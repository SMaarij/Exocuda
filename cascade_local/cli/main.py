"""
cli/main.py -- the `cascade` command (FR-08).

Builds the Pipeline from ~/.cascade/config.json. Only tiers that are
actually configured get a backend: nothing is sent to a cloud endpoint
the user didn't set up (NFR-05). Secrets are never stored in the config
file -- it names the environment variable that holds each key.

Config keys (set with `cascade config set KEY VALUE`):
    model                 local model family            (default llama3.2)
    runner                ollama | llamacpp             (default llamacpp when llama-cpp-python
                          is installed, else ollama)
    acceleration          none | prompt_lookup | draft_model   (llamacpp only; default draft_model)
    small_model           fast model for easy queries; the router sends hard ones to `model`.
                          Same family as `model` -> it is also the draft model, loaded once
                          for both. Default on llamacpp: qwen2.5:0.5b-instruct for qwen2.5
                          models, else off; `off` disables
    small_model_min_confidence  re-ask `model` when the small model's mean token probability is
                          below this (default 0.75; off disables)
    draft_model           draft tag for draft_model     (default: small_model when it is
                          the same family, else llama3.2:1b-instruct-q4_K_M)
    draft.policy          cost | p_min | adaptive   (default cost: draft as long as the
                          draft model's confidence pays for what verifying that many tokens
                          costs the big model on this machine, measured once and cached in
                          ~/.cascade/verify_costs.json; 1.59x vs 1.39x for p_min,
                          docs/benchmark/acceleration_qwen7b_cost_aware.json)
    draft.p_min           p_min policy: stop drafting at the first token the draft model is
                          less sure of than this (default 0.75)
    draft.adaptive        adapt draft length to acceptance (default true for policy adaptive)
    draft.tokens          longest draft per step (default 16 for cost and p_min, else 4)
    draft.max_tokens      cap on the adaptive draft length (default 16)
    server.auto           keep models loaded between `cascade ask` calls in a background
                          server on 127.0.0.1 (default true for runner llamacpp)
    server.idle_minutes   the server exits after this long without a request (default 10)
    server.port           (default 8765; 11434 after `cascade install --as-ollama`)
    ollama_upstream       where the real Ollama runs when Cascade has its port (set by
                          `cascade install --as-ollama`: http://127.0.0.1:11435)
    byok.base_url         OpenAI-compatible root, e.g. https://api.openai.com
    byok.model            e.g. gpt-4o-mini
    byok.api_key_env      env var holding the key       (default OPENAI_API_KEY)
    byok.input_price_per_1k / byok.output_price_per_1k  (optional, for cost estimates)
    self_hosted.url       a Cascade server you or your team run, e.g. http://gpu-box:8000
    self_hosted.api_key_env  env var holding its csk_ key (default CASCADE_SERVER_KEY)
    system_prompt         instruction sent with every request (default DEFAULT_SYSTEM_PROMPT;
                          set to "" to send none)
    local.max_tokens      longest local answer, in tokens (default 512)
    cache.uncached_tiers  tiers never served from the cache (default ["self_hosted"]: live data)
    cache.<field>         any CacheConfig field (engine/intelligence/cache.py), e.g.
                          cache.similarity_threshold 0.94, cache.term_swap_guard lenient,
                          cache.time_sensitive_patterns '["\\btoday\\b"]'
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

CONFIG_PATH = Path.home() / ".cascade" / "config.json"

# Model output routinely contains characters a Windows codepage (cp1252)
# can't encode when stdout is piped; print a replacement char, never crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

app = typer.Typer(name="cascade", help="Cascade: run local LLMs faster and with less memory. Cache, small/big model routing, lossless speculative decoding.", no_args_is_help=True)
config_app = typer.Typer(help="Show or change ~/.cascade/config.json.", no_args_is_help=True)
app.add_typer(config_app, name="config")

# Without an instruction, small local models answer "who wrote X" with an
# essay: measured 256 tokens / 66 s for "explain what an api is" vs 33 tokens
# / 8 s with this one (llama3.2:3b on CPU). Override with `config set system_prompt`.
DEFAULT_SYSTEM_PROMPT = (
    "Answer directly and concisely. Start with the answer itself. Match the length of your answer "
    "to the question: a short factual question gets one or two sentences; give detail or steps "
    "only when the question asks for them."
)

_ACCELERATION_LAYERS = {"none": "NONE", "prompt_lookup": "LOOKAHEAD", "draft_model": "DRAFT_MODEL"}


# ---- Config -------------------------------------------------------------------


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("ollama_upstream"):  # `cascade install --as-ollama` moved the real Ollama
        os.environ["CASCADE_OLLAMA_URL"] = config["ollama_upstream"]
    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _get(config: dict, dotted: str, default=None):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _parse_value(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _byok_settings(config: dict) -> Optional[dict]:
    """BYOK settings with the key resolved from the environment, or None if not fully configured."""
    if not _get(config, "byok.base_url") or not _get(config, "byok.model"):
        return None
    key = os.environ.get(_get(config, "byok.api_key_env", "OPENAI_API_KEY"), "")
    return {
        "base_url": _get(config, "byok.base_url"),
        "model": _get(config, "byok.model"),
        "api_key": key,
        "input_price_per_1k": float(_get(config, "byok.input_price_per_1k", 0.0)),
        "output_price_per_1k": float(_get(config, "byok.output_price_per_1k", 0.0)),
    }


def _self_hosted_settings(config: dict) -> Optional[dict]:
    url = _get(config, "self_hosted.url")
    if not url:
        return None
    key = os.environ.get(_get(config, "self_hosted.api_key_env", "CASCADE_SERVER_KEY"), "")
    return {"base_url": url, "api_key": key}


# ---- Pipeline construction (FR-08.1) ----------------------------------------------


def build_local_backend(config: dict):
    from cascade_local.engine.execution.acceleration_engine import AccelerationLayer, DraftPair
    from cascade_local.engine.execution.backend import LocalExecutionBackend

    model = config.get("model", "llama3.2")
    if _runner_kind(config) == "llamacpp":
        from cascade_local.engine.execution.llamacpp_runner import DEFAULT_DRAFT_MODEL_TAG, LlamaCppRunner

        draft = config.get("draft_model") or _small_model_as_draft(config) or DEFAULT_DRAFT_MODEL_TAG
        layer = AccelerationLayer[_ACCELERATION_LAYERS.get(config.get("acceleration", "draft_model"), "NONE")]
        return LocalExecutionBackend(
            runner=LlamaCppRunner(draft_model_tag=draft, **_draft_settings(config)),
            model_id=model,
            preferred_layer=layer,
            validated_pairs=[DraftPair(model, draft, True)],
            **_local_limits(config),
        )
    return LocalExecutionBackend(model_id=model, **_local_limits(config))


# Measured defaults: with no `small_model` set, a model of these families gets this
# small model for easy queries, which also serves as its draft model (one instance in
# RAM). qwen2.5: BENCHMARK_REPORT Addenda 12-13.
DEFAULT_SMALL_MODELS = {"qwen2.5": "qwen2.5:0.5b-instruct"}
_OFF = ("", "off", "none", "false", "0", "no")


def _llama_cpp_available() -> bool:
    return importlib.util.find_spec("llama_cpp") is not None


def _runner_kind(config: dict) -> str:
    """`runner` if set, else llamacpp when llama-cpp-python is installed (speculative
    decoding, lower RAM, shared small model), else ollama."""
    runner = config.get("runner")
    if runner:
        return runner
    return "llamacpp" if _llama_cpp_available() else "ollama"


def _small_model(config: dict) -> Optional[str]:
    """`small_model` if set (off disables), else the measured default for the model's
    family on the llama.cpp runner, else None."""
    small = _get(config, "small_model")
    if isinstance(small, str) and small.strip().lower() in _OFF:
        return None
    if small:
        return small
    if _runner_kind(config) != "llamacpp":
        return None
    return DEFAULT_SMALL_MODELS.get(_family_base(config.get("model", "llama3.2")))


def _family_base(model: str) -> str:
    """"qwen2.5-coder:7b" -> "qwen2.5": models with the same base share a tokenizer."""
    return model.split(":", 1)[0].split("-", 1)[0].lower()


def _small_model_as_draft(config: dict) -> Optional[str]:
    """
    The small model's real tag when it can also draft for `model` (same tokenizer
    family), else None. The llama.cpp runner then loads that file once for both
    jobs (llamacpp_runner._SHARED_MODELS), saving the separate draft model's RAM.
    qwen2.5:0.5b-instruct drafts for qwen2.5-coder:7b as well as the coder draft
    does: 1.376x vs 1.379x, lossless (docs/benchmark/acceleration_qwen7b_general_draft_3rounds.json).
    """
    from cascade_local.engine.execution.backend import _resolve_real_tag
    from cascade_local.engine.execution.model_ladder import DEFAULT_LADDER

    small = _small_model(config)
    if not small or _family_base(small) != _family_base(config.get("model", "llama3.2")):
        return None
    floor = min(DEFAULT_LADDER, key=lambda r: r.min_tok_s)
    return f"{small.split(':', 1)[0]}:{_resolve_real_tag(small, floor.label)}" if ":" in small else None


def build_small_local_backend(config: dict, main_backend):
    """
    The optional fast model for easy queries (config `small_model`, e.g.
    qwen2.5:1.5b-instruct), or None. It shares the main backend's hardware
    profile -- one profiling run, no profile cache thrash between two models --
    and always runs its smallest quant level, without speculative decoding.
    """
    from cascade_local.engine.execution.backend import LocalExecutionBackend
    from cascade_local.engine.execution.model_ladder import DEFAULT_LADDER, ModelLadder

    small = _small_model(config)
    if not small:
        return None
    floor = ModelLadder(rungs=(min(DEFAULT_LADDER, key=lambda r: r.min_tok_s),))
    runner = None
    if _runner_kind(config) == "llamacpp":
        from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner

        runner = LlamaCppRunner(draft_model_tag=None)
    return LocalExecutionBackend(runner=runner, model_id=small, profiler=main_backend.profiler, ladder=floor,
                                 **_local_limits(config))


def _small_confidence_floor(config: dict) -> Optional[float]:
    """`small_model_min_confidence`: re-ask the big model when the small model's mean
    token probability is below this (default 0.75; off disables)."""
    value = _get(config, "small_model_min_confidence", 0.75)
    if isinstance(value, str):
        value = None if value.strip().lower() in ("", "off", "none", "false", "0") else float(value)
    return float(value) if value else None


def _draft_settings(config: dict) -> dict:
    """Default: cost-aware drafting (up to 16 tokens, this machine's measured verify
    costs), the fastest mode in docs/benchmark/acceleration_qwen7b_cost_aware.json.
    `draft.policy p_min` uses confidence drafting; `adaptive` (or `draft.p_min off`
    under p_min) adaptive length drafting."""
    policy = str(_get(config, "draft.policy", "") or "").strip().lower()
    p_min = _get(config, "draft.p_min", DEFAULT_DRAFT_P_MIN)
    if isinstance(p_min, str):
        p_min = None if p_min.strip().lower() in ("", "off", "none", "false", "0") else float(p_min)
    if not policy:  # an explicit draft.p_min (older configs) still means the p_min policy
        policy = "p_min" if _get(config, "draft.p_min") not in (None, "") else "cost"
    if policy not in ("cost", "p_min", "adaptive"):
        raise SystemExit(f"draft.policy must be cost, p_min or adaptive, not {policy!r}")
    settings: dict = {}
    if policy == "cost":
        settings["draft_verify_costs"] = "auto"
        settings["draft_tokens"] = 16
        default_adaptive = False
    elif policy == "p_min" and p_min:
        settings["draft_p_min"] = float(p_min)
        settings["draft_tokens"] = 16
        default_adaptive = False
    else:
        default_adaptive = True
    adaptive = _get(config, "draft.adaptive", default_adaptive)
    if isinstance(adaptive, str):
        adaptive = adaptive.strip().lower() not in ("false", "0", "no", "off")
    settings["adaptive_draft"] = bool(adaptive)
    for key, arg in (("draft.tokens", "draft_tokens"), ("draft.max_tokens", "max_draft_tokens")):
        value = _get(config, key)
        if value not in (None, ""):
            settings[arg] = int(value)
    return settings


DEFAULT_DRAFT_P_MIN = 0.75


def _local_limits(config: dict) -> dict:
    max_tokens = _get(config, "local.max_tokens")
    return {"max_tokens": int(max_tokens)} if max_tokens else {}


def build_pipeline(decision_logging: bool = True):
    """Separate function so tests can monkeypatch in a fake Pipeline (NFR-09)."""
    from cascade_local.engine.intelligence.cache import SemanticCache
    from cascade_local.engine.intelligence.context_compressor import ContextCompressor
    from cascade_local.engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH
    from cascade_local.engine.intelligence.embeddings import FastEmbedEmbedder
    from cascade_local.engine.intelligence.pipeline import Pipeline
    from cascade_local.engine.intelligence.router.classifier import SemanticRouterClassifier
    from cascade_local.gateway.backend import SelfHostedBackend
    from cascade_local.gateway.byok import CloudBYOKBackend

    config = load_config()
    embedder = FastEmbedEmbedder()
    byok = _byok_settings(config)
    self_hosted = _self_hosted_settings(config)

    local = build_local_backend(config)
    return Pipeline(
        backend=local,
        small_local_backend=build_small_local_backend(config, local),
        small_token_confidence_floor=_small_confidence_floor(config),
        self_hosted_backend=SelfHostedBackend(**self_hosted) if self_hosted else None,
        cloud_byok_backend=CloudBYOKBackend(**byok) if byok and byok["api_key"] else None,
        compressor=ContextCompressor(embedder),
        cache=SemanticCache(embedder, config=_cache_config(config)),
        router=SemanticRouterClassifier(embedder),
        decision_log_path=DEFAULT_DECISION_LOG_PATH if decision_logging else None,
        system_prompt=_get(config, "system_prompt", DEFAULT_SYSTEM_PROMPT),
        **({"uncached_tiers": tuple(_get(config, "cache.uncached_tiers"))} if _get(config, "cache.uncached_tiers") is not None else {}),
    )


def _cache_config(config: dict):
    from cascade_local.engine.intelligence.cache import CacheConfig

    """Any CacheConfig field set under `cache.` in the config file overrides its default."""
    from dataclasses import fields

    settings = _get(config, "cache", {}) or {}
    overrides = {}
    for field in fields(CacheConfig):
        if field.name in settings:
            value = settings[field.name]
            overrides[field.name] = tuple(value) if isinstance(value, list) else value
    floor = overrides.get("judge_floor")
    if isinstance(floor, str):  # `cascade config set cache.judge_floor off` turns the cache's judge off
        overrides["judge_floor"] = None if floor.strip().lower() in ("", "off", "none", "false") else float(floor)
    return CacheConfig(**overrides)


def hardware_profiler(config: dict, runner=None):
    """HardwareProfiler for the configured `model`, at the same real tags the local backend loads."""
    from cascade_local.engine.execution.backend import profile_quant_levels
    from cascade_local.engine.execution.hardware_profiler import HardwareProfiler
    from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

    model = config.get("model", "llama3.2")
    return HardwareProfiler(
        runner if runner is not None else OllamaReferenceModelRunner(),
        reference_model_id=model,
        quant_levels=profile_quant_levels(model),
    )


def warn_if_unprofiled() -> None:
    """The first local answer on a new machine runs the hardware check first; say so instead of hanging silently."""
    if hardware_profiler(load_config()).load_cached_profile() is None:
        typer.secho(
            "No hardware profile yet: running the one-time hardware check before answering "
            "(several minutes on CPU). Run `cascade init` to do this ahead of time.",
            fg=typer.colors.YELLOW,
            err=True,
        )


# ---- Output -----------------------------------------------------------------------


def _status_footer(result: dict) -> str:
    parts = [result["tier"]]
    if result.get("escalated"):
        parts.append("escalated")
    if result["cache_hit"]:
        parts.append("cache hit")
    elif result.get("tok_s"):
        parts.append(f"{result['tok_s']:.1f} tok/s")
    parts.append(f"{result['latency_ms']:.0f}ms")
    parts.append(f"${result['est_cost_usd']:.4f}")
    return "[" + " | ".join(parts) + "]"


# ---- Commands ---------------------------------------------------------------------


@app.command()
def init() -> None:
    """Measure this machine (CPU, RAM, GPU, model speed) and write the config."""
    config = load_config()
    typer.echo(
        f"Profiling your hardware (benchmarks {config.get('model', 'llama3.2')}; "
        "downloads it first if it isn't local)..."
    )
    profile = hardware_profiler(config).profile()

    local_capable = profile["local_capable"]
    best_tok_s = max(profile["measured_tok_s"].values()) if profile["measured_tok_s"] else 0.0
    typer.echo(f"local_capable: {local_capable}")
    typer.echo(f"best measured tok/s: {best_tok_s:.1f}")

    if not local_capable:
        typer.secho(
            "This machine doesn't clear any local model rung -- configure a cloud tier "
            "(`cascade config set byok.base_url ...` or `self_hosted.url ...`).",
            fg=typer.colors.YELLOW,
        )

    config = load_config()
    config["initialized"] = True
    config["local_capable"] = local_capable
    save_config(config)
    typer.secho(f"Config written to {CONFIG_PATH}", fg=typer.colors.GREEN)


# ---- cascade setup ----------------------------------------------------------------

# (minimum total RAM in GB, model, download size of model + 0.5B in GB): the largest that fits.
# Peak RAM with the 0.5B loaded beside it: 7B ~6.3 GB, 3B ~2.9 GB, 1.5B ~1.6 GB.
SETUP_CHOICES = (
    (12.0, "qwen2.5-coder", 5.1),
    (8.0, "qwen2.5:3b-instruct", 2.3),
    (0.0, "qwen2.5:1.5b-instruct", 1.4),
)
LLAMACPP_INSTALL = [sys.executable, "-m", "pip", "install", "llama-cpp-python",
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"]


def recommend_model(total_ram_gb: float) -> tuple[str, float]:
    """The model `cascade setup` picks for this much RAM, and its download size (GB)."""
    for floor, model, size in SETUP_CHOICES:
        if total_ram_gb >= floor:
            return model, size
    return SETUP_CHOICES[-1][1], SETUP_CHOICES[-1][2]


def setup_tags(model: str) -> list[str]:
    """The Ollama tags `cascade setup` makes sure are present: the model's floor rung and,
    for a family with a default small model, that model (it also drafts for the big one)."""
    from cascade_local.engine.execution.backend import _resolve_real_tag
    from cascade_local.engine.execution.model_ladder import DEFAULT_LADDER

    floor = min(DEFAULT_LADDER, key=lambda r: r.min_tok_s).label
    tags = [f"{model.split(':', 1)[0]}:{_resolve_real_tag(model, floor)}"]
    small = DEFAULT_SMALL_MODELS.get(_family_base(model))
    if small:
        tags.append(f"{small.split(':', 1)[0]}:{_resolve_real_tag(small, floor)}")
    return tags


def _pull(tag: str) -> bool:
    """Download one model through Ollama, showing progress. True on success."""
    import httpx

    from cascade_local.engine.execution.model_runner import ollama_url

    shown = ""
    try:
        with httpx.stream("POST", ollama_url().rstrip("/") + "/api/pull", json={"model": tag, "stream": True},
                          timeout=httpx.Timeout(30.0, read=None)) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    typer.secho(f"\n   {event['error']}", fg=typer.colors.RED)
                    return False
                total, done = event.get("total"), event.get("completed")
                text = (f"{event.get('status', '')} {100 * done / total:5.1f}%" if total and done
                        else event.get("status", ""))
                if text != shown:
                    typer.echo(f"\r   {text:<60}", nl=False)
                    shown = text
    except httpx.HTTPError as exc:
        typer.secho(f"\n   download failed: {exc}", fg=typer.colors.RED)
        return False
    typer.echo("")
    return True


@app.command()
def setup(
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before downloading or installing."),
    model: Optional[str] = typer.Option(None, help="Use this model instead of the one picked for your RAM."),
    test: bool = typer.Option(True, help="Ask a test question at the end."),
) -> None:
    """Set Cascade up in one go: check Ollama, pick and download models that fit this
    machine, turn on speculative decoding, measure the machine, and ask a test question."""
    import psutil

    from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

    def step(text: str) -> None:
        typer.secho(f"\n> {text}", bold=True)

    step("Checking Ollama")
    ollama = OllamaReferenceModelRunner()
    if not ollama.is_available():
        typer.secho("Ollama isn't running. Install it from https://ollama.com, start it, and run "
                    "`cascade setup` again.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo("   Ollama is running.")

    step("Choosing models for this machine")
    total_gb = psutil.virtual_memory().total / 2**30
    chosen, size_gb = recommend_model(total_gb)
    if model:
        chosen, size_gb = model, 0.0
    tags = setup_tags(chosen)
    extra = " (plus qwen2.5 0.5B, which answers easy questions and drafts for it)" if len(tags) > 1 else ""
    typer.echo(f"   {total_gb:.0f} GB RAM -> {chosen}{extra}")

    step("Downloading models")
    have = set(ollama.local_models())
    missing = [t for t in tags if t not in have]
    if not missing:
        typer.echo("   Already downloaded.")
    else:
        size = f" (about {size_gb:.1f} GB)" if size_gb else ""
        if not yes and not typer.confirm(f"   Download {', '.join(missing)}{size}?", default=True):
            typer.echo("   Skipped. Run `cascade setup` again when you're ready.")
            raise typer.Exit(1)
        for tag in missing:
            typer.echo(f"   {tag}")
            if not _pull(tag):
                raise typer.Exit(1)

    step("Speculative decoding")
    if _llama_cpp_available():
        typer.echo("   llama-cpp-python is installed: lossless speculative decoding is on.")
    else:
        typer.echo("   llama-cpp-python makes answers 1.5-2x faster with identical output, and lowers RAM.")
        if yes or typer.confirm("   Install it now (prebuilt CPU package)?", default=True):
            import subprocess

            if subprocess.run(LLAMACPP_INSTALL).returncode == 0:
                importlib.invalidate_caches()
                typer.echo("   Installed.")
            else:
                typer.secho("   Install failed; Cascade will use Ollama directly. To retry: "
                            + " ".join(LLAMACPP_INSTALL[2:]), fg=typer.colors.YELLOW)
        else:
            typer.echo("   Skipped; Cascade will use Ollama directly (no speculative decoding).")

    config = load_config()
    config["model"] = chosen
    save_config(config)

    step("Measuring this machine (a few minutes on a CPU, once)")
    profile = hardware_profiler(config, runner=ollama).profile()
    best = max(profile["measured_tok_s"].values(), default=0.0)
    typer.echo(f"   {best:.1f} tokens/s with plain decoding.")
    config = load_config()
    config["initialized"] = True
    config["local_capable"] = profile["local_capable"]
    save_config(config)
    if not profile["local_capable"]:
        typer.secho("   This machine is below the local speed floor; try a smaller model "
                    "(`cascade setup --model qwen2.5:1.5b-instruct`) or a cloud tier.", fg=typer.colors.YELLOW)

    if test:
        step("Test question (the first answer also loads the models, about a minute)")
        ask("In one sentence, what is a hash table?", no_log=True, stream=True)

    typer.secho("\nCascade is ready.", fg=typer.colors.GREEN, bold=True)
    typer.echo("   cascade chat                    chat in the terminal")
    typer.echo("   cascade install --as-ollama     make `ollama run` and Ollama apps use Cascade")
    typer.echo("   cascade stats                   see what Cascade has saved you")


@app.command()
def ask(
    query: str,
    no_log: bool = typer.Option(False, help="Disable decision logging for this request."),
    stream: bool = typer.Option(True, help="Print the answer as it's generated."),
) -> None:
    """Ask one question and print the answer (scriptable)."""
    warn_if_unprofiled()
    messages = [{"role": "user", "content": query}]
    printed: list[str] = []

    def show(piece: str) -> None:
        printed.append(piece)
        typer.echo(piece, nl=False)

    result = _ask_warm_server(messages, log=not no_log, on_text=show if stream else None)
    if result is not None:
        typer.echo("" if printed else result["text"])
        typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK, err=True)
        return
    if printed:  # the server failed mid-answer: start the answer again on a new line
        typer.echo()
    pipeline = build_pipeline(decision_logging=not no_log)
    if stream:
        for piece in pipeline.run_stream(messages):
            typer.echo(piece, nl=False)
        typer.echo()
        return
    result = pipeline.run(messages)
    typer.echo(result["text"])
    typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK, err=True)


def _server_settings(config: dict) -> tuple[bool, int, float]:
    """(auto-start, port, idle minutes). Auto-start defaults on for the llama.cpp runner, which
    loads models in this process; Ollama already keeps its models loaded between calls."""
    from cascade_local.cli import warm_server

    auto = _get(config, "server.auto", _runner_kind(config) == "llamacpp")
    if isinstance(auto, str):
        auto = auto.strip().lower() not in ("false", "0", "no", "off")
    return (bool(auto), int(_get(config, "server.port", warm_server.DEFAULT_PORT)),
            float(_get(config, "server.idle_minutes", warm_server.DEFAULT_IDLE_MINUTES)))


def _ask_warm_server(messages: list, log: bool, on_text=None) -> Optional[dict]:
    """The answer from the background server -- started first if needed and enabled -- or None
    to answer in this process instead."""
    from cascade_local.cli import warm_server

    auto, port, idle = _server_settings(load_config())
    state = warm_server.health(port)
    if state is None:
        if not auto:
            return None
        typer.secho("Starting the background model server (first question loads the models; "
                    f"they stay loaded for {idle:g} idle minutes)...", fg=typer.colors.BRIGHT_BLACK, err=True)
        warm_server.spawn(port, idle, log_path=CONFIG_PATH.parent / "server.log")
    if not warm_server.wait_ready(port):
        return None
    if on_text is not None:
        return warm_server.ask(messages, log=log, port=port, on_text=on_text)
    return warm_server.ask(messages, log=log, port=port)


@app.command()
def serve(
    port: int = typer.Option(None, help="Port on 127.0.0.1 (default: config server.port, 8765)."),
    idle_minutes: float = typer.Option(None, help="Free the models' RAM after this long without a request."),
    keep_listening: bool = typer.Option(False, help="After idle, unload the models but keep serving (reload on demand) "
                                                    "instead of exiting."),
    lazy: bool = typer.Option(False, help="Listen at once; load the models on the first request (login auto-start)."),
) -> None:
    """Keep the models loaded for `cascade ask` and for Ollama / OpenAI clients (see `cascade start`)."""
    import threading

    from cascade_local.cli import autostart, warm_server
    from cascade_local.engine.execution.model_runner import ollama_url

    config = load_config()
    _, default_port, default_idle = _server_settings(config)
    port = port or default_port
    if config.get("ollama_upstream"):  # Cascade owns Ollama's port; it runs the real Ollama itself
        threading.Thread(target=autostart.start_ollama, args=(config["ollama_upstream"],), daemon=True).start()
    server = warm_server.WarmServer(lambda: build_pipeline(decision_logging=True), config_path=CONFIG_PATH,
                                    idle_seconds=(idle_minutes or default_idle) * 60, port=port,
                                    keep_listening=keep_listening, model_name=config.get("model", "llama3.2"),
                                    upstream=ollama_url(), lazy=lazy)
    typer.echo(f"Cascade model server on 127.0.0.1:{port}, " + ("models load on the first request" if lazy else "loading models..."))
    server.serve_forever(load_in_background=not lazy)


@app.command()
def start() -> None:
    """
    Run Cascade in the background for every program on this machine: Ollama and OpenAI
    clients pointed at it get Cascade's cache, routing, speculative decoding and lower RAM.
    It stays reachable; after `server.idle_minutes` without a request it unloads the models
    to free their RAM and reloads them on the next one. `cascade stop` ends it.
    """
    from cascade_local.cli import warm_server

    _, port, idle = _server_settings(load_config())
    state = warm_server.health(port)
    if state is None:
        warm_server.spawn(port, idle, log_path=CONFIG_PATH.parent / "server.log", keep_listening=True)
        typer.echo("Starting Cascade in the background (loading models, about a minute on a CPU)...")
        if not warm_server.wait_ready(port):
            typer.secho(f"The server did not come up; see {CONFIG_PATH.parent / 'server.log'}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    _print_usage(port)
    typer.echo("Stop it with:  cascade stop")


def _print_usage(port: int) -> None:
    from cascade_local.cli import autostart

    typer.echo(f"Cascade is running on http://127.0.0.1:{port}\n")
    typer.echo("Use it from any tool, unchanged:")
    if port == autostart.OLLAMA_PORT:
        typer.echo("  Ollama clients:  nothing to set -- Cascade is on Ollama's own port. e.g.  ollama run qwen2.5-coder")
        typer.echo(f"                   (other models, pull and list go to the real Ollama on {autostart.MOVED_OLLAMA_URL})")
    else:
        typer.echo(f"  Ollama clients:  OLLAMA_HOST=127.0.0.1:{port} in that terminal or app, then e.g.  ollama run qwen2.5-coder")
        typer.echo("                   (not system-wide: the Ollama app itself listens on OLLAMA_HOST;")
        typer.echo("                   `cascade install --as-ollama` makes even this unnecessary)")
    typer.echo(f"  OpenAI clients:  base URL http://127.0.0.1:{port}/v1   (any API key)")


@app.command()
def stop() -> None:
    """Stop the background model server and free its RAM."""
    from cascade_local.cli import warm_server

    port = _server_settings(load_config())[1]
    typer.echo("Stopped." if warm_server.stop(port) else "No model server running.")


@app.command()
def install(
    as_ollama: bool = typer.Option(False, "--as-ollama", help="Take Ollama's port 11434, so `ollama run` and Ollama "
                                                              "apps use Cascade with no setting; the real Ollama "
                                                              "moves to 11435 and runs every other model."),
) -> None:
    """
    Start Cascade whenever you log in. It loads no model until a program asks for one and
    unloads them after `server.idle_minutes`, so an idle machine holds no model in RAM.
    `cascade uninstall` undoes everything.
    """
    from cascade_local.cli import autostart, warm_server

    config = load_config()
    old_port = _server_settings(config)[1]
    if as_ollama:
        config["server"] = dict(config.get("server") or {}, port=autostart.OLLAMA_PORT)
        config["ollama_upstream"] = autostart.MOVED_OLLAMA_URL
    else:
        if config.get("ollama_upstream"):  # was --as-ollama: give Ollama its port back
            _give_ollama_its_port_back(config)
    save_config(config)
    load_config()  # applies ollama_upstream to this process
    port = _server_settings(config)[1]
    if old_port != port:
        warm_server.stop(old_port)

    where = autostart.install_entry()
    typer.echo(f"Cascade will start at login ({where}).")
    if as_ollama:
        step = autostart.set_aside_ollama_autostart()
        if step and step.startswith("moved"):
            typer.echo(f"Ollama: {step}.")
        elif step:
            typer.secho(f"One manual step: {step}.", fg=typer.colors.YELLOW)
        if autostart.server_kind(f"http://127.0.0.1:{port}") == "ollama":
            if os.name == "nt":
                ended = autostart.stop_ollama_app()
                typer.echo(f"Stopped the running Ollama app ({ended} processes); Cascade restarts Ollama on "
                           f"{autostart.MOVED_OLLAMA_URL}.")
                _wait_port_free(port)
            else:
                typer.secho(f"Ollama is still running on port {port}; Cascade takes the port once it is stopped.",
                            fg=typer.colors.YELLOW)
                raise typer.Exit(1)
    if warm_server.health(port) is None:
        warm_server.spawn(port, _server_settings(config)[2], log_path=CONFIG_PATH.parent / "server.log",
                          keep_listening=True, lazy=True)  # what the login entry runs, now
        if not warm_server.wait_ready(port, timeout=60):
            typer.secho(f"Cascade did not start; see {CONFIG_PATH.parent / 'server.log'}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
    _print_usage(port)
    typer.echo("Undo with:  cascade uninstall")


def _wait_port_free(port: int, timeout: float = 15.0) -> None:
    import time

    from cascade_local.cli import autostart

    deadline = time.monotonic() + timeout
    while autostart.server_kind(f"http://127.0.0.1:{port}") is not None and time.monotonic() < deadline:
        time.sleep(0.5)


def _give_ollama_its_port_back(config: dict) -> None:
    from cascade_local.cli import autostart, warm_server

    warm_server.stop(_server_settings(config)[1])
    autostart.stop_started_ollama()
    config.pop("ollama_upstream", None)
    os.environ.pop("CASCADE_OLLAMA_URL", None)
    server = config.get("server") or {}
    if server.get("port") == autostart.OLLAMA_PORT:
        server.pop("port")
        if server:
            config["server"] = server
        else:
            config.pop("server", None)
    link = autostart.restore_ollama_autostart()
    if link is not None:
        _wait_port_free(autostart.OLLAMA_PORT)  # Cascade has released Ollama's port
        autostart.start_ollama_app(link)
        typer.echo("Ollama's login shortcut is back, and the Ollama app is running again.")


@app.command()
def uninstall() -> None:
    """Stop starting Cascade at login, stop it now, and give Ollama its port back."""
    from cascade_local.cli import autostart, warm_server

    config = load_config()
    removed = autostart.remove_entry()
    if config.get("ollama_upstream"):
        _give_ollama_its_port_back(config)
        save_config(config)
    else:
        warm_server.stop(_server_settings(config)[1])
    typer.echo("Cascade no longer starts at login and is stopped." if removed else "Cascade was not set to start at login.")


@app.command()
def chat(no_log: bool = typer.Option(False, help="Disable decision logging for this session.")) -> None:
    """Chat interactively, with conversation history."""
    warn_if_unprofiled()
    pipeline = None  # built only if the background server can't answer
    messages: list[dict] = []
    typer.echo("Cascade chat -- type 'exit' to quit.")

    while True:
        user_input = typer.prompt("you")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user_input})
        typer.echo("cascade: ", nl=False)
        printed: list[str] = []

        def show(piece: str) -> None:
            printed.append(piece)
            typer.echo(piece, nl=False)

        # The background server keeps the models (and this conversation's KV cache) loaded
        # between turns and streams the answer as it is written.
        result = _ask_warm_server(messages, log=not no_log, on_text=show) if pipeline is None else None
        if result is None:
            if printed:
                typer.echo("\ncascade: ", nl=False)
            if pipeline is None:
                pipeline = build_pipeline(decision_logging=not no_log)
            result = pipeline.run(messages)
            printed = []
        messages.append({"role": "assistant", "content": result["text"]})
        typer.echo("" if printed else result["text"])
        typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK)


@app.command()
def stats(
    log_path: Optional[Path] = None,
    days: Optional[float] = typer.Option(None, help="Only the last N days."),
) -> None:
    """Show what Cascade has saved you: cache hits, answers by tier, speed-up."""
    import time

    from cascade_local.engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH, load_decisions, summarize_decisions

    path = log_path or DEFAULT_DECISION_LOG_PATH
    since = time.time() - days * 86400 if days else None
    summary = summarize_decisions(load_decisions(path), since=since)

    if summary["total_requests"] == 0:
        typer.echo(f"No requests logged yet at {path}." if not days else f"No requests in the last {days:g} days.")
        raise typer.Exit()

    def secs(ms: Optional[float]) -> str:
        return "-" if ms is None else (f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms")

    def duration(s: float) -> str:
        return f"{s / 3600:.1f} h" if s >= 3600 else (f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s")

    total = summary["total_requests"]
    window = f" in the last {days:g} days" if days else ""
    typer.secho(f"{total} questions answered{window}", bold=True)
    labels = {"cache": "from the cache", "small": "by the small model", "local": "by the main local model",
              "cloud_byok": "by your cloud provider", "self_hosted": "by your Cascade server"}
    for src in ("cache", "small", "local", "cloud_byok", "self_hosted"):
        n = summary["by_source"].get(src, 0)
        if n:
            typer.echo(f"  {n:5d}  {labels[src]:<26} ({n / total:5.1%})   typical answer {secs(summary['median_ms'].get(src))}")
    if summary["time_saved_s"] > 0:
        typer.secho(f"Time saved by the cache: about {duration(summary['time_saved_s'])}", fg=typer.colors.GREEN)
    if summary["speculative_speedup"] is not None and min(summary["speculative_samples"]) >= 10:
        typer.echo(f"Speculative decoding:    {summary['speculative_speedup']:.2f}x the speed of plain decoding "
                   f"on your answers")
    on_machine = sum(summary["by_source"].get(src, 0) for src in ("cache", "small", "local")) / total
    typer.echo(f"Answered on this machine: {on_machine:.1%}")
    if summary["total_cost_usd"] > 0:
        typer.echo(f"Est. cloud cost:          ${summary['total_cost_usd']:.4f}")


@app.command()
def profile() -> None:
    """Re-measure this machine after a hardware change."""
    typer.echo(json.dumps(hardware_profiler(load_config()).profile(), indent=2))


@app.command()
def doctor() -> None:
    """Check that every configured piece of Cascade can actually run."""
    import importlib.util

    import httpx

    from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

    config = load_config()
    problems = 0

    def report(ok: bool, label: str, hint: str = "") -> None:
        nonlocal problems
        problems += 0 if ok else 1
        mark = typer.style("ok  ", fg=typer.colors.GREEN) if ok else typer.style("FIX ", fg=typer.colors.RED)
        typer.echo(f"{mark} {label}" + (f"\n       -> {hint}" if hint and not ok else ""))

    def note(label: str) -> None:
        typer.echo(f"{typer.style('--  ', fg=typer.colors.BRIGHT_BLACK)} {label}")

    report(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}", "Cascade needs Python 3.10+")

    ollama = OllamaReferenceModelRunner()
    up = ollama.is_available()
    report(up, "Ollama server reachable", "install from https://ollama.com and start it")
    if up:
        model = config.get("model", "llama3.2")
        local = [m for m in ollama.local_models() if m.startswith(model + ":")]
        report(bool(local), f"local {model} models: {', '.join(local) or 'none'}", f"run `ollama pull {model}:3b-instruct-q4_K_M` or `cascade init`")

    cached = hardware_profiler(config, runner=ollama).load_cached_profile()
    if cached is None:
        report(False, f"hardware profile for {config.get('model', 'llama3.2')}", "run `cascade init` (or `cascade profile`)")
    else:
        best = max(cached["measured_tok_s"].values(), default=0.0)
        report(cached["local_capable"], f"hardware profile: local_capable={cached['local_capable']}, best {best:.1f} tok/s",
               "this machine is below the local floor -- configure a cloud tier")

    runner_kind = _runner_kind(config)
    if runner_kind == "llamacpp":
        report(importlib.util.find_spec("llama_cpp") is not None, "llama-cpp-python installed (runner=llamacpp)", "pip install llama-cpp-python")
        note(f"acceleration: {config.get('acceleration', 'draft_model')}; small model: {_small_model(config) or 'off'}")
    else:
        note("runner: ollama (no speculative decoding; set runner=llamacpp to enable it)")

    byok = _byok_settings(config)
    if byok is None:
        note("cloud_byok tier: not configured (hard queries stay local)")
    else:
        env = _get(config, "byok.api_key_env", "OPENAI_API_KEY")
        report(bool(byok["api_key"]), f"cloud_byok: {byok['model']} at {byok['base_url']}", f"set the {env} environment variable")

    self_hosted = _self_hosted_settings(config)
    if self_hosted is None:
        note("self_hosted tier: not configured")
    else:
        try:
            healthy = httpx.get(self_hosted["base_url"].rstrip("/") + "/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            healthy = False
        report(healthy, f"self_hosted server at {self_hosted['base_url']}", "is the server running? `python -m cascade_local.gateway.serve run` on that machine")
        env = _get(config, "self_hosted.api_key_env", "CASCADE_SERVER_KEY")
        report(bool(self_hosted["api_key"]), "self_hosted API key present", f"set the {env} environment variable")

    typer.echo()
    typer.echo("All good." if problems == 0 else f"{problems} thing(s) to fix.")
    raise typer.Exit(code=0 if problems == 0 else 1)


@config_app.command("show")
def config_show() -> None:
    """Print the current config."""
    typer.echo(json.dumps(load_config(), indent=2))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a (dotted) key, e.g. `cascade config set byok.model gpt-4o-mini`."""
    if "api_key" in key and not key.endswith("_env"):
        typer.secho("Refusing to store a key in the config file -- put it in an environment variable and set *.api_key_env instead.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    config = load_config()
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _parse_value(value)
    save_config(config)
    typer.echo(f"{key} = {json.dumps(node[parts[-1]])}")


@config_app.command("unset")
def config_unset(key: str) -> None:
    """Remove a (dotted) key."""
    config = load_config()
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.get(part, {})
    node.pop(parts[-1], None)
    save_config(config)
    typer.echo(f"removed {key}")


if __name__ == "__main__":
    app()
