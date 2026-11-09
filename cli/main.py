"""
cli/main.py -- the `cascade` command (FR-08).

Builds the Pipeline from ~/.cascade/config.json. Only tiers that are
actually configured get a backend: nothing is sent to a cloud endpoint
the user didn't set up (NFR-05). Secrets are never stored in the config
file -- it names the environment variable that holds each key.

Config keys (set with `cascade config set KEY VALUE`):
    model                 local model family            (default llama3.2)
    runner                ollama | llamacpp             (default ollama)
    acceleration          none | prompt_lookup | draft_model   (llamacpp only)
    small_model           fast model for easy queries, e.g. qwen2.5:1.5b-instruct (default
                          off); the router sends hard ones to `model`
    draft_model           draft tag for draft_model     (default llama3.2:1b-instruct-q4_K_M)
    draft.p_min           stop drafting at the first token the draft model is less sure of
                          than this (default 0.75, fastest in
                          docs/benchmark/acceleration_qwen7b_pmin_3rounds.json); off disables
    draft.adaptive        adapt draft length to acceptance (default true when p_min is off)
    draft.tokens          draft tokens per step (default 16 with p_min, else 4)
    draft.max_tokens      cap on the adaptive draft length (default 16)
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

app = typer.Typer(name="cascade", help="Cascade -- adaptive local/cloud LLM routing.", no_args_is_help=True)
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
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
    from engine.execution.acceleration_engine import AccelerationLayer, DraftPair
    from engine.execution.backend import LocalExecutionBackend

    model = config.get("model", "llama3.2")
    if config.get("runner", "ollama") == "llamacpp":
        from engine.execution.llamacpp_runner import DEFAULT_DRAFT_MODEL_TAG, LlamaCppRunner

        draft = config.get("draft_model", DEFAULT_DRAFT_MODEL_TAG)
        layer = AccelerationLayer[_ACCELERATION_LAYERS.get(config.get("acceleration", "none"), "NONE")]
        return LocalExecutionBackend(
            runner=LlamaCppRunner(draft_model_tag=draft, **_draft_settings(config)),
            model_id=model,
            preferred_layer=layer,
            validated_pairs=[DraftPair(model, draft, True)],
            **_local_limits(config),
        )
    return LocalExecutionBackend(model_id=model, **_local_limits(config))


def build_small_local_backend(config: dict, main_backend):
    """
    The optional fast model for easy queries (config `small_model`, e.g.
    qwen2.5:1.5b-instruct), or None. It shares the main backend's hardware
    profile -- one profiling run, no profile cache thrash between two models --
    and always runs its smallest quant level, without speculative decoding.
    """
    from engine.execution.backend import LocalExecutionBackend
    from engine.execution.model_ladder import DEFAULT_LADDER, ModelLadder

    small = _get(config, "small_model")
    if not small:
        return None
    floor = ModelLadder(rungs=(min(DEFAULT_LADDER, key=lambda r: r.min_tok_s),))
    runner = None
    if config.get("runner", "ollama") == "llamacpp":
        from engine.execution.llamacpp_runner import LlamaCppRunner

        runner = LlamaCppRunner(draft_model_tag=None)
    return LocalExecutionBackend(runner=runner, model_id=small, profiler=main_backend.profiler, ladder=floor,
                                 **_local_limits(config))


def _draft_settings(config: dict) -> dict:
    """Default: confidence drafting (p_min 0.75, up to 16 tokens), the fastest mode in
    docs/benchmark/acceleration_qwen7b_pmin_3rounds.json. `draft.p_min off` falls back
    to adaptive length drafting."""
    p_min = _get(config, "draft.p_min", DEFAULT_DRAFT_P_MIN)
    if isinstance(p_min, str):
        p_min = None if p_min.strip().lower() in ("", "off", "none", "false", "0") else float(p_min)
    settings: dict = {}
    if p_min:
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
    from engine.intelligence.cache import SemanticCache
    from engine.intelligence.context_compressor import ContextCompressor
    from engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH
    from engine.intelligence.embeddings import FastEmbedEmbedder
    from engine.intelligence.pipeline import Pipeline
    from engine.intelligence.router.classifier import SemanticRouterClassifier
    from gateway.backend import SelfHostedBackend
    from gateway.byok import CloudBYOKBackend

    config = load_config()
    embedder = FastEmbedEmbedder()
    byok = _byok_settings(config)
    self_hosted = _self_hosted_settings(config)

    local = build_local_backend(config)
    return Pipeline(
        backend=local,
        small_local_backend=build_small_local_backend(config, local),
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
    from engine.intelligence.cache import CacheConfig

    """Any CacheConfig field set under `cache.` in the config file overrides its default."""
    from dataclasses import fields

    settings = _get(config, "cache", {}) or {}
    overrides = {}
    for field in fields(CacheConfig):
        if field.name in settings:
            value = settings[field.name]
            overrides[field.name] = tuple(value) if isinstance(value, list) else value
    return CacheConfig(**overrides)


def hardware_profiler(config: dict, runner=None):
    """HardwareProfiler for the configured `model`, at the same real tags the local backend loads."""
    from engine.execution.backend import profile_quant_levels
    from engine.execution.hardware_profiler import HardwareProfiler
    from engine.execution.model_runner import OllamaReferenceModelRunner

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
    """FR-08.1: profile this machine's hardware and write the config."""
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


@app.command()
def ask(
    query: str,
    no_log: bool = typer.Option(False, help="Disable decision logging for this request."),
    stream: bool = typer.Option(False, help="Print the answer as it's generated."),
) -> None:
    """FR-08.2: one-shot, scriptable use."""
    warn_if_unprofiled()
    pipeline = build_pipeline(decision_logging=not no_log)
    messages = [{"role": "user", "content": query}]
    if stream:
        for piece in pipeline.run_stream(messages):
            typer.echo(piece, nl=False)
        typer.echo()
        return
    result = pipeline.run(messages)
    typer.echo(result["text"])
    typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK, err=True)


@app.command()
def chat(no_log: bool = typer.Option(False, help="Disable decision logging for this session.")) -> None:
    """FR-08.2: interactive, multi-turn session."""
    warn_if_unprofiled()
    pipeline = build_pipeline(decision_logging=not no_log)
    messages: list[dict] = []
    typer.echo("Cascade chat -- type 'exit' to quit.")

    while True:
        user_input = typer.prompt("you")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user_input})
        result = pipeline.run(messages)
        messages.append({"role": "assistant", "content": result["text"]})
        typer.echo(f"cascade: {result['text']}")
        typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK)


@app.command()
def stats(log_path: Optional[Path] = None) -> None:
    """FR-08.3: cache hit rate, local-resolution rate, and speculative speedup from the local log."""
    from engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH, load_decisions, summarize_decisions

    path = log_path or DEFAULT_DECISION_LOG_PATH
    summary = summarize_decisions(load_decisions(path))

    if summary["total_requests"] == 0:
        typer.echo(f"No requests logged yet at {path}.")
        raise typer.Exit()

    typer.echo(f"Requests logged:       {summary['total_requests']}")
    typer.echo(f"Cache hit rate:        {summary['cache_hit_rate']:.1%}")
    typer.echo(f"Local-resolution rate: {summary['local_resolution_rate']:.1%}")
    if summary["speculative_speedup"] is not None:
        typer.echo(f"Speculative speedup:   {summary['speculative_speedup']:.2f}x")
    else:
        typer.echo("Speculative speedup:   not enough data yet (need both speculative and non-speculative local requests)")
    typer.echo(f"Total est. cost:       ${summary['total_cost_usd']:.4f}")


@app.command()
def profile() -> None:
    """Re-run HardwareProfiler after a hardware change (FR-01.4)."""
    typer.echo(json.dumps(hardware_profiler(load_config()).profile(), indent=2))


@app.command()
def doctor() -> None:
    """Check that every configured piece of Cascade can actually run."""
    import importlib.util

    import httpx

    from engine.execution.model_runner import OllamaReferenceModelRunner

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

    runner_kind = config.get("runner", "ollama")
    if runner_kind == "llamacpp":
        report(importlib.util.find_spec("llama_cpp") is not None, "llama-cpp-python installed (runner=llamacpp)", "pip install llama-cpp-python")
        note(f"acceleration: {config.get('acceleration', 'none')}")
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
        report(healthy, f"self_hosted server at {self_hosted['base_url']}", "is the server running? `python -m gateway.serve run` on that machine")
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
