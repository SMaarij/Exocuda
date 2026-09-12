"""
cli/main.py

Track C, Tasks 1, 12, 13, 14 (FR-08.1-08.3).

Wires the full Pipeline together exactly as
docs/TRACK_B_HANDOFF_FOR_C.md Section 4, Step 2 shows -- this is the
one place all three tracks' code actually meets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

CONFIG_PATH = Path.home() / ".cascade" / "config.json"

app = typer.Typer(name="cascade", help="Cascade -- adaptive local/cloud LLM routing.")


# ---- Pipeline construction (FR-08.1's dependency graph) --------------------


def build_pipeline(decision_logging: bool = True):
    """
    Constructs the full Pipeline. A separate function (not inlined into
    each command) so tests can monkeypatch it with a fake Pipeline
    instead of needing a real Ollama instance or CascadeCloud endpoint
    (NFR-09).
    """
    from engine.execution.backend import LocalExecutionBackend
    from engine.intelligence.cache import SemanticCache
    from engine.intelligence.context_compressor import ContextCompressor
    from engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH
    from engine.intelligence.embeddings import FastEmbedEmbedder
    from engine.intelligence.pipeline import Pipeline
    from engine.intelligence.router.classifier import SemanticRouterClassifier
    from gateway.backend import CascadeCloudBackend

    config = load_config()
    embedder = FastEmbedEmbedder()
    cascade_backend = CascadeCloudBackend(
        base_url=config.get("cascade_cloud_url", "https://api.cascade.dev"),
        api_key=config.get("cascade_cloud_api_key", ""),
    )

    return Pipeline(
        backend=LocalExecutionBackend(),
        cascade_backend=cascade_backend,
        compressor=ContextCompressor(embedder),
        cache=SemanticCache(embedder),
        router=SemanticRouterClassifier(embedder),
        decision_log_path=DEFAULT_DECISION_LOG_PATH if decision_logging else None,
    )


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---- Status footer (FR-08.2) -----------------------------------------------


def _status_footer(result: dict) -> str:
    return (
        f"[{result['tier']}"
        f" | cache_hit={result['cache_hit']}"
        f" | {result['latency_ms']:.0f}ms"
        f" | ${result['est_cost_usd']:.4f}]"
    )


# ---- Commands ---------------------------------------------------------------


@app.command()
def init() -> None:
    """FR-08.1: detect hardware, pick an initial local model rung, write config.yaml -- under 5 minutes on a typical connection."""
    from engine.execution.hardware_profiler import HardwareProfiler
    from engine.execution.model_runner import OllamaReferenceModelRunner

    typer.echo("Profiling your hardware...")
    profiler = HardwareProfiler(OllamaReferenceModelRunner())
    profile = profiler.profile()

    local_capable = profile["local_capable"]
    best_tok_s = max(profile["measured_tok_s"].values()) if profile["measured_tok_s"] else 0.0

    typer.echo(f"local_capable: {local_capable}")
    typer.echo(f"best measured tok/s: {best_tok_s:.1f}")

    if not local_capable:
        typer.secho(
            "This machine doesn't clear any local model rung -- Cascade "
            "will route your requests to CascadeCloud instead.",
            fg=typer.colors.YELLOW,
        )

    config = load_config()
    config["initialized"] = True
    config["local_capable"] = local_capable
    save_config(config)
    typer.secho(f"Config written to {CONFIG_PATH}", fg=typer.colors.GREEN)


@app.command()
def ask(query: str, no_log: bool = typer.Option(False, help="Disable decision logging for this request.")) -> None:
    """FR-08.2: one-shot, scriptable use."""
    pipeline = build_pipeline(decision_logging=not no_log)
    result = pipeline.run([{"role": "user", "content": query}])
    typer.echo(result["text"])
    typer.secho(_status_footer(result), fg=typer.colors.BRIGHT_BLACK, err=True)


@app.command()
def chat(no_log: bool = typer.Option(False, help="Disable decision logging for this session.")) -> None:
    """FR-08.2: interactive, multi-turn session."""
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
    """FR-08.3: aggregate cache hit rate, local-resolution rate, and speculative-decoding speedup from the local log file."""
    from engine.intelligence.decision_log import DEFAULT_DECISION_LOG_PATH, load_decisions, summarize_decisions

    path = log_path or DEFAULT_DECISION_LOG_PATH
    records = load_decisions(path)
    summary = summarize_decisions(records)

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
    """Re-run HardwareProfiler after a hardware change (FR-01.4, exposed via the CLI)."""
    from engine.execution.hardware_profiler import HardwareProfiler
    from engine.execution.model_runner import OllamaReferenceModelRunner

    profiler = HardwareProfiler(OllamaReferenceModelRunner())
    hw_profile = profiler.profile()
    typer.echo(json.dumps(hw_profile, indent=2))


if __name__ == "__main__":
    app()
