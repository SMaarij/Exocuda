"""
scripts/benchmark_acceleration.py

NFR-01 measurement: real decode speed of each acceleration layer on THIS
machine, using LlamaCppRunner (the runner that can actually speculate).

    none           plain autoregressive decoding (baseline)
    prompt_lookup  Layer 1: n-gram drafts from the context
    draft_model    Layer 3: a small same-family draft model

Greedy decoding (temperature 0) so the lossless claim is checkable: each
accelerated output is compared token-for-token with the baseline output.

Usage:
    python scripts/benchmark_acceleration.py [--model llama3.2]
        [--quant 3b-instruct-q4_K_M] [--draft llama3.2:1b-instruct-q4_K_M]
        [--max-tokens 128] [--out docs/benchmark/acceleration_results.json]

Integration scope (NFR-09): needs Ollama running and llama-cpp-python.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = "llama3.2"
DEFAULT_QUANT = "3b-instruct-q4_K_M"

_CODE = '''def merge_sorted(a, b):
    i, j, out = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
'''
_TEXT = (
    "Cascade routes each LLM request to the cheapest place that can answer it well: "
    "a local model on the user's own machine, the user's own cloud API key, or a hosted "
    "fallback. A semantic cache answers repeated questions instantly, and a context "
    "compressor trims long conversations before they are sent anywhere."
)

# Context-grounded prompts (where copying from the input is natural) and
# open-ended ones (where it isn't) -- prompt lookup should only help the first kind.
PROMPTS = [
    ("grounded", "Add type hints and a docstring to this function. Return only the code.\n\n" + _CODE),
    ("grounded", "Rewrite this function with descriptive variable names. Return only the code.\n\n" + _CODE),
    ("grounded", "Repeat the following paragraph, but in British spelling:\n\n" + _TEXT),
    ("open", "Explain what a hash table is in one paragraph."),
    ("open", "Write a haiku about the ocean."),
    ("open", "Give three tips for writing readable Python."),
]

MODES = ("none", "prompt_lookup", "draft_model")


def run(model: str, quant: str, draft: str, max_tokens: int) -> dict:
    from engine.execution.llamacpp_runner import LlamaCppRunner

    runner = LlamaCppRunner(draft_model_tag=draft, temperature=0.0)
    results: dict[str, list[dict]] = {}
    for mode in MODES:
        if not runner.set_acceleration(mode):
            print(f"  {mode}: not supported by runner, skipped")
            continue
        runner.load(model, quant)
        runner.generate("Say OK.", 8)  # warm-up: first call pays one-off allocation costs
        rows = []
        for kind, prompt in PROMPTS:
            t = time.perf_counter()
            sample = runner.generate(prompt, max_tokens)
            wall_s = time.perf_counter() - t
            rows.append({
                "kind": kind,
                "decode_tok_s": round(sample.tok_s, 3),
                "wall_s": round(wall_s, 2),
                "tokens": sample.tokens_generated + 1,
                "acceptance_rate": None if sample.acceptance_rate is None else round(sample.acceptance_rate, 3),
                "ran": sample.acceleration,
                "text": sample.output_text,
            })
            print(f"  {mode:<14} {kind:<9} {sample.tok_s:6.2f} tok/s  wall {wall_s:6.1f}s  "
                  f"acc={rows[-1]['acceptance_rate']}", flush=True)
        results[mode] = rows
        runner.unload()
    return results


def summarize(results: dict) -> dict:
    base = results.get("none", [])
    summary = {}
    for mode, rows in results.items():
        per_kind = {}
        for kind in ("grounded", "open", "all"):
            idx = [i for i, r in enumerate(rows) if kind == "all" or r["kind"] == kind]
            if not idx:
                continue
            tok_s = statistics.mean(rows[i]["decode_tok_s"] for i in idx)
            wall = sum(rows[i]["wall_s"] for i in idx)
            base_tok_s = statistics.mean(base[i]["decode_tok_s"] for i in idx) if base else 0.0
            base_wall = sum(base[i]["wall_s"] for i in idx) if base else 0.0
            identical = sum(1 for i in idx if base and rows[i]["text"] == base[i]["text"])
            accs = [rows[i]["acceptance_rate"] for i in idx if rows[i]["acceptance_rate"] is not None]
            per_kind[kind] = {
                "mean_decode_tok_s": round(tok_s, 3),
                "decode_speedup_vs_plain": round(tok_s / base_tok_s, 3) if base_tok_s else None,
                "total_wall_s": round(wall, 1),
                "wall_speedup_vs_plain": round(base_wall / wall, 3) if wall else None,
                "mean_acceptance_rate": round(statistics.mean(accs), 3) if accs else None,
                "outputs_identical_to_plain": f"{identical}/{len(idx)}",
            }
        summary[mode] = per_kind
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", default=DEFAULT_QUANT)
    parser.add_argument("--draft", default="llama3.2:1b-instruct-q4_K_M")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--out", default="docs/benchmark/acceleration_results.json")
    args = parser.parse_args(argv)

    print(f"Benchmarking {args.model}:{args.quant} (draft {args.draft}), {len(PROMPTS)} prompts x {len(MODES)} modes")
    results = run(args.model, args.quant, args.draft, args.max_tokens)
    summary = summarize(results)

    print("\nmode            subset     decode tok/s  speedup  wall speedup  acceptance  identical")
    for mode, kinds in summary.items():
        for kind, s in kinds.items():
            print(f"{mode:<15} {kind:<9} {s['mean_decode_tok_s']:>12.2f} {s['decode_speedup_vs_plain'] or 0:>8.2f}x"
                  f" {s['wall_speedup_vs_plain'] or 0:>11.2f}x  {str(s['mean_acceptance_rate']):>10}  {s['outputs_identical_to_plain']:>9}")

    best = max((s["all"]["decode_speedup_vs_plain"] or 0) for m, s in summary.items() if m != "none") if len(summary) > 1 else 0
    print(f"\nNFR-01 (>= 1.5x): {'PASS' if best >= 1.5 else 'NOT MET'} on this machine (best decode speedup {best:.2f}x)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "model": f"{args.model}:{args.quant}",
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "runner": "llama-cpp-python (CPU build)",
        "summary": summary,
        "per_prompt": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
