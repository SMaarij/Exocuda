"""
scripts/benchmark_acceleration.py

NFR-01 measurement: real decode speed of each acceleration layer on THIS
machine, using LlamaCppRunner (the runner that can actually speculate).

    none           plain autoregressive decoding (baseline)
    prompt_lookup  Layer 1: n-gram drafts from the context
    draft_model    Layer 3: a small same-family draft model

Greedy decoding (temperature 0) so the lossless claim is checkable, two ways:
  - identical: the accelerated text equals the plain text;
  - greedy check (LlamaCppRunner.verify_greedy): every output token is the
    target model's top choice when recomputed one token at a time, or within
    the floating-point noise measured between batched and one-at-a-time
    evaluation of that same sequence. Verifying drafts evaluates tokens in
    batches, so at a near-tie the accelerated text can take the other branch
    while still being a correct greedy decode; a violation would be a real bug.

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


def parse_mode(spec: str, draft_tokens: int, lookup_tokens: int) -> tuple[str, dict]:
    """'draft_model', 'draft_model:k=2', 'draft_model:adaptive', 'draft_model:adaptive:pause=0', 'draft_model:pmin=0.75:k=16',
    'draft_model:draft=qwen2.5@0.5b-instruct', 'prompt_lookup:k=3' -> (runner mode, LlamaCppRunner settings). Unspecified settings use
    the CLI flags."""
    mode, *options = spec.split(":")
    settings = {"draft_tokens": draft_tokens, "lookup_tokens": lookup_tokens}
    for option in options:
        if option == "adaptive":
            settings["adaptive_draft"] = True
        elif option.startswith("k="):
            settings["lookup_tokens" if mode == "prompt_lookup" else "draft_tokens"] = int(option[2:])
        elif option.startswith("pause="):
            settings["max_draft_pause"] = int(option[6:])
        elif option == "nolookup":  # cost-aware without lookup drafting
            settings["draft_lookup"] = False
        elif option == "shadow":  # the draft model runs while the target verifies
            settings["shadow_draft"] = True
        elif option == "nohistory":  # lookup drafting within each request only
            settings["draft_lookup_history"] = False
        elif option == "costauto":  # cost-aware, this machine's measured table (the CLI default)
            settings["draft_verify_costs"] = "auto"
        elif option == "cost":  # cost-aware draft length, CPU cost table
            from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

            settings["draft_verify_costs"] = CPU_VERIFY_COSTS
        elif option.startswith("pmin="):
            settings["draft_p_min"] = float(option[5:])
        elif option.startswith("draft="):
            # '@' stands for ':' inside the tag, since ':' separates options
            settings["draft_model_tag"] = option[6:].replace("@", ":")
        else:
            raise ValueError(f"unknown mode option {option!r} in {spec!r}")
    return mode, settings


def run(model: str, quant: str, draft: str, max_tokens: int, draft_tokens: int = 4, lookup_tokens: int = 10,
        modes: tuple[str, ...] = MODES, repeats: int = 1, use_mmap: bool = False) -> dict:
    from engine.execution.llamacpp_runner import LlamaCppRunner

    results: dict[str, list[dict]] = {}
    # Rounds interleave the modes (none, A, B, none, A, B, ...) so slow drift
    # on the machine -- heat, background load -- hits every mode alike.
    for round_no, spec in ((r, spec) for r in range(repeats) for spec in modes):
        mode, settings = parse_mode(spec, draft_tokens, lookup_tokens)
        settings.setdefault("draft_model_tag", draft)
        runner = LlamaCppRunner(temperature=0.0, use_mmap=use_mmap, **settings)
        if not runner.set_acceleration(mode):
            print(f"  {spec}: not supported by runner, skipped")
            continue
        runner.load(model, quant)
        runner.generate("Say OK.", 8)  # warm-up: first call pays one-off allocation costs
        rows = []
        for kind, prompt in PROMPTS:
            t = time.perf_counter()
            sample = runner.generate(prompt, max_tokens)
            wall_s = time.perf_counter() - t
            ids = (list(runner.last_prompt_ids), list(runner.last_output_ids))
            rows.append({
                "_ids": ids,
                "round": round_no,
                "kind": kind,
                "decode_tok_s": round(sample.tok_s, 3),
                "wall_s": round(wall_s, 2),
                "tokens": sample.tokens_generated + 1,
                "acceptance_rate": None if sample.acceptance_rate is None else round(sample.acceptance_rate, 3),
                "ran": sample.acceleration,
                "rss_gb": round(sample.peak_memory_gb, 3),
                "text": sample.output_text,
            })
            print(f"  {spec:<20} {kind:<9} {sample.tok_s:6.2f} tok/s  wall {wall_s:6.1f}s  "
                  f"acc={rows[-1]['acceptance_rate']}", flush=True)
        # Verified after the timed runs so verification never affects speed.
        for row in rows:
            check = runner.verify_greedy(*row.pop("_ids"))
            row["greedy_check"] = {
                "tokens": check.tokens, "exact": check.exact, "within_noise": check.within_noise,
                "violations": check.violations, "noise_tolerance": round(check.noise_tolerance, 4),
            }
        results.setdefault(spec, []).extend(rows)
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
            checks = [rows[i].get("greedy_check") for i in idx if rows[i].get("greedy_check")]
            rounds = sorted({rows[i].get("round", 0) for i in idx})
            round_speedups = []
            for r in rounds:
                ri = [i for i in idx if rows[i].get("round", 0) == r]
                b = statistics.mean(base[i]["decode_tok_s"] for i in ri) if base else 0.0
                round_speedups.append(round(statistics.mean(rows[i]["decode_tok_s"] for i in ri) / b, 3) if b else None)
            rss = [rows[i]["rss_gb"] for i in idx if rows[i].get("rss_gb")]
            per_kind[kind] = {
                "mean_decode_tok_s": round(tok_s, 3),
                "decode_speedup_vs_plain": round(tok_s / base_tok_s, 3) if base_tok_s else None,
                "total_wall_s": round(wall, 1),
                "wall_speedup_vs_plain": round(base_wall / wall, 3) if wall else None,
                "decode_speedup_per_round": round_speedups,
                "peak_rss_gb": max(rss) if rss else None,
                "mean_acceptance_rate": round(statistics.mean(accs), 3) if accs else None,
                "outputs_identical_to_plain": f"{identical}/{len(idx)}",
                "greedy_check_violations": sum(c["violations"] for c in checks),
                "greedy_check_near_ties": sum(c["within_noise"] for c in checks),
                "greedy_check_tokens": sum(c["tokens"] for c in checks),
            }
        summary[mode] = per_kind
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", default=DEFAULT_QUANT)
    parser.add_argument("--draft", default="llama3.2:1b-instruct-q4_K_M")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=4, help="tokens the draft model proposes per step")
    parser.add_argument("--lookup-tokens", type=int, default=10, help="tokens prompt lookup proposes per step")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="comma-separated; 'none' first. Options: draft_model:k=N, draft_model:adaptive, prompt_lookup:k=N")
    parser.add_argument("--repeats", type=int, default=1, help="rounds; modes are interleaved within each round")
    parser.add_argument("--mmap", action="store_true", help="memory-map the model file (uses more RAM, see Addendum 7)")
    parser.add_argument("--out", default="docs/benchmark/acceleration_results.json")
    args = parser.parse_args(argv)

    print(f"Benchmarking {args.model}:{args.quant} (draft {args.draft}), {len(PROMPTS)} prompts x {len(args.modes.split(','))} modes")
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    results = run(args.model, args.quant, args.draft, args.max_tokens, args.draft_tokens, args.lookup_tokens, modes,
                  repeats=args.repeats, use_mmap=args.mmap)
    summary = summarize(results)

    print("\nmode            subset     decode tok/s  speedup  wall speedup  acceptance  identical  greedy check")
    for mode, kinds in summary.items():
        for kind, s in kinds.items():
            print(f"{mode:<21} {kind:<9} {s['mean_decode_tok_s']:>12.2f} {s['decode_speedup_vs_plain'] or 0:>8.2f}x"
                  f" {s['wall_speedup_vs_plain'] or 0:>11.2f}x  {str(s['mean_acceptance_rate']):>10}  {s['outputs_identical_to_plain']:>9}"
                  f"  {s['greedy_check_violations']} violations, {s['greedy_check_near_ties']} near-ties"
                  f" / {s['greedy_check_tokens']} tokens  RSS {s['peak_rss_gb']} GB"
                  + (f"  rounds {s['decode_speedup_per_round']}" if len(s['decode_speedup_per_round']) > 1 else ""))
    violations = sum(k["all"]["greedy_check_violations"] for k in summary.values())
    print(f"\nLossless (greedy check): {'PASS' if violations == 0 else 'FAIL'} -- {violations} tokens outside measured noise")

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
        "draft_tokens": args.draft_tokens,
        "lookup_tokens": args.lookup_tokens,
        "repeats": args.repeats,
        "use_mmap": args.mmap,
        "runner": "llama-cpp-python (CPU build)",
        "summary": summary,
        "per_prompt": results,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
