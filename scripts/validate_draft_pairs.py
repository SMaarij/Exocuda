"""
scripts/validate_draft_pairs.py

Validates a Layer 3 (target, draft) pair on THIS machine by actually
running speculative decoding with LlamaCppRunner, and records it in
~/.cascade/validated_pairs.json (read by AccelerationEngine) only if:
  1. the measured draft acceptance rate is >= --min-acceptance (FR-03.3), and
  2. decoding with the draft is genuinely faster than without it.

Usage:
    python scripts/validate_draft_pairs.py --model llama3.2 \
        --quant 3b-instruct-q4_K_M --draft llama3.2:1b-instruct-q4_K_M

The draft must share the target's tokenizer (same model family).
Integration scope (NFR-09): needs Ollama running and llama-cpp-python.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_PAIRS_FILE = Path.home() / ".cascade" / "validated_pairs.json"
DEFAULT_MIN_ACCEPT = 0.60
DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in two sentences.",
    "Explain Newton's first law in one paragraph.",
    "List five common uses of the Python dict type.",
]


def measure(model: str, quant: str, draft: str, prompts: list[str], max_tokens: int) -> dict:
    from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner
    from cascade_local.engine.execution.model_runner import ACCEL_DRAFT_MODEL, ACCEL_NONE

    runner = LlamaCppRunner(draft_model_tag=draft, temperature=0.0)
    speeds: dict[str, list[float]] = {}
    acceptance: list[float] = []
    for mode in (ACCEL_NONE, ACCEL_DRAFT_MODEL):
        runner.set_acceleration(mode)
        runner.load(model, quant)
        runner.generate("Say OK.", 8)
        speeds[mode] = []
        for prompt in prompts:
            sample = runner.generate(prompt, max_tokens)
            speeds[mode].append(sample.tok_s)
            if sample.acceptance_rate is not None:
                acceptance.append(sample.acceptance_rate)
            print(f"  {mode:<12} {sample.tok_s:6.2f} tok/s  acceptance={sample.acceptance_rate}", flush=True)
        runner.unload()
    plain, drafted = statistics.mean(speeds[ACCEL_NONE]), statistics.mean(speeds[ACCEL_DRAFT_MODEL])
    return {
        "acceptance_rate": round(statistics.mean(acceptance), 4) if acceptance else 0.0,
        "plain_tok_s": round(plain, 3),
        "draft_tok_s": round(drafted, 3),
        "speedup": round(drafted / plain, 3) if plain else 0.0,
    }


def save(pairs_path: Path, entry: dict) -> None:
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", []) if pairs_path.exists() else []
    existing = [p for p in existing if (p["target_model_id"], p["draft_model_id"]) != (entry["target_model_id"], entry["draft_model_id"])]
    existing.append(entry)
    pairs_path.write_text(json.dumps({"pairs": existing}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Layer 3 draft-model pair")
    parser.add_argument("--model", default="llama3.2", help="target model family (what LocalExecutionBackend's model_id is)")
    parser.add_argument("--quant", default="3b-instruct-q4_K_M")
    parser.add_argument("--draft", default="llama3.2:1b-instruct-q4_K_M")
    parser.add_argument("--pairs-file", default=str(DEFAULT_PAIRS_FILE))
    parser.add_argument("--min-acceptance", type=float, default=DEFAULT_MIN_ACCEPT)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args(argv)

    print(f"Validating {args.model}:{args.quant} <- {args.draft}")
    result = measure(args.model, args.quant, args.draft, DEFAULT_PROMPTS, args.max_tokens)
    validated = result["acceptance_rate"] >= args.min_acceptance and result["speedup"] > 1.0
    entry = {"target_model_id": args.model, "draft_model_id": args.draft, "validated": validated, **result}
    save(Path(args.pairs_file), entry)

    print(f"\nacceptance {result['acceptance_rate']:.1%}, speedup {result['speedup']:.2f}x -> "
          f"{'VALIDATED' if validated else 'NOT validated (Layer 3 stays off for this model on this machine)'}")
    print(f"Saved to {args.pairs_file}")
    return 0 if validated else 1


if __name__ == "__main__":
    sys.exit(main())
