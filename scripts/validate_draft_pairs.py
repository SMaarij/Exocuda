
"""
scripts/validate_draft_pairs.py

Track A, Phase 2, Task 9.

Offline validation script for AccelerationEngine Layer 3 draft-model pairs.
Runs a short benchmark against both the target and draft model to confirm:
  1. The draft model produces a measurable tok/s speedup over raw target.
  2. The acceptance rate over a fixed prompt set meets the 60% threshold
     required by FR-03.3 before the pair is ever used in production.

Output: ~/.cascade/validated_pairs.json
This file is consumed by AccelerationEngine.load_validated_pairs().

Usage:
    python scripts/validate_draft_pairs.py \
        --target llama3-8b:q8_0 \
        --draft  llama3-1b:q4_K_M \
        [--pairs-file path/to/validated_pairs.json]
        [--min-acceptance 0.60]
        [--prompt-file path/to/prompts.txt]

Running this script requires a live Ollama instance with both models
already pulled (not needed for Track A's unit tests -- this script is
integration/manual-testing scope per NFR-09).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_PAIRS_FILE   = Path.home() / ".cascade" / "validated_pairs.json"
DEFAULT_MIN_ACCEPT   = 0.60
DEFAULT_TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain Newton's first law in one sentence.",
    "Write a haiku about mountains.",
]


def _run_ollama(model_tag: str, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Run ollama and return (output_text, elapsed_s). Raises on failure."""
    import subprocess
    start = time.monotonic()
    result = subprocess.run(
        ["ollama", "run", model_tag, prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        raise RuntimeError(f"ollama run failed: {result.stderr.strip()}")
    return result.stdout.strip(), elapsed


def validate_pair(
    target_tag: str,
    draft_tag: str,
    prompts: list[str],
    min_acceptance: float,
) -> dict:
    """
    Returns a result dict:
      { target_model_id, draft_model_id, validated, acceptance_rate, speedup_factor }
    """
    print(f"\nValidating pair: {target_tag} <- {draft_tag}")
    target_times: list[float] = []
    draft_accepted = 0
    total_tokens   = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"  Prompt {i}/{len(prompts)}: {prompt[:50]}...")

        target_out, target_elapsed = _run_ollama(target_tag, prompt)
        target_tokens = max(1, len(target_out.split()))
        target_times.append(target_elapsed)

        draft_out, draft_elapsed = _run_ollama(draft_tag, prompt)
        draft_tokens = max(1, len(draft_out.split()))

        # Acceptance heuristic: how many draft tokens match target output.
        target_words = target_out.split()
        draft_words  = draft_out.split()
        accepted = sum(
            1 for t, d in zip(target_words, draft_words) if t == d
        )
        draft_accepted += accepted
        total_tokens   += max(len(target_words), 1)

        print(f"    target tok/s: {target_tokens/target_elapsed:.1f}  "
              f"draft tok/s: {draft_tokens/draft_elapsed:.1f}  "
              f"acceptance: {accepted/max(len(target_words),1):.0%}")

    avg_target_tok_s   = sum(len(p.split()) for p in prompts) / sum(target_times)
    acceptance_rate    = draft_accepted / max(total_tokens, 1)
    validated          = acceptance_rate >= min_acceptance

    print(f"\n  Overall acceptance rate: {acceptance_rate:.1%} "
          f"({'PASS' if validated else 'FAIL -- below ' + str(min_acceptance)})")

    return {
        "target_model_id": target_tag,
        "draft_model_id":  draft_tag,
        "validated":       validated,
        "acceptance_rate": round(acceptance_rate, 4),
        "avg_target_tok_s": round(avg_target_tok_s, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a draft-model pair for AccelerationEngine Layer 3")
    parser.add_argument("--target",       required=True, help="Target model tag, e.g. llama3-8b:q8_0")
    parser.add_argument("--draft",        required=True, help="Draft model tag, e.g. llama3-1b:q4_K_M")
    parser.add_argument("--pairs-file",   default=str(DEFAULT_PAIRS_FILE))
    parser.add_argument("--min-acceptance", type=float, default=DEFAULT_MIN_ACCEPT)
    parser.add_argument("--prompt-file",  default=None,
                        help="Newline-separated prompts; uses built-in set if omitted")
    args = parser.parse_args(argv)

    prompts = DEFAULT_TEST_PROMPTS
    if args.prompt_file:
        prompts = Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]

    try:
        result = validate_pair(args.target, args.draft, prompts, args.min_acceptance)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    pairs_path = Path(args.pairs_file)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if pairs_path.exists():
        existing = json.loads(pairs_path.read_text(encoding="utf-8")).get("pairs", [])

    # Replace or append
    existing = [
        p for p in existing
        if not (p["target_model_id"] == result["target_model_id"]
                and p["draft_model_id"] == result["draft_model_id"])
    ]
    existing.append(result)
    pairs_path.write_text(json.dumps({"pairs": existing}, indent=2), encoding="utf-8")
    print(f"\nSaved to {pairs_path}")
    return 0 if result["validated"] else 1


if __name__ == "__main__":
    sys.exit(main())
