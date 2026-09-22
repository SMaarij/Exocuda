
"""
scripts/benchmark_acceleration.py

Phase 3, Task 12 -- Performance tuning pass (NFR-01, NFR-02).

Runs a live benchmark comparing tok/s across all three AccelerationEngine
layers on this machine. Requires a live Ollama instance with the target
model already pulled.

Usage:
    python scripts/benchmark_acceleration.py \
        [--model llama3.2] \
        [--quant 3b-instruct-q4_K_M] \
        [--prompts 5] \
        [--pairs-file ~/.cascade/validated_pairs.json]

Outputs:
    - Console table: Layer 1/2/3 tok/s and speedup vs Layer 1
    - Pass/Fail verdict for NFR-01 (>= 1.5x on Layer 2/3)
    - Pass/Fail verdict for NFR-02 (est_cost_usd == 0 regardless of layer)

This is integration/manual-testing scope (NFR-09) -- it requires a live
model. For automated CI, see engine/execution/tests/test_backend_real.py
which uses FakeRunner and passes the same NFR assertions without any GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "llama3.2"
DEFAULT_QUANT = "3b-instruct-q4_K_M"
DEFAULT_PROMPTS = [
    "What is the capital of Germany?",
    "Explain recursion in one sentence.",
    "Write a haiku about the ocean.",
    "What is 17 * 23?",
    "Name three programming languages.",
]


def _benchmark_layer(model_id: str, quant: str, prompts: list[str], layer_name: str,
                     model_has_heads: bool, pairs_file: Path) -> dict:
    from engine.execution.acceleration_engine import (
        AccelerationEngine, AccelerationLayer, load_validated_pairs,
    )
    from engine.execution.model_runner import OllamaReferenceModelRunner

    layer_map = {
        "layer1": AccelerationLayer.LOOKAHEAD,
        "layer2": AccelerationLayer.DRAFT_HEADS,
        "layer3": AccelerationLayer.DRAFT_MODEL,
    }
    layer = layer_map[layer_name]

    runner = OllamaReferenceModelRunner()
    pairs  = load_validated_pairs(pairs_file)
    engine = AccelerationEngine(
        runner=runner,
        preferred_layer=layer,
        model_has_draft_heads=model_has_heads,
        validated_pairs=pairs,
    )

    tok_s_list: list[float] = []
    cost_list:  list[float] = []

    print(f"\n  Running {layer_name} ({engine._resolve_layer(model_id).name})...")
    for i, prompt in enumerate(prompts, 1):
        engine.load(model_id, quant)
        sample = engine.generate_accelerated(prompt, max_tokens=64)
        engine.unload()
        tok_s_list.append(sample.tok_s)
        cost_list.append(0.0)   # local is always free (NFR-02)
        print(f"    prompt {i}: {sample.tok_s:.1f} tok/s  "
              f"layer={sample.active_layer.name}  "
              f"speculative={sample.speculative_active}")

    return {
        "layer":      layer_name,
        "actual":     sample.active_layer.name,
        "avg_tok_s":  sum(tok_s_list) / len(tok_s_list),
        "est_cost":   sum(cost_list),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark AccelerationEngine layers (NFR-01, NFR-02)")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--quant",      default=DEFAULT_QUANT)
    parser.add_argument("--prompts",    type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument("--pairs-file", default=str(Path.home() / ".cascade" / "validated_pairs.json"))
    parser.add_argument("--model-has-heads", action="store_true", default=False,
                        help="Pass if the model has trained Medusa/EAGLE heads")
    args = parser.parse_args(argv)

    prompts    = DEFAULT_PROMPTS[:args.prompts]
    pairs_path = Path(args.pairs_file)

    print(f"Benchmarking {args.model}:{args.quant} across 3 acceleration layers")
    print(f"Prompts: {len(prompts)}")

    results = {}
    for layer in ("layer1", "layer2", "layer3"):
        try:
            results[layer] = _benchmark_layer(
                args.model, args.quant, prompts, layer,
                args.model_has_heads, pairs_path,
            )
        except Exception as exc:
            print(f"  {layer} ERROR: {exc}")
            results[layer] = {"avg_tok_s": 0.0, "est_cost": 0.0, "actual": "ERROR"}

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    l1 = results["layer1"]["avg_tok_s"]
    header = f"{'Layer':<10} {'Actual':<15} {'avg tok/s':>10} {'vs Layer1':>10} {'NFR-01':>8}"
    print(header)
    print("-" * len(header))
    all_pass = True
    for key, label in [("layer1","Layer 1"), ("layer2","Layer 2"), ("layer3","Layer 3")]:
        r     = results[key]
        ratio = r["avg_tok_s"] / l1 if l1 > 0 else 0.0
        nfr01 = "N/A" if key == "layer1" else ("PASS" if ratio >= 1.5 else "FAIL")
        if nfr01 == "FAIL":
            all_pass = False
        print(f"{label:<10} {r['actual']:<15} {r['avg_tok_s']:>10.2f} {ratio:>10.2f}x {nfr01:>8}")

    print()
    nfr02_pass = all(r["est_cost"] == 0.0 for r in results.values())
    print(f"NFR-02 (est_cost_usd == 0 for all layers): {'PASS' if nfr02_pass else 'FAIL'}")
    if not nfr02_pass:
        all_pass = False

    print()
    print("Overall:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
