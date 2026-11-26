"""Usage: python scripts/analysis/wide_verify_costs.py [MODEL_TAG_OR_PATH]

The verify-cost curve beyond 17 tokens. Prompt processing runs at ~150 tokens/s on this laptop
(large batches take llama.cpp's GEMM path), but a 2-16 token verify costs 1.4-4.5 single steps.
Does a wide batch (32-256 tokens) fall into the fast regime, and what do logits at every position
(needed to verify) add? Times one eval of L tokens after a 256-token prefix, KV cache rewound
between runs, with and without per-position logits.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llama_cpp import Llama  # noqa: E402

from engine.execution.llamacpp_runner import WORKING_MEMORY_SETTINGS, LlamaCppRunner  # noqa: E402

PROBE = b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n" * 40
SIZES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def curve(path: str, logits_all: bool, reps: int = 3) -> dict[int, float]:
    llm = Llama(path, n_ctx=1024, verbose=False, use_mmap=False, logits_all=logits_all, **WORKING_MEMORY_SETTINGS)
    ids = list(llm.tokenize(PROBE))
    llm.eval(ids[:256])
    base = llm.n_tokens
    out = {}
    for size in SIZES:
        ts = []
        for _ in range(reps):
            llm.n_tokens = base
            llm._ctx.kv_cache_seq_rm(-1, base, -1)
            t = time.perf_counter()
            llm.eval(ids[256: 256 + size])
            ts.append(1000 * (time.perf_counter() - t))
        out[size] = statistics.median(ts)
    del llm
    return out


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:7b-instruct-q4_K_M"
    path = tag if Path(tag).exists() else LlamaCppRunner(draft_model_tag=None)._resolve_path(tag)
    plain = curve(path, False)
    full = curve(path, True)
    one = plain[1]
    print(f"{'L':>4} {'ms (last logits)':>17} {'ms (all logits)':>16} {'cost (steps)':>13} {'per token':>10}")
    for size in SIZES:
        print(f"{size:4d} {plain[size]:17.1f} {full[size]:16.1f} {full[size] / one:13.2f} {full[size] / one / size:10.3f}",
              flush=True)


if __name__ == "__main__":
    main()
