"""Usage: python scripts/analysis/decode_threads.py [MODEL_TAG_OR_PATH] [--rounds N]

Is the 7B's single-token decode bound by memory bandwidth or by arithmetic on this CPU? Times
single-token decode and 4/8-token verifies at n_threads (decode) x n_threads_batch (verify)
settings, rounds interleaved so drift and heat hit every setting alike. If decode were bandwidth
bound, more threads (the SMT siblings) would not help; if arithmetic bound, they would.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from engine.execution.llamacpp_runner import WORKING_MEMORY_SETTINGS, LlamaCppRunner  # noqa: E402

PROBE = b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n" * 16
SETTINGS = [(4, 8), (6, 8), (8, 8), (3, 8), (4, 4)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3
    tag = args[0] if args else "qwen2.5-coder:7b-instruct-q4_K_M"
    path = tag if Path(tag).exists() else LlamaCppRunner(draft_model_tag=None)._resolve_path(tag)
    llm = Llama(path, n_ctx=2048, verbose=False, use_mmap=False, **WORKING_MEMORY_SETTINGS)
    ids = list(llm.tokenize(PROBE))[:300]
    llm.eval(ids[:256])
    base = llm.n_tokens
    res: dict[tuple, dict[int, list[float]]] = {s: {1: [], 4: [], 8: []} for s in SETTINGS}
    for _ in range(rounds):
        for s in SETTINGS:
            llama_cpp.llama_set_n_threads(llm._ctx.ctx, *s)
            for size in (1, 4, 8):
                for i in range(8):
                    llm.n_tokens = base
                    llm._ctx.kv_cache_seq_rm(-1, base, -1)
                    t = time.perf_counter()
                    llm.eval(ids[256 + i: 256 + i + size])
                    res[s][size].append(1000 * (time.perf_counter() - t))
    for s in SETTINGS:
        m = {k: statistics.median(v) for k, v in res[s].items()}
        print(f"decode threads {s[0]}, batch threads {s[1]}: 1 token {m[1]:6.1f} ms  "
              f"4 tokens {m[4]:6.1f} ms  8 tokens {m[8]:6.1f} ms", flush=True)


if __name__ == "__main__":
    main()
