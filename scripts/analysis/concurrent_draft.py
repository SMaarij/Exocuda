"""Usage: python scripts/analysis/concurrent_draft.py

Can the draft model run *while* the 7B verifies, on a laptop CPU? Measures the 7B's 4-token
verify latency (the batch the cost-aware policy picks most) with T threads alone, and with
T-k threads while the 0.5B decodes concurrently on k threads in another Python thread
(llama.cpp calls release the GIL), plus how many 0.5B tokens fit into one 7B verify.
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llama_cpp import Llama  # noqa: E402

from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402

PROBE = b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n" * 16


def path(tag: str) -> str:
    return LlamaCppRunner(draft_model_tag=None)._resolve_path(tag)


def load(tag: str, threads: int) -> Llama:
    return Llama(path(tag), n_ctx=2048, n_threads=threads, n_threads_batch=threads, verbose=False,
                 use_mmap=False, flash_attn=True)


def verify_times(llm: Llama, ids: list[int], repeats: int = 25, batch: int = 4, on_start=None) -> list[float]:
    llm.reset()
    llm.eval(ids)
    base = llm.n_tokens
    if on_start:
        on_start()
    out = []
    for i in range(repeats):
        llm.n_tokens = base
        t = time.perf_counter()
        llm.eval(ids[i % 40: i % 40 + batch])
        out.append(time.perf_counter() - t)
    return out


def main() -> None:
    results = {}
    target_tag, draft_tag = "qwen2.5-coder:7b-instruct-q4_K_M", "qwen2.5:0.5b-instruct"
    for t7, t05 in ((4, 0), (4, 1), (4, 2), (4, 4), (3, 1)):
        big = load(target_tag, t7)
        ids = list(big.tokenize(PROBE))[:256]
        stop = threading.Event()
        drafted = [0]
        draft_ms: list[float] = []
        worker = None
        if t05:
            small = load(draft_tag, t05)
            small.eval(ids[:200])

            def run() -> None:
                k = 0
                while not stop.is_set():
                    t = time.perf_counter()
                    small.eval([ids[k % 50]])
                    draft_ms.append(1000 * (time.perf_counter() - t))
                    drafted[0] += 1
                    k += 1
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            time.sleep(0.5)
        def start() -> None:
            drafted[0] = 0
            draft_ms.clear()
        times = verify_times(big, ids, on_start=start)
        stop.set()
        if worker:
            worker.join()
        med = 1000 * statistics.median(times)
        per_verify = drafted[0] / len(times) if t05 else 0.0
        results[f"7B {t7} threads + 0.5B {t05}"] = {"verify4_ms": round(med, 1), "draft_tokens_per_verify": round(per_verify, 2),
                                                     "draft_ms": round(statistics.median(draft_ms), 1) if draft_ms else None}
        print(f"7B {t7} thr, 0.5B {t05} thr concurrently: 7B 4-token verify {med:6.1f} ms; "
              f"0.5B tokens per verify {per_verify:.2f}"
              + (f" ({statistics.median(draft_ms):.1f} ms each)" if draft_ms else ""), flush=True)
        del big
    out = ROOT / "docs" / "benchmark" / "concurrent_draft.json"
    if "--save" in sys.argv:
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
