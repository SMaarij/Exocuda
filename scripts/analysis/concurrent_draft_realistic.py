"""Usage: python scripts/analysis/concurrent_draft_realistic.py

The 7B verify latency at the runner's real thread settings (n_threads 4, n_threads_batch 8 by
default) and at batch threads 4, alone and with the 0.5B decoding concurrently on 2 or 4 threads.
Corrects concurrent_draft.py, which pinned batch threads to 4 (Addendum 25).
"""
import statistics
import sys
import threading
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import llama_cpp  # noqa: E402
from llama_cpp import Llama  # noqa: E402
from scripts.analysis.concurrent_draft import PROBE, path  # noqa: E402

big = Llama(path("qwen2.5-coder:7b-instruct-q4_K_M"), n_ctx=2048, verbose=False, use_mmap=False, flash_attn=True)
small = Llama(path("qwen2.5:0.5b-instruct"), n_ctx=2048, verbose=False, use_mmap=False, flash_attn=True)
print("defaults", big.context_params.n_threads, big.context_params.n_threads_batch)
ids = list(big.tokenize(PROBE))[:256]
big.eval(ids)
small.eval(ids[:200])
base = big.n_tokens
sbase = small.n_tokens


def verify(batch, reps=20):
    out = []
    for i in range(reps):
        big.n_tokens = base
        t = time.perf_counter()
        big.eval(ids[i: i + batch])
        out.append(time.perf_counter() - t)
    return 1000 * statistics.median(out)


for bt in (8, 4):
    llama_cpp.llama_set_n_threads(big._ctx.ctx, 4, bt)
    for st in (0, 2, 4):
        stop = threading.Event()
        made = [0]
        w = None
        if st:
            llama_cpp.llama_set_n_threads(small._ctx.ctx, st, st)

            def run():
                k = 0
                while not stop.is_set():
                    if small.n_tokens > sbase + 100:
                        small.n_tokens = sbase
                    small.eval([ids[k % 50]])
                    made[0] += 1
                    k += 1
            w = threading.Thread(target=run, daemon=True)
            w.start()
            time.sleep(0.3)
        made[0] = 0
        t0 = time.perf_counter()
        res = {b: verify(b) for b in (1, 2, 4, 8)}
        el = time.perf_counter() - t0
        stop.set()
        if w:
            w.join()
        print(f"7B batch threads {bt}, 0.5B on {st}: verify ms " +
              " ".join(f"L{b}={v:.0f}" for b, v in res.items()) + f"  draft tok/s {made[0] / el:.1f}", flush=True)
