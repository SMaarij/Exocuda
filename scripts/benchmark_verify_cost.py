"""
scripts/benchmark_verify_cost.py -- what does verifying a draft cost the target model?

Speculative decoding pays off when checking N drafted tokens in one batch costs
much less than N single-token steps. On a GPU a small batch costs about one step;
on a CPU it does not, and the shape of that cost decides how long drafts should
be. This times one eval of N tokens (after a fixed prompt, KV cache rewound
between runs) for each N, with the runner's own load settings.

    python scripts/benchmark_verify_cost.py --model qwen2.5-coder:7b-instruct-q4_K_M --sizes 1,2,3,4,5,8,9,16,17
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(model: str, sizes: list[int], repeats: int, n_threads: int | None, n_threads_batch: int | None,
            extra: dict) -> dict:
    from llama_cpp import Llama

    from cascade_local.engine.execution.llamacpp_runner import WORKING_MEMORY_SETTINGS
    from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

    path = OllamaReferenceModelRunner().model_file_path(model)
    kwargs = dict(WORKING_MEMORY_SETTINGS) | extra
    if n_threads:
        kwargs["n_threads"] = n_threads
    if n_threads_batch:
        kwargs["n_threads_batch"] = n_threads_batch
    llm = Llama(model_path=path, n_ctx=2048, use_mmap=False, verbose=False, logits_all=True, **kwargs)
    prompt = llm.tokenize(b"Explain how a hash table works, with an example in Python. " * 20)[:300]
    llm.eval(prompt)
    base = llm.n_tokens
    filler = llm.tokenize(b" the quick brown fox jumps over the lazy dog and keeps running far away", add_bos=False)
    filler = (filler * 4)[:max(sizes)]
    llm.eval(filler[:4])  # warm the batch path
    rows = {}
    for n in sizes:
        times = []
        for _ in range(repeats):
            llm.n_tokens = base
            llm._ctx.kv_cache_seq_rm(-1, base, -1)
            t0 = time.perf_counter()
            llm.eval(filler[:n])
            times.append(time.perf_counter() - t0)
        rows[n] = round(statistics.median(times) * 1000, 1)
        print(f"  {n:3d} tokens: {rows[n]:7.1f} ms  ({rows[n] / rows[sizes[0]]:.2f}x one token)", flush=True)
    return {"model": model, "settings": {k: v for k, v in kwargs.items()}, "ms": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M")
    parser.add_argument("--sizes", default="1,2,3,4,5,6,7,8,9,12,16,17")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--threads-batch", type=int)
    parser.add_argument("--extra", default="{}", help="JSON of extra Llama() settings")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    result = measure(args.model, sizes, args.repeats, args.threads, args.threads_batch, json.loads(args.extra))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
