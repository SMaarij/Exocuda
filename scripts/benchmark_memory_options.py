"""
scripts/benchmark_memory_options.py -- what do llama.cpp's working-memory options
cost and save on this machine?

The model weights are fixed by the quantization; what a context adds on top is
the KV cache (grows with n_ctx and layers) and the compute buffer (grows with
n_ubatch). Each option runs in its own process so committed memory is clean:

    base     f16 KV cache, flash attention off, n_ubatch 512 (llama.cpp default)
    fa       flash attention on
    fa_q8    flash attention + q8_0 K and V cache (half the KV memory)
    fa_q8_ub256 / ub128   ... with a smaller compute batch

Reported: committed private memory and working set after a 400-token prompt and
64 greedy tokens, prompt and decode speed, the KV/compute sizes llama.cpp logs,
and whether the 64 tokens match `base`.

    python scripts/benchmark_memory_options.py --model qwen2.5-coder:7b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GGML_TYPE_Q8_0 = 8
OPTIONS = {
    "base": {},
    "fa": {"flash_attn": True},
    "fa_q8": {"flash_attn": True, "type_k": GGML_TYPE_Q8_0, "type_v": GGML_TYPE_Q8_0},
    "k8": {"type_k": GGML_TYPE_Q8_0},  # K cache only: V needs flash attention to be quantized
    "fa_ub256": {"flash_attn": True, "n_ubatch": 256},
    "ub256": {"n_ubatch": 256},
    "fa_q8_ub256": {"flash_attn": True, "type_k": GGML_TYPE_Q8_0, "type_v": GGML_TYPE_Q8_0, "n_ubatch": 256},
    "fa_q8_ub128": {"flash_attn": True, "type_k": GGML_TYPE_Q8_0, "type_v": GGML_TYPE_Q8_0, "n_ubatch": 128},
}
PROMPT = ("Explain, step by step, how a hash table handles collisions with separate chaining and with "
          "open addressing, compare their memory use and cache behaviour, and give a short Python example "
          "of each. ") * 6


def run_one(model: str, option: str, n_ctx: int, prompt_tokens: int = 400) -> dict:
    import time

    import numpy as np
    import psutil
    from llama_cpp import Llama

    from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

    path = OllamaReferenceModelRunner().model_file_path(model)
    llm = Llama(model_path=path, n_ctx=n_ctx, use_mmap=False, verbose=False, **OPTIONS[option])
    llm.scores = np.ndarray((1, llm.n_vocab()), dtype=np.single)  # as the runner does
    tokens = llm.tokenize(PROMPT.encode())
    while len(tokens) < prompt_tokens:
        tokens = tokens + tokens
    tokens = tokens[:prompt_tokens]
    t0 = time.perf_counter()
    out = []
    t_first = None
    for tok in llm.generate(tokens, top_k=1, temp=0.0, reset=True):
        if t_first is None:
            t_first = time.perf_counter()
        out.append(int(tok))
        if len(out) >= 64:
            break
    t1 = time.perf_counter()
    mem = psutil.Process().memory_full_info()
    return {"option": option, "private_gb": round(mem.private / 2**30, 3), "rss_gb": round(mem.rss / 2**30, 3),
            "prompt_s": round(t_first - t0, 2), "decode_tok_s": round(63 / (t1 - t_first), 2), "tokens": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--options", default=",".join(OPTIONS))
    parser.add_argument("--prompt-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(args.model, args.one, args.n_ctx, args.prompt_tokens)), flush=True)
        return
    rows = []
    for option in args.options.split(","):
        child = subprocess.run([sys.executable, __file__, "--one", option, "--model", args.model,
                                "--n-ctx", str(args.n_ctx), "--prompt-tokens", str(args.prompt_tokens)], capture_output=True, text=True, encoding="utf-8")
        line = next((l for l in child.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"{option}: failed\n{child.stderr[-2000:]}")
            continue
        rows.append(json.loads(line[7:]))
    base = next((r for r in rows if r["option"] == "base"), None)
    print(f"\nmodel {args.model}, n_ctx {args.n_ctx}")
    print(f"{'option':13s} {'committed':>10s} {'RSS':>8s} {'prompt':>8s} {'decode':>10s}  same 64 tokens as base")
    for r in rows:
        same = "-" if base is None else (
            "yes" if r["tokens"] == base["tokens"] else
            f"no (first diff at {next(i for i, (a, b) in enumerate(zip(r['tokens'], base['tokens'])) if a != b)})")
        print(f"{r['option']:13s} {r['private_gb']:9.2f}G {r['rss_gb']:7.2f}G {r['prompt_s']:7.1f}s "
              f"{r['decode_tok_s']:6.2f}tok/s  {same}")
    if args.out:
        args.out.write_text(json.dumps({"model": args.model, "n_ctx": args.n_ctx, "prompt_tokens": args.prompt_tokens,
                                        "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
