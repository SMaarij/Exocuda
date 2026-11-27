"""Usage: python scripts/analysis/head_share.py [--workdir DIR]

How much of one 7B decode step (and of a verify, with logits at every position) is the output head (the V x d vocabulary projection)? Times
single-token decode of GGUFs holding 1 and 2 of the 28 blocks: T(k) = rest + k * layer, so
head + embedding = 2 T(1) - T(2), alongside the full model's step and its 4- and 8-token verifies.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from llama_cpp import Llama  # noqa: E402

from cascade_local.engine.execution.llamacpp_runner import WORKING_MEMORY_SETTINGS, LlamaCppRunner  # noqa: E402
from layer_skip_acceptance import truncate  # noqa: E402

PROBE = b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n" * 16


def times(path: str, rounds: int = 40) -> dict[int, float]:
    llm = Llama(path, n_ctx=1024, verbose=False, use_mmap=False, logits_all=True, **WORKING_MEMORY_SETTINGS)
    ids = list(llm.tokenize(PROBE))[:300]
    llm.eval(ids[:256])
    base = llm.n_tokens
    out = {}
    for size in (1, 4, 8):
        ts = []
        for i in range(rounds if size == 1 else rounds // 2):
            llm.n_tokens = base
            llm._ctx.kv_cache_seq_rm(-1, base, -1)
            t = time.perf_counter()
            llm.eval(ids[256 + i % 30: 256 + i % 30 + size])
            ts.append(1000 * (time.perf_counter() - t))
        out[size] = statistics.median(ts)
    del llm
    return out


def main() -> None:
    work = Path(sys.argv[sys.argv.index("--workdir") + 1]) if "--workdir" in sys.argv else ROOT.parent / "tmp_layerskip"
    work.mkdir(exist_ok=True)
    src = LlamaCppRunner(draft_model_tag=None)._resolve_path("qwen2.5-coder:7b-instruct-q4_K_M")
    res = {}
    for k in (1, 2):
        p = work / f"blocks{k}.gguf"
        truncate(src, list(range(k)), p)
        res[k] = times(str(p))
        p.unlink()
    full = times(src)
    for size in (1, 4, 8):
        head = 2 * res[1][size] - res[2][size]
        layer = res[2][size] - res[1][size]
        print(f"{size} token(s): full {full[size]:6.1f} ms, head+embedding {head:5.1f} ms "
              f"({100 * head / full[size]:.1f}%), one block {layer:5.1f} ms", flush=True)


if __name__ == "__main__":
    main()
