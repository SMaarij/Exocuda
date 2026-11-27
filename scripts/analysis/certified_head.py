"""Usage: python scripts/analysis/certified_head.py DATA.json [--positions N] [--save]

Certified sparse output head: can greedy decoding find the 7B's argmax token *exactly* without
computing all 152k logits? For a hidden state h and output rows w_i, a precomputed bound B_i(h) >=
w_i . h lets every row with B_i(h) < m be skipped, where m is the exact logit of a candidate (in
speculative decoding: the drafted token, which the target usually agrees with). Rows whose bound
reaches m are computed exactly; the winner is certified either way -- lossless.

The study reconstructs h at the 7B's own decoding positions from its logits (z = W h, solved by
least squares against the dequantized output matrix; the residual is reported) and measures, for
each bound family, the fraction of the head's work that remains:

- norm:     B_i = |w_i| |h|                       (no setup work per token)
- cluster:  rows k-means-clustered; B_i = mu_c . h + r_c |h|, per cluster  (C dot products first)
- lowrank:  W ~ U S V^T (rank r); B_i = (U S)_i . (V^T h) + |res_i| |h|     (r(d + V) work first)
- clust+lr: cluster centre plus a rank-r model of the within-cluster offsets (tightest here)

Work is counted in multiply-adds relative to the full V x d product.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402

TARGET = "qwen2.5-coder:7b-instruct-q4_K_M"


def output_matrix(path: str) -> np.ndarray:
    reader = gguf.GGUFReader(path)
    t = next(t for t in reader.tensors if t.name == "output.weight")
    w = dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0]))
    return np.ascontiguousarray(w, dtype=np.float32)


def hidden_states(path: str, w: np.ndarray, seqs: list[dict], per_seq: int) -> tuple[np.ndarray, np.ndarray]:
    """h at decoding positions, solved from the model's own logits; returns (H, Z)."""
    n_ctx = max(len(s["prompt"]) + len(s["target"]) for s in seqs) + 16
    llm = Llama(path, n_ctx=n_ctx, logits_all=True, verbose=False, use_mmap=False, flash_attn=True)
    zs = []
    for s in seqs:
        llm.reset()
        ids = list(s["prompt"]) + list(s["target"])
        llm.eval(ids)
        start = len(s["prompt"]) - 1
        pos = np.linspace(start, len(ids) - 2, num=min(per_seq, len(s["target"])), dtype=int)
        zs.append(np.asarray(llm.scores)[pos].astype(np.float64))
    del llm
    z = np.concatenate(zs)                       # (n, V)
    w64 = w.astype(np.float64)
    gram = w64.T @ w64                           # (d, d)
    h = np.linalg.solve(gram, w64.T @ z.T).T     # (n, d)
    return h.astype(np.float32), z.astype(np.float32)


def kmeans(w: np.ndarray, c: int, iters: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cent = w[rng.choice(len(w), c, replace=False)].copy()
    wn = (w * w).sum(1)
    for _ in range(iters):
        assign = np.empty(len(w), dtype=np.int32)
        for i in range(0, len(w), 16384):
            blk = w[i:i + 16384]
            d = wn[i:i + 16384, None] - 2 * blk @ cent.T + (cent * cent).sum(1)[None]
            assign[i:i + 16384] = d.argmin(1)
        for k in range(c):
            m = assign == k
            if m.any():
                cent[k] = w[m].mean(0)
    return cent, assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--positions", type=int, default=12, help="positions sampled per sequence")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    seqs = json.loads(Path(a.data).read_text())["seqs"]
    path = LlamaCppRunner(draft_model_tag=None)._resolve_path(TARGET)
    t0 = time.time()
    w = output_matrix(path)
    V, d = w.shape
    print(f"output matrix {V} x {d} ({time.time() - t0:.0f} s)", flush=True)
    h, z = hidden_states(path, w, seqs, a.positions)
    rec = h @ w.T
    err = np.abs(rec - z).max(1)
    gap = np.sort(z, 1)
    gap = gap[:, -1] - gap[:, -2]
    print(f"{len(h)} positions; reconstruction max |error| median {np.median(err):.3f}; "
          f"top-1 margin median {np.median(gap):.2f}; argmax preserved {np.mean(rec.argmax(1) == z.argmax(1)):.3f}",
          flush=True)
    logits = h @ w.T                             # exact, in the reconstructed space
    m = logits.max(1)                            # the candidate = the true winner (draft accepted)
    hn = np.linalg.norm(h, axis=1)
    full = V * d
    out: dict[str, dict] = {}

    rown = np.linalg.norm(w, axis=1)
    keep = (rown[None] * hn[:, None] >= m[:, None]).mean(1)
    out["norm"] = {"work": float(np.median(keep))}
    print(f"norm bound: rows to compute median {np.median(keep):.3f}", flush=True)

    # The bound offsets below are scaled by |h|, so the gap between the winner and the
    # mean logit matters: report it.
    print(f"winner minus mean logit, in units of |h|: median {np.median((m - logits.mean(1)) / hn):.3f}", flush=True)

    for c in (1024, 4096):
        cent, assign = kmeans(w, c)
        r = np.zeros(c, dtype=np.float32)
        np.maximum.at(r, assign, np.linalg.norm(w - cent[assign], axis=1))
        sizes = np.bincount(assign, minlength=c)
        b = h @ cent.T + r[None] * hn[:, None]   # (n, c)
        alive = b >= m[:, None]
        work = (c + (alive * sizes[None]).sum(1)) / V
        out[f"cluster{c}"] = {"work": float(np.median(work))}
        print(f"cluster bound C={c}: work median {np.median(work):.3f} (p90 {np.quantile(work, 0.9):.3f})", flush=True)
        for rank in (32, 128):
            off = w - cent[assign]
            # one shared basis for within-cluster offsets
            samp = off[np.random.default_rng(1).choice(V, 20000, replace=False)]
            _, _, vt = np.linalg.svd(samp, full_matrices=False)
            basis = vt[:rank]                     # (rank, d)
            coef = off @ basis.T                  # (V, rank)
            res = np.linalg.norm(off - coef @ basis, axis=1)
            hp = h @ basis.T                      # (n, rank)
            rmax = np.zeros(c, dtype=np.float32)
            np.maximum.at(rmax, assign, res)
            # stage 1: cluster bound with the residual radius after the low-rank part is known per row
            # B_i = mu_c.h + coef_i.hp + res_i |h|; per-cluster prefilter with max over rows of coef.hp
            exact_rows = 0
            works = []
            ch = h @ cent.T
            for j in range(len(h)):
                row_b = ch[j, assign] + coef @ hp[j] + res * hn[j]
                works.append((c + V * rank / d + (row_b >= m[j]).sum()) / V)
                exact_rows += 1
            works = np.array(works)
            out[f"cluster{c}+lr{rank}"] = {"work": float(np.median(works))}
            print(f"cluster C={c} + rank {rank} offsets: work median {np.median(works):.3f} "
                  f"(p90 {np.quantile(works, 0.9):.3f})", flush=True)

    for rank in (64, 256):
        samp = w[np.random.default_rng(2).choice(V, 30000, replace=False)]
        _, _, vt = np.linalg.svd(samp, full_matrices=False)
        basis = vt[:rank]
        coef = w @ basis.T
        res = np.linalg.norm(w - coef @ basis, axis=1)
        b = (h @ basis.T) @ coef.T + res[None] * hn[:, None]
        alive = (b >= m[:, None]).sum(1)
        work = (rank * (V + d) / d + alive) / V
        out[f"lowrank{rank}"] = {"work": float(np.median(work))}
        print(f"low-rank bound r={rank}: work median {np.median(work):.3f}", flush=True)
    if a.save:
        (ROOT / "docs" / "benchmark" / "certified_head.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
