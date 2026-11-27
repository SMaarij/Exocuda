"""Usage:
  python scripts/analysis/eagle_gate.py collect OUT.npz [--tokens N]
  python scripts/analysis/eagle_gate.py probe OUT.npz

Information gate for EAGLE-style drafting on this stack: does the 7B's own hidden state help
predict its next greedy token beyond what the 0.5B draft knows?

collect: runs local text (Python sources and Markdown docs already on disk) through the 7B with
per-position logits and final hidden states (embedding=True, pooling none: the post-norm state h,
with W h = logits), and through the 0.5B; stores h_t (float16), the 7B's argmax and the 0.5B's
top-4 tokens/probabilities at every position, and the input token ids.

probe: fits, in closed form (ridge), the simplest EAGLE-like predictor
    h_{t+1} ~ A h_t + B e(x_{t+1}) + c        (e = the 7B's input embedding of the next token)
on training documents, and on held-out documents compares, for the 7B's greedy token at t+1:
the 0.5B's top-1; the predictor's full-vocabulary argmax (W h_hat); and the 0.5B's top-4
re-ranked by log p_0.5B + alpha * (W h_hat)[candidate].
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TARGET = "E:/Ollama/models/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"
WINDOW = 384


def documents() -> list[tuple[str, str]]:
    site = Path(np.__file__).resolve().parents[1]
    code = sorted(p for p in site.glob("*/*.py") if 4000 < p.stat().st_size < 60000)
    prose = sorted(p for p in list(site.rglob("*.md")) + list((ROOT / "docs").glob("*.md")) if p.stat().st_size > 3000)
    rng = random.Random(0)
    rng.shuffle(code)
    rng.shuffle(prose)
    out = [("code", str(p)) for p in code[:120]] + [("prose", str(p)) for p in prose[:120]]
    rng.shuffle(out)
    return out


def collect(out: Path, budget: int) -> None:
    import llama_cpp
    from llama_cpp import Llama

    from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner

    big = Llama(TARGET, n_ctx=WINDOW + 16, verbose=False, use_mmap=False, flash_attn=True, n_ubatch=256,
                logits_all=True, embedding=True, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE)
    small = Llama(LlamaCppRunner(draft_model_tag=None)._resolve_path("qwen2.5:0.5b-instruct"), n_ctx=WINDOW + 16,
                  verbose=False, use_mmap=False, flash_attn=True, logits_all=True)
    d = big.n_embd()
    H, Y, X, T4, P4, G, DOC = [], [], [], [], [], [], []
    total = 0
    for doc_id, (group, path) in enumerate(documents()):
        if total >= budget:
            break
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        ids = big.tokenize(text.encode("utf-8"))[1: WINDOW + 1]
        if len(ids) < 64:
            continue
        big.reset()
        big.eval(ids)
        z = np.asarray(big.scores[: len(ids)])
        h = np.stack([np.ctypeslib.as_array(llama_cpp.llama_get_embeddings_ith(big._ctx.ctx, i), shape=(d,))
                      for i in range(len(ids))])
        small.reset()
        small.eval(ids)
        zs = np.asarray(small.scores[: len(ids)])[:, : z.shape[1]]
        top = np.argsort(-zs, axis=1)[:, :4]
        e = np.exp(zs - zs.max(1, keepdims=True))
        prob = np.take_along_axis(e / e.sum(1, keepdims=True), top, 1)
        H.append(h.astype(np.float16))
        Y.append(z.argmax(1).astype(np.int32))
        X.append(np.asarray(ids, dtype=np.int32))
        T4.append(top.astype(np.int32))
        P4.append(prob.astype(np.float32))
        G.append(np.full(len(ids), 0 if group == "code" else 1, dtype=np.int8))
        DOC.append(np.full(len(ids), doc_id, dtype=np.int32))
        total += len(ids)
        print(f"{total} tokens ({group})", flush=True)
    np.savez(out, H=np.concatenate(H), Y=np.concatenate(Y), X=np.concatenate(X), T4=np.concatenate(T4),
             P4=np.concatenate(P4), G=np.concatenate(G), DOC=np.concatenate(DOC))


def probe(path: Path) -> None:
    import gguf
    from gguf.quants import dequantize

    data = np.load(path)
    H, Y, X, T4, P4, G, DOC = (data[k] for k in ("H", "Y", "X", "T4", "P4", "G", "DOC"))
    reader = gguf.GGUFReader(TARGET)
    tens = {t.name: t for t in reader.tensors}
    d = H.shape[1]
    W = dequantize(tens["output.weight"].data, tens["output.weight"].tensor_type).reshape(-1, d).astype(np.float32)
    E = dequantize(tens["token_embd.weight"].data, tens["token_embd.weight"].tensor_type).reshape(-1, d).astype(np.float32)
    # pairs (t, t+1) within one document: input h_t, x_{t+1}; target h_{t+1}; label Y[t+1]
    same = DOC[:-1] == DOC[1:]
    idx = np.nonzero(same)[0]
    docs = np.unique(DOC)
    rng = np.random.default_rng(0)
    test_docs = set(rng.choice(docs, size=max(1, len(docs) // 5), replace=False).tolist())
    is_test = np.array([DOC[i] in test_docs for i in idx])
    tr, te = idx[~is_test], idx[is_test]
    hs = float(np.median(np.linalg.norm(H[tr].astype(np.float32), axis=1)))
    es = float(np.median(np.linalg.norm(E[X[tr + 1]], axis=1)))

    def feats(i):
        return np.hstack([H[i].astype(np.float32) / hs, E[X[i + 1]] / es, np.ones((len(i), 1), np.float32)])

    F = feats(tr)
    T = H[tr + 1].astype(np.float32)
    print(f"train pairs {len(tr)}, test pairs {len(te)}", flush=True)
    for lam in (0.0003, 0.003, 0.03):
        gram = F.T @ F + lam * len(tr) / 1000 * np.eye(F.shape[1], dtype=np.float32)
        M = np.linalg.solve(gram, F.T @ T)
        Ft = feats(te)
        hhat = Ft @ M
        logits = hhat @ W.T
        lab = Y[te + 1]
        full = (logits.argmax(1) == lab)
        small1 = (T4[te + 1, 0] == lab)
        cand = T4[te + 1]
        cand_logit = np.take_along_axis(logits, cand, 1)
        logp = np.log(np.maximum(P4[te + 1], 1e-9))
        cover = (cand == lab[:, None]).any(1)
        line = [f"ridge {lam:6.1f}"]
        for g, name in ((0, "code"), (1, "prose")):
            m = G[te + 1] == g
            best = max((((np.take_along_axis(cand, (logp + a * cand_logit).argmax(1)[:, None], 1)[:, 0] == lab)[m]).mean(), a)
                       for a in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 1e6))
            line.append(f"{name}: 0.5B top-1 {small1[m].mean():.3f} | linear predictor {full[m].mean():.3f} | "
                        f"0.5B top-4 re-ranked {best[0]:.3f} (alpha {best[1]:g}) | top-4 cover {cover[m].mean():.3f}")
        print("\n   ".join(line), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "probe"])
    ap.add_argument("path")
    ap.add_argument("--tokens", type=int, default=30000)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(Path(a.path), a.tokens)
    else:
        probe(Path(a.path))


if __name__ == "__main__":
    main()
