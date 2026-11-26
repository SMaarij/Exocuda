"""Usage: python scripts/analysis/online_rerank_sim.py DATA.json

Online re-ranking of the draft's top-4 from verify outcomes. The 7B's greedy token is in the 0.5B's
top 4 far more often than it is the 0.5B's first choice (open text: 0.92 vs 0.72). Every verify
step reveals which candidate the 7B wanted, a free label. A tiny online model, updated with a
perceptron rule and no gradients through any LM, rescores the candidates:

    score(c) = log p(c) + bias[c] + pair[top1, c] (+ context pair[prev token, c])

Accuracy is measured online (predict, then learn) in the order the sequences were generated, per
group and with groups mixed; "oracle" is top-4 coverage.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def run(seqs, lr=1.0, use_pair=True, use_ctx=True, use_bias=True, margin=0.5):
    bias = defaultdict(float)
    pair = defaultdict(float)
    ctx = defaultdict(float)
    n = base = hit = 0
    for s in seqs:
        prev = None
        for t, top, p in zip(s["target"], s["top"], s["p"]):
            def score(i):
                c = top[i]
                v = math.log(max(p[i], 1e-9))
                if use_bias:
                    v += bias[c]
                if use_pair:
                    v += pair[(top[0], c)]
                if use_ctx and prev is not None:
                    v += ctx[(prev, c)]
                return v
            scores = [score(i) for i in range(len(top))]
            pick = max(range(len(top)), key=lambda i: scores[i])
            n += 1
            base += top[0] == t
            hit += top[pick] == t
            if t in top:  # the label: which candidate the 7B chose
                j = top.index(t)
                # perceptron with margin: push the chosen candidate above every other one
                for i in range(len(top)):
                    if i != j and scores[i] + margin > scores[j]:
                        for table, key_j, key_i in ((bias, top[j], top[i]) if use_bias else (None, None, None),
                                                    (pair, (top[0], top[j]), (top[0], top[i])) if use_pair else (None, None, None),
                                                    (ctx, (prev, top[j]), (prev, top[i])) if use_ctx and prev is not None else (None, None, None)):
                            if table is not None:
                                table[key_j] += lr
                                table[key_i] -= lr
            prev = t
    return n, base / n, hit / n


def main() -> None:
    seqs = json.loads(Path(sys.argv[1]).read_text())["seqs"]
    for name, sub in (("code", [s for s in seqs if s["group"] == "code"]),
                      ("open", [s for s in seqs if s["group"] == "open"]),
                      ("mixed", seqs)):
        cov = sum(t in top for s in sub for t, top in zip(s["target"], s["top"])) / sum(len(s["target"]) for s in sub)
        print(f"{name}: oracle top-4 {cov:.3f}")
        for label, kw in (("bias", dict(use_pair=False, use_ctx=False)),
                          ("pair", dict(use_bias=False, use_ctx=False)),
                          ("ctx", dict(use_bias=False, use_pair=False)),
                          ("all", {})):
            for lr in (0.25, 1.0):
                n, base, hit = run(sub, lr=lr, **kw)
                print(f"   {label:5s} lr {lr:4.2f}: top-1 {base:.3f} -> {hit:.3f} ({(hit - base) * 100:+.1f} pts, n={n})")


if __name__ == "__main__":
    main()
