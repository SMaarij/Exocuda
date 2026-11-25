"""Usage: python scripts/analysis/sibling_replay.py DATA.json   (from draft_top2_coverage.py, top-4 per position)

Exact replay of "sibling" drafts against linear drafts, charged at the measured CPU verify costs.

Shape (c, k, m): a chain of c draft-top-1 tokens; at the next position the draft's top k tokens
side by side (tree siblings, verified in one batch with llama.cpp sequence ids); the top-1
sibling continues for m more top-1 tokens. k = 1 is a plain linear draft of c + 1 + m tokens.
Batch = 1 + c + k + m. At each step the policy picks the shape with the best expected tokens
per unit of cost; the outcome is read off the 7B's real tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

D = 0.105
MAXB = 17  # largest batch in the cost table


def options(p, i, n, costs, max_k):
    out = []
    p1 = [x[0] for x in p]
    pc, chain_exp = 1.0, 0.0
    for c in range(0, MAXB):
        u = i + c
        if u >= n:
            break
        for k in range(1, max_k + 1):
            sib = sum(p[u][r] for r in range(k))
            for m in range(0, MAXB):
                batch = 1 + c + k + m
                if batch > MAXB or u + m >= n:
                    break
                cont, w = 0.0, 1.0
                for t in range(1, m + 1):
                    w *= p1[u + t]
                    cont += w
                exp = chain_exp + pc * (sib + p1[u] * cont)
                out.append((exp, costs[batch] + D * (c + 1 + m), (c, k, m)))
        pc *= p1[u]
        chain_exp += pc
    return out


def replay(seqs, costs, max_k):
    toks = cost = 0.0
    shapes = {}
    for s in seqs:
        tgt, top, p = s["target"], s["top"], s["p"]
        n, i = len(tgt), 0
        while i < n:
            exp, cst, (c, k, m) = max(options(p, i, n, costs, max_k), key=lambda o: (1 + o[0]) / o[1])
            if exp == 0.0 and False:
                pass
            plain = (1.0 / costs[1])
            if (1 + exp) / cst < plain:  # nothing beats plain decoding
                cost += costs[1]
                i += 1
                continue
            shapes[(c, k, m)] = shapes.get((c, k, m), 0) + 1
            acc = 0
            ok = True
            for t in range(c):
                if top[i + t][0] != tgt[i + t]:
                    ok = False
                    break
                acc += 1
            u = i + c
            if ok and tgt[u] in top[u][:k]:
                acc += 1
                if tgt[u] == top[u][0]:
                    for t in range(1, m + 1):
                        if top[u + t][0] != tgt[u + t]:
                            break
                        acc += 1
            cost += cst
            i += acc + 1
        toks += n
    return toks / cost, shapes


def main() -> None:
    seqs = json.loads(Path(sys.argv[1]).read_text())["seqs"]
    tables = json.loads((Path.home() / ".cascade" / "verify_costs.json").read_text())
    for key, costs in tables.items():
        costs = [float(x) for x in costs[:MAXB + 1]]
        print(f"cost table {key[:12]}: " + " ".join(f"{x:.2f}" for x in costs[1:]))
        for group in ("code", "open"):
            sub = [s for s in seqs if s["group"] == group]
            base, _ = replay(sub, costs, 1)
            for k in (2, 3, 4):
                r, shapes = replay(sub, costs, k)
                top = sorted(shapes.items(), key=lambda x: -x[1])[:4]
                print(f"  {group}: linear {base:.3f}x  siblings up to {k}: {r:.3f}x ({(r / base - 1) * 100:+.1f}%)  "
                      f"common shapes (c,k,m): {top}")


if __name__ == "__main__":
    main()
