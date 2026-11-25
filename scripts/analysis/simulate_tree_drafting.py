"""Usage: python scripts/analysis/simulate_tree_drafting.py docs/benchmark/draft_top2_coverage.json

Replay linear vs two-branch drafting on recorded sequences (greedy target).

At each step the policy sees the 0.5B's top-2 tokens and probabilities along the true text
(valid for every token that can still be accepted) and picks the draft with the best expected
tokens per unit of cost; the outcome is then read off the 7B's real tokens.
Cost unit: one single-token 7B step. Draft cost 0.105 per 0.5B token (Addendum 18).
"""
import json
import sys

costs = [0.0, 1.0, 1.38, 1.77, 1.45, 1.94, 2.35, 3.01, 3.06, 3.48, 3.82, 4.27, 4.14, 4.65, 5.25, 5.46, 4.47, 6.03]
D = 0.105
MAXN = 16  # verify batch size limit (nodes + 1 <= 17)

data = json.load(open(sys.argv[1]))["seqs"]


def linear_options(p1, i, n):
    """(expected, cost, spec) for chain lengths 0..L."""
    opts = [(0.0, costs[1], ("lin", 0))]
    whole, exp = 1.0, 0.0
    for L in range(1, min(MAXN, n - i) + 1):
        whole *= p1[i + L - 1]
        exp += whole
        opts.append((exp, costs[L + 1] + D * L, ("lin", L)))
    return opts


def tree_options(p1, p2, i, n, max_m=6):
    opts = []
    pc = 1.0
    chain_exp = 0.0
    for c in range(0, MAXN):
        u = i + c
        if u >= n:
            break
        for m in range(0, max_m + 1):
            nodes = c + 2 + 2 * m
            if nodes > MAXN or u + m >= n:
                break
            cont, w = 0.0, 1.0
            for k in range(1, m + 1):
                w *= p1[u + k]
                cont += w
            exp = chain_exp + pc * (p1[u] + p2[u]) * (1 + cont)
            opts.append((exp, costs[nodes + 1] + D * (c + 1 + 2 * m), ("tree", c, m)))
        pc *= p1[u]
        chain_exp += pc
    return opts


def run(seq, use_tree, cap_m=6):
    tgt, top, p = seq["target"], seq["top"], seq["p"]
    n = len(tgt)
    p1 = [x[0] for x in p]
    p2 = [x[1] for x in p]
    i, cost, steps = 0, 0.0, 0
    while i < n:
        opts = linear_options(p1, i, n)
        if use_tree:
            opts += tree_options(p1, p2, i, n, cap_m)
        exp, c, spec = max(opts, key=lambda o: (1 + o[0]) / o[1])
        acc = 0
        if spec[0] == "lin":
            for k in range(spec[1]):
                if top[i + k][0] == tgt[i + k]:
                    acc += 1
                else:
                    break
        else:
            _, ch, m = spec
            ok = True
            for k in range(ch):
                if top[i + k][0] == tgt[i + k]:
                    acc += 1
                else:
                    ok = False
                    break
            u = i + ch
            if ok and tgt[u] in (top[u][0], top[u][1]):
                acc += 1
                for k in range(1, m + 1):
                    if top[u + k][0] == tgt[u + k]:
                        acc += 1
                    else:
                        break
        i += acc + 1
        cost += c
        steps += 1
    return n, cost


for group in ("code", "open"):
    seqs = [s for s in data if s["group"] == group]
    for name, tree in (("linear (cost-aware, 0.8.0)", False), ("linear or two-branch", True)):
        toks = cst = 0.0
        for s in seqs:
            n, c = run(s, tree)
            toks += n
            cst += c
        print(f"{group:5s} {name:28s} speed-up vs plain {toks / cst:.3f}x")
