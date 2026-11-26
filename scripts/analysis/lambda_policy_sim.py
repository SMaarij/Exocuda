"""Usage: python scripts/analysis/lambda_policy_sim.py DATA.json

Draft length for long-run throughput instead of per-step rate.

The shipped cost-aware policy picks, each step, the draft length L maximizing that step's rate
(1 + E[accepted | L]) / C(L), with C(L) = verify(L+1) + draft cost. Long-run throughput of a
renewal process is maximized instead by the L maximizing (1 + E[accepted | L]) - lam * C(L), where
lam is the optimal long-run rate itself (Dinkelbach). The two differ whenever a step's options
have different costs: the ratio rule ignores that a cheap step lets the next one start sooner.
lam is estimated online as tokens / cost so far (or fixed at the best value, as an upper bound).

Exact replay on the 7B's greedy outputs with the 0.5B's per-position probabilities (DATA.json from
draft_top2_coverage.py) and this machine's measured verify-cost table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_L = 16


def load_costs() -> list[float]:
    tables = json.loads((Path.home() / ".cascade" / "verify_costs.json").read_text())
    return [float(x) for x in tables[[k for k in tables if k.startswith("v2")][0]][:MAX_L + 2]]


def accepted(top, tgt, i, L):
    a = 0
    while a < L and top[i + a][0] == tgt[i + a]:
        a += 1
    return a


def step_cost(costs, L, delta):
    # catch-up eval of the newly committed tokens (~1.5 draft evals) + drafting L-1 more, then verify
    return (1.5 * delta + delta * max(L - 1, 0) if L else 0.0) + costs[L + 1]


def options(p1, i, n, costs, delta):
    out, whole, exp = [(0, 1.0, step_cost(costs, 0, delta))], 1.0, 0.0
    for L in range(1, min(MAX_L, n - i) + 1):
        whole *= p1[i + L - 1]
        exp += whole
        out.append((L, 1.0 + exp, step_cost(costs, L, delta)))
    return out


def run(seq, costs, delta, policy, lam0=None):
    tgt, top = seq["target"], seq["top"]
    p1 = [x[0] for x in seq["p"]]
    n, i, t = len(tgt), 0, 0.0
    lam = lam0
    while i < n:
        opts = options(p1, i, n, costs, delta)
        if policy == "ratio":
            L = max(opts, key=lambda o: o[1] / o[2])[0]
        else:
            cur = lam if lam is not None else 1.0 / costs[1]
            L = max(opts, key=lambda o: o[1] - cur * o[2])[0]
        t += step_cost(costs, L, delta)
        i += accepted(top, tgt, i, L) + 1
        if policy == "online":
            lam = i / t
    return n, t


def agg(seqs, *a, **k):
    tok = cost = 0.0
    for s in seqs:
        n, t = run(s, *a, **k)
        tok += n
        cost += t
    return tok / cost


def main() -> None:
    seqs = json.loads(Path(sys.argv[1]).read_text())["seqs"]
    costs = load_costs()
    delta = 0.13
    for group in ("code", "open"):
        sub = [s for s in seqs if s["group"] == group]
        base = agg(sub, costs, delta, "ratio")
        onl = agg(sub, costs, delta, "online")
        best = max((agg(sub, costs, delta, "fixed", lam0=l / 100), l / 100) for l in range(40, 400, 5))
        print(f"{group}: ratio (shipped) {base:.3f}x | lambda online {onl:.3f}x ({(onl / base - 1) * 100:+.1f}%) | "
              f"lambda best fixed {best[0]:.3f}x at {best[1]:.2f} ({(best[0] / base - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
