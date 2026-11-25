"""Usage: python scripts/analysis/cost_shape_replay.py [draft_top2_coverage.json] [verify_costs.json key]

Does the *shape* of the verification-cost curve matter for choosing draft lengths?

Published adaptive-length methods model verification cost as constant (SpecDec++: length set by
acceptance alone) or as a smooth function of the number of tokens (SmartSpec / AdaSpec-style
latency models, typically affine). On a CPU the measured curve is not smooth: llama.cpp's
repacked quantized kernels make some batch sizes cheaper than smaller ones (4 tokens cost less
than 3, 16 less than 15). This replays the cost-aware policy on recorded sequences (0.5B top-1
tokens and probabilities along the 7B's greedy text) three ways -- the length is chosen with
each cost model, but every step is charged at the MEASURED cost:

  measured   lengths chosen with the measured table (what Cascade ships)
  affine     lengths chosen with the least-squares line through the measured table
  flat       lengths chosen as if verification cost did not depend on length (cost = 1)
             -- acceptance-only stopping, capped at 16
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = 0.105  # 0.5B drafted-token cost relative to one 7B step (Addendum 18)
MAX_L = 16


def choose(p1: list[float], i: int, n: int, model: list[float]) -> int:
    best, best_rate, whole, exp = 0, 1.0 / model[1], 1.0, 0.0
    for L in range(1, min(MAX_L, n - i) + 1):
        whole *= p1[i + L - 1]
        exp += whole
        rate = (1 + exp) / (model[L + 1] + D * L)
        if rate > best_rate:
            best, best_rate = L, rate
    return best


def replay(seqs: list[dict], model: list[float], true: list[float]) -> float:
    toks = cost = 0.0
    for s in seqs:
        tgt, top, p = s["target"], s["top"], s["p"]
        p1 = [x[0] for x in p]
        n, i = len(tgt), 0
        while i < n:
            L = choose(p1, i, n, model)
            acc = 0
            for k in range(L):
                if top[i + k][0] != tgt[i + k]:
                    break
                acc += 1
            cost += true[L + 1] + D * L
            i += acc + 1
        toks += n
    return toks / cost


def models(true: list[float]) -> dict[str, list[float]]:
    xs = np.arange(1, len(true))
    a, b = np.polyfit(xs, np.asarray(true[1:]), 1)
    affine = [0.0] + [float(a * x + b) for x in xs]
    # normalise so one token costs 1 in every model (only ratios matter)
    affine = [0.0] + [v / affine[1] for v in affine[1:]]
    flat = [0.0] + [1.0] * (len(true) - 1)
    return {"measured": true, "affine": affine, "flat": flat}


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/benchmark/draft_top2_coverage.json"
    seqs = json.loads(data.read_text())["seqs"]
    tables = json.loads((Path.home() / ".cascade" / "verify_costs.json").read_text())
    for key, true in tables.items():
        true = [float(x) for x in true[:MAX_L + 2]]
        print(f"cost table {key[:40]}...: " + " ".join(f"{x:.2f}" for x in true[1:]))
        for name, model in models(true).items():
            r = {g: replay([s for s in seqs if s["group"] == g], model, true) for g in ("code", "open")}
            print(f"  lengths chosen with {name:9s} code {r['code']:.3f}x  open {r['open']:.3f}x")


if __name__ == "__main__":
    main()
