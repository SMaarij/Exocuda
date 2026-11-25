"""Usage: python scripts/analysis/shadow_draft_sim.py DATA.json

Time-based replay of shadow drafting vs the shipped sequential scheme.

Unit of time: one single-token 7B step. Sequential (shipped): each step the 0.5B catches up
(one eval of the newly committed tokens, ~1.5 token-evals) and drafts L-1 more tokens, then the
7B verifies L+1 tokens -- the two take turns on the same cores. Shadow: the 0.5B runs on the
spare SMT threads all the time, extending its own greedy continuation; the 7B never waits: each
step it verifies the longest worthwhile prefix of the draft tokens already made (possibly none).
A draft token is usable only while the draft's chain agrees with what the 7B committed; after a
rejection the 0.5B re-syncs (catch-up eval) and starts over.

Measured on this laptop (concurrent_draft.json): 0.5B token on 4 threads ~0.13 of a 7B step
when alone; on 2 SMT threads beside the 7B ~0.2-0.3, while slowing the 7B verify by ~3%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_L = 16


def rate_choice(p1, i, avail, costs, d_cost):
    best, best_rate, whole, exp = 0, 1.0 / costs[1], 1.0, 0.0
    for L in range(1, min(MAX_L, avail) + 1):
        whole *= p1[i + L - 1]
        exp += whole
        rate = (1 + exp) / (costs[L + 1] + d_cost * L)
        if rate > best_rate:
            best, best_rate = L, rate
    return best


def accepted(top, tgt, i, L):
    a = 0
    while a < L and top[i + a][0] == tgt[i + a]:
        a += 1
    return a


def sequential(seq, costs, delta):
    tgt, top, p = seq["target"], seq["top"], seq["p"]
    p1 = [x[0] for x in p]
    n, i, t = len(tgt), 0, 0.0
    while i < n:
        L = rate_choice(p1, i, n - i, costs, delta)
        t += 1.5 * delta + delta * max(L - 1, 0) + costs[L + 1]  # catch-up + drafting, then verify
        i += accepted(top, tgt, i, L) + 1
    return n / t


def shadow(seq, costs, delta, slow=1.03, catchup=1.5):
    tgt, top, p = seq["target"], seq["top"], seq["p"]
    p1 = [x[0] for x in p]
    n, i, t = len(tgt), 0, 0.0
    ready = 0.0       # draft tokens made beyond the committed position (fractional while in flight)
    debt = catchup    # draft work still needed before it produces tokens (re-sync)
    while i < n:
        avail = min(int(ready), n - i)
        L = rate_choice(p1, i, avail, costs, 0.0)  # drafts already exist: no draft cost on the critical path
        dur = costs[L + 1] * slow
        # the draft works during this verify
        work = dur / delta
        used = min(work, debt)
        debt -= used
        ready += work - used
        a = accepted(top, tgt, i, L)
        t += dur
        if a == L and i + L < n and top[i + L][0] == tgt[i + L]:
            # every draft accepted and the draft also predicted the 7B's own next token: its chain stays valid
            ready -= L + 1
            ready = max(ready, 0.0)
        else:
            ready, debt = 0.0, catchup
        i += a + 1
    return n / t


def shadow_wait(seq, costs, delta, slow=1.03, catchup=1.5):
    """Shadow drafting where the 7B may also wait for k more draft tokens when that pays,
    counting the wait as time (the draft keeps working; the 7B idles)."""
    tgt, top, p = seq["target"], seq["top"], seq["p"]
    p1 = [x[0] for x in p]
    n, i, t = len(tgt), 0, 0.0
    ready, debt = 0.0, catchup
    while i < n:
        best = (1.0 / (costs[1] * slow), 0, 0.0)  # (rate, L, wait)
        whole, exp = 1.0, 0.0
        for L in range(1, min(MAX_L, n - i) + 1):
            whole *= p1[i + L - 1]
            exp += whole
            wait = max(0.0, (debt + max(L - ready, 0.0)) * delta)
            rate = (1 + exp) / (wait + costs[L + 1] * slow)
            if rate > best[0]:
                best = (rate, L, wait)
        _, L, wait = best
        dur = wait + costs[L + 1] * slow
        work = dur / delta
        used = min(work, debt)
        debt -= used
        ready += work - used
        a = accepted(top, tgt, i, L)
        t += dur
        if a == L and i + L < n and top[i + L][0] == tgt[i + L]:
            ready = max(ready - (L + 1), 0.0)
        else:
            ready, debt = 0.0, catchup
        i += a + 1
    return n / t


def shadow_switch(seq, costs, delta_smt, delta_fast=0.13, slow=1.03, catchup=1.5):
    """Shadow drafting with thread switching: while the 7B waits for drafts, the 0.5B runs on all
    cores (delta_fast, the shipped speed); while the 7B verifies, it runs on the SMT siblings
    (delta_smt). Waiting is chosen only when it pays (as shadow_wait)."""
    tgt, top, p = seq["target"], seq["top"], seq["p"]
    p1 = [x[0] for x in p]
    n, i, t = len(tgt), 0, 0.0
    ready, debt = 0.0, catchup
    while i < n:
        best = (1.0 / (costs[1] * slow), 0, 0.0)
        whole, exp = 1.0, 0.0
        for L in range(1, min(MAX_L, n - i) + 1):
            whole *= p1[i + L - 1]
            exp += whole
            wait = max(0.0, (debt + max(L - ready, 0.0)) * delta_fast)
            rate = (1 + exp) / (wait + costs[L + 1] * slow)
            if rate > best[0]:
                best = (rate, L, wait)
        _, L, wait = best
        # waiting: fast drafting until L tokens exist
        work = wait / delta_fast
        used = min(work, debt)
        debt -= used
        ready += work - used
        verify = costs[L + 1] * slow
        work = verify / delta_smt
        used = min(work, debt)
        debt -= used
        ready += work - used
        a = accepted(top, tgt, i, L)
        t += wait + verify
        if a == L and i + L < n and top[i + L][0] == tgt[i + L]:
            ready = max(ready - (L + 1), 0.0)
        else:
            ready, debt = 0.0, catchup
        i += a + 1
    return n / t


def main() -> None:
    seqs = json.loads(Path(sys.argv[1]).read_text())["seqs"]
    tables = json.loads((Path.home() / ".cascade" / "verify_costs.json").read_text())
    costs = [float(x) for x in tables[[k for k in tables if k.startswith("v2")][0]][:MAX_L + 2]]
    for group in ("code", "open"):
        sub = [s for s in seqs if s["group"] == group]
        tot = sum(len(s["target"]) for s in sub)

        def agg(f, *a):
            return tot / sum(len(s["target"]) / f(s, *a) for s in sub)
        seq = agg(sequential, costs, 0.13)
        print(f"{group}: sequential (shipped, draft 0.13/token) {seq:.3f}x")
        for delta in (0.2, 0.25, 0.3):
            sh = agg(shadow, costs, delta)
            sw = agg(shadow_wait, costs, delta)
            ss = agg(shadow_switch, costs, delta)
            print(f"   draft {delta:.2f}/token on SMT threads: shadow {sh:.3f}x ({(sh / seq - 1) * 100:+.1f}%), "
                  f"shadow+wait {sw:.3f}x ({(sw / seq - 1) * 100:+.1f}%), "
                  f"shadow+switch {ss:.3f}x ({(ss / seq - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
