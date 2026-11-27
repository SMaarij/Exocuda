"""Usage: python scripts/analysis/quant_for_speculation.py DATA.json NAME=PATH [NAME=PATH ...] [--save OUT]

Which weight format makes the 7B fastest *under speculative decoding* on a CPU? Plain decoding is
bound by bytes per token; a speculative step verifies 2-17 tokens in one batch, which costs more
arithmetic per byte, so the format that decodes fastest need not verify fastest. For each 7B file:

- single-token step time and the verify-cost table (1..17 tokens, relative to one step), measured
  exactly as the runner does (measure_verify_costs, runner load settings);
- greedy agreement with the reference target: how often its argmax equals the reference 7B's next
  token on the reference's own outputs (DATA.json from draft_top2_coverage.py), by prompt group;
- the expected speed of the shipped cost-aware 0.5B drafting on it: the exact replay
  (shadow_draft_sim.sequential) with this file's cost table and the 0.5B's cost in its steps,
  converted to tokens/s.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from llama_cpp import Llama  # noqa: E402

from cascade_local.engine.execution.llamacpp_runner import (  # noqa: E402
    WORKING_MEMORY_SETTINGS, LlamaCppRunner, measure_verify_costs)
from shadow_draft_sim import sequential  # noqa: E402

DRAFT = "qwen2.5:0.5b-instruct"


def load(path: str, n_ctx: int, logits_all: bool) -> Llama:
    return Llama(path, n_ctx=n_ctx, logits_all=logits_all, verbose=False, use_mmap=False,
                 **WORKING_MEMORY_SETTINGS)


def step_ms(llm: Llama, seqs: list[dict]) -> float:
    out = []
    for s in seqs[:4]:
        llm.reset()
        llm.eval(list(s["prompt"]))
        for tok in s["target"][:16]:
            t = time.perf_counter()
            llm.eval([tok])
            out.append(time.perf_counter() - t)
    llm.reset()
    return 1000 * statistics.median(out)


def agreement(llm: Llama, seqs: list[dict]) -> dict[str, float]:
    hits: dict[str, list[int]] = {}
    for s in seqs:
        llm.reset()
        ids = list(s["prompt"]) + list(s["target"])
        llm.eval(ids)
        logits = np.asarray(llm.scores[: len(ids)])
        pred = logits[len(s["prompt"]) - 1: len(ids) - 1].argmax(axis=1)
        h = hits.setdefault(s["group"], [0, 0])
        h[0] += int((pred == np.asarray(s["target"])).sum())
        h[1] += len(s["target"])
    llm.reset()
    return {g: round(a / b, 4) for g, (a, b) in hits.items()}


def replay(seqs: list[dict], costs: list[float], delta: float, step: float) -> dict[str, float]:
    out = {}
    for group in ("code", "open"):
        sub = [s for s in seqs if s["group"] == group]
        tot = sum(len(s["target"]) for s in sub)
        speedup = tot / sum(len(s["target"]) / sequential(s, costs, delta) for s in sub)
        out[group] = round(1000 * speedup / step, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--save")
    a = ap.parse_args()
    seqs = json.loads(Path(a.data).read_text())["seqs"]
    n_ctx = max(len(s["prompt"]) + len(s["target"]) for s in seqs) + 32
    draft = load(LlamaCppRunner(draft_model_tag=None)._resolve_path(DRAFT), n_ctx, False)
    draft_ms = step_ms(draft, seqs)
    del draft
    print(f"0.5B draft step {draft_ms:.1f} ms", flush=True)
    results = {"draft_ms": round(draft_ms, 1)}
    for spec in a.models:
        name, path = spec.split("=", 1)
        if not path.endswith(".gguf") and ":" in path:
            path = LlamaCppRunner(draft_model_tag=None)._resolve_path(path)
        llm = load(path, n_ctx, True)
        step = step_ms(llm, seqs)
        costs = measure_verify_costs(llm, 17)
        agr = agreement(llm, seqs)
        del llm
        speed = replay(seqs, costs, draft_ms / step, step)
        size = Path(path).stat().st_size / 2**30
        results[name] = {"size_gb": round(size, 2), "step_ms": round(step, 1),
                         "costs": [round(c, 3) for c in costs[1:]], "agreement": agr,
                         "cascade_tok_s": speed, "plain_tok_s": round(1000 / step, 2)}
        print(f"{name:8s} {size:5.2f} GB  step {step:6.1f} ms  plain {1000 / step:5.2f} tok/s  "
              f"verify4 {costs[4]:.2f} verify8 {costs[8]:.2f}  agreement {agr}  "
              f"with drafting {speed} tok/s", flush=True)
    if a.save:
        Path(a.save).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
