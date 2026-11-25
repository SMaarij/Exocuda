"""Usage: python scripts/analysis/cache_judge_live.py [--out OUT.json]

The semantic cache with the shipped judge, end to end: the real SemanticCache (default
CacheConfig), the 7B in the llama.cpp runner (draft mode, as served) answering
yes_probability(). For each pair in the fixtures and the held-out set, the original is stored
with its answer and the other question looked up; reports rewordings served, near-misses served
(wrong answers), judge calls and their latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402
from engine.intelligence.cache import SemanticCache  # noqa: E402
from engine.intelligence.embeddings import FastEmbedEmbedder  # noqa: E402
from scripts.analysis.cache_judge import pairs, stored_answers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = pairs()
    stored_answers(rows)

    runner = LlamaCppRunner(temperature=0.0, draft_model_tag="qwen2.5:0.5b-instruct", draft_tokens=16,
                            draft_verify_costs="auto")
    runner.set_acceleration("draft_model")
    runner.load("qwen2.5-coder", "7b-instruct-q4_K_M")
    runner.generate("Say OK.", 8)
    times: list[float] = []

    def judge(prompt: str) -> float:
        t = time.perf_counter()
        p = runner.yes_probability([{"role": "user", "content": prompt}])
        times.append(time.perf_counter() - t)
        return p

    emb = FastEmbedEmbedder()
    tmp = Path(tempfile.mkdtemp(prefix="cascade-judge-"))
    for i, r in enumerate(rows):
        cache = SemanticCache(emb, db_path=tmp / f"c{i}.sqlite3", judge=judge)
        cache.write([{"role": "user", "content": r["a"]}], r["answer"])
        t = time.perf_counter()
        r["served"] = cache.lookup([{"role": "user", "content": r["b"]}]) is not None
        r["lookup_s"] = round(time.perf_counter() - t, 3)
        r["judged"] = cache.judged
        cache.close()
    summary = {}
    for subset in ("fixture", "holdout"):
        sub = [r for r in rows if r["set"] == subset]
        same = [r for r in sub if r["same"]]
        diff = [r for r in sub if not r["same"]]
        summary[subset] = {"rewordings_served": f"{sum(r['served'] for r in same)}/{len(same)}",
                           "near_misses_served": f"{sum(r['served'] for r in diff)}/{len(diff)}",
                           "judge_calls": sum(r["judged"] for r in sub)}
        print(subset, summary[subset])
    summary["judge_latency_s"] = {"median": round(statistics.median(times), 2), "max": round(max(times), 2),
                                  "calls": len(times)}
    print("judge latency", summary["judge_latency_s"])
    for r in rows:
        if r["served"] != r["same"]:
            print("  WRONG" if r["served"] else "  missed", r["a"], "|", r["b"])
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
