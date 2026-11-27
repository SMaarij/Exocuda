"""Usage: python scripts/analysis/cache_judge.py [--judges qwen2.5:0.5b-instruct,...] [--out OUT.json]

Can a small model's yes/no judgement let the semantic cache serve more rewordings without
serving near-misses? For every pair in the cache fixtures (cache_paraphrase_pairs.jsonl,
cache_near_miss_pairs.jsonl) and the held-out set (cache_judge_holdout.jsonl) it records the
embedding similarity, the term-swap guard's verdict (against a stored answer) and each judge's
P(yes), then scores serving policies: tuned on the fixtures, tested on the held-out pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llama_cpp import Llama  # noqa: E402

from cascade_local.cli.main import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402
from cascade_local.engine.intelligence.cache import term_swap_conflict  # noqa: E402
from cascade_local.engine.intelligence.embeddings import FastEmbedEmbedder  # noqa: E402

JUDGE_PROMPT = ("Do these two questions ask for exactly the same information, so that one correct answer "
                "answers both? Reply yes or no.\n\nQuestion 1: {a}\nQuestion 2: {b}")
BENCH = ROOT / "docs" / "benchmark"


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (BENCH / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def pairs() -> list[dict]:
    rows = [{"a": r["original"], "b": r["paraphrase"], "same": True, "answer": r.get("answer"), "set": "fixture"}
            for r in load("cache_paraphrase_pairs.jsonl")]
    rows += [{"a": r["original"], "b": r["different"], "same": False, "answer": r.get("answer"), "set": "fixture"}
             for r in load("cache_near_miss_pairs.jsonl")]
    rows += [dict(r, answer=None, set="holdout") for r in load("cache_judge_holdout.jsonl")]
    return rows


def judge_scores(tag: str, rows: list[dict], answer: bool = False) -> list[float]:
    runner = LlamaCppRunner(draft_model_tag=None, temperature=0.0)
    path = runner._resolve_path(tag)
    llm = Llama(path, n_ctx=1024, verbose=False, use_mmap=False, flash_attn=True, logits_all=True)
    runner._llm = llm
    yes = [llm.tokenize(w.encode(), add_bos=False)[0] for w in ("yes", "Yes")]
    no = [llm.tokenize(w.encode(), add_bos=False)[0] for w in ("no", "No")]
    out = []
    for r in rows:
        ids = runner._chat_tokens([{"role": "user", "content": JUDGE_PROMPT.format(a=r["a"], b=r["b"])}])
        llm.reset()
        llm.eval(ids)
        logits = np.asarray(llm.scores[llm.n_tokens - 1])
        out.append(float(1 / (1 + np.exp(np.logaddexp.reduce(logits[no]) - np.logaddexp.reduce(logits[yes])))))
    return out


def stored_answers(rows: list[dict]) -> None:
    """Answers for rows without a recorded one, from the 0.5B (what serves easy questions)."""
    missing = [r for r in rows if not r["answer"]]
    if not missing:
        return
    runner = LlamaCppRunner(draft_model_tag=None, temperature=0.0)
    runner.load("qwen2.5", "0.5b-instruct")
    cache: dict[str, str] = {}
    for r in missing:
        if r["a"] not in cache:
            cache[r["a"]] = runner.generate([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                             {"role": "user", "content": r["a"]}], 64).output_text
        r["answer"] = cache[r["a"]]
    runner.unload()


def score(rows: list[dict], serve) -> dict:
    hits = sum(1 for r in rows if r["same"] and serve(r))
    wrong = sum(1 for r in rows if not r["same"] and serve(r))
    same = sum(r["same"] for r in rows)
    return {"rewordings_served": f"{hits}/{same}", "near_misses_served": f"{wrong}/{len(rows) - same}",
            "hit_rate": round(hits / same, 3), "wrong": wrong}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", default="qwen2.5:0.5b-instruct,qwen2.5:1.5b-instruct-q4_K_M")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = pairs()
    stored_answers(rows)
    emb = FastEmbedEmbedder()
    for r in rows:
        va, vb = np.asarray(emb.embed(r["a"])), np.asarray(emb.embed(r["b"]))
        r["sim"] = float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))
        r["guard"] = term_swap_conflict(r["b"], r["a"], r["answer"], 4, "strict")
    judges = args.judges.split(",")
    for tag in judges:
        for r, p in zip(rows, judge_scores(tag, rows)):
            r[tag] = p

    report: dict = {"policies": {}}
    for subset in ("fixture", "holdout"):
        sub = [r for r in rows if r["set"] == subset]
        report["policies"][f"{subset}: shipped (sim>=0.92, guard)"] = score(sub, lambda r: r["sim"] >= 0.92 and not r["guard"])
        for tag in judges:
            # the strictest judge threshold that serves no fixture near-miss above the similarity floor
            for floor in (0.80, 0.85):
                fix = [r for r in rows if r["set"] == "fixture" and not r["same"] and r["sim"] >= floor and not r["guard"]]
                tau = max([r[tag] for r in fix] + [0.5]) + 1e-6
                name = f"{subset}: sim>={floor}, guard, {tag.split(':')[0]}{tag.split(':')[1][:4]} P(yes)>{tau:.3f}"
                report["policies"][name] = score(
                    sub, lambda r, f=floor, t=tau, j=tag: (r["sim"] >= 0.92 and not r["guard"])
                    or (r["sim"] >= f and not r["guard"] and r[j] > t))
    for name, s in report["policies"].items():
        print(f"{name:70s} rewordings {s['rewordings_served']:>6}  near-misses {s['near_misses_served']:>6}")
    report["rows"] = rows
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
