"""
scripts/benchmark_system.py

Whole-system benchmark: one realistic session of mixed questions run through
the real Pipeline (real embedder, cache, router, local models), against plain
local inference with the same 7B model and system prompt.

The session: easy factual questions, hard code/reasoning questions, exact
repeats, rewordings of earlier questions, and near-misses that look like an
earlier question but need a different answer (a cache that serves them is
wrong). Every answer is checked against expected-answer patterns, so speed
gains that cost correctness show up.

Each configuration runs in its own process, so RAM figures are not mixed:

    python scripts/benchmark_system.py --out docs/benchmark/system_session.json

Configurations:
    plain      -- 7B, plain decoding; no cache, no router (what running the
                  model directly gives you)
    cascade    -- 7B with confidence drafting, cache, compressor, router
    small15    -- cascade + qwen2.5 1.5B for queries the router calls easy
    small05    -- cascade + qwen2.5 0.5B for queries the router calls easy
    shared05   -- small05, but the 0.5B model is also the 7B's draft model:
                  one instance in RAM for both jobs (cli default when
                  small_model is the same family as model)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BIG_MODEL = "qwen2.5-coder"
DRAFT_MODEL = "qwen2.5-coder:0.5b"
SMALL_MODELS = {"small15": "qwen2.5:1.5b-instruct", "small05": "qwen2.5:0.5b-instruct",
                "shared05": "qwen2.5:0.5b-instruct"}
SHARED_DRAFT = {"shared05": "qwen2.5:0.5b-instruct-q4_K_M"}  # the small model's own file
CONFIGS = ("plain", "cascade", "small15", "small05", "shared05")

# (category, question, expected-answer regexes: all must match, case-insensitive)
SESSION: list[tuple[str, str, list[str]]] = [
    ("easy", "What is the capital of Australia?", [r"canberra"]),
    ("easy", "How many days are in a leap year?", [r"366"]),
    ("easy", "What is the chemical symbol for gold?", [r"\bAu\b"]),
    ("hard", "Write a Python function that checks whether a string is a palindrome, ignoring case and non-letter characters.", [r"def ", r"lower|casefold"]),
    ("easy", "Who wrote Romeo and Juliet?", [r"shakespeare"]),
    ("easy", "Convert 5 kilometers to meters.", [r"5,?000"]),
    ("hard", "Write a SQL query that returns the top 3 customers by total order amount, from tables customers(id, name) and orders(customer_id, amount).", [r"join", r"group by", r"limit 3|top 3|top\s*\(3\)"]),
    ("easy", "What is the boiling point of water in Celsius at sea level?", [r"100"]),
    ("easy", "Translate \"thank you\" into Spanish.", [r"gracias"]),
    ("hard", "A train leaves at 9:40 and arrives at 13:15. How long is the journey?", [r"3 hours? (and )?35|3h ?35|215 minutes|3:35"]),
    ("easy", "What is 12 multiplied by 8?", [r"\b96\b"]),
    ("easy", "What does CPU stand for?", [r"central processing unit"]),
    ("hard", "Write a JavaScript function that debounces another function.", [r"setTimeout", r"clearTimeout"]),
    ("repeat", "What is the capital of Australia?", [r"canberra"]),
    ("easy", "Name the largest planet in our solar system.", [r"jupiter"]),
    ("hard", "In Python, def avg(xs): return sum(xs) / len(xs) -- what happens with an empty list, and how do you fix it?", [r"ZeroDivisionError|division by zero|divide by zero"]),
    ("reworded", "Which city is the capital of Australia?", [r"canberra"]),
    ("easy", "Which ocean is the largest?", [r"pacific"]),
    ("near_miss", "What is the capital of Austria?", [r"vienna"]),
    ("hard", "If a shirt costs $40 after a 20% discount, what was the original price?", [r"\$?50\b"]),
    ("reworded", "Who is the author of Romeo and Juliet?", [r"shakespeare"]),
    ("repeat", "Write a Python function that checks whether a string is a palindrome, ignoring case and non-letter characters.", [r"def ", r"lower|casefold"]),
    ("near_miss", "What is the chemical symbol for silver?", [r"\bAg\b"]),
    ("easy", "What is the opposite of \"ancient\"?", [r"modern|new|recent|contemporary"]),
    ("hard", "Write a Python function that merges two sorted lists into one sorted list in linear time.", [r"def ", r"while|for "]),
    ("reworded", "What's the biggest planet in the solar system?", [r"jupiter"]),
    ("near_miss", "What is 12 multiplied by 9?", [r"\b108\b"]),
    ("repeat", "Who wrote Romeo and Juliet?", [r"shakespeare"]),
    ("reworded", "How many days does a leap year have?", [r"366"]),
    ("near_miss", "How many days are in a regular, non-leap year?", [r"365"]),
    ("hard", "Explain the difference between a process and a thread, with one example of when to use each.", [r"process", r"thread", r"memory"]),
    ("reworded", "What does the abbreviation CPU mean?", [r"central processing unit"]),
    ("repeat", "A train leaves at 9:40 and arrives at 13:15. How long is the journey?", [r"3 hours? (and )?35|3h ?35|215 minutes|3:35"]),
    ("reworded", "In Python, write a function to check if a string is a palindrome, ignoring case and non-letter characters.", [r"def ", r"lower|casefold"]),
]


class _FixedProfiler:
    """The measured profile of this laptop for the 7B (4 tok/s): selects the q4_K_M rung.
    Skips a profiling run, which is a one-off cost outside the session being measured."""

    def load_cached_profile(self):
        return {"measured_tok_s": {"7b-instruct-q4_K_M": 4.0}, "vram_gb": 0.0}

    def profile(self):
        return self.load_cached_profile()


def _local_backend(model_id: str, accelerated: bool, max_tokens: int, draft: str = DRAFT_MODEL):
    from engine.execution.acceleration_engine import AccelerationLayer, DraftPair
    from engine.execution.backend import LocalExecutionBackend
    from engine.execution.llamacpp_runner import LlamaCppRunner
    from engine.execution.model_ladder import DEFAULT_LADDER, ModelLadder

    floor = ModelLadder(rungs=(min(DEFAULT_LADDER, key=lambda r: r.min_tok_s),))
    if accelerated:
        runner = LlamaCppRunner(draft_model_tag=draft, temperature=0.0, draft_p_min=0.75, draft_tokens=16)
        return LocalExecutionBackend(runner=runner, model_id=model_id, profiler=_FixedProfiler(), ladder=floor,
                                     preferred_layer=AccelerationLayer.DRAFT_MODEL,
                                     validated_pairs=[DraftPair(model_id, draft, True)], max_tokens=max_tokens)
    runner = LlamaCppRunner(draft_model_tag=None, temperature=0.0)
    return LocalExecutionBackend(runner=runner, model_id=model_id, profiler=_FixedProfiler(), ladder=floor,
                                 max_tokens=max_tokens)


def build(config: str, max_tokens: int, cache_dir: Path):
    from cli.main import DEFAULT_SYSTEM_PROMPT
    from engine.intelligence.pipeline import Pipeline

    if config == "plain":
        big = _local_backend(BIG_MODEL, False, max_tokens)
        return Pipeline(big, confidence_floor=None, system_prompt=DEFAULT_SYSTEM_PROMPT), [big]

    from engine.intelligence.cache import SemanticCache
    from engine.intelligence.context_compressor import ContextCompressor
    from engine.intelligence.embeddings import FastEmbedEmbedder
    from engine.intelligence.router.classifier import SemanticRouterClassifier

    embedder = FastEmbedEmbedder()
    small = _local_backend(SMALL_MODELS[config], False, max_tokens) if config in SMALL_MODELS else None
    big = _local_backend(BIG_MODEL, True, max_tokens, SHARED_DRAFT.get(config, DRAFT_MODEL))
    embedder.embed("warm-up")  # load the embedding model before the session too
    return Pipeline(
        big,
        small_local_backend=small,
        compressor=ContextCompressor(embedder),
        cache=SemanticCache(embedder, db_path=cache_dir / "cache.db"),
        router=SemanticRouterClassifier(embedder),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ), [b for b in (big, small) if b is not None]


def run_config(config: str, max_tokens: int) -> dict:
    import psutil

    proc = psutil.Process()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # sqlite keeps cache.db open on Windows
        t0 = time.perf_counter()
        pipeline, backends = build(config, max_tokens, Path(tmp))
        for backend in backends:  # load every model before the session, as a running chat or server has
            backend.generate([{"role": "user", "content": "Say OK."}])
        load_s = time.perf_counter() - t0
        rss_after_load = round(proc.memory_info().rss / 1e9, 2)
        peak_after_load = round(getattr(proc.memory_info(), "peak_wset", 0) / 1e9, 2)
        rows = []
        for i, (category, question, patterns) in enumerate(SESSION):
            start = time.perf_counter()
            result = pipeline.run([{"role": "user", "content": question}])
            seconds = time.perf_counter() - start
            text = result["text"]
            served = "cache" if result["cache_hit"] else (
                "small" if getattr(pipeline, "last_served_by_small", False) else "big")
            correct = all(re.search(p, text, re.IGNORECASE) for p in patterns)
            rows.append({"i": i, "category": category, "question": question, "seconds": round(seconds, 2),
                         "served_by": served, "escalated": result["escalated"], "correct": correct,
                         "rss_gb": round(proc.memory_info().rss / 1e9, 2), "answer": text[:400]})
            print(f"  [{config}] {i:2d} {category:9s} {served:5s} {seconds:6.1f}s {'ok ' if correct else 'BAD'} {question[:60]}",
                  flush=True)
        total = time.perf_counter() - t0
    mem = proc.memory_info()
    return {"config": config, "rows": rows, "total_s_including_load": round(total, 1), "load_s": round(load_s, 1),
            "rss_after_load_gb": rss_after_load, "peak_after_load_gb": peak_after_load,
            "peak_rss_gb": round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)}


def summarize(result: dict) -> dict:
    rows = result["rows"]
    secs = [r["seconds"] for r in rows]
    by_cat: dict[str, dict] = {}
    for cat in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {"n": len(sub), "mean_s": round(statistics.mean(r["seconds"] for r in sub), 2),
                       "correct": sum(r["correct"] for r in sub)}
    served = {k: sum(r["served_by"] == k for r in rows) for k in ("cache", "small", "big")}
    return {"session_s": round(sum(secs), 1), "mean_s": round(statistics.mean(secs), 2),
            "median_s": round(statistics.median(secs), 2), "correct": sum(r["correct"] for r in rows),
            "questions": len(rows), "served_by": served, "by_category": by_cat,
            "peak_rss_gb": result["peak_rss_gb"], "total_s_including_load": result["total_s_including_load"],
            "load_s": result["load_s"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark/system_session.json"))
    parser.add_argument("--one", help=argparse.SUPPRESS)  # child process: run one config, print JSON
    args = parser.parse_args()

    if args.one:
        print("RESULT " + json.dumps(run_config(args.one, args.max_tokens)), flush=True)
        return

    results = {}
    for config in args.configs.split(","):
        print(f"== {config}", flush=True)
        child = subprocess.run([sys.executable, __file__, "--one", config, "--max-tokens", str(args.max_tokens)],
                               capture_output=True, text=True, encoding="utf-8")
        sys.stdout.write("".join(l + "\n" for l in child.stdout.splitlines() if not l.startswith("RESULT ")))
        line = next((l for l in child.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(child.stderr[-3000:])
            continue
        result = json.loads(line[len("RESULT "):])
        results[config] = {"summary": summarize(result), "rows": result["rows"]}

    base = results.get("plain", {}).get("summary")
    print(f"\n{'config':9s} {'session':>9s} {'mean':>7s} {'median':>7s} {'speed-up':>9s} {'correct':>8s} {'peak RAM':>9s}  served cache/small/big")
    for config, r in results.items():
        s = r["summary"]
        speedup = f"{base['session_s'] / s['session_s']:.2f}x" if base else "-"
        sv = s["served_by"]
        print(f"{config:9s} {s['session_s']:8.0f}s {s['mean_s']:6.1f}s {s['median_s']:6.1f}s {speedup:>9s} "
              f"{s['correct']:>3d}/{s['questions']:<3d} {s['peak_rss_gb']:7.2f}GB  {sv['cache']}/{sv['small']}/{sv['big']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"max_tokens": args.max_tokens, "big_model": BIG_MODEL, "draft_model": DRAFT_MODEL,
                                    "small_models": SMALL_MODELS, "shared_draft": SHARED_DRAFT, "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
