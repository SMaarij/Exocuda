"""
scripts/run_benchmark.py

Section 12 (Master Requirements Doc) — the benchmark suite nobody had
built yet (flagged as TBD in every phase summary: Track A's
docs/hardware_compat_results.md, Track B's and Track C's phase
summaries). This is a first real version, run against a genuinely
held-out reference set, not the training data — and it's honest about
what it can and can't measure without live model infrastructure.

WHAT THIS MEASURES FOR REAL, on this machine, right now:
    - SemanticRouter agreement: the router is trained on
      engine/intelligence/router/coldstart_data.py's BOOTSTRAP_EXAMPLES
      and evaluated against docs/benchmark/reference_queries.jsonl --
      a DIFFERENT, hand-labeled set with no overlap, so this is a real
      held-out accuracy number, not training-set accuracy dressed up.
    - SemanticCache hit rate on paraphrased vs. exact-duplicate queries
      (docs/benchmark/cache_paraphrase_pairs.jsonl), using the real
      FastEmbedEmbedder (bge-small ONNX), not the FakeEmbedder unit
      tests use.
    - A derived cost-reduction estimate from the two measurements above.

WHAT THIS DOES NOT MEASURE, and why (flagged, not faked):
    - Speculative decoding speedup (NFR-01) on real quantized models --
      needs a live Ollama instance with draft/target pairs loaded.
      Track A's scripts/benchmark_acceleration.py is the tool for that,
      separately, and needs to be run explicitly (it touches real
      models and takes real time).
    - Hardware ladder success rate across >=5 real machines (R-06) --
      inherently needs multiple different physical machines; see
      docs/hardware_compat_results.md, still a template.
    - Output quality regression vs. an FP16 baseline (NFR-11) -- needs
      a real quantized model to compare against, not measurable from
      routing/caching logic alone.
    - self-hosted GPU throughput / time-to-first-token reduction --
      needs an actual deployed vLLM/SGLang instance behind
      gateway/backend.py, not a fake CloudInferenceProvider.

Every number this script prints is either measured on this run or
explicitly marked NOT MEASURED -- never estimated and reported as if
it were measured.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from engine.intelligence.cache import CacheConfig, SemanticCache
from engine.intelligence.embeddings import FastEmbedEmbedder
from engine.intelligence.router.classifier import SemanticRouterClassifier

REFERENCE_QUERIES_PATH = Path(__file__).parent.parent / "docs" / "benchmark" / "reference_queries.jsonl"
CACHE_PARAPHRASE_PAIRS_PATH = Path(__file__).parent.parent / "docs" / "benchmark" / "cache_paraphrase_pairs.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "docs" / "BENCHMARK_REPORT.md"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def measure_router_agreement(embedder: FastEmbedEmbedder) -> dict:
    """
    Trains on BOOTSTRAP_EXAMPLES (cold-start data), evaluates against
    the genuinely separate, hand-labeled reference_queries.jsonl.
    """
    reference = _load_jsonl(REFERENCE_QUERIES_PATH)
    router = SemanticRouterClassifier(embedder)
    router.fit()  # trains on BOOTSTRAP_EXAMPLES by default

    correct = 0
    confusion: dict[str, dict[str, int]] = {}
    per_query_results = []
    for row in reference:
        decision = router.classify(row["query"])
        expected = row["expected_tier"]
        is_correct = decision.tier == expected
        correct += int(is_correct)
        confusion.setdefault(expected, {}).setdefault(decision.tier, 0)
        confusion[expected][decision.tier] += 1
        per_query_results.append(
            {"query": row["query"], "expected": expected, "predicted": decision.tier,
             "confidence": decision.confidence, "correct": is_correct}
        )

    return {
        "total": len(reference),
        "correct": correct,
        "agreement_rate": correct / len(reference) if reference else 0.0,
        "confusion": confusion,
        "per_query_results": per_query_results,
    }


def measure_cache_hit_rate(embedder: FastEmbedEmbedder, tmp_db_path: Path) -> dict:
    """
    NFR-06's own target: paraphrased-query hit rate should be >=90% of
    exact-duplicate hit rate. Measures both, for real, with the real
    embedder -- not FakeEmbedder.
    """
    pairs = _load_jsonl(CACHE_PARAPHRASE_PAIRS_PATH)
    cache = SemanticCache(embedder, db_path=tmp_db_path, config=CacheConfig(similarity_threshold=0.85))

    for row in pairs:
        cache.write([{"role": "user", "content": row["original"]}], f"answer to: {row['original']}")

    exact_hits = 0
    paraphrase_hits = 0
    for row in pairs:
        if cache.lookup([{"role": "user", "content": row["original"]}]) is not None:
            exact_hits += 1
        if cache.lookup([{"role": "user", "content": row["paraphrase"]}]) is not None:
            paraphrase_hits += 1

    exact_rate = exact_hits / len(pairs) if pairs else 0.0
    paraphrase_rate = paraphrase_hits / len(pairs) if pairs else 0.0
    return {
        "total_pairs": len(pairs),
        "exact_duplicate_hit_rate": exact_rate,
        "paraphrase_hit_rate": paraphrase_rate,
        "paraphrase_vs_exact_ratio": (paraphrase_rate / exact_rate) if exact_rate > 0 else None,
        "meets_90pct_target": (paraphrase_rate >= 0.9 * exact_rate) if exact_rate > 0 else False,
    }


def estimate_cost_reduction(router_result: dict) -> dict:
    """
    A derived estimate from the router measurement ONLY, explicitly
    labeled as such -- not a live cost measurement (no real cloud API
    calls were made). Local-tier queries cost $0 in cloud spend;
    cloud_byok/self_hosted tier queries would have cost something on
    a pure-cloud baseline regardless of Cascade's involvement.

    Deliberately does NOT fold the cache-hit-rate measurement into this
    number: measure_cache_hit_rate() only re-reads the exact same
    queries it just wrote, so its 100% exact-duplicate rate is a
    correctness check on SemanticCache's lookup logic, not a measurement
    of how often *real, unseen* production traffic repeats. Compounding
    it in here would inflate this estimate with a number that doesn't
    mean what it would look like it means -- see docs/BENCHMARK_REPORT.md's
    own caveat on this. A real cache contribution to cost reduction needs
    actual repeated production traffic, which this benchmark can't
    generate from a 15-pair fixture.
    """
    total = router_result["total"]
    local_count = sum(1 for r in router_result["per_query_results"] if r["predicted"] == "local")
    free_fraction = local_count / total if total else 0.0
    return {
        "local_resolved_fraction": free_fraction,
        "projected_cost_reduction_vs_pure_cloud": free_fraction,
        "note": (
            "Derived from the router measurement alone -- local-tier queries cost $0. "
            "Does NOT include a cache contribution: this benchmark's cache test only "
            "confirms correctness (re-reading the same queries it wrote), not a real "
            "hit rate on unseen production traffic, so folding it in would overstate "
            "this number. A real combined figure needs live traffic, not a fixture."
        ),
    }


def main() -> None:
    print("Loading real FastEmbedEmbedder (bge-small ONNX)...")
    embedder = FastEmbedEmbedder()

    print("\n=== SemanticRouter agreement (held-out reference set) ===")
    router_result = measure_router_agreement(embedder)
    print(f"Total queries:     {router_result['total']}")
    print(f"Correct:           {router_result['correct']}")
    print(f"Agreement rate:    {router_result['agreement_rate']:.1%}  (target: >=85%, NFR/Section 12)")
    print(f"Confusion matrix:  {json.dumps(router_result['confusion'], indent=2)}")

    print("\n=== SemanticCache hit rate (paraphrase vs exact-duplicate) ===")
    tmp_db = Path(__file__).parent.parent / ".benchmark_cache.sqlite3"
    tmp_db.unlink(missing_ok=True)
    cache_result = measure_cache_hit_rate(embedder, tmp_db)
    tmp_db.unlink(missing_ok=True)
    print(f"Exact-duplicate hit rate:  {cache_result['exact_duplicate_hit_rate']:.1%}")
    print(f"Paraphrase hit rate:       {cache_result['paraphrase_hit_rate']:.1%}")
    print(f"Meets >=90%-of-exact target: {cache_result['meets_90pct_target']}")

    print("\n=== Derived cost-reduction estimate (router only -- see note) ===")
    cost_result = estimate_cost_reduction(router_result)
    print(f"Local-resolved fraction:              {cost_result['local_resolved_fraction']:.1%}")
    print(f"Projected cost reduction vs pure cloud: {cost_result['projected_cost_reduction_vs_pure_cloud']:.1%}")
    print(f"Note: {cost_result['note']}")

    _write_report(router_result, cache_result, cost_result)
    print(f"\nFull report written to {REPORT_PATH}")


def _write_report(router_result: dict, cache_result: dict, cost_result: dict) -> None:
    ratio = cache_result["paraphrase_vs_exact_ratio"]
    ratio_str = f"{ratio:.1%}" if ratio is not None else "n/a"
    lines = [
        "# Cascade — Benchmark Report",
        "",
        f"**Run date:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "**Run against:** `docs/benchmark/reference_queries.jsonl` (37 hand-labeled, held-out queries — "
        "NOT the real TriviaQA/GSM8K/HumanEval/MT-Bench 200-query set the master doc's Section 12 "
        "specifies, which needs a programmatic dataset download not done here; flagged as a placeholder "
        "exactly like `router/coldstart_data.py`'s BOOTSTRAP_EXAMPLES).",
        "",
        "## What was actually measured",
        "",
        "### SemanticRouter agreement (held-out, not training-set accuracy)",
        "",
        f"- Total queries: {router_result['total']}",
        f"- Correct: {router_result['correct']}",
        f"- **Agreement rate: {router_result['agreement_rate']:.1%}** (target: ≥85%)",
        f"- Meets target: {'YES' if router_result['agreement_rate'] >= 0.85 else 'NO'}",
        "",
        "Confusion matrix (expected tier → predicted tier counts):",
        "```json",
        json.dumps(router_result["confusion"], indent=2),
        "```",
        "",
        "### SemanticCache hit rate",
        "",
        f"- Exact-duplicate hit rate: {cache_result['exact_duplicate_hit_rate']:.1%}",
        f"- Paraphrase hit rate: {cache_result['paraphrase_hit_rate']:.1%}",
        f"- Paraphrase-vs-exact ratio: {ratio_str}",
        f"- Meets ≥90%-of-exact target: {'YES' if cache_result['meets_90pct_target'] else 'NO'}",
        "",
        "### Derived cost-reduction estimate (router contribution only — not a live cost measurement)",
        "",
        f"- Local-resolved fraction: {cost_result['local_resolved_fraction']:.1%}",
        f"- Projected cost reduction vs. pure-cloud baseline (router alone): {cost_result['projected_cost_reduction_vs_pure_cloud']:.1%}",
        f"- **Note:** {cost_result['note']}",
        "",
        "## What was NOT measured, and why",
        "",
        "| Metric | Why not measured here |",
        "|---|---|",
        "| Speculative decoding speedup (NFR-01) | Needs a live Ollama instance with draft/target pairs loaded — run `scripts/benchmark_acceleration.py` separately against real models. |",
        "| Hardware ladder success rate across ≥5 machines (R-06) | Inherently needs multiple different physical machines — see `docs/hardware_compat_results.md`. |",
        "| Output quality regression vs. FP16 (NFR-11) | Needs a real quantized model loaded to compare against. |",
        "| self-hosted GPU throughput / time-to-first-token | Needs an actual deployed vLLM/SGLang instance behind `gateway/backend.py`, not a fake provider. |",
        "",
        "No number above this line was estimated and reported as measured — every figure came from a real "
        "run of the code in this repo, against the held-out reference set, on this machine, just now.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
