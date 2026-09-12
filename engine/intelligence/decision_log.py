"""
engine/intelligence/decision_log.py

Track B -- added while Track C built `cascade stats` (FR-08.3), which
needs a per-request log to aggregate. This is distinct from
router/shadow_labeling.py's log: shadow labeling samples a small % of
traffic for router-retraining data; this logs every request's routing
outcome (tier, cache_hit, latency, cost) so `cascade stats` has
something real to read. Logged as a gap found and closed, per
shared/CHANGELOG.md's convention for any Track B addition discovered
after the fact.

Off by default (NFR-05's privacy stance): Pipeline only writes here if
constructed with an explicit `decision_log_path`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from shared.types import PipelineResult

DEFAULT_DECISION_LOG_PATH = Path.home() / ".cascade" / "decisions.jsonl"


def log_decision(result: PipelineResult, log_path: Path = DEFAULT_DECISION_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "tier": result["tier"],
        "cache_hit": result["cache_hit"],
        "confidence": result["confidence"],
        "escalated": result["escalated"],
        "speculative_active": result["speculative_active"],
        "tok_s": result["tok_s"],
        "latency_ms": result["latency_ms"],
        "est_cost_usd": result["est_cost_usd"],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_decisions(log_path: Path = DEFAULT_DECISION_LOG_PATH) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_decisions(records: list[dict]) -> dict:
    """FR-08.3: cache hit rate, local-resolution rate, speculative-decoding speedup."""
    if not records:
        return {
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "local_resolution_rate": 0.0,
            "speculative_speedup": None,
            "total_cost_usd": 0.0,
        }

    total = len(records)
    cache_hits = sum(1 for r in records if r["cache_hit"])
    local_resolved = sum(1 for r in records if r["tier"] == "local")
    total_cost = sum(r["est_cost_usd"] for r in records)

    speculative_tok_s = [r["tok_s"] for r in records if r["speculative_active"] and r["tok_s"] > 0]
    non_speculative_tok_s = [
        r["tok_s"] for r in records if not r["speculative_active"] and r["tier"] == "local" and r["tok_s"] > 0
    ]
    speedup = None
    if speculative_tok_s and non_speculative_tok_s:
        avg_spec = sum(speculative_tok_s) / len(speculative_tok_s)
        avg_non_spec = sum(non_speculative_tok_s) / len(non_speculative_tok_s)
        if avg_non_spec > 0:
            speedup = avg_spec / avg_non_spec

    return {
        "total_requests": total,
        "cache_hit_rate": cache_hits / total,
        "local_resolution_rate": local_resolved / total,
        "speculative_speedup": speedup,
        "total_cost_usd": total_cost,
    }
