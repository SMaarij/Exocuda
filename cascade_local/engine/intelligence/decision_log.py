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
from typing import Optional

from cascade_local.shared.types import PipelineResult

DEFAULT_DECISION_LOG_PATH = Path.home() / ".cascade" / "decisions.jsonl"


def log_decision(result: PipelineResult, log_path: Path = DEFAULT_DECISION_LOG_PATH,
                 small: Optional[bool] = None) -> None:
    """Append one answer's routing record. `small`: whether the small local model answered
    (None when there is no small model or the answer did not come from a local model)."""
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
    if small is not None:
        record["small"] = bool(small)
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


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def source(record: dict) -> str:
    """Who answered: "cache", "small" (the small local model), "local" (the main local
    model), or the remote tier's name."""
    if record.get("cache_hit") or record.get("tier") == "cache":
        return "cache"
    if record.get("tier") == "local":
        return "small" if record.get("small") else "local"
    return record.get("tier", "?")


def summarize_decisions(records: list[dict], since: Optional[float] = None) -> dict:
    """Cache hit rate, local-resolution rate and speculative-decoding speedup, plus answers
    by source, their median response times, and the time the cache saved: for each cache hit,
    the median time of a fresh local answer minus the hit's own time. `since` (a Unix time)
    keeps only later records."""
    if since is not None:
        records = [r for r in records if r.get("ts", 0) >= since]
    if not records:
        return {
            "total_requests": 0,
            "cache_hit_rate": 0.0,
            "local_resolution_rate": 0.0,
            "speculative_speedup": None,
            "total_cost_usd": 0.0,
            "by_source": {},
            "median_ms": {},
            "time_saved_s": 0.0,
            "speculative_samples": (0, 0),
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

    by_source: dict[str, int] = {}
    times: dict[str, list[float]] = {}
    for r in records:
        src = source(r)
        by_source[src] = by_source.get(src, 0) + 1
        times.setdefault(src, []).append(float(r.get("latency_ms", 0.0)))
    median_ms = {k: _median(v) for k, v in times.items()}
    fresh = _median(times.get("local", []) + times.get("small", []))
    saved_ms = sum(max(fresh - t, 0.0) for t in times.get("cache", [])) if fresh is not None else 0.0

    return {
        "total_requests": total,
        "cache_hit_rate": cache_hits / total,
        "local_resolution_rate": local_resolved / total,
        "speculative_speedup": speedup,
        "total_cost_usd": total_cost,
        "by_source": by_source,
        "median_ms": median_ms,
        "time_saved_s": saved_ms / 1000.0,
        "speculative_samples": (len(speculative_tok_s), len(non_speculative_tok_s)),
    }
