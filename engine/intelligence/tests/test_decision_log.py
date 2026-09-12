"""engine/intelligence/tests/test_decision_log.py -- supports Track C's `cascade stats` (FR-08.3)."""

from __future__ import annotations

from pathlib import Path

from engine.intelligence.decision_log import load_decisions, log_decision, summarize_decisions


def _result(**overrides) -> dict:
    base = {
        "text": "answer", "tier": "local", "cache_hit": False, "confidence": 0.9,
        "escalated": False, "speculative_active": False, "tok_s": 20.0,
        "latency_ms": 100.0, "est_cost_usd": 0.0,
    }
    base.update(overrides)
    return base


def test_log_decision_creates_parent_directory(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "decisions.jsonl"
    log_decision(_result(), log_path)
    assert log_path.exists()


def test_load_decisions_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_decisions(tmp_path / "nope.jsonl") == []


def test_log_and_load_round_trip(tmp_path: Path) -> None:
    log_path = tmp_path / "decisions.jsonl"
    log_decision(_result(tier="local"), log_path)
    log_decision(_result(tier="cache", cache_hit=True), log_path)

    records = load_decisions(log_path)
    assert len(records) == 2
    assert records[0]["tier"] == "local"
    assert records[1]["cache_hit"] is True


def test_summarize_empty_records() -> None:
    summary = summarize_decisions([])
    assert summary["total_requests"] == 0
    assert summary["cache_hit_rate"] == 0.0
    assert summary["speculative_speedup"] is None


def test_summarize_computes_cache_hit_rate() -> None:
    records = [_result(cache_hit=True), _result(cache_hit=False), _result(cache_hit=False), _result(cache_hit=False)]
    summary = summarize_decisions(records)
    assert summary["cache_hit_rate"] == 0.25


def test_summarize_computes_local_resolution_rate() -> None:
    records = [_result(tier="local"), _result(tier="local"), _result(tier="cascade_cloud")]
    summary = summarize_decisions(records)
    assert summary["local_resolution_rate"] == 2 / 3


def test_summarize_computes_speculative_speedup() -> None:
    records = [
        _result(tier="local", speculative_active=True, tok_s=40.0),
        _result(tier="local", speculative_active=True, tok_s=40.0),
        _result(tier="local", speculative_active=False, tok_s=20.0),
        _result(tier="local", speculative_active=False, tok_s=20.0),
    ]
    summary = summarize_decisions(records)
    assert summary["speculative_speedup"] == 2.0


def test_summarize_speedup_none_when_no_comparison_available() -> None:
    records = [_result(tier="local", speculative_active=True, tok_s=40.0)]
    summary = summarize_decisions(records)
    assert summary["speculative_speedup"] is None


def test_summarize_total_cost() -> None:
    records = [_result(est_cost_usd=0.01), _result(est_cost_usd=0.02)]
    summary = summarize_decisions(records)
    assert summary["total_cost_usd"] == 0.03
