"""engine/intelligence/tests/test_confidence.py -- Track B, Task 9 (FR-05.3)."""

from __future__ import annotations

from cascade_local.engine.intelligence.router.confidence import ConfidenceConfig, estimate_local_confidence


def test_confident_clean_answer_scores_high() -> None:
    score = estimate_local_confidence("The capital of France is Paris.")
    assert score >= 0.9


def test_hedging_phrase_lowers_confidence() -> None:
    hedged = estimate_local_confidence("I'm not sure, but it might be Paris.")
    clean = estimate_local_confidence("The capital of France is Paris.")
    assert hedged < clean


def test_very_short_answer_lowers_confidence() -> None:
    short = estimate_local_confidence("Yes.")
    longer = estimate_local_confidence("Yes, that is correct and here is why.")
    assert short < longer


def test_score_is_always_in_valid_range() -> None:
    for text in ("", "a", "I'm not sure. I don't know. Unclear. As an AI, I cannot be certain."):
        score = estimate_local_confidence(text)
        assert 0.0 <= score <= 1.0


def test_logprobs_path_used_when_supplied() -> None:
    """FR-05.3's primary path -- kept ready for when a backend exposes this."""
    high_conf = estimate_local_confidence("answer", logprobs=[-0.01, -0.02, -0.01])
    low_conf = estimate_local_confidence("answer", logprobs=[-3.0, -4.0, -2.5])
    assert high_conf > low_conf


def test_custom_config_penalties_apply() -> None:
    config = ConfidenceConfig(hedging_penalty=0.5, short_answer_penalty=0.1, min_answer_chars=5)
    score = estimate_local_confidence("I'm not sure about this one", config=config)
    assert score == 0.5  # 1.0 - 0.5 hedging penalty, answer long enough to skip short penalty
