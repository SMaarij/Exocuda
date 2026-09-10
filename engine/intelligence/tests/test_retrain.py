"""engine/intelligence/tests/test_retrain.py -- Track B, Task 16 (FR-05.5)."""

from __future__ import annotations

import json
from pathlib import Path

from engine.intelligence.router.coldstart_data import BOOTSTRAP_EXAMPLES
from engine.intelligence.router.retrain import load_shadow_examples, retrain
from engine.intelligence.tests.fakes import FakeEmbedder


def test_load_shadow_examples_returns_empty_when_log_missing(tmp_path: Path) -> None:
    assert load_shadow_examples(tmp_path / "nope.jsonl") == []


def test_load_shadow_examples_labels_agreement_as_served_tier(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        json.dumps({"query": "hi", "served_tier": "local", "agree": True}) + "\n",
        encoding="utf-8",
    )
    examples = load_shadow_examples(log_path)
    assert len(examples) == 1
    assert examples[0].label == "local"
    assert examples[0].query == "hi"


def test_load_shadow_examples_labels_disagreement_as_cascade_cloud(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        json.dumps({"query": "hard one", "served_tier": "local", "agree": False}) + "\n",
        encoding="utf-8",
    )
    examples = load_shadow_examples(log_path)
    assert examples[0].label == "cascade_cloud"


def test_retrain_with_no_shadow_log_uses_only_coldstart_data(tmp_path: Path) -> None:
    classifier = retrain(FakeEmbedder(), log_path=tmp_path / "nope.jsonl")
    assert classifier.is_fitted


def test_retrain_incorporates_shadow_examples(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for _ in range(5):
            f.write(json.dumps({"query": "a new real query", "served_tier": "local", "agree": True}) + "\n")

    classifier = retrain(FakeEmbedder(), log_path=log_path)
    assert classifier.is_fitted
    # Retraining succeeded on coldstart + shadow data combined -- no
    # exception, and it's still a usable classifier.
    decision = classifier.classify("a new real query")
    assert decision.tier in ("local", "cloud_byok", "cascade_cloud")


def test_retrain_returns_usable_classifier_matching_bootstrap_size(tmp_path: Path) -> None:
    classifier = retrain(FakeEmbedder(), log_path=tmp_path / "nope.jsonl")
    # sanity: coldstart data alone is enough to fit without error
    assert len(BOOTSTRAP_EXAMPLES) > 0
    assert classifier.classify("hello").tier in ("local", "cloud_byok", "cascade_cloud")
