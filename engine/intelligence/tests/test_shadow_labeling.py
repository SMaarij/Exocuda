"""engine/intelligence/tests/test_shadow_labeling.py -- Track B, Task 14 (FR-05.4)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from engine.intelligence.router.shadow_labeling import ShadowLabeler
from shared.types import BackendResult, Message


class _FakeCascadeBackend:
    def __init__(self, text: str = "cloud answer", raise_error: bool = False) -> None:
        self._text = text
        self._raise_error = raise_error

    def generate(self, messages: list[Message]) -> BackendResult:
        if self._raise_error:
            raise RuntimeError("simulated cascade failure")
        return {
            "text": self._text, "confidence": 0.9, "speculative_active": False,
            "tok_s": 10.0, "latency_ms": 5.0, "est_cost_usd": 0.001,
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        yield self._text


def _messages() -> list[Message]:
    return [{"role": "user", "content": "test query"}]


def test_maybe_label_skips_when_no_self_hosted_backend(tmp_path: Path) -> None:
    labeler = ShadowLabeler(self_hosted_backend=None, sample_rate=1.0, log_path=tmp_path / "log.jsonl")
    assert labeler.maybe_label(_messages(), "local", "answer") is False
    assert not (tmp_path / "log.jsonl").exists()


def test_maybe_label_skips_when_not_sampled(tmp_path: Path) -> None:
    rng = random.Random(1)
    labeler = ShadowLabeler(
        self_hosted_backend=_FakeCascadeBackend(), sample_rate=0.0,
        log_path=tmp_path / "log.jsonl", rng=rng,
    )
    assert labeler.maybe_label(_messages(), "local", "answer") is False


def test_maybe_label_writes_log_when_sampled(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    labeler = ShadowLabeler(self_hosted_backend=_FakeCascadeBackend(text="cloud answer"), sample_rate=1.0, log_path=log_path)

    wrote = labeler.maybe_label(_messages(), served_tier="local", served_text="local answer")

    assert wrote is True
    assert log_path.exists()
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["served_tier"] == "local"
    assert record["served_text"] == "local answer"
    assert record["shadow_text"] == "cloud answer"
    assert record["agree"] is False


def test_maybe_label_records_agreement_when_texts_match(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    labeler = ShadowLabeler(self_hosted_backend=_FakeCascadeBackend(text="same answer"), sample_rate=1.0, log_path=log_path)

    labeler.maybe_label(_messages(), served_tier="local", served_text="same answer")

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["agree"] is True


def test_maybe_label_never_raises_when_self_hosted_backend_fails(tmp_path: Path) -> None:
    labeler = ShadowLabeler(self_hosted_backend=_FakeCascadeBackend(raise_error=True), sample_rate=1.0, log_path=tmp_path / "log.jsonl")
    result = labeler.maybe_label(_messages(), "local", "answer")
    assert result is False  # failed silently, did not raise


def test_maybe_label_appends_multiple_records(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    labeler = ShadowLabeler(self_hosted_backend=_FakeCascadeBackend(), sample_rate=1.0, log_path=log_path)

    labeler.maybe_label(_messages(), "local", "a1")
    labeler.maybe_label(_messages(), "local", "a2")

    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
