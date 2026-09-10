"""
engine/intelligence/tests/test_pipeline.py

Track B, Task 2. Tests Pipeline entirely against fake ExecutionBackend
implementations -- no dependency on Track A's real LocalExecutionBackend
and no live GPU/Ollama (NFR-09). This is the "Track B can build and
fully test Pipeline against two fake backends before either real
implementation exists" property the master doc's Section 9 describes.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from engine.execution.backend import LocalExecutionBackendError
from engine.intelligence.pipeline import Pipeline
from shared.types import BackendResult, Message


# ---- Fake ExecutionBackend implementations ---------------------------------


class FakeBackend:
    """Conforms to ExecutionBackend. Returns a canned, well-typed result."""

    def __init__(
        self,
        text: str = "hello from fake backend",
        confidence: float = 0.9,
        speculative_active: bool = False,
        tok_s: float = 42.0,
        est_cost_usd: float = 0.0,
        raise_error: bool = False,
    ) -> None:
        self._text = text
        self._confidence = confidence
        self._speculative_active = speculative_active
        self._tok_s = tok_s
        self._est_cost_usd = est_cost_usd
        self._raise_error = raise_error

    def generate(self, messages: list[Message]) -> BackendResult:
        if self._raise_error:
            raise LocalExecutionBackendError("simulated: all rungs exhausted")
        return {
            "text": self._text,
            "confidence": self._confidence,
            "speculative_active": self._speculative_active,
            "tok_s": self._tok_s,
            "latency_ms": 12.3,
            "est_cost_usd": self._est_cost_usd,
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        if self._raise_error:
            raise LocalExecutionBackendError("simulated: all rungs exhausted")
        for word in self._text.split():
            yield word + " "


@pytest.fixture
def messages() -> list[Message]:
    return [{"role": "user", "content": "Hello"}]


# ---- run() ------------------------------------------------------------------


def test_run_returns_pipeline_result_shape(messages: list[Message]) -> None:
    pipeline = Pipeline(backend=FakeBackend())
    result = pipeline.run(messages)

    for field in (
        "text", "tier", "cache_hit", "confidence", "escalated",
        "speculative_active", "tok_s", "latency_ms", "est_cost_usd",
    ):
        assert field in result


def test_run_reports_local_tier_and_not_escalated_on_success(
    messages: list[Message],
) -> None:
    pipeline = Pipeline(backend=FakeBackend())
    result = pipeline.run(messages)

    assert result["tier"] == "local"
    assert result["escalated"] is False
    assert result["cache_hit"] is False  # no SemanticCache yet (Task 10-13)


def test_run_passes_through_backend_result_fields(messages: list[Message]) -> None:
    backend = FakeBackend(text="specific answer", confidence=0.77, tok_s=55.5)
    pipeline = Pipeline(backend=backend)
    result = pipeline.run(messages)

    assert result["text"] == "specific answer"
    assert result["confidence"] == 0.77
    assert result["tok_s"] == 55.5


def test_run_escalates_to_cascade_when_local_raises_and_cascade_provided(
    messages: list[Message],
) -> None:
    local = FakeBackend(raise_error=True)
    cascade = FakeBackend(text="answered by cascade cloud", est_cost_usd=0.002)
    pipeline = Pipeline(backend=local, cascade_backend=cascade)

    result = pipeline.run(messages)

    assert result["tier"] == "cascade_cloud"
    assert result["escalated"] is True
    assert result["text"] == "answered by cascade cloud"
    assert result["est_cost_usd"] == 0.002


def test_run_reraises_when_local_raises_and_no_cascade_configured(
    messages: list[Message],
) -> None:
    """Mirrors j2_integration_test.py's test_j2_5 exactly."""
    pipeline = Pipeline(backend=FakeBackend(raise_error=True))

    with pytest.raises(LocalExecutionBackendError):
        pipeline.run(messages)


def test_run_latency_ms_is_nonnegative(messages: list[Message]) -> None:
    pipeline = Pipeline(backend=FakeBackend())
    result = pipeline.run(messages)

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


# ---- run_stream() -------------------------------------------------------


def test_run_stream_yields_at_least_one_token(messages: list[Message]) -> None:
    pipeline = Pipeline(backend=FakeBackend(text="one two three"))
    tokens = list(pipeline.run_stream(messages))

    assert len(tokens) >= 1
    assert "".join(tokens).strip() == "one two three"


def test_run_stream_escalates_to_cascade_when_local_raises(
    messages: list[Message],
) -> None:
    local = FakeBackend(raise_error=True)
    cascade = FakeBackend(text="cloud stream")
    pipeline = Pipeline(backend=local, cascade_backend=cascade)

    tokens = list(pipeline.run_stream(messages))

    assert "".join(tokens).strip() == "cloud stream"


def test_run_stream_reraises_when_no_cascade_configured(
    messages: list[Message],
) -> None:
    pipeline = Pipeline(backend=FakeBackend(raise_error=True))

    with pytest.raises(LocalExecutionBackendError):
        list(pipeline.run_stream(messages))


# ---- constructor shape (matches Track A's documented example exactly) ---


def test_pipeline_constructs_with_single_positional_backend_argument(
    messages: list[Message],
) -> None:
    """
    This is the exact call shape j2_integration_test.py uses:
    `FakePipeline(backend)`, one positional argument, no cascade_backend.
    If this test breaks, J-2 will break too.
    """
    pipeline = Pipeline(FakeBackend())
    result = pipeline.run(messages)
    assert result["text"]
