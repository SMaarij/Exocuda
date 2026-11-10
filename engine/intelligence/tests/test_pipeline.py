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
    cascade = FakeBackend(text="answered by the self-hosted server", est_cost_usd=0.002)
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade)

    result = pipeline.run(messages)

    assert result["tier"] == "self_hosted"
    assert result["escalated"] is True
    assert result["text"] == "answered by the self-hosted server"
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
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade)

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
    `FakePipeline(backend)`, one positional argument, no self_hosted_backend.
    If this test breaks, J-2 will break too.
    """
    pipeline = Pipeline(FakeBackend())
    result = pipeline.run(messages)
    assert result["text"]


# ---- NFR-03: a failing remote tier degrades to local ---------------------------


class _FixedRouter:
    def __init__(self, tier: str) -> None:
        self._tier = tier

    def classify(self, query: str):
        class _Decision:
            tier = self._tier
        return _Decision()


class _FailingRemote(FakeBackend):
    def generate(self, messages: list[Message]) -> BackendResult:
        raise RuntimeError("simulated: network down / bad key")


@pytest.mark.parametrize("tier", ["cloud_byok", "self_hosted"])
def test_failing_remote_tier_falls_back_to_local(messages: list[Message], tier: str) -> None:
    remote = {"cloud_byok_backend": _FailingRemote()} if tier == "cloud_byok" else {"self_hosted_backend": _FailingRemote()}
    pipeline = Pipeline(FakeBackend(text="local answer"), router=_FixedRouter(tier), confidence_floor=None, **remote)
    result = pipeline.run(messages)
    assert result["text"] == "local answer"
    assert result["tier"] == "local"


def test_byok_tier_is_used_when_configured(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="local"), cloud_byok_backend=FakeBackend(text="byok answer", est_cost_usd=0.01), router=_FixedRouter("cloud_byok"))
    result = pipeline.run(messages)
    assert (result["text"], result["tier"]) == ("byok answer", "cloud_byok")


def test_failed_cascade_is_not_retried_when_local_also_fails(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(raise_error=True), self_hosted_backend=_FailingRemote(), router=_FixedRouter("self_hosted"))
    with pytest.raises(LocalExecutionBackendError):
        pipeline.run(messages)


def test_failed_escalation_keeps_the_local_answer(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="?"), self_hosted_backend=_FailingRemote(), confidence_floor=0.99)
    result = pipeline.run(messages)
    assert (result["text"], result["tier"], result["escalated"]) == ("?", "local", False)


# ---- small local model for easy queries -------------------------------------

LONG_ANSWER = "Paris is the capital of France."


def test_small_model_serves_queries_routed_local(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="big"), small_local_backend=FakeBackend(text=LONG_ANSWER), router=_FixedRouter("local"))
    result = pipeline.run(messages)
    assert (result["text"], result["tier"], result["escalated"]) == (LONG_ANSWER, "local", False)
    assert pipeline.last_served_by_small


def test_hard_queries_go_to_the_big_local_model_when_no_cloud(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="big answer"), small_local_backend=FakeBackend(text=LONG_ANSWER),
                        router=_FixedRouter("cloud_byok"))
    result = pipeline.run(messages)
    assert (result["text"], result["tier"]) == ("big answer", "local")
    assert not pipeline.last_served_by_small


def test_weak_small_model_answer_escalates_to_the_big_model(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="big answer, confidently"), small_local_backend=FakeBackend(text="I'm not sure"),
                        router=_FixedRouter("local"), confidence_floor=0.8)
    result = pipeline.run(messages)
    assert (result["text"], result["tier"], result["escalated"]) == ("big answer, confidently", "local", True)
    assert not pipeline.last_served_by_small


class _WithLogprobs(FakeBackend):
    def __init__(self, text: str, logprobs: list) -> None:
        super().__init__(text=text)
        self._logprobs = logprobs

    def generate(self, messages):
        return super().generate(messages) | {"logprobs": self._logprobs}


@pytest.mark.parametrize("p, escalated", [(0.95, False), (0.6, True)])
def test_small_answer_with_low_token_probability_escalates(messages: list[Message], p: float, escalated: bool) -> None:
    import math

    small = _WithLogprobs("An insect has two legs.", [math.log(p)] * 6)  # fluent, but unsure
    pipeline = Pipeline(FakeBackend(text="An insect has six legs."), small_local_backend=small, router=_FixedRouter("local"))
    result = pipeline.run(messages)
    assert result["escalated"] is escalated
    assert result["text"] == ("An insect has six legs." if escalated else "An insect has two legs.")


def test_token_probability_check_can_be_turned_off(messages: list[Message]) -> None:
    small = _WithLogprobs(LONG_ANSWER, [-2.0] * 6)
    pipeline = Pipeline(FakeBackend(text="big"), small_local_backend=small, router=_FixedRouter("local"),
                        small_token_confidence_floor=None)
    assert pipeline.run(messages)["text"] == LONG_ANSWER


def test_failing_small_model_falls_back_to_the_big_model(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="big answer"), small_local_backend=FakeBackend(raise_error=True), router=_FixedRouter("local"))
    assert pipeline.run(messages)["text"] == "big answer"


def test_stream_uses_the_small_model_for_local_queries(messages: list[Message]) -> None:
    pipeline = Pipeline(FakeBackend(text="big"), small_local_backend=FakeBackend(text="small one"), router=_FixedRouter("local"))
    assert "".join(pipeline.run_stream(messages)).strip() == "small one"


def test_warm_loads_both_local_models_without_caching_or_logging(tmp_path) -> None:
    class _Counting(FakeBackend):
        calls = 0

        def generate(self, messages):
            type(self).calls += 1
            return super().generate(messages)

    class _Small(_Counting):
        calls = 0

    log = tmp_path / "decisions.jsonl"
    pipeline = Pipeline(_Counting(), small_local_backend=_Small(), decision_log_path=log)
    pipeline.warm()
    assert (_Counting.calls, _Small.calls) == (1, 1)
    assert not log.exists()
