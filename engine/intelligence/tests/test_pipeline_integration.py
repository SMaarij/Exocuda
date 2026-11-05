"""
engine/intelligence/tests/test_pipeline_integration.py

Track B, Task 15: full pipeline integration tests -- cache, router,
compressor, and confidence escalation wired together and exercised
through Pipeline.run(), against fake backends (no dependency on Track
A's real LocalExecutionBackend or a live model, per NFR-09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from engine.execution.backend import LocalExecutionBackendError
from engine.intelligence.cache import CacheConfig, SemanticCache
from engine.intelligence.context_compressor import ContextCompressor
from engine.intelligence.pipeline import Pipeline
from engine.intelligence.router.classifier import SemanticRouterClassifier
from engine.intelligence.router.coldstart_data import RoutingExample
from engine.intelligence.router.shadow_labeling import ShadowLabeler
from engine.intelligence.tests.fakes import FakeEmbedder
from shared.types import BackendResult, Message


class _FakeBackend:
    def __init__(self, text: str = "answer", confidence: float = 0.9, raise_error: bool = False, cost: float = 0.0) -> None:
        self._text = text
        self._confidence = confidence
        self._raise_error = raise_error
        self._cost = cost
        self.call_count = 0

    def generate(self, messages: list[Message]) -> BackendResult:
        self.call_count += 1
        if self._raise_error:
            raise LocalExecutionBackendError("simulated failure")
        return {
            "text": self._text, "confidence": self._confidence, "speculative_active": False,
            "tok_s": 10.0, "latency_ms": 5.0, "est_cost_usd": self._cost,
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        self.call_count += 1
        if self._raise_error:
            raise LocalExecutionBackendError("simulated failure")
        yield self._text


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(FakeEmbedder(), db_path=tmp_path / "cache.sqlite3", config=CacheConfig(similarity_threshold=0.5))


@pytest.fixture
def compressor() -> ContextCompressor:
    return ContextCompressor(FakeEmbedder())


@pytest.fixture
def router() -> SemanticRouterClassifier:
    r = SemanticRouterClassifier(FakeEmbedder())
    r.fit([
        RoutingExample("hello there friend", "local"),
        RoutingExample("hi good morning", "local"),
        RoutingExample("write a compiler in rust", "cloud_byok"),
        RoutingExample("optimize this algorithm", "cloud_byok"),
        RoutingExample("what is today's date now", "self_hosted"),
        RoutingExample("search the current news", "self_hosted"),
    ])
    return r


def _msgs(text: str) -> list[Message]:
    return [{"role": "user", "content": text}]


# ---- Cache integration --------------------------------------------------


def test_cache_hit_returns_without_calling_any_backend(cache: SemanticCache) -> None:
    backend = _FakeBackend(text="fresh answer")
    pipeline = Pipeline(backend=backend, cache=cache)

    pipeline.run(_msgs("what is the capital of france"))  # populates cache
    assert backend.call_count == 1

    result = pipeline.run(_msgs("what is the capital of france"))  # should hit cache
    assert result["cache_hit"] is True
    assert result["tier"] == "cache"
    assert backend.call_count == 1  # no new backend call


def test_cache_miss_calls_backend_and_writes_result(cache: SemanticCache) -> None:
    backend = _FakeBackend(text="the answer")
    pipeline = Pipeline(backend=backend, cache=cache)

    result = pipeline.run(_msgs("what is the capital of germany"))

    assert result["cache_hit"] is False
    assert result["tier"] == "local"
    assert cache.lookup(_msgs("what is the capital of germany")) == "the answer"


def test_time_sensitive_query_never_cached_by_pipeline(cache: SemanticCache) -> None:
    backend = _FakeBackend(text="some date")
    pipeline = Pipeline(backend=backend, cache=cache)

    pipeline.run(_msgs("what's today's date"))
    pipeline.run(_msgs("what's today's date"))

    assert backend.call_count == 2  # never cached, so every call hits the backend


# ---- Router integration --------------------------------------------------


def test_router_directs_hard_query_to_cloud_byok(router: SemanticRouterClassifier) -> None:
    local = _FakeBackend(text="local answer")
    cloud_byok = _FakeBackend(text="cloud answer")
    pipeline = Pipeline(backend=local, cloud_byok_backend=cloud_byok, router=router)

    result = pipeline.run(_msgs("write a compiler in rust"))

    assert result["tier"] == "cloud_byok"
    assert local.call_count == 0
    assert cloud_byok.call_count == 1


def test_router_directs_tool_query_to_self_hosted(router: SemanticRouterClassifier) -> None:
    local = _FakeBackend(text="local answer")
    cascade = _FakeBackend(text="cascade answer")
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, router=router)

    result = pipeline.run(_msgs("search the current news"))

    assert result["tier"] == "self_hosted"
    assert local.call_count == 0
    assert cascade.call_count == 1


def test_router_directs_trivial_query_to_local(router: SemanticRouterClassifier) -> None:
    local = _FakeBackend(text="local answer")
    pipeline = Pipeline(backend=local, router=router)

    result = pipeline.run(_msgs("hello there friend"))

    assert result["tier"] == "local"
    assert local.call_count == 1


def test_router_falls_back_to_local_when_requested_backend_missing(router: SemanticRouterClassifier) -> None:
    """Router says cloud_byok, but no cloud_byok_backend was configured -- falls back to local instead of crashing."""
    local = _FakeBackend(text="local fallback answer")
    pipeline = Pipeline(backend=local, router=router)  # no cloud_byok_backend

    result = pipeline.run(_msgs("write a compiler in rust"))

    assert result["tier"] == "local"
    assert result["text"] == "local fallback answer"


# ---- Confidence-based escalation (FR-05.3) -------------------------------


def test_low_confidence_local_answer_escalates_to_cascade() -> None:
    local = _FakeBackend(text="I'm not sure, but maybe")  # hedged -> low confidence
    cascade = _FakeBackend(text="a confident cascade answer")
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, confidence_floor=0.8)

    result = pipeline.run(_msgs("some question"))

    assert result["tier"] == "self_hosted"
    assert result["escalated"] is True
    assert result["text"] == "a confident cascade answer"


def test_confident_local_answer_does_not_escalate() -> None:
    local = _FakeBackend(text="The definite correct answer is 42.")
    cascade = _FakeBackend(text="should not be called")
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, confidence_floor=0.5)

    result = pipeline.run(_msgs("some question"))

    assert result["tier"] == "local"
    assert result["escalated"] is False
    assert cascade.call_count == 0


def test_confidence_escalation_disabled_when_floor_is_none() -> None:
    local = _FakeBackend(text="I'm not sure, but maybe")
    cascade = _FakeBackend(text="should not be called")
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, confidence_floor=None)

    result = pipeline.run(_msgs("some question"))

    assert result["tier"] == "local"
    assert cascade.call_count == 0


# ---- Compressor integration -----------------------------------------------


def test_compressor_runs_before_cache_and_backend() -> None:
    """A long, mostly-irrelevant conversation gets trimmed before either
    the cache key or the backend prompt is built."""
    local = _FakeBackend(text="answer")
    compressor = ContextCompressor(FakeEmbedder())
    pipeline = Pipeline(backend=local, compressor=compressor)

    messages: list[Message] = [{"role": "user", "content": f"filler {i}"} for i in range(10)]
    messages.append({"role": "user", "content": "the actual question"})

    result = pipeline.run(messages)
    assert result["text"] == "answer"  # ran successfully through the compressed path


# ---- Shadow labeling integration -------------------------------------------


def test_shadow_labeler_invoked_after_response_finalized(tmp_path: Path) -> None:
    local = _FakeBackend(text="local answer")
    cascade = _FakeBackend(text="cloud answer")
    labeler = ShadowLabeler(self_hosted_backend=cascade, sample_rate=1.0, log_path=tmp_path / "shadow.jsonl")
    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, shadow_labeler=labeler, confidence_floor=None)

    result = pipeline.run(_msgs("some question"))

    assert result["tier"] == "local"  # served response is local, unaffected by shadow call
    assert result["text"] == "local answer"
    assert (tmp_path / "shadow.jsonl").exists()  # shadow label was written


# ---- Full stack: compressor + cache + router + confidence ------------------


def test_full_stack_end_to_end(cache: SemanticCache, compressor: ContextCompressor, router: SemanticRouterClassifier) -> None:
    local = _FakeBackend(text="a solid local answer here")
    cloud_byok = _FakeBackend(text="a cloud answer")
    cascade = _FakeBackend(text="a cascade answer")

    pipeline = Pipeline(
        backend=local,
        cloud_byok_backend=cloud_byok,
        self_hosted_backend=cascade,
        compressor=compressor,
        cache=cache,
        router=router,
        confidence_floor=0.5,
    )

    # First call: trivial query, no cache yet -> local tier, gets cached.
    r1 = pipeline.run(_msgs("hello there friend"))
    assert r1["tier"] == "local"
    assert r1["cache_hit"] is False

    # Second identical call: should now hit cache.
    r2 = pipeline.run(_msgs("hello there friend"))
    assert r2["cache_hit"] is True

    # Hard query: router sends it to cloud_byok, bypassing local/cache entirely.
    r3 = pipeline.run(_msgs("write a compiler in rust"))
    assert r3["tier"] == "cloud_byok"
