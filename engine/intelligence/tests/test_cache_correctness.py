"""
engine/intelligence/tests/test_cache_correctness.py

Wrong and stale cache answers found by the 2026-11-06 live benchmark:
  - "...boiling point of water in celsius" was served the stored
    "...in fahrenheit" answer (embeddings score the two as near-identical);
  - "who is the current prime minister" was cached and replayed.
Plus the pipeline's system prompt. All with fakes (NFR-09).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from engine.intelligence.cache import CacheConfig, SemanticCache, is_cacheable, term_swap_conflict
from engine.intelligence.pipeline import Pipeline
from engine.intelligence.tests.fakes import FakeEmbedder
from shared.types import BackendResult, Message


class _SameVectorEmbedder:
    """Every text embeds identically, so only the term-swap guard can tell queries apart."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _cache(tmp_path: Path, **kwargs) -> SemanticCache:
    return SemanticCache(_SameVectorEmbedder(), db_path=tmp_path / "c.sqlite3", config=CacheConfig(**kwargs))


def _msgs(text: str) -> list[Message]:
    return [{"role": "user", "content": text}]


# ---- term_swap_conflict --------------------------------------------------


@pytest.mark.parametrize(
    ("cached", "answer", "new"),
    [
        ("what's the boiling point of water in fahrenheit", "212°F at sea level.", "what is the boiling point of water in celsius"),
        ("what's 15 times 4", "60.", "what's 15 times 5"),
        ("what's the capital of japan", "Tokyo.", "what's the capital of china"),
        ("convert 5 kilograms to pounds", "About 11 pounds.", "convert 5 pounds to kilograms"),
    ],
)
def test_strict_guard_flags_swaps_and_reorders(cached: str, answer: str, new: str) -> None:
    assert term_swap_conflict(new, cached, answer, mode="strict") is True


@pytest.mark.parametrize(
    ("cached", "answer", "new"),
    [
        ("what is the capital of france", "The capital of France is Paris.", "what's france's capital city"),
        ("convert 10 miles to kilometers", "About 16.09 kilometers.", "how many kilometers is 10 miles"),
        ("what year did world war two end", "World War II ended in 1945.", "in what year did world war 2 come to an end"),
        ("how many continents are there", "There are 7 continents.", "what is the total number of continents"),
    ],
)
def test_strict_guard_allows_rewordings_that_add_or_drop_words(cached: str, answer: str, new: str) -> None:
    assert term_swap_conflict(new, cached, answer, mode="strict") is False


def test_lenient_guard_misses_a_swap_behind_a_terse_answer() -> None:
    # The documented trade-off: "212°F" never spells out "fahrenheit".
    args = ("what is the boiling point of water in celsius", "what's the boiling point of water in fahrenheit", "212°F.")
    assert term_swap_conflict(*args, mode="lenient") is False
    assert term_swap_conflict(*args, mode="strict") is True


def test_word_forms_count_as_the_same_term() -> None:
    assert term_swap_conflict("correct spelling of definitely", "how do you spell definitely", "D-E-F-I-N-I-T-E-L-Y.", mode="strict") is False


def test_numbers_never_match_by_prefix() -> None:
    assert term_swap_conflict("what's 12 times 45", "what's 12 times 4", "48.", mode="strict") is True


def test_off_mode_and_bad_mode() -> None:
    assert term_swap_conflict("in celsius", "in fahrenheit", "212°F", mode="off") is False
    with pytest.raises(ValueError):
        term_swap_conflict("a", "b", "c", mode="sometimes")


# ---- SemanticCache with the guard ---------------------------------------


def test_cache_refuses_a_swapped_unit_by_default(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.write(_msgs("what's the boiling point of water in fahrenheit"), "212°F at sea level.")
    assert cache.lookup(_msgs("what is the boiling point of water in celsius")) is None


def test_cache_still_serves_a_rewording(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.write(_msgs("what is the capital of france"), "The capital of France is Paris.")
    assert cache.lookup(_msgs("what's france's capital city")) == "The capital of France is Paris."


def test_cache_falls_through_to_the_next_candidate(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    cache.write(_msgs("boiling point of water in fahrenheit"), "212°F.")
    cache.write(_msgs("boiling point of water in celsius"), "100°C.")
    assert cache.lookup(_msgs("what is the boiling point of water in celsius")) == "100°C."


def test_guard_off_restores_similarity_only_behaviour(tmp_path: Path) -> None:
    cache = _cache(tmp_path, term_swap_guard="off")
    cache.write(_msgs("what's the boiling point of water in fahrenheit"), "212°F.")
    assert cache.lookup(_msgs("what is the boiling point of water in celsius")) == "212°F."


def test_guard_survives_a_restart(tmp_path: Path) -> None:
    _cache(tmp_path).write(_msgs("what's 15 times 4"), "60.")
    reopened = _cache(tmp_path)  # stored query text is reloaded from SQLite
    assert reopened.lookup(_msgs("what's 15 times 5")) is None


# ---- Cacheability patterns ---------------------------------------------


@pytest.mark.parametrize("query", ["who is the current prime minister of the uk", "what are today's headlines"])
def test_default_patterns_catch_live_questions(query: str) -> None:
    assert is_cacheable(query) is False


def test_cacheability_patterns_are_configurable() -> None:
    config = CacheConfig(time_sensitive_patterns=(r"\bstock\b",), personalized_patterns=())
    assert is_cacheable("what is the current prime minister", config) is True
    assert is_cacheable("apple stock price", config) is False


# ---- Pipeline: live-data tier and system prompt --------------------------


class _RecordingBackend:
    def __init__(self, text: str = "answer") -> None:
        self._text = text
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> BackendResult:
        self.calls.append(messages)
        return {"text": self._text, "confidence": 0.9, "speculative_active": False,
                "tok_s": 10.0, "latency_ms": 1.0, "est_cost_usd": 0.0}

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        yield self._text


class _FixedRouter:
    def __init__(self, tier: str) -> None:
        self.tier = tier

    def classify(self, query: str):
        tier = self.tier

        class _Decision:
            pass

        d = _Decision()
        d.tier, d.confidence = tier, 0.9
        return d


def _pipeline(tmp_path: Path, tier: str, **kwargs):
    backend = _RecordingBackend()
    cache = SemanticCache(FakeEmbedder(), db_path=tmp_path / "p.sqlite3")
    return Pipeline(backend=backend, cache=cache, router=_FixedRouter(tier), confidence_floor=None, **kwargs), backend


def test_live_data_tier_is_never_cached(tmp_path: Path) -> None:
    # No time word the patterns could catch: only the router knows this is live data.
    pipeline, backend = _pipeline(tmp_path, "self_hosted")
    pipeline.run(_msgs("who leads the uk government"))
    second = pipeline.run(_msgs("who leads the uk government"))
    assert second["cache_hit"] is False
    assert len(backend.calls) == 2


def test_other_tiers_are_cached(tmp_path: Path) -> None:
    pipeline, backend = _pipeline(tmp_path, "local")
    pipeline.run(_msgs("who wrote hamlet"))
    assert pipeline.run(_msgs("who wrote hamlet"))["cache_hit"] is True
    assert len(backend.calls) == 1


def test_uncached_tiers_is_configurable(tmp_path: Path) -> None:
    pipeline, backend = _pipeline(tmp_path, "self_hosted", uncached_tiers=())
    pipeline.run(_msgs("who leads the uk government"))
    assert pipeline.run(_msgs("who leads the uk government"))["cache_hit"] is True


def test_stream_skips_cache_for_live_data_tier(tmp_path: Path) -> None:
    pipeline, backend = _pipeline(tmp_path, "self_hosted")
    pipeline.run(_msgs("who leads the uk government"))
    assert "".join(pipeline.run_stream(_msgs("who leads the uk government"))) == "answer"
    assert len(backend.calls) == 2


def test_system_prompt_is_prepended(tmp_path: Path) -> None:
    pipeline, backend = _pipeline(tmp_path, "local", system_prompt="Be brief.")
    pipeline.run(_msgs("hi"))
    assert backend.calls[0][0] == {"role": "system", "content": "Be brief."}
    assert backend.calls[0][-1]["content"] == "hi"


def test_existing_system_message_is_kept_as_is(tmp_path: Path) -> None:
    pipeline, backend = _pipeline(tmp_path, "local", system_prompt="Be brief.")
    pipeline.run([{"role": "system", "content": "Talk like a pirate."}, *_msgs("hi")])
    assert [m["content"] for m in backend.calls[0] if m["role"] == "system"] == ["Talk like a pirate."]


@pytest.mark.parametrize("value", [None, ""])
def test_no_system_prompt_sends_messages_unchanged(tmp_path: Path, value) -> None:
    pipeline, backend = _pipeline(tmp_path, "local", system_prompt=value)
    pipeline.run(_msgs("hi"))
    assert backend.calls[0] == _msgs("hi")


def test_unknown_guard_mode_is_rejected_up_front() -> None:
    with pytest.raises(ValueError):
        CacheConfig(term_swap_guard="stirct")
