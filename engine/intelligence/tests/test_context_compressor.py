"""
engine/intelligence/tests/test_context_compressor.py

Track B, Tasks 3-5 (FR-04.1-04.3). All tests use FakeEmbedder --
no model download, no network (NFR-09 spirit).
"""

from __future__ import annotations

from engine.intelligence.context_compressor import (
    ContextCompressor,
    ContextCompressorConfig,
    _summarize,
)
from engine.intelligence.tests.fakes import FakeEmbedder
from shared.types import Message


def _compressor(**config_kwargs) -> ContextCompressor:
    return ContextCompressor(FakeEmbedder(), ContextCompressorConfig(**config_kwargs))


def test_compress_returns_input_unchanged_when_empty() -> None:
    assert _compressor().compress([]) == []


def test_compress_always_keeps_current_query() -> None:
    messages: list[Message] = [
        {"role": "user", "content": "totally unrelated topic about gardening"},
        {"role": "assistant", "content": "sure, here is info about gardening"},
        {"role": "user", "content": "what is the python programming language"},
    ]
    result = _compressor().compress(messages)
    assert result[-1]["content"] == "what is the python programming language"


def test_compress_keeps_a_relevant_turn_from_far_earlier(monkeypatch=None) -> None:
    """FR-04.1's DoD: a relevant turn from 20 messages back survives."""
    messages: list[Message] = [
        {"role": "user", "content": "let's talk about python programming"},
        {"role": "assistant", "content": "sure, python is a great language"},
    ]
    # 18 irrelevant filler turns in between, alternating roles
    for i in range(18):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"random gardening tip number {i}"})
    messages.append({"role": "user", "content": "what is python programming used for"})

    result = _compressor(relevance_threshold=0.3, summarize_dropped=False).compress(messages)
    contents = [m["content"] for m in result]

    # the early, genuinely relevant turn survives
    assert "let's talk about python programming" in contents
    # most of the 18 irrelevant filler turns are gone
    filler_survivors = sum(1 for c in contents if "gardening tip" in c)
    assert filler_survivors < 18


def test_compress_drops_or_summarizes_irrelevant_recent_turn() -> None:
    """An irrelevant turn close to the end still gets filtered, not kept just because of recency."""
    messages: list[Message] = [
        {"role": "user", "content": "explain quantum computing superposition basics"},
        {"role": "assistant", "content": "quantum computing uses qubits"},
        {"role": "user", "content": "completely unrelated: what's a good pizza topping"},
        {"role": "assistant", "content": "pepperoni is popular"},
        {"role": "user", "content": "back to quantum computing superposition, explain more"},
    ]
    result = _compressor(relevance_threshold=0.3, summarize_dropped=False).compress(messages)
    contents = [m["content"] for m in result]
    assert "completely unrelated: what's a good pizza topping" not in contents
    assert "explain quantum computing superposition basics" in contents


def test_compress_summarizes_dropped_turns_when_enabled() -> None:
    long_text = "x" * 200
    messages: list[Message] = [
        {"role": "user", "content": long_text},
        {"role": "user", "content": "totally different query about something else entirely"},
    ]
    result = _compressor(relevance_threshold=0.99, summarize_dropped=True, summary_max_chars=50).compress(messages)
    assert len(result[0]["content"]) <= 50


def test_compress_drops_entirely_when_summarize_disabled() -> None:
    messages: list[Message] = [
        {"role": "user", "content": "irrelevant filler content about cooking"},
        {"role": "user", "content": "a completely different query about astrophysics"},
    ]
    result = _compressor(relevance_threshold=0.99, summarize_dropped=False).compress(messages)
    # only the always-kept current query remains
    assert len(result) == 1
    assert result[0]["content"] == "a completely different query about astrophysics"


def test_compress_preserves_system_messages_always() -> None:
    messages: list[Message] = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "irrelevant filler about weather patterns"},
        {"role": "user", "content": "a query about database indexing"},
    ]
    result = _compressor(relevance_threshold=0.99, summarize_dropped=False).compress(messages)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "you are a helpful assistant"


def test_compress_is_pure_and_deterministic() -> None:
    """FR-04.3: pure function -- same input, same output, no side effects."""
    messages: list[Message] = [
        {"role": "user", "content": "topic one"},
        {"role": "user", "content": "topic two, the actual query"},
    ]
    compressor = _compressor()
    result_a = compressor.compress(messages)
    result_b = compressor.compress(messages)
    assert result_a == result_b
    # input list itself is untouched
    assert messages[0]["content"] == "topic one"


def test_compress_short_conversation_returns_everything() -> None:
    messages: list[Message] = [{"role": "user", "content": "just one message"}]
    result = _compressor().compress(messages)
    assert result == messages


def test_summarize_helper_truncates_long_text() -> None:
    text = "a" * 100
    result = _summarize(text, max_chars=20)
    assert len(result) == 20
    assert result.endswith("…")


def test_summarize_helper_leaves_short_text_untouched() -> None:
    assert _summarize("short", max_chars=20) == "short"
