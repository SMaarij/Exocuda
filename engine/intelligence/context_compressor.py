"""
engine/intelligence/context_compressor.py

Track B, Tasks 3-5 (FR-04.1-04.3).

Replaces fixed-position truncation ("keep the last 6 turns") with
relevance-ranked compression: every candidate turn's embedding is
compared against the current query's embedding, and low-relevance
turns are summarized or dropped regardless of position -- a genuinely
relevant turn from far earlier in the conversation survives, and an
irrelevant recent one doesn't (FR-04.1's DoD).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.intelligence.embeddings import Embedder, cosine_similarity
from shared.types import Message


@dataclass(frozen=True)
class ContextCompressorConfig:
    relevance_threshold: float = 0.35
    summarize_dropped: bool = True
    summary_max_chars: int = 80
    # The current query (and any trailing turns within this count) are
    # always kept verbatim regardless of relevance score -- the pipeline
    # needs the literal question being asked, not a summary of it.
    always_keep_last_n: int = 1


def _summarize(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class ContextCompressor:
    """
    FR-04.3: exposed as a pure function (compress()) given an already-
    constructed Embedder -- no I/O happens inside compress() itself
    beyond the embedder's own (local, CPU-only) inference calls, and a
    FakeEmbedder in tests touches no network or disk at all.
    """

    def __init__(
        self,
        embedder: Embedder,
        config: ContextCompressorConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or ContextCompressorConfig()

    def compress(self, messages: list[Message]) -> list[Message]:
        if not messages:
            return messages

        system_msgs = [m for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        if len(convo) <= self._config.always_keep_last_n:
            return system_msgs + convo

        keep_from = len(convo) - self._config.always_keep_last_n
        always_kept = convo[keep_from:]
        candidates = convo[:keep_from]

        query_text = always_kept[-1]["content"]
        query_vec = self._embedder.embed(query_text)

        kept: list[Message] = []
        for msg in candidates:
            score = cosine_similarity(self._embedder.embed(msg["content"]), query_vec)
            if score >= self._config.relevance_threshold:
                kept.append(msg)
            elif self._config.summarize_dropped:
                kept.append(
                    {
                        "role": msg["role"],
                        "content": _summarize(msg["content"], self._config.summary_max_chars),
                    }
                )
            # else: dropped entirely -- below threshold, summarization disabled

        return system_msgs + kept + always_kept
