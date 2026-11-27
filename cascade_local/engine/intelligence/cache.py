"""
engine/intelligence/cache.py

Track B, Tasks 10-13 (FR-06.1-06.4).

Two layers (Master Doc Component 6), checked in order:
    1. Exact-prefix (Task 13, FR-06.4) -- a hash of the full message
       list. Cheap, always correct, checked first.
    2. Fuzzy semantic (Tasks 10-12, FR-06.1-06.3) -- embedding cosine
       similarity against the last query, vectorized for the <50ms
       lookup target at 10,000 entries.

FLAGGED SCOPE NOTE on the exact-prefix layer: true *partial* KV-cache-
block reuse (splicing a shared system-prompt prefix into a fresh
generation) needs cooperation from the model runtime itself -- today's
ExecutionBackend protocol has no hook for that (see shared/CHANGELOG.md).
This layer implements the exact-full-match case instead: if an entire
message list is byte-identical to a prior one, the cached response
returns with zero backend call at all, which is the degenerate but
always-correct instance of prefix caching (a real partial-prefix hit
saves time-to-first-token; a full-match hit saves the whole call).
Extending to true partial-prefix reuse is a future ExecutionBackend
protocol change requiring all three tracks' sign-off.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from cascade_local.engine.intelligence.embeddings import Embedder
from cascade_local.shared.types import Message

DEFAULT_DB_PATH = Path.home() / ".cascade" / "cache.sqlite3"

# FR-06.2: rule-based by design, unlike SemanticRouter. Cacheability is
# a safety filter -- never serve a stale answer to "what's today's
# date" -- and that is exactly the kind of decision you want
# deterministic, auditable rules for. These are only the defaults: pass
# your own via CacheConfig. The pipeline additionally skips the cache for
# queries the router sends to a live-data tier (Pipeline.uncached_tiers),
# which catches phrasings these patterns don't.
DEFAULT_TIME_SENSITIVE_PATTERNS: tuple[str, ...] = (
    r"\btoday'?s?\b",
    r"\bcurrent(ly)?\b",
    r"\bright now\b",
    r"\blatest\b",
    r"\bthis (week|month|year)\b",
    r"\bwhat time is it\b",
    r"\bnow\b",
)
DEFAULT_PERSONALIZED_PATTERNS: tuple[str, ...] = (
    r"\bmy (account|balance|order|password|email|name)\b",
    r"\byour account\b",
)


GUARD_MODES = ("strict", "lenient", "off")


@dataclass(frozen=True)
class CacheConfig:
    # Cosine similarity a stored query needs to count as "the same question".
    # scripts/run_benchmark.py sweeps this against the paraphrase and
    # near-miss fixtures and reports hit rate and wrong-answer rate per value.
    similarity_threshold: float = 0.92
    default_ttl_seconds: Optional[int] = 7 * 24 * 3600  # None = never expires
    time_sensitive_patterns: tuple[str, ...] = DEFAULT_TIME_SENSITIVE_PATTERNS
    personalized_patterns: tuple[str, ...] = DEFAULT_PERSONALIZED_PATTERNS
    # Embeddings score "...in fahrenheit" / "...in celsius" and "5 kg to lb" /
    # "5 lb to kg" as near-identical. The guard rejects such matches:
    #   "strict"  -- the new question swapped in a term the stored answer never
    #                mentions. Safest; also turns away some rewordings that swap
    #                a verb ("wrote" -> "author"), which then cost a fresh answer.
    #   "lenient" -- only when the answer also mentions the term that was
    #                swapped out. Misses swaps behind terse answers ("212°F").
    #   "off"     -- similarity alone.
    # Both modes also reject the same key terms in a different order.
    term_swap_guard: str = "strict"
    # Candidates the threshold or the guard turn away but that are at least this similar are
    # put to the judge (SemanticCache(judge=...)), a local model asked whether the two
    # questions ask the same thing. None: no judge.
    judge_floor: Optional[float] = 0.80
    judge_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.term_swap_guard not in GUARD_MODES:
            raise ValueError(f"term_swap_guard must be one of {GUARD_MODES}, got {self.term_swap_guard!r}")
    # Two words count as the same term if they share at least this many
    # leading letters and one is a prefix of the other ("current"/"currently").
    min_shared_prefix: int = 4


# The fixed instruction comes first, so a model that keeps its prompt prefix computes it once.
JUDGE_PROMPT = ("Do these two questions ask for exactly the same information, so that one correct answer "
                "answers both? Reply yes or no.\n\nQuestion 1: {cached}\nQuestion 2: {new}")


def is_cacheable(query: str, config: Optional[CacheConfig] = None) -> bool:
    """FR-06.2."""
    config = config or CacheConfig()
    text = query.lower()
    for pattern in config.time_sensitive_patterns + config.personalized_patterns:
        if re.search(pattern, text):
            return False
    return True


# Ways of asking the same question, mapped to one term so the term-swap guard
# sees a rewording, not a swap ("Who is the author of X" / "Who wrote X").
# Only question phrasing belongs here -- never content words whose swap changes
# the answer (places, units, numbers, names).
_SYNONYMS: dict[str, str] = {
    **dict.fromkeys(("author", "authored", "writer", "written", "penned"), "wrote"),
    **dict.fromkeys(("painter",), "painted"),
    **dict.fromkeys(("inventor",), "invented"),
    **dict.fromkeys(("discoverer",), "discovered"),
    **dict.fromkeys(("founder",), "founded"),
    **dict.fromkeys(("biggest",), "largest"),
    **dict.fromkeys(("big",), "large"),
    **dict.fromkeys(("tiniest",), "smallest"),
    **dict.fromkeys(("quickest",), "fastest"),
    **dict.fromkeys(("means", "meaning", "stand", "stands"), "mean"),
}
# Words that frame a question without changing what it asks.
_FILLER = frozenset(("abbreviation", "acronym", "please", "tell", "exactly", "briefly"))


def _terms(text: str) -> list[str]:
    """Content terms in order: lowercase words and numbers minus scikit-learn's English stop words,
    question-framing filler dropped and question-phrasing synonyms mapped to one form."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_SYNONYMS.get(w, w) for w in words
            if w.isdigit() or (len(w) > 1 and w not in ENGLISH_STOP_WORDS and w not in _FILLER)]


def _same_term(a: str, b: str, min_prefix: int) -> bool:
    if a == b:
        return True
    if a.isdigit() or b.isdigit():
        return False
    short, long_ = sorted((a, b), key=len)
    return len(short) >= min_prefix and long_.startswith(short)


def _unmatched(terms: list[str], others: list[str], min_prefix: int) -> list[str]:
    return [t for t in terms if not any(_same_term(t, o, min_prefix) for o in others)]


def term_swap_conflict(
    new_query: str, cached_query: str, cached_answer: str, min_prefix: int = 4, mode: str = "strict"
) -> bool:
    """
    True if serving `cached_answer` (written for `cached_query`) to
    `new_query` would likely be wrong even though the two embed as
    near-identical:
      - a swap: the cached query has a term the new one lacks, and the new
        query has a term the answer never mentions (fahrenheit -> celsius,
        4 -> 5, japan -> china). In "lenient" mode the answer must also
        mention the term that was swapped out.
      - a reorder: exactly the same terms in a different order (a reversed
        conversion or comparison).
    Adding or dropping words, without swapping one for another, passes.
    """
    if mode not in GUARD_MODES:
        raise ValueError(f"term_swap_guard must be one of {GUARD_MODES}, got {mode!r}")
    if mode == "off":
        return False
    new_t, cached_t, answer_t = _terms(new_query), _terms(cached_query), _terms(cached_answer)
    only_cached = _unmatched(cached_t, new_t, min_prefix)
    only_new = _unmatched(new_t, cached_t, min_prefix)
    answer_lacks_added = any(_unmatched([t], answer_t, min_prefix) for t in only_new)
    if mode == "strict":
        swapped = bool(only_cached)
    else:
        swapped = any(not _unmatched([t], answer_t, min_prefix) for t in only_cached)
    if swapped and answer_lacks_added:
        return True
    return set(new_t) == set(cached_t) and new_t != cached_t


def _prefix_key(messages: list[Message]) -> str:
    blob = json.dumps(messages, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class SemanticCache:
    """
    Task 12/FR-06.3 architecture note: the naive design -- re-querying
    every row from SQLite on every lookup() call -- was measured and
    profiled at ~70-90ms at 10,000 entries on this sandbox's (1-core)
    hardware, almost entirely SQLite row-materialization cost, not the
    similarity math itself (which measured ~3-7ms). SQLite here is the
    durable store, not the per-lookup source of truth: embeddings are
    mirrored into an in-memory index (self._embedding_blobs and
    friends), loaded once at construction and updated incrementally on
    write(). lookup()'s semantic layer reads only that in-memory index.
    """

    def __init__(
        self,
        embedder: Embedder,
        db_path: Path = DEFAULT_DB_PATH,
        config: Optional[CacheConfig] = None,
        judge: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        self._embedder = embedder
        self._config = config or CacheConfig()
        # judge(prompt) -> P(yes) or None (cannot judge now); see CacheConfig.judge_floor.
        self.judge = judge
        self.judged = 0  # judge calls, for inspection
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._init_schema()

        # In-process exact-prefix layer (Task 13).
        self._prefix_cache: dict[str, tuple[str, Optional[float]]] = {}

        # In-memory mirror of the semantic layer (Task 12 perf fix, see
        # class docstring). `_matrix` is rebuilt lazily from
        # `_embedding_blobs` only when dirty (after a write), so a run
        # of many lookups with no interleaved writes pays the rebuild
        # cost once, not per lookup.
        self._ids: list[int] = []
        self._embedding_blobs: list[bytes] = []
        self._queries: dict[int, str] = {}
        self._responses: dict[int, str] = {}
        self._expires_at: dict[int, Optional[float]] = {}
        self._hit_counts: dict[int, int] = {}
        self._matrix: Optional[np.ndarray] = None
        self._reindex_from_db()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_embedding BLOB NOT NULL,
                response_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def _reindex_from_db(self) -> None:
        """Rebuilds the in-memory index from SQLite -- called once at
        construction (so an existing on-disk cache is usable
        immediately) and available for tests that seed rows directly."""
        rows = self._conn.execute(
            "SELECT id, query_embedding, response_text, expires_at, hit_count, query_text FROM cache_entries"
        ).fetchall()
        self._ids = [r[0] for r in rows]
        self._embedding_blobs = [r[1] for r in rows]
        self._responses = {r[0]: r[2] for r in rows}
        self._queries = {r[0]: r[5] for r in rows}
        self._expires_at = {r[0]: r[3] for r in rows}
        self._hit_counts = {r[0]: r[4] for r in rows}
        self._matrix = None  # rebuilt lazily on first lookup

    # ---- Public API ------------------------------------------------------

    def lookup(self, messages: list[Message]) -> Optional[str]:
        prefix_hit = self._lookup_prefix(messages)
        if prefix_hit is not None:
            return prefix_hit
        return self._lookup_semantic(messages)

    def write(self, messages: list[Message], response_text: str, ttl_seconds: Optional[int] = ...) -> None:
        query = messages[-1]["content"] if messages else ""
        if not query or not is_cacheable(query, self._config):
            return

        ttl = self._config.default_ttl_seconds if ttl_seconds is ... else ttl_seconds
        expires_at = (time.time() + ttl) if ttl else None

        key = _prefix_key(messages)
        self._prefix_cache[key] = (response_text, expires_at)

        vec = np.array(self._embedder.embed(query), dtype=np.float64)
        blob = vec.tobytes()
        cursor = self._conn.execute(
            "INSERT INTO cache_entries (query_text, query_embedding, response_text, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, blob, response_text, time.time(), expires_at),
        )
        self._conn.commit()

        # Incremental in-memory update -- O(1) append, no full reindex.
        row_id = cursor.lastrowid
        self._ids.append(row_id)
        self._embedding_blobs.append(blob)
        self._responses[row_id] = response_text
        self._queries[row_id] = query
        self._expires_at[row_id] = expires_at
        self._hit_counts[row_id] = 0
        self._matrix = None  # dirty -- rebuilt lazily on next lookup

    def close(self) -> None:
        self._conn.close()

    # ---- Layer 1: exact-prefix (FR-06.4) ---------------------------------

    def _lookup_prefix(self, messages: list[Message]) -> Optional[str]:
        key = _prefix_key(messages)
        hit = self._prefix_cache.get(key)
        if hit is None:
            return None
        response, expires_at = hit
        if expires_at is not None and expires_at <= time.time():
            del self._prefix_cache[key]
            return None
        return response

    # ---- Layer 2: fuzzy semantic (FR-06.1-06.3) --------------------------

    def _lookup_semantic(self, messages: list[Message]) -> Optional[str]:
        query = messages[-1]["content"] if messages else ""
        if not query or not self._ids:
            return None

        if self._matrix is None:
            dim = len(self._embedding_blobs[0]) // 8  # float64 = 8 bytes each
            blob = b"".join(self._embedding_blobs)
            self._matrix = np.frombuffer(blob, dtype=np.float64).reshape(len(self._ids), dim)

        query_vec = np.array(self._embedder.embed(query), dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0.0:
            return None

        now = time.time()
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0.0] = 1e-12
        sims = (self._matrix @ query_vec) / (norms * query_norm)

        expired = np.array(
            [self._expires_at[i] is not None and self._expires_at[i] <= now for i in self._ids]
        )
        sims = np.where(expired, -1.0, sims)

        # Best match first; a candidate the term-swap guard rejects falls
        # through to the next one above the threshold.
        order = np.argsort(-sims)
        for idx in order:
            if float(sims[idx]) < self._config.similarity_threshold:
                break
            row_id = self._ids[int(idx)]
            if term_swap_conflict(
                query,
                self._queries.get(row_id, ""),
                self._responses[row_id],
                self._config.min_shared_prefix,
                self._config.term_swap_guard,
            ):
                continue
            return self._serve(row_id)
        # Rewordings the threshold or the guard turned away ("which ocean is the biggest")
        # look like near-misses ("...the smallest") to rules; a model tells them apart.
        floor = self._config.judge_floor
        if self.judge is None or floor is None or not len(order) or float(sims[order[0]]) < floor:
            return None
        row_id = self._ids[int(order[0])]
        self.judged += 1
        p_yes = self.judge(JUDGE_PROMPT.format(cached=self._queries.get(row_id, ""), new=query))
        if p_yes is not None and p_yes >= self._config.judge_threshold:
            return self._serve(row_id)
        return None

    def _serve(self, row_id: int) -> str:
        self._hit_counts[row_id] += 1
        self._conn.execute(
            "UPDATE cache_entries SET hit_count = ? WHERE id = ?", (self._hit_counts[row_id], row_id)
        )
        self._conn.commit()
        return self._responses[row_id]
