"""
engine/execution/llamacpp_runner.py

In-process llama.cpp runner (via llama-cpp-python) -- the one runner that
can do real speculative decoding, so the one that makes FR-03's layers
real rather than labels:

    ACCEL_PROMPT_LOOKUP  Layer 1: n-gram drafts copied from the context
                         (no extra model).
    ACCEL_DRAFT_MODEL    Layer 3: a small model of the same family drafts
                         `draft_tokens` tokens per step (e.g. llama3.2:1b
                         drafting for llama3.2:3b -- same tokenizer).

Both are lossless by construction: the target model samples every token
itself and a draft token is only kept when it equals that sample. One
caveat, measured rather than assumed (verify_greedy below): checking drafts
evaluates several tokens in one batch, and llama.cpp's batched arithmetic
rounds slightly differently from one-token-at-a-time decoding. Where the
model's top two choices are within that rounding noise, greedy speculative
output can pick the other one. verify_greedy measures the noise for the
exact sequence and checks every output token against it.

Every generation starts from a clean model state. llama-cpp-python
otherwise reuses the KV cache of the previous request's matching prefix
(every chat prompt shares the template header), which made the same prompt
give different output depending on what was asked before it.

Weights come from Ollama's own model store (resolved through Ollama's
/api/show), so nothing is downloaded twice. Optional dependency:
`pip install llama-cpp-python`; importing this module without it is fine,
constructing a runner raises a clear error.
"""

from __future__ import annotations

import atexit
import codecs
import ctypes
import hashlib
import inspect
import json
import os
import platform
import sys
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import psutil

from engine.execution.model_runner import (
    ACCEL_DRAFT_MODEL,
    ACCEL_NONE,
    ACCEL_PROMPT_LOOKUP,
    GenerationSample,
    OllamaReferenceModelRunner,
)

try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    from llama_cpp.llama_speculative import LlamaDraftModel, LlamaPromptLookupDecoding
except ImportError:  # pragma: no cover - exercised only without the optional dep
    llama_cpp = None
    Llama = None
    Jinja2ChatFormatter = None
    LlamaDraftModel = object
    LlamaPromptLookupDecoding = None


def _chat_sampling_defaults() -> dict:
    """llama-cpp-python's own create_chat_completion defaults, read from its
    signature so this runner samples exactly as that API would."""
    if Llama is None:
        return {}
    params = inspect.signature(Llama.create_chat_completion).parameters
    names = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
             "typical_p": "typical_p", "repeat_penalty": "repeat_penalty"}
    return {ours: params[theirs].default for theirs, ours in names.items() if theirs in params}


@dataclass(frozen=True)
class GreedyCheck:
    """Result of LlamaCppRunner.verify_greedy for one output."""

    tokens: int
    exact: int            # tokens that were the model's top choice, recomputed one at a time
    within_noise: int     # not the top choice, but within the measured rounding noise of it
    violations: int       # outside the noise: a real error in the decoding
    noise_tolerance: float  # largest move of a decision-relevant logit gap between one-at-a-time and batched evaluation

    @property
    def lossless(self) -> bool:
        return self.violations == 0

DEFAULT_DRAFT_MODEL_TAG = "llama3.2:1b-instruct-q4_K_M"

# (model_path, needs_all_logits) -> Llama-like object. Injectable for tests.
LlamaFactory = Callable[[str, bool], Any]


class DraftCounter:
    """
    Wraps any draft callable and counts calls and drafted tokens, so the
    runner can report a measured acceptance rate. llama-cpp-python yields
    (accepted drafts + 1) tokens per verification round, and runs
    (draft calls + 1) rounds, hence accepted = generated - calls - 1.
    """

    def __init__(self, draft: Callable[..., Any]) -> None:
        self._draft = draft
        self.calls = 0
        self.drafted = 0

    def __call__(self, input_ids, /, **kwargs):
        out = self._draft(input_ids, **kwargs)
        self.calls += 1
        self.drafted += len(out)
        return out

    def reset(self) -> None:
        self.calls = 0
        self.drafted = 0

    def acceptance_rate(self, tokens_generated: int) -> Optional[float]:
        if self.drafted == 0:
            return None
        accepted = max(0, tokens_generated - self.calls - 1)
        return min(1.0, accepted / self.drafted)


LOOKUP_MIN_MATCH = 3     # shortest repeated n-gram whose continuation is drafted
LOOKUP_MAX_MATCH = 8
LOOKUP_PRIOR = (5.0, 10.0)  # (accepted, judged) assumed before any lookup draft is judged
LOOKUP_HISTORY_TOKENS = 8192  # earlier answers kept for lookup across requests

KV_SLOT_TOKENS = 2048    # saved KV cache per model, in tokens (7B: ~0.11 GB, 0.5B: ~0.03 GB)
KV_SLOT_MIN_TOKENS = 32   # caches shorter than this are not worth saving


def _kv_lcp(cached: Sequence[int], ids: Sequence[int]) -> int:
    m = min(len(cached), len(ids))
    if not m:
        return 0
    same = np.asarray(cached[:m]) == np.asarray(ids[:m])
    return m if same.all() else int(np.argmin(same))


def _kv_get(llm: Any) -> bytes:
    """Sequence 0's KV cache as bytes (llama.cpp's own serialisation)."""
    ctx = llm._ctx.ctx
    size = llama_cpp.llama_state_seq_get_size(ctx, 0)
    buf = (ctypes.c_uint8 * int(size))()
    got = llama_cpp.llama_state_seq_get_data(ctx, buf, size, 0)
    return bytes(memoryview(buf)[: int(got)])


def _kv_set(llm: Any, data: bytes) -> None:
    llm._ctx.kv_cache_seq_rm(-1, 0, -1)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    if llama_cpp.llama_state_seq_set_data(llm._ctx.ctx, buf, len(data), 0) != len(data):
        raise RuntimeError("llama.cpp refused the saved KV cache")

KV_DISK_PATH = Path.home() / ".cascade" / "kv"
KV_DISK_BYTES = 512 * 1024**2  # all models together; oldest files go first
_KV_MAGIC = b"CKV1"


class KVDisk:
    """
    KV caches on disk, one file per cache: magic, token count, the tokens (int32), then
    llama.cpp's serialised state. Survives the model being unloaded (idle unload) and
    the process exiting (`cascade ask`, a restart, a reboot): restoring a 1,000-token
    7B cache reads ~57 MB, re-reading the prompt takes ~55 s on a 4-core CPU.
    """

    def __init__(self, directory: Path, max_bytes: int = KV_DISK_BYTES, root: Optional[Path] = None) -> None:
        self.directory = Path(directory)
        self.max_bytes = max_bytes
        self.root = Path(root) if root is not None else self.directory.parent
        self._index: Optional[dict[tuple[int, ...], Path]] = None

    def entries(self) -> dict[tuple[int, ...], Path]:
        if self._index is None:
            self._index = {}
            for path in sorted(self.directory.glob("*.kv")):
                try:
                    with path.open("rb") as f:
                        if f.read(4) != _KV_MAGIC:
                            continue
                        n = int.from_bytes(f.read(4), "little")
                        tokens = np.frombuffer(f.read(4 * n), dtype=np.int32)
                    if len(tokens) == n:
                        self._index[tuple(int(t) for t in tokens)] = path
                except OSError:
                    continue
        return self._index

    def put(self, tokens: tuple[int, ...], data: bytes) -> None:
        entries = self.entries()
        if tokens in entries:
            return
        # A cache that extends a saved one replaces it.
        for key in [k for k in entries if len(k) < len(tokens) and tokens[: len(k)] == k]:
            entries.pop(key).unlink(missing_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        array = np.asarray(tokens, dtype=np.int32)
        path = self.directory / (hashlib.sha1(array.tobytes()).hexdigest()[:20] + ".kv")
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            f.write(_KV_MAGIC + len(tokens).to_bytes(4, "little"))
            f.write(array.tobytes())
            f.write(data)
        os.replace(tmp, path)
        entries[tokens] = path
        self._evict()

    def take(self, tokens: tuple[int, ...]) -> Optional[bytes]:
        path = self.entries().get(tokens)
        if path is None:
            return None
        try:
            raw = path.read_bytes()
            os.utime(path)  # most recently used
        except OSError:
            self.entries().pop(tokens, None)
            return None
        return raw[8 + 4 * len(tokens):]

    def drop(self, tokens: tuple[int, ...]) -> None:
        path = self.entries().pop(tokens, None)
        if path is not None:
            path.unlink(missing_ok=True)

    def _evict(self) -> None:
        files = sorted(self.root.glob("*/*.kv"), key=lambda q: q.stat().st_mtime)
        total = sum(q.stat().st_size for q in files)
        for path in files:
            if total <= self.max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            if self._index is not None:
                for key, value in list(self._index.items()):
                    if value == path:
                        del self._index[key]


def _kv_disk_for(llm: Any) -> Optional[KVDisk]:
    """The disk store for this model file and context settings (the KV layout)."""
    path = getattr(llm, "model_path", None)
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    params = getattr(llm, "context_params", None)
    layout = [path, os.path.getsize(path)] + [
        str(getattr(params, name, None)) for name in ("n_ctx", "flash_attn_type", "type_k", "type_v")
    ]
    return KVDisk(KV_DISK_PATH / hashlib.sha1("|".join(map(str, layout)).encode()).hexdigest()[:16])


class KVSlots:
    """
    Saved KV caches of one model, most recently used last, keyed by the tokens they hold.

    llama-cpp-python keeps one KV cache, so a request that does not continue the last
    one throws the computed prefix away: two alternating conversations, or the small
    model answering an easy question on the instance that also drafts, each recompute
    the whole history. `switch(llm, ids)` runs before a model evaluates `ids`: when a
    saved cache shares a longer prefix with `ids` than the live one, the live one is
    saved and the better one restored; a long live cache about to be discarded is saved
    first. At most `max_tokens` tokens are kept, oldest dropped first.
    """

    def __init__(self, max_tokens: int = KV_SLOT_TOKENS, min_tokens: int = KV_SLOT_MIN_TOKENS,
                 disk: Optional[KVDisk] = None) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.disk = disk
        self._slots: OrderedDict[tuple[int, ...], bytes] = OrderedDict()
        self.restores = 0  # for inspection

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def tokens(self) -> int:
        return sum(len(k) for k in self._slots)

    def save(self, llm: Any) -> None:
        n = int(llm.n_tokens)
        if n < self.min_tokens or n > self.max_tokens:
            return
        key = tuple(int(t) for t in llm.input_ids[:n])
        if key not in self._slots:
            self._slots[key] = _kv_get(llm)
        self._slots.move_to_end(key)
        while self.tokens > self.max_tokens:
            self._slots.popitem(last=False)

    def switch(self, llm: Any, ids: Sequence[int]) -> None:
        live = int(llm.n_tokens)
        live_lcp = _kv_lcp(llm.input_ids[:live], ids)
        best, best_lcp, on_disk = None, live_lcp, False
        for key in self._slots:
            lcp = _kv_lcp(key, ids)
            if lcp > best_lcp:
                best, best_lcp = key, lcp
        for key in (self.disk.entries() if self.disk is not None else ()):
            lcp = _kv_lcp(key, ids)
            if lcp > best_lcp:
                best, best_lcp, on_disk = key, lcp, True
        if best is not None and best_lcp - live_lcp >= self.min_tokens:
            self.save(llm)
            data = self.disk.take(best) if on_disk else self._slots.pop(best)
            try:
                if data is None:
                    raise RuntimeError("saved KV cache unreadable")
                _kv_set(llm, data)
            except RuntimeError:  # a stale or foreign file: forget it, start clean
                if on_disk:
                    self.disk.drop(best)
                llm.n_tokens = 0
                return
            llm.input_ids[: len(best)] = best
            llm.n_tokens = len(best)
            llm._requires_eval = True
            self.restores += 1
        elif live - live_lcp >= self.min_tokens:
            self.save(llm)

    def spill(self, llm: Any) -> None:
        """Write the live cache and every saved one to disk (before an unload or exit)."""
        if self.disk is None:
            return
        n = int(llm.n_tokens)
        if self.min_tokens <= n:
            self.disk.put(tuple(int(t) for t in llm.input_ids[:n]), _kv_get(llm))
        for key, data in self._slots.items():
            self.disk.put(key, data)


def kv_slots_for(llm: Any, max_tokens: int = KV_SLOT_TOKENS) -> Optional[KVSlots]:
    """The KVSlots shared by every user of this model instance, or None for a model
    without a llama.cpp context (test doubles) or with saving off (max_tokens 0)."""
    if max_tokens <= 0 or llama_cpp is None or getattr(llm, "_ctx", None) is None:
        return None
    if not hasattr(llm, "n_tokens") or not hasattr(llm, "input_ids"):
        return None
    slots = getattr(llm, "_cascade_kv_slots", None)
    if slots is None:
        slots = KVSlots(max_tokens, disk=_kv_disk_for(llm))
        llm._cascade_kv_slots = slots
        atexit.register(spill_kv, weakref.ref(llm))
    return slots


def spill_kv(llm: Any) -> None:
    """Write a model's KV caches to disk now: it is about to be dropped, or the process
    is exiting. Takes the model or a weak reference to it; never raises."""
    if isinstance(llm, weakref.ref):
        llm = llm()
    slots = getattr(llm, "_cascade_kv_slots", None) if llm is not None else None
    if slots is not None:
        try:
            slots.spill(llm)
        except Exception:  # never fail an unload or an exit over a cache
            pass


class ModelDraft(LlamaDraftModel):
    """
    Layer 3 draft: greedy `k` tokens from a small model, reusing its KV cache
    across calls.

    With `adaptive=True` the draft length follows the acceptance it observes,
    using the schedule Hugging Face Transformers applies to assisted
    generation ("heuristic"): every draft fully accepted -> k += 2, otherwise
    k -= 1 (never below 1, never above `max_k`). Acceptance is read from how
    far the sequence advanced since the previous call: the target keeps the
    accepted drafts plus one token of its own. On copy-heavy text drafts grow
    long; on open-ended text they shrink, so rejected drafts waste less work.

    Even a 1-token draft costs extra on a CPU: the draft model's step, plus
    verifying 2 tokens instead of 1 (BENCHMARK_REPORT Addendum 7). So when a
    1-token draft is rejected, drafting pauses for 1, 2, 4, ... calls (up to
    `max_pause`; 0 disables) and those steps run as plain decoding. An
    empty draft gives the target exactly its plain decoding step. The first
    fully accepted draft clears the backoff.

    With `p_min` set, drafting instead stops at the first token the draft
    model is less than `p_min` sure of (its top softmax probability), as
    llama.cpp's own speculative decoding does (`--draft-p-min`, default
    0.75). `k` is then the longest draft. A step with no confident token
    drafts nothing and costs what plain decoding costs.

    With `verify_costs` set (cost-aware drafting), the draft length is chosen by
    what verifying it costs the target. `verify_costs[n]` is the cost of one
    target eval of n tokens relative to one token; on a CPU it is far from flat
    and not even monotonic (7B Q4_K_M, 4 cores: 4 tokens cost 1.55x one token
    but 3 tokens 1.74x; 16 cost 3.95x, 15 cost 4.6x -- the weight kernels work
    on groups of 4). The draft model drafts while the chance that its whole
    draft is accepted (the product of its top probabilities) can still pay
    for a longer batch, then keeps the length L that maximises the expected
    tokens per unit of cost, (1 + E[accepted]) / (verify_costs[L + 1] +
    draft_cost * L), where E[accepted] sums those running products; L = 0
    (plain decoding) when nothing beats 1 / verify_costs[1].

    Lookup drafting (`lookup`, cost-aware only) comes first: when the last 3-8
    tokens already appeared earlier in the prompt or output -- code being
    edited, text being rewritten, an earlier answer being revised -- the tokens
    that followed them are the draft, at no draft-model cost (prompt lookup,
    as in SuffixDecoding and llama.cpp's ngram drafts). How far to trust a copy
    is learned per match length from how far the sequence advanced; when no
    copy pays, the draft model drafts as above. When the current text has no
    such repeat, the answers to earlier requests (the last LOOKUP_HISTORY_TOKENS
    tokens of them) are searched the same way, with their own learned acceptance.
    """

    def __init__(self, llm: Any, k: int = 4, adaptive: bool = False, max_k: int = 16, max_pause: int = 0,
                 p_min: Optional[float] = None, verify_costs: Optional[Sequence[float]] = None,
                 draft_cost: float = 0.1, lookup: bool = True, lookup_history: bool = True) -> None:
        self._llm = llm
        self._initial_k = k
        self._k = k
        self._adaptive = adaptive
        self._max_k = max_k
        self._last_len: Optional[int] = None
        self._last_drafted = 0
        self._max_pause = max_pause
        self._p_min = p_min
        self._verify_costs = list(verify_costs) if verify_costs is not None else None
        if self._verify_costs is not None and len(self._verify_costs) < k + 2:
            raise ValueError(f"verify_costs needs an entry for every batch size up to {k + 1}")
        self._draft_cost = draft_cost
        # Lookup drafting (cost-aware only): when the text just written already appeared
        # earlier in the prompt or output, its continuation is the draft, at no draft-model
        # cost. Per-token acceptance is learned per match length.
        self._lookup = lookup
        self._lookup_history = lookup_history
        self.lookup_stats = {n: list(LOOKUP_PRIOR) for n in range(LOOKUP_MIN_MATCH, LOOKUP_MAX_MATCH + 1)}
        self.history_stats = {n: list(LOOKUP_PRIOR) for n in range(LOOKUP_MIN_MATCH, LOOKUP_MAX_MATCH + 1)}
        self._judge: Optional[tuple[dict, int, int, int]] = None  # (stats, match length, drafted, sequence length)
        self._past = np.array([], dtype=np.int64)  # earlier answers, each followed by -1
        self._request: Optional[tuple[int, list[int]]] = None  # (prompt length, latest ids) of this request
        self.lookup_drafts = 0  # calls answered by lookup, for inspection
        self._pause = 0      # calls left to skip drafting
        self._backoff = 0    # length of the next pause
        self.history: list[int] = []  # draft length used per call, for inspection

    @property
    def k(self) -> int:
        return self._k

    def _adapt(self, length: int) -> None:
        if self._last_len is None or length <= self._last_len:  # first call, or a new request
            self._k = self._initial_k
            self._pause = self._backoff = 0
        elif self._last_drafted:
            accepted = length - self._last_len - 1
            if accepted >= self._last_drafted:
                self._k = min(self._max_k, self._k + 2)
                self._backoff = 0
            elif self._k == 1 and self._max_pause > 0:
                self._backoff = min(self._max_pause, max(1, 2 * self._backoff))
                self._pause = self._backoff
            else:
                self._k = max(1, self._k - 1)
        self._last_len = length

    def __call__(self, input_ids, /, **kwargs):
        if self._adaptive:
            self._adapt(len(input_ids))
            if self._pause > 0:
                self._pause -= 1
                self._last_drafted = 0
                self.history.append(0)
                return np.array([], dtype=np.intc)
        self.history.append(self._k)

        if self._verify_costs is not None:
            ids = list(map(int, input_ids))
            self._judge_lookup(len(ids))
            self._track_request(ids)
            out = self._lookup_draft(ids) if self._lookup else []
            if out:
                self.lookup_drafts += 1
            else:
                out = self._cost_aware_draft(ids)
            self._last_drafted = len(out)
            return np.array(out, dtype=np.intc)
        if self._p_min is not None:
            out = self._confident_draft(list(map(int, input_ids)))
            self._last_drafted = len(out)
            return np.array(out, dtype=np.intc)

        out: list[int] = []
        gen = self._llm.generate(list(map(int, input_ids)), top_k=1, temp=0.0, reset=True)
        try:
            for token in gen:
                out.append(int(token))
                if len(out) >= self._k:
                    break
        finally:
            gen.close()
        self._last_drafted = len(out)
        return np.array(out, dtype=np.intc)

    def _sync(self, ids: list[int]) -> None:
        """Bring the draft's KV cache to `ids`, reusing the prefix it has already seen."""
        llm = self._llm
        slots = kv_slots_for(llm)
        if slots is not None:  # the small model may have answered on this instance meanwhile
            slots.switch(llm, ids)
        n = _kv_lcp(llm.input_ids[: llm.n_tokens], ids)
        n = min(n, len(ids) - 1)  # always evaluate at least the last token, for its logits
        llm.n_tokens = n
        llm.eval(ids[n:])

    def _judge_lookup(self, length: int) -> None:
        """The target kept (sequence advance - 1) of the last lookup draft's tokens."""
        if self._judge is None:
            return
        table, n, drafted, at = self._judge
        self._judge = None
        if length <= at:  # a new request
            return
        accepted = min(drafted, length - at - 1)
        stats = table[n]
        stats[0] += accepted
        stats[1] += accepted + (accepted < drafted)  # tokens after the first rejection: unknown

    def _track_request(self, ids: list[int]) -> None:
        """On a new request, keep the previous request's answer for lookup."""
        if self._request is not None:
            prompt_len, last = self._request
            if len(ids) > len(last) and ids[:len(last)] == last:
                self._request = (prompt_len, ids)
                return
            answer = last[prompt_len:]
            if self._lookup_history and len(answer) > LOOKUP_MIN_MATCH:
                past = np.concatenate([self._past, np.asarray(answer, dtype=np.int64), [-1]])
                self._past = past[-LOOKUP_HISTORY_TOKENS:]
        self._request = (len(ids), ids)

    @staticmethod
    def _latest_match(hay: np.ndarray, suffix: np.ndarray, windows: int) -> Optional[int]:
        """Index just after the latest occurrence of `suffix` among the first `windows`
        start positions of `hay`, or None."""
        n = len(suffix)
        if windows <= 0:
            return None
        hits = np.ones(windows, dtype=bool)
        for j in range(n):
            hits &= hay[j: windows + j] == suffix[j]
        starts = np.flatnonzero(hits)
        return int(starts[-1]) + n if len(starts) else None

    def _lookup_draft(self, ids: list[int]) -> list[int]:
        """The continuation of the most recent earlier occurrence of the longest repeated
        suffix (LOOKUP_MIN_MATCH..LOOKUP_MAX_MATCH tokens) -- in this request, else in
        earlier answers -- cut to the cost-aware length for its learned per-token
        acceptance; [] when no match pays."""
        arr = np.asarray(ids, dtype=np.int64)
        size = len(arr)
        top = min(LOOKUP_MAX_MATCH, size - 1)
        for n in range(top, LOOKUP_MIN_MATCH - 1, -1):
            # an earlier occurrence, not the suffix itself
            start = self._latest_match(arr, arr[size - n:], size - n)
            if start is not None and start < size:
                return self._cut(self.lookup_stats, n, ids[start: start + self._k], size)
        past = self._past
        for n in range(min(top, LOOKUP_MAX_MATCH), LOOKUP_MIN_MATCH - 1, -1):
            start = self._latest_match(past, arr[size - n:], len(past) - n)
            if start is None:
                continue
            continuation = past[start: start + self._k]
            stop = np.flatnonzero(continuation < 0)
            continuation = continuation[: stop[0]] if len(stop) else continuation
            if len(continuation):
                return self._cut(self.history_stats, n, [int(t) for t in continuation], size)
        return []

    def _cut(self, table: dict, n: int, continuation: list[int], size: int) -> list[int]:
        """The prefix of `continuation` that maximises expected tokens per unit of
        verify cost at the per-token acceptance learned for match length n."""
        accepted, judged = table[n]
        rho = accepted / judged
        costs = self._verify_costs
        best_len, best_rate, whole, expected = 0, 1.0 / costs[1], 1.0, 0.0
        for length in range(1, len(continuation) + 1):
            whole *= rho
            expected += whole
            rate = (1.0 + expected) / costs[length + 1]
            if rate > best_rate:
                best_len, best_rate = length, rate
        if not best_len:
            return []
        self._judge = (table, n, best_len, size)
        return continuation[:best_len]

    def _cost_aware_draft(self, ids: list[int]) -> list[int]:
        costs, d = self._verify_costs, self._draft_cost
        self._sync(ids)
        out: list[int] = []
        best_len, best_rate = 0, 1.0 / costs[1]
        whole, expected = 1.0, 0.0  # P(every drafted token accepted), E[accepted]
        while len(out) < self._k:
            logits = self._last_logits()
            token = int(np.argmax(logits))
            whole *= 1.0 / float(np.sum(np.exp(logits - logits[token])))
            out.append(token)
            expected += whole
            n = len(out)
            rate = (1.0 + expected) / (costs[n + 1] + d * n)
            if rate > best_rate:
                best_len, best_rate = n, rate
            # Stop once no longer draft could win, even if every further token were
            # accepted with the current whole-draft probability.
            if not any((1.0 + expected + whole * (m - n)) / (costs[m + 1] + d * m) > best_rate
                       for m in range(n + 1, self._k + 1)):
                break
            if n < self._k:
                self._llm.eval([token])
        return out[:best_len]

    def _confident_draft(self, ids: list[int]) -> list[int]:
        llm = self._llm
        self._sync(ids)
        out: list[int] = []
        while len(out) < self._k:
            logits = self._last_logits()
            token = int(np.argmax(logits))
            top_p = 1.0 / float(np.sum(np.exp(logits - logits[token])))
            if top_p < self._p_min:
                break
            out.append(token)
            if len(out) < self._k:
                llm.eval([token])
        return out

    def _last_logits(self) -> np.ndarray:
        llm = self._llm
        return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)


# Relative cost of one target eval of n tokens (index n; index 0 unused), measured
# for qwen2.5-coder 7B Q4_K_M on a 4-core CPU (scripts/benchmark_verify_cost.py,
# docs/benchmark/verify_cost_7b.json); the fallback when "auto" cannot measure.
# Multiples of 4 are cheap: the repacked Q4_K kernels work on 4 rows at a time.
CPU_VERIFY_COSTS = (0.0, 1.0, 1.36, 1.74, 1.55, 1.87, 2.28, 2.68, 3.0, 3.34, 3.49, 3.74, 3.57, 3.9, 4.35, 4.6,
                    3.95, 5.1)

VERIFY_COSTS_PATH = Path.home() / ".cascade" / "verify_costs.json"
MODEL_PATHS_PATH = Path.home() / ".cascade" / "model_paths.json"  # Ollama tag -> GGUF file
AUTO = "auto"


def _verify_cost_key(llm: Any, path: str, n: int) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    threads = getattr(getattr(llm, "context_params", None), "n_threads", None)
    version = getattr(sys.modules.get("llama_cpp"), "__version__", "?")
    return "|".join(map(str, ("v2", os.path.basename(path), size, threads, platform.processor() or platform.machine(),
                              os.cpu_count(), version, n)))


def measure_verify_costs(llm: Any, n: int, repeats: int = 3) -> list[float]:
    """Time one eval of 1..n tokens after a 256-token prefix, about a chat prompt (a shorter
    one makes attention, and so verifying, look cheaper); KV cache rewound between runs,
    and return each relative to one token, index 0 unused. Leaves the model's context empty."""
    prefix = llm.tokenize(b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n" * 16)[:256]
    filler = llm.tokenize(b" while i < len(a) and j < len(b): out.append(a[i]); i += 1", add_bos=False)
    filler = (filler * (n // max(len(filler), 1) + 1))[:n]
    try:
        llm.n_tokens = 0
        llm._ctx.kv_cache_clear()
        llm.eval(prefix)
        base = llm.n_tokens
        llm.eval(filler[:4])  # warm the batch path
        ms = []
        for size in range(1, n + 1):
            times = []
            for _ in range(repeats):
                llm.n_tokens = base
                llm._ctx.kv_cache_seq_rm(-1, base, -1)
                t0 = time.perf_counter()
                llm.eval(filler[:size])
                times.append(time.perf_counter() - t0)
            ms.append(sorted(times)[len(times) // 2])
    finally:
        llm.n_tokens = 0
        llm._ctx.kv_cache_clear()
    return [0.0] + [round(t / ms[0], 3) for t in ms]


def verify_costs_for(llm: Any, path: Optional[str], n: int, cache_path: Optional[Path] = None) -> list[float]:
    """This model's verify-cost table on this machine: measured once (about half a
    minute for a 7B on a 4-core CPU) and cached; CPU_VERIFY_COSTS if it cannot be measured."""
    cache_path = cache_path or VERIFY_COSTS_PATH
    fallback = list(CPU_VERIFY_COSTS[: n + 1]) + [CPU_VERIFY_COSTS[-1] * i / 17
                                                    for i in range(len(CPU_VERIFY_COSTS), n + 1)]
    if not path or not hasattr(llm, "_ctx"):
        return fallback
    key = _verify_cost_key(llm, path, n)
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cached = {}
    if isinstance(cached.get(key), list) and len(cached[key]) == n + 1:
        return cached[key]
    print(f"cascade: measuring what verifying drafts costs {os.path.basename(path)} on this machine "
          "(once, about half a minute)...", file=sys.stderr, flush=True)
    try:
        costs = measure_verify_costs(llm, n)
    except Exception:  # noqa: BLE001 -- any failure: use the reference table
        return fallback
    cached[key] = costs
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=1))
    except OSError:
        pass
    return costs


# Positions whose logits a speculative eval keeps: the verification batch is the
# last sampled token plus the draft, so this must exceed every draft length.
OUTPUT_WINDOW = 64


def _install_windowed_eval(llm: Any, window: int = OUTPUT_WINDOW) -> None:
    """
    Make a speculative-mode Llama compute logits only for the last `window`
    positions of each eval, and stop copying them into `llm.scores`.

    llama-cpp-python with a draft model asks llama.cpp for logits at *every*
    position, including every prompt token, and copies them into a Python
    buffer of only n_batch (512) rows. That cost an extra 152k-wide output
    projection per prompt token, an output buffer of n_batch x vocab floats in
    llama.cpp, and crashed outright once prompt + output passed 512 tokens.
    Sampling reads logits from the llama.cpp context (negative indices count
    from the last output), so verifying a draft needs only the last
    len(draft) + 1 of them. `llm.keep_scores = True` restores the stock eval,
    with a full-context `scores` buffer, for verify_greedy.
    """
    # No strong reference from the closure back to llm: a llm -> eval -> llm
    # cycle kept every replaced 4-5 GB model alive until the garbage collector
    # happened to run (RSS 9.4 GB after a reload instead of 5.4 GB).
    stock_eval = type(llm).eval
    ref = weakref.ref(llm)
    llm.keep_scores = False

    def eval(tokens) -> None:
        llm = ref()
        if llm.keep_scores:
            if llm.scores.shape[0] < llm._n_ctx:
                llm.scores = np.zeros((llm._n_ctx, llm._n_vocab), dtype=np.single)  # pages touched lazily
            return stock_eval(llm, tokens)
        tokens = list(tokens)
        first_output = len(tokens) - window
        llm._ctx.kv_cache_seq_rm(-1, llm.n_tokens, -1)
        for i in range(0, len(tokens), llm.n_batch):
            batch = tokens[i:i + llm.n_batch]
            n_past = llm.n_tokens
            llm._batch.set_batch(batch=batch, n_past=n_past, logits_all=False)  # last position always on
            for j in range(max(0, first_output - i), len(batch)):
                llm._batch.batch.logits[j] = True
            llm._ctx.decode(llm._batch)
            llm.input_ids[n_past:n_past + len(batch)] = batch
            llm.n_tokens += len(batch)
            llm._requires_eval = False

    llm.eval = eval


# Plain (one-logit) models loaded in this process, by file and settings. A model
# loaded as the small backend for easy queries and the same file used as the big
# model's draft are then one instance in RAM, not two. Safe because every caller
# re-syncs the KV cache with its own tokens before use (generate() resets;
# ModelDraft compares the cached prefix), and the Pipeline runs one request at a
# time. Weak references: the pool never keeps a model alive by itself.
_SHARED_MODELS: "weakref.WeakValueDictionary[tuple, Any]" = weakref.WeakValueDictionary()


WORKING_MEMORY_SETTINGS = {"flash_attn": True, "n_ubatch": 256}
if os.environ.get("CASCADE_LLAMA_SETTINGS"):  # JSON override, for measuring alternatives
    import json as _json

    WORKING_MEMORY_SETTINGS = _json.loads(os.environ["CASCADE_LLAMA_SETTINGS"])


def _default_llama_factory(n_ctx: int, n_threads: Optional[int], use_mmap: bool = False) -> LlamaFactory:
    def make(path: str, needs_all_logits: bool) -> Any:
        key = (os.path.realpath(path), n_ctx, n_threads, use_mmap)
        if not needs_all_logits:
            shared = _SHARED_MODELS.get(key)
            if shared is not None:
                return shared
        # A placeholder draft at construction is the only way to make
        # llama-cpp-python keep per-position logits, which verifying
        # drafts needs. The real draft is swapped in afterwards.
        placeholder = LlamaPromptLookupDecoding() if needs_all_logits else None
        # use_mmap=False: llama.cpp repacks the weights into a private
        # CPU-friendly layout, so with the file also memory-mapped the
        # weights sat in RAM twice (qwen2.5-coder 7B Q4_K_M: 7.2 GB resident
        # with mmap, 4.6 GB without -- BENCHMARK_REPORT Addendum 7).
        # flash_attn + n_ubatch 256: smaller compute buffer, same output, no slowdown
        # (7B: -0.15 GB committed at n_ctx 2048, -0.40 GB at 8192; prompt and decode speed
        # within run-to-run noise; identical greedy tokens -- BENCHMARK_REPORT Addendum 15).
        # The KV cache stays f16: q8_0 saves another ~0.2 GB but costs 12-44% prompt speed.
        llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, use_mmap=use_mmap,
                    verbose=False, draft_model=placeholder, **WORKING_MEMORY_SETTINGS)
        # llama-cpp-python allocates a float32 logits buffer of n_ctx x n_vocab rows when
        # all logits are kept (qwen2.5 7B: 2048 x 152k = 1.24 GB) and n_batch rows
        # otherwise (0.31 GB). Neither the windowed eval nor plain decoding writes it
        # (plain eval stopped filling it in 0.3); verify_greedy re-creates the large one
        # only while it runs. Committed memory, not just resident: drop it.
        if hasattr(llm, "n_vocab"):  # (test doubles have no buffer)
            llm.scores = np.ndarray((1, llm.n_vocab()), dtype=np.single)
        if needs_all_logits:
            _install_windowed_eval(llm)
        else:
            try:
                _SHARED_MODELS[key] = llm
            except TypeError:  # not weak-referenceable (a test double): just don't share
                pass
        return llm
    return make


class LlamaCppRunner:
    """ReferenceModelRunner + AcceleratedRunner over llama.cpp."""

    def __init__(
        self,
        ollama: Optional[OllamaReferenceModelRunner] = None,
        draft_model_tag: Optional[str] = DEFAULT_DRAFT_MODEL_TAG,
        draft_tokens: int = 4,
        adaptive_draft: bool = False,
        max_draft_tokens: int = 16,
        max_draft_pause: int = 0,
        draft_p_min: Optional[float] = None,
        draft_verify_costs: Optional[Sequence[float] | str] = None,
        draft_lookup: bool = True,
        draft_lookup_history: bool = True,
        lookup_tokens: int = 10,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = False,
        llama_factory: Optional[LlamaFactory] = None,
        temperature: Optional[float] = None,
        reuse_prefix: bool = True,
        native_chat: bool = True,
        kv_slot_tokens: int = KV_SLOT_TOKENS,
    ) -> None:
        self._kv_slot_tokens = kv_slot_tokens
        self._temperature = temperature
        self._reuse_prefix = reuse_prefix
        # With native_chat, LocalExecutionBackend hands over the message list and each
        # message keeps its own role in the model's chat template. Consecutive turns then
        # share the earlier answers as prompt prefix too, which reuse_prefix skips.
        self.accepts_messages = native_chat
        self.last_prompt_s: Optional[float] = None  # time to the first token of the last generate()
        if llama_factory is None and Llama is None:
            raise RuntimeError("LlamaCppRunner needs llama-cpp-python: pip install llama-cpp-python")
        self._ollama = ollama or OllamaReferenceModelRunner()
        self._draft_model_tag = draft_model_tag
        self._draft_tokens = draft_tokens
        self._adaptive_draft = adaptive_draft
        self._max_draft_tokens = max_draft_tokens
        longest = max(draft_tokens, max_draft_tokens, lookup_tokens)
        if longest >= OUTPUT_WINDOW:
            raise ValueError(f"draft length {longest} must be below {OUTPUT_WINDOW}")
        self._max_draft_pause = max_draft_pause
        self._draft_p_min = draft_p_min
        self._draft_verify_costs = draft_verify_costs
        self._resolved_verify_costs: Optional[list[float]] = None
        self._draft_lookup = draft_lookup
        self._draft_lookup_history = draft_lookup_history
        self._lookup_tokens = lookup_tokens
        self._factory = llama_factory or _default_llama_factory(n_ctx, n_threads, use_mmap)
        self._mode = ACCEL_NONE
        self._llm: Any = None
        self._llm_has_all_logits = False
        self._model_path: Optional[str] = None
        self._draft_llm: Any = None
        self._counter: Optional[DraftCounter] = None
        self._sampling = _chat_sampling_defaults()
        if temperature is not None:
            self._sampling["temp"] = temperature
        # Token ids of the last generate() call, for verify_greedy().
        self.last_prompt_ids: list[int] = []
        self.last_output_ids: list[int] = []
        # Log-probability of each output token under the model, recorded in plain
        # mode only (speculative steps sample several tokens per eval). An
        # answer-quality signal for Pipeline's escalation check (FR-05.3).
        self.last_token_logprobs: Optional[list[float]] = None

    # ---- weights ---------------------------------------------------------

    def local_models(self) -> list[str]:
        """Tags already downloaded (Ollama's store holds the GGUFs this runner loads)."""
        return self._ollama.local_models()

    def _resolve_path(self, tag: str) -> str:
        # A GGUF found before loads without Ollama (it may be stopped, or restarting).
        try:
            known = json.loads(MODEL_PATHS_PATH.read_text())
        except (OSError, ValueError):
            known = {}
        if isinstance(known.get(tag), str) and os.path.isfile(known[tag]):
            return known[tag]
        path = self._resolve_path_from_ollama(tag)
        known[tag] = path
        try:
            MODEL_PATHS_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_PATHS_PATH.write_text(json.dumps(known, indent=1))
        except OSError:
            pass
        return path

    def _resolve_path_from_ollama(self, tag: str) -> str:
        if tag not in self._ollama.local_models():
            self._ollama._request("/api/pull", {"model": tag, "stream": False}, timeout=1800)
        path = self._ollama.model_file_path(tag)
        if not path:
            raise RuntimeError(f"Ollama has no GGUF file for {tag}")
        return path

    @property
    def self_regulating(self) -> bool:
        """True when confidence drafting is active: unsure drafts are skipped per token, so
        AccelerationEngine should not step down on a low acceptance rate."""
        return self._mode == ACCEL_DRAFT_MODEL and (self._draft_p_min is not None
                                                    or self._draft_verify_costs is not None)

    def _verify_costs(self) -> Optional[Sequence[float]]:
        """`draft_verify_costs="auto"`: this model's table on this machine, measured once."""
        if self._draft_verify_costs != AUTO:
            return self._draft_verify_costs
        if self._resolved_verify_costs is None:
            self._resolved_verify_costs = verify_costs_for(self._llm, self._model_path, self._draft_tokens + 1)
        return self._resolved_verify_costs

    def _build(self) -> None:
        needs_all_logits = self._mode != ACCEL_NONE
        self._llm = None  # release the old model before loading the new one: never two copies at once
        self._llm = self._factory(self._model_path, needs_all_logits)
        self._llm_has_all_logits = needs_all_logits
        self._resolved_verify_costs = None  # per model: re-read (or measure) its table
        self._install_draft()

    def _install_draft(self) -> None:
        if self._llm is None:
            return
        if self._mode == ACCEL_PROMPT_LOOKUP:
            self._counter = DraftCounter(LlamaPromptLookupDecoding(num_pred_tokens=self._lookup_tokens))
        elif self._mode == ACCEL_DRAFT_MODEL:
            if self._draft_llm is None:
                self._draft_llm = self._factory(self._resolve_path(self._draft_model_tag), False)
            self._counter = DraftCounter(ModelDraft(
                self._draft_llm, k=self._draft_tokens, adaptive=self._adaptive_draft, max_k=self._max_draft_tokens,
                max_pause=self._max_draft_pause, p_min=self._draft_p_min, verify_costs=self._verify_costs(),
                lookup=self._draft_lookup, lookup_history=self._draft_lookup_history,
            ))
        else:
            self._counter = None
        self._llm.draft_model = self._counter

    # ---- AcceleratedRunner ---------------------------------------------

    def set_acceleration(self, mode: str) -> bool:
        if mode not in (ACCEL_NONE, ACCEL_PROMPT_LOOKUP, ACCEL_DRAFT_MODEL):
            return False
        if mode == ACCEL_DRAFT_MODEL and not self._draft_model_tag:
            return False
        if mode == ACCEL_PROMPT_LOOKUP and LlamaPromptLookupDecoding is None:
            return False
        self._mode = mode
        if self._llm is not None:
            if mode != ACCEL_NONE and not self._llm_has_all_logits:
                self._build()
            else:
                self._install_draft()
        return True

    # ---- ReferenceModelRunner -------------------------------------------

    def load(self, model_id: str, quant_level: str) -> None:
        self._model_path = self._resolve_path(f"{model_id}:{quant_level}")
        self._build()

    streams = True  # generate() takes on_text

    def generate(self, prompt: str | list[dict], max_tokens: int,
                 on_text: Optional[Callable[[str], None]] = None) -> GenerationSample:
        """With `on_text`, each piece of the answer is passed to it as soon as it is decoded
        (leading whitespace dropped, as in the returned text), for streaming replies."""
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first.")
        if self._counter is not None:
            self._counter.reset()

        prompt_ids = self._chat_tokens(prompt)
        if not self._reuse_prefix:
            self._llm.reset()
        else:
            slots = kv_slots_for(self._llm, self._kv_slot_tokens)
            if slots is not None:
                slots.switch(self._llm, prompt_ids)
        # Otherwise llama-cpp-python keeps the KV cache of the longest prefix this prompt
        # shares with the previous request (system prompt, earlier turns) and evaluates only
        # the rest. Same arithmetic, so the output is the same up to the floating-point noise
        # verify_greedy() measures (where the prefill is split differently).
        prompt_start = time.perf_counter()
        out: list[int] = []
        logprobs: Optional[list[float]] = [] if self._counter is None else None
        first_t: Optional[float] = None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace") if on_text is not None else None
        started = False  # past the leading whitespace
        gen = self._llm.generate(prompt_ids, reset=True, **self._sampling)
        try:
            for token in gen:
                token = int(token)
                if logprobs is not None:
                    logprobs.append(self._token_logprob(token))
                if self._is_end_of_generation(token):
                    break
                if first_t is None:
                    first_t = time.perf_counter()
                    self.last_prompt_s = first_t - prompt_start
                out.append(token)
                if decoder is not None:
                    piece = decoder.decode(self._llm.detokenize([token]))
                    if not started:
                        piece = piece.lstrip()
                        started = bool(piece)
                    if piece:
                        on_text(piece)
                if len(out) >= max_tokens:
                    break
        finally:
            gen.close()
        last_t = time.perf_counter()

        self.last_prompt_ids, self.last_output_ids = prompt_ids, out
        self.last_token_logprobs = logprobs[:len(out)] if logprobs is not None else None
        text = self._llm.detokenize(out).decode("utf-8", errors="replace") if out else ""
        # Decode time from first to last token excludes prompt processing,
        # matching what Ollama's eval_duration measures.
        decode_s = (last_t - first_t) if first_t is not None else 0.0
        decoded_after_first = max(len(out) - 1, 0)
        return GenerationSample(
            tokens_generated=decoded_after_first,
            elapsed_s=decode_s,
            # In-process, so this process's RSS is the real footprint.
            peak_memory_gb=psutil.Process().memory_info().rss / (1024**3),
            output_text=text.strip(),
            acceleration=self._mode if self._counter is not None else ACCEL_NONE,
            acceptance_rate=self._counter.acceptance_rate(len(out)) if self._counter is not None else None,
        )

    def yes_probability(self, prompt: str | list[dict]) -> float:
        """P(the reply starts with yes) against no, from one evaluation of the prompt in the
        chat template -- a yes/no judgement without generating. Reuses the prompt prefix
        it shares with the cache (and saved caches) like generate()."""
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first.")
        llm = self._llm
        ids = self._chat_tokens(prompt)
        slots = kv_slots_for(llm, self._kv_slot_tokens) if self._reuse_prefix else None
        if slots is not None:
            slots.switch(llm, ids)
        n = min(_kv_lcp(llm.input_ids[: llm.n_tokens], ids), len(ids) - 1) if self._reuse_prefix else 0
        llm.n_tokens = n
        llm.eval(ids[n:])
        logits = self._final_logits()
        words = self._yes_no_tokens()
        yes = np.logaddexp.reduce(logits[words[0]])
        no = np.logaddexp.reduce(logits[words[1]])
        return float(1.0 / (1.0 + np.exp(no - yes)))

    def _yes_no_tokens(self) -> tuple[list[int], list[int]]:
        cached = getattr(self, "_yes_no", None)
        if cached is None:
            first = lambda w: int(self._llm.tokenize(w.encode(), add_bos=False)[0])  # noqa: E731
            cached = self._yes_no = ([first("yes"), first("Yes")], [first("no"), first("No")])
        return cached

    def _final_logits(self) -> np.ndarray:
        llm = self._llm
        ctx = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if llama_cpp is not None and ctx is not None:
            ptr = llama_cpp.llama_get_logits_ith(ctx, -1)
            return np.ctypeslib.as_array(ptr, shape=(llm.n_vocab(),)).astype(np.float64)
        return np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)  # test doubles

    def _token_logprob(self, token: int) -> float:
        """log p(token) from the logits it was just sampled from (plain mode: the
        context holds the last position's logits until the next eval)."""
        llm = self._llm
        try:
            logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
        except (AttributeError, TypeError, ValueError):
            return 0.0  # a test double without llama.cpp internals: no signal
        top = logits.max()
        return float(logits[token] - top - np.log(np.sum(np.exp(logits - top))))

    def verify_greedy(self, prompt_ids: Optional[list[int]] = None, output_ids: Optional[list[int]] = None) -> GreedyCheck:
        """
        Is `output_ids` what plain greedy decoding would produce, up to
        floating-point noise? Defaults to the last generation.

        1. Recompute the target model's logits for the sequence one token at
           a time -- exactly how plain decoding computes them -- and find,
           per position, how far the chosen token is behind the top one.
        2. Measure the noise on this same sequence: recompute it in batches
           of the sizes speculative decoding uses (the whole sequence, the
           draft length + 1, the lookup length + 1) and record how much the
           gap between the top token and each competitor that matters (the
           runner-up and the chosen token) moves. The tolerance is the largest
           such move -- the noise in exactly the quantity greedy decoding
           compares, not in the ~100k logits it never looks at.
        A chosen token behind the top one by more than that is a violation.
        """
        prompt_ids = self.last_prompt_ids if prompt_ids is None else prompt_ids
        output_ids = self.last_output_ids if output_ids is None else output_ids
        if not output_ids:
            return GreedyCheck(0, 0, 0, 0, 0.0)
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first.")
        if not self._llm_has_all_logits:  # plain mode keeps only the last logits; reload keeping all
            self._llm = self._factory(self._model_path, True)
            self._llm_has_all_logits = True
            self._install_draft()
        llm = self._llm
        n = len(output_ids)
        start = len(prompt_ids) - 1
        batch_sizes = sorted({len(output_ids), self._draft_tokens + 1, self._lookup_tokens + 1}
                             | ({self._max_draft_tokens + 1} if self._adaptive_draft else set()))

        def logits(chunk: int) -> np.ndarray:
            llm.reset()
            llm.eval(list(prompt_ids))
            rest = list(output_ids[:-1])
            for i in range(0, len(rest), chunk):
                llm.eval(rest[i:i + chunk])
            return np.array(llm.scores[start:start + n], dtype=np.float64)

        saved_draft, llm.draft_model = llm.draft_model, None
        windowed = hasattr(llm, "keep_scores")
        if windowed:
            llm.keep_scores = True
        try:
            sequential = logits(1)
            batched = [logits(size) for size in batch_sizes]
        finally:
            llm.reset()
            llm.draft_model = saved_draft
            if windowed:
                llm.keep_scores = False
                llm.scores = np.ndarray((llm.n_batch, llm._n_vocab), dtype=np.single)  # free verification rows

        rows = np.arange(n)
        order = np.argsort(sequential, axis=1)
        top, runner_up = order[:, -1], order[:, -2]
        chosen = np.asarray(output_ids)
        tolerance = 0.0
        for other in batched:
            for competitor in (runner_up, chosen):
                gap_seq = sequential[rows, top] - sequential[rows, competitor]
                gap_other = other[rows, top] - other[rows, competitor]
                tolerance = max(tolerance, float(np.max(np.abs(gap_seq - gap_other))))

        shortfall = sequential[rows, top] - sequential[rows, chosen]
        exact = int(np.sum(shortfall == 0.0))
        within = int(np.sum((shortfall > 0.0) & (shortfall <= tolerance)))
        return GreedyCheck(
            tokens=n,
            exact=exact,
            within_noise=within,
            violations=n - exact - within,
            noise_tolerance=tolerance,
        )

    # ---- prompt formatting -----------------------------------------------

    def _chat_tokens(self, prompt: str | list[dict]) -> list[int]:
        """The prompt in the model's own chat template, as create_chat_completion would
        format it: a string becomes one user turn, a message list keeps its roles."""
        llm = self._llm
        messages = (
            [{"role": "user", "content": prompt}] if isinstance(prompt, str)
            else [{"role": m["role"], "content": m["content"]} for m in prompt]
        )
        template = (getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template")
        if template and Jinja2ChatFormatter is not None:
            eos, bos = llm.token_eos(), llm.token_bos()
            formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=llm.detokenize([eos], special=True).decode("utf-8", errors="replace") if eos != -1 else "",
                bos_token=llm.detokenize([bos], special=True).decode("utf-8", errors="replace") if bos != -1 else "",
            )
            result = formatter(messages=messages)
            return list(llm.tokenize(result.prompt.encode("utf-8"), add_bos=not result.added_special, special=True))
        text = prompt if isinstance(prompt, str) else "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        return list(llm.tokenize(text.encode("utf-8"), add_bos=True))

    def _is_end_of_generation(self, token: int) -> bool:
        vocab = getattr(getattr(self._llm, "_model", None), "vocab", None)
        if llama_cpp is not None and vocab is not None:
            return bool(llama_cpp.llama_vocab_is_eog(vocab, token))
        return token == self._llm.token_eos()

    def unload(self) -> None:
        # Keep what the models computed: the next load restores it from disk instead of
        # re-reading the conversation (idle unload, config reload).
        spill_kv(self._llm)
        spill_kv(self._draft_llm)
        self._llm = None
        self._draft_llm = None
        self._counter = None
