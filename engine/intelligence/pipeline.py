"""
engine/intelligence/pipeline.py

Track B -- Intelligence Layer. Tasks 2-15 landed here incrementally;
this is now the full pipeline: ContextCompressor -> SemanticCache ->
SemanticRouter -> execution -> confidence check -> cache write ->
shadow labeling (Master Doc Section 4's architecture diagram).

Pipeline is one of the two shared seams (Master Doc Section 9): the
orchestrator Track C's CLI and gateway call, and which itself calls
whichever ExecutionBackend is selected -- Track A's LocalExecutionBackend
or Track C's SelfHostedBackend -- through the shared ExecutionBackend
protocol only. Neither backend knows the other exists (NFR-07).

Constructor shape matches Track A's own documented example exactly
(docs/archive/TRACK_A_HANDOFF_FOR_B_AND_C.md, Section 4a) so that the J-2
integration test's `FakePipeline(backend)` single-argument call keeps
working unmodified: `backend` is the only required argument, everything
else -- self_hosted_backend included -- is optional and defaults to a
behavior identical to the Task 2 skeleton. Verified: j2_integration_test.py
still passes 5/5 with zero test-body changes after every task below
landed (run it yourself: pytest engine/execution/j2_integration_test.py).

KNOWN WRINKLE -- flagged, not blocking (unchanged since Task 2):
    The one place this file reaches outside shared/types.py is the
    `LocalExecutionBackendError` import below. Strictly, Section 10's
    guiding rule says Track B never imports from engine/execution/
    directly -- but Track A's own handoff doc (Section 4a) writes this
    exact import as the expected integration pattern, so it's treated
    here as an accepted, documented exception. The clean long-term fix
    (a shared ExecutionBackendError base class in shared/types.py) is
    logged in shared/CHANGELOG.md as a future all-three-tracks decision.

STREAMING LIMITATION -- documented, not a bug:
    run_stream() applies compression, cache, and router exactly like
    run(), but does NOT apply confidence-based escalation (FR-05.3):
    that check needs the full answer text, which contradicts actually
    streaming tokens as they arrive. A local answer that starts
    streaming is committed to finishing as a local answer. The
    non-streaming run() has no such constraint and applies the full
    confidence check.
"""

from __future__ import annotations

import inspect
import time
from typing import Callable, Iterator, Optional

from pathlib import Path

from engine.execution.backend import LocalExecutionBackendError
from engine.intelligence.cache import SemanticCache
from engine.intelligence.context_compressor import ContextCompressor
from engine.intelligence.decision_log import log_decision
from engine.intelligence.router.classifier import SemanticRouterClassifier
from engine.intelligence.router.confidence import estimate_local_confidence
from engine.intelligence.router.shadow_labeling import ShadowLabeler
from shared.types import BackendResult, ExecutionBackend, Message, PipelineResult


class Pipeline:
    """Track B's orchestrator -- the seam Track C's CLI and gateway call."""

    def __init__(
        self,
        backend: ExecutionBackend,
        self_hosted_backend: Optional[ExecutionBackend] = None,
        cloud_byok_backend: Optional[ExecutionBackend] = None,
        compressor: Optional[ContextCompressor] = None,
        cache: Optional[SemanticCache] = None,
        router: Optional[SemanticRouterClassifier] = None,
        shadow_labeler: Optional[ShadowLabeler] = None,
        confidence_floor: Optional[float] = 0.5,
        decision_log_path: Optional[Path] = None,
        system_prompt: Optional[str] = None,
        uncached_tiers: tuple[str, ...] = ("self_hosted",),
        small_local_backend: Optional[ExecutionBackend] = None,
        small_token_confidence_floor: Optional[float] = 0.75,
    ) -> None:
        self._backend = backend
        # Optional fast, small local model for queries the router labels
        # "local" (trivial/moderate). `backend` then serves only the hard
        # ones -- cloud_byok-labelled queries with no cloud configured, and
        # small-model answers that look weak. None keeps one local model.
        self._small_local_backend = small_local_backend
        # Small-model answers whose mean token probability is below this go to
        # `backend` too. 0.75 catches 7 of 11 wrong qwen2.5 0.5B answers to easy
        # questions and re-asks 16 of 117 right ones
        # (docs/benchmark/small_confidence_*.json, BENCHMARK_REPORT Addendum 13).
        self._small_token_confidence_floor = small_token_confidence_floor
        self.last_served_by_small = False
        self._self_hosted_backend = self_hosted_backend
        self._cloud_byok_backend = cloud_byok_backend
        self._compressor = compressor
        self._cache = cache
        # The cache asks the main local model whether a close-but-unsure match asks the same
        # thing (CacheConfig.judge_floor). It judges only once that model is loaded.
        if cache is not None and getattr(cache, "judge", None) is None and callable(getattr(backend, "yes_probability", None)):
            cache.judge = backend.yes_probability
        self._router = router
        self._shadow_labeler = shadow_labeler
        self._confidence_floor = confidence_floor
        # Off by default (NFR-05's privacy stance) -- only written to if
        # a path is explicitly configured. See decision_log.py's module
        # docstring for why this exists.
        self._decision_log_path = decision_log_path
        # Prepended as a system message when the conversation has none.
        # None or "" sends the conversation unchanged.
        self._system_prompt = system_prompt or None
        # Tiers whose answers depend on live data or tools (the router's
        # self_hosted label means exactly that -- see coldstart_data.py), so
        # they are never served from, or written to, the cache.
        self._uncached_tiers = tuple(uncached_tiers)

    def warm(self) -> None:
        """Load the local models now (one tiny generation each), not on the first real question.
        Nothing is cached or logged."""
        probe: list[Message] = [{"role": "user", "content": "Say OK."}]
        for backend in (self._backend, self._small_local_backend):
            if backend is not None:
                backend.generate(probe)

    def release(self) -> None:
        """Unload the local models (idle unload). Runners that can keep what they computed
        (the llama.cpp runner's KV caches) write it to disk first."""
        for backend in (self._backend, self._small_local_backend):
            release = getattr(backend, "release", None)
            if callable(release):
                release()

    # ---- Pipeline seam (Section 9) --------------------------------------

    def run(self, messages: list[Message], on_text: Optional[Callable[[str], None]] = None) -> PipelineResult:
        """`on_text` receives the main local model's answer piece by piece as it is generated,
        for streaming replies. Answers that may still be replaced (the small model's, or a
        local one a remote tier may override) are not streamed: the caller sends the
        returned text, or what of it was not streamed yet."""
        stream_ok = on_text is not None and not (
            self._confidence_floor is not None and self._self_hosted_backend is not None)
        self._on_text = on_text if stream_ok else None
        try:
            return self._run(messages)
        finally:
            self._on_text = None

    def _big_generate(self, messages: list[Message]) -> BackendResult:
        """The main local backend's answer, streamed through on_text when it can stream."""
        on_text = getattr(self, "_on_text", None)
        if on_text is not None:
            try:
                accepts = "on_text" in inspect.signature(self._backend.generate).parameters
            except (TypeError, ValueError):
                accepts = False
            if accepts:
                return self._backend.generate(messages, on_text=on_text)
        return self._backend.generate(messages)

    def _run(self, messages: list[Message]) -> PipelineResult:
        start = time.monotonic()
        compressed = self._compressor.compress(messages) if self._compressor else messages
        compressed = self._with_system_prompt(compressed)

        tier = self._decide_initial_tier(compressed)
        use_cache = self._cache is not None and tier not in self._uncached_tiers

        if use_cache:
            cached = self._cache.lookup(compressed)
            if cached is not None:
                return self._finalize(self._cache_hit_result(cached, start))

        result, tier, escalated = self._execute(compressed, tier)

        if self.last_served_by_small and self._small_answer_is_weak(result):
                try:
                    result = self._big_generate(compressed)
                    escalated = True
                    self.last_served_by_small = False
                except Exception:
                    pass  # NFR-03: keep the small model's answer rather than fail

        if tier == "local" and self._confidence_floor is not None and self._self_hosted_backend is not None:
            local_confidence = estimate_local_confidence(result["text"])
            if local_confidence < self._confidence_floor:
                try:
                    result = self._self_hosted_backend.generate(compressed)
                    tier = "self_hosted"
                    escalated = True
                except Exception:
                    pass  # NFR-03: keep the local answer rather than fail the request

        if use_cache:
            self._cache.write(compressed, result["text"])

        if self._shadow_labeler is not None:
            self._shadow_labeler.maybe_label(compressed, tier, result["text"])

        latency_ms = (time.monotonic() - start) * 1000
        return self._finalize({
            "text": result["text"],
            "tier": tier,
            "cache_hit": False,
            "confidence": result["confidence"],
            "escalated": escalated,
            "speculative_active": result["speculative_active"],
            "tok_s": result["tok_s"],
            "latency_ms": round(latency_ms, 2),
            "est_cost_usd": result["est_cost_usd"],
        })

    def _with_system_prompt(self, messages: list[Message]) -> list[Message]:
        if not self._system_prompt or any(m["role"] == "system" for m in messages):
            return messages
        return [{"role": "system", "content": self._system_prompt}, *messages]

    def _finalize(self, result: PipelineResult) -> PipelineResult:
        """Every path through run() ends here -- the one place decision logging (FR-08.3) happens, off by default."""
        if self._decision_log_path is not None:
            log_decision(result, self._decision_log_path)
        return result

    def _small_answer_is_weak(self, result: BackendResult) -> bool:
        """Text heuristic (hedging, too short) against confidence_floor, and, when the
        backend reports token log-probabilities, their mean probability against
        small_token_confidence_floor."""
        if self._confidence_floor is not None and estimate_local_confidence(result["text"]) < self._confidence_floor:
            return True
        logprobs = result.get("logprobs")
        floor = self._small_token_confidence_floor
        return bool(logprobs) and floor is not None and estimate_local_confidence(result["text"], logprobs) < floor

    def run_stream(self, messages: list[Message]) -> Iterator[str]:
        """See module docstring's STREAMING LIMITATION note."""
        compressed = self._compressor.compress(messages) if self._compressor else messages
        compressed = self._with_system_prompt(compressed)

        tier = self._decide_initial_tier(compressed)
        if self._cache is not None and tier not in self._uncached_tiers:
            cached = self._cache.lookup(compressed)
            if cached is not None:
                yield cached
                return

        stream_backend = self._backend_for_tier(tier)
        if tier == "local" and self._small_local_backend is not None:
            stream_backend = self._small_local_backend

        try:
            yield from stream_backend.generate_stream(compressed)
        except LocalExecutionBackendError:
            if self._self_hosted_backend is None:
                raise
            yield from self._self_hosted_backend.generate_stream(compressed)

    # ---- Internal: tier decision & execution -----------------------------

    def _decide_initial_tier(self, messages: list[Message]) -> str:
        """
        Router-driven when a SemanticRouter is configured; otherwise
        defaults to "local" -- the exact Task 2 behavior, so J-2 (which
        never configures a router) is unaffected.
        """
        if self._router is None:
            return "local"
        query = messages[-1]["content"] if messages else ""
        if not query:
            return "local"
        decision = self._router.classify(query)
        return decision.tier

    def _backend_for_tier(self, tier: str) -> ExecutionBackend:
        if tier == "cloud_byok" and self._cloud_byok_backend is not None:
            return self._cloud_byok_backend
        if tier == "self_hosted" and self._self_hosted_backend is not None:
            return self._self_hosted_backend
        return self._backend  # local, or a requested tier with no backend configured -> fall back to local

    def _execute(self, messages: list[Message], tier: str) -> tuple[BackendResult, str, bool]:
        """
        Returns (result, actual_tier_served, escalated). actual_tier_served
        can differ from the requested `tier` if the requested backend
        wasn't configured (falls back to local) or local execution
        raised LocalExecutionBackendError (escalates to cascade, if
        configured -- and re-raises, uncaught, if not: this is the
        exact behavior j2_integration_test.py's test_j2_5 verifies).
        """
        cascade_failed = False
        self.last_served_by_small = False
        if tier == "local" and self._small_local_backend is not None:
            try:
                result = self._small_local_backend.generate(messages)
                self.last_served_by_small = True
                return result, "local", False
            except Exception:
                pass  # NFR-03: the main local model takes over
        remote = self._backend_for_tier(tier)
        if remote is not self._backend:
            try:
                return remote.generate(messages), tier, False
            except Exception:
                # NFR-03: a remote tier being down (no network, bad key,
                # quota) degrades to local instead of failing the request.
                cascade_failed = tier == "self_hosted"

        # "local", a requested tier whose backend isn't configured, or a failed remote tier.
        try:
            result = self._big_generate(messages)
            return result, "local", False
        except LocalExecutionBackendError:
            if self._self_hosted_backend is None or cascade_failed:
                raise
            result = self._self_hosted_backend.generate(messages)
            return result, "self_hosted", True

    def _cache_hit_result(self, cached_text: str, start: float) -> PipelineResult:
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "text": cached_text,
            "tier": "cache",
            "cache_hit": True,
            "confidence": 1.0,
            "escalated": False,
            "speculative_active": False,
            "tok_s": 0.0,
            "latency_ms": round(latency_ms, 2),
            "est_cost_usd": 0.0,
        }
