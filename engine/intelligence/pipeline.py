"""
engine/intelligence/pipeline.py

Track B -- Intelligence Layer. Tasks 2-15 landed here incrementally;
this is now the full pipeline: ContextCompressor -> SemanticCache ->
SemanticRouter -> execution -> confidence check -> cache write ->
shadow labeling (Master Doc Section 4's architecture diagram).

Pipeline is one of the two shared seams (Master Doc Section 9): the
orchestrator Track C's CLI and gateway call, and which itself calls
whichever ExecutionBackend is selected -- Track A's LocalExecutionBackend
or Track C's CascadeCloudBackend -- through the shared ExecutionBackend
protocol only. Neither backend knows the other exists (NFR-07).

Constructor shape matches Track A's own documented example exactly
(docs/TRACK_A_HANDOFF_FOR_B_AND_C.md, Section 4a) so that the J-2
integration test's `FakePipeline(backend)` single-argument call keeps
working unmodified: `backend` is the only required argument, everything
else -- cascade_backend included -- is optional and defaults to a
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

import time
from typing import Iterator, Optional

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
        cascade_backend: Optional[ExecutionBackend] = None,
        cloud_byok_backend: Optional[ExecutionBackend] = None,
        compressor: Optional[ContextCompressor] = None,
        cache: Optional[SemanticCache] = None,
        router: Optional[SemanticRouterClassifier] = None,
        shadow_labeler: Optional[ShadowLabeler] = None,
        confidence_floor: Optional[float] = 0.5,
        decision_log_path: Optional[Path] = None,
    ) -> None:
        self._backend = backend
        self._cascade_backend = cascade_backend
        self._cloud_byok_backend = cloud_byok_backend
        self._compressor = compressor
        self._cache = cache
        self._router = router
        self._shadow_labeler = shadow_labeler
        self._confidence_floor = confidence_floor
        # Off by default (NFR-05's privacy stance) -- only written to if
        # a path is explicitly configured. See decision_log.py's module
        # docstring for why this exists.
        self._decision_log_path = decision_log_path

    # ---- Pipeline seam (Section 9) --------------------------------------

    def run(self, messages: list[Message]) -> PipelineResult:
        start = time.monotonic()
        compressed = self._compressor.compress(messages) if self._compressor else messages

        if self._cache is not None:
            cached = self._cache.lookup(compressed)
            if cached is not None:
                return self._finalize(self._cache_hit_result(cached, start))

        tier = self._decide_initial_tier(compressed)
        result, tier, escalated = self._execute(compressed, tier)

        if tier == "local" and self._confidence_floor is not None and self._cascade_backend is not None:
            local_confidence = estimate_local_confidence(result["text"])
            if local_confidence < self._confidence_floor:
                tier = "cascade_cloud"
                escalated = True
                result = self._cascade_backend.generate(compressed)

        if self._cache is not None:
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

    def _finalize(self, result: PipelineResult) -> PipelineResult:
        """Every path through run() ends here -- the one place decision logging (FR-08.3) happens, off by default."""
        if self._decision_log_path is not None:
            log_decision(result, self._decision_log_path)
        return result

    def run_stream(self, messages: list[Message]) -> Iterator[str]:
        """See module docstring's STREAMING LIMITATION note."""
        compressed = self._compressor.compress(messages) if self._compressor else messages

        if self._cache is not None:
            cached = self._cache.lookup(compressed)
            if cached is not None:
                yield cached
                return

        tier = self._decide_initial_tier(compressed)
        stream_backend = self._backend_for_tier(tier)

        try:
            yield from stream_backend.generate_stream(compressed)
        except LocalExecutionBackendError:
            if self._cascade_backend is None:
                raise
            yield from self._cascade_backend.generate_stream(compressed)

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
        if tier == "cascade_cloud" and self._cascade_backend is not None:
            return self._cascade_backend
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
        if tier == "cloud_byok" and self._cloud_byok_backend is not None:
            return self._cloud_byok_backend.generate(messages), "cloud_byok", False

        if tier == "cascade_cloud" and self._cascade_backend is not None:
            return self._cascade_backend.generate(messages), "cascade_cloud", False

        # "local", or a requested tier whose backend isn't configured yet.
        try:
            result = self._backend.generate(messages)
            return result, "local", False
        except LocalExecutionBackendError:
            if self._cascade_backend is None:
                raise
            result = self._cascade_backend.generate(messages)
            return result, "cascade_cloud", True

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
