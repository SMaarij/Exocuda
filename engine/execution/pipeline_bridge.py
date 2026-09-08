
"""
engine/execution/pipeline_bridge.py

                ????????????????????????????????????????????????????????
                ?  ?  TRACK-B-NOT-PRESENT  ?                        ?
                ?                                                      ?
                ?  This file exists ONLY because Track B's real        ?
                ?  Pipeline (engine/intelligence/pipeline.py) does      ?
                ?  not yet exist.                                       ?
                ?                                                      ?
                ?  It is a minimal shim that calls LocalExecutionBackend ?
                ?  directly so Track A can run end-to-end integration    ?
                ?  tests (j2_integration_test.py) right now.            ?
                ?                                                      ?
                ?  HOW TO REMOVE THIS WHEN TRACK B LANDS:              ?
                ?  1. Track B publishes engine/intelligence/pipeline.py  ?
                ?     with a Pipeline class accepting an ExecutionBackend.?
                ?  2. Delete THIS file entirely.                        ?
                ?  3. In j2_integration_test.py replace:               ?
                ?       from engine.execution.pipeline_bridge import    ?
                ?           FakePipeline                                ?
                ?     with:                                             ?
                ?       from engine.intelligence.pipeline import        ?
                ?           Pipeline as FakePipeline                    ?
                ?  4. Run: pytest engine/execution/j2_integration_test.py?
                ?  5. Green = J-2 verified. Ship.                       ?
                ?                                                      ?
                ????????????????????????????????????????????????????????
"""

from __future__ import annotations

import time
from typing import Iterator

from engine.execution.backend import LocalExecutionBackend
from shared.types import BackendResult, ExecutionBackend, Message, PipelineResult

# ?? Public API notice ??????????????????????????????????????????????????????
# FakePipeline mirrors the interface Track B's real Pipeline will expose
# (per Section 9 of the master doc): __init__(backend: ExecutionBackend),
# run(messages) -> PipelineResult, run_stream(messages) -> Iterator[str].
# When Track B lands, swapping the import is sufficient -- no other changes.


class FakePipeline:
    """
    ? TRACK-B-NOT-PRESENT: minimal Pipeline shim for Track A integration
    testing. Calls LocalExecutionBackend directly; Track B's real Pipeline
    will also call through ExecutionBackend but will add routing, caching,
    and confidence escalation on top.

    Do NOT use in production. Do NOT import from gateway/ or cli/.
    """

    def __init__(self, backend: ExecutionBackend) -> None:  # same sig as Track B
        self._backend = backend

    def run(self, messages: list[Message]) -> PipelineResult:
        """
        Call generate() and wrap the BackendResult into a PipelineResult.
        Track B's real Pipeline will apply routing and caching here.
        """
        start = time.monotonic()
        result: BackendResult = self._backend.generate(messages)
        latency_ms = (time.monotonic() - start) * 1000

        return {
            "text":               result["text"],
            "tier":               "local",  # ? FakePipeline always claims "local"
            "cache_hit":          False,    # ? no cache in the shim
            "confidence":         result["confidence"],
            "escalated":          False,    # ? no escalation logic in the shim
            "speculative_active": result["speculative_active"],
            "tok_s":              result["tok_s"],
            "latency_ms":         round(latency_ms, 2),
            "est_cost_usd":       result["est_cost_usd"],
        }

    def run_stream(self, messages: list[Message]) -> Iterator[str]:
        """Streaming variant -- passes through generate_stream() directly."""
        yield from self._backend.generate_stream(messages)
