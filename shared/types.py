"""
shared/types.py

The two seams (Master Requirements Doc, Section 9) that let three
independently-working tracks build Cascade without ever importing from
each other's folders:

  1. Pipeline         -- Track B's orchestrator, called by Track C's CLI
                          and gateway API.
  2. ExecutionBackend  -- implemented independently by Track A
                          (LocalExecutionBackend) and Track C
                          (CascadeCloudBackend); called only by Track B's
                          Pipeline. Neither backend knows the other exists.

CHANGES TO THIS FILE REQUIRE ALL THREE TRACKS' REVIEW, EVERY TIME, NO
EXCEPTIONS (Section 10.3) -- this is the one deliberately synchronous
gate in the whole workflow. Record every change in shared/CHANGELOG.md;
do not rely on it being remembered or said out loud on a call.

--------------------------------------------------------------------------
Track A's Phase 1 (Weeks 1-2) usage of this file:
    - HardwareProfile is produced EXCLUSIVELY by
      engine/execution/hardware_profiler.py (FR-01.3). No other module in
      the codebase may re-derive cpu/ram/gpu/disk facts.
    - BackendResult is returned by engine/execution/backend.py's
      LocalExecutionBackend stub (Task 4, this phase) and, later, its
      real implementation (Phase 2, Task 11).
    - ExecutionBackend is the protocol LocalExecutionBackend conforms to,
      so Track B can call it with zero knowledge of what's underneath.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterator, Literal, Optional, Protocol, TypedDict, runtime_checkable

# Bump this whenever a field is added/removed/retyped on HardwareProfile.
# ModelLadder (Phase 2) and anything else that consumes a HardwareProfile
# should refuse to trust an object whose schema_version it doesn't recognize
# rather than guess at missing/renamed fields.
HARDWARE_PROFILE_SCHEMA_VERSION = 1


class Message(TypedDict):
    """One turn of conversation passed into Pipeline / ExecutionBackend."""

    role: Literal["system", "user", "assistant"]
    content: str


class HardwareProfile(TypedDict):
    """
    The versioned, MEASURED description of one machine.

    Produced ONLY by HardwareProfiler (FR-01.3, FR-01.4) -- no other
    component in the codebase (ModelLadder, Pipeline, the CLI) may
    re-derive hardware facts; every consumer treats this object as the
    single source of truth for "what can this machine actually do."

    Field notes:
        schema_version        -- see HARDWARE_PROFILE_SCHEMA_VERSION above.
        profiled_at            -- ISO-8601 UTC timestamp of this profiling run.
        os                     -- "linux" | "darwin" | "windows".
        cpu_cores              -- logical core count (FR-01.1).
        ram_gb                 -- total system RAM, GB (FR-01.1).
        gpu_vendor              -- "nvidia" | "apple" | "amd" | None (FR-01.1).
        vram_gb                -- 0.0 when gpu_vendor is None (FR-01.1).
        disk_throughput_mb_s   -- measured sequential write throughput (FR-01.1).
        disk_free_gb           -- free space on the install volume (FR-01.1).
        measured_tok_s         -- {quant_level_label: measured tokens/sec},
                                   >=2 entries per FR-01.2.
        peak_memory_gb         -- {quant_level_label: peak RSS/VRAM during
                                   that quant level's benchmark run}, same
                                   keys as measured_tok_s.
        local_capable          -- FR-01.5: False means no candidate rung
                                   cleared the configured latency floor --
                                   Pipeline routes this profile to
                                   CascadeCloud (FR-07) by default.
    """

    schema_version: int
    profiled_at: str
    os: Literal["linux", "darwin", "windows"]
    cpu_cores: int
    ram_gb: float
    gpu_vendor: Optional[Literal["nvidia", "apple", "amd"]]
    vram_gb: float
    disk_throughput_mb_s: float
    disk_free_gb: float
    measured_tok_s: dict[str, float]
    peak_memory_gb: dict[str, float]
    local_capable: bool


class BackendResult(TypedDict):
    """
    The typed result any ExecutionBackend implementation (local or
    hosted) returns to Pipeline. Identical shape regardless of which tier
    produced it -- this is what lets Pipeline treat
    Track A's LocalExecutionBackend and Track C's CascadeCloudBackend
    interchangeably (NFR-07).
    """

    text: str
    confidence: float
    speculative_active: bool
    tok_s: float
    latency_ms: float
    est_cost_usd: float


class PipelineResult(TypedDict):
    """What Pipeline.run() / Pipeline.run_stream() return to callers
    (Track C's CLI and gateway API). Track A does not construct this
    type directly -- it is assembled by Track B's Pipeline from a
    BackendResult plus routing metadata."""

    text: str
    tier: Literal["cache", "local", "cloud_byok", "cascade_cloud"]
    cache_hit: bool
    confidence: float
    escalated: bool
    speculative_active: bool
    tok_s: float
    latency_ms: float
    est_cost_usd: float


@runtime_checkable
class ExecutionBackend(Protocol):
    """
    Implemented independently by:
        - Track A: engine/execution/backend.py -> LocalExecutionBackend
        - Track C: gateway/backend.py           -> CascadeCloudBackend

    Track B's Pipeline calls ONLY this protocol -- never anything inside
    engine/execution/ or gateway/ directly (NFR-07, Section 10.4's
    "definition of joined" checklist). Neither backend implementation
    knows the other exists.

    A shared contract test suite (shared/tests/test_execution_backend_contract.py)
    is expected to run against every implementation in CI, per Risk R-12 --
    that suite is out of scope for this Phase 1 stub but the protocol
    below is written so that suite can be added without changing this file.
    """

    def generate(self, messages: list[Message]) -> BackendResult:
        ...

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        ...
