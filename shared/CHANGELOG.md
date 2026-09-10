# shared/ CHANGELOG

Durable record of any change to either seam (`Pipeline` or
`ExecutionBackend`) or to `HardwareProfile`/`BackendResult`/`PipelineResult`.
Every entry below must have been reviewed by all three tracks before merge
(Section 10.3). This file is the source of truth over anything said on a
call or in chat.

## [Unreleased] -- Phase 1, Week 1

### Added (Track A, proposed for J-1 freeze)

- `shared/types.py` created with `Message`, `HardwareProfile`,
  `BackendResult`, `PipelineResult`, and the `ExecutionBackend` Protocol.
- `HardwareProfile` extends the Section 9 sketch with the full field set
  FR-01.1 requires (`cpu_cores`, `gpu_vendor`, `disk_throughput_mb_s`,
  `disk_free_gb`, `peak_memory_gb`, `schema_version`, `profiled_at`, `os`)
  in addition to the `ram_gb` / `vram_gb` / `measured_tok_s` /
  `local_capable` fields shown in the master doc's illustrative snippet.
  Rationale: FR-01.1 explicitly requires the system to detect all of
  these; Section 9's snippet was a minimal illustration of the seam
  shape, not an exhaustive schema. Added `schema_version` (starts at
  `HARDWARE_PROFILE_SCHEMA_VERSION = 1`) so ModelLadder and future
  consumers can detect an incompatible profile instead of guessing at
  missing fields.
- `ExecutionBackend.generate()` / `.generate_stream()` signatures match
  the master doc exactly (list[Message] -> BackendResult /
  Iterator[str]).

### Status

- Once frozen, any further change to this file requires a new dated
  entry here plus all three tracks' review on the PR (Section 10.3),
  no exceptions.

## [Unreleased] -- Phase 2, Week of 11 September 2026

### J-1 sign-off (Track B)

- **Track B reviews and signs off on `shared/types.py` as proposed by
  Track A** -- the extended `HardwareProfile` schema (assumption #1 in
  `docs/TRACK_A_PHASE1_SUMMARY.md`), `BackendResult`, `PipelineResult`,
  and the `ExecutionBackend` protocol are all workable as-is for
  `engine/intelligence/pipeline.py`. No changes requested.
- **Schema is now frozen** per Section 13.1. Track C sign-off still
  outstanding -- please add an entry here when reviewed.

### Added (Track B) -- not a change to shared/types.py itself

- `engine/intelligence/pipeline.py`: `Pipeline` class implementing the
  Section 9 Pipeline seam (Task 2). Constructor is
  `Pipeline(backend: ExecutionBackend, cascade_backend: ExecutionBackend
  | None = None)` -- matches the single-positional-argument call shape
  in `docs/TRACK_A_HANDOFF_FOR_B_AND_C.md` Section 4a and in
  `engine/execution/j2_integration_test.py`'s `FakePipeline(backend)`
  calls exactly, so J-2 required zero test-body changes (verified: all
  5 J-2 tests green). `cascade_backend` is optional and defaults to
  `None`; with no cascade_backend configured, a `LocalExecutionBackendError`
  propagates uncaught rather than being silently swallowed (test_j2_5's
  expectation).
- **J-2 verified and closed.** `engine/execution/pipeline_bridge.py`
  (the `FakePipeline` shim) has been deleted per Track A's handoff
  instructions, Section 4b/6.

### Flagged for future review -- not acted on unilaterally

- `engine/intelligence/pipeline.py` currently imports
  `LocalExecutionBackendError` directly from `engine/execution/backend.py`
  to know when to escalate to `cascade_backend`. This is the exact
  pattern Track A's own handoff doc documents (Section 4a), but it is
  in tension with Section 10's guiding rule that Track B should import
  only from `shared/types.py`. Clean fix for a later revision: add a
  shared `ExecutionBackendError` base class to `shared/types.py` that
  both `LocalExecutionBackendError` (Track A) and any future
  `CascadeCloudBackendError` (Track C) subclass, so `Pipeline` can catch
  one shared, protocol-level error type instead of reaching into either
  track's module. Needs all three tracks' sign-off per Section 10.3
  before it's made -- not done here since it touches the frozen file
  and isn't blocking Phase 2 work.

## [Unreleased] -- Phase 2/3, Tasks 3-16 complete (Track B)

Full detail in `docs/TRACK_B_PHASE2_3_SUMMARY.md`. Summary of what's
relevant to the other two tracks specifically:

- **`Pipeline`'s constructor grew new optional parameters**
  (`cloud_byok_backend`, `compressor`, `cache`, `router`,
  `shadow_labeler`, `confidence_floor`) -- all optional, all default to
  the exact Task 2 behavior. `Pipeline(backend)` (J-2's call shape)
  is unaffected; verified green after every task landed.
- **`FR-05.3` gap confirmed, not resolved:** no `ExecutionBackend`
  implementation exposes logprobs/entropy today, so local-tier
  confidence always uses the heuristic fallback. If Track C's
  `CascadeCloudBackend` (or a future Track A revision) ever exposes
  token-level logprobs via `BackendResult`, that requires a
  `shared/types.py` change and thus all-three sign-off -- flag it here
  when it happens rather than adding a field ad hoc.
- **Open ownership question, not blocking:** cloud-BYOK (LiteLLM)
  wiring. `Pipeline` accepts and correctly dispatches to an optional
  `cloud_byok_backend: ExecutionBackend`, but nobody has built the
  actual LiteLLM-backed implementation yet. Worth a quick team decision
  on whether that's Track B or Track C's file to own.
- **Track C, when you build `gateway/backend.py`:** `Pipeline` already
  calls `cascade_backend.generate()` / `.generate_stream()` correctly
  in three situations -- router-selected `cascade_cloud`, confidence-
  based escalation from a weak local answer, and `LocalExecutionBackendError`
  escalation -- all verified against fakes conforming to `ExecutionBackend`.
  No changes needed in `engine/intelligence/` once your real backend
  lands; that's what the protocol is for.
