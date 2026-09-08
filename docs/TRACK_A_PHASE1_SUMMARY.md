# Track A — Execution Engine — Phase 1 (Weeks 1–2) — Complete

Owner: Track A. Scope: Section 13's Phase 1 row for Track A + Tasks 1–4
of the Individual Task Breakdown. Requirements covered: FR-01.1–01.5.

## What was built

| File | Purpose |
|---|---|
| `shared/types.py` | The `HardwareProfile`, `BackendResult`, `PipelineResult`, `Message` types and the `ExecutionBackend` protocol — Section 9's seam contract. Proposed for the J-1 freeze (see below). |
| `shared/CHANGELOG.md` | Durable record of the seam decisions made this phase, per Section 10.2. |
| `engine/execution/hardware_profiler.py` | `HardwareProfiler` — Tasks 1, 2, 3. Implements FR-01.1–01.5 in full. |
| `engine/execution/model_runner.py` | `ReferenceModelRunner` protocol + `OllamaReferenceModelRunner`. The seam that lets the benchmark loop be unit-tested with no live GPU/Ollama (NFR-09). |
| `engine/execution/backend.py` | `LocalExecutionBackend` — Task 4. Stub conforming to `ExecutionBackend`, ready for the Phase 1 join point. |
| `engine/execution/tests/test_hardware_profiler.py` | 27 tests, all hardware/subprocess calls mocked. |
| `engine/execution/tests/test_backend.py` | 8 tests covering the stub's contract conformance and typed output. |

## Task-by-task definition-of-done verification

**Task 1 — hardware detection (FR-01.1).**
`detect_cpu_cores`, `detect_ram_gb`, `detect_gpu` (NVIDIA via `nvidia-smi`,
Apple Silicon unified memory, AMD via `rocm-smi`, graceful `None`
otherwise), `detect_disk_throughput_mb_s` (real timed write, not a spec
lookup), `detect_disk_free_gb`. Verified against this machine directly:

```
OS: linux · CPU cores: 1 · RAM: 3.9 GB · GPU: None (0.0 GB)
Disk throughput: 588.7 MB/s · Disk free: 9.98 GB
```

These are the real, correct values for this sandbox — the DoD's "correct
values on your own machine" bar. All 13 detection tests mock
`subprocess`/`psutil`/`platform` so they never depend on what happens to
be true of whichever machine runs CI.

**Task 2 — timed benchmark loop (FR-01.2).**
`HardwareProfiler.run_benchmark()` loads a reference model at each
configured quant level (3 by default: `q4_K_M`, `q5_K_M`, `q8_0`),
times one generation per level, and returns `(measured_tok_s,
peak_memory_gb)`. Verified:
- Stays inside a 60s budget — a deadline is computed once and no new
  quant level starts once fewer than 5s remain (`test_run_benchmark_respects_time_budget`).
- A quant level that fails to load (OOM) is skipped, not fatal —
  `test_run_benchmark_skips_a_level_that_ooms_but_keeps_the_rest`
  confirms the other levels still get measured (NFR-03).
- Repeatability: `test_run_benchmark_repeatable_within_tolerance` runs
  the loop 3x against a deterministic fake and asserts <10% drift —
  this locks in that `run_benchmark()` is a pure function of the
  injected runner's outputs, with no shared state leaking between
  calls, which is what real-world repeatability depends on once a real
  `ReferenceModelRunner` is swapped in.

**Task 3 — versioned `HardwareProfile`, `local_capable`, re-profile support (FR-01.3–01.5).**
`HardwareProfiler.profile()` is the single method that assembles
detection + benchmark into one `HardwareProfile`, and it is the ONLY
place in the codebase that computes hardware facts — everything else
is expected to call `load_cached_profile()` instead of re-deriving
anything (FR-01.3). It:
- Stamps `schema_version` (currently `1`, from `shared/types.py`'s
  `HARDWARE_PROFILE_SCHEMA_VERSION`) and a UTC `profiled_at` timestamp.
- Computes `local_capable` from the best measured tok/s against a
  configurable floor (`DEFAULT_MIN_TOK_S_FLOOR = 3.0`) — see the
  **assumption flagged below**.
- Persists to `~/.cascade/hardware_profile.json` by default (path is
  constructor-injectable for tests), so `cascade profile` (FR-01.4) is
  just "call `profile()` again" — no reinstall, no special first-run
  state. `test_profile_supports_repeated_reprofiling_without_reinstall`
  simulates exactly this: profile, simulate a hardware regression,
  re-profile, confirm the cached profile updates in place.
- Rejects a cached profile written by a mismatched schema version
  rather than trusting stale/incompatible fields silently.

**Task 4 — stub `LocalExecutionBackend` (Phase 1 join point).**
`engine/execution/backend.py`'s `LocalExecutionBackend` returns a
canned but fully well-typed `BackendResult` for both `generate()` and
`generate_stream()`. It asserts `isinstance(LocalExecutionBackend(),
ExecutionBackend)` at import time, so any future accidental drift from
the protocol fails immediately and loudly rather than silently at
integration time. Confirmed by direct simulation of a Track-B-style
caller that only imports `shared.types.ExecutionBackend` and never
imports `engine.execution.backend` internals directly.

## Test results

```
34 passed in 0.82s
Coverage: hardware_profiler.py 90%, backend.py 95%, package overall 79%
(exceeds NFR-09's 70% floor)
```

`model_runner.py`'s `OllamaReferenceModelRunner` (the *production*
runner) is intentionally left largely uncovered by unit tests — per
NFR-09 it must not depend on a live Ollama instance, and exercising it
for real is integration/manual-testing scope, not this suite's job.
The `ReferenceModelRunner` Protocol itself, and everything
`HardwareProfiler` does with it, is fully covered via the deterministic
`FakeReferenceModelRunner` test double.

## Assumptions and decisions flagged for review

1. **`HardwareProfile` schema is larger than Section 9's illustrative
   snippet.** The master doc's Section 9 code block shows `HardwareProfile`
   with only `ram_gb`, `vram_gb`, `measured_tok_s`, `local_capable`.
   FR-01.1 explicitly requires CPU cores, GPU vendor, disk throughput,
   and disk free space to be detected too, so this implementation adds
   `cpu_cores`, `gpu_vendor`, `disk_throughput_mb_s`, `disk_free_gb`,
   `peak_memory_gb`, `schema_version`, `profiled_at`, and `os`. Recorded
   in `shared/CHANGELOG.md` with rationale, flagged **pending J-1
   sign-off** from Track B and Track C before it's treated as frozen —
   per Section 13.1, all three tracks need to agree on `shared/types.py`
   before further engine/gateway/platform code is written against it.

2. **`local_capable` floor is a Phase 1 placeholder, not ModelLadder's
   real rung logic.** FR-01.5 ties `local_capable` to "clears no local
   rung's latency floor," but rungs don't exist until ModelLadder ships
   in Phase 2 (Task 5). This implementation uses one configurable
   constant (`DEFAULT_MIN_TOK_S_FLOOR = 3.0`) as the "untrained starting
   point" heuristic the master doc's Section 4 preamble explicitly
   allows, and documents inline that Phase 2's ModelLadder is expected
   to supersede this with a real per-rung check.

3. **Reference model identity (`cascade-reference-3b`) is a placeholder
   tag**, not yet a decision about which actual small model
   HardwareProfiler benchmarks with — that's a joint decision (it also
   affects Task 9's draft/target pair validation in Phase 2) rather than
   something to lock in unilaterally in Phase 1.

## What Track B can do right now

Import `shared.types.ExecutionBackend` and
`engine.execution.backend.LocalExecutionBackend`, and start building
Pipeline against it — nothing above requires Track B to know anything
about `HardwareProfiler`, `ModelLadder`, or model loading internals.

## What's explicitly NOT in this phase (by design, per Section 13)

- Real model loading/execution inside `LocalExecutionBackend` — Phase 2,
  Task 11.
- `ModelLadder`, `AccelerationEngine` — Phase 2, Tasks 5–10.
- Anything in `gateway/`, `cli/`, or `engine/intelligence/` — other
  tracks' folders; Task A never imports from them (Section 10's guiding
  rule).

## Next action needed from the team (not code)

**J-1**: All three tracks review and freeze `shared/types.py` (Section
13.1) before Phase 2 engine/gateway/platform code is written against
it — specifically the schema additions to `HardwareProfile` flagged in
assumption #1 above.
