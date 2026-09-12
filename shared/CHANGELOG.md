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

## [Unreleased] -- Track C complete (gateway/, cli/)

Full detail in `docs/TRACK_C_PHASE_SUMMARY.md`.

### J-1 sign-off (Track C)

- **Track C reviews and signs off on `shared/types.py`.**
  `ExecutionBackend`, `BackendResult`, `HardwareProfile`, and
  `PipelineResult` are all workable as-is for `gateway/backend.py` and
  `cli/main.py`. No changes requested.
- **`shared/types.py` is now fully frozen** -- all three tracks have
  signed off (Track A implicitly by building against it since Phase 1,
  Track B and Track C explicitly in this file).

### J-4 verified (Task 17, previously blocked)

- **The real three-way run is done.** `tests/integration/test_j4_three_way_run.py`
  wires Track A's real `LocalExecutionBackend`, Track B's real
  `Pipeline`, and Track C's real `CascadeCloudBackend` together -- this
  was blocked in `docs/TRACK_B_PHASE2_3_SUMMARY.md` pending
  `CascadeCloudBackend`'s existence. 4 tests, all green.

### Added (Track C) -- not a change to shared/types.py itself

- `gateway/backend.py`: `CascadeCloudBackend`, conforms to
  `ExecutionBackend` exactly like `LocalExecutionBackend` does.
- `gateway/api/app.py`: the FastAPI gateway -- where spend-cap
  enforcement actually has to live for real (server-side; a client-side
  check alone is bypassable).
- `cli/main.py`: `init`/`ask`/`chat`/`stats`/`profile`, wiring the full
  `Pipeline` per `docs/TRACK_B_HANDOFF_FOR_C.md` Section 4.

### Flagged for review -- not acted on unilaterally

- **A real bug, found and fixed:** `UsageMeter`'s first design split
  "check the spend cap" and "record the spend" across two separate lock
  acquisitions -- individually thread-safe, but not atomic together,
  which reopens the exact race a spend cap exists to prevent. Fixed:
  `check_and_reserve()` now checks-and-reserves atomically in one lock
  acquisition; `adjust()` reconciles the estimate with the real cost
  afterward. Verified with a 20-thread concurrent test.
- **Gap found and closed, touching a Track B file:** `cascade stats`
  (FR-08.3) needed a per-request decision log that `Pipeline` never
  wrote (only the sampled shadow-labeling log existed). Added
  `engine/intelligence/decision_log.py` and one new optional
  `decision_log_path` constructor parameter on `Pipeline` -- off by
  default (NFR-05). Flagged here since it's Track B's file changed by
  Track C's work, not done silently.
- **A doc bug found, not fixed unilaterally:** `docs/TRACK_A_HANDOFF_FOR_B_AND_C.md`
  Section 7's Track C example imports `OllamaReferenceModelRunner` from
  `engine.execution.hardware_profiler`, but it actually lives in
  `engine.execution.model_runner` -- the example as written would raise
  `ImportError`. `cli/main.py` uses the corrected two-line import.
  Worth Track A updating their own doc to match.
- **Open, unresolved:** streaming CascadeCloud requests don't reconcile
  their spend reservation to a real token count (the streaming loop
  doesn't get an exact usage payload back). The reservation stands as
  the final charge for now -- a candidate fix once a wholesale
  provider's streaming usage reporting (e.g. OpenAI-compatible
  `stream_options: {"include_usage": true}`) is wired in.

## [Unreleased] -- Section 12 benchmark suite, first real run

Full detail in `docs/BENCHMARK_REPORT.md`. One item here since it
touches a Track A file:

- **Real bug found and fixed in `engine/execution/model_runner.py`**
  (Track A's file): `OllamaReferenceModelRunner` crashed with
  `UnicodeDecodeError` on Windows -- `subprocess`'s `text=True` defaults
  to the system codepage (cp1252 here), and both `ollama pull`'s
  progress output and real model-generated text routinely contain
  bytes outside it. Fixed by passing `encoding="utf-8", errors="replace"`
  explicitly on every subprocess call in that file. Found by actually
  running `scripts/benchmark_acceleration.py` against a real Ollama
  install for the first time -- exactly the kind of thing NFR-09's
  "integration/manual-testing scope, not this suite's job" carve-out
  for this file exists to catch eventually. No unit-test change needed
  (`FakeReferenceModelRunner` never touches a real subprocess), but
  worth Track A knowing their file changed.
- **NFR-01 (speculative decoding speedup) remains unmeasured, not
  failed.** A real Layer 1 vs. 2 vs. 3 comparison needs a model with
  trained draft heads or a validated draft/target pair
  (`scripts/validate_draft_pairs.py`) -- neither exists yet, so every
  layer currently falls back to Layer 1 regardless of hardware. Also
  needs a machine with enough free RAM to reliably load the target
  model within the existing timeout. See `docs/BENCHMARK_REPORT.md`'s
  addendum for what was actually attempted and why it's inconclusive
  rather than a real pass/fail.

## [Unreleased] -- Two placeholders resolved, one deeper issue found

Both touch Track A's files (`engine/execution/`), flagged here per
Section 10.3.

### Resolved: the reference model placeholder (HardwareProfiler only)

- `DEFAULT_REFERENCE_MODEL_ID` changed from the non-existent
  `"cascade-reference-3b"` to `"llama3.2"`, and `DEFAULT_QUANT_LEVELS`
  from bare `("q4_K_M", "q5_K_M", "q8_0")` to
  `("3b-instruct-q4_K_M", "3b-instruct-q5_K_M", "3b-instruct-q8_0")` --
  verified real against `ollama.com/library/llama3.2/tags` (not
  guessed). `cascade profile` / `cascade init` can now actually pull
  and benchmark a real model.
- Two `test_hardware_profiler.py` tests were silently relying on the
  module default instead of passing their own `quant_levels` --
  fixed to be explicit, which is what they should have done regardless
  of what the default happens to be.

### Found, NOT resolved: production model loading needs a real design decision

- This fix **only** covers `HardwareProfiler`'s benchmarking probe.
  `ModelLadder.select_rung()` only ever reads `max(measured_tok_s.values())`
  -- it never matches by label -- so `HardwareProfiler`'s quant-level
  strings and `ModelLadder.DEFAULT_LADDER`'s rung labels
  (`"q4_K_M"`/`"q5_K_M"`/`"q8_0"`) are **not actually coupled today**,
  which is why this fix was safe to make alone.
- But `backend.py`'s `LocalExecutionBackend` still defaults
  `model_id="cascade-reference-3b"` and builds real Ollama tags as
  `f"{model_id}:{rung.label}"` -- e.g. `"cascade-reference-3b:q4_K_M"`.
  Real Ollama tags for a given model+quant combo are usually one dash-
  joined string after a single colon (e.g. `llama3.2:3b-instruct-q4_K_M`),
  not `model:bare_quant_label`. Simply pointing `model_id` at `"llama3.2"`
  here would build an invalid tag (`"llama3.2:q4_K_M"` is not a real tag).
- Fixing this for real needs one of: (a) a model whose real tags
  genuinely are `model:q4_K_M`-shaped (uncommon), or (b) a rung-label
  -> real-tag resolution layer between `ModelLadder` and whatever
  builds the final Ollama tag. Either is a real design decision, not a
  constant swap -- flagged here rather than forced through with an
  incorrect quick patch that would silently fail on `ollama pull` in
  production.

## [Unreleased] -- Reference-model fix verified live + a real memory-tracking bug found

Once `llama3.2:3b` finished pulling on this machine, `HardwareProfiler.profile()`
was run for real against it for the first time in this project's
history: `local_capable: True`, `3.571 tok/s` measured on
`llama3.2:3b-instruct-q4_K_M`. Full numbers in
`docs/BENCHMARK_REPORT.md`'s Addendum 2.

That run surfaced one more real bug in `engine/execution/model_runner.py`
(Track A's file): `peak_memory_gb` always measured ~0.0. Root cause --
it used `psutil.Process(proc.pid).memory_info()` on the `ollama run`
CLI subprocess, but `ollama run` is a thin client; the model's real
memory footprint lives in the separate `ollama serve` server process
(confirmed via `ollama ps`). Fixed by polling Ollama's own `/api/ps`
HTTP endpoint on a background thread while generation runs, instead of
inspecting the wrong process's memory. Re-verified live:
`peak_memory_gb` now reports `0.534` for this model, a real, plausible
number. `psutil` is no longer imported in this file (still a
dependency of `hardware_profiler.py`, unaffected).

7 new tests cover `_query_loaded_model_bytes()`'s parsing logic
(mocked HTTP responses -- no live Ollama needed, consistent with
NFR-09's carve-out for this file being about the *runner itself*, not
logic that can be tested without one). 257 tests passing total.
