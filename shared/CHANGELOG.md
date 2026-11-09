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

## [Unreleased] -- Production model-tag mismatch resolved (`backend.py`)

Resolves the "Found, NOT resolved" item flagged above. `LocalExecutionBackend`
(`engine/execution/backend.py`) still defaulted `model_id="cascade-reference-3b"`
and built real Ollama tags as `f"{model_id}:{rung.label}"` -- e.g.
`"cascade-reference-3b:q4_K_M"`, which is not a real, pullable tag even
after pointing `model_id` at a real model, since `ModelLadder`'s rung
labels (`"q4_K_M"`/`"q5_K_M"`/`"q8_0"`) are bare abstract strings, not
real Ollama tag suffixes (real tags are one dash-joined string, e.g.
`llama3.2:3b-instruct-q4_K_M`).

Went with option (b) from the earlier entry -- a resolution layer,
not a constant swap:

- Added `_RUNG_LABEL_TO_REAL_TAG` (a `{model_id: {rung_label: real_tag_suffix}}`
  map) and `_resolve_real_tag(model_id, rung_label)` to `backend.py`.
  `ModelLadder` itself is untouched -- it stays generic/abstract, exactly
  as designed; the resolution lives entirely on the caller side, in the
  one place that actually needs a real tag.
- `_generate_with_stepdown()` now calls
  `_resolve_real_tag(self._model_id, rung.label)` before passing the
  result to `self._engine.load()`, instead of passing `rung.label`
  straight through unresolved.
- `model_id` default changed from `"cascade-reference-3b"` to
  `"llama3.2"`, with a registered mapping for its three default rungs.
- A model_id/label combo with no registered mapping falls back to the
  bare label unresolved -- this fails loudly and honestly at the
  `ollama pull` step for an unmapped production model (a clear "tag not
  found" error), rather than silently guessing a tag that might not
  exist. This also keeps every existing test that builds its own
  `ModelLadder` with synthetic labels (e.g. `"floor"`) working
  unchanged, since those never match an entry in the map.

Test fallout fixed in `test_backend_real.py`: three tests asserting on
the exact quant-level string passed to `FakeRunner.load()`
(`test_backend_selects_highest_viable_rung`,
`test_backend_steps_down_automatically_on_oom`,
`test_backend_raises_when_all_rungs_exhausted`) now expect the resolved
`"3b-instruct-*"` tags for the default `"llama3.2"` model_id. Two Layer-3
tests (`test_speculative_active_true_with_layer3`,
`test_nfr01_layer3_achieves_1_5x_over_layer1`) had `DraftPair` fixtures
targeting the old `"cascade-reference-3b"` model_id -- updated to
`"llama3.2"` so `AccelerationEngine._find_validated_pair()` still
matches. `scripts/benchmark_acceleration.py`'s `--model`/`--quant`
CLI defaults updated the same way (same tag-shape bug, different call
site, not covered by pytest since it's a live-only manual script).

Verified live end-to-end, not just unit-tested: with a synthetic
HardwareProfile forcing rung `q4_K_M`,
`LocalExecutionBackend(model_id="llama3.2").generate(...)` built and
pulled the real tag `llama3.2:3b-instruct-q4_K_M` (already present
locally from the earlier verification run) and returned real generated
text (`"Hello there friend."`, `1.685 tok/s`). 257 tests passing
(3 test-fallout fixes, no new tests needed -- this is a resolution
layer over existing, already-tested load-call plumbing).

## [0.3.1] -- 2026-11-06 -- fixes from the first live end-to-end benchmark

A 67-request run through the real pipeline (llama3.2:3b on CPU) found wrong
and stale cache answers, essay-length answers to one-line questions, and a
benchmark that measured looser settings than the CLI uses.

### Changed (Pipeline seam)

- `Pipeline.__init__` gains `system_prompt: Optional[str] = None` (prepended
  as a system message when the conversation has none) and
  `uncached_tiers: tuple[str, ...] = ("self_hosted",)`.
- The router now runs **before** the cache. Queries it sends to an
  uncached tier are neither served from nor written to the cache. Rationale:
  the self_hosted label means "needs live data or tools", so a cached answer
  is stale by definition ("who is the current prime minister" was cached
  and replayed). Costs one embedding per cache hit.

### Fixed

- **Wrong cache answers.** "...boiling point of water in celsius" was
  served the stored Fahrenheit answer: the embeddings score the two at 0.925.
  `SemanticCache` now applies a term-swap guard (`CacheConfig.term_swap_guard`,
  `"strict"` by default): a match is rejected when the new question swaps in a
  term the stored answer never mentions, or has the same key terms reordered.
  Stop words come from scikit-learn; no hand-written word lists. On the new
  19-pair near-miss fixture: 4/19 wrong answers before, 0/19 after (9 of the
  pairs were written after the guard). Cost: reworded questions answered
  from the cache fell from 71% to 43% on the paraphrase fixture, because
  rewordings that swap a verb ("wrote" -> "author") now get a fresh answer.
- Time-sensitive and personal patterns are now `CacheConfig` fields, and the
  defaults catch "current"/"today" on their own.
- **Long answers.** The CLI sends `DEFAULT_SYSTEM_PROMPT` (configurable,
  `""` disables); `local.max_tokens` is configurable.
- `cascade ask`/`chat` say when the one-time hardware check is about to run.
- `scripts/run_benchmark.py` used a 0.85 threshold while the CLI uses 0.92,
  overstating the paraphrase hit rate (93% vs 71% live). It now uses
  `CacheConfig()`, real recorded answers
  (`scripts/record_benchmark_answers.py`), reports wrong answers, sweeps
  threshold x guard mode, and rewrites only its own section of
  `docs/BENCHMARK_REPORT.md` instead of the whole file.

## [0.3.0] -- 2026-11-05 -- open-source release

Cascade is now an MIT-licensed open-source project with no paid hosted tier.

### Changed -- seam (`shared/types.py`)
- `PipelineResult.tier`: the literal `"cascade_cloud"` is renamed `"self_hosted"`. The router's
  label space, `Pipeline(cascade_backend=...)` (now `self_hosted_backend=`), `ShadowLabeler`,
  retraining and the benchmark fixtures follow. The tier's meaning is unchanged: a remote server
  the router sends live-data/tool queries to and the confidence check escalates to.

### Removed -- Track C (`gateway/`)
- `gateway/billing/` (Stripe client, `UsageMeter`, `SpendCapExceededError`), the `stripe`
  dependency, all per-token prices, the `cascade.cost_usd` response field and client-side
  spend metering.

### Changed -- Track C
- `CascadeCloudBackend` -> `SelfHostedBackend` (`gateway/backend.py`); `est_cost_usd` is 0.0.
- Server store: `accounts(cap_usd, spent_usd, stripe_customer_id)` -> `users(quota_tokens,
  used_tokens)`; `QuotaExceededError`; over-quota requests get HTTP 429 (was 402);
  `/v1/usage` reports tokens. New `reset_usage()`; `revoke_api_key()` returns whether a key was revoked.
- `gateway.serve`: `create-account --cap` -> `create-user --quota`; new `revoke-key`, `usage`,
  `reset-usage`. `CASCADE_GATEWAY_DB` -> `CASCADE_SERVER_DB` (default `~/.cascade/server.db`);
  `CASCADE_PRICE_PER_1K` and `STRIPE_API_KEY` removed.
- CLI config: `cascade_cloud.url` -> `self_hosted.url`; key env default `CASCADE_API_KEY` ->
  `CASCADE_SERVER_KEY`.

### Added
- `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`, package metadata in `pyproject.toml`.
- Track hand-off notes and the business/competitive planning docs moved to `docs/archive/`.

## [0.2.0] -- 2026-11-05 -- completion pass (solo; `shared/types.py` unchanged)

No change to either seam or to any `shared/types.py` type. Changes behind them:

### Changed -- Track A (`engine/execution/`)
- **Removed fabricated speedups.** `AccelerationEngine` no longer divides measured time by
  `LAYER*_SPEEDUP_FACTOR` constants (deleted). A layer is active only if the runner accepted it
  (`AcceleratedRunner.set_acceleration`) and reports in its `GenerationSample.acceleration` that
  it ran; acceptance rates are measured, not constants. `GenerationSample` gained
  `acceleration` and `acceptance_rate` fields (defaults keep every existing fake valid).
- **New `llamacpp_runner.py`**: real prompt-lookup (Layer 1) and draft-model (Layer 3)
  speculative decoding via llama-cpp-python, reusing Ollama's GGUF files. Optional dependency.
- **`OllamaReferenceModelRunner` rewritten on Ollama's HTTP API**: real `eval_count`/`eval_duration`,
  model kept resident, pull only when missing, unload via `keep_alive: 0`, `127.0.0.1` default.
- `LocalExecutionBackend` keeps the model loaded between requests, names the underlying error
  when every rung fails, and defaults to **no acceleration** -- measured slower on CPU
  (docs/BENCHMARK_REPORT.md Addendum 3: prompt lookup 0.60x, 1B draft 0.89x; NFR-01 not met).
- `scripts/validate_draft_pairs.py` and `scripts/benchmark_acceleration.py` now measure real
  acceptance and speed with LlamaCppRunner.

### Changed -- Track B (`engine/intelligence/pipeline.py`)
- A failing remote tier (BYOK or CascadeCloud: network, bad key, spend cap) now degrades to
  local instead of failing the request; a failed confidence escalation keeps the local answer
  (NFR-03).

### Changed -- Track C (`gateway/`, `cli/`)
- `CloudInferenceProvider` takes real chat messages (no `USER:` flattening) and streaming
  returns real usage (`stream_options.include_usage`).
- **Gateway is OpenAI-compatible** -- the old response shape could not be parsed by
  `CascadeCloudBackend`, so client and server could never talk. Now tested together.
- **API-key authentication** replaces the self-asserted `x-cascade-user` header. Keys stored as
  SHA-256 hashes in a new SQLite `GatewayStore`, which also persists spend and caps atomically.
- `gateway/serve.py`: `create-account`, `create-key`, `run`, configured by environment variables.
- New `gateway/byok.py`: `CloudBYOKBackend`, the missing `cloud_byok` tier.
- CLI: cloud tiers exist only when configured (no default `api.cascade.dev` endpoint); keys
  come from environment variables and `config set` refuses to store them; new `doctor` and
  `config` commands, `ask --stream`; `cascade` console script in `pyproject.toml`.

### Decisions recorded
- LiteLLM was not added: one OpenAI-compatible client covers the same providers with no extra dependency.
- BYOK lives in `gateway/` alongside the other OpenAI-compatible client code.

## [0.3.2] -- 2026-11-07 -- speculative decoding verified lossless and tuned

No seam changes (`Pipeline`, `ExecutionBackend` and the shared dataclasses are unchanged).

### Fixed

- **Output depended on the previous request.** llama-cpp-python reused the KV cache of the
  previous prompt's matching prefix. `LlamaCppRunner.generate` now resets state on every call,
  formats prompts with the model's own chat template, uses llama-cpp-python's chat sampling
  defaults, and stops on any end-of-generation token.

### Added

- `LlamaCppRunner.verify_greedy()` -> `GreedyCheck(tokens, exact, within_noise, violations,
  noise_tolerance)`. It recomputes each output token one at a time and checks it against the
  floating-point noise measured between batched and one-at-a-time evaluation.
  `scripts/benchmark_acceleration.py` checks every output and prints PASS/FAIL.
- Adaptive draft length (`LlamaCppRunner(adaptive_draft=..., max_draft_tokens=...)`), using the
  HF Transformers heuristic schedule. CLI config keys: `draft.adaptive` (default true),
  `draft.tokens`, `draft.max_tokens`.
- `benchmark_acceleration.py` options: `--draft-tokens`, `--lookup-tokens`, and `--modes`
  (e.g. `none,draft_model:k=8,draft_model:adaptive`).

### Measured (CPU-only laptop, BENCHMARK_REPORT Addendum 6)

- qwen2.5-coder 7B + 0.5B draft, adaptive: 1.10x decode overall, 1.29x on grounded prompts;
  fixed k=4 reached 1.35x on grounded prompts in one run. Greedy check 0 violations everywhere.
  NFR-01 (1.5x) not met on CPU.

## [Unreleased] -- 2026-11-07 -- hardware profile records the measured model

### Changed (`shared/types.py`)

- `HardwareProfile` gains an optional `reference_model_id: str`, the model the benchmark ran on.
  `HARDWARE_PROFILE_SCHEMA_VERSION` stays 1. The field is optional and additive, so a bump would
  only throw away every cached profile. Profiles without it are read as measured on `llama3.2`,
  which is what every earlier version measured.
- `HardwareProfiler.load_cached_profile()` returns None for a profile measured on a different
  model than the profiler's `reference_model_id`, so changing `model` triggers re-profiling.
  `LocalExecutionBackend` and the CLI (`init`, `profile`, `doctor`, the first-run warning) now
  profile the configured model at its `_resolve_real_tag` tags, not always llama3.2.
- The profiler benchmarks only the quant levels already present locally (when the runner can
  list them). If none are present, it tries just the first level, so profiling downloads at
  most one model.

## [0.4.0] -- 2026-11-09 -- small-model routing, background model server; 2.19x per session

No seam changes. `Pipeline` gains an optional `small_local_backend` keyword and a `warm()` method;
`PipelineResult` is unchanged (small-model answers report tier "local").

### Added

- `Pipeline(small_local_backend=...)`: queries the router labels "local" go to a small fast model;
  weak answers (estimate_local_confidence below the floor) and cloud_byok-labelled queries without a
  cloud backend go to the main local model. CLI key `small_model` (recommended
  `qwen2.5:1.5b-instruct`), sharing the main model's hardware profile.
- Model ids with an explicit size, e.g. `qwen2.5:1.5b-instruct`.
- Background model server (`cli/warm_server.py`): `cascade ask` starts it on first use (default for
  the llama.cpp runner, `server.auto`), keeps models loaded for `server.idle_minutes` (10), binds
  127.0.0.1 only. `cascade serve`, `cascade stop`.
- `scripts/benchmark_system.py`: 34-question session through the real Pipeline, answers checked.
  1.5B small model + cache + router + 7B with confidence drafting: 2.19x faster than plain 7B,
  median answer 9.6 s -> 2.0 s, 34/34 correct (BENCHMARK_REPORT Addendum 11).

### Fixed

- `AccelerationEngine.load()` loaded a llama.cpp model twice (plain, then rebuilt for speculative
  mode), holding both copies: peak 9.5 GB -> 5.9 GB, load 149 s -> 68 s.
- The acceptance monitor stepped confidence drafting down to prompt lookup (0.60x on CPU) after 20
  requests; self-regulating runners (`LlamaCppRunner.self_regulating`) are no longer stepped down.
- `LlamaCppRunner` frees the old model before rebuilding.

## [0.3.6] -- 2026-11-09 -- fix memory leak from the 0.3.5 windowed eval

No seam changes.

### Fixed

- The 0.3.5 windowed eval kept a strong reference from the model's `eval` back to the model, so
  every replaced model (a reload in `set_acceleration` or `verify_greedy`, or a discarded runner)
  stayed resident until a garbage-collector pass: 9.4 GB after a second 7B load instead of 5.4 GB.
  It now holds a weak reference.
- Confirmed 3-round speed with the windowed eval: 1.69x grounded, 1.08x open, 1.39x overall,
  lossless (`docs/benchmark/acceleration_qwen7b_windowed_3rounds.json`).

## [0.3.5] -- 2026-11-08 -- speculative mode: fix crash past 512 tokens; less RAM

No seam changes.

### Fixed

- Speculative decoding (draft model or prompt lookup) crashed with `ValueError: could not
  broadcast` once prompt + output passed 512 tokens: llama-cpp-python copies every position's
  logits into a buffer of only `n_batch` rows. Speculative-mode models now use an eval that asks
  llama.cpp for logits at the last 64 positions only and skips the copy; `verify_greedy` restores
  the stock eval while it runs. Draft lengths of 64 or more are rejected.

### Changed

- Same fix, measured: a 478-token prompt prefills 11% faster and leaves 0.7 GB less resident
  (6.3 -> 5.6 GB); short prompts 0.13 GB less, same decode speed (BENCHMARK_REPORT Addendum 10).
- `benchmark_acceleration.py` mode option `draft=FAMILY@TAG`. A general-purpose draft
  (qwen2.5:0.5b-instruct) ties the coding draft at 1.38x overall (Addendum 9).

## [0.3.4] -- 2026-11-07 -- confidence drafting; 1.33x overall, open prompts no longer slower

No seam changes.

### Added

- `ModelDraft(p_min=...)` / `LlamaCppRunner(draft_p_min=...)`: confidence drafting. The draft
  model stops at the first token whose top probability is below `p_min`; an empty draft gives the
  target a plain decoding step. The draft model's KV cache is reused for the shared prefix.
- `ModelDraft(max_pause=...)` / `LlamaCppRunner(max_draft_pause=...)`: skip drafting with
  exponential backoff after a rejected 1-token adaptive draft. Off by default (it was slower).
- CLI key `draft.p_min` (default 0.75; `off` restores adaptive drafting).
- `benchmark_acceleration.py` mode options combine: `draft_model:pmin=0.75:k=16`,
  `draft_model:adaptive:pause=0`.

### Changed

- CLI default draft mode: confidence drafting (p_min 0.75, 16-token cap) instead of adaptive.
  Same 3 interleaved rounds: overall 1.19x -> 1.33x, open-ended prompts 0.83x -> 1.01x,
  0 lossless-check violations (`docs/benchmark/acceleration_qwen7b_pmin_3rounds.json`).

## [0.3.3] -- 2026-11-07 -- RAM 7.2 -> 4.6 GB; 1.65x on grounded prompts

No seam changes.

### Changed

- `LlamaCppRunner(use_mmap=False)` is the new default. With mmap, the file pages stayed resident
  next to llama.cpp's repacked weights: qwen2.5-coder 7B resident RAM went from 7.2 GB to 4.6 GB,
  and a 5-token verification batch from 792 ms to 590 ms.

### Added

- `benchmark_acceleration.py`: `--repeats N` (modes interleaved within each round, with a
  per-round speed-up in the summary), `--mmap`, and per-row `rss_gb` with a per-mode `peak_rss_gb`.

### Measured (BENCHMARK_REPORT Addendum 7)

- qwen2.5-coder 7B + 0.5B draft, adaptive, 3 rounds: 1.65x decode on grounded prompts (each round
  at least 1.60x), 1.25x overall, 0.85x on open prompts; greedy check 0 violations.
