# Track C — Platform & Revenue — Complete

Owner: Track C (built by the project lead + AI pairing, following
Track B's handoff doc). Scope: the Individual Task Breakdown's Track C
tasks 1–17 (all phases). Requirements covered: FR-07.1–07.7, FR-08.1–08.3.

## What was built

| File | Purpose |
|---|---|
| `gateway/backend.py` | `CascadeCloudBackend` — conforms to `ExecutionBackend` exactly like Track A's `LocalExecutionBackend` (FR-07.1). |
| `gateway/providers/base.py`, `http_provider.py` | `CloudInferenceProvider` protocol + `HTTPInferenceProvider` — one OpenAI-compatible HTTP client works against any wholesale GPU provider by config alone (FR-07.4). |
| `gateway/billing/meter.py` | `UsageMeter` — atomic, thread-safe spend-cap enforcement (FR-07.2). |
| `gateway/billing/stripe_client.py` | `StripeBillingClient` — metered usage + subscriptions (FR-07.5). |
| `gateway/api/app.py` | FastAPI gateway — the real, server-side billing enforcement point (FR-07.2, FR-07.5, FR-07.6/07.7 by way of whatever engine serves behind it). |
| `cli/main.py` | `init`, `ask`, `chat`, `stats`, `profile` commands (FR-08.1–08.3), wiring the full `Pipeline` per Track B's handoff doc. |
| `engine/intelligence/decision_log.py` | **Gap found and closed** — a per-request decision log `cascade stats` needed that Track B's Pipeline never wrote (see below). |
| `tests/integration/test_j4_three_way_run.py` | **The real J-4 run** — all three tracks' actual classes wired together, closing Track B's previously-blocked Task 17. |

## Task-by-task definition-of-done verification

**Tasks 1–4 (Phase 1).** CLI skeleton, config schema (JSON at
`~/.cascade/config.json`), repo/CI scaffolding (already in place from
Track A), and the `CascadeCloudBackend` stub — done directly as the
real implementation rather than a separate stub-then-real pass, since
Track B's `Pipeline` was already built and waiting for a real backend.

**Tasks 5–7 (Phase 2).** FastAPI gateway (`gateway/api/app.py`),
provider adapter interface (`gateway/providers/base.py`), and the real
`CascadeCloudBackend` — verified: 9 tests on the backend, 4 on the HTTP
provider (against `httpx.MockTransport`, exercising real request/
response/SSE-streaming parsing with zero network calls), 8 on the
FastAPI app (via `TestClient`).

**Tasks 8–9 (Phase 3).** Stripe integration and spend-cap enforcement.
**Real bug found and fixed here:** the first `UsageMeter` design split
"check the cap" and "record the spend" across two separate lock
acquisitions — thread-safe individually, but not atomic together,
which reopens exactly the race a spend cap exists to prevent (two
concurrent requests could both pass the check against the same stale
total before either commits). Fixed by making `check_and_reserve()`
atomically check-and-reserve in one lock acquisition, with a separate
`adjust()` to reconcile the estimate against the real cost afterward.
Verified with a 20-thread concurrent test asserting the cap is
respected *exactly* (10 of 20 $10 requests fit a $100 cap, not "at
most 10" — atomicity makes it exact).

**Task 10–11 (Phase 4, J-4).** The full CascadeCloud gateway live
end-to-end, and verifying FR-07.3 (compression/cache run ahead of
CascadeCloud calls too). **This is the task that was blocked in
Track B's phase summary** — it needed `CascadeCloudBackend` to exist.
`tests/integration/test_j4_three_way_run.py` now wires Track A's real
`LocalExecutionBackend`, Track B's real `Pipeline`, and Track C's real
`CascadeCloudBackend` together and verifies: a `local_capable` machine
never touches CascadeCloud; a machine where every rung OOMs escalates
automatically with a real, non-zero cost from Track C's cost model; a
router-classified tool-required query routes straight to CascadeCloud
without attempting local at all; and a spend cap on the CascadeCloud
side is respected even when Pipeline is the one triggering the call.

**Tasks 12–14 (Phase 5).** CLI polish (streaming-aware status footer,
though see Known Limitations below), `cascade init` (hardware
profiling + config write), `cascade stats` (see the decision-log gap
below). Verified via `typer.testing.CliRunner` with `HardwareProfiler`
and `Pipeline` construction monkeypatched out — 11 tests, no live
Ollama/network needed.

**Tasks 15–17.** Quickstart docs are this document plus
`docs/TRACK_B_HANDOFF_FOR_C.md`'s Section 4 (already written by Track
B, still accurate). Benchmark report (Task 16) is still genuinely
blocked — the Section 12 harness (`scripts/run_benchmark.py`, the
200-query reference suite) doesn't exist yet; no fabricated numbers
here either, consistent with the rest of this repo's documentation.
Public launch (Task 17 in Track C's own numbering — distinct from
Track B's Task 17/J-4 above) hasn't happened; that's a business
decision for the project lead, not a code task.

## A gap found and closed: the decision log

Building `cascade stats` (FR-08.3) surfaced a real hole: Track B's
`Pipeline` had a *sampled* shadow-labeling log
(`router/shadow_labeling.py`) but no log of *every* request's outcome
— which is what `cascade stats`'s cache-hit-rate/local-resolution-rate/
speculative-speedup numbers need to aggregate over. Added
`engine/intelligence/decision_log.py` and one new optional
`decision_log_path` constructor parameter on `Pipeline` (off by default,
per NFR-05's privacy stance — matching Track B's own established
pattern of nothing being logged unless explicitly configured). This is
logged here rather than silently folded in, since it's technically a
Track B file changed by Track C's work; `shared/CHANGELOG.md` has the
formal entry.

## A doc bug found and fixed: Track A's handoff example

`docs/TRACK_A_HANDOFF_FOR_B_AND_C.md` Section 7's Track C code example
reads:
```python
from engine.execution.hardware_profiler import HardwareProfiler, OllamaReferenceModelRunner
```
`OllamaReferenceModelRunner` actually lives in
`engine/execution/model_runner.py`, not re-exported from
`hardware_profiler.py` — this import fails as written. `cli/main.py`
uses the corrected two-line import. Worth Track A updating the handoff
doc's example to match (not done unilaterally here, since it's their
file) — flagged in `shared/CHANGELOG.md`.

## Test results

```
pytest -v --cov=engine --cov=gateway --cov=cli --cov-report=term
242 passed in ~10s

Coverage (gateway/, cli/):
  gateway/backend.py              100%
  gateway/api/app.py               100%
  gateway/billing/meter.py          100%
  gateway/billing/stripe_client.py  100%
  gateway/providers/base.py          100%
  gateway/providers/http_provider.py  94%
  cli/main.py                          85%

Overall repo coverage: 92% (target was >= 70%, NFR-09)
```

Every test above runs with no live GPU, no Ollama, no real network
call, no real Stripe API key — `httpx.MockTransport` for the HTTP
provider, a monkeypatched `stripe` module for billing, `FastAPI`'s
`TestClient` for the gateway, and `typer.testing.CliRunner` with
`Pipeline`/`HardwareProfiler` construction faked out for the CLI.

## Assumptions and decisions flagged for review

1. **One `HTTPInferenceProvider` class for every wholesale GPU
   provider**, rather than a separate adapter per vendor (RunPod,
   Modal, Fireworks). All of them serve an OpenAI-compatible
   chat-completions API when running vLLM/SGLang behind them, so
   swapping providers is a `base_url`/`api_key`/`model_id` config
   change against the same class — a stronger form of FR-07.4 than a
   new adapter per vendor would be. Flagged, not unilaterally final,
   since it assumes every future wholesale provider keeps serving that
   API shape.

2. **`DEFAULT_PRICE_PER_1K_TOKENS_USD = 0.50`** is a deliberately
   round placeholder, exactly like Track A's `DEFAULT_REFERENCE_MODEL_ID`
   — the real number depends on which wholesale provider and model
   Cascade actually contracts, a business decision
   (`docs/planning/BUSINESS_BRIEF.md`), not something to hardcode here.

3. **Billing enforcement lives server-side in `gateway/api/app.py`,
   not client-side in `gateway/backend.py`** — a client-side spend-cap
   check is trivially bypassable by a modified client. `CascadeCloudBackend`
   still has its own `UsageMeter` (useful when embedded directly, e.g.
   in `tests/integration/test_j4_three_way_run.py`, or for a
   self-hosted deployment where the "client" and "server" are the same
   trust boundary), but the gateway app's own per-user meters are what
   actually protect Cascade's margin in a real multi-tenant deployment.

4. **Streaming requests don't reconcile to a real token count** —
   `CascadeCloudBackend.generate_stream()`'s spend reservation stands
   as the final charge, since the streaming loop doesn't get an exact
   usage payload back the way a non-streaming call's response does.
   Flagged as a follow-up once a provider's streaming usage reporting
   is wired in (most OpenAI-compatible APIs support `stream_options:
   {"include_usage": true}` for exactly this).

5. **`UsageMeter`'s per-user registry in `gateway/api/app.py` is
   in-memory**, not backed by a database — correct for a single-process
   deployment, and documented in the code as the seam a persistent-store
   swap plugs into without changing any route.

## What's explicitly NOT done, and why

- **Real benchmark numbers (Track C's own Task 16)** — the Section 12
  harness doesn't exist. Same standard the rest of this repo holds to.
- **A real wholesale GPU provider contract** — `HTTPInferenceProvider`
  is real, tested code, but has never been pointed at an actual vLLM/
  SGLang deployment; that's an infrastructure/business decision, not a
  code gap.
- **Public launch** — a business/marketing action for the project lead.

## Next action needed from the team (not code)

**Track C's J-1 sign-off** — now recorded in `shared/CHANGELOG.md`.
All three tracks have signed off; `shared/types.py` is fully frozen.
