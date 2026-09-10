# Track B Handoff — What Track C Needs to Know

**From:** Track B (Intelligence Layer)
**To:** Track C (Platform & Revenue)
**Date:** 11 September 2026 — Tasks 1–16 complete (Tasks 17–18 blocked on this handoff)
**Repo path:** `engine/intelligence/`

> **You do NOT need to read Track B's source code.** This document
> tells you everything: what exists, what the contract is, and exactly
> where to plug in when your part is ready.

---

## 1. What Track B Built — Plain English

Cascade needs to decide **what to do with each query** before it needs
to decide **who answers it**. Track B owns that decision end to end.

```
┌────────────────────────────────────────────────────────────────────┐
│                 TRACK B  engine/intelligence/                       │
│                                                                      │
│  Step 1 — ContextCompressor (FR-04)                                 │
│    Scores every conversation turn's relevance to the current query  │
│    by embedding similarity, not position. Drops/summarizes the      │
│    irrelevant ones before anything else sees the message list.      │
│                                                                      │
│  Step 2 — SemanticCache (FR-06)                                     │
│    Two layers: exact-prefix (hash match) then fuzzy embedding       │
│    similarity. A hit returns immediately — no backend called at     │
│    all. Never caches time-sensitive/personalized queries.           │
│                                                                      │
│  Step 3 — SemanticRouter (FR-05)                                    │
│    A trained classifier (not a keyword router) picks one of:        │
│    local / cloud_byok / cascade_cloud, with a calibrated confidence. │
│                                                                      │
│  Step 4 — Execute                                                   │
│    Calls whichever ExecutionBackend the router picked. If local     │
│    raises LocalExecutionBackendError, escalates to cascade_backend  │
│    automatically.                                                   │
│                                                                      │
│  Step 5 — Confidence check (FR-05.3)                                │
│    A local answer that looks weak (hedging, too short) gets         │
│    silently re-run on cascade_backend before anything is returned.  │
│                                                                      │
│  Step 6 — Cache write + shadow labeling                             │
│    Writes cacheable answers back to SemanticCache. Optionally       │
│    samples a % of traffic for a background comparison call that     │
│    trains the next router version — zero effect on what's served.  │
│                                                                      │
│  Pipeline — THE ONE CLASS TRACK C CALLS.                            │
│  Public API never changes:                                          │
│    .run(messages)        → PipelineResult                           │
│    .run_stream(messages) → Iterator[str]                            │
└────────────────────────────────────────────────────────────────────┘
         │
         ▼
   PipelineResult  ← THIS is what Track C's CLI/gateway receives
   {
     "text":               "model response here",
     "tier":                "local",   ← or "cache" / "cloud_byok" / "cascade_cloud"
     "cache_hit":           False,
     "confidence":          0.9,
     "escalated":           False,
     "speculative_active":  True,
     "tok_s":                36.2,
     "latency_ms":           840.0,
     "est_cost_usd":         0.0
   }
```

---

## 2. Test Results — 182 Tests, 92% Coverage, No GPU/Network Needed

Every test runs with **no live GPU, no Ollama, no model download**.

```
pytest engine/ -v --cov=engine --cov-report=term

182 collected

test_context_compressor.py          (11 tests)  ........ PASSED
test_coldstart_data.py               ( 6 tests)  ........ PASSED
test_classifier.py                   ( 7 tests)  ........ PASSED
test_confidence.py                   ( 6 tests)  ........ PASSED
test_cache.py                        (16 tests)  ........ PASSED
test_shadow_labeling.py              ( 6 tests)  ........ PASSED
test_retrain.py                      ( 6 tests)  ........ PASSED
test_pipeline.py                     (10 tests)  ........ PASSED
test_pipeline_integration.py         (13 tests)  ........ PASSED
[+ all of Track A's 101 tests, unaffected]        ........ PASSED

Coverage (engine/intelligence/):
  context_compressor.py    100%   relevance scoring, pure function
  classifier.py             100%   trained router, not hardcoded
  confidence.py              100%   heuristic + logprobs-ready path
  shadow_labeling.py         100%   sampling + logging
  cache.py                    94%   two-layer cache
  coldstart_data.py           94%   RouteLLM-shaped loader + bootstrap
  pipeline.py                  92%   the orchestrator itself
  retrain.py                   90%
  embeddings.py                63%   FastEmbedEmbedder's real ONNX path is
                                      intentionally untested here — same
                                      reasoning as Track A's own
                                      OllamaReferenceModelRunner (49%)

182 passed in ~7s
```

**Performance target verified on this machine directly:** SemanticCache's
naive design (re-query SQLite on every lookup) measured 70–93ms at
10,000 entries — almost entirely SQLite row-materialization cost, not
the similarity math (3–7ms alone). Fixed with an in-memory index
mirrored from SQLite; now comfortably clears the <50ms p95 target
(FR-06.3) on this same hardware.

---

## 3. Architecture Flow — How a Request Moves Through Track B

```
User sends a message
         │
         ▼
Pipeline.run(messages)
         │
         ├─► ContextCompressor.compress(messages)
         │       relevance-ranked, not position-based
         │
         ├─► SemanticCache.lookup(compressed)
         │       hit → return immediately, tier="cache", no backend call
         │
         ├─► SemanticRouter.classify(query)
         │       → "local" / "cloud_byok" / "cascade_cloud" + confidence
         │
         ├─► Execute on the selected ExecutionBackend
         │       local raises LocalExecutionBackendError
         │           → auto-escalate to cascade_backend (if configured)
         │       requested backend not configured (e.g. no cloud_byok_backend)
         │           → falls back to local, does not crash
         │
         ├─► estimate_local_confidence(answer_text)
         │       low confidence on a "local" answer + cascade_backend
         │       configured → silently re-run on cascade_backend
         │
         ├─► SemanticCache.write(compressed, final_text)
         │       skipped automatically if the query wasn't cacheable
         │
         ├─► ShadowLabeler.maybe_label(...)  (if configured)
         │       zero effect on the response already finalized above
         │
         └─► returns PipelineResult to Track C
```

---

## 4. Track C Contract — Exactly What You Need to Do

### Step 1 — Implement CascadeCloudBackend (same interface Track A used)

```python
# gateway/backend.py (Track C writes this)
from shared.types import ExecutionBackend, BackendResult, Message
from typing import Iterator

class CascadeCloudBackend:
    def generate(self, messages: list[Message]) -> BackendResult:
        # call your hosted vLLM / SGLang endpoint
        return {
            "text": response_text,
            "confidence": 0.95,
            "speculative_active": False,   # no speculative layer in cloud
            "tok_s": tok_s,
            "latency_ms": elapsed_ms,
            "est_cost_usd": cost,          # you set the real cost here
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        for chunk in your_hosted_stream(messages):
            yield chunk

# REQUIRED: same import-time guard Track A uses
assert isinstance(CascadeCloudBackend(), ExecutionBackend)
```

### Step 2 — Construct Pipeline with everything wired in

This is the **only** place all three tracks' code actually meets. Track
B's `Pipeline` already knows how to call whatever you pass in here —
nothing in `engine/intelligence/` needs to change.

```python
# cli/main.py or gateway/api/app.py (Track C writes this)
from engine.execution.backend import LocalExecutionBackend
from engine.intelligence.pipeline import Pipeline
from engine.intelligence.cache import SemanticCache
from engine.intelligence.context_compressor import ContextCompressor
from engine.intelligence.embeddings import FastEmbedEmbedder
from engine.intelligence.router.classifier import SemanticRouterClassifier
from engine.intelligence.router.shadow_labeling import ShadowLabeler
from gateway.backend import CascadeCloudBackend

embedder = FastEmbedEmbedder()
cascade_backend = CascadeCloudBackend()

pipeline = Pipeline(
    backend=LocalExecutionBackend(),          # Track A
    cascade_backend=cascade_backend,           # Track C, this file
    compressor=ContextCompressor(embedder),    # Track B
    cache=SemanticCache(embedder),             # Track B
    router=SemanticRouterClassifier(embedder), # Track B
    shadow_labeler=ShadowLabeler(cascade_backend, sample_rate=0.05),
)

result = pipeline.run(messages)   # -> PipelineResult
for token in pipeline.run_stream(messages):   # -> streaming
    ...
```

`cloud_byok_backend` (a LiteLLM-backed bring-your-own-key call) is also
an accepted optional constructor argument, but **nobody has built it
yet** — see Section 5, item 3. `Pipeline` falls back to `local` cleanly
if you don't pass one.

### Step 3 — Verify J-4 (the first real three-way run) once your backend exists

There is no automated test file for this yet (unlike J-2's
`j2_integration_test.py`) because it genuinely needs your real
`CascadeCloudBackend`, not a fake. Once it exists:

```python
# Manual smoke test — confirm all three tracks' code chains correctly
from engine.execution.hardware_profiler import HardwareProfiler, OllamaReferenceModelRunner

profiler = HardwareProfiler(OllamaReferenceModelRunner())
profile = profiler.profile()
print(f"local_capable: {profile['local_capable']}")

# If local_capable is False, the pipeline above should route straight
# to cascade_cloud via SemanticRouter or the LocalExecutionBackendError
# escalation path — confirm result["tier"] == "cascade_cloud" and
# result["est_cost_usd"] > 0.0 in that case.
```

Track B's side of this is already tested against fakes shaped exactly
like `CascadeCloudBackend` (see `test_pipeline_integration.py`'s
`test_router_directs_tool_query_to_cascade_cloud` and
`test_low_confidence_local_answer_escalates_to_cascade`) — that's the
ceiling of what one track can verify alone.

---

## 5. Fakes and Placeholders Currently In Use

| # | Fake/Placeholder | Where | What it does | How to replace |
|---|---|---|---|---|
| 1 | `FakeEmbedder` (bag-of-words hash) | All `test_*.py` | Deterministic, no model download | **Keep forever** — tests must never need a network call (NFR-09) |
| 2 | `BOOTSTRAP_EXAMPLES` (28 hand-labeled queries) | `router/coldstart_data.py` | Real RouteLLM dataset needs network access this environment doesn't have | Point `load_coldstart_dataset(dataset_path=...)` at a downloaded RouteLLM `.jsonl`/`.csv` export — no code change needed beyond that |
| 3 | `cloud_byok_backend` unimplemented | `pipeline.py` | `Pipeline` accepts and dispatches to it correctly, but no LiteLLM wrapper exists | Joint decision needed: does Track B or Track C own this file? Not blocking either track today |
| 4 | Heuristic-only local confidence | `router/confidence.py` | No `ExecutionBackend` exposes logprobs/entropy yet | If your `CascadeCloudBackend` (or a future Track A revision) can expose per-token logprobs, that's a `shared/types.py` change needing all-three sign-off — raise it in `shared/CHANGELOG.md` first |
| 5 | No J-4 automated test yet | (would live in `engine/intelligence/tests/` or a new `tests/integration/`) | Can't be written until `CascadeCloudBackend` is real | Once your backend lands, write it mirroring `j2_integration_test.py`'s structure — happy to help wire it up |

---

## 6. Files Quick Reference

### For Track C

| File | What it gives you | Action |
|---|---|---|
| `shared/types.py` | `ExecutionBackend` protocol your `CascadeCloudBackend` must implement | Import from here |
| `engine/intelligence/pipeline.py` | `Pipeline` — construct it with your backend wired in (Section 4, Step 2) | Import `Pipeline`, never its internals |
| `engine/intelligence/embeddings.py` | `FastEmbedEmbedder` — construct once, share across `ContextCompressor`/`SemanticCache`/`SemanticRouter` | Import `FastEmbedEmbedder` |
| `docs/TRACK_B_PHASE2_3_SUMMARY.md` | Full task-by-task detail on everything Track B built | Reference when you need the "why" |
| `shared/CHANGELOG.md` | Open decisions relevant to you (cloud_byok ownership, the confidence/logprobs gap) | Review, add your J-1 sign-off entry — still outstanding |

---

## 7. Run Track B's Tests Yourself

```bash
# 1. Install dependencies (Track A's + Track B's)
pip install -r requirements.txt

# 2. Run all of engine/ (Track A + Track B, 182 tests, no GPU/Ollama/network)
pytest engine/ -v

# Expected: 182 passed in ~7s

# 3. Just Track B's intelligence layer:
pytest engine/intelligence/ -v

# 4. Coverage report:
pytest engine/ --cov=engine --cov-report=term
# Expected: TOTAL 92% (target was >= 70%, NFR-09)

# 5. Confirm J-2 still holds (Track A's + Track B's combined seam):
pytest engine/execution/j2_integration_test.py -v
# Expected: 5 passed
```

---

## Summary — What's Real vs What's Pending

| Status | Item |
|---|---|
| ✅ DONE | ContextCompressor: relevance-ranked, pure function (FR-04) |
| ✅ DONE | SemanticRouter: trained classifier, cold-started, not hardcoded (FR-05.1-05.2) |
| ✅ DONE | Local-tier confidence estimation + escalation (FR-05.3) |
| ✅ DONE | SemanticCache: two layers, <50ms p95 at 10k entries verified (FR-06) |
| ✅ DONE | Shadow-labeling pipeline (FR-05.4) |
| ✅ DONE | Full Pipeline integration: compressor → cache → router → execute → confidence → write → shadow-label |
| ✅ DONE | Retraining job combining cold-start + shadow data (FR-05.5) |
| ✅ DONE | 182 tests, 92% coverage, J-2 still green |
| ⏳ PENDING | Real RouteLLM dataset — bootstrap placeholder in use, joint team decision on data source |
| ⏳ PENDING | Cloud-BYOK (LiteLLM) implementation — ownership undecided |
| ⏳ PENDING | Real token logprobs from any backend — needs a `shared/types.py` change |
| 🔲 BLOCKED ON TRACK C | Task 17 — the real J-4 three-way run, needs `gateway/backend.py` |
| 🔲 BLOCKED ON TRACK C | Task 18 — real benchmark numbers, needs the Section 12 harness |
| 🔲 J-1 NEEDED | Track C's sign-off on `shared/types.py` — Track A and Track B have both signed off |
