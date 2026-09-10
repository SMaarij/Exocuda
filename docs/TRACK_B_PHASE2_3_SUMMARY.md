# Track B — Intelligence Layer — Phases 1-3 (Tasks 1-16) — Complete

Owner: Track B. Scope: the Individual Task Breakdown's Track B tasks
1-16 (Phases 1-3). Requirements covered: FR-04.1–04.3, FR-05.1–05.5,
FR-06.1–06.4.

## What was built

| File | Purpose |
|---|---|
| `engine/intelligence/pipeline.py` | `Pipeline` — the Section 9 orchestrator seam. Wires ContextCompressor → SemanticCache → SemanticRouter → execution → confidence check → cache write → shadow labeling. |
| `engine/intelligence/context_compressor.py` | `ContextCompressor` — relevance-ranked context compression (FR-04). |
| `engine/intelligence/embeddings.py` | `Embedder` protocol, `FastEmbedEmbedder` (production), `cosine_similarity` — shared by all three components below. |
| `engine/intelligence/cache.py` | `SemanticCache` — two-layer cache: exact-prefix + fuzzy semantic (FR-06). |
| `engine/intelligence/router/coldstart_data.py` | RouteLLM-shaped cold-start dataset loader + bootstrap placeholder (FR-05.1). |
| `engine/intelligence/router/classifier.py` | `SemanticRouterClassifier` — trained logistic regression, not a keyword router (FR-05.1, FR-05.2). |
| `engine/intelligence/router/confidence.py` | `estimate_local_confidence()` — FR-05.3. |
| `engine/intelligence/router/shadow_labeling.py` | `ShadowLabeler` — sampled background comparison calls (FR-05.4). |
| `engine/intelligence/router/retrain.py` | `retrain()` — cold-start + shadow-label retraining (FR-05.5). |
| `engine/intelligence/tests/` | 96 tests across 8 files, all against fakes — no live GPU, no Ollama, no network/model download (NFR-09). |

## Task-by-task definition-of-done verification

**Task 1 — review and sign off `shared/types.py` (J-1).** Signed off;
see `shared/CHANGELOG.md`'s "J-1 sign-off (Track B)" entry. Schema is
now frozen pending Track C's sign-off.

**Task 2 — Pipeline skeleton (Phase 1 join point, J-2).** Constructor
matches Track A's documented single-argument call shape exactly.
Verified: `pytest engine/execution/j2_integration_test.py` — 5/5 green
with zero test-body changes, both before and after every later task
below was layered on top. `pipeline_bridge.py` deleted per Track A's
own instructions once J-2 passed.

**Tasks 3-5 — ContextCompressor (FR-04.1-04.3).** Turn relevance scored
by embedding similarity against the current query, not position; below-
threshold turns summarized or dropped; exposed as a pure function.
Verified: a relevant turn survives from 20 messages back while an
irrelevant recent turn is filtered (`test_compress_keeps_a_relevant_turn_from_far_earlier`,
`test_compress_drops_or_summarizes_irrelevant_recent_turn`).

**Tasks 6-9 — SemanticRouter (FR-05.1-05.3).** Cold-started on a
labeled dataset (`coldstart_data.py`) instead of a hardcoded ruleset —
the direct fix for the V1 heuristic-router complaint. Classifies into
local / cloud_byok / cascade_cloud with a calibrated `predict_proba`
confidence, not a made-up number.

**Task 10-13 — SemanticCache (FR-06.1-06.4).** Two layers: exact-prefix
(hash of the full message list) checked first, fuzzy embedding
similarity second. Cacheability is a rule-based safety filter by
design (never cache "what's today's date") — the router is learned,
the cache filter is deterministic, and that distinction is intentional,
not inconsistent. Performance target verified directly on this
machine: **p95 lookup latency well under the 50ms target at 10,000
entries**, after an architecture fix (see "Assumptions and decisions
flagged for review" below) — the naive design (re-querying SQLite on
every lookup) measured 70-90ms on this sandbox's 1-core hardware before
the fix.

**Task 14 — shadow-labeling pipeline (FR-05.4).** Samples a
configurable percentage of served traffic for a background comparison
call, writes a training-label log, never affects the served response
(verified: `test_maybe_label_never_raises_when_cascade_backend_fails`).

**Task 15 — full pipeline integration tests.** `test_pipeline_integration.py`,
13 tests: cache hit/miss, router-driven dispatch to all three tiers,
confidence-based escalation, compressor-before-cache-and-backend
ordering, shadow labeling, and one full-stack end-to-end scenario
combining all four components at once.

**Task 16 — retraining job (FR-05.5).** `retrain()` combines cold-start
data with accumulated shadow-label logs (weak supervision: agreement →
label as the tier served, disagreement → label as `cascade_cloud`);
callers' interface never changes.

## Test results

```
pytest engine/ -v --cov=engine --cov-report=term
182 passed in 7.59s
Coverage: 92% overall (target was >= 70%, NFR-09)
  context_compressor.py   100%
  classifier.py            100%
  confidence.py            100%
  shadow_labeling.py       100%
  cache.py                  94%
  coldstart_data.py         94%
  retrain.py                90%
  pipeline.py                92%
  embeddings.py              63%  -- FastEmbedEmbedder's real ONNX-loading
                                      path is intentionally untested here,
                                      same reasoning as Track A's own
                                      OllamaReferenceModelRunner (49%):
                                      NFR-09 requires tests never need a
                                      network/model download, not that
                                      100% of every line is exercised.
```

## Assumptions and decisions flagged for review

1. **`Pipeline`'s constructor takes a single required `backend`
   argument, with everything else optional.** This matches Track A's
   own documented example (`docs/TRACK_A_HANDOFF_FOR_B_AND_C.md`,
   Section 4a) and is what makes J-2 pass with zero test changes. Logged
   in `shared/CHANGELOG.md`.

2. **`Pipeline` imports `LocalExecutionBackendError` directly from
   `engine/execution/backend.py`.** This is Track A's own documented
   integration pattern, but is in tension with Section 10's "Track B
   never imports from engine/execution/" rule. Flagged, not fixed
   unilaterally — the clean fix (a shared `ExecutionBackendError` base
   class in `shared/types.py`) needs all three tracks' sign-off since
   it touches the frozen seam file.

3. **`SemanticCache`'s naive per-lookup SQLite re-query was replaced
   with an in-memory index, mirrored from SQLite.** The original design
   (query all rows on every `lookup()` call) measured 70-93ms at 10,000
   entries — nearly all SQLite row-materialization cost, not the
   similarity math (which measured 3-7ms alone). SQLite is now the
   durable store only; `lookup()`'s semantic layer reads an in-memory
   index built once at construction and updated incrementally on
   `write()`. This is an implementation detail entirely internal to
   `cache.py` — no change to `SemanticCache`'s public `lookup()`/`write()`
   API, so it needed no cross-track sign-off.

4. **`FR-05.3`'s "model-internal signals (logprobs/entropy)" branch is
   currently unreachable.** No `ExecutionBackend` implementation exposes
   logprobs today — `BackendResult.confidence` is documented (Track A's
   handoff, Section 1) as the *speculative-decoding acceptance rate*,
   a different signal. `estimate_local_confidence()` always takes the
   heuristic fallback branch right now; this is FR-05.3's own
   documented "only when it does not [expose those signals]" case, kept
   ready for when a backend does. Flagged in `shared/CHANGELOG.md`.

5. **RouteLLM's real public dataset was not used.** This environment has
   no network access to download it. `coldstart_data.py`'s
   `BOOTSTRAP_EXAMPLES` (28 hand-labeled examples across all three
   tiers) exercises the classifier's training/inference *pipeline*
   correctly — it is explicitly not a claim about real-world routing
   accuracy. Swapping in real data needs no code change, only a
   `dataset_path` pointing at a downloaded `.jsonl`/`.csv` export.

6. **Cloud-BYOK (LiteLLM) itself was not implemented.** `Pipeline`
   accepts an optional `cloud_byok_backend: ExecutionBackend` and
   dispatches to it correctly when configured, falling back to local
   otherwise (tested: `test_router_falls_back_to_local_when_requested_backend_missing`).
   Building the actual LiteLLM wrapper wasn't in Track B's task list and
   its ownership (Track B, since it's "just an SDK call," vs. Track C,
   since it's cloud/billing-adjacent) is an open question worth a quick
   team decision, not blocking anything above.

## What's explicitly NOT done, and why

- **Task 17 (wire real router into the first three-way end-to-end run)**
  is the Phase 4 join point (J-4) — it requires Track C's real
  `CascadeCloudBackend` to exist, which it doesn't yet (`gateway/` is
  empty). Track B's side is built and verified against the
  `ExecutionBackend` protocol via fakes shaped exactly like what Track C
  is expected to build (`test_router_directs_tool_query_to_cascade_cloud`
  and others) — that's as far as one track can verify alone by design
  (Section 10's whole point). The real J-4 run happens once Track C
  lands `gateway/backend.py`.
- **Task 18 (contribute real router-agreement/cache-hit-rate numbers to
  the Section 12 benchmark report)** needs the 200-query reference
  suite and `scripts/run_benchmark.py`, neither of which exist yet
  (Track A's own hardware-compat table is also still `TBD` for the same
  reason — no fabricated numbers, consistent with this repo's own
  standard). Ready to contribute the moment that harness exists;
  `docs/hardware_compat_results.md`'s pattern is the template to follow.

## What Track C can do right now

Import `shared.types.ExecutionBackend` and build `CascadeCloudBackend`
against it exactly as Track A's handoff doc describes — `Pipeline`
already calls `cascade_backend.generate()`/`.generate_stream()`
correctly whenever the router picks that tier, is confidence-escalated
into it, or `LocalExecutionBackendError` is raised. No changes needed
in `engine/intelligence/` once `gateway/backend.py` lands; that's the
whole point of the protocol.

## Next action needed from the team (not code)

**Track C's J-1 sign-off** on `shared/types.py` is still outstanding
(Track A and Track B have both signed off — see `shared/CHANGELOG.md`).
