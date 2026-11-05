# Cascade — Master Requirements & Development Document (V2.2)

> Regenerated as markdown for the repo. This is the authoritative spec
> all three tracks build against — Section numbers below match what
> `shared/types.py`, `shared/CHANGELOG.md`, and every Track A/B handoff
> doc in `docs/` already reference (e.g. "Section 9," "NFR-07," "R-04").

## 1. Executive Summary

Cascade makes a given piece of hardware run a meaningfully better LLM
experience than it could unassisted — by measuring the hardware
instead of assuming it, speeding up generation on it instead of only
picking a smaller model, and offering metered access to real compute
(CascadeCloud) when the hardware simply isn't there. It ships as a
free local engine and a paid hosted tier from the same codebase.

**What changed from V1:** a hardcoded keyword router and a fixed local
model were both guesses dressed up as logic. V2 replaces every guess
with a measured or learned mechanism: adaptive quantization ladder,
a trained (not hardcoded) SemanticRouter, learned context compression,
and CascadeCloud as a genuine revenue engine, not just a routing
convenience layer.

## 2. Problem Statement

| Technique | What it changes | Why hardcoding can't replace it |
|---|---|---|
| Hardware profiling + adaptive quantization | Which model/quant level runs, from measured tok/s on this machine | A fixed "if RAM > 16GB" table is wrong for half of real machines |
| Speculative decoding (lookahead / draft heads / paired draft model) | How fast a model generates | Requires real benchmarking per model pair, not a static rule |
| MoE expert offloading | Whether a large model fits at all | Depends on the input at inference time |
| Learned semantic routing | Which tier answers a query | Query difficulty isn't keyword-shaped |
| Learned context compression | How much history survives | Relevance has to be measured, not assumed by position |

## 3. Project Overview

**Mission:** make the hardware a user already owns run a materially
better LLM experience, and offer metered hosted execution as the
fallback and revenue engine when it can't.

**Target users:** developers/students on 8–16GB RAM laptops with no
discrete GPU; users with genuinely inadequate hardware (Chromebooks,
old laptops); small teams wanting to cut a shared LLM bill without
running infrastructure.

## 4. System Architecture — Seven Components

```
Query ─▶ HardwareProfiler (once, at install)
              │  produces a measured HardwareProfile
              ▼
         ModelLadder ──▶ picks model+quant that clears a real tok/s bar
              │  loads target model (+ draft model if eligible)
              ▼
  ContextCompressor ─▶ SemanticCache ──▶ [hit? return, done]
                              │ miss
                              ▼
                       SemanticRouter (trained classifier)
                 ┌────────────┼─────────────┐
            local (speculative)   cloud BYOK   CascadeCloud (hosted, metered)
                 └────────────┬─────────────┘
                              ▼
                     Confidence Check ──▶ [low? escalate]
                              ▼
                Cache Writer + Decision Logger (→ retrains SemanticRouter)
```

1. **HardwareProfiler** (Track A) — detects CPU/RAM/GPU/disk, benchmarks
   real tok/s at multiple quant levels. ★ Novel: nothing in the
   competitive landscape auto-benchmarks the actual machine.
2. **ModelLadder** (Track A) — adaptive quantization selection with
   OOM/throughput-collapse fallback and MoE expert offloading.
3. **AccelerationEngine** (Track A) — three lossless speedup layers:
   Layer 1 lookahead decoding (default, zero dependencies), Layer 2
   self-speculative draft heads, Layer 3 paired draft-model fallback.
   All output-distribution-preserving by construction — none trade
   quality for speed.
4. **ContextCompressor** (Track B) — relevance-ranked compression via
   embedding similarity, not turn position.
5. **SemanticRouter** (Track B) — cold-started trained classifier
   (local/cloud_byok/cascade_cloud), not a hardcoded ruleset. Confidence
   from model-internal signals where exposed, heuristic fallback
   otherwise.
6. **SemanticCache** (Track B) — two layers: exact-prefix (hash match)
   and fuzzy embedding similarity.
7. **CascadeCloud Gateway** (Track C) — hosted execution for
   local-incapable hardware, served through continuous batching +
   PagedAttention + prefix caching for margin protection, metered and
   billed.

## 5. Technology Stack & Hardware Tiers

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Native ML/embeddings ecosystem |
| Local inference | llama.cpp / Ollama | Already solves quantization + hardware offload |
| Cloud calls (BYOK) | LiteLLM SDK | One interface, config-only provider swap |
| Embeddings | fastembed (ONNX, bge-small) | CPU-only, no torch dependency |
| Router classifier | scikit-learn | Small, fast, trains in seconds |
| Cache storage | SQLite + numpy | Zero extra service at single-user scale |
| Hosted gateway | FastAPI + Postgres + Redis | Standard metered-API stack |
| Hosted serving engine | vLLM / SGLang | Continuous batching + PagedAttention protect margin |
| Billing | Stripe | Metered usage + subscriptions in one integration |
| CLI | Typer | Typed, minimal boilerplate |

**Hardware tiers** — the literal decision logic HardwareProfiler
applies, not just a reference table:

| Tier | GPU VRAM | RAM | Capability |
|---|---|---|---|
| Local-incapable | None | <8GB | Routes 100% to CascadeCloud automatically |
| Minimum (local) | None (CPU) | 8GB | Smallest rung only (~3B, Q4) |
| Recommended | 6–8GB | 16GB | Mid-ladder (~7-8B) with speculative decoding |
| Optimal | 16–24GB | 32GB | Top rung (~14B+) or MoE with expert offloading |

**Quantization quality floor:** GGUF Q4_K_M retains ~92% of FP16
quality; anything below 4-bit is an explicit opt-in, never a default
(NFR-10, NFR-11).

## 6. Functional Requirements (summary — see `shared/types.py` docstrings
and Track A/B handoff docs for full SHALL-statement detail)

- **FR-01 HardwareProfiler**, **FR-02 ModelLadder**, **FR-03
  AccelerationEngine** — Track A, complete (`docs/TRACK_A_PHASE1_SUMMARY.md`,
  `docs/TRACK_A_HANDOFF_FOR_B_AND_C.md`).
- **FR-04 ContextCompressor**, **FR-05 SemanticRouter**, **FR-06
  SemanticCache** — Track B, complete (`docs/TRACK_B_PHASE2_3_SUMMARY.md`,
  `docs/TRACK_B_HANDOFF_FOR_C.md`).
- **FR-07 CascadeCloud Gateway**, **FR-08 CLI & Setup** — Track C.

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Performance | Layer 2/3 acceleration ≥1.5x tok/s vs Layer 1 alone |
| NFR-02 | Performance | Cache-hit responses <150ms end to end |
| NFR-03 | Reliability | Any component failure degrades gracefully, never crashes |
| NFR-04 | Economics | CascadeCloud gross margin tracked per request, never sustained-negative |
| NFR-05 | Privacy | Local processing never transmits off-device; CascadeCloud discloses what is transmitted |
| NFR-06 | Portability | Runs on macOS/Windows/Linux, 8GB RAM minimum |
| NFR-07 | Modularity | Each component replaceable independently via the shared Pipeline/ExecutionBackend seams |
| NFR-08 | Reproducibility | Deterministic rung selection given identical HardwareProfile |
| NFR-09 | Testability | ≥70% coverage per component, no live GPU/Ollama/network required |
| NFR-10 | Quality | No lossy optimization enabled by default; explicit opt-in with disclosed trade-off |
| NFR-11 | Quality | Default quantization ≤~2% perplexity degradation vs FP16, evidence-based |

## 8. The Seams — Shared Interface Contracts (Section 9)

Two contracts, not one, because three independent tracks need to call
each other's code without importing it:

```python
# shared/types.py

class ExecutionBackend(Protocol):        # Track A and Track C each implement this
    def generate(self, messages: list[Message]) -> BackendResult: ...
    def generate_stream(self, messages: list[Message]) -> Iterator[str]: ...

class Pipeline:                          # Track B owns this — the orchestrator
    def __init__(self, backend: ExecutionBackend,
                 cascade_backend: ExecutionBackend | None = None, ...) -> None: ...
    def run(self, messages: list[Message]) -> PipelineResult: ...
    def run_stream(self, messages: list[Message]) -> Iterator[str]: ...
```

Track B's Pipeline calls `local_backend.generate()` or
`cascade_backend.generate()` — never anything inside `engine/execution/`
or `gateway/` directly. See `shared/types.py` for the full, current,
frozen schema (all three tracks have signed off).

## 9. Remote Execution & Team Workflow (Section 10)

Three tracks, each owning a genuinely different kind of problem:

| Track | Owns | Repo folder |
|---|---|---|
| Track A — Execution Engine | HardwareProfiler, ModelLadder, AccelerationEngine (FR-01–03) | `engine/execution/` |
| Track B — Intelligence Layer | ContextCompressor, SemanticRouter, SemanticCache, Pipeline (FR-04–06) | `engine/intelligence/` |
| Track C — Platform & Revenue | CascadeCloud Gateway, CLI (FR-07–08) | `gateway/`, `cli/` |
| Shared — all three | `shared/types.py` — both seams | `shared/` |

**Guiding rule:** if a task requires importing something from inside
another track's folder, it's modeled wrong. Everything flows through
`Pipeline.run()`/`.run_stream()` or the `ExecutionBackend` protocol.

**Git workflow:** `track-a/fr-0X-<name>`, `track-b/fr-0X-<name>`,
`track-c/fr-0X-<name>` branches. PRs touching only one track's own
folder need just that track's self-review + green CI. PRs touching
`shared/types.py` need all three tracks' review, every time, no
exceptions.

## 10. Business Model & Monetization

**CascadeCloud's actual margin mechanism:** because ContextCompressor
and SemanticCache run ahead of every CascadeCloud call exactly as they
do locally, Cascade's wholesale GPU cost per served answer is lower
than a competitor reselling raw inference with no optimization layer.
That gap is the margin.

| Buyer | Buys | How they're reached |
|---|---|---|
| Underpowered-hardware users | CascadeCloud pay-as-you-go | HardwareProfiler itself is the sales moment |
| Small dev teams | Team subscription | Product-led: shared-cache savings surfaced in-tool |
| Cost-conscious AI startups | CascadeCloud + Pro | Benchmark content (real numbers) on HN/r/LocalLLaMA |
| Regulated/privacy orgs | Enterprise/on-prem | Case-study-led, after a reference customer exists |

## 11. Benchmarks & Evaluation Criteria

200-query reference suite (TriviaQA/GSM8K/HumanEval/MT-Bench subsets).
Key targets: speculative speedup ≥1.5x, router-vs-oracle agreement
≥85%, cache hit rate ≥20%, hardware ladder success rate ≥90% on 8GB+
machines, cost reduction ≥50% vs pure-cloud baseline. **Partly run**
(2026-11): router agreement, cache hit rate and the acceleration
speedup have measured results in `docs/BENCHMARK_REPORT.md` (the
speedup target is not met on CPU-only hardware); the full 200-query
dataset suite and the multi-machine ladder test are still to do — see
`docs/PROJECT_STATUS.md`.

## 12. Risk Register (selected)

| ID | Risk | Mitigation |
|---|---|---|
| R-01 | Layer 2/3 acceleration only benefits models with validated heads/pairs | Layer 1 (lookahead) ships as the zero-dependency default |
| R-04 | Three tracks can't build a router, speculative decoding, and a billed gateway simultaneously | Five-phase sequencing + the two seams + Section 10's join points |
| R-12 | Two independent ExecutionBackend implementations could interpret the protocol differently | Shared contract test suite both must pass in CI |

Full register: 12 risks, see the original working document for R-02,
R-03, R-05–R-11.

## 13. Glossary

See `shared/types.py` module docstrings and `docs/TRACK_A_HANDOFF_FOR_B_AND_C.md` /
`docs/TRACK_B_HANDOFF_FOR_C.md` for term-by-term definitions (ladder
rung, draft/target pair, acceptance rate, local-incapable,
shadow-labeling, cold start, the seams, ExecutionBackend, BackendResult,
join point).

---

*This markdown version condenses the original styled working document
into a repo-native reference. Section numbers are preserved so
cross-references elsewhere in the codebase (e.g. "Section 9," "R-04")
still resolve correctly.*
