# Cascade — Competitive Landscape

> Regenerated as markdown for the repo. Research current as of the
> original working document; treat funding/pricing figures as
> directional — this space moves fast.

## Executive Summary

No product currently combines local execution + intelligent
local/cloud routing + semantic caching + team-shared cache in one
buyable thing, aimed at people who don't already have enterprise cloud
spend to optimize. Four adjacent categories each own a piece of it —
any one of them could extend into Cascade's space faster than a small
team can build up to theirs, so this gap is real but narrow.

## Category A — Local-Inference Apps

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **Msty** *(closest existing analog)* | Local-first desktop AI studio: run open models on your own hardware, or connect cloud providers, from one interface. Free core; Enterprise $300/seat/yr | No adaptive hardware profiling, no speculative decoding, no learned router, no shared team cache |
| **Ollama** | CLI + local HTTP server; the substrate most other tools wrap. $65M Series B (Jul 2026), 8.9M developers | Infrastructure layer, not a competitor — Cascade needs Ollama to keep improving |
| **LM Studio** | Polished GUI for model discovery + local serving | No routing/caching/acceleration layer |
| **Jan** | 100% offline, newcomer-friendly | Same gap as above |
| **AnythingLLM** | Document/RAG-centric local agent app | Same gap as above |

## Category B — Cloud Model Routers

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **OpenRouter** | Unified API/marketplace across 500+ cloud models, automatic fallback | Zero local execution — never reduces dependence on paid cloud tokens |
| **Not Diamond** | Predictive per-query routing across cloud models. $2.3M pre-seed | Cloud-only |
| **Martian** | "Model mapping" interpretability-based routing. $9M seed | Cloud-only |
| **Unify** | Runtime-benchmarked quality/cost/speed routing | Cloud-only |
| **RouteLLM (academic)** | Proves learned routing cuts cost 40-85% vs a single strong model | Research release only — no packaging, no local-tier integration, no hosted revenue path. **Cascade's SemanticRouter cold-starts directly on this dataset's approach** (`engine/intelligence/router/coldstart_data.py`) |

## Category C — LLM Gateways & Semantic Caching

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **Portkey** | Gateway + semantic caching (~20% hit rate) + observability | Built for enterprise cloud API traffic, not individuals without capable hardware |
| **Helicone** | Lightweight observability, 20-95% savings via caching | Same gap |
| **LiteLLM** | Free, self-hosted multi-provider proxy — Cascade uses this directly for BYOK cloud calls | Base layer, not a competing product |
| **Kong AI Gateway** | Semantic caching + routing, enterprise API-management-native | Same gap as Portkey |
| **Cloudflare AI Gateway** | Edge caching — exact-match only, no semantic layer | Misses paraphrased duplicate queries entirely |

## Category D — On-Device Mobile & Edge AI

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **PocketPal AI** | Open-source on-device chat app (llama.cpp), free | Chat-app scope only, no routing/caching/hosted fallback |
| **Private LLM** | Privacy-marketed on-device iOS app | Same gap |
| **Nexa AI** | On-device inference framework sold to device makers | Not consumer-facing |

## Category E — Platform-Native Hybrid AI

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **Apple Intelligence + Private Cloud Compute** | On-device first, audited cloud escalation | Locked to Apple silicon/models; proves the pattern works at consumer scale, which is marketing Cascade doesn't have to pay for |
| **Microsoft Copilot+ / Phi Silica** | NPU-local small model, Azure cloud escalation | Locked to Windows; same escalation logic as Cascade, offered as a platform API not a cross-platform product |

## Category F — Enterprise LLM FinOps

| Company | What it does | Gap Cascade closes |
|---|---|---|
| **CloudZero** | Unit-economics attribution for AI/GPU spend | Reports spend after the fact, can't intercept the next request |
| **Vantage** | Cloud-first spend tracking, AI as one more provider | Same — attribution, not reduction |

## Head-to-Head Capability Matrix

| Capability | Cascade | Msty | OpenRouter/ND | Portkey/Kong | PocketPal | Apple Intel. |
|---|---|---|---|---|---|---|
| Runs on user's own hardware | ✓ | ✓ | — | — | ✓ | ✓ |
| Automatic local⇄cloud routing per request | ✓ | — | — | — | — | ✓ |
| Semantic caching | ✓ | — | — | ✓ | — | — |
| Shared cache across a team | ✓ | — | — | — | — | — |
| Works with any cloud provider | ✓ | ✓ | ✓ | ✓ | — | — |
| Usable by non-developers, zero config | ✓ | ✓ | — | — | ✓ | ✓ |

## What This Means for Cascade

- **The white space is real but narrow** — no competitor occupies more
  than 2-3 rows of the matrix above. It's defensible only as long as
  Cascade actually ships the full routing + caching + sharing
  combination, not just a local-model wrapper Msty already provides
  for free.
- **Msty is the nearest threat**, not the funded cloud routers — it
  already has the local+cloud interface and a paying enterprise tier.
  Watch its release notes for automatic routing or shared caching.
- **The platform giants (Apple, Microsoft) set user expectations, not
  the bar to beat** — they're proving local-first-with-cloud-escalation
  is a normal, trustworthy pattern to millions of users, which is
  marketing Cascade doesn't have to pay for.
