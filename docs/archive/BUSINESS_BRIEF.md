# Cascade — Business Brief

> Regenerated as markdown for the repo. Companion to
> `docs/planning/MASTER_REQUIREMENTS.md` (Section 10, Business Model)
> and `docs/planning/COMPETITIVE_LANDSCAPE.md`.

## Executive Summary

Cascade is not a model, and it is not an inference engine. It is the
layer between a user and whichever model actually answers their
question — deciding, per request, whether a small local model is
enough, or whether the query should escalate to a larger cloud model,
or to Cascade's own hosted tier.

The same pipeline supports two products from one codebase: a **free,
self-hosted engine** for individual developers, and a **hosted
service (CascadeCloud)** for people and teams who don't want to manage
local infrastructure. The free tier builds the user base and the
routing data; the hosted tier is where revenue comes from.

**Recommendation (unchanged since V1):** ship the self-hosted engine
first — it's buildable without infrastructure, costs nothing to
distribute, and generates real usage data. Layer CascadeCloud on top
once the routing logic is proven.

## The Problem

Good LLMs need hardware most people don't have. Quantized local models
(Ollama, llama.cpp) make a model *fit*, but the output is noticeably
worse than a paid API for hard questions. Pure cloud APIs give the
good model but cost money on every request. Nobody sells the thing in
between: a system that uses local compute for what it's good at, saves
cloud spend for questions that need it, and never re-answers a
question it's already solved.

## Business Model: Open Core, Hosted Upsell

| | Free product | Paid product |
|---|---|---|
| **Model** | Self-hosted, MIT-licensed, runs on the user's machine, BYOK cloud fallback | Managed hosting: cross-device sync, shared team cache, admin dashboard, CascadeCloud metered inference |
| **Role** | Distribution channel and credibility | Revenue |
| **Why it works** | Free users generate the routing data that makes the paid tier smarter, at no cost to build | Structurally can't be replicated by the free tier — team-shared caching and zero-setup convenience require a server somebody runs |

## Revenue Streams

1. **CascadeCloud pay-as-you-go** (primary) — metered inference for
   local-incapable devices, or opt-in convenience. Margin protected by
   running the same compression/caching pipeline ahead of every hosted
   call (see Master Requirements Section 10).
2. **Pro subscription** — cross-device sync, priority routing updates.
3. **Team subscription** — shared cache + admin dashboard. The
   ROI-obvious sell: "N teammates stopped re-asking the same questions."
4. **Enterprise/on-prem** — custom deployment inside a company's own
   infrastructure. Highest price, comes after case studies exist.

## Who Buys, and How They're Reached

| Buyer | Buys | How reached |
|---|---|---|
| Underpowered-hardware users (students, older laptops) | CascadeCloud pay-as-you-go | HardwareProfiler's own "local-incapable" flag *is* the honest sales moment |
| Small dev teams with growing API spend | Team subscription | Product-led — the tool itself surfaces shared-cache savings once several teammates use it |
| Cost-conscious AI-feature startups | CascadeCloud + Pro | Benchmark content (real measured numbers) on HN/r/LocalLLaMA — this audience trusts numbers over marketing |
| Regulated/privacy-sensitive orgs | Enterprise/on-prem | Case-study-led, after a reference customer exists — not cold outreach |

Individual solo developers are **not** the buyer — they're the
distribution mechanism. They install the free tool, tell other
developers, and some bring it into a team, which is when it converts
to revenue.

## Pricing (illustrative — set final numbers once real cloud-call costs
are known at the target cache-hit rate)

| Tier | Price | Includes |
|---|---|---|
| Free — self-hosted | $0 | Full local pipeline, BYOK, community support |
| Pro — hosted | $12/mo | Cross-device sync, hosted routing, usage analytics |
| Team — hosted | $8/seat/mo | Shared team cache, admin dashboard, seat management |
| Enterprise | Custom | On-prem deployment, custom routing rules, SLA |

## Risks

- **Local models keep improving for free** (Meta, Alibaba, Google
  shrinking the gap every few months) — mitigation: the value is the
  routing/caching/acceleration layer, not the models themselves.
- **Caching can return stale answers** to time-sensitive queries —
  mitigation: `SemanticCache.is_cacheable()` explicitly excludes those
  (see `engine/intelligence/cache.py`).
- **Usage-based reselling carries financial risk** — mitigation: stay
  BYOK as long as possible; CascadeCloud's spend caps (FR-07.2) limit
  exposure before billing risk is taken on.
