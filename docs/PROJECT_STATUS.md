# Cascade — Project Status (2026-11-05)

Where the project stands after the completion pass: what works and was verified live, what
was found broken and fixed, and what is left. Measured numbers are in
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md); the decision log is
[`../shared/CHANGELOG.md`](../shared/CHANGELOG.md).

## Requirement status

| Req | Component | Status | Evidence |
|---|---|---|---|
| FR-01 | HardwareProfiler | Done, verified live | `cascade init`: local_capable, 4.5 tok/s, 0.55 GB measured |
| FR-02 | ModelLadder (OOM step-down, real Ollama tags) | Done | unit tests; live load of `llama3.2:3b-instruct-q4_K_M` |
| FR-03 | AccelerationEngine (layers, fallback, acceptance monitor) | Done and real; **not faster on CPU** | Addendum 3: prompt lookup 0.60x, draft model 0.89x; off by default |
| FR-04 | ContextCompressor | Done | unit tests |
| FR-05 | SemanticRouter (3-way) | Done | 89.2% held-out agreement; live routing to `local` and `cloud_byok` |
| FR-06 | SemanticCache | Done, verified live | paraphrase hit; 22–55 ms warm lookup |
| FR-07 | CascadeCloud gateway | Done in code, verified live locally | auth, caps, streaming and billing exercised against a real upstream |
| FR-08 | CLI | Done | `init / ask / chat / stats / profile / doctor / config` |
| NFR-01 | ≥1.5x acceleration | **Not met** on the dev laptop | needs a GPU measurement |
| NFR-02 | cache hit <150 ms | Met (warm) | first lookup per process ~2.4 s (model load) |
| NFR-03 | graceful degradation | Met | failed/over-cap cloud tiers fall back to local |
| NFR-05 | local never transmits | Met | no default cloud endpoint; no per-request registry pull |
| NFR-09 | tests without GPU/Ollama/network | Met | 331+ tests, 94% line coverage |

## Fixed in the completion pass

Real defects, each found by reading the code or running it live:

1. **Fabricated acceleration speedups.** Reported tok/s was divided by hard-coded 1.15–1.80x
   factors; NFR-01 tests passed by construction. Replaced by real speculative decoding (llama.cpp
   runner) and pass-through of measured numbers only.
2. **Local runner rebuilt on Ollama's HTTP API.** The old one shelled out to `ollama run`, ran
   `ollama pull` (a network call) on every request, unloaded the model after each request, and
   counted tokens by splitting on whitespace.
3. **CascadeCloud client and server spoke different protocols** — the client could never parse
   the gateway's responses. The gateway is now OpenAI-compatible and a test drives the real client
   against the real app.
4. **Gateway had no authentication** — identity was a self-asserted header, so anyone could bill
   another user or dodge a cap. Now API keys (stored only as SHA-256 hashes) with revocation.
5. **Spend caps were in memory** (lost on restart). Now SQLite with atomic check-and-reserve,
   tested under 30 concurrent threads.
6. **Streaming requests were billed at the worst-case estimate.** Now reconciled to real usage
   (`stream_options.include_usage`) on both client and server.
7. **The `cloud_byok` tier had no backend** — hard queries silently ran locally. Added
   `CloudBYOKBackend` for any OpenAI-compatible provider.
8. **The CLI pointed CascadeCloud at `https://api.cascade.dev` by default**, a domain the project
   doesn't control — prompts could have been sent there. Cloud tiers are now off until configured.
9. **Draft-pair validation measured nothing real** (compared words of two independent outputs)
   and wrote IDs the engine never matched. Rewritten to measure real acceptance and speedup.
10. **Windows: ~2 s added to every local request** (`localhost` → IPv6 stall). Now `127.0.0.1`.
11. **Windows: CLI crashed printing characters like `√`** when piped. Fixed.
12. **A failing cloud tier crashed the request** instead of falling back. Fixed (NFR-03).
13. **Messages were flattened into one `USER: ...` string** for cloud calls. Now passed as real chat messages.

## Left to do — needs a decision, an account, or hardware (not code)

| Item | Why it can't be finished in code | What to do |
|---|---|---|
| Stripe billing live | needs your Stripe account and a metered price | create a Meter named `cascade_tokens` in Stripe, then set `STRIPE_API_KEY` on the gateway |
| Real hosted GPU upstream | business/provider decision and spend | pick a provider serving an OpenAI-compatible API (e.g. a vLLM deployment) and set `CASCADE_UPSTREAM_URL/MODEL/API_KEY` |
| Price per 1k tokens | business decision (`BUSINESS_BRIEF.md`) | set `CASCADE_PRICE_PER_1K`; $0.50 is a placeholder |
| NFR-01 on a GPU | needs a GPU machine | run `scripts/benchmark_acceleration.py` there with a CUDA/Metal llama-cpp-python build |
| Hardware ladder on ≥5 machines (R-06) | needs 5 physical machines | run `cascade init` on each and fill `docs/hardware_compat_results.md` |
| License | your choice | add a LICENSE file (README currently says none chosen) |
| Public launch / hosting the gateway | business action | deploy `python -m gateway.serve run` behind HTTPS |

## Left to do — code, smaller

- **200-query benchmark suite** (TriviaQA/GSM8K/HumanEval/MT-Bench subsets) and a router trained
  on RouteLLM's public data; today's numbers come from small hand-made fixtures.
- **Real token streaming for the local tier** — `generate_stream()` currently generates the full
  answer and then yields it word by word.
- **Multi-turn chat on the local tier** is sent as one flattened prompt (the cloud tiers now send
  real messages).
- **Fallback visibility** — when a cloud tier fails and Cascade answers locally, the footer says
  `local` but doesn't say a fallback happened.
- **Postgres store** for a multi-host gateway (SQLite is correct for one host).
