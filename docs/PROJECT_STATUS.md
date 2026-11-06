# Cascade — Project Status (2026-11-05)

Cascade is an MIT-licensed open-source project. This page says what works, what was verified live,
and what's next. Measured numbers are in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md); the change
log is [`../shared/CHANGELOG.md`](../shared/CHANGELOG.md).

## Requirement status

| Req | Component | Status | Evidence |
|---|---|---|---|
| FR-01 | HardwareProfiler | Done, verified live | `cascade init`: local_capable, 4.5 tok/s, 0.55 GB measured |
| FR-02 | ModelLadder (OOM step-down, real Ollama tags) | Done | unit tests; live load of `llama3.2:3b-instruct-q4_K_M` |
| FR-03 | AccelerationEngine (layers, fallback, acceptance monitor) | Done and real; **not faster on CPU** | prompt lookup 0.60x, draft model 0.89x; off by default |
| FR-04 | ContextCompressor | Done | unit tests |
| FR-05 | SemanticRouter (3-way) | Done | 89.2% held-out agreement; live routing to `local` and `cloud_byok` |
| FR-06 | SemanticCache | Done, verified live | 0/19 wrong answers on near-miss pairs; 42.9% of rewordings served (strict guard) |
| FR-07 | Self-hosted Cascade server | Done, verified live locally | auth, quotas and streaming exercised against a real upstream |
| FR-08 | CLI | Done | `init / ask / chat / stats / profile / doctor / config` |
| NFR-01 | ≥1.5x acceleration | **Not met** on the dev laptop | needs a GPU measurement |
| NFR-02 | cache hit <150 ms | Met (warm) | 55 ms median live; first lookup per process ~2.4 s (model load) |
| NFR-03 | graceful degradation | Met | failed or over-quota remote tiers fall back to local |
| NFR-05 | local never transmits | Met | no default remote endpoint; no per-request registry pull |
| NFR-09 | tests without GPU/Ollama/network | Met | 313 tests |

## 0.3.1 — fixes from the first live end-to-end benchmark

- The cache served a wrong answer (Celsius question → stored Fahrenheit answer) and replayed a
  live-data answer ("current prime minister"). Fixed with a term-swap guard and by skipping the
  cache for the live-data tier; 0/19 wrong answers on the near-miss fixture, down from 4/19.
- A default, configurable system prompt: median answer time 23.7 s → 4.7 s over 67 live requests.
- The benchmark now uses the CLI's real settings and recorded model answers, and sweeps every cache
  setting. Details: `BENCHMARK_REPORT.md`, Addendum 5.
- Open: reworded questions served from cache fell to 42.9% — the price of zero wrong answers with
  these embeddings. A better duplicate-question model is a good contribution.

## 0.3.0 — open-source release

- Paid hosted tier dropped. Stripe billing, prices and dollar spend caps were removed.
- The `cascade_cloud` tier is now `self_hosted`: a Cascade server you run on your own GPU machine
  and share with your team, with per-user API keys and optional **token** quotas
  (`python -m gateway.serve create-user/create-key/revoke-key/usage/reset-usage/run`).
- MIT license, CONTRIBUTING, SECURITY, package metadata. Internal planning docs moved to
  [`archive/`](archive/).

## Fixed in the 0.2.0 completion pass

1. **Fabricated acceleration speedups.** Reported tok/s was divided by hard-coded 1.15–1.80x
   factors; NFR-01 tests passed by construction. Replaced by real speculative decoding (llama.cpp
   runner) and pass-through of measured numbers only.
2. **Local runner rebuilt on Ollama's HTTP API.** The old one shelled out to `ollama run`, pulled
   from the registry on every request, unloaded the model after each request, and counted tokens
   by splitting on whitespace.
3. **Client and server spoke different protocols.** The server is now OpenAI-compatible and a
   test drives the real client against the real app.
4. **No authentication on the server** — identity was a self-asserted header. Now API keys
   (stored only as SHA-256 hashes) with revocation.
5. **Usage limits were in memory** (lost on restart). Now SQLite with atomic check-and-reserve,
   tested under 30 concurrent threads.
6. **The `cloud_byok` tier had no backend** — hard queries silently ran locally.
7. **The CLI sent remote-tier traffic to a domain the project doesn't control by default.**
   Remote tiers are now off until configured.
8. **Windows: ~2 s added to every local request** (`localhost` → IPv6 stall), and a crash printing
   characters like `√` when piped. Both fixed.
9. **A failing remote tier crashed the request** instead of falling back (NFR-03).

## Good places to contribute

Needs hardware or data rather than code:

- **NFR-01 on a GPU.** Run `scripts/benchmark_acceleration.py` with a CUDA or Metal
  llama-cpp-python build and share the numbers.
- **Hardware ladder on more machines.** Run `cascade init` and add a row to
  `docs/hardware_compat_results.md`.

Code:

- **200-query benchmark suite** (TriviaQA/GSM8K/HumanEval/MT-Bench subsets) and a router trained on
  RouteLLM's public data; today's numbers come from small hand-made fixtures.
- **Real token streaming for the local tier** — `generate_stream()` currently generates the full
  answer and then yields it word by word.
- **Multi-turn chat on the local tier** is sent as one flattened prompt (remote tiers send real
  messages).
- **Fallback visibility** — when a remote tier fails and Cascade answers locally, the footer says
  `local` but doesn't say a fallback happened.
- **Tool calling.** The router sends live-data queries ("what's the weather now") to `self_hosted`,
  but no tier executes tools yet.
- **Cache recall without wrong answers.** The strict term-swap guard turns away rewordings that
  swap a verb ("wrote" → "author"). A duplicate-question model that can tell those from
  "fahrenheit" → "celsius" would win back hits; `scripts/run_benchmark.py` measures both sides.
- **Postgres store** for a multi-host server (SQLite is correct for one host).
- **PyPI release.** The name `cascade` is likely taken on PyPI; a distinct package name is needed.
