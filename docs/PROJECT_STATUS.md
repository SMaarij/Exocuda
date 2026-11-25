# Cascade — Project Status (2026-11-25)

Cascade is an MIT-licensed open-source project. This page says what works, what was verified live,
and what's next. Measured numbers are in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md); the change
log is [`../shared/CHANGELOG.md`](../shared/CHANGELOG.md).

## Requirement status

| Req | Component | Status | Evidence |
|---|---|---|---|
| FR-01 | HardwareProfiler | Done, verified live | `cascade init`: local_capable, 4.5 tok/s, 0.55 GB measured |
| FR-02 | ModelLadder (OOM step-down, real Ollama tags) | Done | unit tests; live load of `llama3.2:3b-instruct-q4_K_M` |
| FR-03 | AccelerationEngine (layers, fallback, acceptance monitor) | Done, verified live | cost-aware draft model plus lookup drafting on by default: 1.58x overall on the idle CPU-only laptop (Addendum 17); lookup adds +0.13x under load (Addendum 19); lossless |
| FR-04 | ContextCompressor | Done | unit tests |
| FR-05 | SemanticRouter (3-way) | Done | 89.2% held-out agreement; live routing to `local` and `cloud_byok` |
| FR-06 | SemanticCache | Done, verified live | 0/19 wrong answers on near-miss pairs; 42.9% of rewordings served (strict guard) |
| FR-07 | Self-hosted Cascade server | Done, verified live locally | auth, quotas and streaming exercised against a real upstream |
| FR-08 | CLI | Done | `init / ask / chat / stats / profile / doctor / config` |
| NFR-01 | ≥1.5x acceleration | **Met overall** (1.58x; grounded 1.88x, open-ended 1.27x) | cost-aware draft length, per-machine verify costs, 2 interleaved rounds, verified lossless (Addendum 17) |
| NFR-02 | cache hit <150 ms | Met (warm) | 55 ms median live; first lookup per process ~2.4 s (model load) |
| NFR-03 | graceful degradation | Met | failed or over-quota remote tiers fall back to local |
| NFR-05 | local never transmits | Met | no default remote endpoint; no per-request registry pull |
| NFR-09 | tests without GPU/Ollama/network | Met | 313 tests |

## 0.10.1 — the CLI streams

- `cascade ask` streams by default (first words 2.8 s into a 23.9 s answer); `cascade chat` uses the
  background server, so models and the conversation stay loaded between turns and sessions.

## 0.10.0 — answers start in seconds, and survive unloads

- Replies stream: `ollama run`, Ollama apps and OpenAI clients see the first words after ~2–5 s instead of
  after the whole answer (36–59 s).
- What the models computed is written to disk on idle unload and exit and restored on the next load:
  first token 3.1 s instead of 18.1 s after a reload; a restarted server's first answer starts in 5.2 s
  instead of 12.1 s.
- Full setup re-checked with native chat turns: 34/34 correct, 1.57x faster than the plain 7B session.

## 0.9.0 — conversations

- Chat messages keep their roles in the model's own template, so earlier answers are reused too: first
  token from turn 2 on 1.96 s instead of 5.65 s, a 4-turn chat 70 s instead of 93 s; 34/34 correct.
- Saved KV caches (up to 2,048 tokens per model): two alternating conversations get their first token
  in about 2–4 s instead of 10.9 s.
- Lookup drafting also searches earlier answers; the embedding model's cache moved out of the temp folder.

## 0.8.0 — reuse what is already computed

- Conversations keep the 7B's computed prompt prefix: from turn 2 on, the first token arrives in about
  4 s instead of about 42 s (677-token system prompt), with unchanged output.
- Lookup drafting: text already in the context is offered as a free draft when it pays. On a busy
  laptop: 1.49x overall vs 1.36x without it, open-ended 1.24x vs 1.08x, lossless (Addendum 19).
- Models found once load without Ollama running (`~/.cascade/model_paths.json`).

## 0.7.0 — 1.58x overall

- The 7B drafts as far as it pays: the draft length is chosen by what verifying it costs on this
  machine (measured once, ~25 s on first load). 1.58x overall (was 1.39x), lossless.

## 0.6.0 — install once, use plain `ollama`

- `cascade install --as-ollama`: Cascade starts at login on Ollama's port; `ollama run` and Ollama
  apps use it with no setting, other models pass through to the real Ollama.
- Idle machine: 0.02 GB before the first request, 0.65 GB after the idle unload.

## 0.5.1 — smaller working memory

- The 7B's compute buffer is smaller (flash attention, n_ubatch 256): -0.15 GB at 2k context,
  -0.40 GB at 8k, identical answers, speed-up unchanged (1.38x vs 1.35–1.36x same day).

## 0.5.0 — works under the tools people already use; fastest setup is the default

- `cascade start`: the Ollama CLI, Ollama apps and OpenAI-SDK tools run on Cascade unchanged
  (cache hit 0.7 s, easy answers ~2 s), verified with the real `ollama run`.
- Setting only `model qwen2.5-coder` now gives the measured fastest setup automatically.
- Memory: Cascade with 7B + 0.5B + embedder holds 5.47 GB; Ollama holds 5.06 GB for the 7B alone.

## 0.4.2 — the small model re-asks the 7B when unsure; 0.5B setup recommended

- Small-model answers with mean token probability below 0.75 go to the 7B. On 128 checked easy
  questions the 0.5B then gets 4 wrong, fewer than the 1.5B (6) or the 0.5B alone (11).
- Session: 2.11x faster than plain, 34/34 correct, 5.9 GB peak (Addendum 13).

## 0.4.1 — small model at no extra RAM; more cache hits on rewordings

- The 0.5B small model doubles as the 7B's draft model, one instance in RAM: whole session 1.65x faster
  than plain at 5.88 GB peak, the same as without a small model (1.5B setup: 1.79x at 7.0 GB).
- Reworded questions hit the cache more often (fixture 42.9% -> 50.0%) with still 0/19 wrong answers.
- Rejected: the 1.5B as draft model (1.08x vs 1.58x). Details: BENCHMARK_REPORT Addendum 12.

## 0.4.0 — whole system 2.19x faster per session; models stay loaded

- Measured end to end on a 34-question session (easy, hard, repeats, rewordings, near-misses), every
  answer checked: plain 7B 431 s; Cascade with a 1.5B model for easy questions **197 s (2.19x)**,
  median answer 9.6 s → 2.0 s, **34/34 correct**, peak RAM 7.0 GB. Addendum 11.
- `cascade ask` keeps models loaded in a background server (127.0.0.1, exits after 10 idle minutes):
  warm answers ~2 s instead of 60–90 s of model loading per call.
- Fixed: double 7B load (peak 9.5 → 5.9 GB) and a step-down that made answers after the 20th up to
  3.5x slower.

## 0.3.6 — memory leak fixed; 1.39x overall confirmed

- 0.3.5's fix left a reference cycle that kept replaced models in RAM until garbage collection
  (up to 9.4 GB with two 7B copies). Fixed; RAM is back to 4.6 GB plain, ~5.5 GB with the draft.
- Confirmation run: 1.69x on code/rewrite prompts, 1.08x open-ended, **1.39x overall**, lossless.

## 0.3.5 — long prompts no longer crash speculative mode; less RAM

- Speculative decoding crashed once prompt + output passed 512 tokens (a llama-cpp-python buffer
  sized to the batch, not the context). Fixed by computing logits only where verification needs
  them: a 478-token prompt now prefills 11% faster in 0.7 GB less RAM. Addendum 10.
- A general-purpose draft model ties the coding one (1.38x overall), so the draft model is not
  what limits open-ended prompts. Addendum 9.

## 0.3.4 — confidence drafting: open-ended prompts no longer slower

- The draft model now stops drafting at the first token it is less than 75% sure of (llama.cpp's
  `--draft-p-min`), and the CLI uses this by default. Same rounds against the old default:
  overall 1.19x → **1.33x**, open-ended prompts 0.83x → **1.01x**, code/rewrite prompts 1.64x.
  Lossless check passes. Addendum 8.
- Pausing the draft after rejections was tried and did not help; it stays off.

## 0.3.3 — RAM 7.2 → 4.6 GB, and 1.65x on grounded prompts

- llama.cpp kept the memory-mapped model file resident next to its repacked copy of the weights.
  Loading without mmap (now the default): 7B RAM 7.2 GB → 4.6 GB, plain decoding ~20% faster, and
  draft verification ~25% cheaper.
- qwen2.5-coder 7B + 0.5B adaptive draft: 1.65x on code/rewrite prompts and 1.25x overall,
  consistent across 3 interleaved rounds, 0 lossless-check violations. Addendum 7.

## 0.3.2 — speculative decoding verified lossless and tuned

- The "accelerated output differs from plain output" problem had two causes. A real bug, KV-cache
  state leaking between requests, is fixed. The rest is floating-point noise at near-ties:
  `LlamaCppRunner.verify_greedy()` now proves every output token is a correct greedy choice within
  the measured noise, with 0 violations across all runs.
- Adaptive draft length, now the CLI default, and a better pair (qwen2.5-coder 7B + 0.5B draft):
  1.10x overall and 1.29–1.35x on code and rewrite prompts on the CPU-only laptop. NFR-01 (1.5x)
  still needs a GPU. Details: `BENCHMARK_REPORT.md`, Addendum 6.

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
