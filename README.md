# Cascade

### Run the right model in the right place — automatically.

Cascade is an open-source, local-first LLM runtime. It benchmarks your
machine, runs a model locally when your hardware can handle the query,
answers repeated questions from a semantic cache, and sends hard queries
to a provider or server **you** chose — never anywhere you didn't
configure.

```text
query ─▶ ContextCompressor ─▶ SemanticCache ──hit──▶ answer (instant, offline)
                                   │ miss
                                   ▼
                            SemanticRouter (trained classifier)
              ┌────────────────────┼─────────────────────┐
           local                cloud_byok            self_hosted
   Ollama or llama.cpp      your own API key,      a Cascade server you
   (+ speculative decoding)  any OpenAI-compatible  run on a GPU box and
                             provider               share with your team
              └────────────────────┼─────────────────────┘
                                   ▼
          confidence check ─▶ escalate if weak ─▶ cache write + decision log
```

Only `local` is on by default. Every other tier is opt-in, and every tier
degrades gracefully: if a remote tier is down, mis-keyed or over its quota,
the request is answered locally instead of failing.

---

## Quickstart

Requirements: Python 3.10+ and [Ollama](https://ollama.com) running locally.

```bash
git clone https://github.com/SMaarij/Exocuda.git
cd Exocuda
pip install -e .
```

```bash
cascade init        # benchmarks this machine (downloads llama3.2 3B on first run)
cascade doctor      # checks every configured piece and says how to fix what isn't
cascade ask "What is a hash table?"
cascade chat
cascade stats       # cache hit rate, local-resolution rate, speculative speedup
```

Each answer ends with a status footer, e.g. `[local | 9.8 tok/s | 2140ms | $0.0000]`.
The dollar figure is only ever non-zero for your own BYOK provider, and only if you
tell Cascade its prices.

If you skip `cascade init`, the first `ask` runs the same one-time hardware check first
(several minutes on a CPU) and says so.

### Fastest setup (measured 1.8–2.2x on a mixed session, all answers correct)

```bash
cascade config set runner llamacpp
cascade config set model qwen2.5-coder                 # 7B for hard questions
cascade config set acceleration draft_model            # confidence drafting with qwen2.5-coder:0.5b
cascade config set draft_model qwen2.5-coder:0.5b
cascade config set small_model qwen2.5:1.5b-instruct   # fast model for easy questions
```

The first `cascade ask` starts a background model server on 127.0.0.1 and loads the models
(about a minute on a CPU laptop); after that, easy answers take about 2 seconds and repeated
questions come back from the cache in well under one. The server exits after 10 idle minutes
(`server.idle_minutes`) and frees the RAM; `cascade stop` frees it at once,
`cascade config set server.auto off` disables it.

**Low-RAM variant (1.65x, 5.9 GB peak instead of 7.0 GB):** use the 0.5B model for easy questions;
it then also serves as the 7B's draft model, loaded once:

```bash
cascade config set small_model qwen2.5:0.5b-instruct
cascade config unset draft_model
```

### Answer length and the cache

Every setting has a default you can change; nothing below is fixed in code.

```bash
cascade config set system_prompt "Answer in one sentence."   # sent with every request
cascade config set system_prompt '""'                         # send none
cascade config set local.max_tokens 256                       # longest local answer (default 512)
cascade config set cache.similarity_threshold 0.94            # stricter matching (default 0.92)
cascade config set cache.term_swap_guard lenient              # strict (default) | lenient | off
cascade config set cache.uncached_tiers '["self_hosted"]'     # never cache these tiers (default)
```

The default `system_prompt` asks for a direct answer sized to the question. Without one, a
3B model on a CPU spent 66 s writing 256 tokens to explain what an API is, versus 8 s and
33 tokens with it.

The cache matches questions by meaning, which on its own can't tell "boiling point in
Fahrenheit" from "boiling point in Celsius". So it also rejects a match when the new
question swaps in a term the stored answer never mentions, or has the same key words in a
different order ("5 kg to pounds" / "5 pounds to kg"). Questions the router sends to the
live-data tier, or that mention things like "today" or "current", are never cached.
`python -m scripts.run_benchmark` shows the trade-off of each cache setting on the
benchmark fixtures.

### Optional: hard queries to your own API key (`cloud_byok`)

Any OpenAI-compatible provider works (OpenAI, OpenRouter, Groq, Together, DeepSeek, a vLLM box...).
Keys are read from environment variables; `cascade config` refuses to store a key in the file.

```bash
cascade config set byok.base_url https://api.openai.com
cascade config set byok.model gpt-4o-mini
cascade config set byok.api_key_env OPENAI_API_KEY      # then export OPENAI_API_KEY=...
cascade config set byok.input_price_per_1k 0.00015      # optional, for the cost footer
cascade config set byok.output_price_per_1k 0.0006
```

### Optional: real speculative decoding (llama.cpp runner)

```bash
pip install -e ".[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cascade config set runner llamacpp
cascade config set acceleration draft_model       # or prompt_lookup, or none
cascade config set model qwen2.5-coder            # 7B target
cascade config set draft_model qwen2.5-coder:0.5b   # same tokenizer, ~14x smaller
```

The draft length adapts to how often drafts are accepted (`draft.adaptive`, on by default).
Every accelerated output can be checked against plain greedy decoding with
`LlamaCppRunner.verify_greedy()`, and the benchmark script does this for every output.

The llama.cpp runner loads the GGUF files Ollama already downloaded — nothing is fetched twice.
Whether acceleration actually helps depends on the machine; measure it first
(`python scripts/benchmark_acceleration.py`) — see [Benchmarks](#benchmarks).

### Optional: share a GPU box with your team (`self_hosted`)

Run a Cascade server on the machine with the GPU. It is OpenAI-compatible and sits in front of
any OpenAI-compatible upstream — vLLM or SGLang for real throughput, Ollama works for trying it out.
Each person gets their own API key and, optionally, a token quota.

On the GPU machine:

```bash
python -m gateway.serve create-user alice --quota 2000000   # omit --quota for unlimited
python -m gateway.serve create-key alice                    # prints csk_... once; only a hash is stored
CASCADE_UPSTREAM_URL=http://127.0.0.1:8001 CASCADE_UPSTREAM_MODEL=my-model \
  python -m gateway.serve run --host 0.0.0.0 --port 8000
```

On each laptop:

```bash
cascade config set self_hosted.url http://gpu-box:8000      # then export CASCADE_SERVER_KEY=csk_...
```

Other server commands: `revoke-key`, `usage USER`, `reset-usage USER`. Settings are environment
variables: `CASCADE_SERVER_DB`, `CASCADE_UPSTREAM_URL`, `CASCADE_UPSTREAM_MODEL`,
`CASCADE_UPSTREAM_API_KEY`. The server speaks plain HTTP — put it behind HTTPS (Caddy, nginx,
Tailscale...) before exposing it beyond a trusted network.

---

## What's in the box

| Component | Where | What it does |
|---|---|---|
| HardwareProfiler | `engine/execution/hardware_profiler.py` | Detects CPU/RAM/GPU/disk and benchmarks real tok/s at several quant levels |
| ModelLadder | `engine/execution/model_ladder.py` | Picks the highest quant rung the measured profile clears; steps down on OOM |
| Ollama runner | `engine/execution/model_runner.py` | Ollama HTTP API: real token counts and decode timings, model kept resident, pulls only when missing |
| llama.cpp runner | `engine/execution/llamacpp_runner.py` | In-process llama.cpp with prompt-lookup (Layer 1) and draft-model (Layer 3) speculative decoding |
| AccelerationEngine | `engine/execution/acceleration_engine.py` | Layer fallback chain + acceptance-rate monitor; reports only what the runner measured |
| ContextCompressor, SemanticCache, SemanticRouter, Pipeline | `engine/intelligence/` | Relevance-ranked compression, exact + fuzzy cache, trained 3-way router, orchestration |
| CloudBYOKBackend | `gateway/byok.py` | The `cloud_byok` tier, any OpenAI-compatible endpoint |
| SelfHostedBackend | `gateway/backend.py` | Client for the `self_hosted` tier |
| Cascade server | `gateway/api/app.py`, `gateway/store.py`, `gateway/serve.py` | OpenAI-compatible server: API-key auth, per-user token quotas (SQLite), streaming |
| CLI | `cli/main.py` | `init`, `ask`, `chat`, `stats`, `profile`, `doctor`, `config` |

All components meet at `shared/types.py` (`ExecutionBackend`, `BackendResult`, `PipelineResult`, ...),
so each backend is swappable without touching the Pipeline.

## Privacy

- The `local` tier and the cache never send anything off your machine.
- There is no default remote endpoint and no telemetry. A remote tier only exists once you configure it.
- Ollama's model registry is contacted only when a model you need isn't downloaded yet.
- API keys live in environment variables, never in `~/.cascade/config.json`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

No test needs a GPU, a running Ollama or the network. Live behaviour is covered by the scripts in
`scripts/` and recorded in [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

## Benchmarks

Measured on the development laptop (CPU only), not estimated. Full details, methodology and raw
numbers: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

| Metric | Target | Measured | |
|---|---|---|---|
| Router agreement on held-out queries | ≥85% | 89.2% (37 hand-labelled queries) | met |
| Wrong answers served from cache (19 near-miss pairs) | 0 | 0/19 (4/19 before the 0.3.1 guard) | met |
| Reworded questions answered from cache | ≥90% of exact | 42.9% (strict guard, threshold 0.92) | **not met** — traded for zero wrong answers |
| Cache-hit latency (live) | <150 ms | 55 ms median, 78 ms worst | met |
| Median answer time, 67 live requests | – | 4.7 s (was 23.7 s before the default system prompt) | |
| **Whole session, 34 mixed questions** (easy, hard, repeats, rewordings, near-misses) | – | **1.79–2.19x faster than the plain 7B** (two runs; 386 s → 216 s and 431 s → 197 s), median answer ~2 s, **34/34 correct**, peak RAM 7.0 GB (cache + router + 1.5B for easy questions + 7B with confidence drafting); low-RAM variant with the 0.5B shared as draft: 1.65x at 5.9 GB | |
| Speculative decoding speedup, CPU-only laptop | ≥1.5x | **1.69x on code/rewrite prompts**, 1.39x overall, 1.08x on open-ended prompts (qwen2.5-coder 7B + 0.5B draft, confidence drafting, 3 rounds); output verified lossless | met on grounded prompts, not yet overall |
| RAM, 7B Q4_K_M via llama.cpp | – | 4.6 GB (was 7.2 GB with the model file memory-mapped); +0.75 GB with the draft model | |
| Local decode speed, llama3.2 3B Q4_K_M | – | 4.5–8.7 tok/s (Ollama), 6.6 tok/s (llama.cpp CPU) | |

The 37-query router set, 15 paraphrase pairs and 19 near-miss pairs are small hand-made fixtures. Results from other
hardware — especially GPUs — are very welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Status and roadmap

Alpha. See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for what works, what was verified
live, and what's next — good places to contribute are listed there.

## Contributing

Issues and pull requests are welcome — start with [CONTRIBUTING.md](CONTRIBUTING.md).
Security problems: see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
