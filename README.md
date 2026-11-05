# Cascade

### Run the right model in the right place — automatically.

Cascade is a local-first LLM runtime. It benchmarks your machine, runs a
model locally when your hardware can handle the query, answers repeated
questions from a semantic cache, and routes hard queries to your own
cloud API key or to a hosted, metered gateway — only when you've
configured one.

```text
query ─▶ ContextCompressor ─▶ SemanticCache ──hit──▶ answer (free, instant)
                                   │ miss
                                   ▼
                            SemanticRouter (trained classifier)
              ┌────────────────────┼─────────────────────┐
           local                cloud_byok           cascade_cloud
   Ollama or llama.cpp      your own key, any      Cascade's gateway:
   (+ speculative decoding)  OpenAI-compatible      auth, spend caps,
                             provider               Stripe metering
              └────────────────────┼─────────────────────┘
                                   ▼
          confidence check ─▶ escalate if weak ─▶ cache write + decision log
```

Every tier degrades gracefully: if a cloud tier is down, mis-keyed or over
its cap, the request is answered locally instead of failing.

---

## Quickstart

Requirements: Python 3.10+, [Ollama](https://ollama.com) running locally.

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
cascade stats       # cache hit rate, local-resolution rate, speculative speedup, cost
```

Each answer ends with a status footer, e.g. `[local | 9.8 tok/s | 2140ms | $0.0000]`.

### Optional: hard queries to your own API key (cloud_byok)

Any OpenAI-compatible provider works (OpenAI, OpenRouter, Groq, Together, DeepSeek, a vLLM box...).
Keys are read from environment variables; `cascade config` refuses to store a key in the file.

```bash
cascade config set byok.base_url https://api.openai.com
cascade config set byok.model gpt-4o-mini
cascade config set byok.api_key_env OPENAI_API_KEY      # then export OPENAI_API_KEY=...
cascade config set byok.input_price_per_1k 0.00015      # optional, for cost estimates
cascade config set byok.output_price_per_1k 0.0006
```

### Optional: real speculative decoding (llama.cpp runner)

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cascade config set runner llamacpp
cascade config set acceleration prompt_lookup     # or draft_model, or none
```

The llama.cpp runner loads the GGUF files Ollama already downloaded — nothing is fetched twice.
Whether acceleration actually helps depends on the machine; measure it first
(`python scripts/benchmark_acceleration.py`) — see [Benchmarks](#benchmarks).

### Optional: run CascadeCloud (the hosted, metered tier)

The gateway is OpenAI-compatible. Point it at any OpenAI-compatible upstream
(a vLLM/SGLang deployment in production; Ollama works for local testing).

```bash
python -m gateway.serve create-account alice --cap 5.00
python -m gateway.serve create-key alice                 # prints csk_... once; only a hash is stored
CASCADE_UPSTREAM_URL=http://127.0.0.1:11434 python -m gateway.serve run --port 8000
```

```bash
cascade config set cascade_cloud.url http://127.0.0.1:8000   # then export CASCADE_API_KEY=csk_...
```

Server settings (all environment variables): `CASCADE_GATEWAY_DB`, `CASCADE_UPSTREAM_URL`,
`CASCADE_UPSTREAM_MODEL`, `CASCADE_UPSTREAM_API_KEY`, `CASCADE_PRICE_PER_1K`, `STRIPE_API_KEY`
(enables metered Stripe billing).

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
| CascadeCloudBackend | `gateway/backend.py` | Client for the hosted tier; streamed requests are billed from real usage |
| Gateway | `gateway/api/app.py`, `gateway/store.py`, `gateway/serve.py` | OpenAI-compatible server: API-key auth, persistent spend caps (SQLite), streaming, Stripe metering |
| CLI | `cli/main.py` | `init`, `ask`, `chat`, `stats`, `profile`, `doctor`, `config` |

All components meet at `shared/types.py` (`ExecutionBackend`, `BackendResult`, `PipelineResult`, ...),
so each backend is swappable without touching the Pipeline (NFR-07).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

No test needs a GPU, a running Ollama, the network or Stripe (NFR-09). Live behaviour is covered
by the scripts in `scripts/` and recorded in `docs/BENCHMARK_REPORT.md`.

## Benchmarks

Measured on this project's development laptop, not estimated. Full details, methodology and raw
numbers: [`docs/BENCHMARK_REPORT.md`](docs/BENCHMARK_REPORT.md).

| Metric | Target | Measured | |
|---|---|---|---|
| Router agreement on held-out queries | ≥85% | 89.2% (37 hand-labelled queries) | met |
| Paraphrase cache hit rate vs exact | ≥90% | 93.3% | met |
| Warm cache-hit lookup latency (NFR-02) | <150 ms | 22–55 ms | met |
| Speculative decoding speedup (NFR-01), CPU-only laptop | ≥1.5x | prompt lookup 0.60x, 1B draft model 0.89x | **not met** — acceleration is off by default |
| Local decode speed, llama3.2 3B Q4_K_M | – | 4.5–8.7 tok/s (Ollama), 6.6 tok/s (llama.cpp CPU) | |

The 37-query router set and 15 paraphrase pairs are small hand-made fixtures, not the planned
200-query TriviaQA/GSM8K/HumanEval/MT-Bench suite.

## Status

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for what is done, what was found and fixed,
and what still needs a decision or an account rather than code.

## License

No license file has been chosen yet — all rights reserved by the authors until one is added.
