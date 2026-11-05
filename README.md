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
cascade config set acceleration prompt_lookup     # or draft_model, or none
```

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
| Paraphrase cache hit rate vs exact | ≥90% | 93.3% | met |
| Warm cache-hit lookup latency | <150 ms | 22–55 ms | met |
| Speculative decoding speedup, CPU-only laptop | ≥1.5x | prompt lookup 0.60x, 1B draft model 0.89x | **not met** — acceleration is off by default |
| Local decode speed, llama3.2 3B Q4_K_M | – | 4.5–8.7 tok/s (Ollama), 6.6 tok/s (llama.cpp CPU) | |

The 37-query router set and 15 paraphrase pairs are small hand-made fixtures. Results from other
hardware — especially GPUs — are very welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Status and roadmap

Alpha. See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for what works, what was verified
live, and what's next — good places to contribute are listed there.

## Contributing

Issues and pull requests are welcome — start with [CONTRIBUTING.md](CONTRIBUTING.md).
Security problems: see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
