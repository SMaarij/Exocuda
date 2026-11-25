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

### Fastest setup (measured 1.65–2.11x on a mixed session, all answers correct)

```bash
pip install -e ".[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
cascade config set model qwen2.5-coder     # everything else below is automatic
```

With llama-cpp-python installed, Cascade picks its fastest setup by itself: the llama.cpp runner,
cost-aware drafting for the 7B (the draft length is chosen by what verifying it costs on your
machine, measured once on first load), and for qwen2.5 models a 0.5B model that answers easy
questions and also drafts for the 7B, loaded once. When the 0.5B is unsure of an answer (mean token probability
below `small_model_min_confidence`, default 0.75) the 7B answers instead: on 128 checked easy
questions that leaves 4 wrong answers, against 11 for the 0.5B alone and 6 for the 1.5B model
(BENCHMARK_REPORT Addendum 13). Any of it can be changed or switched off with `cascade config set`
(`runner ollama`, `acceleration none`, `small_model off`).

The first `cascade ask` starts a background model server on 127.0.0.1 and loads the models
(about a minute on a CPU laptop); after that, easy answers take about 2 seconds and repeated
questions come back from the cache in well under one. The server exits after 10 idle minutes
(`server.idle_minutes`) and frees the RAM; `cascade stop` frees it at once,
`cascade config set server.auto off` disables it.

### Install once, then just use your usual commands

```bash
cascade install --as-ollama
```

Cascade now starts whenever you log in and sits on Ollama's own port (11434), so `ollama run`,
Ollama-based apps and anything else that talks to Ollama use Cascade **with no setting at all**:

```bash
ollama run qwen2.5-coder "What is a hash table?"   # answered by Cascade: cache, routing, drafting, less RAM
ollama run llama3.2:3b                              # any other model: passed to the real Ollama, unchanged
ollama pull / list / rm / ps                        # passed to the real Ollama
```

The real Ollama moves to port 11435, started by Cascade. Cascade answers its configured model family
(`cascade config set model ...`) and passes every other request through untouched. It loads no model
until a program asks for one and unloads them after `server.idle_minutes` (default 10), so an idle
machine holds nothing in RAM: measured 0.02 GB before the first request, 6.29 GB with the 7B, the
0.5B and the embedder loaded, 0.65 GB after the idle unload (BENCHMARK_REPORT Addendum 16).

On Windows `cascade install --as-ollama` also sets aside Ollama's own login shortcut and stops the
running Ollama app; on macOS and Linux it tells you the one step that needs your consent (removing
Ollama from Login Items, or `sudo systemctl disable --now ollama`). `cascade uninstall` removes
the login entry, stops Cascade and its Ollama, puts Ollama's shortcut back and restarts the Ollama
app. Plain `cascade install` (without `--as-ollama`) starts Cascade at login on port 8765 and leaves
Ollama alone.

### Use Cascade from the tools you already have, without installing

```bash
cascade start
```

Cascade keeps running in the background and answers the **Ollama** and **OpenAI** APIs, so existing
tools run on it unchanged and get its cache, small/big routing, speculative decoding and memory
savings:

```bash
# Ollama CLI and Ollama-based apps: point them at Cascade in that terminal or app
OLLAMA_HOST=127.0.0.1:8765 ollama run qwen2.5-coder "What is a hash table?"   # bash
$env:OLLAMA_HOST="127.0.0.1:8765"; ollama run qwen2.5-coder                    # PowerShell
```

```python
# OpenAI SDK, LangChain, IDE assistants, Open WebUI ...: base URL + any key
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")
client.chat.completions.create(model="any", messages=[{"role": "user", "content": "Hi"}])
```

Set `OLLAMA_HOST` per terminal or app, not system-wide: the Ollama app itself listens on that
address. Cascade answers its configured model family (and names Ollama doesn't have, such as
`gpt-4o` from an OpenAI tool); other Ollama models, pulls and listings go to the real Ollama. After `server.idle_minutes` without a
request it unloads the models to free their RAM, stays reachable, and reloads them on the next
request; `cascade stop` ends it.

Models load with flash attention and a smaller compute batch, which saves 0.15 GB at 2k context and 0.40 GB
at 8k for the 7B with identical output (Addendum 15).

Measured on a 4-core laptop: the 7B through Ollama alone holds 5.06 GB; Cascade holds 5.47 GB for the
7B **plus** the 0.5B model and the embedding model, and answers repeated questions in 0.7 s and easy
ones in about 2 s (BENCHMARK_REPORT Addendum 14).

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
| Reworded questions answered from cache | ≥90% of exact | **54/55** rewordings (14/15 fixture — the 15th is time-sensitive and never cached — and 40/40 held out) with 0/59 near-misses served: the main local model judges close-but-unsure matches, 2.4 s median (0.11.0); 42.9% before | met |
| Cache-hit latency (live) | <150 ms | 55 ms median, 78 ms worst | met |
| Median answer time, 67 live requests | – | 4.7 s (was 23.7 s before the default system prompt) | |
| **Whole session, 34 mixed questions** (easy, hard, repeats, rewordings, near-misses) | – | **1.79–2.19x faster than the plain 7B** (two runs; 386 s → 216 s and 431 s → 197 s), median answer ~2 s, **34/34 correct**, peak RAM 7.0 GB (cache + router + 1.5B for easy questions + 7B with confidence drafting); 0.5B shared as draft with unsure answers re-asked: 1.65–2.11x at 5.9 GB, now the recommended setup | |
| Speculative decoding speedup, CPU-only laptop | ≥1.5x | **1.58x overall**, 1.88x on code/rewrite prompts, 1.27x on open-ended prompts (qwen2.5-coder 7B + 0.5B draft, cost-aware draft length, 2 rounds); output verified lossless. 0.8.0 adds lookup drafting: on a busy laptop 1.49x vs 1.36x without it, open-ended 1.24x vs 1.08x (CPU llama.cpp speculative is usually ~1.0x on chat) | met |
| Time to first token, later chat turns | – | ~4 s instead of ~42 s (computed prompt prefix kept, 677-token system prompt); 0.9.0 native chat turns: 1.96 s vs 5.65 s, whole 4-turn chat 70 s vs 93 s; two alternating chats 2–4 s vs 10.9 s with saved KV caches | |
| Wait before the first words appear (`ollama run`, OpenAI streaming) | – | ~2–5 s instead of the whole answer (36–59 s): replies stream token by token (0.10.0) | |
| First token after an idle unload or restart | – | 3.1 s instead of 18.1 s (4-turn chat): KV caches saved to `~/.cascade/kv` and restored (0.10.0) | |
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
