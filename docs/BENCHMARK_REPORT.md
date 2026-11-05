# Cascade — Benchmark Report

**Run date:** 2026-09-12 16:03:05 Pakistan Standard Time
**Run against:** `docs/benchmark/reference_queries.jsonl` (37 hand-labeled, held-out queries — NOT the real TriviaQA/GSM8K/HumanEval/MT-Bench 200-query set the master doc's Section 12 specifies, which needs a programmatic dataset download not done here; flagged as a placeholder exactly like `router/coldstart_data.py`'s BOOTSTRAP_EXAMPLES).

## What was actually measured

### SemanticRouter agreement (held-out, not training-set accuracy)

- Total queries: 37
- Correct: 33
- **Agreement rate: 89.2%** (target: ≥85%)
- Meets target: YES

Confusion matrix (expected tier → predicted tier counts):
```json
{
  "local": {
    "local": 14,
    "cloud_byok": 1
  },
  "cloud_byok": {
    "cloud_byok": 11,
    "local": 1
  },
  "cascade_cloud": {
    "cascade_cloud": 8,
    "cloud_byok": 1,
    "local": 1
  }
}
```

### SemanticCache hit rate

- Exact-duplicate hit rate: 100.0%
- Paraphrase hit rate: 93.3%
- Paraphrase-vs-exact ratio: 93.3%
- Meets ≥90%-of-exact target: YES

### Derived cost-reduction estimate (router contribution only — not a live cost measurement)

- Local-resolved fraction: 43.2%
- Projected cost reduction vs. pure-cloud baseline (router alone): 43.2%
- **Note:** Derived from the router measurement alone -- local-tier queries cost $0. Does NOT include a cache contribution: this benchmark's cache test only confirms correctness (re-reading the same queries it wrote), not a real hit rate on unseen production traffic, so folding it in would overstate this number. A real combined figure needs live traffic, not a fixture.

## What was NOT measured, and why

| Metric | Why not measured here |
|---|---|
| Speculative decoding speedup (NFR-01) | Needs a live Ollama instance with draft/target pairs loaded — run `scripts/benchmark_acceleration.py` separately against real models. |
| Hardware ladder success rate across ≥5 machines (R-06) | Inherently needs multiple different physical machines — see `docs/hardware_compat_results.md`. |
| Output quality regression vs. FP16 (NFR-11) | Needs a real quantized model loaded to compare against. |
| CascadeCloud GPU throughput / time-to-first-token | Needs an actual deployed vLLM/SGLang instance behind `gateway/backend.py`, not a fake provider. |

No number above this line was estimated and reported as measured — every figure came from a real run of the code in this repo, against the held-out reference set, on this machine, just now.

## Addendum — live-model (speculative decoding) attempt

Ollama is actually installed on this machine, with two real models
pulled (`qwen2.5-coder:7b`, `deepseek-coder-v2:16b`), so `scripts/benchmark_acceleration.py`
was run for real rather than skipped outright. Two things came out of
it, one fixed, one still blocking:

**Fixed:** `engine/execution/model_runner.py`'s `OllamaReferenceModelRunner`
crashed with `UnicodeDecodeError` on Windows — `subprocess`'s `text=True`
defaults to the system codepage (cp1252 here), and both `ollama pull`'s
progress output and real model-generated text routinely contain bytes
outside it. Fixed by passing `encoding="utf-8", errors="replace"`
explicitly on every subprocess call in that file. This is a real,
permanent fix, independent of anything below — logged in `shared/CHANGELOG.md`.

**Still blocking, and not this machine's fault specifically:** even
with the encoding bug fixed, a real Layer 1 vs. Layer 2 vs. Layer 3
speedup comparison needs one of two things neither of which exists yet:
a model with trained Medusa/EAGLE draft heads, or a validated
draft/target pair (`~/.cascade/validated_pairs.json`, produced by
`scripts/validate_draft_pairs.py` — which itself needs a joint
decision on which draft model to pair with which target, exactly the
open item Track A's own handoff doc flagged). Without either, every
layer falls back to Layer 1 (lookahead decoding) — so even a
fully-successful run on unlimited hardware would show Layer 1 ≈ Layer 2
≈ Layer 3 here, which is a **methodology gap, not a real NFR-01
result.**

On top of that, this machine has 2.8GB RAM free against a 4.7GB
model, and 2 of 3 layer-load attempts hit the 55-second timeout
(1 succeeded, at 3.6 tok/s). That number is real but not meaningful on
its own — it reflects this specific run's cold-load timing under RAM
pressure, not a property of the acceleration layers.

**Bottom line: NFR-01's speculative-decoding speedup target remains
genuinely unmeasured**, not failed — the prior "Overall: FAIL" the
script itself printed is an artifact of load timeouts and the missing
draft-pair/heads prerequisite, not evidence the acceleration layers
don't work. Measuring this for real needs: (1) a joint decision on a
draft/target model pair, (2) `scripts/validate_draft_pairs.py` run
against it, and (3) a machine with enough free RAM to reliably load
the target model within the timeout.

## Addendum 2 — the reference-model placeholder fix, verified live

Separately, `HardwareProfiler`'s reference-model placeholder
(`"cascade-reference-3b"`, which doesn't exist) was resolved to
`"llama3.2"` with real, verified quant tags. Once `llama3.2:3b`
actually finished pulling on this machine (slow — bandwidth, not a
code issue), `HardwareProfiler.profile()` was run for real,
end-to-end, for the first time in this project's history:

```
local_capable: True
measured_tok_s: {'3b-instruct-q4_K_M': 3.571}
peak_memory_gb: {'3b-instruct-q4_K_M': 0.534}
wall time: 75.1s (1 of 3 quant levels measured within the 60s time budget)
```

That run surfaced one more real bug: `peak_memory_gb` was always
measuring ~0, because it used `psutil` on the `ollama run` CLI
subprocess's PID — but `ollama run` is a thin client; the model's real
memory footprint lives in the separate `ollama serve` server process.
Fixed by polling Ollama's own `/api/ps` endpoint (which reports the
real figure) on a background thread while generation runs, instead of
inspecting the wrong process. Re-run after the fix: `peak_memory_gb`
now reports a real, plausible `0.534` GB for this quantized 3B model,
not zero.
## Addendum 3 — acceleration measured for real (NFR-01), 2026-11-05

### What changed before measuring

Until this run, `AccelerationEngine.generate_accelerated()` divided the
real measured generation time by hard-coded constants (1.15x for "Layer
1", 1.80x / 1.75x for Layers 2/3). Every `tok_s` the local backend
reported was inflated by at least 15%, and the NFR-01 unit tests passed
by construction. Those multipliers are gone: the engine now reports only
what the runner measured, and a layer counts as active only if the
runner actually ran it. The Ollama runner cannot do speculative decoding
at all, so it honestly reports none.

Real speculative decoding was added through `engine/execution/llamacpp_runner.py`
(llama-cpp-python, CPU wheel), reusing the GGUF files Ollama already stores:

- **Layer 1 — prompt lookup:** n-gram drafts copied from the context, no extra model.
- **Layer 3 — draft model:** `llama3.2:1b-instruct-q4_K_M` drafting 4 tokens per step for `llama3.2:3b-instruct-q4_K_M` (same tokenizer).
- Layer 2 (Medusa/EAGLE heads) remains unavailable: no trained heads exist for this model.

### Setup

`scripts/benchmark_acceleration.py`: 6 prompts (3 context-grounded —
code rewrites and a paragraph copy — and 3 open-ended), 128 max tokens,
greedy decoding (temperature 0), a warm-up call per mode, decode speed
timed from first to last token. Machine: Windows 11 laptop, 8 CPU
threads, 15.7 GB RAM, NVIDIA MX330 2 GB (unused — CPU wheel). Raw
per-prompt data: `docs/benchmark/acceleration_results.json`.

### Results

| Mode | Decode tok/s | vs plain | Grounded prompts | Open prompts | Mean draft acceptance | Output identical to plain |
|---|---:|---:|---:|---:|---:|---:|
| Plain decoding | 6.56 | 1.00x | 1.00x | 1.00x | – | – |
| Layer 1: prompt lookup | 3.94 | **0.60x** | 0.70x | 0.50x | 14.8% | 2 / 6 |
| Layer 3: 1B draft model | 5.85 | **0.89x** | 0.98x (1.14x wall-clock) | 0.80x | 70.9% | 2 / 6 |

**NFR-01 (≥1.5x) is not met on this machine.** Neither speculative
layer speeds up decoding on a CPU-only laptop. The draft model is
accepted often (71%; 87% on grounded prompts), but on a CPU verifying a
batch of drafted tokens costs nearly as much as generating them, so
there is little to gain. Speculative decoding pays off when the target
model is memory-bandwidth-bound with spare compute, which typically
means a GPU. The one prompt where prompt lookup helped (copying a
paragraph, 63% acceptance) ran at 1.19x.

**"Lossless" doesn't hold exactly in practice.** Only 2 of 6 accelerated
outputs were token-identical to plain greedy output. The algorithm
is lossless mathematically, but llama.cpp computes a batch of tokens
with slightly different floating-point results than one token at a
time, so greedy decoding can take a different path. The outputs stay
fluent and on-topic, but "identical output" is not something to promise.

### Consequences in the code

- Acceleration is now **off by default** (`LocalExecutionBackend(preferred_layer=NONE)`);
  it is an explicit per-machine opt-in (`cascade config set runner llamacpp`,
  `cascade config set acceleration draft_model`).
- `scripts/validate_draft_pairs.py` now measures a real acceptance rate
  and speedup, and only registers a pair if it is actually faster on the
  machine it ran on. On this laptop it would not register `llama3.2:1b → 3b`.
- The acceptance-rate monitor (FR-03.3) now works from real
  measurements, and can switch prompt lookup off too.

### Still unmeasured

NFR-01 on a GPU machine, which is where it is most likely to pass. Run
`python scripts/benchmark_acceleration.py` on one (with a CUDA/Metal
build of llama-cpp-python) and append the result here.

## Addendum 4 — end-to-end live run of every tier, 2026-11-05

Run on the same laptop through the real `cascade` CLI, with an isolated config directory:

| Check | Result |
|---|---|
| `cascade init` (HardwareProfiler via Ollama HTTP API) | local_capable=True, 4.5 tok/s on `3b-instruct-q4_K_M`, 0.55 GB measured |
| Local answer, short factual question | 4.9 s end to end (13.1 s before the `127.0.0.1` fix below) |
| Close paraphrase of a previous question | cache hit; warm lookup 22–55 ms (NFR-02 <150 ms met); first lookup in a new process ~2.4 s while the embedding model loads |
| Hard coding question with BYOK configured | routed to `cloud_byok` and answered by the BYOK endpoint (Ollama's OpenAI-compatible API standing in for a real provider) |
| Gateway: no key / spoofed `x-cascade-user` header | 401 / 401 |
| Gateway: completion and streamed completion | billed from real usage (35 and 37 tokens) |
| Gateway: request over the account's spend cap | 402 before any upstream call |
| Router picks `cascade_cloud` but the gateway refuses (cap) | answered locally instead of failing (NFR-03) |

Bugs this run found and fixed:

- **~2 s added to every local request on Windows.** `localhost` resolves to IPv6 `::1` first; Ollama listens on IPv4 only, so each urllib call stalled ~2 s before falling back (2.03 s vs 0.001 s measured). All local defaults now use `127.0.0.1`.
- **CLI crash on non-cp1252 characters** (e.g. `√`) when output is piped on Windows. The CLI now prints a replacement character instead of crashing.
- **SQLite schema setup** committed the wrapping transaction early (caught by the new tests before shipping).

The semantic cache threshold (0.92 cosine) is deliberately conservative: a paraphrase that also moves an instruction around ("…Answer in one sentence." → "In one sentence, …") scored 0.875 and was correctly treated as a miss.
