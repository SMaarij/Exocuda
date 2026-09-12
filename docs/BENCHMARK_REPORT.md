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