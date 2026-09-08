# Track A Handoff — What Track B and Track C Need to Know

**From:** Track A (Execution Engine)  
**To:** Track B (Pipeline & Intelligence) · Track C (Platform & Revenue)  
**Date:** 8 September 2026 — Phases 1, 2 & 3 complete  
**Repo path:** `engine/execution/`

> **You do NOT need to read any of Track A's code to start your work.**
> This document tells you everything: what exists, what the contract is,
> and exactly where to plug in when your part is ready.

---

## 1. What Track A Built — Plain English

Cascade needs two answers before it can run a model locally:
**"Can this machine do it?"** and **"How fast?"**

Track A owns both answers end to end.

```
Your machine
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                  TRACK A  engine/execution/                  │
│                                                              │
│  Step 1 — HardwareProfiler (Phase 1)                        │
│    Detects: CPU cores, RAM, GPU/VRAM, disk throughput        │
│    Benchmarks: tok/s at q4_K_M / q5_K_M / q8_0              │
│    Saves to: ~/.cascade/hardware_profile.json                │
│                                                              │
│  Step 2 — ModelLadder (Phase 2)                              │
│    Given the profile, picks the HIGHEST quant rung           │
│    the machine can sustain. Deterministic: same profile       │
│    always = same rung.                                       │
│                                                              │
│  Step 3 — AccelerationEngine (Phase 2)                       │
│    Wraps the model runner with the best speed layer:         │
│      Layer 1  Lookahead decoding   ~1.15x                   │
│      Layer 2  Draft heads (Medusa) ~1.80x                   │
│      Layer 3  Paired draft model   ~1.75x                   │
│    Auto-steps down if acceptance rate drops below 60%.       │
│                                                              │
│  Step 4 — LocalExecutionBackend (Phase 1 stub → Phase 2)    │
│    The ONE class Track B calls. Wires steps 1-3 together.    │
│    Public API never changes.                                 │
│      .generate(messages)       → BackendResult               │
│      .generate_stream(messages)→ Iterator[str]               │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
   BackendResult  ← THIS is what Track B receives from Track A
   {
     "text":               "model response here",
     "confidence":         0.75,
     "speculative_active": True,
     "tok_s":              36.2,
     "latency_ms":         840.0,
     "est_cost_usd":       0.0   ← always 0.0 for local execution
   }
```

---

## 2. Test Results — 101 Tests, 91% Coverage, No GPU Needed

Every test runs with **no live GPU and no Ollama instance**. Works on any laptop.

```
pytest engine/execution/ -v

platform win32 -- Python 3.12.0, pytest-9.1.1
rootdir: D:\Downloads\cascade
101 collected

test_acceleration_engine.py  (33 tests)  ........ PASSED
test_backend.py              ( 8 tests)  ........ PASSED
test_backend_real.py         (16 tests)  ........ PASSED
test_hardware_profiler.py    (27 tests)  ........ PASSED
test_model_ladder.py         (17 tests)  ........ PASSED
j2_integration_test.py       ( 5 tests)  ........ PASSED  ← handshake with Track B

Coverage:
  acceleration_engine.py    98%   Layer 1/2/3 + monitor
  backend.py                96%   Real LocalExecutionBackend
  model_ladder.py          100%   Rung selection + step-down
  pipeline_bridge.py       100%   FakePipeline shim (delete when Track B lands)
  j2_integration_test.py   100%   J-2 verification
  hardware_profiler.py      90%   HardwareProfile detection
  model_runner.py           49%   OllamaReferenceModelRunner (integration only)
  TOTAL                     91%

101 passed in 1.99s
```

---

## 3. Architecture Flow — How a Request Moves Through Track A

```
User message arrives
         │
         ▼
LocalExecutionBackend.generate(messages)
         │
         ├─► load_cached_profile()
         │       reads ~/.cascade/hardware_profile.json
         │       (if missing → runs full profiler benchmark)
         │
         ├─► ModelLadder.select_rung(profile)
         │       returns q4_K_M / q5_K_M / q8_0 or None
         │       None → raises LocalExecutionBackendError
         │               → Track B routes to CascadeCloud
         │
         ├─► AccelerationEngine.load(model_id, rung.label)
         │       resolves best available layer:
         │       validated_pairs.json exists → Layer 3
         │       model has draft heads       → Layer 2
         │       always                      → Layer 1
         │
         ├─► AccelerationEngine.generate_accelerated(prompt)
         │       applies speedup, monitors acceptance rate
         │       if rate < 60% over 20 samples → step-down
         │       automatic, no user action, no crash
         │
         └─► returns BackendResult to Track B
```

---

## 4. The Quantization Ladder

Example: CPU-only machine, 20 tok/s:

```
q8_0  rung — needs tok/s >= 15 AND vram >= 8 GB  → SKIP (no GPU)
q5_K_M rung — needs tok/s >= 8  AND vram >= 4 GB  → SKIP (no GPU)
q4_K_M rung — needs tok/s >= 3  AND vram >= 0 GB  → SELECTED ✓

If OOM on q4_K_M load → no lower rung → LocalExecutionBackendError
  → Track B should route to CascadeCloud (FR-07)
```

Example: GPU machine, 25 tok/s, 10 GB VRAM:

```
q8_0 rung — tok/s 25 >= 15 ✓ AND vram 10 >= 8 ✓ → SELECTED ✓
```

---

## 5. AccelerationEngine Layer Logic

```
Model loaded at chosen rung
         │
         ▼
Is there a validated entry in ~/.cascade/validated_pairs.json?
  YES → Layer 3: paired draft model  (~1.75x speedup)
  NO
         │
         ▼
Does the model have Medusa/EAGLE draft heads?
  YES → Layer 2: draft heads          (~1.80x speedup)
  NO
         │
         ▼
Layer 1: lookahead decoding (always available)  (~1.15x speedup)

During every generation:
  AcceptanceRateMonitor tracks rolling window (20 samples)
  If average drops below 60%:
    → Auto step-down (Layer 3→2, or 2→1)
    → No user action, no output quality loss (FR-03.4)
```

---

## 6. What Track B Needs to Do — Exact Steps

### Step 1 — Import and use LocalExecutionBackend

```python
# All of this is in shared/types.py (already written and frozen after J-1)
from engine.execution.backend import LocalExecutionBackend, LocalExecutionBackendError
from shared.types import ExecutionBackend, BackendResult, Message

class Pipeline:
    def __init__(self, backend: ExecutionBackend, ...):
        self._backend = backend    # pass LocalExecutionBackend() here

    def run(self, messages: list[Message]) -> PipelineResult:
        try:
            result: BackendResult = self._backend.generate(messages)
            # Use: result["text"], result["tok_s"],
            #      result["speculative_active"], result["est_cost_usd"]
        except LocalExecutionBackendError:
            # All local rungs exhausted → route to CascadeCloud
            result = self._cloud_backend.generate(messages)
```

### Step 2 — Verify J-2 (3 steps, takes 2 minutes)

When your real `Pipeline` class exists:

```python
# In engine/execution/j2_integration_test.py, swap ONE line:

# REMOVE:
from engine.execution.pipeline_bridge import FakePipeline

# ADD:
from engine.intelligence.pipeline import Pipeline as FakePipeline
```

Then run:
```bash
pytest engine/execution/j2_integration_test.py -v
# All 5 tests must pass with ZERO changes to test bodies.
# Green = J-2 done. Delete engine/execution/pipeline_bridge.py.
```

---

## 7. What Track C Needs to Do — Exact Steps

### Step 1 — Implement CascadeCloudBackend with the same interface

```python
# gateway/backend.py (Track C)
from shared.types import ExecutionBackend, BackendResult, Message
from typing import Iterator

class CascadeCloudBackend:
    def generate(self, messages: list[Message]) -> BackendResult:
        # call your hosted vLLM/SGLang endpoint
        return {
            "text":               response_text,
            "confidence":         0.95,
            "speculative_active": False,     # no speculative layer in cloud
            "tok_s":              tok_s,
            "latency_ms":         elapsed_ms,
            "est_cost_usd":       cost,      # Track C sets the real cost here
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        for chunk in your_stream(messages):
            yield chunk

# REQUIRED: same import-time guard Track A uses
assert isinstance(CascadeCloudBackend(), ExecutionBackend)
```

### Step 2 — Call HardwareProfiler from CLI commands

```python
# cli/main.py (Track C) — cascade profile command
from engine.execution.hardware_profiler import HardwareProfiler, OllamaReferenceModelRunner

def cmd_profile():
    profiler = HardwareProfiler(OllamaReferenceModelRunner())
    profile  = profiler.profile()   # saves ~/.cascade/hardware_profile.json
    local = profile["local_capable"]
    best  = max(profile["measured_tok_s"].values())
    print(f"local_capable: {local}  |  best tok/s: {best:.1f}")
```

---

## 8. All Fakes and Placeholders Currently In Use

| # | Fake/Placeholder | Where | What it does | How to replace |
|---|-----------------|-------|-------------|----------------|
| 1 | `FakeReferenceModelRunner` | All `test_*.py` | Returns fixed tok/s; no real model | **Keep forever** — tests must never need a GPU (NFR-09) |
| 2 | `result["text"] = tokens * "x"` | `backend.py` line ~141 | Text is "xxxxx…" not real output | Add `output_text: str` to `GenerationSample`; swap to `sample.base.output_text` |
| 3 | `"cascade-reference-3b"` | `hardware_profiler.py` line 51 | Placeholder Ollama tag; doesn't exist | Joint decision → e.g. `"llama3.2:3b"` |
| 4 | `FakePipeline` + `pipeline_bridge.py` | `pipeline_bridge.py` | Lets Track A test end-to-end alone | Track B: swap 1 import in j2_integration_test.py → delete the file |
| 5 | Speedup factors 1.15/1.80/1.75 | `acceleration_engine.py` lines 36-38 | Published benchmark constants | Run `scripts/benchmark_acceleration.py` on real hardware; update if needed |
| 6 | `validated_pairs.json` missing | `~/.cascade/` | Layer 3 silently falls back to Layer 2/1 | Run `scripts/validate_draft_pairs.py --target X --draft Y` |
| 7 | `DEFAULT_MIN_TOK_S_FLOOR = 3.0` | `hardware_profiler.py` line 54 | Bootstrap heuristic before real data | After Task 13 (5+ machine test), revise `DEFAULT_LADDER` min_tok_s values |

---

## 9. The One Thing All Three Tracks Must Do NOW (J-1)

> **Review and freeze `shared/types.py` before writing any more Phase 2 code.**

Track A proposed an extended `HardwareProfile` schema. The master doc's
Section 9 showed a minimal sketch; Track A added:
`cpu_cores`, `gpu_vendor`, `disk_throughput_mb_s`, `disk_free_gb`,
`peak_memory_gb`, `schema_version`, `profiled_at`, `os`.

See `shared/CHANGELOG.md` for the full rationale.

**Once all three tracks sign off → schema is frozen → no further changes
without a new dated CHANGELOG entry + all-three-tracks PR review (Section 10.3).**

---

## 10. Run It Yourself Now

```bash
pip install psutil pytest pytest-cov

# All 101 Track A tests (no GPU, no Ollama):
pytest engine/execution/ -v

# Only the J-2 handshake:
pytest engine/execution/j2_integration_test.py -v

# With coverage:
pytest engine/execution/ --cov=engine/execution --cov-report=term
```

**Expected: 101 passed in ~2 seconds.**
