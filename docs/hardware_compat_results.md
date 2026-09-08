# Track A ? Hardware Compatibility Results

**Phase 3, Task 13** ? Definition of done: ?90% of 8 GB RAM+ machines get
a working local rung (R-06, Section 12 benchmark target).

Required: ? 5 genuinely different real machines tested.

---

## Summary

| Machines tested | Working local rung (of 8 GB+ subset) | Pass rate | Target |
|-----------------|--------------------------------------|-----------|--------|
| TBD             | TBD / TBD                            | TBD       | ? 90%  |

---

## Machine Results

| # | Machine description | OS | CPU cores | RAM (GB) | GPU / VRAM | Best tok/s | Rung selected | Local rung? |
|---|--------------------|----|-----------|----------|------------|-----------|---------------|-------------|
| 1 | (your own machine) | -  | -         | -        | -          | -         | -             | -           |
| 2 | -                  | -  | -         | -        | -          | -         | -             | -           |
| 3 | -                  | -  | -         | -        | -          | -         | -             | -           |
| 4 | -                  | -  | -         | -        | -          | -         | -             | -           |
| 5 | -                  | -  | -         | -        | -          | -         | -             | -           |

---

## How to Run on a New Machine

`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the hardware profiler
python -c "
from engine.execution.hardware_profiler import HardwareProfiler, OllamaReferenceModelRunner
p = HardwareProfiler(OllamaReferenceModelRunner())
profile = p.profile()
print(profile)
"

# 3. Or run the full benchmark
python scripts/benchmark_acceleration.py
`

Record the output in the table above.

---

## Notes / Edge Cases

- Machines with < 8 GB RAM are expected to fail (not counted in the 90% target).
- CPU-only 8 GB machines should land on the q4_K_M (floor) rung.
- Apple Silicon: unified memory counts as VRAM; gpu_vendor = "apple".
- Windows: disk throughput may be lower due to NTFS overhead -- this does
  not affect rung selection (only tok/s matters for the ladder).

---

> **Status:** In progress ? fill in table as machines are tested.
