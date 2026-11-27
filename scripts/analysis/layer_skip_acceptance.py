"""Usage: python scripts/analysis/layer_skip_acceptance.py DATA.json [--workdir DIR]

Self-speculative decoding with skipped layers, on a CPU: would the 7B with some of its layers
removed be a better draft than the 0.5B? For each layer pattern it writes a GGUF holding only the
kept transformer blocks (renumbered; embeddings, final norm and output head kept), then measures
on the 7B's own greedy outputs (DATA.json from draft_top2_coverage.py):

- per-token agreement: how often the truncated model's argmax equals the 7B's next token given
  the 7B's text so far (the per-token acceptance of a greedy draft), by prompt group;
- its single-token decode time relative to the full 7B (the draft cost).

Addendum 26 compares these against the break-even acceptance computed from the measured
verify-cost table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from llama_cpp import Llama  # noqa: E402

from cascade_local.engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402

TARGET = "qwen2.5-coder:7b-instruct-q4_K_M"


def truncate(src: str, keep: list[int], out: Path) -> None:
    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(out), arch=arch, endianess=reader.endianess)
    count_key = f"{arch}.block_count"
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = len(keep) if field.name == count_key else field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    renumber = {old: new for new, old in enumerate(keep)}
    tensors = []
    for t in reader.tensors:
        name = t.name
        if name.startswith("blk."):
            _, idx, rest = name.split(".", 2)
            if int(idx) not in renumber:
                continue
            name = f"blk.{renumber[int(idx)]}.{rest}"
        tensors.append((name, t))
    for name, t in tensors:
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def load(path: str) -> Llama:
    return Llama(path, n_ctx=4096, logits_all=True, verbose=False, use_mmap=False, flash_attn=True)


def agreement(llm: Llama, seqs: list[dict]) -> dict[str, float]:
    hits: dict[str, list[int]] = {}
    for s in seqs:
        llm.reset()
        ids = list(s["prompt"]) + list(s["target"])
        llm.eval(ids)
        logits = np.asarray(llm.scores[: len(ids)])
        start = len(s["prompt"]) - 1
        pred = logits[start: len(ids) - 1].argmax(axis=1)
        h = hits.setdefault(s["group"], [0, 0])
        h[0] += int((pred == np.asarray(s["target"])).sum())
        h[1] += len(s["target"])
    return {g: round(a / b, 4) for g, (a, b) in hits.items()}


def step_ms(llm: Llama, seq: dict) -> float:
    llm.reset()
    llm.eval(list(seq["prompt"]))
    base = llm.n_tokens
    out = []
    for tok in seq["target"][:24]:
        t = time.perf_counter()
        llm.eval([tok])
        out.append(time.perf_counter() - t)
    llm.n_tokens = base
    return 1000 * statistics.median(out)


def patterns(n: int) -> dict[str, list[int]]:
    """Kept-layer sets: first/last layers matter most in the literature; skip in the middle."""
    out = {"full": list(range(n))}
    for frac in (0.6, 0.5, 0.4):
        k = round(n * frac)
        head, tail = 3, 2
        mid = k - head - tail
        inner = list(range(head, n - tail))
        pick = [inner[round(i * (len(inner) - 1) / max(mid - 1, 1))] for i in range(mid)]
        out[f"keep{k}_spread"] = sorted(set(list(range(head)) + pick + list(range(n - tail, n))))
        out[f"keep{k}_dropmid"] = sorted(set(list(range(k - tail)) + list(range(n - tail, n))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--workdir", default=str(ROOT.parent / "tmp_layerskip"))
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    seqs = json.loads(Path(a.data).read_text())["seqs"]
    src = LlamaCppRunner(draft_model_tag=None)._resolve_path(TARGET)
    n = gguf.GGUFReader(src).fields["qwen2.block_count"].contents()
    work = Path(a.workdir)
    work.mkdir(exist_ok=True)
    results = {}
    full_ms = None
    for name, keep in patterns(n).items():
        if name == "full":
            llm = load(src)
        else:
            out = work / "skip.gguf"
            truncate(src, keep, out)
            llm = load(str(out))
        ms = step_ms(llm, seqs[0])
        full_ms = full_ms or ms
        agr = agreement(llm, seqs)
        results[name] = {"keep": keep, "agreement": agr, "step_ms": round(ms, 1), "cost": round(ms / full_ms, 3)}
        print(f"{name:16s} layers {len(keep):2d}  cost {ms / full_ms:.2f}  agreement {agr}", flush=True)
        del llm
    if a.save:
        (ROOT / "docs" / "benchmark" / "layer_skip_acceptance.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
