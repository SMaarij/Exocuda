"""Usage: python scripts/analysis/evaluate_draft_model.py qwen2.5:0.5b-instruct [more tags...] [--out OUT.json]

Would another draft model make the 7B faster? For each candidate draft model (an Ollama tag):

1. tokenizer check: it must give the 7B's token ids, or llama.cpp drafting can't use it;
2. agreement: the recorded 7B greedy answers (6 benchmark prompts, 8 chat turns) are re-scored
   with the candidate in one eval; how often is its top token the 7B's, by group;
3. cost: its median single-token eval time relative to the 7B's on this machine;
4. predicted speed-up: the shipped cost-aware drafting policy replayed on those sequences with
   the measured verify costs and this draft cost (no overheads, so read it as a ranking).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llama_cpp import Llama  # noqa: E402

from cli.main import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from engine.execution.llamacpp_runner import LlamaCppRunner, verify_costs_for  # noqa: E402
from scripts.benchmark_acceleration import PROMPTS  # noqa: E402
from scripts.benchmark_chat import USER_TURNS, USER_TURNS_B  # noqa: E402

TARGET = "qwen2.5-coder:7b-instruct-q4_K_M"
PROBE = (b"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n" * 16)


def model_path(tag: str) -> str:
    if Path(tag).is_file():  # a GGUF file, e.g. a re-quantized draft
        return tag
    runner = LlamaCppRunner(draft_model_tag=None)
    return runner._resolve_path(tag)


def recorded_cases() -> list[tuple[object, str, str]]:
    race = json.loads((ROOT / "docs/benchmark/acceleration_qwen7b_lookup.json").read_text())
    rows = race["per_prompt"]["none"][: len(PROMPTS)]
    cases = [(PROMPTS[i][1], rows[i]["text"], "code" if PROMPTS[i][0] == "grounded" else "open")
             for i in range(len(PROMPTS))]
    chat = json.loads((ROOT / "docs/benchmark/chat_turns_qwen7b.json").read_text())
    msgs = {c: [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] for c in "ab"}
    for r in chat["alternating"]["rounds"]["native:alternate"][0]:
        m = msgs[r["chat"]]
        m.append({"role": "user", "content": (USER_TURNS if r["chat"] == "a" else USER_TURNS_B)[r["turn"] - 1]})
        cases.append((list(m), r["text"], "code" if r["chat"] == "a" else "open"))
        m.append({"role": "assistant", "content": r["text"]})
    return cases


def single_token_ms(llm: Llama, repeats: int = 40) -> float:
    ids = llm.tokenize(PROBE)[:200]
    llm.reset()
    llm.eval(ids)
    times = []
    for i in range(repeats):
        t = time.perf_counter()
        llm.eval([ids[i % 50]])
        times.append(time.perf_counter() - t)
    times.sort()
    return 1000 * times[len(times) // 2]


def simulate(seqs: list[dict], costs: list[float], draft_cost: float) -> float:
    """Tokens per unit cost of the cost-aware linear policy (ModelDraft._cost_aware_draft)."""
    toks = total = 0.0
    for s in seqs:
        tgt, top, p1 = s["target"], s["top1"], s["p1"]
        n, i = len(tgt), 0
        while i < n:
            best, best_rate, whole, exp = 0, 1.0 / costs[1], 1.0, 0.0
            for L in range(1, min(16, n - i) + 1):
                whole *= p1[i + L - 1]
                exp += whole
                rate = (1 + exp) / (costs[L + 1] + draft_cost * L)
                if rate > best_rate:
                    best, best_rate = L, rate
            acc = 0
            for k in range(best):
                if top[i + k] != tgt[i + k]:
                    break
                acc += 1
            total += costs[best + 1] + draft_cost * best
            i += acc + 1
        toks += n
    return toks / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("drafts", nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--at-cost", type=float, help="also predict each candidate as if it cost this much "
                        "(what a model with its agreement would give at that price)")
    args = parser.parse_args()

    target = Llama(model_path(TARGET), n_ctx=2048, verbose=False, use_mmap=False, flash_attn=True)
    target_ms = single_token_ms(target)
    costs = verify_costs_for(target, model_path(TARGET), 17)
    probe_ids = list(target.tokenize("Hello, world! def f(x): return x ** 2  # ünïcödé".encode()))
    cases = recorded_cases()
    chat = LlamaCppRunner(draft_model_tag=None)
    chat._llm = target
    prompts = [chat._chat_tokens(p) for p, _, _ in cases]
    answers = [list(target.tokenize(a.encode(), add_bos=False)) for _, a, _ in cases]
    del target
    print(f"7B single token: {target_ms:.0f} ms", flush=True)

    results = {}
    for tag in args.drafts:
        llm = Llama(model_path(tag), n_ctx=4096, logits_all=True, verbose=False, use_mmap=False, flash_attn=True)
        same_tokens = list(llm.tokenize("Hello, world! def f(x): return x ** 2  # ünïcödé".encode())) == probe_ids
        ms = single_token_ms(llm)
        seqs = []
        agree = {"code": [0, 0], "open": [0, 0]}
        for ids, out, (_, _, group) in zip(prompts, answers, cases):
            llm.reset()
            llm.eval(ids + out)
            logits = np.asarray(llm.scores[len(ids) - 1: len(ids) + len(out) - 1])
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            top1 = probs.argmax(axis=1)
            seqs.append({"group": group, "target": out, "top1": top1.tolist(), "p1": probs.max(axis=1).tolist()})
            agree[group][0] += int((top1 == np.asarray(out)).sum())
            agree[group][1] += len(out)
        d = ms / target_ms
        row = {"same_tokens": same_tokens, "single_token_ms": round(ms, 1), "draft_cost": round(d, 3),
               "agreement": {g: round(a / n, 3) for g, (a, n) in agree.items()},
               "predicted": {g: round(simulate([s for s in seqs if s["group"] == g], costs, d), 3)
                             for g in ("code", "open")}}
        if args.at_cost:
            row["predicted_at_cost"] = {g: round(simulate([s for s in seqs if s["group"] == g], costs, args.at_cost), 3)
                                        for g in ("code", "open")}
        results[tag] = row
        print(f"{tag:34s} tokens {'same' if same_tokens else 'DIFFERENT'}  {ms:5.1f} ms (cost {d:.3f})  "
              f"agree code {row['agreement']['code']:.1%} open {row['agreement']['open']:.1%}  "
              f"predicted code {row['predicted']['code']:.3f}x open {row['predicted']['open']:.3f}x"
              + (f"  | at cost {args.at_cost}: code {row['predicted_at_cost']['code']:.3f}x "
                 f"open {row['predicted_at_cost']['open']:.3f}x" if args.at_cost else ""), flush=True)
        del llm
    if args.out:
        args.out.write_text(json.dumps({"target_ms": target_ms, "results": results}, indent=1))


if __name__ == "__main__":
    main()
