"""Usage: python scripts/analysis/draft_top2_coverage.py OUT.json

How often is the 7B's greedy token the 0.5B's 2nd choice? (open-ended text)

Uses the 7B's recorded greedy answers (benchmark JSONs), re-scores them with the 0.5B in one
eval with all logits, and tallies the rank of the 7B token in the 0.5B's distribution, by the
0.5B's top-1 probability.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llama_cpp import Llama  # noqa: E402

from cli.main import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from engine.execution.llamacpp_runner import LlamaCppRunner  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
paths = json.loads((Path.home() / ".cascade" / "model_paths.json").read_text())
llm = Llama(paths["qwen2.5:0.5b-instruct"], n_ctx=4096, logits_all=True, verbose=False, n_threads=4)
runner = LlamaCppRunner(draft_model_tag=None)
runner._llm = llm

cases = []  # (prompt messages or str, 7B answer text, kind)
race = json.loads((ROOT / "docs/benchmark/acceleration_qwen7b_lookup.json").read_text())
from scripts.benchmark_acceleration import PROMPTS  # noqa: E402
seen = set()
for row in race["per_prompt"]["none"]:
    key = (row["kind"], row["text"])
    if key in seen:
        continue
    seen.add(key)
    prompt = next(p for k, p in PROMPTS if k == row["kind"] and True)
    cases.append((row, row["kind"]))
# map rows to prompts by order within kind
by_kind = {}
for k, p in PROMPTS:
    by_kind.setdefault(k, []).append(p)
rows_none = race["per_prompt"]["none"][: len(PROMPTS)]
pairs = [(PROMPTS[i][1], rows_none[i]["text"], PROMPTS[i][0]) for i in range(len(PROMPTS))]

chat = json.loads((ROOT / "docs/benchmark/chat_turns_qwen7b.json").read_text())
from scripts.benchmark_chat import USER_TURNS, USER_TURNS_B  # noqa: E402
for rounds_key in ("alternating",):
    rows = chat[rounds_key]["rounds"]["native:alternate"][0]
    msgs = {"a": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
            "b": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]}
    for r in rows:
        turns = USER_TURNS if r["chat"] == "a" else USER_TURNS_B
        m = msgs[r["chat"]]
        m.append({"role": "user", "content": turns[r["turn"] - 1]})
        pairs.append((list(m), r["text"], "chat_" + ("code" if r["chat"] == "a" else "open")))
        m.append({"role": "assistant", "content": r["text"]})

bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
seqs = []
stats = {}  # kind -> bin -> [n, rank0, rank1, rank2_3]
for prompt, answer, kind in pairs:
    ids = runner._chat_tokens(prompt)
    out = list(llm.tokenize(answer.encode("utf-8"), add_bos=False, special=False))
    llm.reset()
    llm.eval(ids + out)
    logits = np.asarray(llm.scores[len(ids) - 1: len(ids) + len(out) - 1])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    order = np.argsort(-probs, axis=1)[:, :4]
    group = "open" if "open" in kind else "code"
    seqs.append({"group": group, "prompt": list(map(int, ids)), "target": out, "top": order[:, :4].tolist(),
                 "p": np.take_along_axis(probs, order[:, :4], axis=1).tolist()})
    for j, tok in enumerate(out):
        p1 = probs[j, order[j, 0]]
        b = next(i for i in range(len(bins) - 1) if bins[i] <= p1 < bins[i + 1])
        cell = stats.setdefault(group, {}).setdefault(b, [0, 0, 0, 0])
        cell[0] += 1
        rank = list(order[j]).index(tok) if tok in order[j] else 9
        if rank == 0:
            cell[1] += 1
        elif rank == 1:
            cell[2] += 1
        elif rank <= 3:
            cell[3] += 1

result = {}
for group, cells in stats.items():
    print(f"== {group}")
    tot = [0, 0, 0, 0]
    for b in sorted(cells):
        n, r0, r1, r23 = cells[b]
        tot = [t + c for t, c in zip(tot, cells[b])]
        print(f"  top1 p in [{bins[b]:.1f},{bins[b+1]:.1f}): n={n:4d}  top1 {r0/n:5.1%}  2nd {r1/n:5.1%}  3rd-4th {r23/n:5.1%}")
    n, r0, r1, r23 = tot
    print(f"  all: n={n}  top1 {r0/n:.1%}  2nd {r1/n:.1%}  3rd-4th {r23/n:.1%}  -> top1 miss covered by 2nd: {r1/max(n-r0,1):.1%}")
    result[group] = {str(b): cells[b] for b in cells}
Path(sys.argv[1]).write_text(json.dumps({"stats": result, "seqs": seqs})) if len(sys.argv) > 1 else None
