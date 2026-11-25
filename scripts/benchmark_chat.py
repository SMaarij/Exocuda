"""Multi-turn chat on the llama.cpp runner: time to first token and decode speed per turn.

Each configuration holds one conversation (Cascade's default system prompt and USER_TURNS),
feeding its own answers back as the next turn's history -- what a chat client does. Rounds
interleave the configurations so drift on the machine hits each alike.

Configurations (options joined by ':'):
    flat       the conversation flattened into one "ROLE: content" user turn (Cascade <= 0.8)
    native     every message keeps its role in the model's own chat template
    nohistory  no lookup drafting from earlier answers
    alternate  two conversations (USER_TURNS, USER_TURNS_B) taking turns on one model
    noslots    no saved KV caches (kv_slot_tokens 0): each switch recomputes the history

    python scripts/benchmark_chat.py --configs "flat:nohistory,native" --repeats 2 --out chat.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

USER_TURNS = [
    "Write a Python function parse_date(s) that parses 'YYYY-MM-DD' strings into datetime.date. Return only the code.",
    "Also accept 'DD/MM/YYYY'. Return the whole function again.",
    "Raise a ValueError with a clear message for anything else. Return the whole function again.",
    "Add a docstring and type hints. Return the whole function again.",
]

USER_TURNS_B = [
    "Explain in two sentences what a Python generator is.",
    "Show a short example of one that yields the Fibonacci numbers.",
    "How would I take only the first 10 values from it?",
    "And what changes if I make it an async generator?",
]


def run_config(spec: str, model: str, quant: str, draft: str, max_tokens: int) -> list[dict]:
    from cli.main import DEFAULT_SYSTEM_PROMPT
    from engine.execution.backend import _messages_to_prompt
    from engine.execution.llamacpp_runner import LlamaCppRunner

    options = set(spec.split(":"))
    runner = LlamaCppRunner(temperature=0.0, draft_model_tag=draft, draft_tokens=16, draft_verify_costs="auto",
                            draft_lookup_history="nohistory" not in options,
                            **({"kv_slot_tokens": 0} if "noslots" in options else {}))
    runner.set_acceleration("draft_model")
    runner.load(model, quant)
    runner.generate("Say OK.", 8)  # warm-up
    chats = {"a": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]}
    if "alternate" in options:
        chats["b"] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        order = [(chat, i) for i in range(len(USER_TURNS)) for chat in ("a", "b")]
    else:
        order = [("a", i) for i in range(len(USER_TURNS))]
    rows = []
    for chat, i in order:
        turn, text = i + 1, (USER_TURNS if chat == "a" else USER_TURNS_B)[i]
        messages = chats[chat]
        messages.append({"role": "user", "content": text})
        prompt = messages if "native" in options else _messages_to_prompt(messages)
        t = time.perf_counter()
        sample = runner.generate(prompt, max_tokens)
        wall = time.perf_counter() - t
        messages.append({"role": "assistant", "content": sample.output_text})
        rows.append({
            "chat": chat, "turn": turn, "prompt_tokens": len(runner.last_prompt_ids), "tokens": sample.tokens_generated + 1,
            "first_token_s": round(runner.last_prompt_s or 0.0, 2), "decode_tok_s": round(sample.tok_s, 3),
            "wall_s": round(wall, 2), "acceptance_rate": sample.acceptance_rate, "text": sample.output_text,
        })
        print(f"  {spec:<16} {chat} turn {turn}  prompt {rows[-1]['prompt_tokens']:4d} tok  first token "
              f"{rows[-1]['first_token_s']:6.2f}s  {sample.tok_s:5.2f} tok/s  wall {wall:6.1f}s", flush=True)
    runner.unload()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder")
    parser.add_argument("--quant", default="7b-instruct-q4_K_M")
    parser.add_argument("--draft", default="qwen2.5:0.5b-instruct")
    parser.add_argument("--configs", default="flat:nohistory,native")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    specs = args.configs.split(",")
    results: dict[str, list[list[dict]]] = {spec: [] for spec in specs}
    for _ in range(args.repeats):
        for spec in specs:
            results[spec].append(run_config(spec, args.model, args.quant, args.draft, args.max_tokens))

    print(f"\n{'config':<16} {'turns 2+ first token':>21} {'decode tok/s':>13} {'total wall':>11}")
    summary = {}
    for spec, rounds in results.items():
        later = [r["first_token_s"] for rows in rounds for r in rows if r["turn"] > 1]
        tok = sum(r["tokens"] for rows in rounds for r in rows)
        decode_s = sum(r["tokens"] / r["decode_tok_s"] for rows in rounds for r in rows if r["decode_tok_s"])
        wall = statistics.mean(sum(r["wall_s"] for r in rows) for rows in rounds)
        summary[spec] = {"later_first_token_s": round(statistics.mean(later), 2),
                         "decode_tok_s": round(tok / decode_s, 3), "mean_total_wall_s": round(wall, 1)}
        s = summary[spec]
        print(f"{spec:<16} {s['later_first_token_s']:>20.2f}s {s['decode_tok_s']:>13.2f} {s['mean_total_wall_s']:>10.1f}s")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rounds": results}, indent=1))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
