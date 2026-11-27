"""
scripts/record_benchmark_answers.py

Records a real local-model answer for every cached question in the cache
benchmark fixtures (the "original" of each pair in
docs/benchmark/cache_paraphrase_pairs.jsonl and
docs/benchmark/cache_near_miss_pairs.jsonl), and stores it in the row's
"answer" field.

Why: SemanticCache's term-swap guard looks at the stored answer, so the
benchmark has to use answers a model actually gives -- not the question
text standing in for one, which makes the guard look far stricter than it
is on real traffic.

The answers are generated the way `cascade ask` generates them: the CLI's
default system prompt, flattened into the local prompt by the same
function LocalExecutionBackend uses.

Usage (needs Ollama running):
    python -m scripts.record_benchmark_answers [--model llama3.2] [--quant 3b-instruct-q4_K_M]
        [--max-tokens 256] [--system-prompt TEXT] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade_local.cli.main import DEFAULT_SYSTEM_PROMPT
from cascade_local.engine.execution.backend import _messages_to_prompt
from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner

BENCHMARK_DIR = Path(__file__).parent.parent / "docs" / "benchmark"
FIXTURES = ("cache_paraphrase_pairs.jsonl", "cache_near_miss_pairs.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--quant", default="3b-instruct-q4_K_M")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--overwrite", action="store_true", help="Re-record rows that already have an answer.")
    args = parser.parse_args(argv)

    runner = OllamaReferenceModelRunner()
    runner.load(args.model, args.quant)
    source = f"{args.model}:{args.quant}"

    for name in FIXTURES:
        path = BENCHMARK_DIR / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row.get("answer") and not args.overwrite:
                continue
            messages = [{"role": "user", "content": row["original"]}]
            if args.system_prompt:
                messages.insert(0, {"role": "system", "content": args.system_prompt})
            sample = runner.generate(_messages_to_prompt(messages), args.max_tokens)
            row["answer"] = sample.output_text.strip()
            row["answer_source"] = source
            print(f"{row['original'][:50]:<50} -> {row['answer'][:70]!r}", flush=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
