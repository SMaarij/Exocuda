"""
scripts/calibrate_small_confidence.py -- which signal tells a wrong small-model
answer from a right one?

Pipeline sends easy questions to the small model and escalates answers that
look weak to the big model. This asks a set of easy questions with checkable
answers (regexes) and records, per answer, the text heuristic and token-
probability features from the llama.cpp runner. For each feature it prints
how well it ranks wrong answers below right ones (AUC) and, per threshold,
how many wrong answers it would catch and how many right ones it would
escalate needlessly.

    python scripts/calibrate_small_confidence.py --models qwen2.5:0.5b-instruct,qwen2.5:1.5b-instruct
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS: list[tuple[str, str]] = [
    ("Name a primary colour.", r"\b(red|blue|yellow)\b"),
    ("What is the capital of Canada?", r"ottawa"),
    ("What is the capital of Turkey?", r"ankara"),
    ("What is the capital of Brazil?", r"bras[ií]lia"),
    ("What is the capital of Switzerland?", r"\bbern"),
    ("What is the capital of Australia?", r"canberra"),
    ("What is the capital of New Zealand?", r"wellington"),
    ("What is the capital of Nigeria?", r"abuja"),
    ("What is the capital of Pakistan?", r"islamabad"),
    ("What is the capital of Morocco?", r"rabat"),
    ("What is the capital of Vietnam?", r"hanoi"),
    ("What is the capital of India?", r"new delhi"),
    ("Which planet is closest to the Sun?", r"mercury"),
    ("Which planet is known as the Red Planet?", r"mars"),
    ("What is the largest planet in the solar system?", r"jupiter"),
    ("Who painted the Mona Lisa?", r"leonardo|da vinci"),
    ("What is the largest ocean on Earth?", r"pacific"),
    ("How many legs does a spider have?", r"\b(8|eight)\b"),
    ("What is the boiling point of water in Fahrenheit?", r"\b212\b"),
    ("What is the freezing point of water in Celsius?", r"\b(0|zero)\b"),
    ("What is the square root of 144?", r"\b12\b"),
    ("Who was the first person to walk on the Moon?", r"armstrong"),
    ("In what year did World War II end?", r"1945"),
    ("How many continents are there?", r"\b(7|seven)\b"),
    ("What is the smallest prime number?", r"\b(2|two)\b"),
    ("Which gas do plants absorb from the air for photosynthesis?", r"carbon dioxide|co2|co₂"),
    ("Who developed the theory of relativity?", r"einstein"),
    ("How many sides does a hexagon have?", r"\b(6|six)\b"),
    ("What is the chemical symbol for sodium?", r"\bNa\b"),
    ("What is the chemical symbol for iron?", r"\bFe\b"),
    ("What is the chemical symbol for potassium?", r"\bK\b"),
    ("What is the chemical symbol for silver?", r"\bAg\b"),
    ("Which element has atomic number 1?", r"hydrogen"),
    ("How many bones are in the adult human body?", r"\b206\b"),
    ("What is 15% of 200?", r"\b30\b"),
    ("What is 12 squared?", r"\b144\b"),
    ("What is 7 times 8?", r"\b56\b"),
    ("What is 17 times 3?", r"\b51\b"),
    ("What is 1000 divided by 8?", r"\b125\b"),
    ("What is 2 to the power of 10?", r"\b1024\b|1,024"),
    ("Who wrote the novel 1984?", r"orwell"),
    ("Who wrote Pride and Prejudice?", r"austen"),
    ("Who discovered penicillin?", r"fleming"),
    ("What is the currency of Japan?", r"\byen\b"),
    ("What language is mainly spoken in Brazil?", r"portuguese"),
    ("What is the hardest natural substance?", r"diamond"),
    ("How many minutes are in a day?", r"1440|1,440"),
    ("How many seconds are in an hour?", r"3600|3,600"),
    ("How many hours are in a week?", r"\b168\b"),
    ("How many players does a soccer team have on the field?", r"\b(11|eleven)\b"),
    ("What is the tallest mountain in the world?", r"everest"),
    ("Who invented the telephone?", r"\bbell\b"),
    ("Convert 100 degrees Celsius to Fahrenheit.", r"\b212\b"),
    ("What is the plural of mouse?", r"\bmice\b"),
    ("What is the past tense of the verb 'go'?", r"\bwent\b"),
    ("Name a mammal that can fly.", r"\bbats?\b"),
    ("In which country are the pyramids of Giza?", r"egypt"),
    ("How many strings does a standard guitar have?", r"\b(6|six)\b"),
    ("How many days are in February in a non-leap year?", r"\b28\b"),
    ("What is the longest bone in the human body?", r"femur"),
    ("Which organ pumps blood through the body?", r"heart"),
    ("What is the capital city of Spain?", r"madrid"),
    ("Who wrote Hamlet?", r"shakespeare"),
    ("What is the main ingredient in guacamole?", r"avocado"),
]

# A second, disjoint set: thresholds picked on QUESTIONS are checked here.
HOLDOUT: list[tuple[str, str]] = [
    ("How many weeks are in a year?", r"\b52\b"),
    ("How many months have 31 days?", r"\b(7|seven)\b"),
    ("How many sides does an octagon have?", r"\b(8|eight)\b"),
    ("How many sides does a pentagon have?", r"\b(5|five)\b"),
    ("How many zeros are in one million?", r"\b(6|six)\b"),
    ("How many centimetres are in a metre?", r"\b100\b"),
    ("How many grams are in a kilogram?", r"1000|1,000"),
    ("How many minutes are in an hour?", r"\b60\b"),
    ("How many days are in a week?", r"\b(7|seven)\b"),
    ("How many planets are in the solar system?", r"\b(8|eight)\b"),
    ("How many teeth does an adult human usually have?", r"\b32\b"),
    ("How many hearts does an octopus have?", r"\b(3|three)\b"),
    ("How many legs does an insect have?", r"\b(6|six)\b"),
    ("How many colours are in a rainbow?", r"\b(7|seven)\b"),
    ("How many keys are on a standard piano?", r"\b88\b"),
    ("How many cards are in a standard deck, without jokers?", r"\b52\b"),
    ("How many players are on a basketball team on the court?", r"\b(5|five)\b"),
    ("How many degrees are in a right angle?", r"\b90\b"),
    ("How many degrees are in a full circle?", r"\b360\b"),
    ("How many years are in a century?", r"\b100\b"),
    ("What is 9 times 7?", r"\b63\b"),
    ("What is 144 divided by 12?", r"\b12\b"),
    ("What is 25 plus 38?", r"\b63\b"),
    ("What is 100 minus 37?", r"\b63\b"),
    ("What is 13 times 4?", r"\b52\b"),
    ("What is half of 250?", r"\b125\b"),
    ("What is 3 cubed?", r"\b27\b"),
    ("What is 20% of 50?", r"\b10\b"),
    ("What is the square root of 81?", r"\b(9|nine)\b"),
    ("What is 11 times 11?", r"\b121\b"),
    ("What is the capital of Egypt?", r"cairo"),
    ("What is the capital of Kenya?", r"nairobi"),
    ("What is the capital of Argentina?", r"buenos aires"),
    ("What is the capital of South Korea?", r"seoul"),
    ("What is the capital of Germany?", r"berlin"),
    ("What is the capital of Norway?", r"oslo"),
    ("What is the capital of Peru?", r"\blima\b"),
    ("What is the capital of Thailand?", r"bangkok"),
    ("What is the capital of Portugal?", r"lisbon"),
    ("What is the capital of the United States?", r"washington"),
    ("What is the chemical symbol for gold?", r"\bAu\b"),
    ("What is the chemical symbol for lead?", r"\bPb\b"),
    ("What is the chemical symbol for copper?", r"\bCu\b"),
    ("What is the chemical symbol for oxygen?", r"\bO\b"),
    ("Which planet has the most prominent rings?", r"saturn"),
    ("What is the closest star to Earth?", r"\bsun\b"),
    ("What do bees make?", r"honey"),
    ("What is the largest mammal?", r"blue whale"),
    ("What is the fastest land animal?", r"cheetah"),
    ("Which animal is known as the king of the jungle?", r"lion"),
    ("Who painted The Starry Night?", r"van gogh"),
    ("Who wrote Romeo and Juliet?", r"shakespeare"),
    ("Who wrote The Odyssey?", r"homer"),
    ("Who was the first President of the United States?", r"washington"),
    ("What is the opposite of 'hot'?", r"\bcold\b"),
    ("What is the plural of child?", r"children"),
    ("What is the past tense of 'eat'?", r"\bate\b"),
    ("Name a fruit that is yellow.", r"banana|lemon|pineapple|mango"),
    ("Name a bird that cannot fly.", r"penguin|ostrich|emu|kiwi|cassowary|dodo"),
    ("What colour do you get by mixing blue and yellow?", r"green"),
    ("What is the largest continent?", r"asia"),
    ("What is the longest river in Africa?", r"\bnile\b"),
    ("What is the currency of the United Kingdom?", r"pound"),
    ("In which country is the Eiffel Tower?", r"france"),
]

FEATURES = ("heuristic", "mean_p", "min_p", "low3_p", "first_p")


def features(text: str, logprobs: list[float] | None) -> dict:
    from cascade_local.engine.intelligence.router.confidence import estimate_local_confidence

    f = {"heuristic": estimate_local_confidence(text)}
    if logprobs:
        lp = sorted(logprobs)
        f["mean_p"] = math.exp(sum(logprobs) / len(logprobs))
        f["min_p"] = math.exp(lp[0])
        f["low3_p"] = math.exp(sum(lp[:3]) / len(lp[:3]))
        f["first_p"] = math.exp(logprobs[0])
    return f


def auc(right: list[float], wrong: list[float]) -> float | None:
    """P(a random right answer scores above a random wrong one)."""
    if not right or not wrong:
        return None
    wins = sum((r > w) + 0.5 * (r == w) for r in right for w in wrong)
    return wins / (len(right) * len(wrong))


def run_model(model: str, max_tokens: int, questions=QUESTIONS) -> list[dict]:
    from benchmark_system import _local_backend
    from cascade_local.cli.main import DEFAULT_SYSTEM_PROMPT

    backend = _local_backend(model, False, max_tokens)
    rows = []
    for question, pattern in questions:
        result = backend.generate([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                                   {"role": "user", "content": question}])
        text = result["text"]
        row = {"question": question, "answer": text, "correct": bool(re.search(pattern, text, re.I)),
               **features(text, backend._runner.last_token_logprobs)}
        rows.append(row)
        print(f"  {'ok ' if row['correct'] else 'BAD'} mean_p={row.get('mean_p', 0):.3f} min_p={row.get('min_p', 0):.3f}"
              f"  {question[:40]:40s} -> {text[:60]!r}", flush=True)
    return rows


def report(model: str, rows: list[dict]) -> dict:
    right = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    out = {"correct": len(right), "questions": len(rows), "auc": {}, "thresholds": {}}
    print(f"\n{model}: {len(right)}/{len(rows)} correct")
    for f in FEATURES:
        a = auc([r[f] for r in right if f in r], [r[f] for r in wrong if f in r])
        out["auc"][f] = a
        print(f"  AUC {f:9s} {a if a is None else round(a, 3)}")
    for f in ("mean_p", "low3_p", "min_p"):
        for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            caught = sum(r.get(f, 1) < t for r in wrong)
            needless = sum(r.get(f, 1) < t for r in right)
            out["thresholds"][f"{f}<{t}"] = {"wrong_caught": caught, "right_escalated": needless}
            print(f"  escalate if {f} < {t:4}: catches {caught}/{len(wrong)} wrong, escalates {needless}/{len(right)} right")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="qwen2.5:0.5b-instruct,qwen2.5:1.5b-instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--set", choices=("main", "holdout"), default="main")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmark/small_confidence_calibration.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    results = {}
    for model in args.models.split(","):
        print(f"== {model}", flush=True)
        rows = run_model(model, args.max_tokens, HOLDOUT if args.set == "holdout" else QUESTIONS)
        results[model] = {"summary": report(model, rows), "rows": rows}
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
