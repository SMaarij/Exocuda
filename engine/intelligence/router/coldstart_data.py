"""
engine/intelligence/router/coldstart_data.py

Track B, Task 6 (FR-05.1).

Loads routing-outcome training data for SemanticRouter's classifier, so
it ships cold-started on real prior data rather than an empty or
handwritten ruleset (FR-05.1) -- this is the direct fix for the
hardcoded-heuristic complaint the V1 router drew.

Label space (RoutingLabel): "local" / "cloud_byok" / "self_hosted".
Note this is the post-cache-miss execution tier, not a difficulty
label -- by the time SemanticRouter runs, SemanticCache has already had
its chance at a hit (see the pipeline architecture: Compressor -> Cache
-> [hit? return] -> Router -> execution). "cache" is therefore not a
routing label the classifier needs to predict.

Label rationale (used to build BOOTSTRAP_EXAMPLES below):
    local          -- trivial/moderate queries a local model handles well.
    cloud_byok     -- hard reasoning/coding queries where answer quality
                      benefits from a larger model, and no live external
                      data or tool call is needed.
    self_hosted  -- queries needing live/current data or a tool call
                      (search, current date, account lookups) -- FR-07.1
                      is the tier with tool-calling enabled; local has no
                      tool execution in this version.

PENDING (flagged exactly like Track A's own documented placeholders --
e.g. DEFAULT_REFERENCE_MODEL_ID in hardware_profiler.py): this
environment has no network access to download RouteLLM's actual public
dataset (github.com/lm-sys/RouteLLM). BOOTSTRAP_EXAMPLES below is a
small, clearly-labeled placeholder large enough to exercise the
classifier's training/inference *pipeline* end to end -- it is NOT a
claim about routing accuracy on real traffic (see NFR-01's benchmark
target in Section 12, which still needs to be run against real data).
Swap in real data by pointing `dataset_path` at a downloaded RouteLLM
export (.jsonl or .csv with "query"/"label" fields) -- no code change
needed beyond that.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

RoutingLabel = Literal["local", "cloud_byok", "self_hosted"]


@dataclass(frozen=True)
class RoutingExample:
    query: str
    label: RoutingLabel


# PENDING: replace with a real RouteLLM export once network access
# exists in this environment. See module docstring.
BOOTSTRAP_EXAMPLES: tuple[RoutingExample, ...] = (
    # -- local: trivial/moderate, a small local model handles these fine
    RoutingExample("hi there", "local"),
    RoutingExample("hello, how are you", "local"),
    RoutingExample("what is 2 plus 2", "local"),
    RoutingExample("what's the capital of france", "local"),
    RoutingExample("thanks, that helps", "local"),
    RoutingExample("can you say that again more simply", "local"),
    RoutingExample("summarize this paragraph in one sentence", "local"),
    RoutingExample("explain how photosynthesis works in simple terms", "local"),
    RoutingExample("convert 10 miles to kilometers", "local"),
    RoutingExample("what does the word ubiquitous mean", "local"),
    RoutingExample("give me a synonym for happy", "local"),
    RoutingExample("what year did world war two end", "local"),
    # -- cloud_byok: hard reasoning/coding, no live data needed
    RoutingExample("write a python function to reverse a linked list", "cloud_byok"),
    RoutingExample("prove that the square root of two is irrational", "cloud_byok"),
    RoutingExample("design a distributed rate limiter and explain the tradeoffs", "cloud_byok"),
    RoutingExample("debug this recursive function that causes a stack overflow", "cloud_byok"),
    RoutingExample("explain the CAP theorem and give a real world example", "cloud_byok"),
    RoutingExample("write a detailed proof of the pythagorean theorem", "cloud_byok"),
    RoutingExample("optimize this sql query for a table with 10 million rows", "cloud_byok"),
    RoutingExample("explain how transformers work in deep learning in depth", "cloud_byok"),
    RoutingExample("refactor this class to follow solid principles", "cloud_byok"),
    RoutingExample("walk through the time complexity of quicksort step by step", "cloud_byok"),
    # -- self_hosted: needs live/current data or a tool call
    RoutingExample("what's today's date", "self_hosted"),
    RoutingExample("search the web for the latest news on the election", "self_hosted"),
    RoutingExample("what's the current stock price of apple", "self_hosted"),
    RoutingExample("book me a flight to tokyo next friday", "self_hosted"),
    RoutingExample("what's the weather like right now in london", "self_hosted"),
    RoutingExample("look up my account balance", "self_hosted"),
    RoutingExample("what time is it in tokyo currently", "self_hosted"),
    RoutingExample("find the latest reviews for this restaurant", "self_hosted"),
)


def load_coldstart_dataset(dataset_path: Optional[Path] = None) -> list[RoutingExample]:
    """
    FR-05.1: returns a real, non-empty training set -- from `dataset_path`
    if given and it exists, otherwise the bootstrap placeholder above.
    """
    if dataset_path is not None and Path(dataset_path).exists():
        return _load_from_path(Path(dataset_path))
    return list(BOOTSTRAP_EXAMPLES)


def _load_from_path(path: Path) -> list[RoutingExample]:
    if path.suffix == ".jsonl":
        examples: list[RoutingExample] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                examples.append(RoutingExample(query=row["query"], label=row["label"]))
        return examples
    if path.suffix == ".csv":
        examples = []
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                examples.append(RoutingExample(query=row["query"], label=row["label"]))
        return examples
    raise ValueError(f"Unsupported cold-start dataset format: {path.suffix}")
