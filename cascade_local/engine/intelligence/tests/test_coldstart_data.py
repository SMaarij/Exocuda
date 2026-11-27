"""engine/intelligence/tests/test_coldstart_data.py -- Track B, Task 6 (FR-05.1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cascade_local.engine.intelligence.router.coldstart_data import (
    BOOTSTRAP_EXAMPLES,
    load_coldstart_dataset,
)


def test_bootstrap_dataset_is_non_empty() -> None:
    assert len(BOOTSTRAP_EXAMPLES) > 0


def test_bootstrap_dataset_covers_all_three_labels() -> None:
    labels = {e.label for e in BOOTSTRAP_EXAMPLES}
    assert labels == {"local", "cloud_byok", "self_hosted"}


def test_load_coldstart_dataset_defaults_to_bootstrap_when_no_path() -> None:
    examples = load_coldstart_dataset()
    assert examples == list(BOOTSTRAP_EXAMPLES)


def test_load_coldstart_dataset_defaults_to_bootstrap_when_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    examples = load_coldstart_dataset(missing)
    assert examples == list(BOOTSTRAP_EXAMPLES)


def test_load_coldstart_dataset_reads_real_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "routing_data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "hello", "label": "local"}) + "\n")
        f.write(json.dumps({"query": "write a compiler", "label": "cloud_byok"}) + "\n")

    examples = load_coldstart_dataset(path)
    assert len(examples) == 2
    assert examples[0].query == "hello"
    assert examples[0].label == "local"


def test_load_coldstart_dataset_reads_real_csv_file(tmp_path: Path) -> None:
    path = tmp_path / "routing_data.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "label"])
        writer.writeheader()
        writer.writerow({"query": "what's the weather now", "label": "self_hosted"})

    examples = load_coldstart_dataset(path)
    assert len(examples) == 1
    assert examples[0].label == "self_hosted"
