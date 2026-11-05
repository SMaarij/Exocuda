"""
engine/intelligence/router/retrain.py

Track B, Task 16 (FR-05.5).

Scheduled or manual retraining from accumulated shadow-label logs, with
no code changes required on the calling side -- callers (a cron job,
or a `cascade retrain` CLI command Track C could add) just call
retrain(); the classifier's own interface never changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from engine.intelligence.embeddings import Embedder
from engine.intelligence.router.classifier import SemanticRouterClassifier
from engine.intelligence.router.coldstart_data import RoutingExample, load_coldstart_dataset
from engine.intelligence.router.shadow_labeling import DEFAULT_LOG_PATH


def load_shadow_examples(log_path: Path = DEFAULT_LOG_PATH) -> list[RoutingExample]:
    """
    Turns accumulated shadow-label log lines into RoutingExamples via
    weak supervision: a served answer that *agreed* with the shadow
    self_hosted answer is evidence "local" was the right call for
    that query; disagreement is evidence it should have gone to
    self_hosted instead.
    """
    if not log_path.exists():
        return []

    examples: list[RoutingExample] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["served_tier"] if row.get("agree") else "self_hosted"
            if label not in ("local", "cloud_byok", "self_hosted"):
                continue
            examples.append(RoutingExample(query=row["query"], label=label))
    return examples


def retrain(
    embedder: Embedder,
    log_path: Path = DEFAULT_LOG_PATH,
    extra_examples: Optional[list[RoutingExample]] = None,
) -> SemanticRouterClassifier:
    """Always includes the cold-start set, plus whatever real usage data has accumulated."""
    examples = list(load_coldstart_dataset())
    examples.extend(load_shadow_examples(log_path))
    if extra_examples:
        examples.extend(extra_examples)

    classifier = SemanticRouterClassifier(embedder)
    classifier.fit(examples)
    return classifier
