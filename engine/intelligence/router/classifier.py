"""
engine/intelligence/router/classifier.py

Track B, Tasks 7-8 (FR-05.1, FR-05.2).

Trains a real classifier (logistic regression, per Tech Stack 5.1) on
BOOTSTRAP_EXAMPLES/real cold-start data instead of shipping a hardcoded
if/elif keyword router. Features: the query's embedding plus a handful
of cheap statistical features (FR-05.1: "query embedding plus a handful
of statistical features").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from engine.intelligence.embeddings import Embedder
from engine.intelligence.router.coldstart_data import (
    RoutingExample,
    RoutingLabel,
    load_coldstart_dataset,
)

_TIME_SENSITIVE_WORDS = ("today", "now", "current", "currently", "latest", "search", "news", "right now")


@dataclass(frozen=True)
class RoutingDecision:
    tier: RoutingLabel
    confidence: float


def _statistical_features(query: str) -> list[float]:
    """A few cheap, query-shape signals alongside the embedding (FR-05.1)."""
    lower = query.lower()
    return [
        float(len(query)),
        float(query.count("```")),  # code-block marker -> leans hard/cloud_byok
        float(sum(lower.count(w) for w in _TIME_SENSITIVE_WORDS)),  # leans cascade_cloud
        float(query.count("?")),
    ]


class SemanticRouterClassifier:
    """
    FR-05.2: classifies a query into local / cloud_byok / cascade_cloud
    with a calibrated confidence score (predict_proba, not a made-up number).
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._model: Optional[LogisticRegression] = None
        self._classes: list[str] = []

    def fit(self, examples: Optional[list[RoutingExample]] = None) -> None:
        examples = examples if examples is not None else load_coldstart_dataset()
        if not examples:
            raise ValueError("Cannot fit SemanticRouterClassifier on zero examples")

        X = self._featurize_many([e.query for e in examples])
        y = [e.label for e in examples]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self._model = model
        self._classes = list(model.classes_)

    def classify(self, query: str) -> RoutingDecision:
        if self._model is None:
            self.fit()
        assert self._model is not None

        X = np.array([self._featurize(query)])
        proba = self._model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        return RoutingDecision(tier=self._classes[idx], confidence=float(proba[idx]))

    def _featurize(self, query: str) -> list[float]:
        return list(self._embedder.embed(query)) + _statistical_features(query)

    def _featurize_many(self, queries: list[str]) -> np.ndarray:
        return np.array([self._featurize(q) for q in queries])

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
