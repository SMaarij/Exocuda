"""
engine/intelligence/tests/fakes.py

Shared test doubles for Track B's test suites. FakeEmbedder mirrors
Track A's FakeReferenceModelRunner pattern (engine/execution/model_runner.py):
deterministic, dependency-free, no network/model download -- but similar
input text still produces similar vectors, so tests that assert
"a relevant turn survives, an irrelevant one doesn't" behave meaningfully
rather than on pure chance.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

_DIM = 64


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-dimension vector."""

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        return vec

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
