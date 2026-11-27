"""
engine/intelligence/embeddings.py

Track B -- shared by ContextCompressor (FR-04), SemanticRouter (FR-05),
and SemanticCache (FR-06). One embedding backend, one place that knows
how to compute cosine similarity, so all three components treat
"how similar are these two texts" identically.

Tech Stack 5.1: fastembed (ONNX, bge-small) -- CPU-only, no torch
dependency, matching the low-end-hardware target user.

Follows the same injectable-dependency pattern Track A used for
ReferenceModelRunner (engine/execution/model_runner.py): an Embedder
Protocol, a real fastembed-backed implementation for production, and a
deterministic fake for tests (see engine/intelligence/tests/fakes.py)
so no test needs a model download (NFR-09's "no live GPU" spirit,
extended here to "no network needed to run the suite").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """
    Production embedder. Lazy-loads fastembed's ONNX model on first real
    use (not at construction time), so importing this module -- or even
    constructing this class -- never triggers a model download; only
    calling .embed()/.embed_batch() does.
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL_NAME
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from fastembed import TextEmbedding  # deferred: heavy import

            # fastembed's default cache is the system temp folder, which Windows' storage
            # cleanup empties: the tokenizer files vanished while the ONNX file stayed, and
            # every load then failed. Keep it with Cascade's other state unless set.
            cache_dir = os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cascade" / "fastembed")
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=cache_dir)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        return [vec.tolist() for vec in model.embed(list(texts))]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure function, no I/O -- shared by all three components' similarity checks."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
