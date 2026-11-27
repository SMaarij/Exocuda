"""FastEmbedEmbedder keeps fastembed's model cache out of the system temp folder."""
from __future__ import annotations

import sys
import types
from pathlib import Path

from cascade_local.engine.intelligence.embeddings import FastEmbedEmbedder


def _fake_fastembed(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    class TextEmbedding:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding))
    return calls


def test_model_cache_lives_under_dot_cascade(monkeypatch, tmp_path) -> None:
    calls = _fake_fastembed(monkeypatch)
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    FastEmbedEmbedder()._ensure_loaded()
    assert calls[0]["cache_dir"] == str(tmp_path / ".cascade" / "fastembed")


def test_fastembed_cache_path_still_wins(monkeypatch, tmp_path) -> None:
    calls = _fake_fastembed(monkeypatch)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "mine"))
    FastEmbedEmbedder()._ensure_loaded()
    assert calls[0]["cache_dir"] == str(tmp_path / "mine")
