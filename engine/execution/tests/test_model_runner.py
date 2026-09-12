"""
engine/execution/tests/test_model_runner.py

Unlike the rest of OllamaReferenceModelRunner (integration scope only,
per NFR-09 -- needs a live Ollama install), `_query_loaded_model_bytes()`
is pure parsing logic over an HTTP response and IS unit-testable by
mocking `urllib.request.urlopen`, with no live Ollama needed. Added
after a real live run (docs/BENCHMARK_REPORT.md's addendum) found the
original psutil-on-`ollama run`'s-PID approach always measured ~0 --
`ollama run` is a thin client; the model's real memory lives in the
ollama server process, reported via /api/ps.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import patch

from engine.execution.model_runner import OllamaReferenceModelRunner


def _fake_response(payload: dict):
    return BytesIO(json.dumps(payload).encode("utf-8"))


def test_query_loaded_model_bytes_returns_zero_when_no_models_loaded() -> None:
    runner = OllamaReferenceModelRunner()
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _fake_response({"models": []})
        assert runner._query_loaded_model_bytes() == 0


def test_query_loaded_model_bytes_prefers_size_vram() -> None:
    runner = OllamaReferenceModelRunner()
    payload = {"models": [{"size": 1000, "size_vram": 2000}]}
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _fake_response(payload)
        assert runner._query_loaded_model_bytes() == 2000


def test_query_loaded_model_bytes_falls_back_to_size_when_no_vram() -> None:
    runner = OllamaReferenceModelRunner()
    payload = {"models": [{"size": 1500}]}
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _fake_response(payload)
        assert runner._query_loaded_model_bytes() == 1500


def test_query_loaded_model_bytes_takes_max_across_multiple_models() -> None:
    runner = OllamaReferenceModelRunner()
    payload = {"models": [{"size_vram": 1000}, {"size_vram": 5000}, {"size_vram": 2000}]}
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _fake_response(payload)
        assert runner._query_loaded_model_bytes() == 5000


def test_query_loaded_model_bytes_returns_zero_on_connection_error() -> None:
    """The Ollama server not being reachable must never raise into the caller -- generate() should still complete."""
    runner = OllamaReferenceModelRunner()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        assert runner._query_loaded_model_bytes() == 0


def test_query_loaded_model_bytes_returns_zero_on_malformed_json() -> None:
    runner = OllamaReferenceModelRunner()
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = BytesIO(b"not json")
        assert runner._query_loaded_model_bytes() == 0


def test_uses_configured_api_base_url() -> None:
    runner = OllamaReferenceModelRunner(api_base_url="http://custom-host:9999")
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = _fake_response({"models": []})
        runner._query_loaded_model_bytes()
        called_url = mock_urlopen.call_args[0][0]
        assert called_url == "http://custom-host:9999/api/ps"
