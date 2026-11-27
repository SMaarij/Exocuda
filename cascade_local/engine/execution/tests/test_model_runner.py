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

from cascade_local.engine.execution.model_runner import OllamaReferenceModelRunner


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


# ---- HTTP-API runner ------------------------------------------------------------

import pytest

from cascade_local.engine.execution.model_runner import OllamaError, parse_generate_response


def test_parse_generate_response_uses_ollamas_own_token_count_and_decode_time() -> None:
    sample = parse_generate_response(
        {"response": "  Paris.  ", "eval_count": 40, "eval_duration": 2_000_000_000}, wall_s=9.0, loaded_bytes=2 * 1024**3
    )
    assert sample.output_text == "Paris."
    assert sample.tokens_generated == 40
    assert sample.tok_s == pytest.approx(20.0)  # 40 tokens / 2s decode, not / 9s wall
    assert sample.peak_memory_gb == pytest.approx(2.0)
    assert sample.acceleration == "none"
    assert sample.acceptance_rate is None


def test_parse_generate_response_falls_back_to_wall_time() -> None:
    assert parse_generate_response({"response": "x", "eval_count": 10}, wall_s=5.0, loaded_bytes=0).elapsed_s == 5.0


class _RecordingRunner(OllamaReferenceModelRunner):
    def __init__(self, local: list[str]) -> None:
        super().__init__()
        self._local = local
        self.calls: list[tuple[str, dict | None]] = []

    def _request(self, path, payload=None, timeout=10.0):
        self.calls.append((path, payload))
        if path == "/api/tags":
            return {"models": [{"name": n} for n in self._local]}
        if path == "/api/generate" and payload.get("prompt"):
            return {"response": "hi", "eval_count": 3, "eval_duration": 1_000_000_000}
        return {}


def test_load_does_not_pull_when_model_is_already_local() -> None:
    runner = _RecordingRunner(local=["llama3.2:3b-instruct-q4_K_M"])
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert "/api/pull" not in [p for p, _ in runner.calls]


def test_load_pulls_only_a_missing_model() -> None:
    runner = _RecordingRunner(local=[])
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert ("/api/pull", {"model": "llama3.2:3b-instruct-q4_K_M", "stream": False}) in runner.calls


def test_generate_requires_load_and_sends_max_tokens() -> None:
    runner = _RecordingRunner(local=["m:q"])
    with pytest.raises(OllamaError):
        runner.generate("p", 5)
    runner.load("m", "q")
    with patch.object(OllamaReferenceModelRunner, "_query_loaded_model_bytes", return_value=0):
        sample = runner.generate("p", 7)
    assert sample.tok_s == pytest.approx(3.0)
    assert runner.calls[-1][1]["options"] == {"num_predict": 7}


def test_unload_releases_model_via_keep_alive_zero() -> None:
    runner = _RecordingRunner(local=["m:q"])
    runner.load("m", "q")
    runner.unload()
    assert runner.calls[-1] == ("/api/generate", {"model": "m:q", "keep_alive": 0})


def test_unreachable_server_raises_helpful_error() -> None:
    runner = OllamaReferenceModelRunner(api_base_url="http://127.0.0.1:9")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(OllamaError, match="Can't reach Ollama"):
            runner.load("m", "q")
        assert runner.is_available() is False


def test_http_error_body_is_surfaced() -> None:
    runner = OllamaReferenceModelRunner()
    err = urllib.error.HTTPError("u", 500, "x", {}, BytesIO(b'{"error":"model requires more system memory"}'))
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(OllamaError, match="more system memory"):
            runner.local_models()


def test_model_file_path_reads_from_line() -> None:
    runner = OllamaReferenceModelRunner()
    with patch.object(OllamaReferenceModelRunner, "_request", return_value={"modelfile": "# x\nFROM /models/blobs/sha256-ab\nTEMPLATE y"}):
        assert runner.model_file_path("m:q") == "/models/blobs/sha256-ab"
