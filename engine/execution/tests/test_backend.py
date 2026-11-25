
"""
Tests for engine/execution/backend.py -- Phase 1 stub contract (Task 4).

These tests prove what the J-1 join point requires: Track B can construct
LocalExecutionBackend and call it with no knowledge of Track A internals
and get a well-typed, protocol-conformant response every time.

Now that backend.py is the real Phase 2 implementation (Task 11), these
tests inject a FakeRunner + pre-written HardwareProfile so they remain
independent of a live GPU or Ollama instance (NFR-09).
The public API contract is IDENTICAL to the stub -- Track B's calling code
does not change. That is the definition of the J-2 seam holding.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.execution.backend import LocalExecutionBackend
from engine.execution.hardware_profiler import HardwareProfiler
from engine.execution.model_runner import GenerationSample, ReferenceModelRunner
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION, BackendResult, ExecutionBackend, Message


# --------------------------------------------------------------------------
# Fake runner (same pattern as test_hardware_profiler.py)
# --------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self, tok_s: float = 20.0) -> None:
        self._tok_s = tok_s
        self._loaded: str | None = None
        self.prompts: list = []

    def load(self, model_id: str, quant_level: str) -> None:
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        self.prompts.append(prompt)
        return GenerationSample(
            tokens_generated=max_tokens,
            elapsed_s=max_tokens / self._tok_s,
            peak_memory_gb=2.0,
        )

    def unload(self) -> None:
        self._loaded = None


def _make_backend(tmp_path: Path, simulated_latency_ms: float = 0.0) -> LocalExecutionBackend:
    """Construct a backend with a fast fake runner and a pre-written profile."""
    runner = _FakeRunner(tok_s=20.0)
    profile = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux", "cpu_cores": 8, "ram_gb": 16.0,
        "gpu_vendor": None, "vram_gb": 0.0,
        "disk_throughput_mb_s": 500.0, "disk_free_gb": 100.0,
        "measured_tok_s": {"q4_K_M": 20.0, "q5_K_M": 16.0},
        "peak_memory_gb": {"q4_K_M": 2.0, "q5_K_M": 3.0},
        "local_capable": True,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    profiler = HardwareProfiler(runner=runner, profile_path=path)
    return LocalExecutionBackend(
        runner=runner,
        profiler=profiler,
        simulated_latency_ms=simulated_latency_ms,
    )


# --------------------------------------------------------------------------
# Phase 1 contract tests (unchanged API surface -- J-2 seam holds)
# --------------------------------------------------------------------------


def test_local_execution_backend_conforms_to_protocol(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    assert isinstance(backend, ExecutionBackend)


def test_generate_returns_well_typed_backend_result(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    messages: list[Message] = [{"role": "user", "content": "hello"}]

    result: BackendResult = backend.generate(messages)

    assert set(result.keys()) == {
        "text", "confidence", "speculative_active",
        "tok_s", "latency_ms", "est_cost_usd",
    }
    assert isinstance(result["text"],               str)
    assert isinstance(result["confidence"],         float)
    assert isinstance(result["speculative_active"], bool)
    assert isinstance(result["tok_s"],              float)
    assert isinstance(result["latency_ms"],         float)
    assert isinstance(result["est_cost_usd"],       float)


def test_generate_local_execution_is_always_free(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    result  = backend.generate([{"role": "user", "content": "anything"}])
    assert result["est_cost_usd"] == 0.0


def test_generate_ignores_message_content_but_accepts_any_valid_messages(tmp_path) -> None:
    backend  = _make_backend(tmp_path)
    messages: list[Message] = [
        {"role": "system",    "content": "you are a helpful assistant"},
        {"role": "user",      "content": "what is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user",      "content": "and 3+3?"},
    ]
    result = backend.generate(messages)
    assert isinstance(result["text"], str)


def test_generate_stream_yields_a_nonempty_sequence_of_strings(tmp_path) -> None:
    backend = _make_backend(tmp_path)
    chunks  = list(backend.generate_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_generate_stream_reassembles_to_nonempty_text(tmp_path) -> None:
    backend   = _make_backend(tmp_path)
    full_text = "".join(backend.generate_stream([{"role": "user", "content": "hi"}]))
    assert len(full_text.strip()) > 0


def test_simulated_latency_is_reflected_in_latency_ms(tmp_path) -> None:
    backend = _make_backend(tmp_path, simulated_latency_ms=20.0)
    result  = backend.generate([{"role": "user", "content": "hi"}])
    assert result["latency_ms"] >= 15.0   # scheduler slack allowed


def test_zero_latency_backend_is_fast(tmp_path) -> None:
    backend = _make_backend(tmp_path, simulated_latency_ms=0.0)
    result  = backend.generate([{"role": "user", "content": "hi"}])
    assert result["latency_ms"] < 500.0


def test_runner_gets_a_flat_prompt_unless_it_formats_chat_turns_itself(tmp_path) -> None:
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    backend = _make_backend(tmp_path)
    runner = backend._runner
    backend.generate(messages)
    assert runner.prompts[-1] == "SYSTEM: be brief\nUSER: hi"
    runner.accepts_messages = True
    backend.generate(messages)
    assert runner.prompts[-1] == messages
