
"""
Tests for the real engine/execution/backend.py -- Tasks 11, 12, 14.

These complement test_backend.py (which tests the Phase 1 stub contract).
All model/runner calls go through FakeReferenceModelRunner (NFR-09).

Phase 3 Task 12 assertions are tagged #NFR-01 and #NFR-02.
Phase 3 Task 14 (coverage) is satisfied by running the full suite with
    pytest engine/execution/ --cov=engine/execution --cov-fail-under=70
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.execution.acceleration_engine import (
    AccelerationLayer,
    DraftPair,
)
from engine.execution.backend import (
    LocalExecutionBackend,
    LocalExecutionBackendError,
)
from engine.execution.hardware_profiler import HardwareProfiler
from engine.execution.model_ladder import ModelLadder, QuantRung
from engine.execution.model_runner import GenerationSample, ReferenceModelRunner
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION, ExecutionBackend, Message


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeRunner:
    """Deterministic runner double -- no live GPU, no Ollama (NFR-09)."""

    def __init__(
        self,
        tok_s: float = 20.0,
        fail_quants: frozenset[str] = frozenset(),
    ) -> None:
        self._tok_s       = tok_s
        self._fail_quants = fail_quants
        self._loaded: str | None = None
        self.load_calls: list[tuple] = []
        self.unload_calls: int = 0

    def load(self, model_id: str, quant_level: str) -> None:
        self.load_calls.append((model_id, quant_level))
        if quant_level in self._fail_quants:
            raise RuntimeError(f"simulated OOM on {quant_level}")
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        return GenerationSample(
            tokens_generated=max_tokens,
            elapsed_s=max_tokens / self._tok_s,
            peak_memory_gb=2.0,
        )

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = None


def _write_profile(tmp_path: Path, tok_s: float = 20.0, vram_gb: float = 0.0) -> Path:
    """Write a synthetic HardwareProfile to disk and return its path."""
    profile = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux", "cpu_cores": 8, "ram_gb": 16.0,
        "gpu_vendor": "nvidia" if vram_gb > 0 else None,
        "vram_gb": vram_gb,
        "disk_throughput_mb_s": 500.0, "disk_free_gb": 100.0,
        "measured_tok_s": {"q4_K_M": tok_s, "q5_K_M": tok_s * 0.8},
        "peak_memory_gb": {"q4_K_M": 2.0, "q5_K_M": 3.0},
        "local_capable": tok_s >= 3.0,
    }
    path = tmp_path / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile))
    return path


def _backend(
    tmp_path: Path,
    tok_s: float = 20.0,
    vram_gb: float = 0.0,
    fail_quants: frozenset[str] = frozenset(),
    preferred_layer: AccelerationLayer = AccelerationLayer.LOOKAHEAD,
    model_has_draft_heads: bool = False,
    validated_pairs: list | None = None,
) -> LocalExecutionBackend:
    runner   = FakeRunner(tok_s=tok_s, fail_quants=fail_quants)
    profile_path = _write_profile(tmp_path, tok_s=tok_s, vram_gb=vram_gb)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    return LocalExecutionBackend(
        runner=runner,
        profiler=profiler,
        preferred_layer=preferred_layer,
        model_has_draft_heads=model_has_draft_heads,
        validated_pairs=validated_pairs or [],
    )


SAMPLE_MESSAGES: list[Message] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "What is 2+2?"},
]


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_real_backend_conforms_to_execution_backend_protocol(tmp_path) -> None:
    backend = _backend(tmp_path)
    assert isinstance(backend, ExecutionBackend)


# --------------------------------------------------------------------------
# Basic generate / generate_stream
# --------------------------------------------------------------------------


def test_generate_returns_well_typed_backend_result(tmp_path) -> None:
    backend = _backend(tmp_path)
    result  = backend.generate(SAMPLE_MESSAGES)

    assert isinstance(result["text"],               str)
    assert isinstance(result["confidence"],         float)
    assert isinstance(result["speculative_active"], bool)
    assert isinstance(result["tok_s"],              float)
    assert isinstance(result["latency_ms"],         float)
    assert isinstance(result["est_cost_usd"],       float)
    assert result["tok_s"]     >= 0.0
    assert result["latency_ms"] >= 0.0
    assert result["est_cost_usd"] == 0.0   # local is always free


def test_generate_stream_yields_tokens(tmp_path) -> None:
    backend = _backend(tmp_path)
    tokens  = list(backend.generate_stream(SAMPLE_MESSAGES))
    assert len(tokens) >= 1


# --------------------------------------------------------------------------
# Rung selection integration
# --------------------------------------------------------------------------


def test_backend_selects_highest_viable_rung(tmp_path) -> None:
    """Backend uses ModelLadder to pick rung; fast machine gets high rung."""
    runner = FakeRunner(tok_s=20.0)
    profile_path = _write_profile(tmp_path, tok_s=20.0, vram_gb=8.0)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    ladder   = ModelLadder(rungs=(
        QuantRung("q4_K_M", 3.0,  0.0),
        QuantRung("q5_K_M", 8.0,  4.0),
        QuantRung("q8_0",  15.0,  8.0),
    ))
    backend  = LocalExecutionBackend(runner=runner, profiler=profiler, ladder=ladder)
    backend.generate(SAMPLE_MESSAGES)
    # The last load call should be for the highest rung that cleared
    last_quant = runner.load_calls[-1][1]
    assert last_quant == "q8_0"


# --------------------------------------------------------------------------
# FR-02.3: OOM step-down
# --------------------------------------------------------------------------


def test_backend_steps_down_automatically_on_oom(tmp_path) -> None:
    """OOM on q4_K_M triggers step-down and retry at floor rung."""
    runner   = FakeRunner(tok_s=5.0, fail_quants=frozenset({"q4_K_M"}))
    profile_path = _write_profile(tmp_path, tok_s=5.0)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    # Ladder with two rungs so there IS a rung to step down to
    ladder   = ModelLadder(rungs=(
        QuantRung("floor", 3.0, 0.0),
        QuantRung("q4_K_M", 4.0, 0.0),
    ))
    backend  = LocalExecutionBackend(runner=runner, profiler=profiler, ladder=ladder)
    # Should not raise -- steps down to "floor" automatically
    result   = backend.generate(SAMPLE_MESSAGES)
    assert isinstance(result["text"], str)
    # floor rung was loaded after q4_K_M OOMed
    loaded_quants = [c[1] for c in runner.load_calls]
    assert "floor" in loaded_quants


def test_backend_raises_when_all_rungs_exhausted(tmp_path) -> None:
    """LocalExecutionBackendError when every rung OOMs."""
    runner   = FakeRunner(tok_s=20.0, fail_quants=frozenset({"q4_K_M", "q5_K_M"}))
    profile_path = _write_profile(tmp_path, tok_s=20.0)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    ladder   = ModelLadder(rungs=(
        QuantRung("q4_K_M", 3.0, 0.0),
        QuantRung("q5_K_M", 5.0, 0.0),
    ))
    backend  = LocalExecutionBackend(runner=runner, profiler=profiler, ladder=ladder)
    with pytest.raises(LocalExecutionBackendError):
        backend.generate(SAMPLE_MESSAGES)


def test_backend_raises_when_no_rung_selectable(tmp_path) -> None:
    """Profile below all floors -> select_rung returns None -> error."""
    runner   = FakeRunner(tok_s=1.0)
    profile_path = _write_profile(tmp_path, tok_s=1.0)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    backend  = LocalExecutionBackend(runner=runner, profiler=profiler)
    with pytest.raises(LocalExecutionBackendError):
        backend.generate(SAMPLE_MESSAGES)


# --------------------------------------------------------------------------
# Acceleration layer integration
# --------------------------------------------------------------------------


def test_speculative_active_false_with_layer1(tmp_path) -> None:
    backend = _backend(tmp_path, preferred_layer=AccelerationLayer.LOOKAHEAD)
    result  = backend.generate(SAMPLE_MESSAGES)
    assert result["speculative_active"] is False


def test_speculative_active_true_with_layer2(tmp_path) -> None:
    backend = _backend(
        tmp_path,
        preferred_layer=AccelerationLayer.DRAFT_HEADS,
        model_has_draft_heads=True,
    )
    result = backend.generate(SAMPLE_MESSAGES)
    assert result["speculative_active"] is True


def test_speculative_active_true_with_layer3(tmp_path) -> None:
    backend = _backend(
        tmp_path,
        preferred_layer=AccelerationLayer.DRAFT_MODEL,
        validated_pairs=[DraftPair("cascade-reference-3b", "draft", True)],
    )
    result = backend.generate(SAMPLE_MESSAGES)
    assert result["speculative_active"] is True


# --------------------------------------------------------------------------
# Phase 3 Task 12 -- NFR-01: Layer 2/3 >= 1.5x tok/s over Layer 1 baseline
# --------------------------------------------------------------------------


def _tok_s_for_layer(tmp_path: Path, layer: AccelerationLayer) -> float:
    heads = layer == AccelerationLayer.DRAFT_HEADS
    pair  = (
        [DraftPair("cascade-reference-3b", "draft", True)]
        if layer == AccelerationLayer.DRAFT_MODEL
        else []
    )
    backend = _backend(
        tmp_path,
        tok_s=20.0,
        preferred_layer=layer,
        model_has_draft_heads=heads,
        validated_pairs=pair,
    )
    return backend.generate(SAMPLE_MESSAGES)["tok_s"]


def test_nfr01_layer2_achieves_1_5x_over_layer1(tmp_path) -> None:
    """NFR-01: Layer 2 (draft heads) must be >= 1.5x Layer 1 tok/s."""  # NFR-01
    layer1 = _tok_s_for_layer(tmp_path / "l1", AccelerationLayer.LOOKAHEAD)
    layer2 = _tok_s_for_layer(tmp_path / "l2", AccelerationLayer.DRAFT_HEADS)
    assert layer2 >= layer1 * 1.5, (
        f"NFR-01 FAIL: Layer 2 tok/s ({layer2:.1f}) < 1.5x Layer 1 ({layer1:.1f})"
    )


def test_nfr01_layer3_achieves_1_5x_over_layer1(tmp_path) -> None:
    """NFR-01: Layer 3 (draft model) must be >= 1.5x Layer 1 tok/s."""  # NFR-01
    layer1 = _tok_s_for_layer(tmp_path / "l1", AccelerationLayer.LOOKAHEAD)
    layer3 = _tok_s_for_layer(tmp_path / "l3", AccelerationLayer.DRAFT_MODEL)
    assert layer3 >= layer1 * 1.5, (
        f"NFR-01 FAIL: Layer 3 tok/s ({layer3:.1f}) < 1.5x Layer 1 ({layer1:.1f})"
    )


# --------------------------------------------------------------------------
# Phase 3 Task 12 -- NFR-02: cache-hit path unaffected by execution layer
# --------------------------------------------------------------------------


def test_nfr02_est_cost_usd_always_zero_for_local(tmp_path) -> None:
    """NFR-02: local execution never incurs cost regardless of layer."""  # NFR-02
    for layer in (
        AccelerationLayer.LOOKAHEAD,
        AccelerationLayer.DRAFT_HEADS,
    ):
        heads = layer == AccelerationLayer.DRAFT_HEADS
        result = _backend(tmp_path, preferred_layer=layer, model_has_draft_heads=heads).generate(SAMPLE_MESSAGES)
        assert result["est_cost_usd"] == 0.0


# --------------------------------------------------------------------------
# Real-output-text fix (found via the Section 12 benchmark run,
# docs/BENCHMARK_REPORT.md's addendum) -- confirms output_text now
# actually flows from the runner through to BackendResult["text"],
# and that fake runners which never set it keep working via the
# fallback (backend.py's `or` -- see its own comment).
# --------------------------------------------------------------------------


class _FakeRunnerWithRealText(FakeRunner):
    """Same double as above, but simulates a runner that DOES report real output text (i.e. the real OllamaReferenceModelRunner's behavior)."""

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        return GenerationSample(
            tokens_generated=max_tokens,
            elapsed_s=max_tokens / self._tok_s,
            peak_memory_gb=2.0,
            output_text="this is the real generated answer",
        )


def test_generate_uses_real_output_text_when_runner_provides_it(tmp_path: Path) -> None:
    runner = _FakeRunnerWithRealText(tok_s=20.0)
    profile_path = _write_profile(tmp_path, tok_s=20.0)
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    backend = LocalExecutionBackend(runner=runner, profiler=profiler)

    result = backend.generate(SAMPLE_MESSAGES)

    assert result["text"] == "this is the real generated answer"


def test_generate_falls_back_to_placeholder_when_runner_has_no_output_text(tmp_path: Path) -> None:
    """FakeRunner (the default double used everywhere else in this file) never sets output_text -- confirms the fallback keeps the field a non-empty string, not silently empty."""
    result = _backend(tmp_path).generate(SAMPLE_MESSAGES)
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
