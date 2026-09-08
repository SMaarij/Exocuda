
"""
Tests for engine/execution/acceleration_engine.py -- Tasks 7-10.

Per NFR-09: all tests run with no live GPU, no live Ollama.
All runner calls go through FakeRunner (deterministic).
"""

from __future__ import annotations

import pytest

from engine.execution.acceleration_engine import (
    DEFAULT_ACCEPTANCE_THRESHOLD,
    LAYER1_SPEEDUP_FACTOR,
    LAYER2_SPEEDUP_FACTOR,
    LAYER3_SPEEDUP_FACTOR,
    AcceleratedGenerationSample,
    AccelerationEngine,
    AccelerationLayer,
    AcceptanceRateMonitor,
    DraftPair,
    load_validated_pairs,
)
from engine.execution.model_runner import GenerationSample


# --------------------------------------------------------------------------
# Fake runner
# --------------------------------------------------------------------------


class FakeRunner:
    """Deterministic ReferenceModelRunner double."""
    def __init__(self, tok_s: float = 20.0, fail_load: bool = False) -> None:
        self._tok_s     = tok_s
        self._fail_load = fail_load
        self._loaded: str | None = None
        self.load_calls: list[tuple] = []
        self.unload_calls: int = 0

    def load(self, model_id: str, quant_level: str) -> None:
        self.load_calls.append((model_id, quant_level))
        if self._fail_load:
            raise RuntimeError("simulated OOM")
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None, "generate() called before load()"
        return GenerationSample(
            tokens_generated=max_tokens,
            elapsed_s=max_tokens / self._tok_s,
            peak_memory_gb=2.0,
        )

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = None


# --------------------------------------------------------------------------
# AcceptanceRateMonitor (FR-03.3)
# --------------------------------------------------------------------------


class TestAcceptanceRateMonitor:

    def test_should_not_step_down_before_window_full(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=5)
        for _ in range(4):          # one short of the window
            monitor.record(0.0)     # all zeros -- would trigger if window full
        assert monitor.should_step_down() is False

    def test_should_step_down_when_average_below_threshold(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=5)
        for _ in range(5):
            monitor.record(0.30)    # avg = 0.30 < 0.60
        assert monitor.should_step_down() is True

    def test_should_not_step_down_when_average_at_or_above_threshold(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=5)
        for _ in range(5):
            monitor.record(0.80)
        assert monitor.should_step_down() is False

    def test_reset_clears_window(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=3)
        for _ in range(3):
            monitor.record(0.0)
        assert monitor.should_step_down() is True
        monitor.reset()
        assert monitor.should_step_down() is False

    def test_current_average_is_none_when_empty(self) -> None:
        assert AcceptanceRateMonitor().current_average is None

    def test_current_average_reflects_recorded_samples(self) -> None:
        monitor = AcceptanceRateMonitor(window=4)
        for v in [0.5, 0.7, 0.9]:
            monitor.record(v)
        assert monitor.current_average == pytest.approx((0.5 + 0.7 + 0.9) / 3)


# --------------------------------------------------------------------------
# Layer resolution (FR-03.2)
# --------------------------------------------------------------------------


class TestLayerResolution:

    def test_layer1_is_default_when_no_heads_no_pair(self) -> None:
        """FR-03.1: Layer 1 always available, zero draft model required."""
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.LOOKAHEAD,
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD

    def test_layer2_activates_when_model_has_heads(self) -> None:
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_HEADS,
            model_has_draft_heads=True,
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.DRAFT_HEADS

    def test_layer2_falls_back_to_layer1_when_no_heads(self) -> None:
        """FR-03.2: silent fallback, never a crash or quality regression."""
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_HEADS,
            model_has_draft_heads=False,  # no heads available
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD

    def test_layer3_activates_when_validated_pair_exists(self) -> None:
        pair   = DraftPair("target-model", "draft-model", validated=True)
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            validated_pairs=[pair],
        )
        engine.load("target-model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.DRAFT_MODEL

    def test_layer3_falls_back_to_layer2_when_pair_missing_but_heads_exist(self) -> None:
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            model_has_draft_heads=True,
            validated_pairs=[],         # no pair for this model
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.DRAFT_HEADS

    def test_layer3_falls_back_to_layer1_when_no_pair_and_no_heads(self) -> None:
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            model_has_draft_heads=False,
            validated_pairs=[],
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD

    def test_unvalidated_pair_is_ignored(self) -> None:
        pair   = DraftPair("model", "draft", validated=False)
        engine = AccelerationEngine(
            runner=FakeRunner(),
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            validated_pairs=[pair],
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD


# --------------------------------------------------------------------------
# Speedup factors (NFR-01: Layer 2/3 >= 1.5x)
# --------------------------------------------------------------------------


class TestSpeedupFactors:

    def _tok_s_for_layer(self, layer: AccelerationLayer) -> float:
        runner = FakeRunner(tok_s=20.0)
        heads  = layer == AccelerationLayer.DRAFT_HEADS
        pair   = (
            [DraftPair("model", "draft", True)]
            if layer == AccelerationLayer.DRAFT_MODEL
            else []
        )
        engine = AccelerationEngine(
            runner=runner,
            preferred_layer=layer,
            model_has_draft_heads=heads,
            validated_pairs=pair,
        )
        engine.load("model", "q4_K_M")
        sample = engine.generate_accelerated("prompt", 64)
        engine.unload()
        return sample.tok_s

    def test_layer1_provides_modest_speedup_over_baseline(self) -> None:
        tok_s  = self._tok_s_for_layer(AccelerationLayer.LOOKAHEAD)
        # Layer 1 multiplier > 1.0
        assert tok_s > 20.0 * 1.0

    def test_layer2_meets_nfr01_1_5x_target_over_layer1(self) -> None:
        """NFR-01: Layer 2 >= 1.5x vs Layer 1 baseline."""
        layer1_tok_s = self._tok_s_for_layer(AccelerationLayer.LOOKAHEAD)
        layer2_tok_s = self._tok_s_for_layer(AccelerationLayer.DRAFT_HEADS)
        assert layer2_tok_s >= layer1_tok_s * 1.5

    def test_layer3_meets_nfr01_1_5x_target_over_layer1(self) -> None:
        """NFR-01: Layer 3 >= 1.5x vs Layer 1 baseline."""
        layer1_tok_s = self._tok_s_for_layer(AccelerationLayer.LOOKAHEAD)
        layer3_tok_s = self._tok_s_for_layer(AccelerationLayer.DRAFT_MODEL)
        assert layer3_tok_s >= layer1_tok_s * 1.5


# --------------------------------------------------------------------------
# Auto step-down on acceptance-rate collapse (FR-03.3, FR-03.4)
# --------------------------------------------------------------------------


class TestAutoStepDown:

    def _engine_with_full_bad_monitor(self, layer: AccelerationLayer, window: int = 3) -> AccelerationEngine:
        """Return an engine whose monitor is already full of low-rate samples."""
        runner  = FakeRunner(tok_s=20.0)
        heads   = layer == AccelerationLayer.DRAFT_HEADS
        pair    = (
            [DraftPair("model", "draft", True)]
            if layer == AccelerationLayer.DRAFT_MODEL
            else []
        )
        monitor = AcceptanceRateMonitor(threshold=DEFAULT_ACCEPTANCE_THRESHOLD, window=window)
        for _ in range(window):
            monitor.record(0.10)  # well below 0.60
        engine = AccelerationEngine(
            runner=runner,
            preferred_layer=layer,
            monitor=monitor,
            model_has_draft_heads=heads,
            validated_pairs=pair,
        )
        engine.load("model", "q4_K_M")
        return engine

    def test_layer2_steps_down_to_layer1_when_acceptance_rate_collapses(self) -> None:
        engine = self._engine_with_full_bad_monitor(AccelerationLayer.DRAFT_HEADS)
        assert engine.active_layer == AccelerationLayer.DRAFT_HEADS
        engine.generate_accelerated("prompt", 10)  # triggers monitor check
        # After step-down:
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD

    def test_layer3_steps_down_to_layer2_when_heads_available(self) -> None:
        runner  = FakeRunner(tok_s=20.0)
        monitor = AcceptanceRateMonitor(threshold=0.60, window=3)
        for _ in range(3):
            monitor.record(0.10)
        engine  = AccelerationEngine(
            runner=runner,
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            monitor=monitor,
            model_has_draft_heads=True,
            validated_pairs=[DraftPair("model", "draft", True)],
        )
        engine.load("model", "q4_K_M")
        engine.generate_accelerated("prompt", 10)
        assert engine.active_layer == AccelerationLayer.DRAFT_HEADS

    def test_step_down_resets_monitor(self) -> None:
        engine = self._engine_with_full_bad_monitor(AccelerationLayer.DRAFT_HEADS)
        engine.generate_accelerated("prompt", 10)   # triggers step-down
        # Monitor should be fresh (no old bad samples carrying over)
        assert engine.monitor.current_average is None

    def test_layer1_never_steps_down(self) -> None:
        """Layer 1 (lookahead) never produces rejectable speculative tokens."""
        runner  = FakeRunner(tok_s=20.0)
        monitor = AcceptanceRateMonitor(threshold=0.60, window=1)
        monitor.record(0.0)   # force monitor full and bad
        engine  = AccelerationEngine(
            runner=runner,
            preferred_layer=AccelerationLayer.LOOKAHEAD,
            monitor=monitor,
        )
        engine.load("model", "q4_K_M")
        engine.generate_accelerated("prompt", 10)
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD   # unchanged

    def test_output_is_valid_after_step_down(self) -> None:
        """FR-03.4: no quality regression -- result is still a valid sample."""
        engine = self._engine_with_full_bad_monitor(AccelerationLayer.DRAFT_HEADS)
        sample = engine.generate_accelerated("prompt", 64)
        assert isinstance(sample, AcceleratedGenerationSample)
        assert sample.tok_s > 0.0
        assert sample.peak_memory_gb >= 0.0


# --------------------------------------------------------------------------
# speculative_active flag
# --------------------------------------------------------------------------


def test_speculative_active_false_for_layer1() -> None:
    engine = AccelerationEngine(FakeRunner(), AccelerationLayer.LOOKAHEAD)
    engine.load("model", "q4_K_M")
    sample = engine.generate_accelerated("p", 10)
    assert sample.speculative_active is False


def test_speculative_active_true_for_layer2() -> None:
    engine = AccelerationEngine(
        FakeRunner(), AccelerationLayer.DRAFT_HEADS,
        model_has_draft_heads=True,
    )
    engine.load("model", "q4_K_M")
    sample = engine.generate_accelerated("p", 10)
    assert sample.speculative_active is True


def test_speculative_active_true_for_layer3() -> None:
    engine = AccelerationEngine(
        FakeRunner(), AccelerationLayer.DRAFT_MODEL,
        validated_pairs=[DraftPair("model", "draft", True)],
    )
    engine.load("model", "q4_K_M")
    sample = engine.generate_accelerated("p", 10)
    assert sample.speculative_active is True


# --------------------------------------------------------------------------
# load_validated_pairs
# --------------------------------------------------------------------------


def test_load_validated_pairs_returns_empty_when_file_missing(tmp_path) -> None:
    assert load_validated_pairs(tmp_path / "nonexistent.json") == []


def test_load_validated_pairs_parses_valid_file(tmp_path) -> None:
    import json
    data = {"pairs": [
        {"target_model_id": "llama3", "draft_model_id": "llama1", "validated": True},
    ]}
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    pairs = load_validated_pairs(p)
    assert len(pairs) == 1
    assert pairs[0].target_model_id == "llama3"


def test_load_validated_pairs_skips_unvalidated(tmp_path) -> None:
    import json
    data = {"pairs": [
        {"target_model_id": "m1", "draft_model_id": "d1", "validated": True},
        {"target_model_id": "m2", "draft_model_id": "d2", "validated": False},
    ]}
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    pairs = load_validated_pairs(p)
    assert len(pairs) == 1
    assert pairs[0].target_model_id == "m1"
