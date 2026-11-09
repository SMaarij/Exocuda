"""
Tests for engine/execution/acceleration_engine.py (FR-03).

NFR-09: no live GPU or Ollama. FakeRunner is a plain runner (like
Ollama); FakeAcceleratedRunner simulates a runner that can genuinely
switch decoding modes (like LlamaCppRunner) and reports what it did.
"""

from __future__ import annotations

import json

import pytest

from engine.execution.acceleration_engine import (
    ACCEL_DRAFT_HEADS,
    DEFAULT_ACCEPTANCE_THRESHOLD,
    AcceleratedGenerationSample,
    AccelerationEngine,
    AccelerationLayer,
    AcceptanceRateMonitor,
    DraftPair,
    load_validated_pairs,
)
from engine.execution.model_runner import (
    ACCEL_DRAFT_MODEL,
    ACCEL_NONE,
    ACCEL_PROMPT_LOOKUP,
    GenerationSample,
)

ALL_MODES = frozenset({ACCEL_PROMPT_LOOKUP, ACCEL_DRAFT_HEADS, ACCEL_DRAFT_MODEL})


class FakeRunner:
    """Plain runner: no set_acceleration, always reports ACCEL_NONE."""

    def __init__(self, tok_s: float = 20.0) -> None:
        self._tok_s = tok_s
        self._loaded: str | None = None
        self.unload_calls = 0

    def load(self, model_id: str, quant_level: str) -> None:
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None, "generate() called before load()"
        return GenerationSample(tokens_generated=max_tokens, elapsed_s=max_tokens / self._tok_s, peak_memory_gb=2.0)

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = None


class FakeAcceleratedRunner(FakeRunner):
    def __init__(self, tok_s: float = 20.0, supported: frozenset[str] = ALL_MODES, acceptance_rate: float = 0.8) -> None:
        super().__init__(tok_s)
        self._supported = supported
        self._acceptance_rate = acceptance_rate
        self.mode = ACCEL_NONE
        self.mode_requests: list[str] = []

    def set_acceleration(self, mode: str) -> bool:
        self.mode_requests.append(mode)
        if mode != ACCEL_NONE and mode not in self._supported:
            return False
        self.mode = mode
        return True

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        base = super().generate(prompt, max_tokens)
        speculative = self.mode != ACCEL_NONE
        return GenerationSample(
            tokens_generated=base.tokens_generated,
            elapsed_s=base.elapsed_s,
            peak_memory_gb=base.peak_memory_gb,
            acceleration=self.mode,
            acceptance_rate=self._acceptance_rate if speculative else None,
        )


def _engine(layer: AccelerationLayer, runner=None, heads: bool = False, pair: bool = False, monitor=None) -> AccelerationEngine:
    engine = AccelerationEngine(
        runner=runner or FakeAcceleratedRunner(),
        preferred_layer=layer,
        monitor=monitor,
        model_has_draft_heads=heads,
        validated_pairs=[DraftPair("model", "draft", True)] if pair else [],
    )
    engine.load("model", "q4_K_M")
    return engine


# ---- AcceptanceRateMonitor (FR-03.3) -----------------------------------------


class TestAcceptanceRateMonitor:
    def test_should_not_step_down_before_window_full(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=5)
        for _ in range(4):
            monitor.record(0.0)
        assert monitor.should_step_down() is False

    def test_should_step_down_when_average_below_threshold(self) -> None:
        monitor = AcceptanceRateMonitor(threshold=0.60, window=5)
        for _ in range(5):
            monitor.record(0.30)
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
        monitor.reset()
        assert monitor.should_step_down() is False

    def test_current_average(self) -> None:
        assert AcceptanceRateMonitor().current_average is None
        monitor = AcceptanceRateMonitor(window=4)
        for v in [0.5, 0.7, 0.9]:
            monitor.record(v)
        assert monitor.current_average == pytest.approx(0.7)


# ---- Layer resolution (FR-03.2) ------------------------------------------------


class TestLayerResolution:
    def test_plain_runner_never_reports_acceleration(self) -> None:
        """A runner that can't change decoding (e.g. Ollama) is honestly NONE, whatever was preferred."""
        engine = _engine(AccelerationLayer.DRAFT_MODEL, runner=FakeRunner(), heads=True, pair=True)
        assert engine.active_layer == AccelerationLayer.NONE
        assert engine.generate_accelerated("p", 10).speculative_active is False

    def test_layer1_activates_on_capable_runner(self) -> None:
        assert _engine(AccelerationLayer.LOOKAHEAD).active_layer == AccelerationLayer.LOOKAHEAD

    def test_layer2_activates_when_model_has_heads(self) -> None:
        assert _engine(AccelerationLayer.DRAFT_HEADS, heads=True).active_layer == AccelerationLayer.DRAFT_HEADS

    def test_layer2_falls_back_to_layer1_when_no_heads(self) -> None:
        assert _engine(AccelerationLayer.DRAFT_HEADS, heads=False).active_layer == AccelerationLayer.LOOKAHEAD

    def test_layer3_activates_when_validated_pair_exists(self) -> None:
        assert _engine(AccelerationLayer.DRAFT_MODEL, pair=True).active_layer == AccelerationLayer.DRAFT_MODEL

    def test_layer3_falls_back_to_layer2_when_pair_missing_but_heads_exist(self) -> None:
        assert _engine(AccelerationLayer.DRAFT_MODEL, heads=True).active_layer == AccelerationLayer.DRAFT_HEADS

    def test_layer3_falls_back_to_layer1_when_no_pair_and_no_heads(self) -> None:
        assert _engine(AccelerationLayer.DRAFT_MODEL).active_layer == AccelerationLayer.LOOKAHEAD

    def test_unvalidated_pair_is_ignored(self) -> None:
        engine = AccelerationEngine(
            runner=FakeAcceleratedRunner(),
            preferred_layer=AccelerationLayer.DRAFT_MODEL,
            validated_pairs=[DraftPair("model", "draft", validated=False)],
        )
        engine.load("model", "q4_K_M")
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD

    def test_walks_down_when_runner_rejects_eligible_layer(self) -> None:
        """Eligible for Layer 2 (heads), but the runner can't do it -> Layer 1."""
        runner = FakeAcceleratedRunner(supported=frozenset({ACCEL_PROMPT_LOOKUP}))
        engine = _engine(AccelerationLayer.DRAFT_HEADS, runner=runner, heads=True)
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD
        assert runner.mode_requests == [ACCEL_DRAFT_HEADS, ACCEL_PROMPT_LOOKUP]

    def test_falls_to_none_when_runner_supports_nothing(self) -> None:
        runner = FakeAcceleratedRunner(supported=frozenset())
        engine = _engine(AccelerationLayer.LOOKAHEAD, runner=runner)
        assert engine.active_layer == AccelerationLayer.NONE
        assert runner.mode == ACCEL_NONE

    def test_preferring_none_disables_acceleration(self) -> None:
        runner = FakeAcceleratedRunner()
        assert _engine(AccelerationLayer.NONE, runner=runner).active_layer == AccelerationLayer.NONE
        assert runner.mode == ACCEL_NONE


# ---- Measured numbers pass through unchanged -----------------------------------


class TestNoFabricatedSpeedup:
    @pytest.mark.parametrize("layer", list(AccelerationLayer))
    def test_tok_s_is_exactly_the_runners_measurement(self, layer: AccelerationLayer) -> None:
        engine = _engine(layer, runner=FakeAcceleratedRunner(tok_s=20.0), heads=True, pair=True)
        assert engine.generate_accelerated("p", 64).tok_s == pytest.approx(20.0)

    def test_acceptance_rate_is_the_runners_measurement(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_MODEL, runner=FakeAcceleratedRunner(acceptance_rate=0.73), pair=True)
        assert engine.generate_accelerated("p", 10).acceptance_rate == pytest.approx(0.73)

    def test_reported_layer_is_what_the_runner_says_ran(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_MODEL, pair=True)
        sample = engine.generate_accelerated("p", 10)
        assert sample.active_layer == AccelerationLayer.DRAFT_MODEL
        assert sample.speculative_active is True


# ---- Auto step-down (FR-03.3, FR-03.4) -----------------------------------------


def _bad_monitor(window: int = 3) -> AcceptanceRateMonitor:
    monitor = AcceptanceRateMonitor(threshold=DEFAULT_ACCEPTANCE_THRESHOLD, window=window)
    for _ in range(window):
        monitor.record(0.10)
    return monitor


class TestAutoStepDown:
    def test_layer2_steps_down_to_layer1_when_acceptance_collapses(self) -> None:
        runner = FakeAcceleratedRunner(acceptance_rate=0.1)
        engine = _engine(AccelerationLayer.DRAFT_HEADS, runner=runner, heads=True, monitor=_bad_monitor())
        engine.generate_accelerated("p", 10)
        assert engine.active_layer == AccelerationLayer.LOOKAHEAD
        assert runner.mode == ACCEL_PROMPT_LOOKUP

    def test_layer3_steps_down_to_layer2_when_heads_available(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_MODEL, runner=FakeAcceleratedRunner(acceptance_rate=0.1), heads=True, pair=True, monitor=_bad_monitor())
        engine.generate_accelerated("p", 10)
        assert engine.active_layer == AccelerationLayer.DRAFT_HEADS

    def test_layer1_steps_down_to_plain_decoding(self) -> None:
        """Prompt lookup has real rejections, so it can be disabled too."""
        runner = FakeAcceleratedRunner(acceptance_rate=0.1)
        engine = _engine(AccelerationLayer.LOOKAHEAD, runner=runner, monitor=_bad_monitor())
        engine.generate_accelerated("p", 10)
        assert engine.active_layer == AccelerationLayer.NONE
        assert runner.mode == ACCEL_NONE

    def test_step_down_resets_monitor(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_HEADS, runner=FakeAcceleratedRunner(acceptance_rate=0.1), heads=True, monitor=_bad_monitor())
        engine.generate_accelerated("p", 10)
        assert engine.monitor.current_average is None

    def test_good_acceptance_keeps_layer(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_MODEL, runner=FakeAcceleratedRunner(acceptance_rate=0.9), pair=True, monitor=AcceptanceRateMonitor(window=2))
        for _ in range(5):
            engine.generate_accelerated("p", 10)
        assert engine.active_layer == AccelerationLayer.DRAFT_MODEL

    def test_plain_samples_never_feed_the_monitor(self) -> None:
        monitor = AcceptanceRateMonitor(window=1)
        engine = _engine(AccelerationLayer.LOOKAHEAD, runner=FakeRunner(), monitor=monitor)
        engine.generate_accelerated("p", 10)
        assert monitor.current_average is None

    def test_output_is_valid_after_step_down(self) -> None:
        engine = _engine(AccelerationLayer.DRAFT_HEADS, runner=FakeAcceleratedRunner(acceptance_rate=0.1), heads=True, monitor=_bad_monitor())
        sample = engine.generate_accelerated("p", 64)
        assert isinstance(sample, AcceleratedGenerationSample)
        assert sample.tok_s > 0.0

    def test_unload_resets_active_layer(self) -> None:
        engine = _engine(AccelerationLayer.LOOKAHEAD)
        engine.unload()
        assert engine.active_layer == AccelerationLayer.NONE
        assert engine.speculative_active is False


# ---- load_validated_pairs --------------------------------------------------------


def test_load_validated_pairs_returns_empty_when_file_missing(tmp_path) -> None:
    assert load_validated_pairs(tmp_path / "nonexistent.json") == []


def test_load_validated_pairs_returns_empty_on_malformed_file(tmp_path) -> None:
    p = tmp_path / "pairs.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_validated_pairs(p) == []


def test_load_validated_pairs_skips_unvalidated(tmp_path) -> None:
    data = {"pairs": [
        {"target_model_id": "m1", "draft_model_id": "d1", "validated": True},
        {"target_model_id": "m2", "draft_model_id": "d2", "validated": False},
    ]}
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    pairs = load_validated_pairs(p)
    assert [pp.target_model_id for pp in pairs] == ["m1"]


class TestLoadOrderAndSelfRegulation:
    def test_layer_is_chosen_before_the_model_loads(self) -> None:
        """Switching a loaded llama.cpp model to speculative mode rebuilds it (two copies at once)."""
        events: list[str] = []

        class _Recording(FakeAcceleratedRunner):
            def set_acceleration(self, mode: str) -> bool:
                events.append(f"mode:{mode}")
                return super().set_acceleration(mode)

            def load(self, model_id: str, quant_level: str) -> None:
                events.append("load")
                super().load(model_id, quant_level)

        _engine(AccelerationLayer.DRAFT_MODEL, runner=_Recording(), pair=True)
        assert events == [f"mode:{ACCEL_DRAFT_MODEL}", "load"]

    def test_self_regulating_runner_is_never_stepped_down(self) -> None:
        class _Confident(FakeAcceleratedRunner):
            self_regulating = True

        runner = _Confident(acceptance_rate=0.1)
        engine = _engine(AccelerationLayer.DRAFT_MODEL, runner=runner, pair=True, monitor=_bad_monitor())
        for _ in range(3):
            engine.generate_accelerated("p", 10)
        assert engine.active_layer == AccelerationLayer.DRAFT_MODEL
        assert runner.mode == ACCEL_DRAFT_MODEL
