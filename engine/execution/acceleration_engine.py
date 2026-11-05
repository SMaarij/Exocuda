"""
engine/execution/acceleration_engine.py

Owns FR-03:
    FR-03.1  Layer 1: draft-free speculative decoding (prompt-lookup
             n-gram drafts) -- needs no extra model and no training.
    FR-03.2  Layer 2: draft heads (Medusa/EAGLE) -- only when the model
             has trained heads. Layer 3: paired draft model -- only when a
             validated pair exists in validated_pairs.json. Each falls
             back one level when unavailable.
    FR-03.3  Rolling acceptance-rate monitor: a rate < 60% over the window
             automatically disables the current layer.
    FR-03.4  No output quality change when a layer is disabled.

Every number this engine reports is the runner's own measurement. A
layer only counts as active when the runner both accepted it
(`AcceleratedRunner.set_acceleration`) and reports in the sample that it
actually ran; a runner that can't accelerate (e.g. Ollama) always reports
plain decoding. Whether a layer is *faster* is an empirical question per
machine -- see scripts/benchmark_acceleration.py and
docs/BENCHMARK_REPORT.md for measured results.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from engine.execution.model_runner import (
    ACCEL_DRAFT_MODEL,
    ACCEL_NONE,
    ACCEL_PROMPT_LOOKUP,
    AcceleratedRunner,
    GenerationSample,
    ReferenceModelRunner,
)

DEFAULT_ACCEPTANCE_THRESHOLD = 0.60   # step down below 60% acceptance rate
DEFAULT_ACCEPTANCE_WINDOW    = 20     # rolling window size (samples)

DEFAULT_PAIRS_PATH = Path.home() / ".cascade" / "validated_pairs.json"

ACCEL_DRAFT_HEADS = "draft_heads"


class AccelerationLayer(Enum):
    """
    NONE        -- plain autoregressive decode.
    LOOKAHEAD   -- Layer 1: draft-free speculative decoding (prompt lookup).
    DRAFT_HEADS -- Layer 2: requires trained heads on the loaded model.
    DRAFT_MODEL -- Layer 3: requires a validated draft-model pair.
    """
    NONE        = auto()
    LOOKAHEAD   = auto()
    DRAFT_HEADS = auto()
    DRAFT_MODEL = auto()


_MODE_FOR_LAYER = {
    AccelerationLayer.NONE:        ACCEL_NONE,
    AccelerationLayer.LOOKAHEAD:   ACCEL_PROMPT_LOOKUP,
    AccelerationLayer.DRAFT_HEADS: ACCEL_DRAFT_HEADS,
    AccelerationLayer.DRAFT_MODEL: ACCEL_DRAFT_MODEL,
}
_LAYER_FOR_MODE = {mode: layer for layer, mode in _MODE_FOR_LAYER.items()}


class AcceptanceRateMonitor:
    """Rolling-window acceptance-rate tracker (FR-03.3). Pure accumulator."""

    def __init__(
        self,
        threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        window: int = DEFAULT_ACCEPTANCE_WINDOW,
    ) -> None:
        self._threshold = threshold
        self._window: deque[float] = deque(maxlen=window)

    def record(self, acceptance_rate: float) -> None:
        self._window.append(acceptance_rate)

    def should_step_down(self) -> bool:
        """True only once the window is full AND its average is below threshold."""
        if len(self._window) < self._window.maxlen:  # type: ignore[arg-type]
            return False
        return (sum(self._window) / len(self._window)) < self._threshold

    def reset(self) -> None:
        self._window.clear()

    @property
    def current_average(self) -> Optional[float]:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)


@dataclass
class DraftPair:
    """One validated (target, draft) model pair for Layer 3."""
    target_model_id: str
    draft_model_id: str
    validated: bool = True


def load_validated_pairs(pairs_path: Path) -> list[DraftPair]:
    """Load the registry written by scripts/validate_draft_pairs.py; [] if absent or malformed."""
    if not pairs_path.exists():
        return []
    try:
        data = json.loads(pairs_path.read_text(encoding="utf-8"))
        return [
            DraftPair(
                target_model_id=p["target_model_id"],
                draft_model_id=p["draft_model_id"],
                validated=p.get("validated", True),
            )
            for p in data.get("pairs", [])
            if p.get("validated", True)
        ]
    except (json.JSONDecodeError, KeyError):
        return []


@dataclass(frozen=True)
class AcceleratedGenerationSample:
    """A runner's GenerationSample plus the layer that actually ran."""
    base: GenerationSample
    active_layer: AccelerationLayer
    acceptance_rate: float   # measured; 1.0 when nothing speculative ran

    @property
    def tok_s(self) -> float:
        return self.base.tok_s

    @property
    def peak_memory_gb(self) -> float:
        return self.base.peak_memory_gb

    @property
    def speculative_active(self) -> bool:
        return self.active_layer != AccelerationLayer.NONE


class AccelerationEngine:
    """
    Acceleration wrapper around a ReferenceModelRunner, slotting into
    LocalExecutionBackend where a bare runner would.

    At load(): resolve the highest layer this model is eligible for
    (FR-03.2), then ask the runner to use it, walking down
    DRAFT_MODEL -> DRAFT_HEADS -> LOOKAHEAD -> NONE until the runner
    accepts one. After each generation, the runner's measured acceptance
    rate feeds the monitor; a sustained low rate steps down a layer
    (FR-03.3).
    """

    def __init__(
        self,
        runner: ReferenceModelRunner,
        preferred_layer: AccelerationLayer = AccelerationLayer.LOOKAHEAD,
        monitor: Optional[AcceptanceRateMonitor] = None,
        pairs_path: Path = DEFAULT_PAIRS_PATH,
        model_has_draft_heads: bool = False,
        validated_pairs: Optional[list[DraftPair]] = None,
    ) -> None:
        self._runner = runner
        self._monitor = monitor or AcceptanceRateMonitor()
        self._preferred_layer = preferred_layer
        self._model_has_draft_heads = model_has_draft_heads
        self._validated_pairs: list[DraftPair] = (
            validated_pairs
            if validated_pairs is not None
            else load_validated_pairs(pairs_path)
        )
        self._active_layer: AccelerationLayer = AccelerationLayer.NONE
        self._loaded_model_id: Optional[str] = None

    def load(self, model_id: str, quant_level: str) -> None:
        self._runner.load(model_id, quant_level)
        self._loaded_model_id = model_id
        self._active_layer = self._apply_layer(self._resolve_layer(model_id))

    def unload(self) -> None:
        self._runner.unload()
        self._loaded_model_id = None
        self._active_layer = AccelerationLayer.NONE

    def generate_accelerated(self, prompt: str, max_tokens: int) -> AcceleratedGenerationSample:
        base = self._runner.generate(prompt, max_tokens)
        ran_layer = _LAYER_FOR_MODE.get(base.acceleration, AccelerationLayer.NONE)
        acceptance = base.acceptance_rate if base.acceptance_rate is not None else 1.0

        if base.acceptance_rate is not None and ran_layer != AccelerationLayer.NONE:
            self._monitor.record(base.acceptance_rate)
            if self._monitor.should_step_down():
                self._step_down_layer()

        return AcceleratedGenerationSample(base=base, active_layer=ran_layer, acceptance_rate=acceptance)

    # ---- Internal --------------------------------------------------------

    def _resolve_layer(self, model_id: str) -> AccelerationLayer:
        """Highest layer this model is eligible for, before asking the runner."""
        if self._preferred_layer == AccelerationLayer.DRAFT_MODEL:
            if self._find_validated_pair(model_id) is not None:
                return AccelerationLayer.DRAFT_MODEL
            if self._model_has_draft_heads:
                return AccelerationLayer.DRAFT_HEADS
            return AccelerationLayer.LOOKAHEAD
        if self._preferred_layer == AccelerationLayer.DRAFT_HEADS:
            if self._model_has_draft_heads:
                return AccelerationLayer.DRAFT_HEADS
            return AccelerationLayer.LOOKAHEAD
        return self._preferred_layer

    def _lower_layer(self, layer: AccelerationLayer) -> AccelerationLayer:
        if layer == AccelerationLayer.DRAFT_MODEL:
            return AccelerationLayer.DRAFT_HEADS if self._model_has_draft_heads else AccelerationLayer.LOOKAHEAD
        if layer == AccelerationLayer.DRAFT_HEADS:
            return AccelerationLayer.LOOKAHEAD
        return AccelerationLayer.NONE

    def _apply_layer(self, layer: AccelerationLayer) -> AccelerationLayer:
        """Ask the runner for `layer`, walking down until it accepts one."""
        if not isinstance(self._runner, AcceleratedRunner):
            return AccelerationLayer.NONE
        while layer != AccelerationLayer.NONE:
            if self._runner.set_acceleration(_MODE_FOR_LAYER[layer]):
                return layer
            layer = self._lower_layer(layer)
        self._runner.set_acceleration(ACCEL_NONE)
        return AccelerationLayer.NONE

    def _find_validated_pair(self, model_id: str) -> Optional[DraftPair]:
        for pair in self._validated_pairs:
            if pair.target_model_id == model_id and pair.validated:
                return pair
        return None

    def _step_down_layer(self) -> None:
        self._active_layer = self._apply_layer(self._lower_layer(self._active_layer))
        self._monitor.reset()

    @property
    def active_layer(self) -> AccelerationLayer:
        return self._active_layer

    @property
    def monitor(self) -> AcceptanceRateMonitor:
        return self._monitor

    @property
    def speculative_active(self) -> bool:
        return self._active_layer != AccelerationLayer.NONE
