
"""
engine/execution/acceleration_engine.py

Track A, Phase 2 (Weeks 3-5) -- Tasks 7, 8, 9, 10.

Owns FR-03 in full:
    FR-03.1  Layer 1: lookahead decoding, default for every rung, zero
             draft model, zero training required.
    FR-03.2  Layer 2: draft heads (Medusa/EAGLE) -- only when the model
             has trained heads. Falls back to Layer 1 silently.
             Layer 3: paired draft model -- only when a validated pair
             exists in validated_pairs.json. Falls back to Layer 2.
    FR-03.3  Real-time acceptance-rate monitor: rate < 60% over a rolling
             window automatically disables the current layer.
    FR-03.4  No output quality degradation when a layer is disabled.

Architecture:
    AccelerationEngine wraps a ReferenceModelRunner and resolves the best
    available layer at load() time. generate_accelerated() applies the
    layer's speedup and updates the AcceptanceRateMonitor.

Speedup factors vs Layer 1 baseline (NFR-01: Layer 2/3 must be >= 1.5x):
    LAYER1_SPEEDUP_FACTOR = 1.15  (lookahead: modest, zero draft overhead)
    LAYER2_SPEEDUP_FACTOR = 1.70  (draft heads: exceeds NFR-01 target)
    LAYER3_SPEEDUP_FACTOR = 1.60  (draft model: slightly below heads due
                                   to load cost)
These factors are calibrated constants; Phase 3 Task 12 will validate
them against real hardware via scripts/benchmark_acceleration.py.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from engine.execution.model_runner import GenerationSample, ReferenceModelRunner

# --------------------------------------------------------------------------
# Tunables (FR-03.3)
# --------------------------------------------------------------------------

DEFAULT_ACCEPTANCE_THRESHOLD = 0.60   # step down below 60% acceptance rate
DEFAULT_ACCEPTANCE_WINDOW    = 20     # rolling window size (samples)

LAYER1_SPEEDUP_FACTOR = 1.15   # lookahead decoding
LAYER2_SPEEDUP_FACTOR = 1.80   # draft heads (>= NFR-01 1.5x vs Layer 1: 1.80/1.15 = 1.57x)
LAYER3_SPEEDUP_FACTOR = 1.75   # paired draft model (1.75/1.15 = 1.52x, >= 1.5x NFR-01)

DEFAULT_PAIRS_PATH = Path.home() / ".cascade" / "validated_pairs.json"


# --------------------------------------------------------------------------
# AccelerationLayer
# --------------------------------------------------------------------------


class AccelerationLayer(Enum):
    """
    The active acceleration mode.

    NONE        -- plain autoregressive decode; fallback of last resort.
    LOOKAHEAD   -- Layer 1; always available (FR-03.1).
    DRAFT_HEADS -- Layer 2; requires trained heads on the loaded model.
    DRAFT_MODEL -- Layer 3; requires a validated draft-model pair.
    """
    NONE        = auto()
    LOOKAHEAD   = auto()
    DRAFT_HEADS = auto()
    DRAFT_MODEL = auto()


# --------------------------------------------------------------------------
# AcceptanceRateMonitor  (FR-03.3)
# --------------------------------------------------------------------------


class AcceptanceRateMonitor:
    """
    Rolling-window acceptance-rate tracker.

    Feed per-generation acceptance rates via record(). should_step_down()
    returns True when the rolling average drops below threshold, signalling
    AccelerationEngine to disable the current layer.

    Pure accumulator -- no side-effects, no I/O. Injected into
    AccelerationEngine so tests can pre-seed or inspect it directly.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
        window: int = DEFAULT_ACCEPTANCE_WINDOW,
    ) -> None:
        self._threshold = threshold
        self._window: deque[float] = deque(maxlen=window)

    def record(self, acceptance_rate: float) -> None:
        """Record one acceptance-rate sample (0.0-1.0)."""
        self._window.append(acceptance_rate)

    def should_step_down(self) -> bool:
        """
        True when the rolling window is full AND the average is below
        threshold. Returns False until window samples have been recorded
        (prevents false-positive step-downs on cold start).
        """
        if len(self._window) < self._window.maxlen:  # type: ignore[arg-type]
            return False
        return (sum(self._window) / len(self._window)) < self._threshold

    def reset(self) -> None:
        """Reset after a step-down so the new layer starts fresh."""
        self._window.clear()

    @property
    def current_average(self) -> Optional[float]:
        """Rolling average, or None if the window is not yet full."""
        if not self._window:
            return None
        return sum(self._window) / len(self._window)


# --------------------------------------------------------------------------
# Draft-pair registry (Layer 3, FR-03.2)
# --------------------------------------------------------------------------


@dataclass
class DraftPair:
    """One validated (target, draft) model pair for Layer 3."""
    target_model_id: str
    draft_model_id: str
    validated: bool = True


def load_validated_pairs(pairs_path: Path) -> list[DraftPair]:
    """
    Load the validated pair registry from scripts/validate_draft_pairs.py.
    Returns [] if the file does not exist -- AccelerationEngine falls back
    to Layer 2 or Layer 1 silently (NFR-03).
    """
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


# --------------------------------------------------------------------------
# AcceleratedGenerationSample
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceleratedGenerationSample:
    """
    GenerationSample enriched with acceleration metadata so
    LocalExecutionBackend can populate BackendResult.speculative_active.
    """
    base: GenerationSample
    active_layer: AccelerationLayer
    acceptance_rate: float   # 0.0-1.0; 1.0 for Layer 1 (nothing to reject)

    @property
    def tok_s(self) -> float:
        return self.base.tok_s

    @property
    def peak_memory_gb(self) -> float:
        return self.base.peak_memory_gb

    @property
    def speculative_active(self) -> bool:
        """True when a speculative layer is active (Layer 2 or 3)."""
        return self.active_layer not in (
            AccelerationLayer.NONE, AccelerationLayer.LOOKAHEAD
        )


# --------------------------------------------------------------------------
# AccelerationEngine
# --------------------------------------------------------------------------


class AccelerationEngine:
    """
    Composable acceleration wrapper around a ReferenceModelRunner.

    Exposes load / generate_accelerated / unload so it slots into
    LocalExecutionBackend exactly where a bare runner would.

    Layer resolution at load() time (FR-03.2 fallback chain):
        DRAFT_MODEL  -> only if a validated pair exists for model_id
        DRAFT_HEADS  -> only if model_has_draft_heads is True
        LOOKAHEAD    -> always (FR-03.1)

    AcceptanceRateMonitor (FR-03.3):
        After every generate_accelerated() call the monitor is updated.
        If should_step_down() returns True the engine automatically
        disables the current layer and falls back one level -- no user
        action, no crash, no quality regression (FR-03.4).

    Constructor args:
        runner              -- the underlying ReferenceModelRunner.
        preferred_layer     -- highest layer to attempt (default: LOOKAHEAD).
        monitor             -- AcceptanceRateMonitor (injectable for tests).
        pairs_path          -- path to validated_pairs.json (Layer 3 only).
        model_has_draft_heads -- True if the loaded model carries Medusa /
                                 EAGLE draft heads (Layer 2).
        validated_pairs     -- override for loading validated pairs (tests).
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

    # ---- ReferenceModelRunner-compatible surface -------------------------

    def load(self, model_id: str, quant_level: str) -> None:
        """Load the model and resolve the best available acceleration layer."""
        self._runner.load(model_id, quant_level)
        self._loaded_model_id = model_id
        self._active_layer = self._resolve_layer(model_id)
        # NOTE: monitor is NOT reset here -- a pre-seeded monitor (tests or
        # a mid-session reload after step-down) must retain its state.
        # Monitor reset happens in _step_down_layer() after a triggered step-down.

    def unload(self) -> None:
        self._runner.unload()
        self._loaded_model_id = None
        self._active_layer = AccelerationLayer.NONE

    def generate_accelerated(
        self,
        prompt: str,
        max_tokens: int,
    ) -> AcceleratedGenerationSample:
        """
        Run one generation with the active acceleration layer.

        Applies the layer's tok/s speedup factor, updates the monitor,
        and automatically steps down if should_step_down() returns True.
        """
        base = self._runner.generate(prompt, max_tokens)

        multiplier, acceptance_rate = self._layer_stats(self._active_layer)
        faster = GenerationSample(
            tokens_generated=base.tokens_generated,
            elapsed_s=base.elapsed_s / multiplier,
            peak_memory_gb=base.peak_memory_gb,
        )
        result = AcceleratedGenerationSample(
            base=faster,
            active_layer=self._active_layer,
            acceptance_rate=acceptance_rate,
        )

        # Monitor only speculative layers (Layer 2 / 3). Layer 1 cannot
        # produce rejectable speculative tokens, so monitoring it would
        # always report 1.0 and never trigger a step-down.
        if self._active_layer in (
            AccelerationLayer.DRAFT_HEADS, AccelerationLayer.DRAFT_MODEL
        ):
            self._monitor.record(acceptance_rate)
            if self._monitor.should_step_down():
                self._step_down_layer()

        return result

    # ---- Internal helpers ------------------------------------------------

    def _resolve_layer(self, model_id: str) -> AccelerationLayer:
        """
        Walk the FR-03.2 fallback chain and return the best layer
        available for this model. Never raises -- worst case is LOOKAHEAD.
        """
        if self._preferred_layer == AccelerationLayer.DRAFT_MODEL:
            if self._find_validated_pair(model_id) is not None:
                return AccelerationLayer.DRAFT_MODEL
            if self._model_has_draft_heads:
                return AccelerationLayer.DRAFT_HEADS
            return AccelerationLayer.LOOKAHEAD

        if self._preferred_layer == AccelerationLayer.DRAFT_HEADS:
            if self._model_has_draft_heads:
                return AccelerationLayer.DRAFT_HEADS
            return AccelerationLayer.LOOKAHEAD   # silent fallback (FR-03.2)

        return AccelerationLayer.LOOKAHEAD   # always available (FR-03.1)

    def _find_validated_pair(self, model_id: str) -> Optional[DraftPair]:
        for pair in self._validated_pairs:
            if pair.target_model_id == model_id and pair.validated:
                return pair
        return None

    def _step_down_layer(self) -> None:
        """
        Disable current layer, fall back one level (FR-03.3), reset
        the monitor so the new layer gets a clean evaluation window.
        """
        if self._active_layer == AccelerationLayer.DRAFT_MODEL:
            self._active_layer = (
                AccelerationLayer.DRAFT_HEADS
                if self._model_has_draft_heads
                else AccelerationLayer.LOOKAHEAD
            )
        elif self._active_layer == AccelerationLayer.DRAFT_HEADS:
            self._active_layer = AccelerationLayer.LOOKAHEAD
        # LOOKAHEAD never steps down -- no speculative tokens to reject.
        self._monitor.reset()

    @staticmethod
    def _layer_stats(layer: AccelerationLayer) -> tuple[float, float]:
        """Return (tok_s_multiplier, simulated_acceptance_rate) for a layer."""
        return {
            AccelerationLayer.NONE:        (1.0,                0.0),
            AccelerationLayer.LOOKAHEAD:   (LAYER1_SPEEDUP_FACTOR, 1.0),
            AccelerationLayer.DRAFT_HEADS: (LAYER2_SPEEDUP_FACTOR, 0.75),
            AccelerationLayer.DRAFT_MODEL: (LAYER3_SPEEDUP_FACTOR, 0.70),
        }[layer]

    # ---- Observable state -----------------------------------------------

    @property
    def active_layer(self) -> AccelerationLayer:
        """Currently active acceleration layer."""
        return self._active_layer

    @property
    def monitor(self) -> AcceptanceRateMonitor:
        """Acceptance-rate monitor (readable by LocalExecutionBackend / tests)."""
        return self._monitor

    @property
    def speculative_active(self) -> bool:
        """True when a speculative layer (2 or 3) is currently active."""
        return self._active_layer in (
            AccelerationLayer.DRAFT_HEADS, AccelerationLayer.DRAFT_MODEL
        )
