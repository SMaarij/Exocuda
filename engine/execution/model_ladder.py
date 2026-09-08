
"""
engine/execution/model_ladder.py

Track A, Phase 2 (Weeks 3-5) -- Tasks 5, 6.

Owns FR-02 in full:
    FR-02.1  Deterministic rung selection given an identical HardwareProfile.
    FR-02.2  Picks the HIGHEST rung whose tok/s floor and VRAM ceiling the
             HardwareProfile clears.
    FR-02.3  OOM triggers automatic step-down; no user action required.
    FR-02.4  Mid-session throughput collapse triggers the same step-down.

Relationship to Phase 1:
    HardwareProfiler used DEFAULT_MIN_TOK_S_FLOOR = 3.0 as a bootstrap
    heuristic before this module existed. LocalExecutionBackend (Task 11)
    calls ModelLadder.select_rung() instead -- the per-rung precision
    supersedes the single flat floor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from shared.types import HardwareProfile

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

DEFAULT_COLLAPSE_THRESHOLD = 0.5   # step-down when rolling avg < 50% of floor
DEFAULT_COLLAPSE_WINDOW    = 10    # number of tok/s samples in rolling window


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantRung:
    """
    One rung of the quantization ladder.

    label       -- quant level string, e.g. "q4_K_M"
    min_tok_s   -- minimum tok/s required to be selected for this rung
    max_vram_gb -- VRAM ceiling; 0.0 means CPU-only machines are eligible
    """

    label: str
    min_tok_s: float
    max_vram_gb: float


# Default 3-rung ladder mirroring Phase 1 quant levels.
DEFAULT_LADDER: tuple[QuantRung, ...] = (
    QuantRung(label="q4_K_M", min_tok_s=3.0,  max_vram_gb=0.0),   # floor rung
    QuantRung(label="q5_K_M", min_tok_s=8.0,  max_vram_gb=4.0),
    QuantRung(label="q8_0",   min_tok_s=15.0, max_vram_gb=8.0),
)


# --------------------------------------------------------------------------
# ModelLadder
# --------------------------------------------------------------------------


class ModelLadder:
    """
    Ranked quantization ladder with deterministic rung selection (FR-02.1,
    FR-02.2) and automatic step-down on OOM or throughput collapse
    (FR-02.3, FR-02.4).

    All state derives from the injected rungs tuple plus the HardwareProfile
    passed at call time -- no mutable per-request state stored here.
    Every public method is pure and thread-safe.
    """

    def __init__(
        self,
        rungs: tuple[QuantRung, ...] = DEFAULT_LADDER,
        collapse_window: int = DEFAULT_COLLAPSE_WINDOW,
        collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD,
    ) -> None:
        if not rungs:
            raise ValueError("ModelLadder requires at least one rung.")
        # Sort ascending by min_tok_s so the selection loop keeps the last
        # (highest quality) rung that passes all criteria.
        self._rungs: tuple[QuantRung, ...] = tuple(
            sorted(rungs, key=lambda r: r.min_tok_s)
        )
        self._collapse_window = collapse_window
        self._collapse_threshold = collapse_threshold

    # ---- FR-02.1 / FR-02.2 : deterministic selection ------------------

    def select_rung(self, profile: HardwareProfile) -> Optional[QuantRung]:
        """
        Return the highest rung this profile can sustain, or None if no
        rung clears the floor (Pipeline should route to CascadeCloud).

        Pure function of profile -- same input always returns same output
        (FR-02.1 / NFR-08).
        """
        best_tok_s = max(profile["measured_tok_s"].values(), default=0.0)
        vram_gb    = profile["vram_gb"]

        selected: Optional[QuantRung] = None
        for rung in self._rungs:   # ascending; overwrite keeps the highest
            tok_ok  = best_tok_s >= rung.min_tok_s
            vram_ok = rung.max_vram_gb == 0.0 or vram_gb >= rung.max_vram_gb
            if tok_ok and vram_ok:
                selected = rung
        return selected

    # ---- FR-02.3 : explicit step-down -----------------------------------

    def step_down(self, current_rung: QuantRung) -> Optional[QuantRung]:
        """
        Return the next lower rung below current_rung, or None if
        current_rung is already the floor.

        Called on OOM (FR-02.3) or detect_throughput_collapse() == True
        (FR-02.4). Raises ValueError if rung is not in this ladder.
        """
        for i, rung in enumerate(self._rungs):
            if rung == current_rung:
                return self._rungs[i - 1] if i > 0 else None
        raise ValueError(
            f"Rung {current_rung.label!r} not in ladder. "
            f"Known: {[r.label for r in self._rungs]}"
        )

    # ---- FR-02.4 : throughput-collapse detection ------------------------

    def detect_throughput_collapse(
        self,
        tok_s_stream: Iterable[float],
        current_rung: QuantRung,
    ) -> bool:
        """
        Consume a stream of real-time tok/s samples. Return True the first
        moment the rolling-window average drops below
            collapse_threshold * current_rung.min_tok_s.

        Returns False if the stream ends without a collapse.
        """
        floor  = current_rung.min_tok_s * self._collapse_threshold
        window: deque[float] = deque(maxlen=self._collapse_window)

        for sample in tok_s_stream:
            window.append(sample)
            if len(window) == self._collapse_window:
                if (sum(window) / self._collapse_window) < floor:
                    return True
        return False

    # ---- Convenience properties -----------------------------------------

    @property
    def rungs(self) -> tuple[QuantRung, ...]:
        """All rungs, ascending by min_tok_s."""
        return self._rungs

    @property
    def floor_rung(self) -> QuantRung:
        """Lowest-quality rung -- last fallback before cloud routing."""
        return self._rungs[0]

    @property
    def ceiling_rung(self) -> QuantRung:
        """Highest-quality rung -- best case for a capable machine."""
        return self._rungs[-1]
