
"""
Tests for engine/execution/model_ladder.py -- Tasks 5, 6.

Per NFR-09: all tests run with no live GPU, no live Ollama.
HardwareProfile fixtures are constructed directly; no real hardware calls.
"""

from __future__ import annotations

import pytest

from engine.execution.model_ladder import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_COLLAPSE_WINDOW,
    DEFAULT_LADDER,
    ModelLadder,
    QuantRung,
)
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _profile(
    tok_s: float = 20.0,
    vram_gb: float = 0.0,
    quant_label: str = "q4_K_M",
) -> dict:
    """Minimal HardwareProfile dict for selection tests."""
    from datetime import datetime, timezone
    return {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux",
        "cpu_cores": 8,
        "ram_gb": 16.0,
        "gpu_vendor": None if vram_gb == 0.0 else "nvidia",
        "vram_gb": vram_gb,
        "disk_throughput_mb_s": 500.0,
        "disk_free_gb": 100.0,
        "measured_tok_s": {quant_label: tok_s},
        "peak_memory_gb": {quant_label: 2.0},
        "local_capable": tok_s >= 3.0,
    }


SIMPLE_LADDER = (
    QuantRung(label="low",  min_tok_s=3.0,  max_vram_gb=0.0),
    QuantRung(label="mid",  min_tok_s=8.0,  max_vram_gb=4.0),
    QuantRung(label="high", min_tok_s=15.0, max_vram_gb=8.0),
)


# --------------------------------------------------------------------------
# FR-02.1: determinism
# --------------------------------------------------------------------------


def test_select_rung_is_deterministic() -> None:
    """Same HardwareProfile always returns the same QuantRung (NFR-08)."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=20.0, vram_gb=8.0)
    results = {ladder.select_rung(profile) for _ in range(100)}
    assert len(results) == 1   # exactly one unique result


# --------------------------------------------------------------------------
# FR-02.2: highest viable rung selection
# --------------------------------------------------------------------------


def test_select_rung_returns_highest_rung_profile_clears() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=20.0, vram_gb=8.0)
    rung    = ladder.select_rung(profile)
    assert rung is not None
    assert rung.label == "high"


def test_select_rung_picks_mid_when_tok_s_below_high_floor() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=10.0, vram_gb=4.0)
    rung    = ladder.select_rung(profile)
    assert rung is not None
    assert rung.label == "mid"


def test_select_rung_returns_floor_when_only_floor_clears() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=5.0, vram_gb=0.0)
    rung    = ladder.select_rung(profile)
    assert rung is not None
    assert rung.label == "low"


def test_select_rung_returns_none_below_all_floors() -> None:
    """Profile that clears no rung -> None (route to the self-hosted tier)."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=1.0, vram_gb=0.0)
    assert ladder.select_rung(profile) is None


def test_select_rung_respects_vram_ceiling() -> None:
    """A fast CPU-only machine must not be assigned a rung needing VRAM."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    profile = _profile(tok_s=20.0, vram_gb=0.0)  # no GPU
    rung    = ladder.select_rung(profile)
    assert rung is not None
    assert rung.label == "low"  # mid and high need vram_gb >= 4/8


def test_select_rung_multiple_measured_levels_uses_best() -> None:
    """When profile has multiple measured tok/s, best is used (FR-02.2)."""
    from datetime import datetime, timezone
    profile = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux", "cpu_cores": 8, "ram_gb": 16.0,
        "gpu_vendor": "nvidia", "vram_gb": 8.0,
        "disk_throughput_mb_s": 500.0, "disk_free_gb": 100.0,
        "measured_tok_s": {"q4_K_M": 5.0, "q8_0": 20.0},
        "peak_memory_gb": {"q4_K_M": 1.0, "q8_0": 4.0},
        "local_capable": True,
    }
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    rung   = ladder.select_rung(profile)
    assert rung is not None
    assert rung.label == "high"   # best tok_s (20.0) clears the high floor


def test_select_rung_empty_measured_tok_s_returns_none() -> None:
    """No benchmark data at all -> no rung can be cleared."""
    from datetime import datetime, timezone
    profile = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux", "cpu_cores": 4, "ram_gb": 8.0,
        "gpu_vendor": None, "vram_gb": 0.0,
        "disk_throughput_mb_s": 200.0, "disk_free_gb": 50.0,
        "measured_tok_s": {},
        "peak_memory_gb": {},
        "local_capable": False,
    }
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    assert ladder.select_rung(profile) is None


# --------------------------------------------------------------------------
# FR-02.3: step_down
# --------------------------------------------------------------------------


def test_step_down_from_highest_returns_middle() -> None:
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    high   = ladder.ceiling_rung
    result = ladder.step_down(high)
    assert result is not None
    assert result.label == "mid"


def test_step_down_from_middle_returns_floor() -> None:
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    mid    = next(r for r in SIMPLE_LADDER if r.label == "mid")
    result = ladder.step_down(mid)
    assert result is not None
    assert result.label == "low"


def test_step_down_from_floor_returns_none() -> None:
    """Already at the floor -> None; caller routes to cloud."""
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    assert ladder.step_down(ladder.floor_rung) is None


def test_step_down_rung_not_in_ladder_raises() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    foreign = QuantRung(label="unknown", min_tok_s=50.0, max_vram_gb=0.0)
    with pytest.raises(ValueError, match="not in ladder"):
        ladder.step_down(foreign)


def test_step_down_single_rung_ladder_returns_none() -> None:
    ladder = ModelLadder(rungs=(QuantRung("only", 3.0, 0.0),))
    assert ladder.step_down(ladder.floor_rung) is None


# --------------------------------------------------------------------------
# FR-02.4: throughput collapse detection
# --------------------------------------------------------------------------


def test_detect_throughput_collapse_triggers_on_sustained_drop() -> None:
    """A window full of low tok/s values triggers a collapse."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER, collapse_window=5, collapse_threshold=0.5)
    rung    = next(r for r in SIMPLE_LADDER if r.label == "low")  # min_tok_s=3.0; floor=1.5
    samples = iter([0.5, 0.5, 0.5, 0.5, 0.5])   # all below 1.5
    assert ladder.detect_throughput_collapse(samples, rung) is True


def test_detect_throughput_collapse_does_not_trigger_on_strong_throughput() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER, collapse_window=5)
    rung    = next(r for r in SIMPLE_LADDER if r.label == "low")
    samples = iter([5.0, 5.0, 5.0, 5.0, 5.0])   # well above 1.5
    assert ladder.detect_throughput_collapse(samples, rung) is False


def test_detect_throughput_collapse_ignores_brief_dips() -> None:
    """A single low sample in a mostly-good stream does not trigger."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER, collapse_window=5, collapse_threshold=0.5)
    rung    = next(r for r in SIMPLE_LADDER if r.label == "low")
    samples = iter([5.0, 5.0, 0.1, 5.0, 5.0])   # one bad sample, window avg still OK
    assert ladder.detect_throughput_collapse(samples, rung) is False


def test_detect_throughput_collapse_returns_false_on_empty_stream() -> None:
    ladder  = ModelLadder(rungs=SIMPLE_LADDER)
    rung    = ladder.floor_rung
    assert ladder.detect_throughput_collapse(iter([]), rung) is False


def test_detect_throughput_collapse_returns_false_if_window_never_fills() -> None:
    """Fewer samples than the window -> no verdict -> False."""
    ladder  = ModelLadder(rungs=SIMPLE_LADDER, collapse_window=10)
    rung    = ladder.floor_rung
    samples = iter([0.0, 0.0, 0.0])   # only 3 samples, window needs 10
    assert ladder.detect_throughput_collapse(samples, rung) is False


# --------------------------------------------------------------------------
# Ladder properties
# --------------------------------------------------------------------------


def test_empty_ladder_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least one rung"):
        ModelLadder(rungs=())


def test_rungs_are_sorted_ascending() -> None:
    """Rungs should be ascending by min_tok_s regardless of input order."""
    shuffled = (
        QuantRung("high", 15.0, 8.0),
        QuantRung("low",  3.0,  0.0),
        QuantRung("mid",  8.0,  4.0),
    )
    ladder = ModelLadder(rungs=shuffled)
    labels = [r.label for r in ladder.rungs]
    assert labels == ["low", "mid", "high"]


def test_floor_and_ceiling_properties() -> None:
    ladder = ModelLadder(rungs=SIMPLE_LADDER)
    assert ladder.floor_rung.label   == "low"
    assert ladder.ceiling_rung.label == "high"


def test_default_ladder_has_three_rungs() -> None:
    ladder = ModelLadder()
    assert len(ladder.rungs) == 3
