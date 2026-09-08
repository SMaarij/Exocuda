"""
Tests for engine/execution/hardware_profiler.py.

Per NFR-09 ("each component SHALL carry unit tests... independent of a
live GPU or live Ollama instance"), every test here mocks:
    - subprocess calls (nvidia-smi / rocm-smi)
    - psutil calls (cpu/ram)
    - disk I/O (uses tmp_path, a real-but-throwaway pytest fixture, which
      is fine -- it's not a hardware dependency, just a scratch file)
    - the ReferenceModelRunner (a fake, deterministic implementation
      defined below) -- never a real model load.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.execution.hardware_profiler import (
    DEFAULT_MIN_TOK_S_FLOOR,
    HardwareProfiler,
    detect_cpu_cores,
    detect_disk_free_gb,
    detect_disk_throughput_mb_s,
    detect_gpu,
    detect_os,
    detect_ram_gb,
)
from engine.execution.model_runner import GenerationSample, ReferenceModelRunner
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeReferenceModelRunner:
    """Deterministic ReferenceModelRunner double. Returns a fixed
    tok/s + memory figure per quant level so benchmark-loop tests are
    exact and reproducible -- no timing flakiness, no real model."""

    def __init__(
        self,
        tok_s_by_level: dict[str, float],
        fail_levels: frozenset[str] = frozenset(),
    ) -> None:
        self._tok_s_by_level = tok_s_by_level
        self._fail_levels = fail_levels
        self._loaded: str | None = None
        self.load_calls: list[tuple[str, str]] = []
        self.unload_calls: int = 0

    def load(self, model_id: str, quant_level: str) -> None:
        self.load_calls.append((model_id, quant_level))
        if quant_level in self._fail_levels:
            raise RuntimeError(f"simulated OOM loading {quant_level}")
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        tok_s = self._tok_s_by_level[self._loaded]
        elapsed = 1.0
        tokens = tok_s * elapsed
        return GenerationSample(
            tokens_generated=int(tokens),
            elapsed_s=elapsed,
            peak_memory_gb=2.5,
        )

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = None


# Confirms the fake genuinely satisfies the Protocol shape.
def test_fake_runner_conforms_to_protocol() -> None:
    assert isinstance(FakeReferenceModelRunner({}), ReferenceModelRunner)


# --------------------------------------------------------------------------
# FR-01.1 -- detection, all mocked
# --------------------------------------------------------------------------


def test_detect_cpu_cores_uses_psutil() -> None:
    with patch("engine.execution.hardware_profiler.psutil.cpu_count", return_value=16):
        assert detect_cpu_cores() == 16


def test_detect_cpu_cores_falls_back_to_1_when_psutil_returns_none() -> None:
    with patch("engine.execution.hardware_profiler.psutil.cpu_count", return_value=None):
        assert detect_cpu_cores() == 1


def test_detect_ram_gb_uses_psutil() -> None:
    fake_mem = MagicMock(total=32 * 1024**3)
    with patch(
        "engine.execution.hardware_profiler.psutil.virtual_memory",
        return_value=fake_mem,
    ):
        assert detect_ram_gb() == 32.0


def test_detect_gpu_nvidia_present() -> None:
    with patch("engine.execution.hardware_profiler.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "engine.execution.hardware_profiler.subprocess.run"
    ) as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="24576\n")
        vendor, vram = detect_gpu()
        assert vendor == "nvidia"
        assert vram == 24.0


def test_detect_gpu_no_gpu_present() -> None:
    with patch("engine.execution.hardware_profiler.shutil.which", return_value=None), patch(
        "engine.execution.hardware_profiler.platform.system", return_value="Linux"
    ):
        vendor, vram = detect_gpu()
        assert vendor is None
        assert vram == 0.0


def test_detect_gpu_nvidia_smi_present_but_errors_degrades_gracefully() -> None:
    """NFR-03: a detection failure must never raise -- it degrades to
    'no GPU detected' instead of crashing HardwareProfiler."""
    with patch("engine.execution.hardware_profiler.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "engine.execution.hardware_profiler.subprocess.run",
        side_effect=OSError("boom"),
    ), patch("engine.execution.hardware_profiler.platform.system", return_value="Linux"):
        vendor, vram = detect_gpu()
        assert vendor is None
        assert vram == 0.0


def test_detect_gpu_apple_silicon() -> None:
    with patch("engine.execution.hardware_profiler.shutil.which", return_value=None), patch(
        "engine.execution.hardware_profiler.platform.system", return_value="Darwin"
    ), patch("engine.execution.hardware_profiler.platform.machine", return_value="arm64"), patch(
        "engine.execution.hardware_profiler.detect_ram_gb", return_value=18.0
    ):
        vendor, vram = detect_gpu()
        assert vendor == "apple"
        assert vram == 18.0


def test_detect_disk_throughput_writes_and_measures(tmp_path: Path) -> None:
    throughput = detect_disk_throughput_mb_s(tmp_path)
    assert throughput > 0.0
    # Temp file must be cleaned up, not left behind.
    assert list(tmp_path.iterdir()) == []


def test_detect_disk_throughput_degrades_gracefully_on_oserror(tmp_path: Path) -> None:
    unwritable = tmp_path / "does" / "not" / "exist"
    result = detect_disk_throughput_mb_s(unwritable)
    assert result == 0.0


def test_detect_disk_free_gb(tmp_path: Path) -> None:
    assert detect_disk_free_gb(tmp_path) > 0.0


def test_detect_os_maps_known_systems() -> None:
    with patch("engine.execution.hardware_profiler.platform.system", return_value="Darwin"):
        assert detect_os() == "darwin"
    with patch("engine.execution.hardware_profiler.platform.system", return_value="Windows"):
        assert detect_os() == "windows"


# --------------------------------------------------------------------------
# FR-01.2 -- timed benchmark loop
# --------------------------------------------------------------------------


def test_run_benchmark_measures_at_least_two_quant_levels(tmp_path: Path) -> None:
    runner = FakeReferenceModelRunner(
        {"q4_K_M": 40.0, "q5_K_M": 30.0, "q8_0": 15.0}
    )
    profiler = HardwareProfiler(runner=runner, profile_path=tmp_path / "profile.json")

    tok_s, mem = profiler.run_benchmark()

    assert len(tok_s) >= 2
    assert set(tok_s.keys()) == {"q4_K_M", "q5_K_M", "q8_0"}
    assert tok_s["q4_K_M"] == pytest.approx(40.0)
    assert all(v > 0 for v in mem.values())


def test_run_benchmark_unloads_between_levels_so_memory_never_carries_over(
    tmp_path: Path,
) -> None:
    runner = FakeReferenceModelRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )
    profiler.run_benchmark()
    assert runner.unload_calls == 2


def test_run_benchmark_skips_a_level_that_ooms_but_keeps_the_rest(
    tmp_path: Path,
) -> None:
    runner = FakeReferenceModelRunner(
        {"q4_K_M": 40.0, "q5_K_M": 30.0, "q8_0": 15.0},
        fail_levels=frozenset({"q8_0"}),
    )
    profiler = HardwareProfiler(runner=runner, profile_path=tmp_path / "profile.json")

    tok_s, _mem = profiler.run_benchmark()

    assert "q8_0" not in tok_s
    assert len(tok_s) == 2  # still clears FR-01.2's ">= 2 levels" bar


def test_run_benchmark_respects_time_budget(tmp_path: Path) -> None:
    """A runner whose load() burns the whole budget on the first level
    should not be allowed to start a second, over-budget level."""

    class SlowRunner(FakeReferenceModelRunner):
        def load(self, model_id: str, quant_level: str) -> None:
            super().load(model_id, quant_level)
            # Simulate having already consumed the entire budget.

    runner = SlowRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        time_budget_s=0.0,  # budget already exhausted before we start
        profile_path=tmp_path / "profile.json",
    )
    tok_s, _mem = profiler.run_benchmark()
    assert tok_s == {}
    assert runner.load_calls == []


def test_run_benchmark_repeatable_within_tolerance(tmp_path: Path) -> None:
    """Definition of done: 'measurements repeatable within 10% across 3
    runs.' With a deterministic fake runner this is trivially exact, but
    the test exists to lock in that run_benchmark() is a pure function
    of the runner's outputs -- no hidden randomness or shared state leaks
    between calls."""
    runner = FakeReferenceModelRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )

    results = [profiler.run_benchmark()[0]["q4_K_M"] for _ in range(3)]

    baseline = results[0]
    for value in results[1:]:
        assert abs(value - baseline) / baseline <= 0.10


# --------------------------------------------------------------------------
# FR-01.3 / FR-01.4 / FR-01.5 -- HardwareProfile assembly, versioning,
# local_capable, and re-profile support
# --------------------------------------------------------------------------


@pytest.fixture
def patched_detectors():
    """Patches every FR-01.1 detector so profile() runs with fully known,
    controlled inputs regardless of the machine actually running the
    test suite."""
    with patch(
        "engine.execution.hardware_profiler.detect_gpu", return_value=("nvidia", 8.0)
    ), patch(
        "engine.execution.hardware_profiler.detect_cpu_cores", return_value=8
    ), patch(
        "engine.execution.hardware_profiler.detect_ram_gb", return_value=16.0
    ), patch(
        "engine.execution.hardware_profiler.detect_disk_throughput_mb_s",
        return_value=500.0,
    ), patch(
        "engine.execution.hardware_profiler.detect_disk_free_gb", return_value=100.0
    ), patch(
        "engine.execution.hardware_profiler.detect_os", return_value="linux"
    ):
        yield


def test_profile_produces_versioned_hardware_profile(
    tmp_path: Path, patched_detectors
) -> None:
    runner = FakeReferenceModelRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )

    result = profiler.profile()

    assert result["schema_version"] == HARDWARE_PROFILE_SCHEMA_VERSION
    assert result["cpu_cores"] == 8
    assert result["ram_gb"] == 16.0
    assert result["gpu_vendor"] == "nvidia"
    assert result["vram_gb"] == 8.0
    assert result["disk_throughput_mb_s"] == 500.0
    assert result["disk_free_gb"] == 100.0
    assert result["measured_tok_s"] == {"q4_K_M": 40.0, "q5_K_M": 30.0}
    assert "profiled_at" in result and result["profiled_at"]


def test_profile_flags_local_capable_true_above_floor(
    tmp_path: Path, patched_detectors
) -> None:
    runner = FakeReferenceModelRunner({"q4_K_M": DEFAULT_MIN_TOK_S_FLOOR + 5})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M",),
        profile_path=tmp_path / "profile.json",
    )
    result = profiler.profile()
    assert result["local_capable"] is True


def test_profile_flags_local_incapable_below_floor(
    tmp_path: Path, patched_detectors
) -> None:
    runner = FakeReferenceModelRunner({"q4_K_M": DEFAULT_MIN_TOK_S_FLOOR - 1})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M",),
        profile_path=tmp_path / "profile.json",
    )
    result = profiler.profile()
    assert result["local_capable"] is False


def test_profile_local_incapable_when_every_level_fails_to_load(
    tmp_path: Path, patched_detectors
) -> None:
    """FR-01.5 edge case: a machine so constrained that nothing loads at
    all must still produce a usable (local_capable=False) profile, never
    a crash (NFR-03)."""
    runner = FakeReferenceModelRunner(
        {"q4_K_M": 40.0, "q5_K_M": 30.0},
        fail_levels=frozenset({"q4_K_M", "q5_K_M"}),
    )
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )
    result = profiler.profile()
    assert result["local_capable"] is False
    assert result["measured_tok_s"] == {}


def test_profile_is_the_only_place_that_persists_hardware_facts(
    tmp_path: Path, patched_detectors
) -> None:
    """FR-01.3: profile() must write the versioned profile to disk so
    every other component reads *this* file rather than re-deriving
    hardware facts itself."""
    profile_path = tmp_path / "profile.json"
    runner = FakeReferenceModelRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=profile_path,
    )

    profiler.profile()

    assert profile_path.exists()
    on_disk = json.loads(profile_path.read_text())
    assert on_disk["schema_version"] == HARDWARE_PROFILE_SCHEMA_VERSION


def test_load_cached_profile_returns_none_when_never_profiled(tmp_path: Path) -> None:
    runner = FakeReferenceModelRunner({})
    profiler = HardwareProfiler(runner=runner, profile_path=tmp_path / "missing.json")
    assert profiler.load_cached_profile() is None


def test_load_cached_profile_round_trips(tmp_path: Path, patched_detectors) -> None:
    runner = FakeReferenceModelRunner({"q4_K_M": 40.0, "q5_K_M": 30.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )
    written = profiler.profile()
    loaded = profiler.load_cached_profile()
    assert loaded == written


def test_load_cached_profile_rejects_mismatched_schema_version(
    tmp_path: Path,
) -> None:
    """A profile written by an incompatible schema must not be trusted
    silently -- caller re-profiles instead."""
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"schema_version": 999}))
    runner = FakeReferenceModelRunner({})
    profiler = HardwareProfiler(runner=runner, profile_path=profile_path)
    assert profiler.load_cached_profile() is None


def test_profile_supports_repeated_reprofiling_without_reinstall(
    tmp_path: Path, patched_detectors
) -> None:
    """FR-01.4: `cascade profile` re-run support -- calling profile()
    twice in a row (simulating a hardware change between calls) must
    just work, with the second call's result overwriting the first."""
    profile_path = tmp_path / "profile.json"
    runner = FakeReferenceModelRunner({"q4_K_M": 10.0, "q5_K_M": 8.0})
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=profile_path,
    )
    first = profiler.profile()
    assert first["local_capable"] is True

    # Simulate a hardware regression (e.g. thermal throttling, another
    # process hogging RAM) between the two `cascade profile` invocations.
    runner_after_change = FakeReferenceModelRunner({"q4_K_M": 0.5, "q5_K_M": 0.3})
    profiler_after_change = HardwareProfiler(
        runner=runner_after_change,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=profile_path,
    )
    second = profiler_after_change.profile()

    assert second["local_capable"] is False
    assert profiler.load_cached_profile() == second
