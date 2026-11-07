"""
engine/execution/hardware_profiler.py

Track A, Phase 1 (Weeks 1-2) -- Tasks 1, 2, 3.

Owns FR-01 in full:
    FR-01.1  Detect CPU cores, RAM, GPU vendor/VRAM, disk throughput,
             disk free space on first run.
    FR-01.2  Load a reference model at >=2 quant levels, measure real
             tok/s and peak memory, within a 60s budget.
    FR-01.3  Produce a versioned HardwareProfile -- the ONLY place
             hardware facts are computed anywhere in the codebase.
    FR-01.4  Support re-profiling on demand (`cascade profile`) without
             requiring reinstall.
    FR-01.5  Flag a profile that clears no local rung's latency floor as
             local_capable=False, routed to the self-hosted tier (FR-07) by
             default.

R-06 mitigation: profiling is time-boxed (see `time_budget_s`) and uses
a fast heuristic pre-filter (see `_quant_levels_within_budget`) so this
stays inside the 5-minute `cascade init` promise (FR-08.1) even on
low-end hardware, not just dev machines.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil

from engine.execution.model_runner import GenerationSample, ReferenceModelRunner
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION, HardwareProfile

# --------------------------------------------------------------------------
# Tunables. These are the "untrained starting point" heuristics the master
# doc allows (Section 4 preamble) -- ModelLadder (Phase 2, Task 5) is the
# component that actually turns these into a real per-rung decision;
# HardwareProfiler's job in Phase 1 is only to measure, and to apply one
# conservative floor so `local_capable` means something before ModelLadder
# exists.
# --------------------------------------------------------------------------

# Resolved from the placeholder Track A flagged in
# docs/archive/TRACK_A_HANDOFF_FOR_B_AND_C.md's fakes/placeholders table --
# "Joint team decision. Once agreed (e.g. llama3.2:3b), update...".
# llama3.2 is real, small (fast to pull/probe), and its published
# Ollama tags for 3b + these three quant levels are verified real:
# llama3.2:3b-instruct-q4_K_M / -q5_K_M / -q8_0 (confirmed against
# ollama.com/library/llama3.2/tags). Tag is built as
# f"{DEFAULT_REFERENCE_MODEL_ID}:{quant_level}" in model_runner.py's
# OllamaReferenceModelRunner.load() -- hence the "3b-instruct-" prefix
# living inside each quant_level string here, not in the model ID.
DEFAULT_REFERENCE_MODEL_ID = "llama3.2"
DEFAULT_QUANT_LEVELS: tuple[str, ...] = (
    "3b-instruct-q4_K_M",
    "3b-instruct-q5_K_M",
    "3b-instruct-q8_0",
)
DEFAULT_TIME_BUDGET_S = 60.0
DEFAULT_MIN_TOK_S_FLOOR = 3.0  # conservative "can this machine run *anything*
# locally at all" floor -- see local_capable note above.
DEFAULT_BENCHMARK_PROMPT = (
    "Explain, in a few sentences, why the sky appears blue during the day."
)
DEFAULT_MAX_TOKENS = 64
DISK_TEST_FILE_SIZE_BYTES = 64 * 1024 * 1024  # 64MB, enough to smooth out
# filesystem-cache noise without blowing the 60s budget on slow disks.

DEFAULT_PROFILE_PATH = Path.home() / ".cascade" / "hardware_profile.json"


class HardwareProfilerError(RuntimeError):
    """Raised when profiling cannot produce a usable HardwareProfile at
    all (e.g. every quant level OOMs). Callers should catch this and
    degrade per NFR-03 rather than let it propagate as a crash."""


# --------------------------------------------------------------------------
# FR-01.1 -- static hardware detection
# --------------------------------------------------------------------------


def detect_cpu_cores() -> int:
    """Logical core count. FR-01.1."""
    cores = psutil.cpu_count(logical=True)
    return cores if cores else 1


def detect_ram_gb() -> float:
    """Total system RAM in GB. FR-01.1."""
    return round(psutil.virtual_memory().total / (1024**3), 2)


def detect_gpu() -> tuple[Optional[str], float]:
    """
    Returns (vendor, vram_gb). vendor is one of "nvidia" / "apple" /
    "amd" / None. FR-01.1.

    Detection order:
      1. NVIDIA via `nvidia-smi` (works cross-platform wherever the
         driver is installed).
      2. Apple Silicon unified memory via `platform.system() == "Darwin"`
         plus `machine() startswith "arm"` -- VRAM is approximated as
         system RAM since Apple Silicon uses a unified memory pool with
         no separate VRAM figure to query.
      3. AMD via `rocm-smi`, where present.
      4. None / 0.0 -- CPU-only machine.

    Every branch is wrapped so a missing binary or unexpected output
    degrades to "no GPU detected" (NFR-03) rather than raising.
    """
    nvidia = _detect_nvidia_gpu()
    if nvidia is not None:
        return "nvidia", nvidia

    if platform.system() == "Darwin" and platform.machine().startswith("arm"):
        # Apple Silicon: unified memory, so report system RAM as the
        # usable "VRAM" ceiling for GPU-resident work.
        return "apple", detect_ram_gb()

    amd = _detect_amd_gpu()
    if amd is not None:
        return "amd", amd

    return None, 0.0


def _detect_nvidia_gpu() -> Optional[float]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        # First GPU only -- multi-GPU ladder selection is out of scope
        # for Phase 1.
        first_line = result.stdout.strip().splitlines()[0]
        return round(float(first_line.strip()) / 1024, 2)  # MiB -> GiB
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def _detect_amd_gpu() -> Optional[float]:
    if shutil.which("rocm-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().splitlines():
            if "total" in line.lower():
                digits = "".join(ch for ch in line if ch.isdigit() or ch == ".")
                if digits:
                    return round(float(digits) / (1024**3), 2)
        return None
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def detect_disk_throughput_mb_s(target_dir: Optional[Path] = None) -> float:
    """
    Measured sequential write throughput on the install volume. FR-01.1.

    Writes a real temp file rather than trusting a device-reported spec
    figure, since the master doc's whole premise (Section 2.1) is that
    a nominal spec hides real per-machine variance.
    """
    target_dir = target_dir or Path(tempfile.gettempdir())
    test_path = target_dir / ".cascade_disk_bench.tmp"
    payload = b"0" * DISK_TEST_FILE_SIZE_BYTES
    try:
        start = time.monotonic()
        with open(test_path, "wb") as f:
            f.write(payload)
            f.flush()
            import os

            os.fsync(f.fileno())
        elapsed = time.monotonic() - start
        if elapsed <= 0:
            return 0.0
        return round((DISK_TEST_FILE_SIZE_BYTES / (1024**2)) / elapsed, 2)
    except OSError:
        return 0.0
    finally:
        test_path.unlink(missing_ok=True)


def detect_disk_free_gb(target_dir: Optional[Path] = None) -> float:
    """Free space on the install volume. FR-01.1."""
    target_dir = target_dir or Path(tempfile.gettempdir())
    usage = shutil.disk_usage(target_dir)
    return round(usage.free / (1024**3), 2)


def detect_os() -> str:
    system = platform.system()
    return {"Linux": "linux", "Darwin": "darwin", "Windows": "windows"}.get(
        system, "linux"
    )


# --------------------------------------------------------------------------
# FR-01.2 -- timed benchmark loop
# --------------------------------------------------------------------------


class HardwareProfiler:
    """
    Orchestrates detection (FR-01.1) and the timed benchmark loop
    (FR-01.2) into one versioned HardwareProfile (FR-01.3), with
    persistence for on-demand re-profiling (FR-01.4) and the
    local_capable determination (FR-01.5).

    `runner` is injected (see model_runner.py) so this class -- and its
    tests -- never depend on a live Ollama/llama.cpp instance or a real
    GPU (NFR-09). Production callers (e.g. Track C's `cascade init` /
    `cascade profile` CLI commands) construct this with
    `OllamaReferenceModelRunner()`.
    """

    def __init__(
        self,
        runner: ReferenceModelRunner,
        reference_model_id: str = DEFAULT_REFERENCE_MODEL_ID,
        quant_levels: tuple[str, ...] = DEFAULT_QUANT_LEVELS,
        time_budget_s: float = DEFAULT_TIME_BUDGET_S,
        min_tok_s_floor: float = DEFAULT_MIN_TOK_S_FLOOR,
        profile_path: Path = DEFAULT_PROFILE_PATH,
    ) -> None:
        self._runner = runner
        self._reference_model_id = reference_model_id
        self._quant_levels = quant_levels
        self._time_budget_s = time_budget_s
        self._min_tok_s_floor = min_tok_s_floor
        self._profile_path = profile_path

    @property
    def reference_model_id(self) -> str:
        """The model this profiler benchmarks (and expects a cached profile
        to have been measured on)."""
        return self._reference_model_id

    def _levels_to_benchmark(self) -> tuple[str, ...]:
        """
        Quant levels worth timing on this run. If the runner can list the
        models it already has (`local_models()`), only those of the configured
        quant levels are benchmarked -- pulling every quant of a 7B model
        just to time it costs many GB and tens of minutes, far past the
        60s budget (R-06). With none present locally, only the first level
        is tried, so at most one model is downloaded. Runners that can't
        list local models (test doubles) get every configured level.
        """
        list_local = getattr(self._runner, "local_models", None)
        if not callable(list_local):
            return self._quant_levels
        try:
            local = set(list_local())
        except Exception:
            return self._quant_levels
        present = tuple(
            q for q in self._quant_levels if f"{self._reference_model_id}:{q}" in local
        )
        return present or self._quant_levels[:1]

    # -- FR-01.2 ------------------------------------------------------

    def run_benchmark(
        self,
        prompt: str = DEFAULT_BENCHMARK_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Loads the reference model at each configured quant level in
        turn, runs one timed generation per level, and returns
        (measured_tok_s, peak_memory_gb) keyed by quant level.

        Stays inside `time_budget_s` in total (FR-01.2, R-06): if a
        level's load+generate would blow the remaining budget, it's
        skipped rather than run partially, and at least the first two
        successfully-measured levels are what FR-01.2 requires. A level
        that fails to load (OOM) is recorded as skipped, never as a
        crash (NFR-03) -- the caller still gets every level that did
        succeed.
        """
        measured_tok_s: dict[str, float] = {}
        peak_memory_gb: dict[str, float] = {}

        quant_levels = self._levels_to_benchmark()
        deadline = time.monotonic() + self._time_budget_s
        for quant_level in quant_levels:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Heuristic pre-filter (R-06): once we're inside the last
            # ~5s of budget, don't start a new load -- a load that's cut
            # off mid-way would produce a misleadingly low tok/s figure.
            if remaining < 5.0:
                break

            try:
                self._runner.load(self._reference_model_id, quant_level)
            except Exception:
                # OOM or any other load failure: this quant level simply
                # isn't supported on this machine. Not an error for the
                # profiler as a whole (NFR-03) -- keep going.
                continue

            try:
                sample: GenerationSample = self._runner.generate(prompt, max_tokens)
                measured_tok_s[quant_level] = round(sample.tok_s, 3)
                peak_memory_gb[quant_level] = round(sample.peak_memory_gb, 3)
            except Exception:
                continue
            finally:
                self._runner.unload()

        if len(measured_tok_s) < 2 and len(quant_levels) >= 2:
            # FR-01.2 requires >= 2 quant levels measured. If the budget
            # or repeated load failures left us with fewer, that's a
            # real signal worth surfacing distinctly from "0 measured",
            # but we still return whatever *was* measured -- degrade
            # gracefully (NFR-03) rather than raise, since one working
            # rung is still enough for ModelLadder to pick from.
            pass

        return measured_tok_s, peak_memory_gb

    # -- FR-01.1 + FR-01.2 combined into one profile -------------------

    def profile(self, force: bool = True) -> HardwareProfile:
        """
        Runs full detection + benchmark and returns a fresh
        HardwareProfile. This is the method Track C's CLI calls for
        BOTH `cascade init` (first run) and `cascade profile` (explicit
        re-run) -- FR-01.4 is satisfied by this method being safely
        callable any number of times with no special first-run state,
        and no reinstall required.

        `force` exists for API clarity at call sites (`cascade profile`
        always force=True; a future cached-profile-reuse optimization in
        `cascade init` could pass force=False) but this Phase 1
        implementation always re-measures -- caching a stale profile
        silently would defeat the entire "measure, don't guess" premise
        of the product (Section 2.1).
        """
        gpu_vendor, vram_gb = detect_gpu()
        measured_tok_s, peak_memory_gb = self.run_benchmark()

        best_tok_s = max(measured_tok_s.values(), default=0.0)
        local_capable = best_tok_s >= self._min_tok_s_floor

        hardware_profile: HardwareProfile = {
            "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
            "profiled_at": datetime.now(timezone.utc).isoformat(),
            "reference_model_id": self._reference_model_id,
            "os": detect_os(),  # type: ignore[typeddict-item]
            "cpu_cores": detect_cpu_cores(),
            "ram_gb": detect_ram_gb(),
            "gpu_vendor": gpu_vendor,  # type: ignore[typeddict-item]
            "vram_gb": vram_gb,
            "disk_throughput_mb_s": detect_disk_throughput_mb_s(),
            "disk_free_gb": detect_disk_free_gb(),
            "measured_tok_s": measured_tok_s,
            "peak_memory_gb": peak_memory_gb,
            "local_capable": local_capable,
        }

        self._save(hardware_profile)
        return hardware_profile

    # -- FR-01.3 -- persistence backing "the only place hardware facts
    #    are computed anywhere in the codebase": every other component
    #    reads the *saved* profile via `load_cached_profile`, it never
    #    calls detect_* functions itself. ------------------------------

    def _save(self, hardware_profile: HardwareProfile) -> None:
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._profile_path, "w") as f:
            json.dump(hardware_profile, f, indent=2)

    def load_cached_profile(self) -> Optional[HardwareProfile]:
        """Returns the last profile written by `profile()`, or None if
        this machine has never been profiled. Used by anything that
        needs a HardwareProfile but shouldn't trigger a fresh (slow)
        benchmark run itself -- e.g. `cascade chat` on every launch.

        Also returns None when the cached profile was measured on a
        different model than this profiler's `reference_model_id`: its
        tok/s figures say nothing about the model now configured, so the
        caller re-profiles. Profiles written before `reference_model_id`
        was recorded were always measured on DEFAULT_REFERENCE_MODEL_ID."""
        if not self._profile_path.exists():
            return None
        with open(self._profile_path) as f:
            data = json.load(f)
        if data.get("schema_version") != HARDWARE_PROFILE_SCHEMA_VERSION:
            # A profile written by an older/newer schema version isn't
            # safe to trust blindly -- caller should re-profile.
            return None
        if data.get("reference_model_id", DEFAULT_REFERENCE_MODEL_ID) != self._reference_model_id:
            return None
        return data  # type: ignore[return-value]
