"""
The hardware profile measures the configured model, not always llama3.2.

Before this, LocalExecutionBackend(model_id="qwen2.5-coder") built a default
HardwareProfiler that benchmarked llama3.2:3b-instruct-* (and pulled them on
first run), and a cached llama3.2 profile was reused for any model. These
tests pin the fix: the profiler benchmarks the backend's model at its
_resolve_real_tag() tags, records which model it measured, and a profile for
another model counts as stale. No Ollama, network or real model (NFR-09).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.execution.backend import (
    LocalExecutionBackend,
    _resolve_real_tag,
    build_profiler,
    profile_quant_levels,
)
from engine.execution.hardware_profiler import DEFAULT_REFERENCE_MODEL_ID, HardwareProfiler
from engine.execution.model_ladder import ModelLadder
from engine.execution.tests.test_hardware_profiler import (  # noqa: F401  (fixture)
    FakeReferenceModelRunner,
    patched_detectors,
)
from engine.execution.tests.test_llamacpp_runner import FakeOllama
from shared.types import HARDWARE_PROFILE_SCHEMA_VERSION

QWEN = "qwen2.5-coder"
QWEN_TAGS = ("7b-instruct-q4_K_M", "7b-instruct-q5_K_M", "7b-instruct-q8_0")
LLAMA_TAGS = ("3b-instruct-q4_K_M", "3b-instruct-q5_K_M", "3b-instruct-q8_0")


class LocalListingRunner(FakeReferenceModelRunner):
    """Fake runner that, like Ollama / LlamaCppRunner, can list local tags."""

    def __init__(self, tok_s_by_level: dict[str, float], local: list[str] | Exception) -> None:
        super().__init__(tok_s_by_level)
        self._local = local

    def local_models(self) -> list[str]:
        if isinstance(self._local, Exception):
            raise self._local
        return list(self._local)


def _qwen_runner() -> FakeReferenceModelRunner:
    return FakeReferenceModelRunner({QWEN_TAGS[0]: 6.0, QWEN_TAGS[1]: 5.0, QWEN_TAGS[2]: 3.5})


def _legacy_profile(**overrides) -> dict:
    """A profile as written before reference_model_id existed (llama3.2-measured)."""
    profile = {
        "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "os": "linux", "cpu_cores": 4, "ram_gb": 16.0,
        "gpu_vendor": None, "vram_gb": 0.0,
        "disk_throughput_mb_s": 500.0, "disk_free_gb": 100.0,
        "measured_tok_s": {LLAMA_TAGS[0]: 12.0, LLAMA_TAGS[1]: 10.0},
        "peak_memory_gb": {LLAMA_TAGS[0]: 2.0, LLAMA_TAGS[1]: 2.5},
        "local_capable": True,
    }
    profile.update(overrides)
    return profile


# ---- tag resolution ----------------------------------------------------------


def test_profile_quant_levels_resolve_ladder_rungs_to_real_tags() -> None:
    assert profile_quant_levels(QWEN) == QWEN_TAGS
    assert profile_quant_levels("llama3.2") == LLAMA_TAGS
    ladder = ModelLadder()
    assert profile_quant_levels(QWEN, ladder) == tuple(
        _resolve_real_tag(QWEN, rung.label) for rung in ladder.rungs
    )


def test_profile_quant_levels_fall_back_to_bare_labels_for_unmapped_models() -> None:
    assert profile_quant_levels("some-model") == ("q4_K_M", "q5_K_M", "q8_0")


# ---- profiler measures and records the configured model -----------------------


def test_profiler_benchmarks_the_configured_model_not_llama(tmp_path: Path, patched_detectors) -> None:
    runner = _qwen_runner()
    profile = build_profiler(runner, QWEN, profile_path=tmp_path / "p.json").profile()

    assert runner.load_calls == [(QWEN, tag) for tag in QWEN_TAGS]
    assert set(profile["measured_tok_s"]) == set(QWEN_TAGS)
    assert profile["reference_model_id"] == QWEN


def test_saved_profile_records_the_measured_model(tmp_path: Path, patched_detectors) -> None:
    path = tmp_path / "p.json"
    build_profiler(_qwen_runner(), QWEN, profile_path=path).profile()
    assert json.loads(path.read_text())["reference_model_id"] == QWEN


def test_default_profiler_still_records_llama(tmp_path: Path, patched_detectors) -> None:
    runner = FakeReferenceModelRunner({tag: 10.0 for tag in LLAMA_TAGS})
    profile = HardwareProfiler(runner, profile_path=tmp_path / "p.json").profile()
    assert profile["reference_model_id"] == DEFAULT_REFERENCE_MODEL_ID


# ---- cache validity follows the model ---------------------------------------------


def test_cached_profile_for_same_model_is_reused(tmp_path: Path, patched_detectors) -> None:
    path = tmp_path / "p.json"
    written = build_profiler(_qwen_runner(), QWEN, profile_path=path).profile()
    assert build_profiler(_qwen_runner(), QWEN, profile_path=path).load_cached_profile() == written


def test_cached_profile_for_another_model_is_stale(tmp_path: Path, patched_detectors) -> None:
    path = tmp_path / "p.json"
    build_profiler(_qwen_runner(), QWEN, profile_path=path).profile()
    llama = build_profiler(FakeReferenceModelRunner({}), "llama3.2", profile_path=path)
    assert llama.load_cached_profile() is None


def test_legacy_profile_without_model_counts_as_llama(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_legacy_profile()))

    assert build_profiler(FakeReferenceModelRunner({}), "llama3.2", profile_path=path).load_cached_profile() is not None
    assert build_profiler(FakeReferenceModelRunner({}), QWEN, profile_path=path).load_cached_profile() is None


# ---- the backend wiring -----------------------------------------------------------


def test_backend_default_profiler_targets_its_model() -> None:
    backend = LocalExecutionBackend(runner=_qwen_runner(), model_id=QWEN)
    assert backend._profiler.reference_model_id == QWEN
    assert backend._profiler._quant_levels == QWEN_TAGS


def test_backend_reprofiles_its_model_when_cache_is_for_llama(tmp_path: Path, patched_detectors) -> None:
    """The reported bug: a qwen2.5-coder backend with a llama3.2 profile on
    disk must re-measure qwen2.5-coder, not trust (or re-measure) llama3.2."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_legacy_profile()))
    runner = _qwen_runner()
    backend = LocalExecutionBackend(
        runner=runner, model_id=QWEN, profiler=build_profiler(runner, QWEN, profile_path=path)
    )

    backend.generate([{"role": "user", "content": "hi"}])

    loaded_models = {model for model, _ in runner.load_calls}
    assert loaded_models == {QWEN}
    saved = json.loads(path.read_text())
    assert saved["reference_model_id"] == QWEN
    assert set(saved["measured_tok_s"]) == set(QWEN_TAGS)


def test_backend_does_not_reprofile_when_cache_matches(tmp_path: Path, patched_detectors) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_legacy_profile(
        reference_model_id=QWEN,
        measured_tok_s={QWEN_TAGS[0]: 6.0}, peak_memory_gb={QWEN_TAGS[0]: 5.0},
    )))
    runner = _qwen_runner()
    backend = LocalExecutionBackend(
        runner=runner, model_id=QWEN, profiler=build_profiler(runner, QWEN, profile_path=path)
    )

    backend.generate([{"role": "user", "content": "hi"}])

    # Only the serving load, no benchmark loop over every rung.
    assert runner.load_calls == [(QWEN, QWEN_TAGS[0])]


# ---- prefer already-downloaded tags (no multi-GB pulls to profile) ----------------


def test_profiler_only_benchmarks_locally_present_tags(tmp_path: Path) -> None:
    runner = LocalListingRunner(
        {tag: 5.0 for tag in QWEN_TAGS},
        local=[f"{QWEN}:{QWEN_TAGS[0]}", f"{QWEN}:{QWEN_TAGS[2]}", "llama3.2:3b-instruct-q4_K_M"],
    )
    measured, _ = build_profiler(runner, QWEN, profile_path=tmp_path / "p.json").run_benchmark()

    assert runner.load_calls == [(QWEN, QWEN_TAGS[0]), (QWEN, QWEN_TAGS[2])]
    assert set(measured) == {QWEN_TAGS[0], QWEN_TAGS[2]}


def test_profiler_pulls_only_the_first_tag_when_none_are_local(tmp_path: Path) -> None:
    runner = LocalListingRunner({tag: 5.0 for tag in QWEN_TAGS}, local=["llama3.2:3b-instruct-q4_K_M"])
    build_profiler(runner, QWEN, profile_path=tmp_path / "p.json").run_benchmark()
    assert runner.load_calls == [(QWEN, QWEN_TAGS[0])]


def test_profiler_tries_every_tag_when_local_listing_fails(tmp_path: Path) -> None:
    runner = LocalListingRunner({tag: 5.0 for tag in QWEN_TAGS}, local=ConnectionError("down"))
    build_profiler(runner, QWEN, profile_path=tmp_path / "p.json").run_benchmark()
    assert runner.load_calls == [(QWEN, tag) for tag in QWEN_TAGS]


def test_llamacpp_runner_lists_local_models_from_ollama() -> None:
    from engine.execution.llamacpp_runner import LlamaCppRunner

    ollama = FakeOllama()
    runner = LlamaCppRunner(ollama=ollama, llama_factory=lambda path, all_logits: None)
    assert runner.local_models() == ollama.local_models()
    assert ollama.pulled == []
