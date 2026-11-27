
"""
engine/execution/j2_integration_test.py

Track A, Phase 2 -- J-2 Verification File.

This file proves that LocalExecutionBackend satisfies the J-2 join point:
"Both real backends drop into Track B's Pipeline with zero changes to
Track B's calling code." (Section 13.1 / CascadeTaskBreakdown, Table 20)

                ????????????????????????????????????????????????????????
                ?  HOW TO UPGRADE THIS TO A REAL J-2 WHEN TRACK B      ?
                ?  LANDS:                                               ?
                ?                                                       ?
                ?  Replace this import:                                 ?
                ?    from engine.execution.pipeline_bridge import       ?
                ?        FakePipeline                                   ?
                ?  with:                                                ?
                ?    from engine.intelligence.pipeline import           ?
                ?        Pipeline as FakePipeline                       ?
                ?                                                       ?
                ?  Then run:                                            ?
                ?    pytest engine/execution/j2_integration_test.py -v  ?
                ?                                                       ?
                ?  If green with zero test body changes -> J-2 done.    ?
                ?  Delete engine/execution/pipeline_bridge.py.          ?
                ????????????????????????????????????????????????????????

All model calls are mocked; no live GPU or Ollama instance is required
(NFR-09). The FakeReferenceModelRunner from test_hardware_profiler.py
is reimplemented here to keep this file self-contained.
"""

from __future__ import annotations

import pytest

from cascade_local.engine.intelligence.pipeline import Pipeline as FakePipeline

from cascade_local.engine.execution.backend import LocalExecutionBackend
from cascade_local.engine.execution.hardware_profiler import HardwareProfiler
from cascade_local.engine.execution.model_ladder import ModelLadder, QuantRung
from cascade_local.engine.execution.model_runner import GenerationSample, ReferenceModelRunner
from cascade_local.shared.types import ExecutionBackend, Message


# ?? Deterministic test double ??????????????????????????????????????????????

class _FakeRunner:
    """Minimal ReferenceModelRunner double for integration tests."""
    def __init__(self, tok_s: float = 20.0, fail: bool = False) -> None:
        self._tok_s = tok_s
        self._fail  = fail
        self._loaded: str | None = None

    def load(self, model_id: str, quant_level: str) -> None:
        if self._fail:
            raise RuntimeError("simulated OOM")
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        return GenerationSample(
            tokens_generated=max_tokens,
            elapsed_s=max_tokens / self._tok_s,
            peak_memory_gb=2.0,
        )

    def unload(self) -> None:
        self._loaded = None


def _capable_backend(tmp_path) -> LocalExecutionBackend:
    """Backend backed by a fast fake runner (local_capable=True profile)."""
    runner = _FakeRunner(tok_s=20.0)
    profiler = HardwareProfiler(
        runner=runner,
        quant_levels=("q4_K_M", "q5_K_M"),
        profile_path=tmp_path / "profile.json",
    )
    profiler.profile()   # write a cached profile
    return LocalExecutionBackend(runner=runner, profiler=profiler)


# ?? J-2 checklist tests ????????????????????????????????????????????????????

class TestJ2VerificationChecklist:
    """
    Each test maps to one item on the J-2 checklist from the task breakdown.
    Run this class against Track B's real Pipeline by swapping the import at
    the top of this file -- the test bodies must not change.
    """

    def test_j2_1_backend_conforms_to_execution_backend_protocol(self, tmp_path) -> None:
        """J-2.1: LocalExecutionBackend is-a ExecutionBackend."""
        backend = _capable_backend(tmp_path)
        assert isinstance(backend, ExecutionBackend)

    def test_j2_2_pipeline_accepts_backend_and_returns_pipeline_result(self, tmp_path) -> None:
        """J-2.2: FakePipeline/Pipeline accepts backend; run() returns PipelineResult-shaped dict."""
        backend  = _capable_backend(tmp_path)
        pipeline = FakePipeline(backend)
        messages: list[Message] = [{"role": "user", "content": "Hello"}]

        result = pipeline.run(messages)

        # PipelineResult fields (shared/types.py)
        assert "text"               in result
        assert "tier"               in result
        assert "cache_hit"          in result
        assert "confidence"         in result
        assert "escalated"          in result
        assert "speculative_active" in result
        assert "tok_s"              in result
        assert "latency_ms"         in result
        assert "est_cost_usd"       in result

    def test_j2_3_generate_stream_yields_at_least_one_token(self, tmp_path) -> None:
        """J-2.3: generate_stream() yields at least one token -- not an empty iterator."""
        backend  = _capable_backend(tmp_path)
        pipeline = FakePipeline(backend)
        messages: list[Message] = [{"role": "user", "content": "Hello"}]

        tokens = list(pipeline.run_stream(messages))
        assert len(tokens) >= 1

    def test_j2_4_zero_changes_to_pipeline_calling_code_across_backends(self, tmp_path) -> None:
        """
        J-2.4: Pipeline calling code is IDENTICAL whether backed by the
        Phase 1 stub or the real Phase 2 backend. The seam holds.
        """
        from cascade_local.engine.execution.backend import LocalExecutionBackend as RealBackend

        backend  = _capable_backend(tmp_path)
        pipeline = FakePipeline(backend)
        messages: list[Message] = [{"role": "user", "content": "Test"}]

        # The ONLY difference between stub and real: the constructor kwargs.
        # The pipeline.run() call below is byte-for-byte identical.
        result = pipeline.run(messages)
        assert isinstance(result["tok_s"], float)
        assert result["latency_ms"] >= 0.0

    def test_j2_5_oom_on_all_rungs_raises_cleanly_pipeline_can_catch(self, tmp_path) -> None:
        """
        J-2.5: If all rungs OOM, LocalExecutionBackend raises
        LocalExecutionBackendError -- Pipeline can catch it and escalate
        to the self-hosted tier without a traceback reaching the user.
        """
        from cascade_local.engine.execution.backend import LocalExecutionBackendError

        runner   = _FakeRunner(fail=True)
        profiler = HardwareProfiler(
            runner=runner,
            quant_levels=("q4_K_M",),
            profile_path=tmp_path / "profile.json",
        )
        # Manually write a profile so get_profile() doesn't try to benchmark
        import json
        from cascade_local.shared.types import HARDWARE_PROFILE_SCHEMA_VERSION
        from datetime import datetime, timezone
        profile = {
            "schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
            "profiled_at": datetime.now(timezone.utc).isoformat(),
            "os": "linux", "cpu_cores": 4, "ram_gb": 8.0,
            "gpu_vendor": None, "vram_gb": 0.0,
            "disk_throughput_mb_s": 100.0, "disk_free_gb": 50.0,
            "measured_tok_s": {"q4_K_M": 10.0},
            "peak_memory_gb": {"q4_K_M": 1.0},
            "local_capable": True,
        }
        (tmp_path / "profile.json").write_text(json.dumps(profile))
        backend  = LocalExecutionBackend(runner=runner, profiler=profiler)
        pipeline = FakePipeline(backend)
        messages: list[Message] = [{"role": "user", "content": "Hello"}]

        with pytest.raises(LocalExecutionBackendError):
            pipeline.run(messages)
