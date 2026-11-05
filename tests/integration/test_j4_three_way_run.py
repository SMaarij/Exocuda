"""
tests/integration/test_j4_three_way_run.py

The first REAL three-way end-to-end run (Section 13.1's Phase 4 join
point, J-4) -- closes Track B's Individual Task Breakdown Task 17,
which was blocked pending Track C's SelfHostedBackend
(docs/archive/TRACK_B_PHASE2_3_SUMMARY.md's "What's explicitly NOT done"
section). Now that gateway/backend.py exists, this test wires all
three tracks' real classes together:

    Track A -- engine.execution.backend.LocalExecutionBackend (real)
    Track B -- engine.intelligence.pipeline.Pipeline (real)
    Track C -- gateway.backend.SelfHostedBackend (real)

Only the innermost dependencies each track already fakes for its own
unit tests are faked here too (a deterministic ReferenceModelRunner for
Track A, a deterministic CloudInferenceProvider for Track C) -- no live
Ollama, GPU, or network call anywhere (NFR-09). This is the ceiling of
what's testable without an actual deployed model/endpoint; a true
production smoke test still needs docs/archive/TRACK_B_HANDOFF_FOR_C.md
Section 4 Step 3's manual verification against real infrastructure.
"""

from __future__ import annotations

from pathlib import Path

from engine.execution.backend import LocalExecutionBackend
from engine.execution.hardware_profiler import HardwareProfiler
from engine.execution.model_runner import GenerationSample
from engine.intelligence.pipeline import Pipeline
from engine.intelligence.router.classifier import SemanticRouterClassifier
from engine.intelligence.router.coldstart_data import RoutingExample
from engine.intelligence.tests.fakes import FakeEmbedder
from gateway.backend import SelfHostedBackend
from gateway.tests.fakes import FakeCloudInferenceProvider
from shared.types import Message


class _FakeRunner:
    """Same deterministic double pattern Track A's own j2_integration_test.py uses."""

    def __init__(self, tok_s: float = 20.0, fail: bool = False) -> None:
        self._tok_s = tok_s
        self._fail = fail
        self._loaded: str | None = None

    def load(self, model_id: str, quant_level: str) -> None:
        if self._fail:
            raise RuntimeError("simulated OOM")
        self._loaded = quant_level

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        assert self._loaded is not None
        return GenerationSample(tokens_generated=max_tokens, elapsed_s=max_tokens / self._tok_s, peak_memory_gb=2.0)

    def unload(self) -> None:
        self._loaded = None


def _local_backend(tmp_path: Path, capable: bool) -> LocalExecutionBackend:
    runner = _FakeRunner(tok_s=20.0, fail=not capable)
    profiler = HardwareProfiler(
        runner=runner, quant_levels=("q4_K_M", "q5_K_M"), profile_path=tmp_path / "profile.json"
    )
    profiler.profile()
    return LocalExecutionBackend(runner=runner, profiler=profiler)


def _msgs(text: str) -> list[Message]:
    return [{"role": "user", "content": text}]


def test_j4_local_capable_machine_stays_on_local_tier(tmp_path: Path) -> None:
    """local_capable=True -> Pipeline never touches the self-hosted server."""
    local = _local_backend(tmp_path, capable=True)
    cascade_provider = FakeCloudInferenceProvider(text="should never be seen")
    cascade = SelfHostedBackend(provider=cascade_provider)

    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, confidence_floor=None)
    result = pipeline.run(_msgs("hello"))

    assert result["tier"] == "local"
    assert result["escalated"] is False
    assert cascade_provider.call_count == 0


def test_j4_local_incapable_machine_escalates_to_self_hosted(tmp_path: Path) -> None:
    """
    All local rungs OOM (LocalExecutionBackendError) -> Pipeline
    escalates to Track C's real SelfHostedBackend -- the exact chain the master doc's
    Section 4 architecture diagram describes: local_capable=False
    (or, here, a runtime OOM) routes to the self-hosted tier, FR-07.
    """
    local = _local_backend(tmp_path, capable=False)  # every rung OOMs
    cascade_provider = FakeCloudInferenceProvider(text="answered by the self-hosted server", prompt_tokens=10, completion_tokens=20)
    cascade = SelfHostedBackend(provider=cascade_provider)

    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, confidence_floor=None)
    result = pipeline.run(_msgs("hello"))

    assert result["tier"] == "self_hosted"
    assert result["escalated"] is True
    assert result["text"] == "answered by the self-hosted server"
    assert result["est_cost_usd"] == 0.0  # your own server: no per-token charge
    assert cascade_provider.call_count == 1


def test_j4_router_sends_tool_required_query_straight_to_self_hosted(tmp_path: Path) -> None:
    """
    SemanticRouter (Track B) classifies a tool-required query and sends
    it directly to the self-hosted server (Track C) without ever attempting local
    (Track A) execution at all -- FR-05.2's routing decision, not an
    error-driven fallback.
    """
    local = _local_backend(tmp_path, capable=True)
    cascade_provider = FakeCloudInferenceProvider(text="today's date is fetched live")
    cascade = SelfHostedBackend(provider=cascade_provider)

    router = SemanticRouterClassifier(FakeEmbedder())
    router.fit([
        RoutingExample("hello there friend", "local"),
        RoutingExample("hi good morning", "local"),
        RoutingExample("what is today's date now", "self_hosted"),
        RoutingExample("search the current news", "self_hosted"),
    ])

    pipeline = Pipeline(backend=local, self_hosted_backend=cascade, router=router, confidence_floor=None)
    result = pipeline.run(_msgs("what is today's date now"))

    assert result["tier"] == "self_hosted"
    assert cascade_provider.call_count == 1

