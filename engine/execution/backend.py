
"""
engine/execution/backend.py

Track A, Phase 2 (Weeks 3-5) -- Task 11.

Real LocalExecutionBackend replacing the Phase 1 stub. Wires together:
    - HardwareProfiler  (Phase 1, Task 3) -- loads the cached HardwareProfile
    - ModelLadder       (Phase 2, Task 5) -- selects the best QuantRung
    - AccelerationEngine (Phase 2, Tasks 7-10) -- wraps the runner with
                          the appropriate speculative layer

The public API (generate / generate_stream) is IDENTICAL to the Phase 1
stub -- Track B's existing tests pass with zero changes. This is the
definition of the J-2 seam holding (Section 13.1 Phase 2 join point).

                    ??????????????????????????????????????
                    ?? J-2 CONNECT HERE ??
                    When Track B's real Pipeline lands in
                    engine/intelligence/pipeline.py, it
                    passes LocalExecutionBackend() to
                    Pipeline.__init__() and calls
                    .generate() / .generate_stream() via
                    the shared ExecutionBackend protocol.
                    Verification: run
                      pytest engine/execution/j2_integration_test.py
                    against Track B's real Pipeline.
                    No changes needed in this file.
                    ??????????????????????????????????????

                    Until Track B's Pipeline exists, use
                    engine/execution/pipeline_bridge.py
                    (FakePipeline) for end-to-end testing.
                    See that file for removal instructions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

from engine.execution.acceleration_engine import (
    AccelerationEngine,
    AccelerationLayer,
    DraftPair,
)
from engine.execution.hardware_profiler import HardwareProfiler
from engine.execution.model_ladder import ModelLadder, QuantRung
from engine.execution.model_runner import OllamaReferenceModelRunner, ReferenceModelRunner
from shared.types import BackendResult, ExecutionBackend, HardwareProfile, Message

# ?? Sentinel for missing Track-B Pipeline ????????????????????????????????
# Raised by run_via_pipeline() in pipeline_bridge.py; NOT raised by
# generate() / generate_stream() (those work without Track B).
TRACK_B_NOT_PRESENT_MSG = (
    "Track B's real Pipeline (engine/intelligence/pipeline.py) is not yet "
    "available. Use engine/execution/pipeline_bridge.FakePipeline for "
    "end-to-end testing until Track B lands."
)


class LocalExecutionBackendError(RuntimeError):
    """Raised when every QuantRung has been exhausted (OOM on all levels)."""


def _messages_to_prompt(messages: list[Message]) -> str:
    """Flatten a message list into a simple role: content prompt string."""
    return "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )


# ModelLadder's QuantRung.label is deliberately abstract ("q4_K_M", "q5_K_M",
# "q8_0" -- see model_ladder.py's DEFAULT_LADDER) because select_rung() only
# ever compares measured_tok_s/vram against those labels, never matches a
# real model's tag by string. But a real Ollama tag is `repository:tag`
# where `tag` is a single dash-joined string (e.g.
# "llama3.2:3b-instruct-q4_K_M"), not "llama3.2:q4_K_M" -- that tag doesn't
# exist. This maps (model_id, abstract rung label) -> the real tag suffix
# for models Cascade ships a mapping for, keeping ModelLadder itself generic
# and untouched.
_RUNG_LABEL_TO_REAL_TAG: dict[str, dict[str, str]] = {
    "llama3.2": {
        "q4_K_M": "3b-instruct-q4_K_M",
        "q5_K_M": "3b-instruct-q5_K_M",
        "q8_0": "3b-instruct-q8_0",
    },
    # 7B target that pairs with qwen2.5-coder:0.5b as a draft model
    # (same tokenizer); see BENCHMARK_REPORT Addendum 6.
    "qwen2.5-coder": {
        "q4_K_M": "7b-instruct-q4_K_M",
        "q5_K_M": "7b-instruct-q5_K_M",
        "q8_0": "7b-instruct-q8_0",
    },
}


def _runner_model(model_id: str) -> str:
    """The family name the runner loads: "qwen2.5" for "qwen2.5:1.5b-instruct"."""
    return model_id.split(":", 1)[0]


def _resolve_real_tag(model_id: str, rung_label: str) -> str:
    """
    A model_id with an explicit size ("qwen2.5:1.5b-instruct") resolves to
    "<size>-<rung>" ("1.5b-instruct-q4_K_M"), loaded under _runner_model().

    Translate an abstract ModelLadder rung label into the real Ollama tag
    suffix for `model_id`. Falls back to the bare label unresolved if
    `model_id` has no registered mapping (e.g. a test double's synthetic
    labels like "floor") -- this fails loudly and honestly at the `ollama
    pull` step for an unmapped production model, rather than silently
    guessing a tag that might not exist.
    """
    if ":" in model_id:
        return f"{model_id.split(':', 1)[1]}-{rung_label}"
    return _RUNG_LABEL_TO_REAL_TAG.get(model_id, {}).get(rung_label, rung_label)


def profile_quant_levels(model_id: str, ladder: Optional[ModelLadder] = None) -> tuple[str, ...]:
    """The real tag suffixes HardwareProfiler should benchmark for `model_id`:
    one per ModelLadder rung, lowest rung first, resolved via _resolve_real_tag()."""
    rungs = (ladder or ModelLadder()).rungs
    return tuple(_resolve_real_tag(model_id, rung.label) for rung in rungs)


def build_profiler(
    runner: ReferenceModelRunner,
    model_id: str,
    ladder: Optional[ModelLadder] = None,
    **profiler_kwargs,
) -> HardwareProfiler:
    """A HardwareProfiler that measures (and caches a profile for) `model_id`.
    Extra keyword arguments (e.g. profile_path) go to HardwareProfiler."""
    return HardwareProfiler(
        runner=runner,
        reference_model_id=model_id,
        quant_levels=profile_quant_levels(model_id, ladder),
        **profiler_kwargs,
    )


class LocalExecutionBackend:
    """
    Real local execution backend. Conforms to shared.types.ExecutionBackend.

    Dependency-injection constructor (for tests and Track B):
        runner    -- the model runner; defaults to OllamaReferenceModelRunner.
        profiler  -- defaults to build_profiler(runner, model_id, ladder),
                     which benchmarks model_id itself; injectable for
                     tests that need a controlled HardwareProfile.
        ladder    -- defaults to ModelLadder(); injectable for tests.
        preferred_layer -- the highest AccelerationLayer to attempt;
                     defaults to LOOKAHEAD (always available, FR-03.1).
        model_id  -- the model tag to load; defaults to "llama3.2", the
                     real reference model used for benchmarking (Section
                     5.1). Abstract ModelLadder rung labels are resolved
                     to real per-model Ollama tag suffixes via
                     _resolve_real_tag() before being passed to the
                     runner -- see that function's docstring.
        model_has_draft_heads -- True if the loaded model carries Medusa /
                     EAGLE draft heads (Layer 2); False by default.
        validated_pairs -- override for Layer 3 pair list; defaults to
                     loading from DEFAULT_PAIRS_PATH.
        simulated_latency_ms -- non-zero only in tests that want realistic
                     timing without a real model.
    """

    def __init__(
        self,
        runner: Optional[ReferenceModelRunner] = None,
        profiler: Optional[HardwareProfiler] = None,
        ladder: Optional[ModelLadder] = None,
        # Off by default: measured slower than plain decoding on CPU-only
        # hardware (docs/benchmark/acceleration_results.json). Opt in per machine.
        preferred_layer: AccelerationLayer = AccelerationLayer.NONE,
        model_id: str = "llama3.2",
        model_has_draft_heads: bool = False,
        validated_pairs: Optional[list[DraftPair]] = None,
        simulated_latency_ms: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        # Use OllamaReferenceModelRunner only as a real production default;
        # tests always inject a FakeReferenceModelRunner (NFR-09).
        self._runner: ReferenceModelRunner = runner or OllamaReferenceModelRunner()
        self._ladder = ladder or ModelLadder()
        self._model_id = model_id
        # Profile the model this backend actually serves, not the llama3.2
        # reference default: its tok/s is what rung selection needs, and a
        # cached profile from another model is re-measured (see
        # HardwareProfiler.load_cached_profile).
        self._profiler = profiler or build_profiler(self._runner, model_id, self._ladder)
        self._engine = AccelerationEngine(
            runner=self._runner,
            preferred_layer=preferred_layer,
            model_has_draft_heads=model_has_draft_heads,
            validated_pairs=validated_pairs or [],
        )
        self._simulated_latency_ms = simulated_latency_ms
        self._max_tokens = max_tokens
        self._loaded: Optional[tuple[str, str]] = None

    @property
    def profiler(self) -> HardwareProfiler:
        return self._profiler

    # ---- ExecutionBackend protocol implementation ----------------------

    def generate(self, messages: list[Message], on_text=None) -> BackendResult:
        """
        Run a full (non-streaming) generation.

        Loads the best available rung, applies AccelerationEngine, and
        automatically steps down on OOM until either a rung succeeds or
        all rungs are exhausted (raises LocalExecutionBackendError).
        """
        start = time.monotonic()
        if self._simulated_latency_ms > 0:
            time.sleep(self._simulated_latency_ms / 1000)

        profile = self._get_profile()
        rung    = self._ladder.select_rung(profile)
        # A runner that formats chat turns itself (LlamaCppRunner) gets the messages, so
        # each keeps its role in the model's own template; others get one flat string.
        prompt  = messages if getattr(self._runner, "accepts_messages", False) is True else _messages_to_prompt(messages)

        sample, used_rung = self._generate_with_stepdown(prompt, rung, on_text)
        wall_ms = (time.monotonic() - start) * 1000

        # Fallback only matters for test doubles that never set output_text.
        text = sample.base.output_text or (sample.base.tokens_generated * "x")

        result: BackendResult = {
            "text":               text,
            "confidence":         sample.acceptance_rate,
            "speculative_active": sample.speculative_active,
            "tok_s":              round(sample.tok_s, 3),
            "latency_ms":         round(wall_ms, 2),   # real wall-clock time
            "est_cost_usd":       0.0,                 # local is always free
        }
        logprobs = getattr(self._runner, "last_token_logprobs", None)
        if isinstance(logprobs, list) and logprobs:
            result["logprobs"] = logprobs
        return result

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        """
        Streaming variant. Yields tokens one by one (word-level here;
        real llama.cpp / Ollama streaming is byte-level -- that detail
        lives in OllamaReferenceModelRunner, not here).
        """
        result = self.generate(messages)
        # Yield word by word to simulate streaming from the caller's view.
        for word in result["text"].split():
            if self._simulated_latency_ms > 0:
                time.sleep(self._simulated_latency_ms / 1000 / 10)
            yield word + " "

    def release(self) -> None:
        """Unload the model now (the runner may save its state first); the next request reloads it."""
        if self._loaded is not None:
            self._engine.unload()
            self._loaded = None

    # ---- Internal helpers -----------------------------------------------

    def _get_profile(self) -> HardwareProfile:
        """
        Return the cached HardwareProfile if present, schema-compatible and
        measured on the profiler's model, otherwise run a fresh profile
        (cascade profile flow, FR-01.4).
        """
        cached = self._profiler.load_cached_profile()
        if cached is not None:
            return cached
        return self._profiler.profile()

    def _generate_with_stepdown(
        self,
        prompt: "str | list[Message]",
        rung: Optional[QuantRung],
        on_text=None,
    ) -> tuple:
        """
        Attempt generation at rung; on OOM step down and retry (FR-02.3).
        Raises LocalExecutionBackendError if all rungs are exhausted.
        """
        last_error: Optional[Exception] = None
        while rung is not None:
            real_tag = _resolve_real_tag(self._model_id, rung.label)
            try:
                # Keep the model resident between requests; reloading per
                # request costs seconds of cold-load each time.
                if self._loaded != (self._model_id, real_tag):
                    self._engine.load(_runner_model(self._model_id), real_tag)
                    self._loaded = (self._model_id, real_tag)
                kwargs = {"on_text": on_text} if on_text is not None else {}
                sample = self._engine.generate_accelerated(prompt, max_tokens=self._max_tokens, **kwargs)
                return sample, rung
            except Exception as exc:
                # OOM or any load failure: try the next lower rung (FR-02.3)
                last_error = exc
                self._engine.unload()
                self._loaded = None
                rung = self._ladder.step_down(rung)

        detail = f" Last error: {last_error}" if last_error else " No rung is viable for this machine's HardwareProfile."
        raise LocalExecutionBackendError(
            "Local execution failed on every QuantRung." + detail
        )


# Fails immediately at import time if the class ever drifts from the
# ExecutionBackend protocol -- same guard as the Phase 1 stub.
assert isinstance(LocalExecutionBackend(), ExecutionBackend)
