"""
engine/execution/tests/test_llamacpp_runner.py

LlamaCppRunner's mode switching and acceptance accounting, with an
injected fake Llama (NFR-09: no model, no llama.cpp call).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.execution.acceleration_engine import AccelerationEngine, AccelerationLayer, DraftPair
from engine.execution.llamacpp_runner import DraftCounter, LlamaCppRunner, ModelDraft
from engine.execution.model_runner import ACCEL_DRAFT_MODEL, ACCEL_NONE, ACCEL_PROMPT_LOOKUP, AcceleratedRunner


class FakeOllama:
    def __init__(self) -> None:
        self.pulled: list[str] = []

    def local_models(self) -> list[str]:
        return ["llama3.2:3b-instruct-q4_K_M", "llama3.2:1b-instruct-q4_K_M"]

    def _request(self, path, payload=None, timeout=10.0):
        self.pulled.append(payload["model"])
        return {}

    def model_file_path(self, tag: str) -> str:
        return f"/blobs/{tag}"


class FakeLlama:
    """Streams `text` word by word; calls the installed draft model `draft_calls` times."""

    def __init__(self, path: str, all_logits: bool, text: str = "one two three four five", draft_calls: int = 2) -> None:
        self.path = path
        self.all_logits = all_logits
        self.draft_model = None
        self._text = text
        self._draft_calls = draft_calls
        self.last_kwargs: dict = {}

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        if self.draft_model is not None:
            for _ in range(self._draft_calls):
                self.draft_model(np.array([1, 2, 3], dtype=np.intc))
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        for word in self._text.split(" "):
            yield {"choices": [{"delta": {"content": word + " "}}]}

    def generate(self, tokens, top_k, temp, reset):
        yield from range(10, 20)

    def tokenize(self, data: bytes, add_bos: bool = True):
        return data.decode().split()


def _runner(**kwargs):
    built: list[FakeLlama] = []

    def factory(path: str, all_logits: bool) -> FakeLlama:
        built.append(FakeLlama(path, all_logits))
        return built[-1]

    runner = LlamaCppRunner(ollama=FakeOllama(), llama_factory=factory, **kwargs)
    return runner, built


def test_is_an_accelerated_runner() -> None:
    assert isinstance(_runner()[0], AcceleratedRunner)


def test_plain_mode_loads_without_all_logits_and_reports_none() -> None:
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    sample = runner.generate("hi", 16)
    assert built[0].path == "/blobs/llama3.2:3b-instruct-q4_K_M"
    assert built[0].all_logits is False
    assert (sample.acceleration, sample.acceptance_rate) == (ACCEL_NONE, None)
    assert sample.output_text == "one two three four five"
    assert sample.tokens_generated == 4  # tokens after the first; decode time starts at token one


def test_switching_to_speculation_after_load_rebuilds_with_all_logits() -> None:
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert runner.set_acceleration(ACCEL_PROMPT_LOOKUP) is True
    assert len(built) == 2 and built[1].all_logits is True
    assert runner.generate("hi", 16).acceleration == ACCEL_PROMPT_LOOKUP


def test_draft_model_mode_loads_the_draft_once() -> None:
    runner, built = _runner()
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    paths = [b.path for b in built]
    assert paths == ["/blobs/llama3.2:3b-instruct-q4_K_M", "/blobs/llama3.2:1b-instruct-q4_K_M"]
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    assert len(built) == 2


def test_draft_model_mode_unavailable_without_a_draft() -> None:
    runner, _ = _runner(draft_model_tag=None)
    assert runner.set_acceleration(ACCEL_DRAFT_MODEL) is False
    assert runner.set_acceleration("draft_heads") is False


def test_temperature_is_forwarded_only_when_set() -> None:
    runner, built = _runner(temperature=0.0)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.generate("hi", 8)
    assert built[0].last_kwargs["temperature"] == 0.0
    runner2, built2 = _runner()
    runner2.load("llama3.2", "3b-instruct-q4_K_M")
    runner2.generate("hi", 8)
    assert "temperature" not in built2[0].last_kwargs


def test_generate_before_load_raises() -> None:
    with pytest.raises(RuntimeError):
        _runner()[0].generate("hi", 8)


def test_engine_drives_the_runner_end_to_end() -> None:
    runner, _ = _runner()
    engine = AccelerationEngine(runner, AccelerationLayer.DRAFT_MODEL, validated_pairs=[DraftPair("llama3.2", "llama3.2:1b", True)])
    engine.load("llama3.2", "3b-instruct-q4_K_M")
    sample = engine.generate_accelerated("hi", 16)
    assert engine.active_layer == AccelerationLayer.DRAFT_MODEL
    assert sample.active_layer == AccelerationLayer.DRAFT_MODEL
    assert sample.speculative_active is True


# ---- acceptance accounting --------------------------------------------------------


def test_draft_counter_acceptance_rate() -> None:
    counter = DraftCounter(lambda ids: np.array([7, 8, 9, 10], dtype=np.intc))
    for _ in range(3):
        counter(np.array([1]))
    # 3 calls drafted 12 tokens; 10 tokens generated over 4 rounds -> 6 accepted.
    assert counter.acceptance_rate(10) == pytest.approx(6 / 12)


def test_draft_counter_handles_no_drafts_and_clamps() -> None:
    counter = DraftCounter(lambda ids: np.array([], dtype=np.intc))
    counter(np.array([1]))
    assert counter.acceptance_rate(50) is None
    full = DraftCounter(lambda ids: np.array([1], dtype=np.intc))
    full(np.array([1]))
    assert full.acceptance_rate(100) == 1.0
    assert full.acceptance_rate(0) == 0.0


def test_model_draft_takes_k_greedy_tokens_and_closes_generator() -> None:
    class DraftLlm:
        closed = False

        def generate(self, tokens, top_k, temp, reset):
            assert (top_k, temp) == (1, 0.0)
            try:
                for t in range(100, 200):
                    yield t
            finally:
                DraftLlm.closed = True

    out = ModelDraft(DraftLlm(), k=3)(np.array([1, 2], dtype=np.intc))
    assert out.tolist() == [100, 101, 102]
    assert DraftLlm.closed is True
