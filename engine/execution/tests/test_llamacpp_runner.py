"""
engine/execution/tests/test_llamacpp_runner.py

LlamaCppRunner's mode switching and acceptance accounting, with an
injected fake Llama (NFR-09: no model, no llama.cpp call).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.execution import llamacpp_runner
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
    """Emits `text` word by word as token ids, then end-of-generation; calls the
    installed draft model `draft_calls` times per generation."""

    EOS, BOS = 0, 1

    def __init__(self, path: str, all_logits: bool, text: str = "one two three four five", draft_calls: int = 2) -> None:
        self.path = path
        self.all_logits = all_logits
        self.draft_model = None
        self.metadata: dict = {}
        self._text = text
        self._draft_calls = draft_calls
        self._vocab: dict[str, int] = {}
        self.last_kwargs: dict = {}
        self.resets = 0

    def _id(self, word: str) -> int:
        return self._vocab.setdefault(word, len(self._vocab) + 2)

    def tokenize(self, data: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        ids = [self._id(w) for w in data.decode().split()]
        return ([self.BOS] if add_bos else []) + ids

    def detokenize(self, ids, special: bool = False) -> bytes:
        words = {v: k for k, v in self._vocab.items()}
        return " ".join(words[i] for i in ids if i in words).encode()

    def token_eos(self) -> int:
        return self.EOS

    def token_bos(self) -> int:
        return self.BOS

    def reset(self) -> None:
        self.resets += 1

    def generate(self, tokens, reset=True, **kwargs):
        self.last_kwargs = kwargs
        if self.draft_model is not None:
            for _ in range(self._draft_calls):
                self.draft_model(np.array([1, 2, 3], dtype=np.intc))
        for word in self._text.split(" "):
            yield self._id(word)
        yield self.EOS


class FakePromptLookup:
    """Stands in for llama_cpp's LlamaPromptLookupDecoding, which is an optional dependency."""

    def __init__(self, num_pred_tokens: int = 10) -> None:
        self.num_pred_tokens = num_pred_tokens

    def __call__(self, input_ids, /, **kwargs):
        return np.array([7, 8], dtype=np.intc)


@pytest.fixture(autouse=True)
def _fake_prompt_lookup(monkeypatch):
    # Without this, these tests pass only where llama-cpp-python happens to be installed.
    monkeypatch.setattr(llamacpp_runner, "LlamaPromptLookupDecoding", FakePromptLookup)


def test_prompt_lookup_unavailable_without_llama_cpp(monkeypatch) -> None:
    monkeypatch.setattr(llamacpp_runner, "LlamaPromptLookupDecoding", None)
    runner, _ = _runner()
    assert runner.set_acceleration(ACCEL_PROMPT_LOOKUP) is False


def _runner(**kwargs):
    built: list[FakeLlama] = []

    def factory(path: str, all_logits: bool) -> FakeLlama:
        built.append(FakeLlama(path, all_logits))
        return built[-1]

    kwargs.setdefault("llama_factory", factory)
    runner = LlamaCppRunner(ollama=FakeOllama(), **kwargs)
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


def test_temperature_overrides_only_temperature() -> None:
    runner, built = _runner(temperature=0.0)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.generate("hi", 8)
    assert built[0].last_kwargs["temp"] == 0.0
    runner2, built2 = _runner()
    runner2.load("llama3.2", "3b-instruct-q4_K_M")
    runner2.generate("hi", 8)
    # Otherwise exactly llama-cpp-python's create_chat_completion defaults.
    assert built2[0].last_kwargs == llamacpp_runner._chat_sampling_defaults()


def test_every_generation_starts_from_a_clean_state() -> None:
    # The same prompt must not depend on what was asked before it.
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.generate("first question", 8)
    runner.generate("second question", 8)
    assert built[0].resets == 2


def test_generation_stops_at_max_tokens_and_records_ids() -> None:
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    sample = runner.generate("hi", 3)
    assert sample.output_text == "one two three"
    assert len(runner.last_output_ids) == 3
    assert runner.last_prompt_ids[0] == FakeLlama.BOS


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


# ---- prompt formatting --------------------------------------------------------


def test_prompt_uses_the_models_chat_template(monkeypatch) -> None:
    class Formatter:
        def __init__(self, template, eos_token, bos_token):
            self.template = template

        def __call__(self, messages):
            class Result:
                prompt = f"<turn> {messages[0]['content']} </turn>"
                added_special = False
            return Result()

    monkeypatch.setattr(llamacpp_runner, "Jinja2ChatFormatter", Formatter)
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    built[0].metadata["tokenizer.chat_template"] = "{{ messages }}"
    runner.generate("hi", 4)
    assert built[0].detokenize(runner.last_prompt_ids[1:]) == b"<turn> hi </turn>"


# ---- verify_greedy --------------------------------------------------------------


class ToyModel:
    """A 4-token vocabulary. The logits after any history favour token
    (last + 1) % 4 by `gap` over the runner-up (last + 2) % 4. Computing
    several tokens in one eval() call shifts every logit by `batch_noise`
    in a position-dependent way -- the effect verify_greedy has to measure."""

    def __init__(self, gap: float = 1.0, batch_noise: float = 0.0) -> None:
        self.gap, self.batch_noise = gap, batch_noise
        self.draft_model = None
        self.metadata: dict = {}
        self._history: list[int] = []
        self._scores: list[np.ndarray] = []

    def reset(self) -> None:
        self._history, self._scores = [], []

    def eval(self, tokens) -> None:
        for token in tokens:
            self._history.append(token)
            row = np.zeros(4)
            row[(token + 1) % 4] = 2.0
            row[(token + 2) % 4] = 2.0 - self.gap
            if len(tokens) > 1:
                row = row + self.batch_noise * np.sin(np.arange(4) + len(self._history))
            self._scores.append(row)

    @property
    def scores(self) -> np.ndarray:
        return np.array(self._scores)


def _verify(model: ToyModel, output: list[int]):
    runner, _ = _runner(llama_factory=lambda path, all_logits: model)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    return runner.verify_greedy([3], output)


def test_greedy_output_verifies_exactly() -> None:
    check = _verify(ToyModel(), [0, 1, 2, 3])
    assert (check.exact, check.within_noise, check.violations) == (4, 0, 0)
    assert check.lossless and check.noise_tolerance == 0.0


def test_runner_up_within_measured_noise_is_not_a_violation() -> None:
    # After 1 the top token is 2 and the runner-up 3 is behind by 0.01; computing
    # the earlier tokens in a batch moves that gap by more than 0.01.
    check = _verify(ToyModel(gap=0.01, batch_noise=0.05), [0, 1, 3])
    assert (check.exact, check.within_noise, check.violations) == (2, 1, 0)
    assert check.lossless and 0.01 < check.noise_tolerance < 1.0


def test_runner_up_beyond_the_noise_is_a_violation() -> None:
    check = _verify(ToyModel(gap=1.0, batch_noise=0.05), [0, 1, 3])
    assert check.violations == 1 and not check.lossless


def test_verify_restores_the_draft_and_handles_empty_output() -> None:
    model = ToyModel()
    runner, _ = _runner(llama_factory=lambda path, all_logits: model)
    runner.set_acceleration(ACCEL_PROMPT_LOOKUP)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    installed = model.draft_model
    assert isinstance(installed, DraftCounter)
    runner.verify_greedy([3], [0])
    assert model.draft_model is installed  # speculation still on after verifying
    assert runner.verify_greedy([3], []).tokens == 0


# ---- adaptive draft length ------------------------------------------------------


class _CountingDraftLlm:
    def generate(self, tokens, top_k, temp, reset):
        yield from range(1000, 1100)


def test_fixed_draft_length_never_changes() -> None:
    draft = ModelDraft(_CountingDraftLlm(), k=4)
    for length in (10, 15, 16):
        assert len(draft(np.zeros(length, dtype=np.intc))) == 4


def test_adaptive_draft_grows_when_fully_accepted_and_shrinks_otherwise() -> None:
    draft = ModelDraft(_CountingDraftLlm(), k=4, adaptive=True, max_k=8)
    draft(np.zeros(10, dtype=np.intc))           # first call: k = 4
    draft(np.zeros(15, dtype=np.intc))           # advanced 5 = 4 accepted + 1 -> all accepted -> 6
    assert draft.k == 6
    draft(np.zeros(22, dtype=np.intc))           # advanced 7 = 6 + 1 -> 8 (capped)
    draft(np.zeros(31, dtype=np.intc))           # advanced 9 = 8 + 1 -> stays at max 8
    assert draft.k == 8
    draft(np.zeros(33, dtype=np.intc))           # advanced 2 -> 1 accepted of 8 -> 7
    assert draft.k == 7
    assert draft.history == [4, 6, 8, 8, 7]


def test_adaptive_draft_never_drops_below_one_and_resets_per_request() -> None:
    draft = ModelDraft(_CountingDraftLlm(), k=2, adaptive=True, max_pause=0)
    draft(np.zeros(10, dtype=np.intc))
    draft(np.zeros(11, dtype=np.intc))           # nothing accepted -> 1
    draft(np.zeros(12, dtype=np.intc))           # still nothing -> stays 1
    assert draft.k == 1
    draft(np.zeros(5, dtype=np.intc))            # shorter sequence = a new request -> back to 2
    assert draft.k == 2


def test_runner_passes_adaptive_settings_to_the_draft() -> None:
    runner, built = _runner(adaptive_draft=True, max_draft_tokens=12)
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    draft = built[0].draft_model._draft
    assert (draft._adaptive, draft._max_k) == (True, 12)


def test_default_factory_loads_without_mmap_unless_asked(monkeypatch) -> None:
    import engine.execution.llamacpp_runner as lr

    calls = []
    monkeypatch.setattr(lr, "Llama", lambda **kwargs: calls.append(kwargs) or object())
    lr._default_llama_factory(512, None)("model.gguf", False)
    lr._default_llama_factory(512, None, use_mmap=True)("model.gguf", False)
    assert [c["use_mmap"] for c in calls] == [False, True]


def test_adaptive_draft_pauses_with_backoff_after_rejected_single_token() -> None:
    draft = ModelDraft(_CountingDraftLlm(), k=1, adaptive=True, max_pause=4)
    lengths = iter(range(10, 100))
    calls = lambda n: [len(draft(np.zeros(next(lengths), dtype=np.intc))) for _ in range(n)]
    # every call advances by 1 token: each 1-token draft is rejected
    assert calls(1) == [1]
    assert calls(2) == [0, 1]           # rejected -> pause 1, then probe
    assert calls(3) == [0, 0, 1]        # rejected again -> pause 2
    assert calls(5) == [0, 0, 0, 0, 1]  # pause 4 (capped)
    assert calls(5) == [0, 0, 0, 0, 1]  # stays at the cap
    assert draft.history[:3] == [1, 0, 1]


def test_adaptive_draft_backoff_clears_on_full_acceptance_and_new_request() -> None:
    draft = ModelDraft(_CountingDraftLlm(), k=1, adaptive=True, max_pause=8)
    draft(np.zeros(10, dtype=np.intc))
    assert len(draft(np.zeros(11, dtype=np.intc))) == 0   # rejected -> pause 1
    assert len(draft(np.zeros(12, dtype=np.intc))) == 1   # probe
    assert len(draft(np.zeros(14, dtype=np.intc))) == 3   # accepted (advanced 2) -> k = 3, backoff cleared
    assert draft._backoff == 0
    draft(np.zeros(15, dtype=np.intc))                    # 0 of 3 accepted -> k = 2
    draft(np.zeros(5, dtype=np.intc))                     # new request: k and pause reset
    assert (draft.k, draft._pause) == (1, 0)


class _ConfidenceLlm:
    """Draft-model fake: the next token after t is (t + 1) % V, with probability
    `confidence[t]` (the rest spread evenly), exposing llama-cpp-python's eval/logits API."""

    def __init__(self, confidence: dict, vocab: int = 8) -> None:
        import ctypes

        self._ctypes = ctypes
        self.confidence, self.vocab = confidence, vocab
        self.input_ids = np.zeros(256, dtype=np.intc)
        self.n_tokens = 0
        self.evaluated: list[list[int]] = []
        self._buf = None
        outer = self

        class _Ctx:
            def get_logits(self):
                return outer._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._ctx = _Ctx()

    def n_vocab(self) -> int:
        return self.vocab

    def eval(self, tokens) -> None:
        tokens = list(tokens)
        self.evaluated.append(tokens)
        self.input_ids[self.n_tokens:self.n_tokens + len(tokens)] = tokens
        self.n_tokens += len(tokens)
        last = tokens[-1]
        p = self.confidence.get(last, 0.9)
        probs = np.full(self.vocab, (1 - p) / (self.vocab - 1))
        probs[(last + 1) % self.vocab] = p
        self._buf = np.log(probs).astype(np.float32)


def test_confident_draft_stops_at_the_first_unsure_token() -> None:
    llm = _ConfidenceLlm({1: 0.95, 2: 0.9, 3: 0.5})
    draft = ModelDraft(llm, k=8, p_min=0.75)
    assert list(draft(np.array([0, 1], dtype=np.intc))) == [2, 3]  # after 3 the draft is only 50% sure


def test_confident_draft_is_empty_when_unsure_and_capped_at_k() -> None:
    assert len(ModelDraft(_ConfidenceLlm({1: 0.3}), k=8, p_min=0.75)(np.array([0, 1], dtype=np.intc))) == 0
    assert len(ModelDraft(_ConfidenceLlm({}), k=3, p_min=0.75)(np.array([0, 1], dtype=np.intc))) == 3


def test_confident_draft_reuses_the_cached_prefix() -> None:
    llm = _ConfidenceLlm({}, vocab=64)
    draft = ModelDraft(llm, k=2, p_min=0.75)
    draft(np.array([5, 6, 7], dtype=np.intc))          # evaluates the prompt, drafts 8, 9
    llm.evaluated.clear()
    draft(np.array([5, 6, 7, 8, 20], dtype=np.intc))    # 8 was cached; target replaced 9 with 20
    assert llm.evaluated[0] == [20]


def test_runner_passes_p_min_to_the_draft() -> None:
    runner, built = _runner(draft_p_min=0.8)
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert built[0].draft_model._draft._p_min == 0.8
