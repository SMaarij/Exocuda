"""
engine/execution/tests/test_llamacpp_runner.py

LlamaCppRunner's mode switching and acceptance accounting, with an
injected fake Llama (NFR-09: no model, no llama.cpp call).
"""

from __future__ import annotations

import json

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


@pytest.fixture(autouse=True)
def _model_paths_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(llamacpp_runner, "MODEL_PATHS_PATH", tmp_path / "model_paths.json")
    monkeypatch.setattr(llamacpp_runner, "KV_DISK_PATH", tmp_path / "kv")


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


def test_generations_keep_the_shared_prompt_prefix_by_default() -> None:
    # llama-cpp-python reuses the KV cache of the prefix shared with the previous request.
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.generate("first question", 8)
    runner.generate("second question", 8)
    assert built[0].resets == 0
    assert runner.last_prompt_s is not None


def test_reuse_prefix_off_starts_every_generation_from_a_clean_state() -> None:
    runner, built = _runner(reuse_prefix=False)
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


def test_plain_generation_records_token_logprobs() -> None:
    import ctypes
    import math

    V = 64
    logits = [0.0] * V
    logits[5] = 2.0

    class _Ctx:
        def __init__(self) -> None:
            self.buf = (ctypes.c_float * V)(*logits)

        def get_logits(self):
            return ctypes.cast(self.buf, ctypes.POINTER(ctypes.c_float))

    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    built[0]._ctx, built[0].n_vocab = _Ctx(), lambda: V
    runner.generate("hi", 3)
    norm = math.log(math.exp(2.0) + V - 1)
    assert runner.last_token_logprobs == pytest.approx([logits[t] - norm for t in runner.last_output_ids])


def test_speculative_generation_records_no_token_logprobs() -> None:
    runner, built = _runner()
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.generate("hi", 3)
    assert runner.last_token_logprobs is None


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


def test_messages_keep_their_roles_in_the_chat_template(monkeypatch) -> None:
    seen = []

    class Formatter:
        def __init__(self, template, eos_token, bos_token):
            pass

        def __call__(self, messages):
            seen.append(messages)

            class Result:
                prompt = " ".join(f"<{m['role']}> {m['content']}" for m in messages)
                added_special = False
            return Result()

    monkeypatch.setattr(llamacpp_runner, "Jinja2ChatFormatter", Formatter)
    runner, built = _runner()
    assert runner.accepts_messages is True
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    built[0].metadata["tokenizer.chat_template"] = "{{ messages }}"
    turns = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
             {"role": "user", "content": "q2"}]
    runner.generate(turns, 4)
    assert seen[-1] == turns
    assert built[0].detokenize(runner.last_prompt_ids[1:]) == b"<user> q1 <assistant> a1 <user> q2"


def test_native_chat_off_asks_for_a_flat_prompt() -> None:
    runner, _ = _runner(native_chat=False)
    assert runner.accepts_messages is False


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


def test_default_factory_uses_the_low_memory_working_settings(monkeypatch) -> None:
    import engine.execution.llamacpp_runner as lr

    calls = []
    monkeypatch.setattr(lr, "Llama", lambda **kwargs: calls.append(kwargs) or object())
    lr._default_llama_factory(512, None)("model.gguf", False)
    assert (calls[0]["flash_attn"], calls[0]["n_ubatch"]) == (True, 256)
    assert "type_k" not in calls[0]  # f16 KV cache: q8_0 costs prompt speed on CPU


def test_default_factory_shares_one_plain_instance_per_file(monkeypatch) -> None:
    """The small backend's model and the same file as a draft are one instance in RAM."""
    import engine.execution.llamacpp_runner as lr

    class _Model:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(lr, "Llama", _Model)
    monkeypatch.setattr(lr, "_install_windowed_eval", lambda llm: None)
    monkeypatch.setattr(lr, "_SHARED_MODELS", __import__("weakref").WeakValueDictionary())
    make = lr._default_llama_factory(512, None)
    small = make("small.gguf", False)
    assert lr._default_llama_factory(512, None)("small.gguf", False) is small  # another runner, same file
    other = make("other.gguf", False)
    assert other is not small
    assert make("small.gguf", True) is not small        # a target model is never shared
    assert lr._default_llama_factory(1024, None)("small.gguf", False) is not small  # other n_ctx
    del small
    import gc; gc.collect()
    assert len(lr._SHARED_MODELS) == 1                  # only other.gguf left: weak references


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


class _WindowLlm:
    """Just enough of a Llama for _install_windowed_eval: records which batch positions ask for logits."""

    def __init__(self, n_batch: int = 4, n_ctx: int = 32, n_vocab: int = 3) -> None:
        import types

        self.n_batch, self._n_ctx, self._n_vocab = n_batch, n_ctx, n_vocab
        self.n_tokens = 0
        self.input_ids = np.zeros(n_ctx, dtype=np.intc)
        self.scores = np.zeros((n_batch, n_vocab), dtype=np.single)
        self.decoded: list[list[bool]] = []
        self.stock_calls: list[list[int]] = []
        llm = self

        class _Batch:
            def __init__(self) -> None:
                self.batch = types.SimpleNamespace(logits=[False] * n_batch, n_tokens=0)

            def set_batch(self, batch, n_past, logits_all) -> None:
                self.batch.n_tokens = len(batch)
                for i in range(len(batch)):
                    self.batch.logits[i] = logits_all
                self.batch.logits[len(batch) - 1] = True

        class _Ctx:
            def kv_cache_seq_rm(self, *args) -> None:
                pass

            def decode(self, batch) -> None:
                llm.decoded.append(list(batch.batch.logits[:batch.batch.n_tokens]))

        self._batch, self._ctx = _Batch(), _Ctx()

    def eval(self, tokens) -> None:
        self.stock_calls.append(list(tokens))


def test_windowed_eval_asks_for_logits_only_at_the_last_positions() -> None:
    from engine.execution.llamacpp_runner import _install_windowed_eval

    llm = _WindowLlm(n_batch=4)
    _install_windowed_eval(llm, window=3)
    llm.eval([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # a prompt longer than one batch
    # set_batch always marks each chunk's last position (one harmless row per chunk)
    assert llm.decoded == [[False, False, False, True], [False, False, False, True], [True, True]]
    assert llm.n_tokens == 10
    assert list(llm.input_ids[:10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert llm.stock_calls == []
    llm.eval([11, 12])  # a verification batch shorter than the window: every position
    assert llm.decoded[-1] == [True, True]
    assert llm.n_tokens == 12


def test_windowed_eval_keep_scores_uses_the_stock_eval_with_a_full_context_buffer() -> None:
    from engine.execution.llamacpp_runner import _install_windowed_eval

    llm = _WindowLlm(n_batch=4, n_ctx=32)
    _install_windowed_eval(llm, window=3)
    llm.keep_scores = True
    llm.eval([1, 2])
    assert llm.stock_calls == [[1, 2]]
    assert llm.scores.shape == (32, 3)
    assert llm.decoded == []


def test_runner_rejects_drafts_longer_than_the_output_window() -> None:
    from engine.execution.llamacpp_runner import OUTPUT_WINDOW

    with pytest.raises(ValueError):
        LlamaCppRunner(llama_factory=lambda path, all_logits: None, max_draft_tokens=OUTPUT_WINDOW)


def test_windowed_eval_does_not_keep_its_model_alive() -> None:
    from engine.execution.llamacpp_runner import _install_windowed_eval

    llm = _WindowLlm()
    _install_windowed_eval(llm)
    # No llm -> eval -> llm cycle: such a cycle kept replaced 4-5 GB models in RAM until a gc pass.
    captured = [cell.cell_contents for cell in llm.eval.__closure__ or ()]
    assert all(obj is not llm for obj in captured)
    assert not any(getattr(obj, "__self__", None) is llm for obj in captured)


def test_runner_is_self_regulating_only_with_confidence_drafting() -> None:
    plain = LlamaCppRunner(llama_factory=lambda path, all_logits: None)
    plain.set_acceleration(ACCEL_DRAFT_MODEL)
    assert not plain.self_regulating
    confident = LlamaCppRunner(llama_factory=lambda path, all_logits: None, draft_p_min=0.75)
    assert not confident.self_regulating
    confident.set_acceleration(ACCEL_DRAFT_MODEL)
    assert confident.self_regulating


# ---- cost-aware draft length ------------------------------------------------------------


def test_cost_aware_draft_rounds_up_to_a_cheap_batch() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    # p_min 0.75 would stop after [2, 3] (a 3-token batch, 1.74x); a 4-token batch costs
    # only 1.55x, so the 50%-sure third token is worth verifying too.
    llm = _ConfidenceLlm({1: 0.95, 2: 0.9, 3: 0.5})
    assert list(ModelDraft(llm, k=16, verify_costs=CPU_VERIFY_COSTS)(np.array([0, 1], dtype=np.intc))) == [2, 3, 4]


def test_cost_aware_draft_skips_drafting_when_verifying_costs_as_much_as_decoding() -> None:
    linear = [0.0] + [float(n) for n in range(1, 18)]  # no batching benefit at all
    llm = _ConfidenceLlm({}, vocab=64)
    assert len(ModelDraft(llm, k=16, verify_costs=linear)(np.array([0, 1], dtype=np.intc))) == 0


def test_cost_aware_draft_drafts_long_when_batches_are_almost_free() -> None:
    flat = [0.0] + [1.0 + 0.01 * n for n in range(1, 18)]  # GPU-like
    llm = _ConfidenceLlm({}, vocab=64)  # 90% sure of every token; drafting itself free
    assert len(ModelDraft(llm, k=16, verify_costs=flat, draft_cost=0.0)(np.array([0, 1], dtype=np.intc))) == 16


def test_cost_aware_draft_stops_drafting_once_no_longer_draft_can_win() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    llm = _ConfidenceLlm({t: 0.2 for t in range(64)}, vocab=64)
    draft = ModelDraft(llm, k=16, verify_costs=CPU_VERIFY_COSTS)
    assert len(draft(np.array([0, 1], dtype=np.intc))) == 0
    assert len(llm.evaluated) == 1  # only the sync eval: no wasted draft steps


def test_cost_aware_draft_needs_a_cost_for_every_batch_size() -> None:
    with pytest.raises(ValueError):
        ModelDraft(_ConfidenceLlm({}), k=16, verify_costs=[0.0, 1.0, 1.3])


def test_runner_passes_verify_costs_to_the_draft() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    runner, built = _runner(draft_verify_costs=CPU_VERIFY_COSTS, max_draft_tokens=16, draft_tokens=16)
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert built[0].draft_model._draft._verify_costs == list(CPU_VERIFY_COSTS)
    assert runner.self_regulating


def test_auto_verify_costs_fall_back_to_the_reference_table_without_a_real_model() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    runner, built = _runner(draft_verify_costs="auto", max_draft_tokens=16, draft_tokens=16)
    runner.set_acceleration(ACCEL_DRAFT_MODEL)
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert built[0].draft_model._draft._verify_costs == list(CPU_VERIFY_COSTS)


class _TimedLlm:
    """Target fake whose eval of n tokens takes 1 + 0.1 n ticks of `clock`, with llama-cpp-python's KV API."""

    def __init__(self) -> None:
        self.clock = 0.0
        self.n_tokens = 0
        self.evals = 0
        outer = self

        class Ctx:
            def kv_cache_clear(self) -> None:
                outer.n_tokens = 0

            def kv_cache_seq_rm(self, seq, start, end) -> None:
                pass

        self._ctx = Ctx()

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        return list(range(len(text.split())))

    def eval(self, tokens) -> None:
        self.evals += 1
        self.n_tokens += len(tokens)
        self.clock += 1.0 + 0.1 * len(tokens)


def test_verify_costs_are_measured_once_per_model_and_machine(tmp_path, monkeypatch) -> None:
    llm = _TimedLlm()
    monkeypatch.setattr(llamacpp_runner.time, "perf_counter", lambda: llm.clock)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 10)
    cache = tmp_path / "verify_costs.json"
    costs = llamacpp_runner.verify_costs_for(llm, str(model), 5, cache_path=cache)
    assert len(costs) == 6 and costs[0] == 0.0 and costs[1] == 1.0
    assert costs[5] == round(1.5 / 1.1, 3) and llm.n_tokens == 0  # context left empty for the next prompt
    evals = llm.evals
    assert llamacpp_runner.verify_costs_for(llm, str(model), 5, cache_path=cache) == costs
    assert llm.evals == evals  # read from the cache, not measured again
    model.write_bytes(b"x" * 20)  # a different model file: measured afresh
    llamacpp_runner.verify_costs_for(llm, str(model), 5, cache_path=cache)
    assert llm.evals > evals


def test_lookup_drafts_the_continuation_of_repeated_text_without_the_draft_model() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    llm = _ConfidenceLlm({}, vocab=64)
    draft = ModelDraft(llm, k=16, verify_costs=CPU_VERIFY_COSTS)
    draft.lookup_stats[3] = [95.0, 100.0]  # lookup drafts after 3-token matches were 95% accepted
    ids = [9, 10, 11, 12, 13, 14, 15, 16, 30, 9, 10, 11]  # "9 10 11" seen before, followed by 12 13 ...
    out = list(draft(np.array(ids, dtype=np.intc)))
    assert len(out) >= 3 and out == [12, 13, 14, 15, 16, 30, 9, 10, 11][: len(out)] and draft.lookup_drafts == 1
    assert llm.evaluated == []  # the draft model never ran


def test_lookup_learns_from_how_far_the_sequence_advanced() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    draft = ModelDraft(_ConfidenceLlm({}, vocab=64), k=16, verify_costs=CPU_VERIFY_COSTS)
    draft.lookup_stats[3] = [90.0, 100.0]
    ids = [9, 10, 11, 12, 13, 14, 15, 16, 30, 9, 10, 11]
    out = list(draft(np.array(ids, dtype=np.intc)))
    draft(np.array(ids + out[:2] + [40], dtype=np.intc))  # target kept 2 drafts, then its own token
    assert draft.lookup_stats[3] == [92.0, 103.0]


def test_lookup_falls_back_to_the_draft_model_when_copies_do_not_pay() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    llm = _ConfidenceLlm({}, vocab=64)
    draft = ModelDraft(llm, k=16, verify_costs=CPU_VERIFY_COSTS)
    draft.lookup_stats[3] = [1.0, 100.0]  # copies after 3-token matches are almost never kept
    ids = [9, 10, 11, 12, 13, 14, 15, 16, 30, 9, 10, 11]
    out = list(draft(np.array(ids, dtype=np.intc)))
    assert draft.lookup_drafts == 0 and out[:1] == [12] and llm.evaluated  # drafted by the model
    off = ModelDraft(_ConfidenceLlm({}, vocab=64), k=16, verify_costs=CPU_VERIFY_COSTS, lookup=False)
    off.lookup_stats[3] = [99.0, 100.0]
    off(np.array(ids, dtype=np.intc))
    assert off.lookup_drafts == 0


def test_a_model_found_before_loads_without_ollama(tmp_path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    llamacpp_runner.MODEL_PATHS_PATH.write_text(json.dumps({"llama3.2:3b-instruct-q4_K_M": str(gguf)}))

    class DownOllama:
        def local_models(self):
            raise AssertionError("Ollama was asked")

    runner = LlamaCppRunner(ollama=DownOllama(), llama_factory=lambda path, all_logits: FakeLlama(path, all_logits))
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    assert runner._model_path == str(gguf)


def test_lookup_searches_earlier_answers_when_this_request_has_no_repeat() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    llm = _ConfidenceLlm({}, vocab=64)
    draft = ModelDraft(llm, k=16, verify_costs=CPU_VERIFY_COSTS)
    draft.history_stats[3] = [95.0, 100.0]
    prompt = [1, 2, 3]
    answer = [20, 21, 22, 23, 24, 25, 26, 27]
    for i in range(len(answer) + 1):  # request 1 decodes `answer` step by step
        draft(np.array(prompt + answer[:i], dtype=np.intc))
    llm.evaluated.clear()
    lookups = draft.lookup_drafts
    out = list(draft(np.array([4, 5, 20, 21, 22], dtype=np.intc)))  # request 2 repeats its start
    assert draft.lookup_drafts == lookups + 1 and llm.evaluated == []
    assert len(out) >= 3 and out == [23, 24, 25, 26, 27][: len(out)]


def test_earlier_answers_never_continue_past_their_end() -> None:
    from engine.execution.llamacpp_runner import CPU_VERIFY_COSTS

    draft = ModelDraft(_ConfidenceLlm({}, vocab=64), k=16, verify_costs=CPU_VERIFY_COSTS)
    draft.history_stats[3] = [99.0, 100.0]
    for ids in ([1, 2], [1, 2, 20, 21, 22, 23, 24], [7, 8], [7, 8, 30, 31, 32, 33], [9]):
        draft(np.array(ids, dtype=np.intc))
    out = list(draft._lookup_draft([5, 21, 22, 23]))
    assert out == [24]  # the first answer ended there; the next one is not its continuation


# ---- KV slots -----------------------------------------------------------------------


class _KVLlm:
    """A model whose KV cache is just the tokens it holds."""

    def __init__(self) -> None:
        self._ctx = object()
        self.input_ids = np.zeros(4096, dtype=np.intc)
        self.n_tokens = 0
        self._requires_eval = False
        self.kv: list[int] = []

    def evaluate(self, ids: list[int]) -> None:
        self.input_ids[: len(ids)] = ids
        self.n_tokens = len(ids)
        self.kv = list(ids)


@pytest.fixture
def _kv_as_tokens(monkeypatch):
    monkeypatch.setattr(llamacpp_runner, "_kv_get", lambda llm: json.dumps(llm.kv).encode())

    def kv_set(llm, data):
        llm.kv = json.loads(data)
    monkeypatch.setattr(llamacpp_runner, "_kv_set", kv_set)


def test_a_conversation_comes_back_to_its_saved_cache(_kv_as_tokens) -> None:
    from engine.execution.llamacpp_runner import KVSlots

    llm, slots = _KVLlm(), KVSlots(max_tokens=1000, min_tokens=10)
    chat_a = list(range(100, 300))
    llm.evaluate(chat_a)
    other = list(range(500, 650))
    slots.switch(llm, other)  # a different conversation: chat A is saved before it is overwritten
    assert len(slots) == 1 and llm.kv == chat_a
    llm.evaluate(other)
    slots.switch(llm, chat_a + [7, 8, 9])  # chat A continues
    assert llm.kv == chat_a and llm.n_tokens == len(chat_a) and llm._requires_eval and slots.restores == 1
    assert list(llm.input_ids[: llm.n_tokens]) == chat_a
    assert len(slots) == 1  # the other conversation was saved in its place


def test_short_or_continuing_caches_are_not_saved(_kv_as_tokens) -> None:
    from engine.execution.llamacpp_runner import KVSlots

    llm, slots = _KVLlm(), KVSlots(max_tokens=1000, min_tokens=10)
    llm.evaluate([1, 2, 3])
    slots.switch(llm, [9, 9, 9])
    llm.evaluate(list(range(50)))
    slots.switch(llm, list(range(50)) + [1, 2])
    assert len(slots) == 0


def test_saved_caches_stay_within_the_token_budget(_kv_as_tokens) -> None:
    from engine.execution.llamacpp_runner import KVSlots

    llm, slots = _KVLlm(), KVSlots(max_tokens=250, min_tokens=10)
    for base in (1000, 2000, 3000):
        llm.evaluate(list(range(base, base + 100)))
        slots.switch(llm, [5] * 20)
    assert len(slots) == 2 and slots.tokens == 200  # the oldest conversation was dropped
    llm.evaluate([5] * 20)
    slots.switch(llm, list(range(1000, 1100)))
    assert slots.restores == 0


def test_kv_slots_are_off_for_test_doubles_and_when_disabled() -> None:
    from engine.execution.llamacpp_runner import kv_slots_for

    assert kv_slots_for(FakeLlama("m", False)) is None
    assert kv_slots_for(_KVLlm(), max_tokens=0) is None
    llm = _KVLlm()
    assert kv_slots_for(llm) is kv_slots_for(llm)  # one store per model instance


# ---- KV caches on disk -------------------------------------------------------------


def test_disk_store_round_trips_and_reindexes(tmp_path) -> None:
    from engine.execution.llamacpp_runner import KVDisk

    disk = KVDisk(tmp_path / "m")
    disk.put((1, 2, 3), b"state-a")
    assert disk.take((1, 2, 3)) == b"state-a"
    again = KVDisk(tmp_path / "m")  # a new process finds it
    assert list(again.entries()) == [(1, 2, 3)] and again.take((1, 2, 3)) == b"state-a"


def test_a_longer_cache_replaces_the_one_it_extends(tmp_path) -> None:
    from engine.execution.llamacpp_runner import KVDisk

    disk = KVDisk(tmp_path / "m")
    disk.put((1, 2, 3), b"short")
    disk.put((1, 2, 3, 4, 5), b"long")
    disk.put((9, 9), b"other")
    assert sorted(disk.entries()) == [(1, 2, 3, 4, 5), (9, 9)]
    assert len(list((tmp_path / "m").glob("*.kv"))) == 2


def test_disk_store_drops_the_oldest_files_over_budget(tmp_path) -> None:
    import os
    from engine.execution.llamacpp_runner import KVDisk

    disk = KVDisk(tmp_path / "m", max_bytes=2500)
    for i in range(3):
        disk.put((i, i, i), bytes(1000))
        for path in (tmp_path / "m").glob("*.kv"):  # older files really are older
            os.utime(path, (path.stat().st_mtime - 10, path.stat().st_mtime - 10))
    assert sorted(disk.entries()) == [(1, 1, 1), (2, 2, 2)]


def test_a_new_process_resumes_a_conversation_from_disk(_kv_as_tokens, tmp_path) -> None:
    from engine.execution.llamacpp_runner import KVDisk, KVSlots

    chat = list(range(100, 300))
    llm, slots = _KVLlm(), KVSlots(max_tokens=1000, min_tokens=10, disk=KVDisk(tmp_path / "m"))
    llm.evaluate(chat)
    slots.spill(llm)  # unload / exit
    fresh, slots2 = _KVLlm(), KVSlots(max_tokens=1000, min_tokens=10, disk=KVDisk(tmp_path / "m"))
    slots2.switch(fresh, chat + [1, 2, 3])
    assert fresh.kv == chat and fresh.n_tokens == len(chat) and slots2.restores == 1


def test_an_unreadable_disk_cache_is_dropped_and_the_model_starts_clean(monkeypatch, _kv_as_tokens, tmp_path) -> None:
    from engine.execution.llamacpp_runner import KVDisk, KVSlots

    disk = KVDisk(tmp_path / "m")
    disk.put(tuple(range(100, 300)), b"not-a-state")

    def refuse(llm, data):
        raise RuntimeError("llama.cpp refused the saved KV cache")
    monkeypatch.setattr(llamacpp_runner, "_kv_set", refuse)
    llm, slots = _KVLlm(), KVSlots(max_tokens=1000, min_tokens=10, disk=disk)
    slots.switch(llm, list(range(100, 300)) + [1])
    assert llm.n_tokens == 0 and slots.restores == 0 and not disk.entries()


def test_unload_writes_the_kv_caches_to_disk(monkeypatch) -> None:
    spilled = []
    monkeypatch.setattr(llamacpp_runner, "spill_kv", lambda llm: spilled.append(llm))
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    runner.unload()
    assert built[0] in spilled



def test_generate_streams_each_decoded_piece() -> None:
    runner, built = _runner()
    runner.load("llama3.2", "3b-instruct-q4_K_M")
    pieces: list[str] = []
    sample = runner.generate("hi", 10, on_text=pieces.append)
    assert pieces == ["one", "two", "three", "four", "five"]
    assert sample.output_text == "one two three four five"
    assert runner.streams is True
