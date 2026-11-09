"""
engine/execution/llamacpp_runner.py

In-process llama.cpp runner (via llama-cpp-python) -- the one runner that
can do real speculative decoding, so the one that makes FR-03's layers
real rather than labels:

    ACCEL_PROMPT_LOOKUP  Layer 1: n-gram drafts copied from the context
                         (no extra model).
    ACCEL_DRAFT_MODEL    Layer 3: a small model of the same family drafts
                         `draft_tokens` tokens per step (e.g. llama3.2:1b
                         drafting for llama3.2:3b -- same tokenizer).

Both are lossless by construction: the target model samples every token
itself and a draft token is only kept when it equals that sample. One
caveat, measured rather than assumed (verify_greedy below): checking drafts
evaluates several tokens in one batch, and llama.cpp's batched arithmetic
rounds slightly differently from one-token-at-a-time decoding. Where the
model's top two choices are within that rounding noise, greedy speculative
output can pick the other one. verify_greedy measures the noise for the
exact sequence and checks every output token against it.

Every generation starts from a clean model state. llama-cpp-python
otherwise reuses the KV cache of the previous request's matching prefix
(every chat prompt shares the template header), which made the same prompt
give different output depending on what was asked before it.

Weights come from Ollama's own model store (resolved through Ollama's
/api/show), so nothing is downloaded twice. Optional dependency:
`pip install llama-cpp-python`; importing this module without it is fine,
constructing a runner raises a clear error.
"""

from __future__ import annotations

import inspect
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import psutil

from engine.execution.model_runner import (
    ACCEL_DRAFT_MODEL,
    ACCEL_NONE,
    ACCEL_PROMPT_LOOKUP,
    GenerationSample,
    OllamaReferenceModelRunner,
)

try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    from llama_cpp.llama_speculative import LlamaDraftModel, LlamaPromptLookupDecoding
except ImportError:  # pragma: no cover - exercised only without the optional dep
    llama_cpp = None
    Llama = None
    Jinja2ChatFormatter = None
    LlamaDraftModel = object
    LlamaPromptLookupDecoding = None


def _chat_sampling_defaults() -> dict:
    """llama-cpp-python's own create_chat_completion defaults, read from its
    signature so this runner samples exactly as that API would."""
    if Llama is None:
        return {}
    params = inspect.signature(Llama.create_chat_completion).parameters
    names = {"temperature": "temp", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
             "typical_p": "typical_p", "repeat_penalty": "repeat_penalty"}
    return {ours: params[theirs].default for theirs, ours in names.items() if theirs in params}


@dataclass(frozen=True)
class GreedyCheck:
    """Result of LlamaCppRunner.verify_greedy for one output."""

    tokens: int
    exact: int            # tokens that were the model's top choice, recomputed one at a time
    within_noise: int     # not the top choice, but within the measured rounding noise of it
    violations: int       # outside the noise: a real error in the decoding
    noise_tolerance: float  # largest move of a decision-relevant logit gap between one-at-a-time and batched evaluation

    @property
    def lossless(self) -> bool:
        return self.violations == 0

DEFAULT_DRAFT_MODEL_TAG = "llama3.2:1b-instruct-q4_K_M"

# (model_path, needs_all_logits) -> Llama-like object. Injectable for tests.
LlamaFactory = Callable[[str, bool], Any]


class DraftCounter:
    """
    Wraps any draft callable and counts calls and drafted tokens, so the
    runner can report a measured acceptance rate. llama-cpp-python yields
    (accepted drafts + 1) tokens per verification round, and runs
    (draft calls + 1) rounds, hence accepted = generated - calls - 1.
    """

    def __init__(self, draft: Callable[..., Any]) -> None:
        self._draft = draft
        self.calls = 0
        self.drafted = 0

    def __call__(self, input_ids, /, **kwargs):
        out = self._draft(input_ids, **kwargs)
        self.calls += 1
        self.drafted += len(out)
        return out

    def reset(self) -> None:
        self.calls = 0
        self.drafted = 0

    def acceptance_rate(self, tokens_generated: int) -> Optional[float]:
        if self.drafted == 0:
            return None
        accepted = max(0, tokens_generated - self.calls - 1)
        return min(1.0, accepted / self.drafted)


class ModelDraft(LlamaDraftModel):
    """
    Layer 3 draft: greedy `k` tokens from a small model, reusing its KV cache
    across calls.

    With `adaptive=True` the draft length follows the acceptance it observes,
    using the schedule Hugging Face Transformers applies to assisted
    generation ("heuristic"): every draft fully accepted -> k += 2, otherwise
    k -= 1 (never below 1, never above `max_k`). Acceptance is read from how
    far the sequence advanced since the previous call: the target keeps the
    accepted drafts plus one token of its own. On copy-heavy text drafts grow
    long; on open-ended text they shrink, so rejected drafts waste less work.

    Even a 1-token draft costs extra on a CPU: the draft model's step, plus
    verifying 2 tokens instead of 1 (BENCHMARK_REPORT Addendum 7). So when a
    1-token draft is rejected, drafting pauses for 1, 2, 4, ... calls (up to
    `max_pause`; 0 disables) and those steps run as plain decoding. An
    empty draft gives the target exactly its plain decoding step. The first
    fully accepted draft clears the backoff.

    With `p_min` set, drafting instead stops at the first token the draft
    model is less than `p_min` sure of (its top softmax probability), as
    llama.cpp's own speculative decoding does (`--draft-p-min`, default
    0.75). `k` is then the longest draft. A step with no confident token
    drafts nothing and costs what plain decoding costs.
    """

    def __init__(self, llm: Any, k: int = 4, adaptive: bool = False, max_k: int = 16, max_pause: int = 0,
                 p_min: Optional[float] = None) -> None:
        self._llm = llm
        self._initial_k = k
        self._k = k
        self._adaptive = adaptive
        self._max_k = max_k
        self._last_len: Optional[int] = None
        self._last_drafted = 0
        self._max_pause = max_pause
        self._p_min = p_min
        self._pause = 0      # calls left to skip drafting
        self._backoff = 0    # length of the next pause
        self.history: list[int] = []  # draft length used per call, for inspection

    @property
    def k(self) -> int:
        return self._k

    def _adapt(self, length: int) -> None:
        if self._last_len is None or length <= self._last_len:  # first call, or a new request
            self._k = self._initial_k
            self._pause = self._backoff = 0
        elif self._last_drafted:
            accepted = length - self._last_len - 1
            if accepted >= self._last_drafted:
                self._k = min(self._max_k, self._k + 2)
                self._backoff = 0
            elif self._k == 1 and self._max_pause > 0:
                self._backoff = min(self._max_pause, max(1, 2 * self._backoff))
                self._pause = self._backoff
            else:
                self._k = max(1, self._k - 1)
        self._last_len = length

    def __call__(self, input_ids, /, **kwargs):
        if self._adaptive:
            self._adapt(len(input_ids))
            if self._pause > 0:
                self._pause -= 1
                self._last_drafted = 0
                self.history.append(0)
                return np.array([], dtype=np.intc)
        self.history.append(self._k)

        if self._p_min is not None:
            out = self._confident_draft(list(map(int, input_ids)))
            self._last_drafted = len(out)
            return np.array(out, dtype=np.intc)

        out: list[int] = []
        gen = self._llm.generate(list(map(int, input_ids)), top_k=1, temp=0.0, reset=True)
        try:
            for token in gen:
                out.append(int(token))
                if len(out) >= self._k:
                    break
        finally:
            gen.close()
        self._last_drafted = len(out)
        return np.array(out, dtype=np.intc)

    def _confident_draft(self, ids: list[int]) -> list[int]:
        llm = self._llm
        # Reuse the draft's KV cache for the prefix it has already seen.
        cached = np.asarray(llm.input_ids[: llm.n_tokens])
        m = min(len(cached), len(ids))
        same = cached[:m] == np.asarray(ids[:m])
        n = m if same.all() else int(np.argmin(same))
        n = min(n, len(ids) - 1)  # always evaluate at least the last token, for its logits
        llm.n_tokens = n
        llm.eval(ids[n:])
        out: list[int] = []
        while len(out) < self._k:
            logits = self._last_logits()
            token = int(np.argmax(logits))
            top_p = 1.0 / float(np.sum(np.exp(logits - logits[token])))
            if top_p < self._p_min:
                break
            out.append(token)
            if len(out) < self._k:
                llm.eval([token])
        return out

    def _last_logits(self) -> np.ndarray:
        llm = self._llm
        return np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)


# Positions whose logits a speculative eval keeps: the verification batch is the
# last sampled token plus the draft, so this must exceed every draft length.
OUTPUT_WINDOW = 64


def _install_windowed_eval(llm: Any, window: int = OUTPUT_WINDOW) -> None:
    """
    Make a speculative-mode Llama compute logits only for the last `window`
    positions of each eval, and stop copying them into `llm.scores`.

    llama-cpp-python with a draft model asks llama.cpp for logits at *every*
    position, including every prompt token, and copies them into a Python
    buffer of only n_batch (512) rows. That cost an extra 152k-wide output
    projection per prompt token, an output buffer of n_batch x vocab floats in
    llama.cpp, and crashed outright once prompt + output passed 512 tokens.
    Sampling reads logits from the llama.cpp context (negative indices count
    from the last output), so verifying a draft needs only the last
    len(draft) + 1 of them. `llm.keep_scores = True` restores the stock eval,
    with a full-context `scores` buffer, for verify_greedy.
    """
    # No strong reference from the closure back to llm: a llm -> eval -> llm
    # cycle kept every replaced 4-5 GB model alive until the garbage collector
    # happened to run (RSS 9.4 GB after a reload instead of 5.4 GB).
    stock_eval = type(llm).eval
    ref = weakref.ref(llm)
    llm.keep_scores = False

    def eval(tokens) -> None:
        llm = ref()
        if llm.keep_scores:
            if llm.scores.shape[0] < llm._n_ctx:
                llm.scores = np.zeros((llm._n_ctx, llm._n_vocab), dtype=np.single)  # pages touched lazily
            return stock_eval(llm, tokens)
        tokens = list(tokens)
        first_output = len(tokens) - window
        llm._ctx.kv_cache_seq_rm(-1, llm.n_tokens, -1)
        for i in range(0, len(tokens), llm.n_batch):
            batch = tokens[i:i + llm.n_batch]
            n_past = llm.n_tokens
            llm._batch.set_batch(batch=batch, n_past=n_past, logits_all=False)  # last position always on
            for j in range(max(0, first_output - i), len(batch)):
                llm._batch.batch.logits[j] = True
            llm._ctx.decode(llm._batch)
            llm.input_ids[n_past:n_past + len(batch)] = batch
            llm.n_tokens += len(batch)
            llm._requires_eval = False

    llm.eval = eval


def _default_llama_factory(n_ctx: int, n_threads: Optional[int], use_mmap: bool = False) -> LlamaFactory:
    def make(path: str, needs_all_logits: bool) -> Any:
        # A placeholder draft at construction is the only way to make
        # llama-cpp-python keep per-position logits, which verifying
        # drafts needs. The real draft is swapped in afterwards.
        placeholder = LlamaPromptLookupDecoding() if needs_all_logits else None
        # use_mmap=False: llama.cpp repacks the weights into a private
        # CPU-friendly layout, so with the file also memory-mapped the
        # weights sat in RAM twice (qwen2.5-coder 7B Q4_K_M: 7.2 GB resident
        # with mmap, 4.6 GB without -- BENCHMARK_REPORT Addendum 7).
        llm = Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, use_mmap=use_mmap,
                    verbose=False, draft_model=placeholder)
        if needs_all_logits:
            _install_windowed_eval(llm)
        return llm
    return make


class LlamaCppRunner:
    """ReferenceModelRunner + AcceleratedRunner over llama.cpp."""

    def __init__(
        self,
        ollama: Optional[OllamaReferenceModelRunner] = None,
        draft_model_tag: Optional[str] = DEFAULT_DRAFT_MODEL_TAG,
        draft_tokens: int = 4,
        adaptive_draft: bool = False,
        max_draft_tokens: int = 16,
        max_draft_pause: int = 0,
        draft_p_min: Optional[float] = None,
        lookup_tokens: int = 10,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        use_mmap: bool = False,
        llama_factory: Optional[LlamaFactory] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._temperature = temperature
        if llama_factory is None and Llama is None:
            raise RuntimeError("LlamaCppRunner needs llama-cpp-python: pip install llama-cpp-python")
        self._ollama = ollama or OllamaReferenceModelRunner()
        self._draft_model_tag = draft_model_tag
        self._draft_tokens = draft_tokens
        self._adaptive_draft = adaptive_draft
        self._max_draft_tokens = max_draft_tokens
        longest = max(draft_tokens, max_draft_tokens, lookup_tokens)
        if longest >= OUTPUT_WINDOW:
            raise ValueError(f"draft length {longest} must be below {OUTPUT_WINDOW}")
        self._max_draft_pause = max_draft_pause
        self._draft_p_min = draft_p_min
        self._lookup_tokens = lookup_tokens
        self._factory = llama_factory or _default_llama_factory(n_ctx, n_threads, use_mmap)
        self._mode = ACCEL_NONE
        self._llm: Any = None
        self._llm_has_all_logits = False
        self._model_path: Optional[str] = None
        self._draft_llm: Any = None
        self._counter: Optional[DraftCounter] = None
        self._sampling = _chat_sampling_defaults()
        if temperature is not None:
            self._sampling["temp"] = temperature
        # Token ids of the last generate() call, for verify_greedy().
        self.last_prompt_ids: list[int] = []
        self.last_output_ids: list[int] = []

    # ---- weights ---------------------------------------------------------

    def local_models(self) -> list[str]:
        """Tags already downloaded (Ollama's store holds the GGUFs this runner loads)."""
        return self._ollama.local_models()

    def _resolve_path(self, tag: str) -> str:
        if tag not in self._ollama.local_models():
            self._ollama._request("/api/pull", {"model": tag, "stream": False}, timeout=1800)
        path = self._ollama.model_file_path(tag)
        if not path:
            raise RuntimeError(f"Ollama has no GGUF file for {tag}")
        return path

    def _build(self) -> None:
        needs_all_logits = self._mode != ACCEL_NONE
        self._llm = self._factory(self._model_path, needs_all_logits)
        self._llm_has_all_logits = needs_all_logits
        self._install_draft()

    def _install_draft(self) -> None:
        if self._llm is None:
            return
        if self._mode == ACCEL_PROMPT_LOOKUP:
            self._counter = DraftCounter(LlamaPromptLookupDecoding(num_pred_tokens=self._lookup_tokens))
        elif self._mode == ACCEL_DRAFT_MODEL:
            if self._draft_llm is None:
                self._draft_llm = self._factory(self._resolve_path(self._draft_model_tag), False)
            self._counter = DraftCounter(ModelDraft(
                self._draft_llm, k=self._draft_tokens, adaptive=self._adaptive_draft, max_k=self._max_draft_tokens,
                max_pause=self._max_draft_pause, p_min=self._draft_p_min
            ))
        else:
            self._counter = None
        self._llm.draft_model = self._counter

    # ---- AcceleratedRunner ---------------------------------------------

    def set_acceleration(self, mode: str) -> bool:
        if mode not in (ACCEL_NONE, ACCEL_PROMPT_LOOKUP, ACCEL_DRAFT_MODEL):
            return False
        if mode == ACCEL_DRAFT_MODEL and not self._draft_model_tag:
            return False
        if mode == ACCEL_PROMPT_LOOKUP and LlamaPromptLookupDecoding is None:
            return False
        self._mode = mode
        if self._llm is not None:
            if mode != ACCEL_NONE and not self._llm_has_all_logits:
                self._build()
            else:
                self._install_draft()
        return True

    # ---- ReferenceModelRunner -------------------------------------------

    def load(self, model_id: str, quant_level: str) -> None:
        self._model_path = self._resolve_path(f"{model_id}:{quant_level}")
        self._build()

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first.")
        if self._counter is not None:
            self._counter.reset()

        prompt_ids = self._chat_tokens(prompt)
        self._llm.reset()  # clean state: output must not depend on the previous request
        out: list[int] = []
        first_t: Optional[float] = None
        gen = self._llm.generate(prompt_ids, reset=True, **self._sampling)
        try:
            for token in gen:
                token = int(token)
                if self._is_end_of_generation(token):
                    break
                if first_t is None:
                    first_t = time.perf_counter()
                out.append(token)
                if len(out) >= max_tokens:
                    break
        finally:
            gen.close()
        last_t = time.perf_counter()

        self.last_prompt_ids, self.last_output_ids = prompt_ids, out
        text = self._llm.detokenize(out).decode("utf-8", errors="replace") if out else ""
        # Decode time from first to last token excludes prompt processing,
        # matching what Ollama's eval_duration measures.
        decode_s = (last_t - first_t) if first_t is not None else 0.0
        decoded_after_first = max(len(out) - 1, 0)
        return GenerationSample(
            tokens_generated=decoded_after_first,
            elapsed_s=decode_s,
            # In-process, so this process's RSS is the real footprint.
            peak_memory_gb=psutil.Process().memory_info().rss / (1024**3),
            output_text=text.strip(),
            acceleration=self._mode if self._counter is not None else ACCEL_NONE,
            acceptance_rate=self._counter.acceptance_rate(len(out)) if self._counter is not None else None,
        )

    def verify_greedy(self, prompt_ids: Optional[list[int]] = None, output_ids: Optional[list[int]] = None) -> GreedyCheck:
        """
        Is `output_ids` what plain greedy decoding would produce, up to
        floating-point noise? Defaults to the last generation.

        1. Recompute the target model's logits for the sequence one token at
           a time -- exactly how plain decoding computes them -- and find,
           per position, how far the chosen token is behind the top one.
        2. Measure the noise on this same sequence: recompute it in batches
           of the sizes speculative decoding uses (the whole sequence, the
           draft length + 1, the lookup length + 1) and record how much the
           gap between the top token and each competitor that matters (the
           runner-up and the chosen token) moves. The tolerance is the largest
           such move -- the noise in exactly the quantity greedy decoding
           compares, not in the ~100k logits it never looks at.
        A chosen token behind the top one by more than that is a violation.
        """
        prompt_ids = self.last_prompt_ids if prompt_ids is None else prompt_ids
        output_ids = self.last_output_ids if output_ids is None else output_ids
        if not output_ids:
            return GreedyCheck(0, 0, 0, 0, 0.0)
        if self._llm is None:
            raise RuntimeError("No model loaded -- call load() first.")
        if not self._llm_has_all_logits:  # plain mode keeps only the last logits; reload keeping all
            self._llm = self._factory(self._model_path, True)
            self._llm_has_all_logits = True
            self._install_draft()
        llm = self._llm
        n = len(output_ids)
        start = len(prompt_ids) - 1
        batch_sizes = sorted({len(output_ids), self._draft_tokens + 1, self._lookup_tokens + 1}
                             | ({self._max_draft_tokens + 1} if self._adaptive_draft else set()))

        def logits(chunk: int) -> np.ndarray:
            llm.reset()
            llm.eval(list(prompt_ids))
            rest = list(output_ids[:-1])
            for i in range(0, len(rest), chunk):
                llm.eval(rest[i:i + chunk])
            return np.array(llm.scores[start:start + n], dtype=np.float64)

        saved_draft, llm.draft_model = llm.draft_model, None
        windowed = hasattr(llm, "keep_scores")
        if windowed:
            llm.keep_scores = True
        try:
            sequential = logits(1)
            batched = [logits(size) for size in batch_sizes]
        finally:
            llm.reset()
            llm.draft_model = saved_draft
            if windowed:
                llm.keep_scores = False
                llm.scores = np.ndarray((llm.n_batch, llm._n_vocab), dtype=np.single)  # free verification rows

        rows = np.arange(n)
        order = np.argsort(sequential, axis=1)
        top, runner_up = order[:, -1], order[:, -2]
        chosen = np.asarray(output_ids)
        tolerance = 0.0
        for other in batched:
            for competitor in (runner_up, chosen):
                gap_seq = sequential[rows, top] - sequential[rows, competitor]
                gap_other = other[rows, top] - other[rows, competitor]
                tolerance = max(tolerance, float(np.max(np.abs(gap_seq - gap_other))))

        shortfall = sequential[rows, top] - sequential[rows, chosen]
        exact = int(np.sum(shortfall == 0.0))
        within = int(np.sum((shortfall > 0.0) & (shortfall <= tolerance)))
        return GreedyCheck(
            tokens=n,
            exact=exact,
            within_noise=within,
            violations=n - exact - within,
            noise_tolerance=tolerance,
        )

    # ---- prompt formatting -----------------------------------------------

    def _chat_tokens(self, prompt: str) -> list[int]:
        """The prompt as one user turn in the model's own chat template, as
        create_chat_completion would format it."""
        llm = self._llm
        template = (getattr(llm, "metadata", None) or {}).get("tokenizer.chat_template")
        if template and Jinja2ChatFormatter is not None:
            eos, bos = llm.token_eos(), llm.token_bos()
            formatter = Jinja2ChatFormatter(
                template=template,
                eos_token=llm.detokenize([eos], special=True).decode("utf-8", errors="replace") if eos != -1 else "",
                bos_token=llm.detokenize([bos], special=True).decode("utf-8", errors="replace") if bos != -1 else "",
            )
            result = formatter(messages=[{"role": "user", "content": prompt}])
            return list(llm.tokenize(result.prompt.encode("utf-8"), add_bos=not result.added_special, special=True))
        return list(llm.tokenize(prompt.encode("utf-8"), add_bos=True))

    def _is_end_of_generation(self, token: int) -> bool:
        vocab = getattr(getattr(self._llm, "_model", None), "vocab", None)
        if llama_cpp is not None and vocab is not None:
            return bool(llama_cpp.llama_vocab_is_eog(vocab, token))
        return token == self._llm.token_eos()

    def unload(self) -> None:
        self._llm = None
        self._draft_llm = None
        self._counter = None
