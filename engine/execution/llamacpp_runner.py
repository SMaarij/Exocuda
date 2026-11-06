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
    """Layer 3 draft: greedy `k` tokens from a small model, reusing its KV cache across calls."""

    def __init__(self, llm: Any, k: int = 4) -> None:
        self._llm = llm
        self._k = k

    def __call__(self, input_ids, /, **kwargs):
        import numpy as np

        out: list[int] = []
        gen = self._llm.generate(list(map(int, input_ids)), top_k=1, temp=0.0, reset=True)
        try:
            for token in gen:
                out.append(int(token))
                if len(out) >= self._k:
                    break
        finally:
            gen.close()
        return np.array(out, dtype=np.intc)


def _default_llama_factory(n_ctx: int, n_threads: Optional[int]) -> LlamaFactory:
    def make(path: str, needs_all_logits: bool) -> Any:
        # A placeholder draft at construction is the only way to make
        # llama-cpp-python keep per-position logits, which verifying
        # drafts needs. The real draft is swapped in afterwards.
        placeholder = LlamaPromptLookupDecoding() if needs_all_logits else None
        return Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, verbose=False, draft_model=placeholder)
    return make


class LlamaCppRunner:
    """ReferenceModelRunner + AcceleratedRunner over llama.cpp."""

    def __init__(
        self,
        ollama: Optional[OllamaReferenceModelRunner] = None,
        draft_model_tag: Optional[str] = DEFAULT_DRAFT_MODEL_TAG,
        draft_tokens: int = 4,
        lookup_tokens: int = 10,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        llama_factory: Optional[LlamaFactory] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._temperature = temperature
        if llama_factory is None and Llama is None:
            raise RuntimeError("LlamaCppRunner needs llama-cpp-python: pip install llama-cpp-python")
        self._ollama = ollama or OllamaReferenceModelRunner()
        self._draft_model_tag = draft_model_tag
        self._draft_tokens = draft_tokens
        self._lookup_tokens = lookup_tokens
        self._factory = llama_factory or _default_llama_factory(n_ctx, n_threads)
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
            self._counter = DraftCounter(ModelDraft(self._draft_llm, k=self._draft_tokens))
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
        batch_sizes = sorted({len(output_ids), self._draft_tokens + 1, self._lookup_tokens + 1})

        def logits(chunk: int) -> np.ndarray:
            llm.reset()
            llm.eval(list(prompt_ids))
            rest = list(output_ids[:-1])
            for i in range(0, len(rest), chunk):
                llm.eval(rest[i:i + chunk])
            return np.array(llm.scores[start:start + n], dtype=np.float64)

        saved_draft, llm.draft_model = llm.draft_model, None
        try:
            sequential = logits(1)
            batched = [logits(size) for size in batch_sizes]
        finally:
            llm.reset()
            llm.draft_model = saved_draft

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
