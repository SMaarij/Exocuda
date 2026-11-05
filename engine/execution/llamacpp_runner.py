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
itself and a draft token is only kept when it equals that sample.

Weights come from Ollama's own model store (resolved through Ollama's
/api/show), so nothing is downloaded twice. Optional dependency:
`pip install llama-cpp-python`; importing this module without it is fine,
constructing a runner raises a clear error.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import psutil

from engine.execution.model_runner import (
    ACCEL_DRAFT_MODEL,
    ACCEL_NONE,
    ACCEL_PROMPT_LOOKUP,
    GenerationSample,
    OllamaReferenceModelRunner,
)

try:
    from llama_cpp import Llama
    from llama_cpp.llama_speculative import LlamaDraftModel, LlamaPromptLookupDecoding
except ImportError:  # pragma: no cover - exercised only without the optional dep
    Llama = None
    LlamaDraftModel = object
    LlamaPromptLookupDecoding = None

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

        pieces: list[str] = []
        first_t: Optional[float] = None
        sampling = {"temperature": self._temperature} if self._temperature is not None else {}
        stream = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            **sampling,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                if first_t is None:
                    first_t = time.perf_counter()
                pieces.append(delta)
        last_t = time.perf_counter()

        text = "".join(pieces)
        tokens = len(self._llm.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
        # Decode time from first to last token excludes prompt processing,
        # matching what Ollama's eval_duration measures.
        decode_s = (last_t - first_t) if first_t is not None else 0.0
        decoded_after_first = max(tokens - 1, 0)
        return GenerationSample(
            tokens_generated=decoded_after_first,
            elapsed_s=decode_s,
            # In-process, so this process's RSS is the real footprint.
            peak_memory_gb=psutil.Process().memory_info().rss / (1024**3),
            output_text=text.strip(),
            acceleration=self._mode if self._counter is not None else ACCEL_NONE,
            acceptance_rate=self._counter.acceptance_rate(tokens) if self._counter is not None else None,
        )

    def unload(self) -> None:
        self._llm = None
        self._draft_llm = None
        self._counter = None
