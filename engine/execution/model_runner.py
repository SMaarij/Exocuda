"""
engine/execution/model_runner.py

FR-01.2 requires HardwareProfiler to "load a reference model at no fewer
than 2 quantization levels and measure real tokens/sec and peak memory."
NFR-09 requires every component's unit tests to reach 70%+ coverage
*independent of a live GPU or live Ollama instance*.

Those two requirements together mean the actual model-loading/generation
mechanics must sit behind a small interface (`ReferenceModelRunner`) that
HardwareProfiler, AccelerationEngine and LocalExecutionBackend depend on
but never construct internally. Production code injects
`OllamaReferenceModelRunner` (Ollama's local HTTP API) or
`engine.execution.llamacpp_runner.LlamaCppRunner` (in-process llama.cpp,
the only runner that can do real speculative decoding); tests inject
deterministic fakes.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 ::1 first and
# stalls ~2s per request before falling back, since Ollama listens on IPv4.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# What a runner actually did for one generation. Reported by the runner,
# never assumed by the caller -- see AccelerationEngine.
ACCEL_NONE = "none"
ACCEL_PROMPT_LOOKUP = "prompt_lookup"
ACCEL_DRAFT_MODEL = "draft_model"


@dataclass(frozen=True)
class GenerationSample:
    """One timed generation against one (model, quant_level) pair."""

    tokens_generated: int
    # Decode time in seconds. Runners that can separate decode time from
    # model-load/prompt-processing time report decode time only, so
    # tok_s is a real decode-speed figure comparable across runs.
    elapsed_s: float
    peak_memory_gb: float
    output_text: str = ""
    acceleration: str = ACCEL_NONE
    # Fraction of drafted tokens the target model accepted; None when no
    # speculative decoding ran.
    acceptance_rate: Optional[float] = None

    @property
    def tok_s(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.tokens_generated / self.elapsed_s


@runtime_checkable
class ReferenceModelRunner(Protocol):
    """
    What the engine needs from "a model backend". Deliberately narrow:
    load one quant level, time one generation, release it.
    """

    def load(self, model_id: str, quant_level: str) -> None:
        """Load `model_id` at `quant_level`. Raise on failure (e.g. OOM)
        so the caller can step down rather than silently skip it."""
        ...

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        """Run one timed generation against the currently loaded model."""
        ...

    def unload(self) -> None:
        """Release the currently loaded model."""
        ...


@runtime_checkable
class AcceleratedRunner(Protocol):
    """
    Optional capability. A runner that can genuinely change how it
    decodes implements this; AccelerationEngine only reports a layer as
    active when the runner accepted it AND the sample says it ran.
    """

    def set_acceleration(self, mode: str) -> bool:
        """Switch decoding mode (ACCEL_* constant). Return False if this
        runner cannot do `mode` -- the caller then tries a lower one."""
        ...


class OllamaError(RuntimeError):
    """Ollama server unreachable, model missing, or a request failed."""


class OllamaReferenceModelRunner:
    """
    Production runner over Ollama's local HTTP API.

    - Real token counts and decode timings (`eval_count`/`eval_duration`)
      instead of a whitespace approximation of CLI stdout.
    - The model stays resident between requests (`keep_alive`); a
      registry pull only happens when the model is missing locally, so a
      normal request never leaves the machine (NFR-05).
    - Peak memory comes from `/api/ps`, which reports the serving
      process's real footprint.

    Does not implement AcceleratedRunner: Ollama exposes no speculative
    decoding control, so every sample honestly reports ACCEL_NONE.
    Integration-test scope only (NFR-09); the HTTP parsing is unit-tested
    by mocking `urllib.request.urlopen`.
    """

    def __init__(
        self,
        api_base_url: str = DEFAULT_OLLAMA_URL,
        keep_alive: str = "10m",
        request_timeout_s: float = 300.0,
        pull_timeout_s: float = 1800.0,
    ) -> None:
        self._api_base_url = api_base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._request_timeout_s = request_timeout_s
        self._pull_timeout_s = pull_timeout_s
        self._loaded_model: Optional[str] = None

    # ---- HTTP plumbing -------------------------------------------------

    def _request(self, path: str, payload: Optional[dict] = None, timeout: float = 10.0) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            f"{self._api_base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise OllamaError(f"Ollama {path} failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise OllamaError(
                f"Can't reach Ollama at {self._api_base_url} ({exc}). "
                "Is it installed and running? See https://ollama.com"
            ) from exc
        return json.loads(body) if body else {}

    def is_available(self) -> bool:
        try:
            self._request("/api/version", timeout=3)
            return True
        except OllamaError:
            return False

    def local_models(self) -> list[str]:
        return [m.get("name", "") for m in self._request("/api/tags").get("models", [])]

    def model_file_path(self, tag: str) -> Optional[str]:
        """Path of the GGUF blob Ollama stores for `tag` (lets LlamaCppRunner reuse Ollama's downloads)."""
        modelfile = self._request("/api/show", {"model": tag}).get("modelfile", "")
        for line in modelfile.splitlines():
            if line.startswith("FROM "):
                return line[len("FROM "):].strip()
        return None

    def _query_loaded_model_bytes(self) -> int:
        """Real memory footprint (bytes) of loaded models, from /api/ps. Never raises."""
        try:
            with urllib.request.urlopen(f"{self._api_base_url}/api/ps", timeout=2) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError):
            return 0
        models = data.get("models") or []
        if not models:
            return 0
        return max(m.get("size_vram") or m.get("size") or 0 for m in models)

    # ---- ReferenceModelRunner ------------------------------------------

    def load(self, model_id: str, quant_level: str) -> None:
        tag = f"{model_id}:{quant_level}"
        if tag not in self.local_models():
            self._request("/api/pull", {"model": tag, "stream": False}, timeout=self._pull_timeout_s)
        # An empty prompt makes Ollama load the weights without generating,
        # so an OOM surfaces here rather than mid-request.
        self._request(
            "/api/generate",
            {"model": tag, "prompt": "", "keep_alive": self._keep_alive},
            timeout=self._request_timeout_s,
        )
        self._loaded_model = tag

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        if self._loaded_model is None:
            raise OllamaError("No model loaded -- call load() first.")
        start = time.monotonic()
        payload = self._request(
            "/api/generate",
            {
                "model": self._loaded_model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": self._keep_alive,
                "options": {"num_predict": max_tokens},
            },
            timeout=self._request_timeout_s,
        )
        wall_s = time.monotonic() - start
        return parse_generate_response(payload, wall_s, self._query_loaded_model_bytes())

    def unload(self) -> None:
        if self._loaded_model is not None:
            try:
                self._request("/api/generate", {"model": self._loaded_model, "keep_alive": 0})
            except OllamaError:
                pass
        self._loaded_model = None


def parse_generate_response(payload: dict, wall_s: float, loaded_bytes: int) -> GenerationSample:
    """Turn an Ollama /api/generate response into a GenerationSample using its own timing fields."""
    eval_count = int(payload.get("eval_count") or 0)
    eval_duration_s = (payload.get("eval_duration") or 0) / 1e9
    return GenerationSample(
        tokens_generated=eval_count,
        elapsed_s=eval_duration_s if eval_duration_s > 0 else wall_s,
        peak_memory_gb=loaded_bytes / (1024**3),
        output_text=(payload.get("response") or "").strip(),
    )
