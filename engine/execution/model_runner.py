"""
engine/execution/model_runner.py

FR-01.2 requires HardwareProfiler to "load a reference model at no fewer
than 2 quantization levels and measure real tokens/sec and peak memory."
NFR-09 requires every component's unit tests to reach 70%+ coverage
*independent of a live GPU or live Ollama instance*.

Those two requirements together mean the actual model-loading/generation
mechanics must sit behind a small interface (`ReferenceModelRunner`) that
HardwareProfiler depends on but never constructs a concrete instance of
internally. Production code injects `OllamaReferenceModelRunner`
(subprocess against a locally installed Ollama/llama.cpp, matching the
Development Stack in Section 5.1); tests inject a deterministic fake that
returns canned tok/s and memory numbers with zero external dependencies.

This mirrors, at a smaller scale, the same seam discipline the rest of
Cascade uses (ExecutionBackend in shared/types.py): HardwareProfiler
codes against the Protocol, never the concrete runner.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import psutil


@dataclass(frozen=True)
class GenerationSample:
    """One timed generation against one (model, quant_level) pair."""

    tokens_generated: int
    elapsed_s: float
    peak_memory_gb: float

    @property
    def tok_s(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.tokens_generated / self.elapsed_s


@runtime_checkable
class ReferenceModelRunner(Protocol):
    """
    What HardwareProfiler needs from "a model backend" to benchmark a
    machine. Deliberately narrow -- this is NOT the full
    AccelerationEngine/ModelLadder model-loading surface (that's Phase
    2's job); it only needs to load one quant level and time one
    generation.
    """

    def load(self, model_id: str, quant_level: str) -> None:
        """Load `model_id` at `quant_level`. Raise on failure (e.g. OOM)
        so the caller can record that quant level as unsupported rather
        than silently skip it."""
        ...

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        """Run one timed generation against the currently loaded model
        and return tokens produced, elapsed wall-clock time, and peak
        memory observed during the call."""
        ...

    def unload(self) -> None:
        """Release the currently loaded model before the next quant
        level is loaded, so peak-memory measurements don't carry over
        between quant levels."""
        ...


class OllamaReferenceModelRunner:
    """
    Production `ReferenceModelRunner`: drives a locally installed Ollama
    instance via subprocess (Section 5.1's local dev stack:
    "Ollama/llama.cpp (local)"). Intentionally NOT used in unit tests
    (NFR-09) -- exercised only by integration/manual testing against a
    real Ollama install.
    """

    def __init__(self, ollama_binary: str = "ollama") -> None:
        self._ollama_binary = ollama_binary
        self._loaded_model: str | None = None

    def _check_available(self) -> None:
        if shutil.which(self._ollama_binary) is None:
            raise RuntimeError(
                f"'{self._ollama_binary}' not found on PATH. Install "
                "Ollama (https://ollama.com) or inject a different "
                "ReferenceModelRunner."
            )

    def load(self, model_id: str, quant_level: str) -> None:
        self._check_available()
        tag = f"{model_id}:{quant_level}"
        # `ollama pull` is a no-op if the tag is already present locally.
        # encoding/errors explicit: `ollama pull`'s progress output uses
        # Unicode block-drawing characters that aren't valid in the
        # cp1252 codec subprocess.run's `text=True` falls back to by
        # default on Windows -- found via a real live-model benchmark
        # run (docs/BENCHMARK_REPORT.md), not a hypothetical.
        result = subprocess.run(
            [self._ollama_binary, "pull", tag],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=55,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to load {tag} via Ollama: {result.stderr.strip()}"
            )
        self._loaded_model = tag

    def generate(self, prompt: str, max_tokens: int) -> GenerationSample:
        if self._loaded_model is None:
            raise RuntimeError("No model loaded -- call load() first.")

        proc = subprocess.Popen(
            [self._ollama_binary, "run", self._loaded_model, prompt],
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        peak_rss_bytes = 0
        start = time.monotonic()
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

        stdout, _stderr = proc.communicate(timeout=55)
        elapsed = time.monotonic() - start

        if ps_proc is not None:
            try:
                peak_rss_bytes = ps_proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                peak_rss_bytes = 0

        # Ollama doesn't report a token count on stdout by default; a
        # whitespace-token approximation is a reasonable stand-in for
        # relative tok/s comparisons across quant levels of the *same*
        # model, which is all HardwareProfiler needs it for.
        tokens_generated = max(1, len(stdout.split()))
        return GenerationSample(
            tokens_generated=min(tokens_generated, max_tokens) or tokens_generated,
            elapsed_s=elapsed,
            peak_memory_gb=peak_rss_bytes / (1024**3),
        )

    def unload(self) -> None:
        if self._loaded_model is not None:
            subprocess.run(
                [self._ollama_binary, "stop", self._loaded_model],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        self._loaded_model = None
