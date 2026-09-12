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

import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationSample:
    """One timed generation against one (model, quant_level) pair."""

    tokens_generated: int
    elapsed_s: float
    peak_memory_gb: float
    # Defaults to "" so every existing FakeReferenceModelRunner /
    # _FakeRunner construction across the test suite (none of which set
    # this) keeps working unchanged -- an empty string is an honest
    # value for a fake that was never asked to produce real text,
    # rather than a placeholder that looks like real output.
    output_text: str = ""

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

    def __init__(self, ollama_binary: str = "ollama", api_base_url: str = "http://localhost:11434") -> None:
        self._ollama_binary = ollama_binary
        self._api_base_url = api_base_url
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

    def _query_loaded_model_bytes(self) -> int:
        """
        Real memory footprint (bytes) for whatever model is currently
        loaded, from Ollama's own /api/ps -- NOT psutil on the `ollama
        run` CLI subprocess's PID. `ollama run` is a thin client; the
        model weights live in the ollama SERVER process (or one it
        manages), which is a different PID entirely. Confirmed by
        running this for real (docs/BENCHMARK_REPORT.md's addendum):
        the psutil-on-proc.pid approach always measured ~0, because it
        was measuring the wrong process's memory the whole time.
        """
        try:
            with urllib.request.urlopen(f"{self._api_base_url}/api/ps", timeout=2) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError):
            return 0
        models = data.get("models") or []
        if not models:
            return 0
        return max(m.get("size_vram") or m.get("size") or 0 for m in models)

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

        # Poll Ollama's own /api/ps on a background thread WHILE
        # generation runs (the model is only reliably reported as
        # "loaded" during that window) and keep the max seen.
        peak_bytes = 0
        stop_polling = threading.Event()

        def _poll() -> None:
            nonlocal peak_bytes
            while not stop_polling.is_set():
                peak_bytes = max(peak_bytes, self._query_loaded_model_bytes())
                stop_polling.wait(0.25)

        poll_thread = threading.Thread(target=_poll, daemon=True)
        start = time.monotonic()
        poll_thread.start()
        try:
            stdout, _stderr = proc.communicate(timeout=55)
        finally:
            stop_polling.set()
            poll_thread.join(timeout=1)
        elapsed = time.monotonic() - start

        # Ollama doesn't report a token count on stdout by default; a
        # whitespace-token approximation is a reasonable stand-in for
        # relative tok/s comparisons across quant levels of the *same*
        # model, which is all HardwareProfiler needs it for.
        tokens_generated = max(1, len(stdout.split()))
        return GenerationSample(
            tokens_generated=min(tokens_generated, max_tokens) or tokens_generated,
            elapsed_s=elapsed,
            peak_memory_gb=peak_bytes / (1024**3),
            output_text=stdout.strip(),
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
