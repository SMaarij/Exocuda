"""
gateway/backend.py

SelfHostedBackend: the client side of the `self_hosted` tier -- a Cascade
server (gateway/api/app.py) that you or your team run on a GPU machine.
Conforms to shared.types.ExecutionBackend exactly like
LocalExecutionBackend does, so Pipeline can call either without knowing
which (NFR-07). The server is OpenAI-compatible, so this uses the same
HTTPInferenceProvider as any other endpoint. Any quota is enforced by the
server; a refusal surfaces here as SelfHostedBackendError and Pipeline
falls back to local.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from gateway.providers.base import CloudInferenceProvider
from gateway.providers.http_provider import HTTPInferenceProvider
from shared.types import BackendResult, ExecutionBackend, Message


class SelfHostedBackendError(RuntimeError):
    """The self-hosted server couldn't serve a request (unreachable, quota reached, or upstream failure)."""


class SelfHostedBackend:
    def __init__(
        self,
        provider: Optional[CloudInferenceProvider] = None,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "",
        model_id: str = "cascade",
        max_tokens: int = 512,
    ) -> None:
        self._provider: CloudInferenceProvider = provider or HTTPInferenceProvider(
            base_url=base_url, api_key=api_key, model_id=model_id
        )
        self._max_tokens = max_tokens

    def generate(self, messages: list[Message]) -> BackendResult:
        start = time.monotonic()
        try:
            sample = self._provider.generate(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            raise SelfHostedBackendError(f"self-hosted server call failed: {exc}") from exc
        wall_ms = (time.monotonic() - start) * 1000
        tok_s = sample.completion_tokens / sample.elapsed_s if sample.elapsed_s > 0 else 0.0

        return {
            "text": sample.text,
            "confidence": 0.95,
            "speculative_active": False,
            "tok_s": round(tok_s, 3),
            "latency_ms": round(wall_ms, 2),
            "est_cost_usd": 0.0,  # your own hardware
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            yield from self._provider.generate_stream(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            raise SelfHostedBackendError(f"self-hosted server stream failed: {exc}") from exc


assert isinstance(SelfHostedBackend(), ExecutionBackend)
