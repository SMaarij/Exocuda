"""
gateway/byok.py

CloudBYOKBackend: the `cloud_byok` tier SemanticRouter routes hard
reasoning/coding queries to. Calls the user's own provider with the
user's own key, straight from their machine -- nothing goes through
any Cascade server.

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, Groq, Together,
DeepSeek, a local vLLM...), via the same HTTPInferenceProvider the self-hosted
tier uses. This covers what the master doc planned LiteLLM for -- one
interface, config-only provider swap -- without the extra dependency.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from cascade_local.gateway.providers.base import CloudInferenceProvider
from cascade_local.gateway.providers.http_provider import HTTPInferenceProvider
from cascade_local.shared.types import BackendResult, ExecutionBackend, Message


class CloudBYOKBackendError(RuntimeError):
    """The user's provider rejected or failed the request."""


class CloudBYOKBackend:
    def __init__(
        self,
        provider: Optional[CloudInferenceProvider] = None,
        base_url: str = "https://api.openai.com",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        input_price_per_1k: float = 0.0,
        output_price_per_1k: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self._provider = provider or HTTPInferenceProvider(base_url=base_url, api_key=api_key, model_id=model)
        self._input_price = input_price_per_1k
        self._output_price = output_price_per_1k
        self._max_tokens = max_tokens

    def generate(self, messages: list[Message]) -> BackendResult:
        start = time.monotonic()
        try:
            sample = self._provider.generate(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            raise CloudBYOKBackendError(f"BYOK provider call failed: {exc}") from exc
        wall_ms = (time.monotonic() - start) * 1000
        # Charged by the user's provider, not Cascade; only an estimate,
        # and 0.0 when the user hasn't configured their provider's prices.
        cost = (sample.prompt_tokens * self._input_price + sample.completion_tokens * self._output_price) / 1000.0
        tok_s = sample.completion_tokens / sample.elapsed_s if sample.elapsed_s > 0 else 0.0
        return {
            "text": sample.text,
            "confidence": 0.95,
            "speculative_active": False,
            "tok_s": round(tok_s, 3),
            "latency_ms": round(wall_ms, 2),
            "est_cost_usd": round(cost, 6),
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        try:
            yield from self._provider.generate_stream(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            raise CloudBYOKBackendError(f"BYOK provider stream failed: {exc}") from exc


assert isinstance(CloudBYOKBackend(), ExecutionBackend)
