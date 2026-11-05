"""
gateway/backend.py

CascadeCloudBackend: the client side of the hosted tier. Conforms to
shared.types.ExecutionBackend exactly like LocalExecutionBackend does, so
Pipeline can call either without knowing which (NFR-07). It talks to
Cascade's own gateway (gateway/api/app.py), which is OpenAI-compatible,
through the same HTTPInferenceProvider used for any other endpoint.

The local UsageMeter here is a convenience for the user (and for
self-hosted setups); the authoritative spend cap is enforced server-side.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from gateway.billing.meter import UsageMeter
from gateway.providers.base import CloudInferenceProvider
from gateway.providers.http_provider import HTTPInferenceProvider
from shared.types import BackendResult, ExecutionBackend, Message

# Dollars per 1,000 tokens (prompt + completion). A placeholder until a
# wholesale provider and price are actually chosen (business decision,
# docs/planning/BUSINESS_BRIEF.md).
DEFAULT_PRICE_PER_1K_TOKENS_USD = 0.50


class CascadeCloudBackendError(RuntimeError):
    """CascadeCloud couldn't serve a request (spend cap hit or provider failure)."""


def estimate_prompt_tokens(messages: list[Message]) -> int:
    """Rough pre-call estimate (~1.3 tokens per word) used only to gate a spend cap before a billable call."""
    return int(sum(len(m["content"].split()) for m in messages) * 1.3) + 4 * len(messages)


class CascadeCloudBackend:
    def __init__(
        self,
        provider: Optional[CloudInferenceProvider] = None,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "",
        model_id: str = "cascade-hosted-default",
        price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS_USD,
        usage_meter: Optional[UsageMeter] = None,
        max_tokens: int = 512,
    ) -> None:
        self._provider: CloudInferenceProvider = provider or HTTPInferenceProvider(
            base_url=base_url, api_key=api_key, model_id=model_id
        )
        self._price_per_1k_tokens = price_per_1k_tokens
        self._usage_meter = usage_meter or UsageMeter()
        self._max_tokens = max_tokens

    def generate(self, messages: list[Message]) -> BackendResult:
        estimated_cost = self._estimate_cost(messages)
        self._usage_meter.check_and_reserve(estimated_cost)  # FR-07.2, before the billable call

        start = time.monotonic()
        try:
            sample = self._provider.generate(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            self._usage_meter.adjust(estimated_cost, 0.0)
            raise CascadeCloudBackendError(f"CascadeCloud provider call failed: {exc}") from exc
        wall_ms = (time.monotonic() - start) * 1000

        actual_cost = self._cost_for_tokens(sample.prompt_tokens + sample.completion_tokens)
        self._usage_meter.adjust(estimated_cost, actual_cost)
        tok_s = sample.completion_tokens / sample.elapsed_s if sample.elapsed_s > 0 else 0.0

        return {
            "text": sample.text,
            "confidence": 0.95,
            "speculative_active": False,
            "tok_s": round(tok_s, 3),
            "latency_ms": round(wall_ms, 2),
            "est_cost_usd": round(actual_cost, 6),
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        estimated_cost = self._estimate_cost(messages)
        self._usage_meter.check_and_reserve(estimated_cost)
        try:
            usage = yield from self._provider.generate_stream(list(messages), max_tokens=self._max_tokens)
        except Exception as exc:
            self._usage_meter.adjust(estimated_cost, 0.0)
            raise CascadeCloudBackendError(f"CascadeCloud provider stream failed: {exc}") from exc
        # Reconcile to the real usage when the server reported it;
        # otherwise the (worst-case) reservation stands.
        if usage is not None:
            self._usage_meter.adjust(estimated_cost, self._cost_for_tokens(usage.total_tokens))

    def _estimate_cost(self, messages: list[Message]) -> float:
        return self._cost_for_tokens(estimate_prompt_tokens(messages) + self._max_tokens)

    def _cost_for_tokens(self, total_tokens: int) -> float:
        return (total_tokens / 1000.0) * self._price_per_1k_tokens


assert isinstance(CascadeCloudBackend(), ExecutionBackend)
