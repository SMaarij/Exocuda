"""
gateway/backend.py

Track C, Tasks 4 & 7 (FR-07.1, FR-07.2, FR-07.6, FR-07.7).

CascadeCloudBackend: conforms to shared.types.ExecutionBackend exactly
like Track A's LocalExecutionBackend does, so Track B's Pipeline can
call either without knowing which one it's talking to (NFR-07). This
is the direct implementation of the interface Track A's and Track B's
handoff docs describe.

                    ┌──────────────────────────────────────┐
                    │  J-4 CONNECT HERE                     │
                    │  Wire this into Pipeline exactly as    │
                    │  docs/TRACK_B_HANDOFF_FOR_C.md          │
                    │  Section 4, Step 2 shows:                │
                    │    Pipeline(backend=LocalExecutionBackend(),│
                    │             cascade_backend=CascadeCloudBackend(),│
                    │             ...)                            │
                    │  Verification: the manual smoke test in      │
                    │  that doc's Section 4, Step 3.                │
                    └──────────────────────────────────────┘
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from gateway.billing.meter import SpendCapExceededError, UsageMeter
from gateway.providers.base import CloudInferenceProvider
from gateway.providers.http_provider import HTTPInferenceProvider
from shared.types import BackendResult, ExecutionBackend, Message

# Cost model: dollars per 1,000 tokens (prompt + completion combined).
# PENDING, exactly like Track A's DEFAULT_REFERENCE_MODEL_ID and
# acceleration-layer speedup constants: a real number depends on which
# wholesale GPU provider and model Cascade actually contracts, which is
# a joint business decision (see docs/planning/BUSINESS_BRIEF.md),
# not something to hardcode unilaterally here. This default is
# deliberately a round, clearly-a-placeholder number.
DEFAULT_PRICE_PER_1K_TOKENS_USD = 0.50


class CascadeCloudBackendError(RuntimeError):
    """
    Raised when CascadeCloud cannot serve a request -- a spend cap was
    hit, or the underlying provider call failed. Mirrors
    LocalExecutionBackendError's role: Pipeline's confidence/escalation
    logic doesn't reach into gateway/ to catch this specifically (it
    only ever catches LocalExecutionBackendError, per Section 9) --
    this exists so gateway/api/ can catch it and return a clean HTTP
    error instead of a raw traceback.
    """


def _messages_to_prompt(messages: list[Message]) -> str:
    """Mirrors engine/execution/backend.py's _messages_to_prompt exactly -- same flattening convention on both ExecutionBackend implementations."""
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)


class CascadeCloudBackend:
    def __init__(
        self,
        provider: Optional[CloudInferenceProvider] = None,
        base_url: str = "http://localhost:8000",
        api_key: str = "",
        model_id: str = "cascade-hosted-default",
        price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS_USD,
        usage_meter: Optional[UsageMeter] = None,
        max_tokens: int = 256,
    ) -> None:
        self._provider: CloudInferenceProvider = provider or HTTPInferenceProvider(
            base_url=base_url, api_key=api_key, model_id=model_id
        )
        self._price_per_1k_tokens = price_per_1k_tokens
        self._usage_meter = usage_meter or UsageMeter()
        self._max_tokens = max_tokens

    # ---- ExecutionBackend protocol implementation ------------------------

    def generate(self, messages: list[Message]) -> BackendResult:
        prompt = _messages_to_prompt(messages)
        estimated_cost = self._estimate_cost(prompt)
        self._usage_meter.check_and_reserve(estimated_cost)  # FR-07.2, before the billable call

        start = time.monotonic()
        try:
            sample = self._provider.generate(prompt, max_tokens=self._max_tokens)
        except Exception as exc:
            # The reservation from check_and_reserve() above still holds
            # even though the call failed -- reconcile it down to zero
            # extra cost rather than leaving a phantom reservation.
            self._usage_meter.adjust(estimated_cost, 0.0)
            raise CascadeCloudBackendError(f"CascadeCloud provider call failed: {exc}") from exc
        wall_ms = (time.monotonic() - start) * 1000

        total_tokens = sample.prompt_tokens + sample.completion_tokens
        actual_cost = self._cost_for_tokens(total_tokens)
        self._usage_meter.adjust(estimated_cost, actual_cost)

        tok_s = sample.completion_tokens / sample.elapsed_s if sample.elapsed_s > 0 else 0.0

        return {
            "text": sample.text,
            "confidence": 0.95,  # no speculative layer in the cloud tier -- see docs/TRACK_A_HANDOFF_FOR_B_AND_C.md Section 1's field table
            "speculative_active": False,
            "tok_s": round(tok_s, 3),
            "latency_ms": round(wall_ms, 2),
            "est_cost_usd": round(actual_cost, 6),
        }

    def generate_stream(self, messages: list[Message]) -> Iterator[str]:
        prompt = _messages_to_prompt(messages)
        estimated_cost = self._estimate_cost(prompt)
        self._usage_meter.check_and_reserve(estimated_cost)

        try:
            for chunk in self._provider.generate_stream(prompt, max_tokens=self._max_tokens):
                yield chunk
        except Exception as exc:
            self._usage_meter.adjust(estimated_cost, 0.0)
            raise CascadeCloudBackendError(f"CascadeCloud provider stream failed: {exc}") from exc
        # No further adjust() call on success: the reservation from
        # check_and_reserve() above stands as the final charge for a
        # streamed request. Streaming responses don't report exact
        # token usage back from this loop, so reconciling down to a
        # real actual_cost isn't possible here -- a documented follow-up
        # once a provider's streaming usage payload is wired in.

    # ---- Cost model --------------------------------------------------------

    def _estimate_cost(self, prompt: str) -> float:
        # Rough pre-call estimate (prompt tokens ~ words * 1.3, plus the
        # worst-case max_tokens) purely to gate the spend cap check
        # before a billable call happens; _cost_for_tokens() below with
        # the provider's REAL token count is what actually gets billed.
        estimated_prompt_tokens = int(len(prompt.split()) * 1.3)
        return self._cost_for_tokens(estimated_prompt_tokens + self._max_tokens)

    def _cost_for_tokens(self, total_tokens: int) -> float:
        return (total_tokens / 1000.0) * self._price_per_1k_tokens


# Fails immediately at import time if the class ever drifts from the
# ExecutionBackend protocol -- same guard Track A uses on
# LocalExecutionBackend (engine/execution/backend.py).
assert isinstance(CascadeCloudBackend(), ExecutionBackend)
