"""
gateway/tests/fakes.py

Shared test doubles for Track C's test suites. Mirrors Track A's
FakeReferenceModelRunner and Track B's FakeEmbedder pattern:
deterministic, no live network/GPU/Stripe call (NFR-09).
"""

from __future__ import annotations

from typing import Generator, Optional

from gateway.billing.stripe_client import BillingClient
from gateway.providers.base import CloudGenerationSample, CloudUsage


class FakeCloudInferenceProvider:
    """Conforms to CloudInferenceProvider. Returns a canned, well-typed sample."""

    def __init__(
        self,
        text: str = "a fake cloud response",
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
        elapsed_s: float = 0.5,
        fail: bool = False,
        report_stream_usage: bool = True,
    ) -> None:
        self._report_stream_usage = report_stream_usage
        self.last_messages: list[dict] = []
        self._text = text
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._elapsed_s = elapsed_s
        self._fail = fail
        self.call_count = 0

    def generate(self, messages: list[dict], max_tokens: int) -> CloudGenerationSample:
        self.call_count += 1
        self.last_messages = messages
        if self._fail:
            raise RuntimeError("simulated provider failure")
        return CloudGenerationSample(
            text=self._text,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            elapsed_s=self._elapsed_s,
        )

    def generate_stream(self, messages: list[dict], max_tokens: int) -> Generator[str, None, Optional[CloudUsage]]:
        self.call_count += 1
        self.last_messages = messages
        if self._fail:
            raise RuntimeError("simulated provider failure")
        for word in self._text.split():
            yield word + " "
        return CloudUsage(self._prompt_tokens, self._completion_tokens) if self._report_stream_usage else None


class FakeBillingClient:
    """Conforms to BillingClient. Records calls in memory instead of hitting Stripe's API."""

    def __init__(self) -> None:
        self.usage_records: list[tuple[str, str, int]] = []
        self.subscriptions: list[tuple[str, str]] = []

    def record_usage(self, customer_id: str, meter_event_name: str, quantity: int) -> None:
        self.usage_records.append((customer_id, meter_event_name, quantity))

    def create_subscription(self, customer_id: str, price_id: str) -> str:
        self.subscriptions.append((customer_id, price_id))
        return f"sub_fake_{len(self.subscriptions)}"


# Static type-check aid -- confirms FakeBillingClient actually satisfies
# the Protocol shape, the same spirit as the isinstance() asserts on
# the real backend classes.
def _conforms_to_billing_client(client: BillingClient) -> None:
    pass


_conforms_to_billing_client(FakeBillingClient())
