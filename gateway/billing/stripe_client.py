"""
gateway/billing/stripe_client.py

Track C, Task 8 (FR-07.5).

Metered usage billing + subscriptions through Stripe, with no plaintext
storage of payment details in Cascade's own systems -- Stripe holds the
card, Cascade only ever holds a customer/subscription ID. Follows the
same injectable-dependency pattern as Track A's ReferenceModelRunner
and Track C's own CloudInferenceProvider: a Protocol, a real
Stripe-backed implementation, and a fake for tests (NFR-09 -- no test
should need a real Stripe API key or network call).
"""

from __future__ import annotations

from typing import Protocol


class BillingClient(Protocol):
    def record_usage(self, customer_id: str, meter_event_name: str, quantity: int) -> None: ...

    def create_subscription(self, customer_id: str, price_id: str) -> str: ...


class StripeBillingClient:
    """Production implementation. Requires a real Stripe secret key -- never logged, never written to shared/CHANGELOG.md or any repo file."""

    def __init__(self, api_key: str) -> None:
        import stripe  # deferred: heavy import, and importing this module must never require a key to exist

        stripe.api_key = api_key
        self._stripe = stripe

    def record_usage(self, customer_id: str, meter_event_name: str, quantity: int) -> None:
        self._stripe.billing.MeterEvent.create(
            event_name=meter_event_name,
            payload={"stripe_customer_id": customer_id, "value": str(quantity)},
        )

    def create_subscription(self, customer_id: str, price_id: str) -> str:
        subscription = self._stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
        )
        return subscription.id
