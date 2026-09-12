"""
gateway/tests/test_stripe_client.py -- Track C, Task 8 (FR-07.5).

Monkeypatches the `stripe` module itself so this exercises
StripeBillingClient's real wiring logic with zero real Stripe API key
or network call (NFR-09).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gateway.billing.stripe_client import BillingClient, StripeBillingClient


class _FakeMeterEvent:
    created: list[dict] = []

    @classmethod
    def create(cls, event_name: str, payload: dict) -> Any:
        cls.created.append({"event_name": event_name, "payload": payload})
        return SimpleNamespace(id="evt_fake")


class _FakeSubscription:
    created: list[dict] = []

    @classmethod
    def create(cls, customer: str, items: list) -> Any:
        cls.created.append({"customer": customer, "items": items})
        return SimpleNamespace(id="sub_fake_123")


class _FakeStripeModule:
    api_key: str = ""
    billing = SimpleNamespace(MeterEvent=_FakeMeterEvent)
    Subscription = _FakeSubscription


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeMeterEvent.created = []
    _FakeSubscription.created = []
    yield


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> StripeBillingClient:
    import sys

    monkeypatch.setitem(sys.modules, "stripe", _FakeStripeModule)
    return StripeBillingClient(api_key="sk_test_fake")


def test_construction_sets_api_key_on_stripe_module(client: StripeBillingClient) -> None:
    assert _FakeStripeModule.api_key == "sk_test_fake"


def test_record_usage_calls_meter_event_create(client: StripeBillingClient) -> None:
    client.record_usage(customer_id="cus_123", meter_event_name="cascade_tokens", quantity=500)

    assert len(_FakeMeterEvent.created) == 1
    call = _FakeMeterEvent.created[0]
    assert call["event_name"] == "cascade_tokens"
    assert call["payload"]["stripe_customer_id"] == "cus_123"
    assert call["payload"]["value"] == "500"


def test_create_subscription_returns_subscription_id(client: StripeBillingClient) -> None:
    sub_id = client.create_subscription(customer_id="cus_123", price_id="price_abc")

    assert sub_id == "sub_fake_123"
    assert _FakeSubscription.created[0]["customer"] == "cus_123"
    assert _FakeSubscription.created[0]["items"] == [{"price": "price_abc"}]


def test_stripe_billing_client_conforms_to_protocol_shape(client: StripeBillingClient) -> None:
    # Structural check -- both required methods exist and are callable.
    assert callable(client.record_usage)
    assert callable(client.create_subscription)
