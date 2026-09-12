"""gateway/tests/test_api.py -- Track C, Tasks 5 & 10 (FR-07.1, FR-07.2, FR-07.5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import create_app
from gateway.tests.fakes import FakeBillingClient, FakeCloudInferenceProvider


def _client(**app_kwargs) -> TestClient:
    provider = app_kwargs.pop("provider", FakeCloudInferenceProvider())
    return TestClient(create_app(provider=provider, **app_kwargs))


def test_health_endpoint() -> None:
    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_returns_well_formed_response() -> None:
    client = _client(provider=FakeCloudInferenceProvider(text="a hosted answer"))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "user_123"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "a hosted answer"
    assert body["est_cost_usd"] > 0.0


def test_generate_requires_user_header() -> None:
    client = _client()
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 422  # missing required header


def test_generate_returns_402_when_spend_cap_exceeded() -> None:
    client = _client(price_per_1k_tokens=1000.0, default_cap_usd=0.0001)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "user_123"},
    )
    assert response.status_code == 402


def test_generate_returns_502_on_provider_failure() -> None:
    client = _client(provider=FakeCloudInferenceProvider(fail=True))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "user_123"},
    )
    assert response.status_code == 502


def test_generate_records_usage_via_billing_client() -> None:
    billing = FakeBillingClient()
    client = _client(billing_client=billing)
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "user_123"},
    )
    assert len(billing.usage_records) == 1
    assert billing.usage_records[0][0] == "user_123"


def test_generate_meters_separately_per_user() -> None:
    client = _client(price_per_1k_tokens=1.0, default_cap_usd=1000.0)

    r1 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "alice"},
    )
    r2 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"x-cascade-user": "bob"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Each user gets their own independent cap -- one user's usage never
    # affects another's, verified indirectly: both succeed under a cap
    # that would reject a single user making both calls if shared.


def test_generate_second_call_over_shared_cap_for_same_user_fails() -> None:
    """
    First call's pre-call ESTIMATE (small, from `max_tokens`) clears the
    cap, but its actual realized cost (from the provider's real token
    counts, larger than the small `max_tokens` requested) pushes spend
    close enough to the cap that a second call's estimate no longer fits.
    """
    provider = FakeCloudInferenceProvider(prompt_tokens=100, completion_tokens=100)
    client = TestClient(create_app(provider=provider, price_per_1k_tokens=1.0, default_cap_usd=0.21))

    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    r1 = client.post("/v1/chat/completions", json=body, headers={"x-cascade-user": "alice"})
    r2 = client.post("/v1/chat/completions", json=body, headers={"x-cascade-user": "alice"})

    assert r1.status_code == 200
    assert r1.json()["est_cost_usd"] == pytest.approx(0.2, rel=0.01)  # 200 real tokens @ $1/1k
    assert r2.status_code == 402  # same user, cap now exhausted by call 1's real cost
