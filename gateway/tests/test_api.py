"""gateway/tests/test_api.py -- FastAPI TestClient, no network/Stripe (NFR-09)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import create_app
from gateway.backend import CascadeCloudBackend
from gateway.providers.http_provider import HTTPInferenceProvider
from gateway.store import GatewayStore
from gateway.tests.fakes import FakeBillingClient, FakeCloudInferenceProvider

BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}


def _setup(provider=None, cap=None, **app_kwargs):
    store = GatewayStore()
    store.create_account("alice", cap_usd=cap)
    key = store.create_api_key("alice")
    app = create_app(provider=provider or FakeCloudInferenceProvider(), store=store, **app_kwargs)
    return TestClient(app), store, {"Authorization": f"Bearer {key}"}


def _sse_events(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[len("data: "):]
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_health_endpoint() -> None:
    client, _, _ = _setup()
    assert client.get("/health").json() == {"status": "ok"}


def test_returns_openai_compatible_completion() -> None:
    client, _, headers = _setup(FakeCloudInferenceProvider(text="hi there", prompt_tokens=10, completion_tokens=20))
    body = client.post("/v1/chat/completions", json=BODY, headers=headers).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hi there"}
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer csk_not-a-real-key"}, {"x-cascade-user": "alice"}])
def test_rejects_missing_or_invalid_key(headers) -> None:
    provider = FakeCloudInferenceProvider()
    client, _, _ = _setup(provider)
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 401
    assert provider.call_count == 0


def test_revoked_key_is_rejected() -> None:
    client, store, headers = _setup()
    store.revoke_api_key(headers["Authorization"].removeprefix("Bearer "))
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 401


def test_rejects_invalid_role_and_empty_messages() -> None:
    client, _, headers = _setup()
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "root", "content": "x"}]}, headers=headers).status_code == 422
    assert client.post("/v1/chat/completions", json={"messages": []}, headers=headers).status_code == 422


def test_rejects_max_tokens_over_limit() -> None:
    client, _, headers = _setup(max_tokens_limit=100)
    assert client.post("/v1/chat/completions", json={**BODY, "max_tokens": 101}, headers=headers).status_code == 400


def test_returns_402_when_spend_cap_exceeded_without_calling_upstream() -> None:
    provider = FakeCloudInferenceProvider()
    client, _, headers = _setup(provider, cap=0.0001, price_per_1k_tokens=100.0)
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 402
    assert provider.call_count == 0


def test_returns_502_and_releases_reservation_on_upstream_failure() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(fail=True))
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 502
    assert store.account("alice").spent_usd == pytest.approx(0.0)


def test_charges_real_usage_and_reports_to_billing() -> None:
    billing = FakeBillingClient()
    client, store, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=100, completion_tokens=100), billing_client=billing, price_per_1k_tokens=1.0)
    body = client.post("/v1/chat/completions", json=BODY, headers=headers).json()
    assert body["cascade"]["cost_usd"] == pytest.approx(0.2)
    assert store.account("alice").spent_usd == pytest.approx(0.2)
    assert billing.usage_records == [("alice", "cascade_tokens", 200)]


def test_billing_uses_stripe_customer_id_when_set() -> None:
    billing = FakeBillingClient()
    client, store, headers = _setup(billing_client=billing)
    store.create_account("alice", stripe_customer_id="cus_123")
    client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert billing.usage_records[0][0] == "cus_123"


def test_second_request_over_cap_fails() -> None:
    client, _, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=100, completion_tokens=100), cap=0.21, price_per_1k_tokens=1.0)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10}
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 200
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 402


def test_usage_endpoint_reports_spend() -> None:
    client, _, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=500, completion_tokens=500), cap=5.0, price_per_1k_tokens=1.0)
    client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert client.get("/v1/usage", headers=headers).json() == {"user_id": "alice", "spent_usd": 1.0, "cap_usd": 5.0}


def test_streaming_yields_openai_chunks_and_usage() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(text="one two", prompt_tokens=5, completion_tokens=5), price_per_1k_tokens=1.0)
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True, "stream_options": {"include_usage": True}}, headers=headers)
    events = _sse_events(response.text)
    deltas = [e["choices"][0]["delta"].get("content") for e in events[:-1] if isinstance(e, dict) and e["choices"]]
    assert "".join(d for d in deltas if d) == "one two "
    usage_event = events[-2]
    assert usage_event["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    assert events[-1] == "[DONE]"
    assert store.account("alice").spent_usd == pytest.approx(0.01)


def test_streaming_upstream_failure_emits_error_and_releases_reservation() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(fail=True))
    events = _sse_events(client.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=headers).text)
    assert "error" in events[0]
    assert store.account("alice").spent_usd == pytest.approx(0.0)


def test_cascade_cloud_backend_talks_to_the_real_gateway() -> None:
    """The client and server speak the same protocol -- this was broken before (different response shapes)."""
    client, store, headers = _setup(FakeCloudInferenceProvider(text="served by gateway", prompt_tokens=10, completion_tokens=10), price_per_1k_tokens=1.0)
    key = headers["Authorization"].removeprefix("Bearer ")
    provider = HTTPInferenceProvider(base_url="http://testserver", api_key=key, model_id="cascade", client=client)
    backend = CascadeCloudBackend(provider=provider, price_per_1k_tokens=1.0)

    result = backend.generate([{"role": "user", "content": "hello"}])
    assert result["text"] == "served by gateway"
    assert result["est_cost_usd"] == pytest.approx(0.02)

    assert "".join(backend.generate_stream([{"role": "user", "content": "hello"}])).strip() == "served by gateway"
    assert store.account("alice").spent_usd == pytest.approx(0.04)
