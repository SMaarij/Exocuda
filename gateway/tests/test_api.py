"""gateway/tests/test_api.py -- FastAPI TestClient, no network (NFR-09)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gateway.api.app import create_app
from gateway.backend import SelfHostedBackend
from gateway.providers.http_provider import HTTPInferenceProvider
from gateway.store import GatewayStore
from gateway.tests.fakes import FakeCloudInferenceProvider

BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}


def _setup(provider=None, quota=None, **app_kwargs):
    store = GatewayStore()
    store.create_user("alice", quota_tokens=quota)
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


def test_returns_429_over_quota_without_calling_upstream() -> None:
    provider = FakeCloudInferenceProvider()
    client, _, headers = _setup(provider, quota=10)
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 429
    assert provider.call_count == 0


def test_returns_502_and_releases_reservation_on_upstream_failure() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(fail=True))
    assert client.post("/v1/chat/completions", json=BODY, headers=headers).status_code == 502
    assert store.account("alice").used_tokens == 0


def test_records_real_usage_not_the_reservation() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=100, completion_tokens=100))
    client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert store.account("alice").used_tokens == 200


def test_second_request_over_quota_fails() -> None:
    client, _, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=100, completion_tokens=100), quota=210)  # 200 used, next reserves 15
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10}
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 200
    assert client.post("/v1/chat/completions", json=body, headers=headers).status_code == 429


def test_usage_endpoint_reports_tokens() -> None:
    client, _, headers = _setup(FakeCloudInferenceProvider(prompt_tokens=500, completion_tokens=500), quota=5000)
    client.post("/v1/chat/completions", json=BODY, headers=headers)
    assert client.get("/v1/usage", headers=headers).json() == {"user_id": "alice", "used_tokens": 1000, "quota_tokens": 5000}


def test_streaming_yields_openai_chunks_and_usage() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(text="one two", prompt_tokens=5, completion_tokens=5))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True, "stream_options": {"include_usage": True}}, headers=headers)
    events = _sse_events(response.text)
    deltas = [e["choices"][0]["delta"].get("content") for e in events[:-1] if isinstance(e, dict) and e["choices"]]
    assert "".join(d for d in deltas if d) == "one two "
    usage_event = events[-2]
    assert usage_event["usage"] == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    assert events[-1] == "[DONE]"
    assert store.account("alice").used_tokens == 10


def test_streaming_upstream_failure_emits_error_and_releases_reservation() -> None:
    client, store, headers = _setup(FakeCloudInferenceProvider(fail=True))
    events = _sse_events(client.post("/v1/chat/completions", json={**BODY, "stream": True}, headers=headers).text)
    assert "error" in events[0]
    assert store.account("alice").used_tokens == 0


def test_self_hosted_backend_talks_to_the_real_server() -> None:
    """The CLI's client and the server speak the same protocol, end to end."""
    client, store, headers = _setup(FakeCloudInferenceProvider(text="served by the server", prompt_tokens=10, completion_tokens=10))
    key = headers["Authorization"].removeprefix("Bearer ")
    provider = HTTPInferenceProvider(base_url="http://testserver", api_key=key, model_id="cascade", client=client)
    backend = SelfHostedBackend(provider=provider)

    result = backend.generate([{"role": "user", "content": "hello"}])
    assert result["text"] == "served by the server"
    assert result["est_cost_usd"] == 0.0

    assert "".join(backend.generate_stream([{"role": "user", "content": "hello"}])).strip() == "served by the server"
    assert store.account("alice").used_tokens == 40
