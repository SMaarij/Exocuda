"""gateway/tests/test_byok.py -- the cloud_byok tier."""

from __future__ import annotations

import pytest

from gateway.byok import CloudBYOKBackend, CloudBYOKBackendError
from gateway.tests.fakes import FakeCloudInferenceProvider
from shared.types import ExecutionBackend

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "prove sqrt(2) is irrational"}]


def test_conforms_to_execution_backend() -> None:
    assert isinstance(CloudBYOKBackend(provider=FakeCloudInferenceProvider()), ExecutionBackend)


def test_generate_returns_provider_text_and_passes_messages_through() -> None:
    provider = FakeCloudInferenceProvider(text="because...")
    result = CloudBYOKBackend(provider=provider).generate(MESSAGES)
    assert result["text"] == "because..."
    assert result["speculative_active"] is False
    assert provider.last_messages == MESSAGES


def test_cost_uses_the_users_configured_prices() -> None:
    provider = FakeCloudInferenceProvider(prompt_tokens=1000, completion_tokens=2000)
    backend = CloudBYOKBackend(provider=provider, input_price_per_1k=0.15, output_price_per_1k=0.60)
    assert backend.generate(MESSAGES)["est_cost_usd"] == pytest.approx(0.15 + 1.20)


def test_cost_is_zero_without_configured_prices() -> None:
    assert CloudBYOKBackend(provider=FakeCloudInferenceProvider()).generate(MESSAGES)["est_cost_usd"] == 0.0


def test_provider_failure_raises_byok_error() -> None:
    backend = CloudBYOKBackend(provider=FakeCloudInferenceProvider(fail=True))
    with pytest.raises(CloudBYOKBackendError):
        backend.generate(MESSAGES)
    with pytest.raises(CloudBYOKBackendError):
        list(backend.generate_stream(MESSAGES))


def test_stream_yields_text() -> None:
    backend = CloudBYOKBackend(provider=FakeCloudInferenceProvider(text="a b c"))
    assert "".join(backend.generate_stream(MESSAGES)).strip() == "a b c"
