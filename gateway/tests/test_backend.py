"""gateway/tests/test_backend.py -- SelfHostedBackend, the client of the self_hosted tier."""

from __future__ import annotations

import pytest

from gateway.backend import SelfHostedBackend, SelfHostedBackendError
from gateway.tests.fakes import FakeCloudInferenceProvider
from shared.types import ExecutionBackend, Message


def _messages() -> list[Message]:
    return [{"role": "user", "content": "hello self-hosted server"}]


def test_backend_conforms_to_execution_backend_protocol() -> None:
    assert isinstance(SelfHostedBackend(provider=FakeCloudInferenceProvider()), ExecutionBackend)


def test_generate_returns_well_typed_backend_result() -> None:
    result = SelfHostedBackend(provider=FakeCloudInferenceProvider(text="a real answer")).generate(_messages())
    for field in ("text", "confidence", "speculative_active", "tok_s", "latency_ms", "est_cost_usd"):
        assert field in result
    assert result["text"] == "a real answer"
    assert result["speculative_active"] is False
    assert result["est_cost_usd"] == 0.0


def test_generate_reports_upstream_throughput() -> None:
    provider = FakeCloudInferenceProvider(completion_tokens=20, elapsed_s=0.5)
    assert SelfHostedBackend(provider=provider).generate(_messages())["tok_s"] == pytest.approx(40.0)


def test_generate_wraps_provider_failure() -> None:
    with pytest.raises(SelfHostedBackendError):
        SelfHostedBackend(provider=FakeCloudInferenceProvider(fail=True)).generate(_messages())


def test_generate_stream_yields_tokens() -> None:
    backend = SelfHostedBackend(provider=FakeCloudInferenceProvider(text="one two three"))
    assert "".join(backend.generate_stream(_messages())).strip() == "one two three"


def test_generate_stream_wraps_provider_failure() -> None:
    with pytest.raises(SelfHostedBackendError):
        list(SelfHostedBackend(provider=FakeCloudInferenceProvider(fail=True)).generate_stream(_messages()))


def test_generate_passes_messages_through_unflattened() -> None:
    provider = FakeCloudInferenceProvider()
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    SelfHostedBackend(provider=provider).generate(messages)
    assert provider.last_messages == messages


def test_default_construction_does_not_touch_network() -> None:
    """Constructing with all defaults (as the module-level isinstance assert does) must never make an HTTP call."""
    assert isinstance(SelfHostedBackend(), ExecutionBackend)
