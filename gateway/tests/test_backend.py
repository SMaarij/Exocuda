"""gateway/tests/test_backend.py -- Track C, Tasks 4 & 7 (FR-07.1, FR-07.2)."""

from __future__ import annotations

import pytest

from gateway.backend import CascadeCloudBackend, CascadeCloudBackendError
from gateway.billing.meter import SpendCapExceededError, UsageMeter
from gateway.tests.fakes import FakeCloudInferenceProvider
from shared.types import ExecutionBackend, Message


def _messages() -> list[Message]:
    return [{"role": "user", "content": "hello cascade cloud"}]


def test_backend_conforms_to_execution_backend_protocol() -> None:
    backend = CascadeCloudBackend(provider=FakeCloudInferenceProvider())
    assert isinstance(backend, ExecutionBackend)


def test_generate_returns_well_typed_backend_result() -> None:
    backend = CascadeCloudBackend(provider=FakeCloudInferenceProvider(text="a real answer"))
    result = backend.generate(_messages())

    for field in ("text", "confidence", "speculative_active", "tok_s", "latency_ms", "est_cost_usd"):
        assert field in result
    assert result["text"] == "a real answer"
    assert result["speculative_active"] is False  # no speculative layer in the cloud tier


def test_generate_cost_is_greater_than_zero() -> None:
    backend = CascadeCloudBackend(provider=FakeCloudInferenceProvider(), price_per_1k_tokens=1.0)
    result = backend.generate(_messages())
    assert result["est_cost_usd"] > 0.0


def test_generate_charges_for_actual_tokens_not_the_estimate() -> None:
    provider = FakeCloudInferenceProvider(prompt_tokens=5, completion_tokens=5)
    meter = UsageMeter(cap_usd=None)
    backend = CascadeCloudBackend(provider=provider, price_per_1k_tokens=10.0, usage_meter=meter)

    backend.generate(_messages())

    expected_cost = (10 / 1000.0) * 10.0  # 10 actual tokens (5 prompt + 5 completion)
    assert meter.spent_usd == pytest.approx(expected_cost, rel=0.01)


def test_generate_raises_spend_cap_exceeded_before_calling_provider() -> None:
    provider = FakeCloudInferenceProvider()
    meter = UsageMeter(cap_usd=0.0001)  # effectively zero budget
    backend = CascadeCloudBackend(provider=provider, price_per_1k_tokens=100.0, usage_meter=meter)

    with pytest.raises(SpendCapExceededError):
        backend.generate(_messages())

    assert provider.call_count == 0  # never called -- the cap was checked first


def test_generate_releases_reservation_on_provider_failure() -> None:
    provider = FakeCloudInferenceProvider(fail=True)
    meter = UsageMeter(cap_usd=100.0)
    backend = CascadeCloudBackend(provider=provider, usage_meter=meter)

    with pytest.raises(CascadeCloudBackendError):
        backend.generate(_messages())

    assert meter.spent_usd == 0.0  # reservation released, not stuck


def test_generate_stream_yields_tokens() -> None:
    backend = CascadeCloudBackend(provider=FakeCloudInferenceProvider(text="one two three"))
    tokens = list(backend.generate_stream(_messages()))
    assert len(tokens) >= 1
    assert "".join(tokens).strip() == "one two three"


def test_generate_stream_releases_reservation_on_failure() -> None:
    provider = FakeCloudInferenceProvider(fail=True)
    meter = UsageMeter(cap_usd=100.0)
    backend = CascadeCloudBackend(provider=provider, usage_meter=meter)

    with pytest.raises(CascadeCloudBackendError):
        list(backend.generate_stream(_messages()))

    assert meter.spent_usd == 0.0


def test_default_construction_does_not_touch_network() -> None:
    """Constructing with all defaults (as the module-level isinstance assert does) must never make an HTTP call."""
    backend = CascadeCloudBackend()
    assert isinstance(backend, ExecutionBackend)
