"""gateway/tests/test_http_provider.py -- httpx.MockTransport, no network (NFR-09)."""

from __future__ import annotations

import json

import httpx
import pytest

from gateway.providers.base import CloudGenerationSample, CloudUsage
from gateway.providers.http_provider import HTTPInferenceProvider, ProviderError

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def _chat_completion_response(text: str, prompt_tokens: int = 8, completion_tokens: int = 12) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _provider_with_mock(handler, api_key: str = "test-key") -> HTTPInferenceProvider:
    return HTTPInferenceProvider(
        base_url="http://fake-vllm-endpoint:8000/",
        api_key=api_key,
        model_id="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _drain(gen):
    chunks = []
    while True:
        try:
            chunks.append(next(gen))
        except StopIteration as stop:
            return chunks, stop.value


def test_generate_parses_openai_compatible_response() -> None:
    provider = _provider_with_mock(lambda r: httpx.Response(200, json=_chat_completion_response("hello from vllm")))
    sample = provider.generate(MESSAGES, max_tokens=50)
    assert isinstance(sample, CloudGenerationSample)
    assert (sample.text, sample.prompt_tokens, sample.completion_tokens) == ("hello from vllm", 8, 12)


def test_generate_sends_messages_unflattened() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion_response("ok"))

    _provider_with_mock(handler).generate(MESSAGES, max_tokens=99)

    assert captured["url"] == "http://fake-vllm-endpoint:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"] == {"model": "test-model", "messages": MESSAGES, "max_tokens": 99, "stream": False}


def test_no_auth_header_without_a_key() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=_chat_completion_response("ok"))

    _provider_with_mock(handler, api_key="").generate(MESSAGES, max_tokens=5)
    assert captured["auth"] is None


def test_generate_raises_provider_error_with_body_on_http_error() -> None:
    provider = _provider_with_mock(lambda r: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(ProviderError, match="401.*bad key"):
        provider.generate(MESSAGES, max_tokens=50)


def test_generate_stream_yields_deltas_and_returns_reported_usage() -> None:
    captured = {}
    sse = "\n".join([
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ]) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    chunks, usage = _drain(_provider_with_mock(handler).generate_stream(MESSAGES, max_tokens=50))
    assert chunks == ["hello ", "world"]
    assert usage == CloudUsage(7, 2)
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_generate_stream_returns_none_when_usage_not_reported() -> None:
    sse = 'data: {"choices":[{"delta":{"content":"x"}}]}\ndata: [DONE]\n'
    provider = _provider_with_mock(lambda r: httpx.Response(200, content=sse))
    assert _drain(provider.generate_stream(MESSAGES, max_tokens=5)) == (["x"], None)


def test_generate_stream_raises_provider_error_on_http_error() -> None:
    provider = _provider_with_mock(lambda r: httpx.Response(503, content=b"overloaded"))
    with pytest.raises(ProviderError, match="503"):
        list(provider.generate_stream(MESSAGES, max_tokens=5))
