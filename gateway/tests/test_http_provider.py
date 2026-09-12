"""
gateway/tests/test_http_provider.py -- Track C, Task 7 (FR-07.1).

Uses httpx.MockTransport so these tests exercise the real request/
response handling code with zero actual network calls (NFR-09).
"""

from __future__ import annotations

import json

import httpx
import pytest

from gateway.providers.base import CloudGenerationSample
from gateway.providers.http_provider import HTTPInferenceProvider


def _chat_completion_response(text: str, prompt_tokens: int = 8, completion_tokens: int = 12) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _provider_with_mock(handler) -> HTTPInferenceProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return HTTPInferenceProvider(
        base_url="http://fake-vllm-endpoint:8000",
        api_key="test-key",
        model_id="test-model",
        client=client,
    )


def test_generate_parses_openai_compatible_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_response("hello from vllm"))

    provider = _provider_with_mock(handler)
    sample = provider.generate("hi", max_tokens=50)

    assert isinstance(sample, CloudGenerationSample)
    assert sample.text == "hello from vllm"
    assert sample.prompt_tokens == 8
    assert sample.completion_tokens == 12


def test_generate_sends_correct_request_shape() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion_response("ok"))

    provider = _provider_with_mock(handler)
    provider.generate("test prompt", max_tokens=99)

    assert captured["url"] == "http://fake-vllm-endpoint:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["max_tokens"] == 99
    assert captured["body"]["messages"] == [{"role": "user", "content": "test prompt"}]
    assert captured["body"]["stream"] is False


def test_generate_raises_on_http_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal server error"})

    provider = _provider_with_mock(handler)
    with pytest.raises(httpx.HTTPStatusError):
        provider.generate("hi", max_tokens=50)


def test_generate_stream_yields_delta_chunks() -> None:
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"hello "}}]}',
        'data: {"choices":[{"delta":{"content":"world"}}]}',
        "data: [DONE]",
    ]
    sse_body = "\n".join(sse_lines) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    provider = _provider_with_mock(handler)
    chunks = list(provider.generate_stream("hi", max_tokens=50))

    assert chunks == ["hello ", "world"]
