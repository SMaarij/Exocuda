"""
gateway/providers/http_provider.py

Production CloudInferenceProvider: any OpenAI-compatible chat-completions
endpoint. `base_url` is the server root without `/v1` (e.g.
https://api.openai.com, https://openrouter.ai/api, http://127.0.0.1:11434
for Ollama, or a vLLM/SGLang deployment).

FR-07.6/07.7 (continuous batching, PagedAttention, prefix caching) are
properties of the serving engine behind the endpoint, not of this client.
"""

from __future__ import annotations

import json
import time
from typing import Generator, Optional

import httpx

from gateway.providers.base import ChatMessages, CloudGenerationSample, CloudUsage


class ProviderError(RuntimeError):
    """The upstream endpoint rejected or failed a request."""


class HTTPInferenceProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_s: float = 120.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                body = response.read().decode("utf-8", errors="replace")
            except httpx.HTTPError:
                body = ""
            raise ProviderError(f"upstream returned {response.status_code}: {body[:300]}")

    def generate(self, messages: ChatMessages, max_tokens: int) -> CloudGenerationSample:
        start = time.monotonic()
        response = self._client.post(
            self._url(),
            headers=self._headers(),
            json={"model": self._model_id, "messages": messages, "max_tokens": max_tokens, "stream": False},
        )
        self._raise_for_status(response)
        payload = response.json()
        usage = payload.get("usage") or {}
        return CloudGenerationSample(
            text=payload["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            elapsed_s=time.monotonic() - start,
        )

    def generate_stream(self, messages: ChatMessages, max_tokens: int) -> Generator[str, None, Optional[CloudUsage]]:
        usage: Optional[CloudUsage] = None
        with self._client.stream(
            "POST",
            self._url(),
            headers=self._headers(),
            json={
                "model": self._model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                # Asks the server for a final chunk carrying real token
                # usage, so streamed requests can be billed exactly.
                "stream_options": {"include_usage": True},
            },
        ) as response:
            self._raise_for_status(response)
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = CloudUsage(chunk["usage"].get("prompt_tokens", 0), chunk["usage"].get("completion_tokens", 0))
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta
        return usage

    def close(self) -> None:
        self._client.close()
