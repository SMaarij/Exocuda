"""
gateway/providers/http_provider.py

Track C, Task 7 (FR-07.1, FR-07.6, FR-07.7).

Production CloudInferenceProvider: calls an OpenAI-compatible
chat-completions endpoint over HTTP. Works unmodified against vLLM,
SGLang, or any wholesale serverless GPU provider that serves that API
shape (FR-07.4's provider-swap-by-config, see base.py's docstring).

FR-07.6/07.7 (continuous batching, PagedAttention, prefix caching,
FlashAttention) are properties of the SERVING ENGINE running behind
this HTTP endpoint (vLLM/SGLang), not something this HTTP client
implements itself -- this class's job is just to call that endpoint
correctly and report real cost/timing back to BackendResult. Which
serving engine is actually running is an infrastructure/deployment
decision (which wholesale GPU provider is contracted, how their vLLM
instance is configured), documented as a joint decision in
shared/CHANGELOG.md the same way Track A flagged the reference model
tag as one.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import httpx

from gateway.providers.base import CloudGenerationSample


class HTTPInferenceProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_s: float = 60.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._client = client or httpx.Client(timeout=timeout_s)

    def generate(self, prompt: str, max_tokens: int) -> CloudGenerationSample:
        start = time.monotonic()
        response = self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        elapsed_s = time.monotonic() - start

        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return CloudGenerationSample(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            elapsed_s=elapsed_s,
        )

    def generate_stream(self, prompt: str, max_tokens: int) -> Iterator[str]:
        with self._client.stream(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                import json

                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta

    def close(self) -> None:
        self._client.close()
