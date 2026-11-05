"""
gateway/providers/base.py

The inference-provider seam (FR-07.4): swapping providers is a config
change, not a code change. One Protocol targeting the OpenAI-compatible
chat-completions shape that vLLM, SGLang, Ollama, OpenAI, OpenRouter,
Groq and most serverless GPU providers all serve -- so one
HTTPInferenceProvider class covers every vendor via base_url/api_key/model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Protocol

ChatMessages = list[dict]


@dataclass(frozen=True)
class CloudUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class CloudGenerationSample:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class CloudInferenceProvider(Protocol):
    def generate(self, messages: ChatMessages, max_tokens: int) -> CloudGenerationSample: ...

    def generate_stream(self, messages: ChatMessages, max_tokens: int) -> Generator[str, None, CloudUsage | None]:
        """Yields text deltas; the generator's return value is the real
        usage when the provider reported it (read it with `yield from`)."""
        ...
