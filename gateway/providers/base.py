"""
gateway/providers/base.py

Track C, Task 6 (FR-07.4).

The wholesale-GPU-provider seam: swapping providers is a config change,
not a code change. Mirrors Track A's ReferenceModelRunner pattern
(engine/execution/model_runner.py) -- a Protocol, a real implementation,
and a deterministic fake for tests (NFR-09: no live network needed to
run the suite).

Design note on "one adapter, many providers": rather than writing a
separate adapter class per wholesale GPU vendor (RunPod, Modal,
Fireworks, ...), CloudInferenceProvider targets the OpenAI-compatible
chat-completions API shape that vLLM and SGLang both serve (Tech Stack
5.1) -- and that every major serverless GPU provider exposes an
OpenAI-compatible endpoint for. Swapping providers is therefore just a
`base_url`/`api_key`/`model_id` config change against the SAME
HTTPInferenceProvider class, which is a stronger form of FR-07.4's
requirement than a new adapter per vendor would be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class CloudGenerationSample:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float


class CloudInferenceProvider(Protocol):
    def generate(self, prompt: str, max_tokens: int) -> CloudGenerationSample: ...

    def generate_stream(self, prompt: str, max_tokens: int) -> Iterator[str]: ...
