"""
engine/intelligence/router/confidence.py

Track B, Task 9 (FR-05.3).

Computes local-tier answer confidence AFTER a BackendResult comes back
-- used by Pipeline to decide whether to escalate a locally-answered
query to cascade_backend even though local execution technically
succeeded (the answer might just be weak).

FLAGGED GAP -- not a shortcut, the spec-anticipated case:
    FR-05.3 says: "compute local-tier confidence from model-internal
    signals (token logprobs/entropy) where the execution backend
    exposes them, falling back to a heuristic only when it does not."

    Today, no ExecutionBackend implementation exposes logprobs/entropy.
    BackendResult.confidence (as Track A implemented it) is the
    speculative-decoding acceptance rate -- a *speed-layer* signal, not
    an answer-quality signal (see docs/TRACK_A_HANDOFF_FOR_B_AND_C.md,
    Section 1's field table: "confidence: Acceptance rate of
    speculative tokens"). So estimate_local_confidence() always takes
    the heuristic fallback branch right now -- that's FR-05.3's own
    documented "only when it does not" case, not a gap I introduced.

    Extending BackendResult with an optional logprobs field is flagged
    in shared/CHANGELOG.md as a future all-three-tracks decision (it
    touches the frozen shared/types.py). The logprobs branch below is
    kept ready for when that lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_HEDGING_PHRASES = (
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "it's hard to say",
    "i might be wrong",
    "i could be wrong",
    "as an ai",
    "i cannot be certain",
    "i can't be certain",
    "unclear",
    "i'm uncertain",
)


@dataclass(frozen=True)
class ConfidenceConfig:
    min_answer_chars: int = 8
    hedging_penalty: float = 0.3
    short_answer_penalty: float = 0.3


def estimate_local_confidence(
    answer_text: str,
    logprobs: list[float] | None = None,
    config: ConfidenceConfig | None = None,
) -> float:
    """
    Returns a confidence score in [0.0, 1.0]. Prefers model-internal
    signals (logprobs) when supplied; otherwise falls back to a
    text-shape heuristic (hedging phrases, suspiciously short answers).
    """
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
        return max(0.0, min(1.0, math.exp(avg_logprob)))

    config = config or ConfidenceConfig()
    text = answer_text.strip()
    score = 1.0

    if len(text) < config.min_answer_chars:
        score -= config.short_answer_penalty

    lowered = text.lower()
    if any(phrase in lowered for phrase in _HEDGING_PHRASES):
        score -= config.hedging_penalty

    return max(0.0, min(1.0, score))
