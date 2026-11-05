"""
engine/intelligence/router/shadow_labeling.py

Track B, Task 14 (FR-05.4).

Samples a configurable percentage of already-served local-tier traffic
for a background comparison call against self_hosted_backend, purely to
generate a training label for the next SemanticRouter retrain -- with
zero effect on the response actually served (call this only AFTER
Pipeline has already returned its result to the caller).

Note on "background": this implementation runs the shadow call
synchronously when invoked, which is correct and fully testable
(NFR-09: no threading needed to verify the sampling/logging logic).
Running it off the request's critical path in production (a thread,
a queue, an async task) is a deployment concern for whoever wires
Pipeline into Track C's gateway/CLI -- ShadowLabeler.maybe_label() is
written so it can be called from a background thread/task with no
changes needed here.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

from shared.types import ExecutionBackend, Message

DEFAULT_LOG_PATH = Path.home() / ".cascade" / "shadow_labels.jsonl"


class ShadowLabeler:
    def __init__(
        self,
        self_hosted_backend: Optional[ExecutionBackend],
        sample_rate: float = 0.05,
        log_path: Path = DEFAULT_LOG_PATH,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._self_hosted_backend = self_hosted_backend
        self._sample_rate = sample_rate
        self._log_path = log_path
        self._rng = rng or random.Random()

    def maybe_label(self, messages: list[Message], served_tier: str, served_text: str) -> bool:
        """Returns True if a shadow label was written, False if skipped (not sampled, or no self_hosted_backend)."""
        if self._self_hosted_backend is None:
            return False
        if self._rng.random() > self._sample_rate:
            return False

        try:
            shadow_result = self._self_hosted_backend.generate(messages)
        except Exception:
            return False  # shadow labeling must never raise into the caller (FR-05.4)

        self._write_log(
            {
                "ts": time.time(),
                "query": messages[-1]["content"] if messages else "",
                "served_tier": served_tier,
                "served_text": served_text,
                "shadow_text": shadow_result["text"],
                "agree": served_text.strip() == shadow_result["text"].strip(),
            }
        )
        return True

    def _write_log(self, record: dict) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
