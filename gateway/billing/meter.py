"""
gateway/billing/meter.py

Track C, Task 9 (FR-07.2).

Real-time spend-cap enforcement: reject a request that would exceed a
configured limit, checked BEFORE the (billable) cloud call happens --
never after, which would mean paying for a request that shouldn't have
been allowed at all.

`check_and_reserve()` both checks AND reserves atomically under one
lock acquisition -- the estimated cost is added to `spent_usd`
immediately, not after a separate `record()` call. Splitting check and
commit across two lock acquisitions would reopen exactly the race this
class exists to prevent (two concurrent requests could both pass the
check against the same stale `spent_usd` before either commits).
`adjust()` reconciles the reservation with the real cost once the
provider call completes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


class SpendCapExceededError(RuntimeError):
    """Raised by UsageMeter.check_and_reserve() when a request would push a user over their configured cap."""


@dataclass
class UsageMeter:
    """One meter per billable identity (a user, an API key, a team -- whatever CascadeCloud decides to cap independently)."""

    cap_usd: Optional[float] = None
    spent_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def check_and_reserve(self, estimated_cost_usd: float) -> None:
        """
        Atomically checks the cap and reserves the estimated cost in one
        lock acquisition. Raises SpendCapExceededError, reserving
        nothing, if the estimate would exceed the cap.
        """
        with self._lock:
            if self.cap_usd is not None and self.spent_usd + estimated_cost_usd > self.cap_usd:
                raise SpendCapExceededError(
                    f"Request would bring spend to ${self.spent_usd + estimated_cost_usd:.4f}, "
                    f"exceeding the ${self.cap_usd:.4f} cap."
                )
            self.spent_usd += estimated_cost_usd

    def adjust(self, estimated_cost_usd: float, actual_cost_usd: float) -> None:
        """Reconciles a prior reservation with the real cost once the provider call completes."""
        with self._lock:
            self.spent_usd += actual_cost_usd - estimated_cost_usd

    def remaining_usd(self) -> Optional[float]:
        if self.cap_usd is None:
            return None
        with self._lock:
            return max(0.0, self.cap_usd - self.spent_usd)
