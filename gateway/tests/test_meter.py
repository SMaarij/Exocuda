"""gateway/tests/test_meter.py -- Track C, Task 9 (FR-07.2)."""

from __future__ import annotations

import threading

import pytest

from gateway.billing.meter import SpendCapExceededError, UsageMeter


def test_no_cap_never_raises() -> None:
    meter = UsageMeter(cap_usd=None)
    meter.check_and_reserve(1_000_000.0)  # should not raise


def test_request_under_cap_is_allowed() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.check_and_reserve(5.0)  # should not raise


def test_request_over_cap_raises() -> None:
    meter = UsageMeter(cap_usd=10.0)
    with pytest.raises(SpendCapExceededError):
        meter.check_and_reserve(11.0)


def test_check_and_reserve_accounts_for_already_spent() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.check_and_reserve(8.0)
    with pytest.raises(SpendCapExceededError):
        meter.check_and_reserve(3.0)  # 8 + 3 = 11 > 10


def test_check_and_reserve_immediately_reserves_the_estimate() -> None:
    meter = UsageMeter(cap_usd=None)
    meter.check_and_reserve(1.5)
    meter.check_and_reserve(2.5)
    assert meter.spent_usd == 4.0


def test_adjust_reconciles_estimate_with_actual_cost() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.check_and_reserve(5.0)  # reserved $5
    meter.adjust(estimated_cost_usd=5.0, actual_cost_usd=3.0)  # actual was only $3
    assert meter.spent_usd == 3.0


def test_adjust_on_failure_releases_the_full_reservation() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.check_and_reserve(5.0)
    meter.adjust(estimated_cost_usd=5.0, actual_cost_usd=0.0)  # call failed, no charge
    assert meter.spent_usd == 0.0


def test_remaining_usd_with_cap() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.check_and_reserve(3.0)
    assert meter.remaining_usd() == 7.0


def test_remaining_usd_never_negative() -> None:
    meter = UsageMeter(cap_usd=10.0)
    meter.spent_usd = 15.0  # over cap somehow (e.g. a manual adjustment)
    assert meter.remaining_usd() == 0.0


def test_remaining_usd_none_when_no_cap() -> None:
    meter = UsageMeter(cap_usd=None)
    assert meter.remaining_usd() is None


def test_check_and_reserve_is_thread_safe() -> None:
    """Concurrent requests racing past a stale `spent` total is exactly how a cap gets bypassed."""
    meter = UsageMeter(cap_usd=100.0)
    allowed = []
    denied = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            meter.check_and_reserve(10.0)  # reserves atomically -- no separate record() needed
            with lock:
                allowed.append(1)
        except SpendCapExceededError:
            with lock:
                denied.append(1)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 10 requests of $10 fit in a $100 cap -- atomicity guarantees
    # this is exact, not just "never more."
    assert len(allowed) == 10
    assert len(denied) == 10
    assert meter.spent_usd == 100.0
