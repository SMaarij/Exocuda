"""gateway/tests/test_store.py -- SQLite store: hashed keys, atomic metering, persistence."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from gateway.billing.meter import SpendCapExceededError
from gateway.store import KEY_PREFIX, GatewayStore


def test_key_authenticates_and_only_its_hash_is_stored(tmp_path) -> None:
    db = tmp_path / "g.db"
    store = GatewayStore(db)
    store.create_account("alice")
    key = store.create_api_key("alice")
    assert key.startswith(KEY_PREFIX)
    assert store.authenticate(key).user_id == "alice"
    stored = sqlite3.connect(db).execute("SELECT key_hash FROM api_keys").fetchone()[0]
    assert key not in stored and len(stored) == 64


def test_unknown_empty_and_revoked_keys_fail() -> None:
    store = GatewayStore()
    store.create_account("alice")
    key = store.create_api_key("alice")
    assert store.authenticate("csk_nope") is None
    assert store.authenticate("") is None
    store.revoke_api_key(key)
    assert store.authenticate(key) is None


def test_key_requires_existing_account() -> None:
    with pytest.raises(KeyError):
        GatewayStore().create_api_key("ghost")


def test_cap_check_reserves_nothing_on_rejection() -> None:
    store = GatewayStore()
    store.create_account("alice", cap_usd=1.0)
    store.check_and_reserve("alice", 0.6)
    with pytest.raises(SpendCapExceededError):
        store.check_and_reserve("alice", 0.6)
    assert store.account("alice").spent_usd == pytest.approx(0.6)


def test_adjust_reconciles_to_actual() -> None:
    store = GatewayStore()
    store.create_account("alice")
    store.check_and_reserve("alice", 0.5)
    store.adjust("alice", 0.5, 0.1)
    assert store.account("alice").spent_usd == pytest.approx(0.1)


def test_spend_survives_restart(tmp_path) -> None:
    db = tmp_path / "g.db"
    store = GatewayStore(db)
    store.create_account("alice", cap_usd=1.0)
    store.check_and_reserve("alice", 0.9)
    reopened = GatewayStore(db)
    with pytest.raises(SpendCapExceededError):
        reopened.check_and_reserve("alice", 0.2)


def test_concurrent_reservations_never_exceed_cap(tmp_path) -> None:
    store = GatewayStore(tmp_path / "g.db")
    store.create_account("alice", cap_usd=1.0)
    accepted = []

    def worker() -> None:
        try:
            store.check_and_reserve("alice", 0.1)
            accepted.append(1)
        except SpendCapExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 10
    assert store.account("alice").spent_usd == pytest.approx(1.0)


def test_create_account_updates_cap_and_keeps_spend() -> None:
    store = GatewayStore()
    store.create_account("alice", cap_usd=1.0)
    store.check_and_reserve("alice", 0.5)
    store.create_account("alice", cap_usd=2.0)
    account = store.account("alice")
    assert (account.cap_usd, account.spent_usd) == (2.0, pytest.approx(0.5))
