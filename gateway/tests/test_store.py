"""gateway/tests/test_store.py -- SQLite store: hashed keys, atomic token quotas, persistence."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from gateway.store import KEY_PREFIX, GatewayStore, QuotaExceededError


def test_key_authenticates_and_only_its_hash_is_stored(tmp_path) -> None:
    db = tmp_path / "s.db"
    store = GatewayStore(db)
    store.create_user("alice")
    key = store.create_api_key("alice")
    assert key.startswith(KEY_PREFIX)
    assert store.authenticate(key).user_id == "alice"
    stored = sqlite3.connect(db).execute("SELECT key_hash FROM api_keys").fetchone()[0]
    assert key not in stored and len(stored) == 64


def test_unknown_empty_and_revoked_keys_fail() -> None:
    store = GatewayStore()
    store.create_user("alice")
    key = store.create_api_key("alice")
    assert store.authenticate("csk_nope") is None
    assert store.authenticate("") is None
    assert store.revoke_api_key(key) is True
    assert store.authenticate(key) is None
    assert store.revoke_api_key(key) is False


def test_key_requires_existing_user() -> None:
    with pytest.raises(KeyError):
        GatewayStore().create_api_key("ghost")


def test_quota_check_reserves_nothing_on_rejection() -> None:
    store = GatewayStore()
    store.create_user("alice", quota_tokens=1000)
    store.check_and_reserve("alice", 600)
    with pytest.raises(QuotaExceededError):
        store.check_and_reserve("alice", 600)
    assert store.account("alice").used_tokens == 600


def test_no_quota_means_unlimited() -> None:
    store = GatewayStore()
    store.create_user("alice")
    store.check_and_reserve("alice", 10**9)


def test_adjust_reconciles_to_actual() -> None:
    store = GatewayStore()
    store.create_user("alice")
    store.check_and_reserve("alice", 500)
    store.adjust("alice", 500, 100)
    assert store.account("alice").used_tokens == 100


def test_usage_survives_restart_and_can_be_reset(tmp_path) -> None:
    db = tmp_path / "s.db"
    store = GatewayStore(db)
    store.create_user("alice", quota_tokens=1000)
    store.check_and_reserve("alice", 900)
    reopened = GatewayStore(db)
    with pytest.raises(QuotaExceededError):
        reopened.check_and_reserve("alice", 200)
    reopened.reset_usage("alice")
    reopened.check_and_reserve("alice", 200)


def test_concurrent_reservations_never_exceed_quota(tmp_path) -> None:
    store = GatewayStore(tmp_path / "s.db")
    store.create_user("alice", quota_tokens=1000)
    accepted = []

    def worker() -> None:
        try:
            store.check_and_reserve("alice", 100)
            accepted.append(1)
        except QuotaExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 10
    assert store.account("alice").used_tokens == 1000


def test_create_user_updates_quota_and_keeps_usage() -> None:
    store = GatewayStore()
    store.create_user("alice", quota_tokens=1000)
    store.check_and_reserve("alice", 500)
    store.create_user("alice", quota_tokens=2000)
    account = store.account("alice")
    assert (account.quota_tokens, account.used_tokens) == (2000, 500)
