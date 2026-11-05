"""
gateway/store.py

Server-side persistence for CascadeCloud: API keys (stored as SHA-256
hashes only -- a leaked database doesn't leak usable keys) and per-user
spend with optional caps. SQLite with `BEGIN IMMEDIATE` transactions, so
check-and-reserve stays atomic across threads and processes on one host,
and caps survive restarts. A multi-host deployment would swap this for
Postgres behind the same methods.
"""

from __future__ import annotations

import hashlib
import contextlib
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from gateway.billing.meter import SpendCapExceededError

KEY_PREFIX = "csk_"


@dataclass(frozen=True)
class Account:
    user_id: str
    cap_usd: Optional[float]
    spent_usd: float
    stripe_customer_id: Optional[str]


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class GatewayStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self._path = str(path)
        # A shared in-memory DB needs one long-lived connection; a file DB
        # opens a connection per operation so it's safe across threads.
        self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None) if self._path == ":memory:" else None
        self._memory_lock = threading.Lock() if self._memory_conn is not None else contextlib.nullcontext()
        # executescript() commits on its own, so the schema is created
        # outside the explicit-transaction helper.
        conn = self._memory_conn or sqlite3.connect(self._path, timeout=10, isolation_level=None)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    user_id TEXT PRIMARY KEY,
                    cap_usd REAL,
                    spent_usd REAL NOT NULL DEFAULT 0,
                    stripe_customer_id TEXT
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES accounts(user_id),
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                """
            )
        finally:
            if self._memory_conn is None:
                conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        # The shared in-memory connection is serialised with a lock; file
        # databases rely on SQLite's own locking via BEGIN IMMEDIATE.
        with self._memory_lock:
            with self._tx_unlocked() as conn:
                yield conn

    @contextmanager
    def _tx_unlocked(self) -> Iterator[sqlite3.Connection]:
        conn = self._memory_conn or sqlite3.connect(self._path, timeout=10, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            if self._memory_conn is None:
                conn.close()

    # ---- accounts & keys -------------------------------------------------

    def create_account(self, user_id: str, cap_usd: Optional[float] = None, stripe_customer_id: Optional[str] = None) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO accounts(user_id, cap_usd, stripe_customer_id) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET cap_usd = excluded.cap_usd, "
                "stripe_customer_id = COALESCE(excluded.stripe_customer_id, accounts.stripe_customer_id)",
                (user_id, cap_usd, stripe_customer_id),
            )

    def create_api_key(self, user_id: str) -> str:
        """Returns the plaintext key -- the only time it exists; only its hash is stored."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM accounts WHERE user_id = ?", (user_id,)).fetchone() is None:
                raise KeyError(f"no account {user_id!r}; create_account() first")
            conn.execute("INSERT INTO api_keys(key_hash, user_id) VALUES (?, ?)", (_hash_key(key), user_id))
        return key

    def revoke_api_key(self, key: str) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (_hash_key(key),))

    def authenticate(self, key: str) -> Optional[Account]:
        if not key:
            return None
        with self._tx() as conn:
            row = conn.execute(
                "SELECT a.user_id, a.cap_usd, a.spent_usd, a.stripe_customer_id FROM api_keys k "
                "JOIN accounts a ON a.user_id = k.user_id WHERE k.key_hash = ? AND k.revoked = 0",
                (_hash_key(key),),
            ).fetchone()
        return Account(*row) if row else None

    def account(self, user_id: str) -> Optional[Account]:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT user_id, cap_usd, spent_usd, stripe_customer_id FROM accounts WHERE user_id = ?", (user_id,)
            ).fetchone()
        return Account(*row) if row else None

    # ---- metering (FR-07.2) -----------------------------------------------

    def check_and_reserve(self, user_id: str, estimated_cost_usd: float) -> None:
        """Atomically reject (reserving nothing) or reserve the estimate."""
        with self._tx() as conn:
            cap, spent = conn.execute("SELECT cap_usd, spent_usd FROM accounts WHERE user_id = ?", (user_id,)).fetchone()
            if cap is not None and spent + estimated_cost_usd > cap:
                raise SpendCapExceededError(
                    f"Request would bring spend to ${spent + estimated_cost_usd:.4f}, exceeding the ${cap:.4f} cap."
                )
            conn.execute("UPDATE accounts SET spent_usd = spent_usd + ? WHERE user_id = ?", (estimated_cost_usd, user_id))

    def adjust(self, user_id: str, estimated_cost_usd: float, actual_cost_usd: float) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE accounts SET spent_usd = spent_usd + ? WHERE user_id = ?",
                (actual_cost_usd - estimated_cost_usd, user_id),
            )
