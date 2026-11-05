"""
gateway/store.py

Persistence for the self-hosted Cascade server: users, API keys (stored as
SHA-256 hashes only -- a leaked database doesn't leak usable keys) and
per-user token usage with an optional quota, so several people can share
one GPU box fairly. SQLite with `BEGIN IMMEDIATE` transactions, so
check-and-reserve stays atomic across threads and processes on one host,
and usage survives restarts. A multi-host deployment would swap this for
Postgres behind the same methods.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

KEY_PREFIX = "csk_"


class QuotaExceededError(RuntimeError):
    """A request would push a user past their token quota."""


@dataclass(frozen=True)
class Account:
    user_id: str
    quota_tokens: Optional[int]
    used_tokens: int


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
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    quota_tokens INTEGER,
                    used_tokens INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id),
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

    # ---- users & keys -----------------------------------------------------

    def create_user(self, user_id: str, quota_tokens: Optional[int] = None) -> None:
        """Create a user, or update an existing user's quota (usage is kept)."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO users(user_id, quota_tokens) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET quota_tokens = excluded.quota_tokens",
                (user_id, quota_tokens),
            )

    def create_api_key(self, user_id: str) -> str:
        """Returns the plaintext key -- the only time it exists; only its hash is stored."""
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone() is None:
                raise KeyError(f"no user {user_id!r}; create_user() first")
            conn.execute("INSERT INTO api_keys(key_hash, user_id) VALUES (?, ?)", (_hash_key(key), user_id))
        return key

    def revoke_api_key(self, key: str) -> bool:
        """Returns True if a key was revoked."""
        with self._tx() as conn:
            cursor = conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_hash = ? AND revoked = 0", (_hash_key(key),))
            return cursor.rowcount > 0

    def authenticate(self, key: str) -> Optional[Account]:
        if not key:
            return None
        with self._tx() as conn:
            row = conn.execute(
                "SELECT u.user_id, u.quota_tokens, u.used_tokens FROM api_keys k "
                "JOIN users u ON u.user_id = k.user_id WHERE k.key_hash = ? AND k.revoked = 0",
                (_hash_key(key),),
            ).fetchone()
        return Account(*row) if row else None

    def account(self, user_id: str) -> Optional[Account]:
        with self._tx() as conn:
            row = conn.execute("SELECT user_id, quota_tokens, used_tokens FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return Account(*row) if row else None

    # ---- usage ------------------------------------------------------------

    def check_and_reserve(self, user_id: str, tokens: int) -> None:
        """Atomically reject (reserving nothing) or reserve the token estimate."""
        with self._tx() as conn:
            quota, used = conn.execute("SELECT quota_tokens, used_tokens FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if quota is not None and used + tokens > quota:
                raise QuotaExceededError(
                    f"request needs up to {tokens} tokens; {user_id} has used {used} of a {quota}-token quota"
                )
            conn.execute("UPDATE users SET used_tokens = used_tokens + ? WHERE user_id = ?", (tokens, user_id))

    def adjust(self, user_id: str, reserved_tokens: int, actual_tokens: int) -> None:
        """Replace a reservation with what the request actually used."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE users SET used_tokens = used_tokens + ? WHERE user_id = ?",
                (actual_tokens - reserved_tokens, user_id),
            )

    def reset_usage(self, user_id: str) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE users SET used_tokens = 0 WHERE user_id = ?", (user_id,))
