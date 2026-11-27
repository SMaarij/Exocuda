"""
Looking at and correcting the semantic cache from the command line (`cascade cache ...`).

Works on the cache's SQLite file directly, so no embedder or model is loaded. A running
background server keeps an in-memory copy of the cache; callers tell it to reload
(SemanticCache.reload, the warm server's /cache/reload) after changing the file.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cascade_local.engine.intelligence.cache import DEFAULT_DB_PATH


@dataclass(frozen=True)
class CacheEntry:
    id: int
    query: str
    response: str
    created_at: float
    expires_at: Optional[float]
    hit_count: int

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()


def _connect(db_path: Path) -> Optional[sqlite3.Connection]:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cache_entries'").fetchone()
    if not exists:
        conn.close()
        return None
    return conn


def list_entries(db_path: Path = DEFAULT_DB_PATH, search: Optional[str] = None,
                 limit: Optional[int] = 20) -> list[CacheEntry]:
    """Newest first. `search` keeps entries whose question or answer contains it (any case)."""
    conn = _connect(db_path)
    if conn is None:
        return []
    try:
        sql = "SELECT id, query_text, response_text, created_at, expires_at, hit_count FROM cache_entries"
        args: list = []
        if search:
            sql += " WHERE lower(query_text) LIKE ? OR lower(response_text) LIKE ?"
            args = [f"%{search.lower()}%"] * 2
        sql += " ORDER BY created_at DESC, id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [CacheEntry(*row) for row in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def count(db_path: Path = DEFAULT_DB_PATH) -> int:
    conn = _connect(db_path)
    if conn is None:
        return 0
    try:
        return conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
    finally:
        conn.close()


def delete(ids: list[int], db_path: Path = DEFAULT_DB_PATH) -> int:
    """Remove these entries; returns how many were removed."""
    conn = _connect(db_path)
    if conn is None or not ids:
        return 0
    try:
        marks = ",".join("?" * len(ids))
        removed = conn.execute(f"DELETE FROM cache_entries WHERE id IN ({marks})", list(ids)).rowcount
        conn.commit()
        return removed
    finally:
        conn.close()


def clear(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Remove every entry; returns how many there were."""
    conn = _connect(db_path)
    if conn is None:
        return 0
    try:
        removed = conn.execute("DELETE FROM cache_entries").rowcount
        conn.commit()
        return removed
    finally:
        conn.close()
