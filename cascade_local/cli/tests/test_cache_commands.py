"""`cascade cache list / forget / clear`, the cache file helpers, and the running server
re-reading the cache after a change. No embedder or model is loaded."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cascade_local.cli.main as cli_main
from cascade_local.cli import warm_server
from cascade_local.cli.main import app
from cascade_local.engine.intelligence import cache_admin

runner = CliRunner()


def _seed(db: Path, rows: list[tuple[str, str]], expired: tuple[int, ...] = ()) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, query_text TEXT NOT NULL, "
        "query_embedding BLOB NOT NULL, response_text TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL, "
        "hit_count INTEGER NOT NULL DEFAULT 0)")
    now = time.time()
    for i, (q, a) in enumerate(rows):
        conn.execute("INSERT INTO cache_entries (query_text, query_embedding, response_text, created_at, expires_at) "
                     "VALUES (?, ?, ?, ?, ?)", (q, b"\0", a, now + i, now - 1 if i in expired else None))
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(cli_main, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cli_main, "_cache_db", lambda: path)
    monkeypatch.setattr(warm_server, "reload_cache", lambda port=None: False)
    _seed(path, [("capital of Australia?", "Canberra"), ("capital of Austria?", "Canberra (wrong)"),
                 ("what is a hash table?", "A map from keys to values.")], expired=(0,))
    return path


def test_helpers_list_newest_first_search_delete_and_clear(db: Path) -> None:
    entries = cache_admin.list_entries(db)
    assert [e.query for e in entries] == ["what is a hash table?", "capital of Austria?", "capital of Australia?"]
    assert entries[-1].expired and not entries[0].expired
    assert [e.query for e in cache_admin.list_entries(db, search="AUSTRIA")] == ["capital of Austria?"]
    assert cache_admin.list_entries(db, limit=1)[0].query == "what is a hash table?"
    assert cache_admin.delete([entries[1].id], db) == 1 and cache_admin.count(db) == 2
    assert cache_admin.clear(db) == 2 and cache_admin.count(db) == 0


def test_helpers_handle_a_missing_cache(tmp_path: Path) -> None:
    missing = tmp_path / "none.sqlite3"
    assert cache_admin.list_entries(missing) == [] and cache_admin.count(missing) == 0
    assert cache_admin.delete([1], missing) == 0 and cache_admin.clear(missing) == 0
    assert not missing.exists()  # looking never creates the file


def test_list_shows_entries_and_how_to_remove_one(db: Path) -> None:
    result = runner.invoke(app, ["cache", "list"])
    assert result.exit_code == 0, result.output
    assert "capital of Austria?" in result.output and "(expired)" in result.output
    assert "cascade cache forget ID" in result.output
    assert "No cached answer matches" in runner.invoke(app, ["cache", "list", "zebra"]).output


def test_forget_by_id_last_and_match_with_confirmation(db: Path) -> None:
    wrong = cache_admin.list_entries(db, search="Austria")[0]
    result = runner.invoke(app, ["cache", "forget", str(wrong.id)], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Forgot 1" in result.output and cache_admin.count(db) == 2

    result = runner.invoke(app, ["cache", "forget", "--last"], input="n\n")
    assert result.exit_code == 1 and cache_admin.count(db) == 2  # declined: nothing removed

    result = runner.invoke(app, ["cache", "forget", "--last", "--yes"])
    assert "hash table" in result.output and cache_admin.count(db) == 1

    result = runner.invoke(app, ["cache", "forget", "--match", "capital", "--yes"])
    assert cache_admin.count(db) == 0
    assert runner.invoke(app, ["cache", "forget"]).exit_code == 1  # nothing said


def test_clear_asks_and_defaults_to_no(db: Path) -> None:
    result = runner.invoke(app, ["cache", "clear"], input="\n")
    assert result.exit_code == 1 and cache_admin.count(db) == 3
    result = runner.invoke(app, ["cache", "clear", "--yes"])
    assert "Removed 3" in result.output and cache_admin.count(db) == 0
    assert "already empty" in runner.invoke(app, ["cache", "clear", "--yes"]).output


def test_a_change_tells_the_running_server(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(warm_server, "reload_cache", lambda port=None: calls.append(port) or True)
    result = runner.invoke(app, ["cache", "forget", "--last", "--yes"])
    assert calls == [warm_server.DEFAULT_PORT]
    assert "running Cascade server has the change" in result.output


def test_server_reload_rereads_the_file_on_the_pipeline_thread() -> None:
    reloaded = []

    class _Cache:
        def reload(self) -> None:
            reloaded.append(True)

    class _Pipeline:
        _cache = _Cache()

    server = warm_server.WarmServer(build=lambda: _Pipeline(), port=0, lazy=True)
    try:
        assert server.reload_cache() is False  # nothing loaded yet: a later build reads the file
        server.load()
        assert server.reload_cache() is True and reloaded == [True]
    finally:
        server.httpd.server_close()


def test_semantic_cache_reload_picks_up_external_deletes(tmp_path: Path) -> None:
    from cascade_local.engine.intelligence.cache import SemanticCache

    class _Embedder:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    path = tmp_path / "c.sqlite3"
    cache = SemanticCache(_Embedder(), db_path=path)
    messages = [{"role": "user", "content": "what is a hash table?"}]
    cache.write(messages, "A map from keys to values.")
    assert cache.lookup(messages) == "A map from keys to values."
    cache_admin.clear(path)
    assert cache.lookup(messages) is not None  # still in memory
    cache.reload()
    assert cache.lookup(messages) is None
    cache.close()
