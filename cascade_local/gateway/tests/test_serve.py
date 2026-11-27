"""gateway/tests/test_serve.py -- user/key management entrypoint and app wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cascade_local.gateway import serve
from cascade_local.gateway.store import GatewayStore


def test_create_user_and_key_persist_to_configured_db(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "server.db"
    monkeypatch.setenv("CASCADE_SERVER_DB", str(db))
    assert serve.main(["create-user", "alice", "--quota", "2500"]) == 0
    assert serve.main(["create-key", "alice"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    account = GatewayStore(db).authenticate(key)
    assert (account.user_id, account.quota_tokens) == ("alice", 2500)

    assert serve.main(["usage", "alice"]) == 0
    assert "0 tokens used (quota: 2500)" in capsys.readouterr().out
    assert serve.main(["revoke-key", key]) == 0
    assert GatewayStore(db).authenticate(key) is None
    assert serve.main(["revoke-key", key]) == 1


def test_reset_usage(tmp_path, monkeypatch) -> None:
    db = tmp_path / "server.db"
    monkeypatch.setenv("CASCADE_SERVER_DB", str(db))
    serve.main(["create-user", "alice"])
    GatewayStore(db).check_and_reserve("alice", 123)
    assert serve.main(["reset-usage", "alice"]) == 0
    assert GatewayStore(db).account("alice").used_tokens == 0


def test_usage_of_unknown_user_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CASCADE_SERVER_DB", str(tmp_path / "server.db"))
    assert serve.main(["usage", "ghost"]) == 1


def test_build_app_reads_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CASCADE_SERVER_DB", str(tmp_path / "server.db"))
    monkeypatch.setenv("CASCADE_UPSTREAM_URL", "http://upstream.invalid")
    client = TestClient(serve.build_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
