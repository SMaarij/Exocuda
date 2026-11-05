"""gateway/tests/test_serve.py -- account/key management entrypoint and app wiring."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gateway import serve
from gateway.store import GatewayStore


def test_create_account_and_key_persist_to_configured_db(tmp_path, monkeypatch, capsys) -> None:
    db = tmp_path / "gw.db"
    monkeypatch.setenv("CASCADE_GATEWAY_DB", str(db))
    assert serve.main(["create-account", "alice", "--cap", "2.5"]) == 0
    assert serve.main(["create-key", "alice"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    account = GatewayStore(db).authenticate(key)
    assert (account.user_id, account.cap_usd) == ("alice", 2.5)


def test_build_app_reads_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CASCADE_GATEWAY_DB", str(tmp_path / "gw.db"))
    monkeypatch.setenv("CASCADE_UPSTREAM_URL", "http://upstream.invalid")
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    client = TestClient(serve.build_app())
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
