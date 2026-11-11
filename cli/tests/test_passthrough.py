"""Cascade in front of a real Ollama: which requests it answers, which it passes through,
lazy loading, and the login entry (`cascade install`). Fake Ollama and fake Pipeline (NFR-09)."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cli import autostart, warm_server


class _FakePipeline:
    def warm(self) -> None:
        pass

    def run(self, messages):
        return {"text": f"cascade: {messages[-1]['content']}", "tier": "local", "cache_hit": False}


class _FakeOllama:
    """Answers like Ollama for one installed model, llama3.2:3b."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def _json(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.requests.append(("GET", self.path))
                if self.path == "/api/tags":
                    self._json(200, {"models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5-coder:7b"}]})
                elif self.path == "/api/version":
                    self._json(200, {"version": "9.9.9"})
                elif self.path == "/api/ps":
                    self._json(200, {"models": [{"name": "llama3.2:3b"}]})
                else:
                    self._json(404, {"error": "nope"})

            def do_DELETE(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.requests.append(("DELETE", self.path + " " + body.decode()))
                self._json(200, {})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                fake.requests.append(("POST", self.path))
                model = body.get("model") or body.get("name")
                if self.path == "/api/pull":  # chunked progress, like Ollama
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for line in (b'{"status":"pulling"}\n', b'{"status":"success"}\n'):
                        self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                elif model not in ("llama3.2:3b", "qwen2.5-coder:7b"):
                    self._json(404, {"error": f"model '{model}' not found"})
                elif self.path == "/api/show":
                    self._json(200, {"modelfile": "FROM real", "capabilities": ["completion", "tools"]})
                elif self.path == "/api/chat":
                    self._json(200, {"model": model, "message": {"role": "assistant", "content": "ollama"}, "done": True})
                elif self.path == "/v1/chat/completions":
                    self._json(200, {"choices": [{"message": {"role": "assistant", "content": "ollama-openai"}}]})
                else:
                    self._json(404, {"error": "nope"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def _start(srv: warm_server.WarmServer) -> int:
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.httpd.server_address[1]


@pytest.fixture
def ollama():
    fake = _FakeOllama()
    yield fake
    fake.httpd.shutdown()


@pytest.fixture
def cascade(ollama):
    srv = warm_server.WarmServer(_FakePipeline, idle_seconds=60, port=0, model_name="qwen2.5-coder",
                                 upstream=ollama.url, lazy=True)
    port = _start(srv)
    yield srv, port
    srv.httpd.shutdown()


def _call(port: int, path: str, body=None, method: str = "POST") -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _chat(port: int, model: str, path: str = "/api/chat") -> tuple[int, dict]:
    status, raw = _call(port, path, {"model": model, "stream": False,
                                     "messages": [{"role": "user", "content": "hi"}]})
    return status, json.loads(raw)


def test_the_configured_family_is_answered_by_cascade(cascade, ollama) -> None:
    _, port = cascade
    for name in ("qwen2.5-coder", "qwen2.5-coder:7b", "cascade"):
        assert _chat(port, name)[1]["message"]["content"] == "cascade: hi"
    assert ("POST", "/api/chat") not in ollama.requests


def test_other_ollama_models_run_on_the_real_ollama(cascade, ollama) -> None:
    _, port = cascade
    status, body = _chat(port, "llama3.2:3b")
    assert (status, body["message"]["content"]) == (200, "ollama")
    status, body = _chat(port, "mistral:7b")  # not installed: Ollama's own 404, so `ollama run` pulls it
    assert status == 404 and "not found" in body["error"]


def test_openai_names_ollama_lacks_are_answered_by_cascade(cascade, ollama) -> None:
    _, port = cascade
    assert _chat(port, "gpt-4o", "/v1/chat/completions")[1]["choices"][0]["message"]["content"] == "cascade: hi"
    assert _chat(port, "llama3.2:3b", "/v1/chat/completions")[1]["choices"][0]["message"]["content"] == "ollama-openai"


def test_model_management_passes_through(cascade, ollama) -> None:
    _, port = cascade
    assert [m["name"] for m in json.loads(_call(port, "/api/tags", method="GET")[1])["models"]] == \
        ["llama3.2:3b", "qwen2.5-coder:7b"]
    assert json.loads(_call(port, "/api/version", method="GET")[1]) == {"version": "9.9.9"}
    status, raw = _call(port, "/api/pull", {"model": "mistral:7b"})
    assert status == 200 and raw.splitlines() == [b'{"status":"pulling"}', b'{"status":"success"}']
    assert _call(port, "/api/delete", {"model": "llama3.2:3b"}, method="DELETE")[0] == 200
    assert ("DELETE", '/api/delete {"model": "llama3.2:3b"}') in ollama.requests
    assert json.loads(_call(port, "/api/show", {"model": "qwen2.5-coder:7b"})[1])["modelfile"] == "FROM real"
    assert _call(port, "/api/show", {"model": "mistral:7b"})[0] == 404


def test_ps_lists_cascades_model_with_ollamas(cascade, ollama) -> None:
    srv, port = cascade
    assert [m["name"] for m in json.loads(_call(port, "/api/ps", method="GET")[1])["models"]] == ["llama3.2:3b"]
    _chat(port, "qwen2.5-coder")
    assert [m["name"] for m in json.loads(_call(port, "/api/ps", method="GET")[1])["models"]] == \
        ["qwen2.5-coder", "llama3.2:3b"]


def test_lazy_server_is_ready_at_once_and_loads_on_the_first_request(cascade) -> None:
    srv, port = cascade
    assert warm_server.health(port) == {"ready": True, "loaded": False, "pid": os.getpid()}
    _chat(port, "qwen2.5-coder")
    assert srv.loaded


def test_upstream_that_is_cascade_itself_is_refused(ollama) -> None:
    inner = warm_server.WarmServer(_FakePipeline, idle_seconds=60, port=0, upstream=ollama.url, lazy=True)
    inner_port = _start(inner)
    outer = warm_server.WarmServer(_FakePipeline, idle_seconds=60, port=0, model_name="qwen2.5-coder",
                                   upstream=f"http://127.0.0.1:{inner_port}", lazy=True)
    outer_port = _start(outer)
    try:
        assert _call(outer_port, "/api/pull", {"model": "x"})[0] == 508
    finally:
        inner.httpd.shutdown()
        outer.httpd.shutdown()
    same = warm_server.WarmServer(_FakePipeline, port=0, upstream="http://127.0.0.1:0")
    assert same.upstream is None  # its own port
    same.httpd.server_close()


def test_without_an_upstream_cascade_answers_everything() -> None:
    srv = warm_server.WarmServer(_FakePipeline, idle_seconds=60, port=0, model_name="qwen2.5-coder", lazy=True)
    port = _start(srv)
    try:
        assert _chat(port, "llama3.2:3b")[1]["message"]["content"] == "cascade: hi"
        assert _call(port, "/api/pull", {"model": "x"}) == (200, b'{"status": "success"}')
    finally:
        srv.httpd.shutdown()


def test_server_kind_tells_cascade_from_ollama(cascade, ollama) -> None:
    _, port = cascade
    assert autostart.server_kind(f"http://127.0.0.1:{port}") == "cascade"
    assert autostart.server_kind(ollama.url) == "ollama"
    assert autostart.server_kind("http://127.0.0.1:9") is None


# ---- the login entry ---------------------------------------------------------------


def test_login_entry_runs_the_lazy_server_without_a_console() -> None:
    command = autostart.launch_command()
    assert command[-2:] == [str(autostart.REPO / "cli" / "autostart.py"), "run"]
    if os.name == "nt" and Path(command[0]).with_name("pythonw.exe").exists():
        assert command[0].lower().endswith("pythonw.exe")
    assert "<string>run</string>" in autostart._launch_agent_plist(command)
    assert "ExecStart=" in autostart._systemd_unit(command) and "WantedBy=default.target" in autostart._systemd_unit(command)


@pytest.mark.skipif(os.name != "nt", reason="HKCU Run key")
def test_windows_login_entry_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(autostart, "APP_NAME", f"CascadeTest{os.getpid()}")
    try:
        assert not autostart.entry_installed()
        autostart.install_entry(["python", "x.py", "run"])
        assert autostart.entry_installed()
    finally:
        assert autostart.remove_entry()
    assert not autostart.entry_installed()


@pytest.mark.skipif(os.name != "nt", reason="Ollama's Startup shortcut is Windows-only")
def test_ollama_login_shortcut_is_set_aside_and_put_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(autostart, "SET_ASIDE_DIR", tmp_path / "aside")
    link = autostart.ollama_startup_shortcut()
    link.parent.mkdir(parents=True)
    link.write_bytes(b"lnk")
    assert autostart.set_aside_ollama_autostart().startswith("moved")
    assert not link.exists() and (tmp_path / "aside" / "Ollama.lnk").exists()
    assert autostart.restore_ollama_autostart() == link and link.read_bytes() == b"lnk"
    assert autostart.restore_ollama_autostart() is None  # nothing set aside any more
