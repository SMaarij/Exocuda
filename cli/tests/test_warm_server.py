"""cli/warm_server.py over a real localhost socket, with a fake Pipeline (NFR-09)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from cli import warm_server


class _FakePipeline:
    def __init__(self) -> None:
        self.threads = {threading.get_ident()}
        self.warmed = False
        self.calls: list = []
        self._decision_log_path = Path("decisions.jsonl")

    def warm(self) -> None:
        self.warmed = True

    def run(self, messages):
        self.threads.add(threading.get_ident())
        self.calls.append((messages, self._decision_log_path))
        return {"text": f"echo {messages[-1]['content']}", "tier": "local", "cache_hit": False}


@pytest.fixture
def server(tmp_path: Path):
    builds: list[_FakePipeline] = []

    def build():
        builds.append(_FakePipeline())
        return builds[-1]

    config = tmp_path / "config.json"
    config.write_text("{}")
    srv = warm_server.WarmServer(build, config_path=config, idle_seconds=60, port=0)
    port = srv.httpd.server_address[1]
    srv.load()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, port, builds, config
    srv.httpd.shutdown()


def test_server_answers_and_warms_the_pipeline(server) -> None:
    srv, port, builds, _ = server
    assert warm_server.health(port)["ready"] is True
    assert builds[0].warmed
    assert warm_server.ask([{"role": "user", "content": "hi"}], port=port)["text"] == "echo hi"


def test_no_log_requests_skip_the_decision_log_only_for_that_request(server) -> None:
    _, port, builds, _ = server
    warm_server.ask([{"role": "user", "content": "a"}], log=False, port=port)
    warm_server.ask([{"role": "user", "content": "b"}], port=port)
    assert [path for _, path in builds[0].calls] == [None, Path("decisions.jsonl")]


def test_config_change_rebuilds_the_pipeline(server) -> None:
    _, port, builds, config = server
    time.sleep(0.05)
    config.write_text('{"model": "other"}')
    warm_server.ask([{"role": "user", "content": "x"}], port=port)
    assert len(builds) == 2


def test_client_returns_none_when_nothing_listens() -> None:
    assert warm_server.health(1) is None
    assert warm_server.ask([{"role": "user", "content": "x"}], port=1) is None


def test_idle_server_shuts_itself_down(tmp_path: Path) -> None:
    srv = warm_server.WarmServer(_FakePipeline, idle_seconds=0.5, port=0)
    srv.load()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_build_and_every_request_run_on_one_thread(server) -> None:
    """The cache's SQLite connection only works on the thread that created it."""
    _, port, builds, _ = server
    for q in ("a", "b", "c"):
        warm_server.ask([{"role": "user", "content": q}], port=port)
    assert len(builds[0].calls) == 3
    assert len(builds[0].threads) == 1


def test_health_reports_loading_until_models_are_ready(tmp_path: Path) -> None:
    gate = threading.Event()

    class _Slow(_FakePipeline):
        def warm(self) -> None:
            gate.wait(5)

    srv = warm_server.WarmServer(_Slow, idle_seconds=60, port=0)
    port = srv.httpd.server_address[1]
    threading.Thread(target=srv.serve_forever, kwargs={"load_in_background": True}, daemon=True).start()
    try:
        assert warm_server.health(port)["ready"] is False
        assert warm_server.ask([{"role": "user", "content": "x"}], port=port) is None  # 503 while loading
        gate.set()
        assert warm_server.wait_ready(port, timeout=5, poll=0.05)
    finally:
        srv.httpd.shutdown()


# ---- Ollama / OpenAI compatibility ---------------------------------------------------


def _post(port: int, path: str, body: dict) -> tuple[int, str, bytes]:
    import json
    import urllib.request

    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def test_openai_chat_completions(server) -> None:
    import json

    _, port, _, _ = server
    _, _, raw = _post(port, "/v1/chat/completions", {"model": "gpt-x", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
    body = json.loads(raw)
    assert body["model"] == "gpt-x"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "echo hi"}


def test_openai_streaming_sends_sse_chunks(server) -> None:
    import json

    _, port, _, _ = server
    _, ctype, raw = _post(port, "/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert ctype.startswith("text/event-stream")
    events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
    assert text == "echo hi"


def test_ollama_chat_streams_ndjson_by_default(server) -> None:
    import json

    _, port, _, _ = server
    _, ctype, raw = _post(port, "/api/chat", {"model": "qwen2.5-coder", "messages": [{"role": "user", "content": "hi"}]})
    lines = [json.loads(l) for l in raw.decode().splitlines()]
    assert ctype.startswith("application/x-ndjson")
    assert "".join(l["message"]["content"] for l in lines) == "echo hi"
    assert lines[-1]["done"] is True and lines[-1]["model"] == "qwen2.5-coder"


def test_ollama_generate_and_load_requests(server) -> None:
    import json

    _, port, builds, _ = server
    body = json.loads(_post(port, "/api/generate", {"model": "m", "prompt": "hi", "stream": False})[2])
    assert (body["response"], body["done"]) == ("echo hi", True)
    load = json.loads(_post(port, "/api/generate", {"model": "m", "stream": False})[2])
    assert load["done_reason"] == "load"
    assert len(builds[0].calls) == 1  # the load request asked nothing


def test_ollama_model_listing_endpoints(server) -> None:
    import json
    import urllib.request

    srv, port, _, _ = server
    get = lambda p: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{p}", timeout=5).read())
    assert get("/api/tags")["models"][0]["name"] == srv.model_name
    assert "version" in get("/api/version")
    assert get("/v1/models")["data"][0]["id"] == srv.model_name
    assert json.loads(_post(port, "/api/show", {"model": "x"})[2])["capabilities"] == ["completion"]


def test_keep_listening_unloads_when_idle_and_reloads_on_demand(tmp_path: Path) -> None:
    builds: list[_FakePipeline] = []
    srv = warm_server.WarmServer(lambda: builds.append(_FakePipeline()) or builds[-1], idle_seconds=60, port=0,
                                 keep_listening=True)
    port = srv.httpd.server_address[1]
    srv.load()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert srv._on_idle() is False          # keeps serving...
        assert not srv.loaded                   # ...with the models freed
        assert warm_server.health(port)["ready"] is True
        assert warm_server.ask([{"role": "user", "content": "again"}], port=port)["text"] == "echo again"
        assert len(builds) == 2 and srv.loaded  # reloaded for the request
    finally:
        srv.httpd.shutdown()


def test_without_keep_listening_idle_means_exit(server) -> None:
    srv, _, _, _ = server
    assert srv._on_idle() is True
