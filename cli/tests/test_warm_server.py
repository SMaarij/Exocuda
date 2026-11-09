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
