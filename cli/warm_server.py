"""
cli/warm_server.py -- keep the Pipeline loaded between `cascade ask` calls.

Every `cascade ask` is a new process. With the llama.cpp runner that meant
reloading the 7B model (~34 s on a CPU laptop), the draft model and the
embedding model (~2.4 s) before every answer -- often far longer than the
answer itself. Like Ollama's keep-alive, `cascade ask` now starts this server
in the background on first use and sends questions to it; it exits after
`server.idle_minutes` without a request, which frees the RAM again.

It binds to 127.0.0.1 only and runs one request at a time (one model, one
KV cache). Nothing here talks to the network beyond localhost (NFR-05).
Standard library only.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

DEFAULT_PORT = 8765
DEFAULT_IDLE_MINUTES = 10
HOST = "127.0.0.1"


class WarmServer:
    """Owns one Pipeline. `build(decision_logging)` must return a fresh Pipeline;
    it is called again when the config file changes."""

    def __init__(self, build: Callable[[], object], config_path: Optional[Path] = None,
                 idle_seconds: float = DEFAULT_IDLE_MINUTES * 60, port: int = DEFAULT_PORT) -> None:
        self._build = build
        self._config_path = config_path
        self._idle_seconds = idle_seconds
        self._lock = threading.Lock()
        # Every Pipeline call runs on this one thread: the cache's SQLite
        # connection only works on the thread that opened it, and the model
        # handles one request at a time anyway. HTTP requests (health checks
        # included) are still served concurrently, so a long answer never
        # makes a second client think the server is gone.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cascade-pipeline")
        self._pipeline = None
        self._config_mtime = self._mtime()
        self._last_activity = time.monotonic()
        self.ready = False
        self.httpd = ThreadingHTTPServer((HOST, port), self._handler())
        self.httpd.handle_error = self._handle_error

    def _mtime(self) -> Optional[float]:
        try:
            return self._config_path.stat().st_mtime if self._config_path else None
        except OSError:
            return None

    def load(self) -> None:
        self._worker.submit(self._load).result()

    def _load(self) -> None:
        pipeline = self._build()
        warm = getattr(pipeline, "warm", None)
        if warm:
            warm()  # load every local model now, not on the first question
        self._pipeline = pipeline
        self.ready = True

    def ask(self, messages: list, log: bool = True) -> dict:
        with self._lock:
            self._last_activity = time.monotonic()
            try:
                return self._worker.submit(self._ask, messages, log).result()
            finally:
                self._last_activity = time.monotonic()

    def _ask(self, messages: list, log: bool) -> dict:
        if self._mtime() != self._config_mtime:  # `cascade config set` changed something
            self._config_mtime = self._mtime()
            self._pipeline = None
            self._load()
        pipeline = self._pipeline
        saved = getattr(pipeline, "_decision_log_path", None)
        if not log:
            pipeline._decision_log_path = None
        try:
            return pipeline.run(messages)
        finally:
            if not log:
                pipeline._decision_log_path = saved

    @staticmethod
    def _handle_error(request, client_address) -> None:
        exc = sys.exc_info()[1]
        if not isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            traceback.print_exc()  # a client giving up on a reply is not worth a traceback

    def idle_for(self) -> float:
        return 0.0 if self._lock.locked() else time.monotonic() - self._last_activity

    def serve_forever(self, load_in_background: bool = False) -> None:
        """Serve until idle. With load_in_background, health checks answer
        {"ready": false} while the models load instead of timing out."""
        if load_in_background:
            threading.Thread(target=self.load, daemon=True).start()

        def watchdog() -> None:
            while True:
                time.sleep(min(30.0, max(1.0, self._idle_seconds / 4)))
                if self.idle_for() > self._idle_seconds:
                    self.httpd.shutdown()
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        try:
            self.httpd.serve_forever()
        finally:
            self.httpd.server_close()

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def _reply(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._reply(200, {"ready": server.ready, "pid": os.getpid()})
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/shutdown":
                    self._reply(200, {"stopping": True})
                    threading.Thread(target=server.httpd.shutdown, daemon=True).start()
                elif self.path == "/ask":
                    if not server.ready:
                        self._reply(503, {"error": "loading"})
                        return
                    try:
                        self._reply(200, server.ask(body["messages"], bool(body.get("log", True))))
                    except Exception as exc:  # report, keep serving
                        traceback.print_exc()  # to ~/.cascade/server.log
                        self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
                else:
                    self._reply(404, {"error": "not found"})

        return Handler


# ---- client side -------------------------------------------------------------------


def _request(port: int, path: str, body: Optional[dict] = None, timeout: float = 2.0) -> Optional[dict]:
    """JSON from the warm server, or None if nothing is listening."""
    url = f"http://{HOST}:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read() or b"{}") | {"_status": exc.code}
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return None


def health(port: int = DEFAULT_PORT) -> Optional[dict]:
    return _request(port, "/health")


def ask(messages: list, log: bool = True, port: int = DEFAULT_PORT) -> Optional[dict]:
    """The Pipeline result from a ready warm server, or None (not running / still loading / failed)."""
    result = _request(port, "/ask", {"messages": messages, "log": log}, timeout=3600)
    if result is not None and "error" in result:
        print(f"Model server error ({result['error']}); answering without it.", file=sys.stderr)
        return None
    if result is None or "_status" in result:
        return None
    return result


def stop(port: int = DEFAULT_PORT) -> bool:
    return _request(port, "/shutdown", {}) is not None


def spawn(port: int = DEFAULT_PORT, idle_minutes: float = DEFAULT_IDLE_MINUTES, log_path: Optional[Path] = None) -> None:
    """Start `cascade serve` detached from this process."""
    cmd = [sys.executable, "-m", "cli.main", "serve", "--port", str(port), "--idle-minutes", str(idle_minutes)]
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]), **kwargs)


def wait_ready(port: int = DEFAULT_PORT, timeout: float = 900.0, poll: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = health(port)
        if state and state.get("ready"):
            return True
        time.sleep(poll)
    return False
