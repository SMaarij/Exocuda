"""
cli/warm_server.py -- keep the Pipeline loaded, and let other programs use it.

Every `cascade ask` is a new process. With the llama.cpp runner that meant
reloading the 7B model (~34 s on a CPU laptop), the draft model and the
embedding model (~2.4 s) before every answer. Like Ollama's keep-alive,
`cascade ask` starts this server in the background on first use and sends
questions to it.

The same server speaks the two APIs local AI tools already use, so they run
on Cascade (cache, small/big routing, speculative decoding, lower RAM) with
no change to the tool:

    Ollama API   /api/chat /api/generate /api/tags /api/show /api/ps /api/version
                 -> `OLLAMA_HOST=127.0.0.1:8765 ollama run qwen2.5-coder`
    OpenAI API   /v1/chat/completions /v1/models
                 -> base URL http://127.0.0.1:8765/v1, any API key

Whatever model a client names, Cascade answers with its configured models
(`cascade config show`) and echoes the requested name back.

Idle behaviour: started by `cascade ask`, the server exits after
`server.idle_minutes` without a request. Started with `cascade start`
(keep_listening), it stays reachable and only unloads the models, freeing
their RAM; the next request loads them again.

It binds to 127.0.0.1 only and runs one request at a time (one model, one
KV cache). Nothing here talks to the network beyond localhost (NFR-05).
Standard library only.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

DEFAULT_PORT = 8765
DEFAULT_IDLE_MINUTES = 10
HOST = "127.0.0.1"
OLLAMA_API_VERSION = "0.40.15"  # what /api/version reports; Ollama clients only compare it


class WarmServer:
    """Owns one Pipeline. `build()` must return a fresh Pipeline; it is called again
    when the config file changes, and after an idle unload."""

    def __init__(self, build: Callable[[], object], config_path: Optional[Path] = None,
                 idle_seconds: float = DEFAULT_IDLE_MINUTES * 60, port: int = DEFAULT_PORT,
                 keep_listening: bool = False, model_name: str = "cascade") -> None:
        self._build = build
        self._config_path = config_path
        self._idle_seconds = idle_seconds
        self._keep_listening = keep_listening
        self.model_name = model_name
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
        self.ready = False   # accepting questions (initial load done)
        self.httpd = ThreadingHTTPServer((HOST, port), self._handler())
        self.httpd.handle_error = self._handle_error

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

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

    def unload(self) -> None:
        self._worker.submit(self._unload).result()

    def _unload(self) -> None:
        self._pipeline = None
        gc.collect()  # the models go now, not at some later collection

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
        if self._pipeline is None:  # first use, config change, or unloaded while idle
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

    def _on_idle(self) -> bool:
        """Free the RAM after idle_seconds. Returns True when the server should exit."""
        if not self._keep_listening:
            return True
        if self.loaded and not self._lock.locked():
            self.unload()
        return False

    def serve_forever(self, load_in_background: bool = False) -> None:
        """Serve until idle (or forever with keep_listening). With load_in_background,
        health checks answer {"ready": false} while the models load instead of timing out."""
        if load_in_background:
            threading.Thread(target=self.load, daemon=True).start()

        def watchdog() -> None:
            while True:
                time.sleep(min(30.0, max(1.0, self._idle_seconds / 4)))
                if self.idle_for() > self._idle_seconds and self._on_idle():
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

            def _stream(self, content_type: str, chunks: list[bytes]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                self.close_connection = True

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path == "/health":
                    self._reply(200, {"ready": server.ready, "loaded": server.loaded, "pid": os.getpid()})
                elif path in ("", "/"):
                    self._reply(200, {"status": "Cascade is running"})
                elif path == "/api/version":
                    self._reply(200, {"version": OLLAMA_API_VERSION})
                elif path == "/api/tags":
                    self._reply(200, {"models": [_ollama_model_entry(server.model_name)]})
                elif path == "/api/ps":
                    self._reply(200, {"models": [_ollama_model_entry(server.model_name)] if server.loaded else []})
                elif path in ("/v1/models", "/models"):
                    self._reply(200, {"object": "list", "data": [
                        {"id": server.model_name, "object": "model", "created": 0, "owned_by": "cascade"}]})
                else:
                    self._reply(404, {"error": "not found"})

            def do_HEAD(self) -> None:  # `ollama` probes the server with HEAD /
                self.send_response(200)
                self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._reply(400, {"error": "invalid JSON"})
                    return
                path = self.path.split("?", 1)[0].rstrip("/")
                try:
                    if path == "/shutdown":
                        self._reply(200, {"stopping": True})
                        threading.Thread(target=server.httpd.shutdown, daemon=True).start()
                    elif path == "/ask":
                        if not server.ready:
                            self._reply(503, {"error": "loading"})
                            return
                        self._reply(200, server.ask(body["messages"], bool(body.get("log", True))))
                    elif path in ("/v1/chat/completions", "/chat/completions"):
                        self._openai_chat(body)
                    elif path == "/api/chat":
                        self._ollama_chat(body)
                    elif path == "/api/generate":
                        self._ollama_generate(body)
                    elif path == "/api/show":
                        self._reply(200, _ollama_show(body.get("model") or body.get("name") or server.model_name))
                    elif path == "/api/pull":
                        self._reply(200, {"status": "success"})  # Cascade manages its own models
                    else:
                        self._reply(404, {"error": "not found"})
                except Exception as exc:  # report, keep serving
                    traceback.print_exc()  # to ~/.cascade/server.log
                    self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

            # ---- OpenAI -----------------------------------------------------

            def _openai_chat(self, body: dict) -> None:
                model = body.get("model") or server.model_name
                result = server.ask(_normalize_messages(body.get("messages", [])))
                text, created = result["text"], int(time.time())
                rid = f"chatcmpl-{os.getpid()}-{time.monotonic_ns()}"
                if body.get("stream"):
                    def chunk(delta: dict, finish: Optional[str]) -> bytes:
                        return b"data: " + json.dumps({
                            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).encode() + b"\n\n"
                    self._stream("text/event-stream", [
                        chunk({"role": "assistant", "content": text}, None), chunk({}, "stop"), b"data: [DONE]\n\n"])
                    return
                self._reply(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "cascade": _cascade_meta(result)})

            # ---- Ollama -----------------------------------------------------

            def _ollama_reply(self, body: dict, final: dict, partial: Optional[dict] = None) -> None:
                if body.get("stream", True):  # Ollama streams unless told not to
                    lines = ([partial] if partial else []) + [final]
                    self._stream("application/x-ndjson", [json.dumps(line).encode() + b"\n" for line in lines])
                else:
                    self._reply(200, final)

            def _ollama_chat(self, body: dict) -> None:
                model = body.get("model") or server.model_name
                messages = _normalize_messages(body.get("messages", []))
                if not messages:  # `ollama run` loads the model with an empty chat first
                    self._ollama_reply(body, {"model": model, "created_at": _now(), "done": True, "done_reason": "load",
                                              "message": {"role": "assistant", "content": ""}})
                    return
                result = server.ask(messages)
                message = {"role": "assistant", "content": result["text"]}
                final = {"model": model, "created_at": _now(), "message": message, "done": True, "done_reason": "stop",
                         **_ollama_timings(result)}
                partial = {"model": model, "created_at": _now(), "message": message, "done": False}
                if body.get("stream", True):
                    final = final | {"message": {"role": "assistant", "content": ""}}
                self._ollama_reply(body, final, partial)

            def _ollama_generate(self, body: dict) -> None:
                model = body.get("model") or server.model_name
                prompt = body.get("prompt") or ""
                if not prompt.strip():  # a load request
                    self._ollama_reply(body, {"model": model, "created_at": _now(), "response": "", "done": True,
                                              "done_reason": "load"})
                    return
                messages = ([{"role": "system", "content": body["system"]}] if body.get("system") else []) + \
                    [{"role": "user", "content": prompt}]
                result = server.ask(messages)
                final = {"model": model, "created_at": _now(), "response": result["text"], "done": True,
                         "done_reason": "stop", **_ollama_timings(result)}
                partial = {"model": model, "created_at": _now(), "response": result["text"], "done": False}
                if body.get("stream", True):
                    final = final | {"response": ""}
                self._ollama_reply(body, final, partial)

        return Handler


# ---- API helpers --------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_messages(messages: list) -> list[dict]:
    """OpenAI content may be a list of parts; keep the text ones. Drop empty turns."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content:
            out.append({"role": m.get("role", "user"), "content": str(content)})
    return out


def _cascade_meta(result: dict) -> dict:
    return {k: result.get(k) for k in ("tier", "cache_hit", "tok_s", "latency_ms", "escalated")}


def _ollama_timings(result: dict) -> dict:
    ns = int(float(result.get("latency_ms") or 0) * 1e6)
    return {"total_duration": ns, "load_duration": 0, "eval_duration": ns, "eval_count": 0,
            "prompt_eval_count": 0, "prompt_eval_duration": 0}


def _ollama_model_entry(name: str) -> dict:
    return {"name": name, "model": name, "modified_at": _now(), "size": 0, "digest": "cascade",
            "details": {"format": "gguf", "family": "cascade", "families": ["cascade"],
                        "parameter_size": "", "quantization_level": ""}}


def _ollama_show(name: str) -> dict:
    return {"modelfile": "", "parameters": "", "template": "{{ .Prompt }}", "license": "",
            "details": _ollama_model_entry(name)["details"], "model_info": {},
            "capabilities": ["completion"], "modified_at": _now()}


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


def spawn(port: int = DEFAULT_PORT, idle_minutes: float = DEFAULT_IDLE_MINUTES, log_path: Optional[Path] = None,
          keep_listening: bool = False) -> None:
    """Start `cascade serve` detached from this process."""
    cmd = [sys.executable, "-m", "cli.main", "serve", "--port", str(port), "--idle-minutes", str(idle_minutes)]
    if keep_listening:
        cmd.append("--keep-listening")
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
