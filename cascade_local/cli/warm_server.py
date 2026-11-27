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

Requests for the configured model family (or for a name no Ollama model has, e.g.
"gpt-4o" from an OpenAI tool) are answered by Cascade, which echoes the requested
name back. Everything else -- other Ollama models, `ollama pull`, `ollama list`,
embeddings -- is passed through unchanged to the real Ollama (`upstream`), so
Cascade can sit on Ollama's own port 11434 (`cascade install --as-ollama`) and
plain `ollama run` works with no setting at all.

Idle behaviour: started by `cascade ask`, the server exits after
`server.idle_minutes` without a request. Started with `cascade start`
(keep_listening), it stays reachable and only unloads the models, freeing
their RAM; the next request loads them again.

With `lazy` (login auto-start) the server listens at once and loads the models
on the first request, so an idle machine holds no model in RAM.

It binds to 127.0.0.1 only and runs one request at a time (one model, one
KV cache). Nothing here talks to the network beyond localhost (NFR-05).
Standard library only.
"""

from __future__ import annotations

import gc
import http.client
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
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
CASCADE_HEADER = "X-Cascade"    # on every reply: tells a proxy loop (upstream = ourselves) apart
# Ollama endpoints Cascade never answers itself: model management and embeddings.
PASSTHROUGH_PATHS = ("/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete", "/api/embed",
                     "/api/embeddings", "/api/blobs")


class WarmServer:
    """Owns one Pipeline. `build()` must return a fresh Pipeline; it is called again
    when the config file changes, and after an idle unload."""

    def __init__(self, build: Callable[[], object], config_path: Optional[Path] = None,
                 idle_seconds: float = DEFAULT_IDLE_MINUTES * 60, port: int = DEFAULT_PORT,
                 keep_listening: bool = False, model_name: str = "cascade",
                 upstream: Optional[str] = None, lazy: bool = False) -> None:
        self._build = build
        # The real Ollama, for everything Cascade does not answer itself. Never our own port.
        self.upstream = upstream.rstrip("/") if upstream and urllib.parse.urlsplit(upstream).port != port else None
        self._upstream_models: tuple[float, set] = (-1e9, set())
        self.lazy = lazy
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
        self.ready = lazy    # accepting questions (initial load done, or loads on demand)
        self.httpd = ThreadingHTTPServer((HOST, port), self._handler())
        self.httpd.handle_error = self._handle_error

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    # ---- which requests Cascade answers --------------------------------------------

    def upstream_models(self) -> set:
        """Model names the real Ollama has (cached 30 s); empty when it is unreachable."""
        stamp, names = self._upstream_models
        if self.upstream and time.monotonic() - stamp > 30:
            try:
                with urllib.request.urlopen(self.upstream + "/api/tags", timeout=2) as resp:
                    names = {m.get("name", "") for m in json.loads(resp.read()).get("models", [])}
            except (OSError, ValueError):
                names = set()
            self._upstream_models = (time.monotonic(), names)
        return names

    def answers(self, model: Optional[str], ollama_api: bool = False) -> bool:
        """True when Cascade answers a request naming `model`; False to pass it to the real Ollama.
        Cascade answers its own model family. Other names go to Ollama on the Ollama API (so a
        missing model gets Ollama's own "pull it" reply); on the OpenAI API only names Ollama has
        do, and anything else ("gpt-4o", "any") is answered by Cascade."""
        if not model or not self.upstream:
            return True
        base = model.split(":", 1)[0].lower()
        if base in ("cascade", self.model_name.split(":", 1)[0].lower()):
            return True
        if ollama_api:
            return False
        known = self.upstream_models()
        return not (model in known or f"{model}:latest" in known)

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

    def reload_cache(self) -> bool:
        """Re-read the cache file (another process changed it). False if no pipeline is loaded;
        a pipeline built later reads the file anyway."""
        def run() -> bool:
            cache = getattr(self._pipeline, "_cache", None)
            reload = getattr(cache, "reload", None)
            if callable(reload):
                reload()
                return True
            return False
        return self._worker.submit(run).result()

    def _unload(self) -> None:
        release = getattr(self._pipeline, "release", None)
        if callable(release):
            try:
                release()  # lets the runners write their KV caches to disk first
            except Exception:
                traceback.print_exc()
        self._pipeline = None
        gc.collect()  # the models go now, not at some later collection

    def ask(self, messages: list, log: bool = True, on_text: Optional[Callable[[str], None]] = None) -> dict:
        """The Pipeline result. `on_text` (called on the pipeline thread) receives the answer
        piece by piece while the main model generates it, when the pipeline streams."""
        with self._lock:
            self._last_activity = time.monotonic()
            try:
                return self._worker.submit(self._ask, messages, log, on_text).result()
            finally:
                self._last_activity = time.monotonic()

    def _ask(self, messages: list, log: bool, on_text: Optional[Callable[[str], None]] = None) -> dict:
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
            if on_text is not None and _accepts_on_text(pipeline.run):
                return pipeline.run(messages, on_text=on_text)
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

            def end_headers(self) -> None:
                self.send_header(CASCADE_HEADER, "1")
                super().end_headers()

            def _proxy(self, body: Optional[bytes] = None) -> None:
                """Pass this request to the real Ollama and stream its reply back unchanged."""
                if not server.upstream:
                    self._reply(404, {"error": "not found"})
                    return
                url = urllib.parse.urlsplit(server.upstream)
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() in ("content-type", "accept", "authorization", "user-agent")}
                try:
                    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=3600)
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                except OSError as exc:
                    self._reply(502, {"error": f"Ollama is not reachable at {server.upstream} ({exc})"})
                    return
                if resp.getheader(CASCADE_HEADER):  # upstream is Cascade itself
                    resp.close()
                    self._reply(508, {"error": f"upstream {server.upstream} is Cascade, not Ollama"})
                    return
                self.send_response(resp.status)
                for name in ("Content-Type", "Content-Length"):
                    if resp.getheader(name):
                        self.send_header(name, resp.getheader(name))
                if not resp.getheader("Content-Length"):
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                while True:  # chunk by chunk, so `ollama pull` progress streams through
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                conn.close()

            def _upstream_json(self, path: str, raw: Optional[bytes] = None) -> Optional[dict]:
                if not server.upstream:
                    return None
                try:
                    req = urllib.request.Request(server.upstream + path, data=raw,
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        return None if resp.headers.get(CASCADE_HEADER) else json.loads(resp.read())
                except (OSError, ValueError):
                    return None

            def _reply(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, content_type: str, chunks: list[bytes]) -> None:
                self._begin_stream(content_type)
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()

            def _begin_stream(self, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def _ask_streaming(self, messages: list, content_type: str, encode: Callable[[str], bytes],
                               log: bool = True) -> dict:
                """Ask the pipeline and stream the answer as it is generated: each piece is
                written as `encode(piece)` the moment the model produces it. Answers that
                arrive whole (cache hit, small model, remote tier) go out in one piece."""
                pieces: "queue.Queue[Optional[str]]" = queue.Queue()
                box: dict = {}

                def work() -> None:
                    try:
                        box["result"] = server.ask(messages, log=log, on_text=pieces.put)
                    except BaseException as exc:  # re-raised on the handler thread
                        box["error"] = exc
                    finally:
                        pieces.put(None)

                threading.Thread(target=work, daemon=True, name="cascade-stream").start()
                streamed: list[str] = []
                piece = pieces.get()
                if piece is None and "error" in box:
                    raise box["error"]  # nothing sent yet: the caller can still reply 500
                self._begin_stream(content_type)
                while piece is not None:
                    streamed.append(piece)
                    self.wfile.write(encode(piece))
                    self.wfile.flush()
                    piece = pieces.get()
                if "error" in box:
                    raise box["error"]
                result = box["result"]
                done, text = "".join(streamed), result["text"]
                rest = text[len(done):] if text.startswith(done) else ("" if done else text)
                if rest:
                    self.wfile.write(encode(rest))
                    self.wfile.flush()
                return result

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0].rstrip("/")
                if path == "/health":
                    self._reply(200, {"ready": server.ready, "loaded": server.loaded, "pid": os.getpid()})
                elif path in ("", "/"):
                    self._reply(200, {"status": "Cascade is running"})
                elif path == "/api/version":
                    self._reply(200, self._upstream_json("/api/version") or {"version": OLLAMA_API_VERSION})
                elif path == "/api/tags":
                    self._reply(200, self._upstream_json("/api/tags")
                                or {"models": [_ollama_model_entry(server.model_name)]})
                elif path == "/api/ps":
                    ours = [_ollama_model_entry(server.model_name)] if server.loaded else []
                    self._reply(200, {"models": ours + (self._upstream_json("/api/ps") or {}).get("models", [])})
                elif path in ("/v1/models", "/models"):
                    self._reply(200, {"object": "list", "data": [
                        {"id": server.model_name, "object": "model", "created": 0, "owned_by": "cascade"}]})
                else:
                    self._proxy()

            def do_HEAD(self) -> None:  # `ollama` probes the server with HEAD /
                self.send_response(200)
                self.end_headers()

            def do_DELETE(self) -> None:  # `ollama rm`
                length = int(self.headers.get("Content-Length", 0))
                self._proxy(self.rfile.read(length) if length else None)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b""
                path = self.path.split("?", 1)[0].rstrip("/")
                if server.upstream and path.startswith(PASSTHROUGH_PATHS):
                    self._proxy(raw)
                    return
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._reply(400, {"error": "invalid JSON"})
                    return
                if path in ("/api/chat", "/api/generate", "/v1/chat/completions", "/chat/completions",
                            "/v1/completions", "/v1/embeddings") and not server.answers(body.get("model"), ollama_api=path.startswith("/api/")):
                    self._proxy(raw)  # another Ollama model: the real Ollama runs it
                    return
                try:
                    if path == "/cache/reload":
                        self._reply(200, {"reloaded": server.reload_cache()})
                    elif path == "/shutdown":
                        self._reply(200, {"stopping": True})
                        threading.Thread(target=server.httpd.shutdown, daemon=True).start()
                    elif path == "/ask":
                        if not server.ready:
                            self._reply(503, {"error": "loading"})
                            return
                        log = bool(body.get("log", True))
                        if body.get("stream"):  # ndjson: {"text": piece} lines, then {"result": ...}
                            result = self._ask_streaming(
                                body["messages"], "application/x-ndjson",
                                lambda piece: json.dumps({"text": piece}).encode() + b"\n", log=log)
                            self.wfile.write(json.dumps({"result": result}).encode() + b"\n")
                            self.wfile.flush()
                        else:
                            self._reply(200, server.ask(body["messages"], log))
                    elif path in ("/v1/chat/completions", "/chat/completions"):
                        self._openai_chat(body)
                    elif path == "/api/chat":
                        self._ollama_chat(body)
                    elif path == "/api/generate":
                        self._ollama_generate(body)
                    elif path == "/api/show":
                        name = body.get("model") or body.get("name") or server.model_name
                        if not server.answers(name, ollama_api=True):
                            self._proxy(raw)  # Ollama's own reply; "not found" makes `ollama run` pull it
                            return
                        real = None if name.split(":", 1)[0].lower() == "cascade" else self._upstream_json("/api/show", raw)
                        self._reply(200, real or _ollama_show(name))
                    elif path == "/api/pull":
                        self._reply(200, {"status": "success"})  # no Ollama to pass it to
                    else:
                        self._proxy(raw)
                except Exception as exc:  # report, keep serving
                    traceback.print_exc()  # to ~/.cascade/server.log
                    self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

            # ---- OpenAI -----------------------------------------------------

            def _openai_chat(self, body: dict) -> None:
                model = body.get("model") or server.model_name
                messages = _normalize_messages(body.get("messages", []))
                created = int(time.time())
                rid = f"chatcmpl-{os.getpid()}-{time.monotonic_ns()}"
                if body.get("stream"):
                    def chunk(delta: dict, finish: Optional[str]) -> bytes:
                        return b"data: " + json.dumps({
                            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).encode() + b"\n\n"
                    first = [True]

                    def encode(piece: str) -> bytes:
                        delta = {"role": "assistant", "content": piece} if first[0] else {"content": piece}
                        first[0] = False
                        return chunk(delta, None)
                    self._ask_streaming(messages, "text/event-stream", encode)
                    self.wfile.write(chunk({}, "stop") + b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                result = server.ask(messages)
                text = result["text"]
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
                if body.get("stream", True):  # Ollama streams unless told not to
                    def encode(piece: str) -> bytes:
                        return json.dumps({"model": model, "created_at": _now(), "done": False,
                                           "message": {"role": "assistant", "content": piece}}).encode() + b"\n"
                    result = self._ask_streaming(messages, "application/x-ndjson", encode)
                    final = {"model": model, "created_at": _now(), "message": {"role": "assistant", "content": ""},
                             "done": True, "done_reason": "stop", **_ollama_timings(result)}
                    self.wfile.write(json.dumps(final).encode() + b"\n")
                    self.wfile.flush()
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
                if body.get("stream", True):
                    def encode(piece: str) -> bytes:
                        return json.dumps({"model": model, "created_at": _now(), "response": piece,
                                           "done": False}).encode() + b"\n"
                    result = self._ask_streaming(messages, "application/x-ndjson", encode)
                    final = {"model": model, "created_at": _now(), "response": "", "done": True,
                             "done_reason": "stop", **_ollama_timings(result)}
                    self.wfile.write(json.dumps(final).encode() + b"\n")
                    self.wfile.flush()
                    return
                result = server.ask(messages)
                final = {"model": model, "created_at": _now(), "response": result["text"], "done": True,
                         "done_reason": "stop", **_ollama_timings(result)}
                partial = {"model": model, "created_at": _now(), "response": result["text"], "done": False}
                if body.get("stream", True):
                    final = final | {"response": ""}
                self._ollama_reply(body, final, partial)

        return Handler


# ---- API helpers --------------------------------------------------------------------


def _accepts_on_text(run: Callable) -> bool:
    try:
        return "on_text" in inspect.signature(run).parameters
    except (TypeError, ValueError):
        return False



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


def ask(messages: list, log: bool = True, port: int = DEFAULT_PORT,
        on_text: Optional[Callable[[str], None]] = None) -> Optional[dict]:
    """The Pipeline result from a ready warm server, or None (not running / still loading / failed).
    With `on_text`, the answer is streamed: each piece is passed to it as the model writes it, and
    together the pieces are the whole answer."""
    if on_text is not None:
        return _ask_streamed(messages, log, port, on_text)
    result = _request(port, "/ask", {"messages": messages, "log": log}, timeout=3600)
    if result is not None and "error" in result:
        print(f"Model server error ({result['error']}); answering without it.", file=sys.stderr)
        return None
    if result is None or "_status" in result:
        return None
    return result


def _ask_streamed(messages: list, log: bool, port: int, on_text: Callable[[str], None]) -> Optional[dict]:
    req = urllib.request.Request(f"http://{HOST}:{port}/ask", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"messages": messages, "log": log, "stream": True}).encode())
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for line in resp:
                item = json.loads(line)
                if "text" in item:
                    on_text(item["text"])
                elif "result" in item:
                    return item["result"]
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read() or b"{}")
        if "error" in body and body.get("error") != "loading":
            print(f"Model server error ({body['error']}); answering without it.", file=sys.stderr)
        return None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError):
        return None
    return None


def reload_cache(port: int = DEFAULT_PORT) -> bool:
    """Tell a running server to re-read the cache file. False if none is running."""
    return _request(port, "/cache/reload", {}) is not None


def stop(port: int = DEFAULT_PORT) -> bool:
    return _request(port, "/shutdown", {}) is not None


def spawn(port: int = DEFAULT_PORT, idle_minutes: float = DEFAULT_IDLE_MINUTES, log_path: Optional[Path] = None,
          keep_listening: bool = False, lazy: bool = False) -> None:
    """Start `cascade serve` detached from this process."""
    cmd = [sys.executable, "-m", "cascade_local.cli.main", "serve", "--port", str(port), "--idle-minutes", str(idle_minutes)]
    if keep_listening:
        cmd.append("--keep-listening")
    if lazy:
        cmd.append("--lazy")
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]), **kwargs)  # holds cascade_local


def wait_ready(port: int = DEFAULT_PORT, timeout: float = 900.0, poll: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = health(port)
        if state and state.get("ready"):
            return True
        time.sleep(poll)
    return False
