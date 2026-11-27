"""
gateway/api/app.py

The self-hosted Cascade server. Run it on a machine with a GPU (in front
of vLLM, SGLang, Ollama or any OpenAI-compatible upstream) and point
everyone's `cascade` CLI at it as their `self_hosted` tier. It is
OpenAI-compatible, so any OpenAI client can use it too.

Every request is authenticated by an API key (`Authorization: Bearer
csk_...`). If the user has a token quota, the worst-case token count is
reserved before the upstream call and reconciled to the upstream's real
usage afterwards. The upstream's own key never leaves this server.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cascade_local.gateway.providers.base import CloudInferenceProvider, CloudUsage
from cascade_local.gateway.store import Account, GatewayStore, QuotaExceededError

DEFAULT_MAX_TOKENS_LIMIT = 4096


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None


def _estimate_tokens(messages: list[ChatMessage], max_tokens: int) -> int:
    """Worst-case reservation: ~1.3 tokens per word of prompt, plus the full completion budget."""
    return int(sum(len(m.content.split()) for m in messages) * 1.3) + 4 * len(messages) + max_tokens


def create_app(
    provider: CloudInferenceProvider,
    store: GatewayStore,
    public_model_name: str = "cascade",
    max_tokens_limit: int = DEFAULT_MAX_TOKENS_LIMIT,
) -> FastAPI:
    app = FastAPI(title="Cascade server")

    def _authenticate(authorization: Optional[str]) -> Account:
        key = (authorization or "").removeprefix("Bearer ").strip()
        account = store.authenticate(key)
        if account is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return account

    def _settle(account: Account, reserved: int, usage: Optional[CloudUsage]) -> None:
        # Without reported usage the worst-case reservation stands.
        if usage is not None:
            store.adjust(account.user_id, reserved, usage.total_tokens)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": public_model_name, "object": "model", "owned_by": "cascade"}]}

    @app.get("/v1/usage")
    def usage(authorization: Optional[str] = Header(default=None)) -> dict:
        account = _authenticate(authorization)
        current = store.account(account.user_id)
        return {"user_id": current.user_id, "used_tokens": current.used_tokens, "quota_tokens": current.quota_tokens}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
        account = _authenticate(authorization)
        if request.max_tokens > max_tokens_limit:
            raise HTTPException(status_code=400, detail=f"max_tokens exceeds limit of {max_tokens_limit}")

        messages = [m.model_dump() for m in request.messages]
        reserved = _estimate_tokens(request.messages, request.max_tokens)
        try:
            store.check_and_reserve(account.user_id, reserved)  # before the upstream call
        except QuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                _stream(account, messages, request.max_tokens, reserved, completion_id, created, include_usage),
                media_type="text/event-stream",
            )

        try:
            sample = provider.generate(messages, max_tokens=request.max_tokens)
        except Exception as exc:
            store.adjust(account.user_id, reserved, 0)
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

        usage_obj = CloudUsage(sample.prompt_tokens, sample.completion_tokens)
        _settle(account, reserved, usage_obj)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": public_model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": sample.text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage_obj.prompt_tokens,
                "completion_tokens": usage_obj.completion_tokens,
                "total_tokens": usage_obj.total_tokens,
            },
        }

    def _stream(
        account: Account,
        messages: list[dict],
        max_tokens: int,
        reserved: int,
        completion_id: str,
        created: int,
        include_usage: bool,
    ) -> Iterator[str]:
        def event(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        def chunk(delta: dict, finish_reason: Optional[str] = None) -> dict:
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": public_model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        usage: Optional[CloudUsage] = None
        try:
            gen = provider.generate_stream(messages, max_tokens=max_tokens)
            while True:
                try:
                    piece = next(gen)
                except StopIteration as stop:
                    usage = stop.value
                    break
                yield event(chunk({"content": piece}))
        except Exception as exc:
            store.adjust(account.user_id, reserved, 0)
            yield event({"error": {"message": f"upstream error: {exc}", "type": "upstream_error"}})
            yield "data: [DONE]\n\n"
            return

        _settle(account, reserved, usage)
        yield event(chunk({}, finish_reason="stop"))
        if include_usage:
            final = chunk({})
            final["choices"] = []
            final["usage"] = (
                {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                if usage else None
            )
            yield event(final)
        yield "data: [DONE]\n\n"

    return app
