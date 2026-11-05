"""
gateway/api/app.py

CascadeCloud's server (FR-07.1, FR-07.2, FR-07.5). OpenAI-compatible, so
CascadeCloudBackend -- or any OpenAI client -- can call it unchanged.

Billing enforcement lives here, server-side, because a client-side cap is
trivially bypassable. Every request is authenticated by an API key
(`Authorization: Bearer csk_...`), checked against the account's spend
cap before the billable upstream call, reconciled to the upstream's real
token usage afterwards, and reported to Stripe when a billing client is
configured. The upstream (a vLLM/SGLang deployment or any wholesale
provider) is reached through the injected CloudInferenceProvider; its key
never leaves this server.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gateway.billing.meter import SpendCapExceededError
from gateway.billing.stripe_client import BillingClient
from gateway.providers.base import CloudInferenceProvider, CloudUsage
from gateway.store import Account, GatewayStore

DEFAULT_PRICE_PER_1K_TOKENS_USD = 0.50  # placeholder, see gateway/backend.py
DEFAULT_MAX_TOKENS_LIMIT = 4096
USAGE_METER_EVENT = "cascade_tokens"


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


def _cost(tokens: int, price_per_1k: float) -> float:
    return (tokens / 1000.0) * price_per_1k


def _estimate_tokens(messages: list[ChatMessage], max_tokens: int) -> int:
    return int(sum(len(m.content.split()) for m in messages) * 1.3) + 4 * len(messages) + max_tokens


def create_app(
    provider: CloudInferenceProvider,
    store: GatewayStore,
    billing_client: Optional[BillingClient] = None,
    price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS_USD,
    public_model_name: str = "cascade",
    max_tokens_limit: int = DEFAULT_MAX_TOKENS_LIMIT,
) -> FastAPI:
    app = FastAPI(title="CascadeCloud Gateway")

    def _authenticate(authorization: Optional[str]) -> Account:
        key = (authorization or "").removeprefix("Bearer ").strip()
        account = store.authenticate(key)
        if account is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return account

    def _bill(account: Account, estimated_cost: float, usage: Optional[CloudUsage]) -> float:
        """Reconcile the reservation to real usage; returns the charged cost."""
        if usage is None:
            return estimated_cost  # no usage reported: the worst-case reservation stands
        actual = _cost(usage.total_tokens, price_per_1k_tokens)
        store.adjust(account.user_id, estimated_cost, actual)
        if billing_client is not None:
            billing_client.record_usage(account.stripe_customer_id or account.user_id, USAGE_METER_EVENT, usage.total_tokens)
        return actual

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
        return {"user_id": current.user_id, "spent_usd": round(current.spent_usd, 6), "cap_usd": current.cap_usd}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
        account = _authenticate(authorization)
        if request.max_tokens > max_tokens_limit:
            raise HTTPException(status_code=400, detail=f"max_tokens exceeds limit of {max_tokens_limit}")

        messages = [m.model_dump() for m in request.messages]
        estimated_cost = _cost(_estimate_tokens(request.messages, request.max_tokens), price_per_1k_tokens)
        try:
            store.check_and_reserve(account.user_id, estimated_cost)  # FR-07.2, before the billable call
        except SpendCapExceededError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if request.stream:
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            return StreamingResponse(
                _stream(account, messages, request.max_tokens, estimated_cost, completion_id, created, include_usage),
                media_type="text/event-stream",
            )

        try:
            sample = provider.generate(messages, max_tokens=request.max_tokens)
        except Exception as exc:
            store.adjust(account.user_id, estimated_cost, 0.0)
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc

        usage_obj = CloudUsage(sample.prompt_tokens, sample.completion_tokens)
        charged = _bill(account, estimated_cost, usage_obj)
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
            "cascade": {"cost_usd": round(charged, 6)},
        }

    def _stream(
        account: Account,
        messages: list[dict],
        max_tokens: int,
        estimated_cost: float,
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
            store.adjust(account.user_id, estimated_cost, 0.0)
            yield event({"error": {"message": f"upstream error: {exc}", "type": "upstream_error"}})
            yield "data: [DONE]\n\n"
            return

        charged = _bill(account, estimated_cost, usage)
        yield event(chunk({}, finish_reason="stop"))
        if include_usage:
            final = chunk({})
            final["choices"] = []
            final["usage"] = (
                {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
                if usage else None
            )
            final["cascade"] = {"cost_usd": round(charged, 6)}
            yield event(final)
        yield "data: [DONE]\n\n"

    return app
