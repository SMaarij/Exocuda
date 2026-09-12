"""
gateway/api/app.py

Track C, Tasks 5 & 10 (FR-07.1, FR-07.2, FR-07.5).

ARCHITECTURE NOTE -- where billing actually has to live:
    `gateway/backend.py`'s CascadeCloudBackend is what a user's own
    Pipeline (running on their machine, per docs/TRACK_B_HANDOFF_FOR_C.md)
    calls when it needs the cloud tier. But its `check_and_reserve()`
    spend-cap check running on the USER's own machine is not something
    a hostile or careless client can be trusted to actually enforce --
    a modified client could just skip the check. The real enforcement
    has to happen server-side, on infrastructure Cascade controls.

    This FastAPI app IS that server. It's what CascadeCloudBackend's
    HTTPInferenceProvider actually points `base_url` at in production
    (not directly at a wholesale GPU provider's raw endpoint) -- this
    app holds the wholesale provider's real API key server-side, checks
    the spend cap per authenticated user here (where it can't be
    bypassed), calls the wholesale provider, records billing via
    Stripe, and returns an OpenAI-compatible response back to the
    client's CascadeCloudBackend.

`create_app()` is a factory, not a module-level singleton -- every
dependency (the wholesale provider, the billing client, pricing) is
injected, matching the pattern every other Track C component uses so
this is fully testable without any real network/Stripe call (NFR-09).
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from gateway.billing.meter import SpendCapExceededError, UsageMeter
from gateway.billing.stripe_client import BillingClient
from gateway.providers.base import CloudInferenceProvider

DEFAULT_PRICE_PER_1K_TOKENS_USD = 0.50  # see gateway/backend.py's identical, identically-flagged placeholder


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 256


class GenerateResponse(BaseModel):
    text: str
    tok_s: float
    latency_ms: float
    est_cost_usd: float


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _cost_for_tokens(total_tokens: int, price_per_1k_tokens: float) -> float:
    return (total_tokens / 1000.0) * price_per_1k_tokens


def create_app(
    provider: CloudInferenceProvider,
    billing_client: Optional[BillingClient] = None,
    price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS_USD,
    default_cap_usd: Optional[float] = None,
) -> FastAPI:
    app = FastAPI(title="CascadeCloud Gateway")

    # One UsageMeter per authenticated user. A real deployment would
    # back this with a database (so caps survive a restart and work
    # across multiple gateway instances) -- this in-memory registry is
    # correct for a single-process deployment and is the seam a future
    # persistent-store swap plugs into without changing any route below.
    meters: dict[str, UsageMeter] = {}

    def _meter_for(user_id: str) -> UsageMeter:
        if user_id not in meters:
            meters[user_id] = UsageMeter(cap_usd=default_cap_usd)
        return meters[user_id]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=GenerateResponse)
    def generate(request: GenerateRequest, x_cascade_user: str = Header(...)) -> GenerateResponse:
        meter = _meter_for(x_cascade_user)
        prompt = _messages_to_prompt(request.messages)

        estimated_tokens = int(len(prompt.split()) * 1.3) + request.max_tokens
        estimated_cost = _cost_for_tokens(estimated_tokens, price_per_1k_tokens)

        try:
            meter.check_and_reserve(estimated_cost)  # FR-07.2 -- the real, server-side enforcement
        except SpendCapExceededError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc

        try:
            sample = provider.generate(prompt, max_tokens=request.max_tokens)
        except Exception as exc:
            meter.adjust(estimated_cost, 0.0)
            raise HTTPException(status_code=502, detail=f"provider error: {exc}") from exc

        actual_tokens = sample.prompt_tokens + sample.completion_tokens
        actual_cost = _cost_for_tokens(actual_tokens, price_per_1k_tokens)
        meter.adjust(estimated_cost, actual_cost)

        if billing_client is not None:
            billing_client.record_usage(x_cascade_user, "cascade_tokens", actual_tokens)

        tok_s = sample.completion_tokens / sample.elapsed_s if sample.elapsed_s > 0 else 0.0
        return GenerateResponse(
            text=sample.text,
            tok_s=round(tok_s, 3),
            latency_ms=round(sample.elapsed_s * 1000, 2),
            est_cost_usd=round(actual_cost, 6),
        )

    return app
