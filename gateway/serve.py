"""
gateway/serve.py -- run CascadeCloud and manage its accounts.

    python -m gateway.serve create-account alice --cap 5.00
    python -m gateway.serve create-key alice          # prints the key once
    python -m gateway.serve run --port 8000

Configuration (environment variables, so secrets never land in a file
in the repo):
    CASCADE_GATEWAY_DB        SQLite path      (default ~/.cascade/gateway.db)
    CASCADE_UPSTREAM_URL      OpenAI-compatible upstream root, e.g. a vLLM
                              deployment (default http://127.0.0.1:11434, Ollama)
    CASCADE_UPSTREAM_MODEL    upstream model id (default llama3.2:3b-instruct-q4_K_M)
    CASCADE_UPSTREAM_API_KEY  upstream key, if it needs one
    CASCADE_PRICE_PER_1K      price charged per 1k tokens (default 0.50)
    STRIPE_API_KEY            enables metered Stripe billing when set
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".cascade" / "gateway.db"


def _store():
    from gateway.store import GatewayStore

    path = Path(os.environ.get("CASCADE_GATEWAY_DB", DEFAULT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    return GatewayStore(path)


def build_app():
    from gateway.api.app import DEFAULT_PRICE_PER_1K_TOKENS_USD, create_app
    from gateway.providers.http_provider import HTTPInferenceProvider

    provider = HTTPInferenceProvider(
        base_url=os.environ.get("CASCADE_UPSTREAM_URL", "http://127.0.0.1:11434"),
        api_key=os.environ.get("CASCADE_UPSTREAM_API_KEY", ""),
        model_id=os.environ.get("CASCADE_UPSTREAM_MODEL", "llama3.2:3b-instruct-q4_K_M"),
    )
    billing = None
    if os.environ.get("STRIPE_API_KEY"):
        from gateway.billing.stripe_client import StripeBillingClient

        billing = StripeBillingClient(os.environ["STRIPE_API_KEY"])
    return create_app(
        provider=provider,
        store=_store(),
        billing_client=billing,
        price_per_1k_tokens=float(os.environ.get("CASCADE_PRICE_PER_1K", DEFAULT_PRICE_PER_1K_TOKENS_USD)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gateway.serve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    acct = sub.add_parser("create-account", help="create or update an account")
    acct.add_argument("user_id")
    acct.add_argument("--cap", type=float, default=None, help="spend cap in USD (omit for none)")
    acct.add_argument("--stripe-customer", default=None)

    key = sub.add_parser("create-key", help="issue an API key for an account")
    key.add_argument("user_id")

    run = sub.add_parser("run", help="serve the gateway")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "create-account":
        _store().create_account(args.user_id, cap_usd=args.cap, stripe_customer_id=args.stripe_customer)
        print(f"account {args.user_id} ready (cap: {'none' if args.cap is None else f'${args.cap:g}'})")
    elif args.cmd == "create-key":
        print(_store().create_api_key(args.user_id))
    else:
        import uvicorn

        uvicorn.run(build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
