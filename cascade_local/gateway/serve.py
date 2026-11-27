"""
gateway/serve.py -- run a self-hosted Cascade server and manage its users.

    python -m cascade_local.gateway.serve create-user alice --quota 2000000
    python -m cascade_local.gateway.serve create-key alice          # prints the key once
    python -m cascade_local.gateway.serve run --host 0.0.0.0 --port 8000

Configuration (environment variables, so secrets never land in a file in
the repo):
    CASCADE_SERVER_DB         SQLite path (default ~/.cascade/server.db)
    CASCADE_UPSTREAM_URL      OpenAI-compatible upstream root, e.g. a vLLM
                              deployment (default http://127.0.0.1:11434, Ollama)
    CASCADE_UPSTREAM_MODEL    upstream model id (default llama3.2:3b-instruct-q4_K_M)
    CASCADE_UPSTREAM_API_KEY  upstream key, if it needs one
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".cascade" / "server.db"


def _store():
    from cascade_local.gateway.store import GatewayStore

    path = Path(os.environ.get("CASCADE_SERVER_DB", DEFAULT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    return GatewayStore(path)


def build_app():
    from cascade_local.gateway.api.app import create_app
    from cascade_local.gateway.providers.http_provider import HTTPInferenceProvider

    provider = HTTPInferenceProvider(
        base_url=os.environ.get("CASCADE_UPSTREAM_URL", "http://127.0.0.1:11434"),
        api_key=os.environ.get("CASCADE_UPSTREAM_API_KEY", ""),
        model_id=os.environ.get("CASCADE_UPSTREAM_MODEL", "llama3.2:3b-instruct-q4_K_M"),
    )
    return create_app(provider=provider, store=_store())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cascade_local.gateway.serve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    user = sub.add_parser("create-user", help="create a user, or change an existing user's quota")
    user.add_argument("user_id")
    user.add_argument("--quota", type=int, default=None, help="token quota (omit for unlimited)")

    key = sub.add_parser("create-key", help="issue an API key for a user (printed once)")
    key.add_argument("user_id")

    revoke = sub.add_parser("revoke-key", help="revoke an API key")
    revoke.add_argument("key")

    reset = sub.add_parser("reset-usage", help="set a user's used tokens back to 0")
    reset.add_argument("user_id")

    usage = sub.add_parser("usage", help="show a user's token usage")
    usage.add_argument("user_id")

    run = sub.add_parser("run", help="serve")
    run.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept other machines (put it behind HTTPS)")
    run.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "create-user":
        _store().create_user(args.user_id, quota_tokens=args.quota)
        print(f"user {args.user_id} ready (quota: {'unlimited' if args.quota is None else f'{args.quota} tokens'})")
    elif args.cmd == "create-key":
        print(_store().create_api_key(args.user_id))
    elif args.cmd == "revoke-key":
        if not _store().revoke_api_key(args.key):
            print("no such active key", file=sys.stderr)
            return 1
        print("revoked")
    elif args.cmd == "reset-usage":
        _store().reset_usage(args.user_id)
        print(f"usage reset for {args.user_id}")
    elif args.cmd == "usage":
        account = _store().account(args.user_id)
        if account is None:
            print(f"no user {args.user_id!r}", file=sys.stderr)
            return 1
        quota = "unlimited" if account.quota_tokens is None else account.quota_tokens
        print(f"{account.user_id}: {account.used_tokens} tokens used (quota: {quota})")
    else:
        import uvicorn

        uvicorn.run(build_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
