"""
Kotak Neo session token generator (tradeApiLogin + tradeApiValidate).

Reads credentials from .env:
  NEO_API_KEY, NEO_MOBILE, NEO_CLIENT_CODE, NEO_MPIN
  NEO_TOTP_SECRET (recommended for automation)

Writes full login + validate responses to token.json in the project root.

Usage:
  python generate_neo_session.py
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv

from live.config import load_neo_credentials
from live.neo_place_order import DEFAULT_TOKEN_FILE
from live.neo_session import ensure_trade_session, is_token_stale, resolve_totp


def _totp_provider(creds):
    if creds.has_totp_secret:
        return lambda: resolve_totp(creds)
    def _prompt() -> str:
        totp = input("Enter 6-digit TOTP from your authenticator: ").strip()
        if not totp or len(totp) != 6 or not totp.isdigit():
            raise RuntimeError("Invalid TOTP")
        return totp
    return _prompt


def main() -> int:
    load_dotenv()
    creds = load_neo_credentials()

    if is_token_stale(DEFAULT_TOKEN_FILE):
        print("Token missing or from a previous trading day — generating fresh session...")
    else:
        print("Regenerating trade session...")

    try:
        session = ensure_trade_session(
            creds, force=True, totp_provider=_totp_provider(creds),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"\nSaved to {DEFAULT_TOKEN_FILE}")
    print("Use session.auth_token as header Auth and session.sid as header Sid for trading APIs.")
    print(f"Place order base URL: {session.base_url}/quick/order/rule/ms/place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
