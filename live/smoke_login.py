"""Smoke test: Neo SDK login + limits (no trading)."""
from __future__ import annotations

import logging
import sys

from live.config import build_live_config, load_live_config, load_neo_credentials
from live.neo_client import make_broker


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_root = load_live_config()
    live_cfg = build_live_config(config_root)
    creds = load_neo_credentials()
    if not creds.consumer_key:
        print("NEO_API_KEY missing in .env", file=sys.stderr)
        return 1
    if not creds.mobile:
        print("NEO_MOBILE missing in .env", file=sys.stderr)
        return 1

    broker = make_broker(creds, live_cfg)
    broker.login()
    limits = broker.limits()
    print("Login OK. Limits response:")
    print(limits)
    broker.logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
