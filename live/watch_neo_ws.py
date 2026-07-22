"""Connect to Kotak Neo websocket and print price + tvalue to the terminal.

Usage:
  .venv/bin/python -m live.watch_neo_ws
  .venv/bin/python -m live.watch_neo_ws --orders
  .venv/bin/python -m live.watch_neo_ws --raw

Ctrl+C to stop. No orders are placed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Iterable, Optional

from live.config import build_live_config, load_live_config, load_neo_credentials
from live.neo_client import make_broker


def _now_label() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _iter_feed_items(message: Any) -> Iterable[dict]:
    """Yield stock_feed item dicts from a websocket payload."""
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            return

    if not isinstance(message, dict):
        return

    if message.get("type") == "stock_feed":
        data = message.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data
        return

    # Flat tick-like payload
    if "iv" in message or "ltp" in message:
        yield message


def _extract_price(item: dict) -> Optional[float]:
    raw = item.get("iv")
    if raw is None:
        raw = item.get("ltp") or item.get("last_traded_price")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_tvalue(item: dict) -> str:
    raw = item.get("tvalue") or item.get("ltt") or item.get("last_traded_time")
    if raw is None:
        return ""
    return str(raw)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print Neo websocket price and tvalue")
    parser.add_argument(
        "--orders",
        action="store_true",
        help="Also subscribe to the order feed",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print full raw message JSON",
    )
    parser.add_argument(
        "--quiet-sdk",
        action="store_true",
        help="Silence websocket-client / SDK log noise",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet_sdk else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.quiet_sdk:
        logging.getLogger("websocket").setLevel(logging.ERROR)
        logging.getLogger("neo_api_client").setLevel(logging.WARNING)

    config_root = load_live_config()
    live_cfg = build_live_config(config_root)
    creds = load_neo_credentials()
    if not creds.consumer_key or not creds.mobile:
        print("NEO_API_KEY / NEO_MOBILE missing in .env", file=sys.stderr)
        return 1

    broker = make_broker(creds, live_cfg)
    msg_count = 0
    tick_count = 0
    running = True

    def on_message(message: Any) -> None:
        nonlocal msg_count, tick_count
        if not running:
            return
        msg_count += 1

        if args.raw:
            parsed = message
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                except json.JSONDecodeError:
                    parsed = message
            print(f"[{_now_label()}] WS #{msg_count} raw={parsed}", flush=True)

        printed = False
        for item in _iter_feed_items(message):
            price = _extract_price(item)
            if price is None:
                continue
            tick_count += 1
            tvalue = _extract_tvalue(item)
            print(f"price={price:.2f} tvalue={tvalue}", flush=True)
            printed = True

        if args.orders and isinstance(message, dict) and message.get("type") in (
            "order_feed",
            "order",
        ):
            print(f"[{_now_label()}] order_feed: {message}", flush=True)
            printed = True

        if not printed and args.raw is False:
            # keep quiet for control/no-price messages
            return

    def on_error(error: Any) -> None:
        print(f"[{_now_label()}] WS ERROR: {error}", file=sys.stderr, flush=True)

    def on_close(*_args: Any) -> None:
        print(f"[{_now_label()}] WS CLOSED", flush=True)

    print("Logging into Neo...", flush=True)
    broker.login()
    broker.set_callbacks(on_message=on_message, on_close=on_close, on_error=on_error)

    print(
        f"Subscribing market feed: {live_cfg.index_segment}|{live_cfg.index_name}",
        flush=True,
    )
    broker.start_market_feed()
    if args.orders:
        print("Subscribing order feed...", flush=True)
        broker.start_order_feed()

    print("Streaming price/tvalue... Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(
            f"\nStopped. messages={msg_count} price_ticks={tick_count}",
            flush=True,
        )
    finally:
        running = False
        try:
            broker.logout()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
