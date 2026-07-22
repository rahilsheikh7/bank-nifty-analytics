"""Connect to Kite websocket and print streamed ticks to the terminal.

Usage:
  .venv/bin/python -m live.watch_kite_ws
  .venv/bin/python -m live.watch_kite_ws --mode full
  .venv/bin/python -m live.watch_kite_ws --token 260105 --compact

Ctrl+C to stop. No orders are placed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

from download_banknifty_history import TARGET_EXCHANGE, TARGET_TRADINGSYMBOL, find_bank_nifty_token


def _now_label() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _tick_summary(tick: dict[str, Any]) -> str:
    token = tick.get("instrument_token", "")
    price = tick.get("last_price", "")
    exchange_ts = tick.get("exchange_timestamp") or tick.get("last_trade_time") or ""
    ohlc = tick.get("ohlc") or {}
    parts = [f"token={token}", f"price={price}", f"exchange_timestamp={exchange_ts}"]
    if ohlc:
        parts.append(
            "ohlc="
            f"O:{ohlc.get('open', '')} "
            f"H:{ohlc.get('high', '')} "
            f"L:{ohlc.get('low', '')} "
            f"C:{ohlc.get('close', '')}"
        )
    return " ".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print Kite websocket ticks")
    parser.add_argument(
        "--token",
        type=int,
        default=0,
        help="Instrument token to subscribe. Default: resolve NSE:NIFTY BANK.",
    )
    parser.add_argument(
        "--mode",
        choices=("ltp", "quote", "full"),
        default="full",
        help="Kite websocket mode.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact JSON for each tick.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only token/price/timestamp/OHLC summary, not full JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        print("KITE_API_KEY / KITE_ACCESS_TOKEN missing in .env", file=sys.stderr)
        return 1

    token = int(args.token)
    if token <= 0:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        try:
            kite.profile()
            token = find_bank_nifty_token(kite)
        except Exception as exc:
            print(f"Could not resolve {TARGET_EXCHANGE}:{TARGET_TRADINGSYMBOL}: {exc}", file=sys.stderr)
            return 1

    ticker = KiteTicker(api_key, access_token)
    tick_count = 0

    def on_ticks(ws: KiteTicker, ticks: list[dict[str, Any]]) -> None:
        nonlocal tick_count
        for tick in ticks:
            tick_count += 1
            print(f"\n[{_now_label()}] tick #{tick_count} {_tick_summary(tick)}", flush=True)
            if not args.summary_only:
                if args.compact:
                    print(json.dumps(tick, separators=(",", ":"), default=_json_default), flush=True)
                else:
                    print(json.dumps(tick, indent=2, default=_json_default), flush=True)

    def on_connect(ws: KiteTicker, response: Any) -> None:
        print(
            f"Connected to Kite websocket. Subscribing token={token} mode={args.mode}. "
            f"response={response}",
            flush=True,
        )
        ws.subscribe([token])
        mode_value = {
            "ltp": ws.MODE_LTP,
            "quote": ws.MODE_QUOTE,
            "full": ws.MODE_FULL,
        }[args.mode]
        ws.set_mode(mode_value, [token])

    def on_close(_ws: KiteTicker, code: int, reason: str) -> None:
        print(f"[{_now_label()}] Kite WS closed code={code} reason={reason}", flush=True)

    def on_error(_ws: KiteTicker, code: int, reason: str) -> None:
        print(f"[{_now_label()}] Kite WS error code={code} reason={reason}", file=sys.stderr, flush=True)

    def on_reconnect(_ws: KiteTicker, attempts_count: int) -> None:
        print(f"[{_now_label()}] Kite WS reconnect attempt={attempts_count}", flush=True)

    def on_noreconnect(_ws: KiteTicker) -> None:
        print(f"[{_now_label()}] Kite WS reconnect abandoned", file=sys.stderr, flush=True)

    ticker.on_ticks = on_ticks
    ticker.on_connect = on_connect
    ticker.on_close = on_close
    ticker.on_error = on_error
    ticker.on_reconnect = on_reconnect
    ticker.on_noreconnect = on_noreconnect

    print(
        f"Connecting to Kite websocket for {TARGET_EXCHANGE}:{TARGET_TRADINGSYMBOL} "
        f"(token={token})...",
        flush=True,
    )
    try:
        ticker.connect(threaded=False)
    except KeyboardInterrupt:
        print(f"\nStopped. ticks={tick_count}", flush=True)
        try:
            ticker.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
