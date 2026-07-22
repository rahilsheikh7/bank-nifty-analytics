"""Poll Kite historical API for the just-completed 1-minute candle.

Usage:
  .venv/bin/python -m live.kite_last_minute_poll --delay 3
  .venv/bin/python -m live.kite_last_minute_poll --delay 5 --once
  .venv/bin/python -m live.kite_last_minute_poll --delay 3 --token 260105

At HH:MM:00 + delay, this fetches the candle labeled HH:MM-1 because the
HH:MM candle has just started and is not complete yet. For example, at
10:55:03 it checks the 10:54:00 candle.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from download_banknifty_history import TARGET_EXCHANGE, TARGET_TRADINGSYMBOL, find_bank_nifty_token

IST = "Asia/Kolkata"


def _now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def _sleep_until_next_poll(delay_sec: float) -> pd.Timestamp:
    now = _now_ist()
    next_minute = now.floor("min") + pd.Timedelta(minutes=1)
    poll_at = next_minute + pd.Timedelta(seconds=delay_sec)
    sleep_for = max(0.0, (poll_at - now).total_seconds())
    time.sleep(sleep_for)
    return poll_at


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S%z")


def _load_kite() -> KiteConnect:
    load_dotenv()
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        raise RuntimeError("KITE_API_KEY / KITE_ACCESS_TOKEN missing in .env")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    kite.profile()
    return kite


def _resolve_token(kite: KiteConnect, token_arg: int) -> int:
    if token_arg > 0:
        return int(token_arg)
    return find_bank_nifty_token(kite)


def _fetch_recent_minute_candles(
    kite: KiteConnect,
    token: int,
    target_minute: pd.Timestamp,
    *,
    lookback_minutes: int,
) -> list[dict]:
    from_ts = target_minute - pd.Timedelta(minutes=max(1, lookback_minutes) - 1)
    to_ts = target_minute
    return kite.historical_data(
        instrument_token=token,
        from_date=from_ts.to_pydatetime(),
        to_date=to_ts.to_pydatetime(),
        interval="minute",
    )


def _find_candle(records: list[dict], target_minute: pd.Timestamp) -> Optional[dict]:
    target = target_minute.floor("min")
    for row in records:
        ts = pd.Timestamp(row.get("date"))
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        if ts.floor("min") == target:
            return row
    return None


def _print_candle(row: dict, *, target_minute: pd.Timestamp, poll_at: pd.Timestamp) -> None:
    ts = pd.Timestamp(row.get("date"))
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    else:
        ts = ts.tz_convert(IST)
    print(
        "poll_at={poll_at} target={target} candle={candle} "
        "O={open} H={high} L={low} C={close} V={volume}".format(
            poll_at=_fmt_ts(poll_at),
            target=_fmt_ts(target_minute),
            candle=_fmt_ts(ts),
            open=row.get("open"),
            high=row.get("high"),
            low=row.get("low"),
            close=row.get("close"),
            volume=row.get("volume"),
        ),
        flush=True,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Poll Kite for just-completed 1m candles")
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds after each minute boundary to poll, e.g. 3 means HH:MM:03.",
    )
    parser.add_argument(
        "--token",
        type=int,
        default=0,
        help="Instrument token. Default resolves NSE:NIFTY BANK.",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=5,
        help="Fetch this many recent candles so late/revised candles are visible.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once at the next minute boundary + delay, then exit.",
    )
    args = parser.parse_args(argv)

    if args.delay < 0:
        print("--delay must be >= 0", file=sys.stderr)
        return 2

    kite = _load_kite()
    token = _resolve_token(kite, args.token)
    print(
        f"Polling Kite historical minute candles for {TARGET_EXCHANGE}:{TARGET_TRADINGSYMBOL} "
        f"token={token} delay={args.delay}s lookback={args.lookback_minutes}m",
        flush=True,
    )

    while True:
        scheduled_poll_at = _sleep_until_next_poll(args.delay)
        actual_poll_at = _now_ist()
        # At 10:55:03, the completed minute candle is labeled 10:54:00.
        target_minute = scheduled_poll_at.floor("min") - pd.Timedelta(minutes=1)

        try:
            records = _fetch_recent_minute_candles(
                kite,
                token,
                target_minute,
                lookback_minutes=args.lookback_minutes,
            )
        except Exception as exc:
            print(
                f"poll_at={_fmt_ts(actual_poll_at)} target={_fmt_ts(target_minute)} ERROR {exc}",
                file=sys.stderr,
                flush=True,
            )
            if args.once:
                return 1
            continue

        candle = _find_candle(records, target_minute)
        if candle is None:
            print(
                f"poll_at={_fmt_ts(actual_poll_at)} target={_fmt_ts(target_minute)} "
                f"NOT_AVAILABLE returned={len(records)}",
                flush=True,
            )
        else:
            _print_candle(candle, target_minute=target_minute, poll_at=actual_poll_at)

        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
