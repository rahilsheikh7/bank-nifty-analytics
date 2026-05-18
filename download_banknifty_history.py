"""
Download 1-minute OHLCV candles for the Bank Nifty index (NSE: NIFTY BANK) from 2020-01-01 through today.

Minute data is fetched in 60-day chunks (Kite limit per request), then merged.
Requires KITE_API_KEY and KITE_ACCESS_TOKEN in .env (and a valid session from generate_token.py).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY = os.environ.get("KITE_API_KEY", "").strip()
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "").strip()

# Bank Nifty spot/index on NSE (Zerodha instrument list)
TARGET_EXCHANGE = "NSE"
TARGET_TRADINGSYMBOL = "NIFTY BANK"

# Inclusive start of history window
HISTORY_START = date(2020, 1, 1)

# Kite historical API: "minute" interval allows at most ~60 days per call
MINUTE_CHUNK_DAYS = 60
# Gentle pacing between chunk calls (historical API is rate-limited)
CHUNK_SLEEP_SEC = 1


def find_bank_nifty_token(kite: KiteConnect) -> int:
    rows = kite.instruments(TARGET_EXCHANGE)
    for row in rows:
        if row.get("tradingsymbol") == TARGET_TRADINGSYMBOL:
            return int(row["instrument_token"])
    raise RuntimeError(
        f"Could not find {TARGET_TRADINGSYMBOL} on {TARGET_EXCHANGE}. "
        "Check Zerodha instrument naming or your connection."
    )


def fetch_minute_history_chunked(
    kite: KiteConnect,
    instrument_token: int,
    from_d: date,
    to_d: date,
) -> list[dict]:
    """Pull minute candles from from_d through to_d using 60-day windows."""
    all_rows: list[dict] = []
    chunk_start = from_d
    chunk_index = 0
    while chunk_start <= to_d:
        chunk_end = min(
            chunk_start + timedelta(days=MINUTE_CHUNK_DAYS - 1),
            to_d,
        )
        chunk_index += 1
        print(f"  Chunk {chunk_index}: {chunk_start} .. {chunk_end}")
        batch = kite.historical_data(
            instrument_token=instrument_token,
            from_date=chunk_start,
            to_date=chunk_end,
            interval="minute",
        )
        all_rows.extend(batch)
        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= to_d:
            time.sleep(CHUNK_SLEEP_SEC)
    return all_rows


def main() -> None:
    if not API_KEY or not ACCESS_TOKEN:
        print(
            "Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env.\n"
            "Run generate_token.py first, then copy the printed token into .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    # Smoke test
    try:
        profile = kite.profile()
        print("Connected as:", profile.get("user_name") or profile.get("user_id"))
    except Exception as exc:
        print("Profile check failed — token may be expired. Re-run generate_token.py.", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(1)

    token = find_bank_nifty_token(kite)
    to_d = date.today()
    from_d = HISTORY_START

    print(
        f"Fetching 1-minute candles for {TARGET_EXCHANGE}:{TARGET_TRADINGSYMBOL} "
        f"(token {token}) from {from_d} to {to_d} (chunked)..."
    )

    records = fetch_minute_history_chunked(kite, token, from_d, to_d)

    if not records:
        print("No rows returned. Check dates, subscription, or API limits.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="first").sort_values("date")

    out = "banknifty_1min_from2020.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
