"""Fetch Bank Nifty 1-min history from Zerodha Kite for live-bot warmup."""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from download_banknifty_history import fetch_minute_history_chunked, find_bank_nifty_token
from live.bar_cache import _normalize_df

logger = logging.getLogger("live.kite_warmup")

IST = "Asia/Kolkata"


def _trading_days_back(from_day: date, sessions: int) -> date:
    d = from_day
    counted = 0
    while counted < sessions:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            counted += 1
    return d


def fetch_kite_1min(
    warmup_sessions: int = 80,
    *,
    to_day: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """Pull ~``warmup_sessions`` trading days of 1-min OHLCV from Kite.

    Returns a normalized DataFrame on success, or ``None`` on failure (caller may
    fall back to the local bar cache).
    """
    load_dotenv()
    api_key = os.environ.get("KITE_API_KEY", "").strip()
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        logger.warning(
            "Kite credentials missing (KITE_API_KEY / KITE_ACCESS_TOKEN). "
            "Run generate_token.py and update .env."
        )
        return None

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        kite.profile()
    except Exception as exc:
        logger.warning("Kite access token invalid or expired: %s", exc)
        return None

    to_d = to_day or date.today()
    from_d = _trading_days_back(to_d, warmup_sessions)
    try:
        token = find_bank_nifty_token(kite)
        records = fetch_minute_history_chunked(kite, token, from_d, to_d)
    except Exception as exc:
        logger.warning("Kite history fetch failed: %s", exc)
        return None

    if not records:
        logger.warning("Kite returned no 1-min data for %s .. %s", from_d, to_d)
        return None

    df = pd.DataFrame.from_records(records)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize(IST)
    else:
        df["date"] = df["date"].dt.tz_convert(IST)
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    out = _normalize_df(df)
    logger.info(
        "Kite warmup: %d bars from %s to %s (%d sessions)",
        len(out),
        out.index.min(),
        out.index.max(),
        len({ts.date() for ts in out.index if ts.weekday() < 5}),
    )
    return out
