"""Load warmup 1-minute history: daily Kite fetch + Neo live bar cache merge."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import pandas as pd

from indicators import timeframe_to_rule
from live.bar_cache import (
    apply_bar_retention,
    count_trading_sessions,
    load_bars_csv,
    merge_bars,
    resolve_bars_path,
    save_bars_csv,
)
from live.kite_warmup import fetch_kite_1min

logger = logging.getLogger("live.warmup")

IST = "Asia/Kolkata"


@dataclass
class WarmupStatus:
    bar_count: int
    session_count: int
    target_sessions: int
    ema_length: int
    ema_timeframe: str
    primary_timeframe: str
    ready: bool
    source: str
    message: str


def _bars_per_session(primary_timeframe: str) -> int:
    minutes = 375
    primary_min = pd.Timedelta(timeframe_to_rule(primary_timeframe)).total_seconds() / 60
    return max(1, int(minutes // primary_min))


def _session_open_today(session_start: time) -> pd.Timestamp:
    today = date.today()
    return pd.Timestamp(datetime.combine(today, session_start), tz=IST)


def assess_warmup(
    df: pd.DataFrame,
    *,
    target_sessions: int,
    max_bars: int = 0,
    ema_length: int,
    ema_timeframe: str,
    primary_timeframe: str,
    source: str = "unknown",
) -> WarmupStatus:
    sessions = count_trading_sessions(df)
    bar_count = len(df)

    ema_rule = timeframe_to_rule(ema_timeframe)
    primary_rule = timeframe_to_rule(primary_timeframe)
    ema_minutes = pd.Timedelta(ema_rule).total_seconds() / 60
    primary_minutes = pd.Timedelta(primary_rule).total_seconds() / 60
    ema_bars_needed = int(ema_length * (ema_minutes / primary_minutes)) + 5

    min_sessions = max(1, (ema_bars_needed // _bars_per_session(primary_timeframe)) + 1)
    ready = sessions >= min_sessions and bar_count >= ema_bars_needed

    if df.empty:
        msg = (
            "No warmup bars available. For kite_daily: run generate_token.py and "
            "ensure Kite credentials are valid, or let Neo accumulate bars in bars_csv."
        )
    elif not ready:
        msg = (
            f"Warmup partial ({source}): {sessions} session(s), {bar_count} bars "
            f"(need ~{min_sessions} sessions / {ema_bars_needed} bars for EMA{ema_length})."
        )
    else:
        msg = (
            f"Warmup OK ({source}): {sessions} sessions, {bar_count} 1-min bars "
            f"(retention: {target_sessions} sessions"
            f"{f', max {max_bars} bars' if max_bars > 0 else ''})."
        )

    return WarmupStatus(
        bar_count=bar_count,
        session_count=sessions,
        target_sessions=target_sessions,
        ema_length=ema_length,
        ema_timeframe=ema_timeframe,
        primary_timeframe=primary_timeframe,
        ready=ready,
        source=source,
        message=msg,
    )


def _bars_per_trading_session() -> int:
    return 375


def effective_warmup_sessions(
    warmup_sessions: int,
    *,
    max_bars: int = 0,
    bars_per_session: int = 375,
) -> int:
    """Limit Kite fetch to roughly what retention will keep."""
    if max_bars <= 0:
        return warmup_sessions
    needed = max(5, (max_bars // bars_per_session) + 2)
    return min(warmup_sessions, needed)


def build_warm_1min(
    warmup_sessions: int = 80,
    *,
    warmup_max_bars: int = 0,
    bars_csv: str | Path,
    ema_length: int = 200,
    ema_timeframe: str = "5m",
    primary_timeframe: str = "5m",
    warmup_source: str = "kite_daily",
    kite_fallback_on_error: bool = True,
    session_start: time = time(9, 15),
) -> tuple[pd.DataFrame, WarmupStatus]:
    """Build the rolling 1-min warmup frame for indicator seeding."""
    path = resolve_bars_path(bars_csv)
    overlay = load_bars_csv(path)
    source = "neo_cache"
    fetch_sessions = effective_warmup_sessions(
        warmup_sessions,
        max_bars=warmup_max_bars,
        bars_per_session=_bars_per_trading_session(),
    )

    if warmup_source.strip().lower() == "kite_daily":
        if fetch_sessions < warmup_sessions:
            logger.info(
                "Fetching %d sessions from Kite (capped by warmup_max_bars=%d)",
                fetch_sessions,
                warmup_max_bars,
            )
        else:
            logger.info("Fetching %d sessions of 1-min history from Kite...", fetch_sessions)
        kite_df = fetch_kite_1min(fetch_sessions)
        if kite_df is not None:
            cutoff = _session_open_today(session_start)
            merged = merge_bars(kite_df, overlay, prefer_overlay_from=cutoff)
            source = "kite_daily+neo_overlay" if not overlay.empty else "kite_daily"
        elif kite_fallback_on_error:
            logger.warning("Kite fetch failed; falling back to bars_csv at %s", path)
            merged = overlay
            source = "neo_cache_fallback"
        else:
            raise RuntimeError(
                "Kite warmup fetch failed and kite_fallback_on_error is false. "
                "Run generate_token.py and update KITE_ACCESS_TOKEN."
            )
    else:
        merged = overlay

    merged = apply_bar_retention(
        merged,
        max_sessions=warmup_sessions,
        max_bars=warmup_max_bars,
    )
    save_bars_csv(merged, path)

    status = assess_warmup(
        merged,
        target_sessions=warmup_sessions,
        max_bars=warmup_max_bars,
        ema_length=ema_length,
        ema_timeframe=ema_timeframe,
        primary_timeframe=primary_timeframe,
        source=source,
    )
    if status.ready:
        logger.info(status.message)
    else:
        logger.warning(status.message)
    return merged, status
