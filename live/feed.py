"""Real-time tick aggregation: Neo index ticks -> 1-min OHLC -> primary bar events."""
from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd

from indicators import resample_ohlcv, timeframe_is_hour_based, timeframe_to_timedelta
from live.bar_cache import BarCache
from live.config import LiveConfig

logger = logging.getLogger("live.feed")
IST = "Asia/Kolkata"


def _to_ist(ts: datetime) -> pd.Timestamp:
    if ts.tzinfo is None:
        return pd.Timestamp(ts, tz=IST)
    return pd.Timestamp(ts).tz_convert(IST)


def _minute_floor(ts: pd.Timestamp) -> pd.Timestamp:
    ts = _to_ist(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
    return ts.floor("min")


def _in_session(ts: pd.Timestamp, start: time, end: time) -> bool:
    t = ts.time()
    return start <= t <= end


class MinuteAggregator:
    """Bucket index ticks into 1-minute OHLC bars (IST, session-gated)."""

    def __init__(
        self,
        live_cfg: LiveConfig,
        on_minute_bar: Callable[[pd.Series], None],
        initial_df: Optional[pd.DataFrame] = None,
        bar_cache: Optional[BarCache] = None,
    ):
        self.cfg = live_cfg
        self.on_minute_bar = on_minute_bar
        self.bar_cache = bar_cache
        self.df_1m = initial_df.copy() if initial_df is not None else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        self._cur_minute: Optional[pd.Timestamp] = None
        self._o = self._h = self._l = self._c = None
        self._last_close: Optional[float] = None
        self.tick_count = 0

    @property
    def frame(self) -> pd.DataFrame:
        return self.df_1m

    def on_tick(self, price: float, ts: Optional[datetime] = None) -> None:
        self.tick_count += 1
        now = _to_ist(ts or datetime.now(tz=pd.Timestamp.now(tz=IST).tz))
        if not _in_session(now, self.cfg.session_start, self.cfg.session_end):
            return

        minute = _minute_floor(now)
        if self._cur_minute is None:
            self._start_minute(minute, price)
            return

        if minute > self._cur_minute:
            self._finalize_minute()
            self._start_minute(minute, price)
            return

        self._h = max(self._h, price)
        self._l = min(self._l, price)
        self._c = price

    def _start_minute(self, minute: pd.Timestamp, price: float) -> None:
        self._cur_minute = minute
        self._o = self._h = self._l = self._c = price

    def _finalize_minute(self) -> None:
        if self._cur_minute is None or self._c is None:
            return
        minute = self._cur_minute
        bar = pd.Series(
            {
                "open": self._o,
                "high": self._h,
                "low": self._l,
                "close": self._c,
                "volume": 0.0,
            },
            name=minute,
        )
        self.df_1m.loc[minute] = bar
        self.df_1m = self.df_1m.sort_index()
        self._last_close = float(self._c)
        logger.info(
            "1-min bar %s O=%.2f H=%.2f L=%.2f C=%.2f",
            minute.strftime("%H:%M"),
            self._o,
            self._h,
            self._l,
            self._c,
        )
        if self.bar_cache is not None:
            self.bar_cache.record_bar(bar)
        self.on_minute_bar(bar)
        self._cur_minute = None

    def flush(self) -> None:
        """Finalize the in-progress minute (e.g. at session end)."""
        if self._cur_minute is not None:
            self._finalize_minute()


class PrimaryBarClock:
    """Detect completed primary-timeframe bars by reusing backtest resample logic."""

    def __init__(
        self,
        primary_timeframe: str,
        hourly_end_minute: Optional[int],
        on_primary_bar: Callable[[pd.Timestamp, pd.DataFrame], None],
    ):
        self.primary_timeframe = primary_timeframe
        self.hourly_end_minute = hourly_end_minute
        self.on_primary_bar = on_primary_bar
        self._last_emitted: Optional[pd.Timestamp] = None
        self._bar_delta = timeframe_to_timedelta(primary_timeframe)
        self._right_labeled = hourly_end_minute is not None and timeframe_is_hour_based(primary_timeframe)

    def _required_last_minute(self, primary_ts: pd.Timestamp) -> pd.Timestamp:
        """Return the final 1-minute candle needed before this primary bar is complete."""
        if self._right_labeled:
            # Hour bars are labeled at the right edge, e.g. 10:15 spans 09:15-10:14.
            return primary_ts - pd.Timedelta(minutes=1)
        # Minute-based bars are left-labeled, e.g. 13:45 spans 13:45-13:49.
        return primary_ts + self._bar_delta - pd.Timedelta(minutes=1)

    def seed(self, df_1m: pd.DataFrame) -> None:
        """Mark existing completed history as already emitted before live ticks start."""
        if df_1m.empty:
            return
        last_1m = df_1m.index.max()
        primary = resample_ohlcv(
            df_1m,
            self.primary_timeframe,
            hourly_end_minute=self.hourly_end_minute,
        )
        completed = [ts for ts in primary.index if last_1m >= self._required_last_minute(ts)]
        if completed:
            self._last_emitted = completed[-1]
            logger.info("Primary bar clock seeded at %s", self._last_emitted)

    def check(self, df_1m: pd.DataFrame) -> None:
        if df_1m.empty:
            return
        last_1m = df_1m.index.max()
        primary = resample_ohlcv(
            df_1m,
            self.primary_timeframe,
            hourly_end_minute=self.hourly_end_minute,
        )
        if primary.empty:
            return
        for primary_ts in primary.index:
            if self._last_emitted is not None and primary_ts <= self._last_emitted:
                continue
            required_last = self._required_last_minute(primary_ts)
            if last_1m < required_last:
                logger.debug(
                    "Primary bar %s not complete yet (need 1-min %s, have %s)",
                    primary_ts,
                    required_last,
                    last_1m,
                )
                continue
            self._last_emitted = primary_ts
            logger.info("Primary bar completed: %s", primary_ts)
            self.on_primary_bar(primary_ts, df_1m)


def _parse_tick_item(item: dict) -> Optional[Tuple[float, datetime]]:
    """Extract one index tick from a Neo HSM feed item (short keys: iv, tk, ...)."""
    price = item.get("iv")
    if price is None:
        price = item.get("ltp") or item.get("last_traded_price")
    if price is None:
        return None
    try:
        px = float(price)
    except (TypeError, ValueError):
        return None

    ts: Optional[datetime] = None
    raw_ts = item.get("tvalue") or item.get("ltt") or item.get("last_traded_time")
    if raw_ts is not None:
        try:
            ts = pd.Timestamp(raw_ts, tz=IST).to_pydatetime()
        except (TypeError, ValueError):
            ts = None
    if ts is None:
        ts = datetime.now(tz=pd.Timestamp.now(tz=IST).tz)
    return px, ts


def extract_index_ticks(message: Any) -> List[Tuple[float, datetime]]:
    """Extract index ticks from Neo websocket payloads (flat dict or stock_feed wrapper)."""
    ticks: List[Tuple[float, datetime]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            mtype = node.get("type")
            if mtype == "stock_feed":
                data = node.get("data")
                if isinstance(data, list):
                    for item in data:
                        _walk(item)
                elif isinstance(data, dict):
                    _walk(data)
                return
            if mtype in ("order_feed", "order", "quotes", "cn"):
                return
            tick = _parse_tick_item(node)
            if tick:
                ticks.append(tick)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(message)
    return ticks


def parse_index_tick(message: Any) -> Optional[Tuple[float, datetime]]:
    """Extract the first index tick from a Neo WebSocket message."""
    ticks = extract_index_ticks(message)
    return ticks[0] if ticks else None
