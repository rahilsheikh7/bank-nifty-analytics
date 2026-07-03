"""Persist Neo-built 1-minute index bars to CSV for indicator warmup."""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

logger = logging.getLogger("live.bar_cache")

IST = "Asia/Kolkata"
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BARS_CSV = Path(__file__).resolve().parent / "cache" / "banknifty_1min_neo.csv"

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def resolve_bars_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"])
            out = out.set_index("date")
        else:
            raise ValueError("Bar data must have a DatetimeIndex or a 'date' column")
    if out.index.tz is None:
        out.index = out.index.tz_localize(IST)
    else:
        out.index = out.index.tz_convert(IST)
    out = out.sort_index()
    for col in _OHLCV_COLS:
        if col not in out.columns:
            out[col] = 0.0 if col == "volume" else float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")]
    return out[_OHLCV_COLS]


def count_trading_sessions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    days = {ts.date() for ts in df.index if ts.weekday() < 5}
    return len(days)


def merge_bars(
    kite_df: pd.DataFrame,
    overlay_df: pd.DataFrame,
    *,
    prefer_overlay_from: Union[pd.Timestamp, datetime, date],
) -> pd.DataFrame:
    """Merge Kite history with Neo/cache overlay.

    For timestamps on or after ``prefer_overlay_from`` (typically today 09:15 IST),
    overlay rows win on duplicate minutes. Earlier timestamps use Kite.
    """
    base = _normalize_df(kite_df) if not kite_df.empty else pd.DataFrame(columns=_OHLCV_COLS)
    overlay = _normalize_df(overlay_df) if not overlay_df.empty else pd.DataFrame(columns=_OHLCV_COLS)
    if base.empty:
        return overlay
    if overlay.empty:
        return base

    cutoff = pd.Timestamp(prefer_overlay_from)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(IST)
    else:
        cutoff = cutoff.tz_convert(IST)

    pre_overlay = overlay[overlay.index < cutoff]
    post_overlay = overlay[overlay.index >= cutoff]

    merged = pd.concat([base, pre_overlay])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    if not post_overlay.empty:
        merged = pd.concat([merged, post_overlay])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    return merged[_OHLCV_COLS]


def trim_to_sessions(df: pd.DataFrame, max_sessions: int) -> pd.DataFrame:
    """Keep only the most recent ``max_sessions`` weekday sessions."""
    if df.empty or max_sessions <= 0:
        return df
    df = _normalize_df(df)
    session_days: list[date] = []
    for ts in df.index:
        d = ts.date()
        if ts.weekday() < 5 and (not session_days or session_days[-1] != d):
            session_days.append(d)
    session_days = sorted(set(session_days))
    if len(session_days) <= max_sessions:
        return df
    keep = set(session_days[-max_sessions:])
    return df[df.index.map(lambda ts: ts.date() in keep)]


def load_bars_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.info("No bar cache at %s; starting empty (bars will accumulate from Neo ticks)", path)
        return pd.DataFrame(columns=_OHLCV_COLS)
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        logger.warning("Bar cache at %s is empty/corrupt; ignoring it for this startup", path)
        return pd.DataFrame(columns=_OHLCV_COLS)
    except ParserError as exc:
        logger.warning("Bar cache at %s could not be parsed (%s); ignoring it for this startup", path, exc)
        return pd.DataFrame(columns=_OHLCV_COLS)
    return _normalize_df(df)


def save_bars_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _normalize_df(df)
    out.index.name = "date"
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    out.reset_index().to_csv(tmp_path, index=False)
    tmp_path.replace(path)
    logger.debug("Saved %d 1-min bars to %s", len(out), path)


class BarCache:
    """Thread-safe rolling store of 1-min bars built from Neo index ticks."""

    def __init__(self, path: str | Path, *, max_sessions: int = 80):
        self.path = resolve_bars_path(path)
        self.max_sessions = max_sessions
        self._lock = threading.Lock()
        self.df = trim_to_sessions(load_bars_csv(self.path), max_sessions)

    def reload(self) -> pd.DataFrame:
        with self._lock:
            self.df = trim_to_sessions(load_bars_csv(self.path), self.max_sessions)
            return self.df.copy()

    def record_bar(self, bar: pd.Series, *, persist: bool = True) -> None:
        """Append or update one finalized 1-min bar."""
        with self._lock:
            self.df.loc[bar.name] = bar[_OHLCV_COLS]
            self.df = trim_to_sessions(self.df, self.max_sessions)
            if persist:
                save_bars_csv(self.df, self.path)

    def merge_frame(self, df_1m: pd.DataFrame, *, persist: bool = True) -> None:
        """Merge a full in-memory frame (e.g. on shutdown) and trim."""
        with self._lock:
            if df_1m.empty:
                return
            merged = pd.concat([self.df, _normalize_df(df_1m)])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.df = trim_to_sessions(merged, self.max_sessions)
            if persist:
                save_bars_csv(self.df, self.path)

    def flush(self) -> None:
        with self._lock:
            save_bars_csv(self.df, self.path)

    @property
    def sessions(self) -> int:
        return count_trading_sessions(self.df)

    @property
    def bar_count(self) -> int:
        return len(self.df)
