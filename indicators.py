from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

_ST_BASE_COLS = (
    "supertrend",
    "direction",
    "st_bull",
    "st_bear",
    "st_bull_flip",
    "st_bear_flip",
)


def parse_timeframe_parts(timeframe: str) -> Tuple[int, str]:
    """Return (value, unit_lower) for strings like ``10m``, ``1H``, ``30min``."""
    raw = str(timeframe or "").strip()
    match = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", raw)
    if not match:
        raise ValueError(f"Invalid timeframe: {timeframe!r}")
    return int(match.group(1)), match.group(2).lower()


def timeframe_is_hour_based(timeframe: str) -> bool:
    """True for ``1H``, ``2h``, etc. (not minute-only bars like ``60m``)."""
    _, unit = parse_timeframe_parts(timeframe)
    return unit in {"h", "hr", "hrs", "hour", "hours"}


def timeframe_to_rule(timeframe: str) -> str:
    """Convert config values like 10m, 30min, 1H to pandas resample rules."""
    value, unit = parse_timeframe_parts(timeframe)
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return f"{value}min"
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return f"{value}h"
    raise ValueError(f"Unsupported timeframe unit: {timeframe!r}")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(timeframe_to_rule(timeframe))


def resample_ohlcv(
    df_1m: pd.DataFrame,
    timeframe: str,
    *,
    hourly_end_minute: Optional[int] = None,
) -> pd.DataFrame:
    """
    Resample 1-minute Bank Nifty data to a higher timeframe.

    Default ``offset=15min`` aligns minute-based bars (e.g. 5m, 10m) to the 09:15
    Indian cash session.

    For **hour-based** timeframes (``1H``, ``2h``, …), pass ``hourly_end_minute``
    (e.g. ``45``) to label buckets at …:45 with ``closed='right'`` / ``label='right'``
    so each hourly candle completes at 09:45, 10:45, etc.
    """
    rule = timeframe_to_rule(timeframe)
    if hourly_end_minute is not None and timeframe_is_hour_based(timeframe):
        off = f"{int(hourly_end_minute) % 60}min"
        out = (
            df_1m.resample(
                rule,
                origin="start_day",
                offset=off,
                closed="right",
                label="right",
            )
            .agg(OHLCV_AGG)
            .dropna(subset=["open", "high", "low", "close"])
        )
        return out

    out = (
        df_1m.resample(rule, origin="start_day", offset="15min")
        .agg(OHLCV_AGG)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=int(length), adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / int(length), adjust=False).mean()


def get_supertrend_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    atr = calculate_atr(high, low, close, atr_length)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + (float(multiplier) * atr)
    basic_lower = hl2 - (float(multiplier) * atr)

    n = len(close)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)

    close_arr = close.to_numpy()
    upper_arr = basic_upper.to_numpy()
    lower_arr = basic_lower.to_numpy()

    upper_band[0] = upper_arr[0]
    lower_band[0] = lower_arr[0]
    direction[0] = 1
    supertrend[0] = upper_band[0]

    for i in range(1, n):
        if upper_arr[i] < upper_band[i - 1] or close_arr[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_arr[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if lower_arr[i] > lower_band[i - 1] or close_arr[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_arr[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if direction[i - 1] == 1:
            if close_arr[i] > upper_band[i - 1]:
                direction[i] = -1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = 1
                supertrend[i] = upper_band[i]
        else:
            if close_arr[i] < lower_band[i - 1]:
                direction[i] = 1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = -1
                supertrend[i] = lower_band[i]

    direction_s = pd.Series(direction, index=close.index, name="direction")
    st_bull = direction_s < 0
    st_bear = direction_s > 0
    prev_direction = direction_s.shift(1)
    return pd.DataFrame(
        {
            "supertrend": pd.Series(supertrend, index=close.index),
            "direction": direction_s,
            "st_bull": st_bull,
            "st_bear": st_bear,
            "st_bull_flip": st_bull & (prev_direction > 0),
            "st_bear_flip": st_bear & (prev_direction < 0),
        }
    )


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_length: int = 14,
    adx_smoothing: int = 14,
) -> pd.DataFrame:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr_smooth = tr.ewm(alpha=1 / int(di_length), adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / int(di_length), adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / int(di_length), adjust=False).mean() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / int(adx_smoothing), adjust=False).mean()
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


def _rename_st_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    return df.rename(columns={col: f"{col}_{suffix}" for col in _ST_BASE_COLS})


def attach_long_short_indicators(
    df: pd.DataFrame,
    long_supertrend_entry: Dict[str, Any],
    short_supertrend_entry: Dict[str, Any],
    long_adx: Dict[str, Any],
    short_adx: Dict[str, Any],
    long_supertrend_exit: Dict[str, Any] | None = None,
    short_supertrend_exit: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    long_supertrend_exit = long_supertrend_exit or long_supertrend_entry
    short_supertrend_exit = short_supertrend_exit or short_supertrend_entry

    st_l = _rename_st_columns(
        get_supertrend_signals(
            df["high"],
            df["low"],
            df["close"],
            int(long_supertrend_entry.get("atr_length", 10)),
            float(long_supertrend_entry.get("multiplier", 3.0)),
        ),
        "long",
    )
    st_s = _rename_st_columns(
        get_supertrend_signals(
            df["high"],
            df["low"],
            df["close"],
            int(short_supertrend_entry.get("atr_length", 10)),
            float(short_supertrend_entry.get("multiplier", 3.0)),
        ),
        "short",
    )
    st_lx = _rename_st_columns(
        get_supertrend_signals(
            df["high"],
            df["low"],
            df["close"],
            int(long_supertrend_exit.get("atr_length", 10)),
            float(long_supertrend_exit.get("multiplier", 3.0)),
        ),
        "long_exit",
    )
    st_sx = _rename_st_columns(
        get_supertrend_signals(
            df["high"],
            df["low"],
            df["close"],
            int(short_supertrend_exit.get("atr_length", 10)),
            float(short_supertrend_exit.get("multiplier", 3.0)),
        ),
        "short_exit",
    )

    out = pd.concat([df, st_l, st_s, st_lx, st_sx], axis=1)

    di_l = int(long_adx.get("di_length", 14))
    sm_l = int(long_adx.get("adx_smoothing", 14))
    th_l = float(long_adx.get("threshold", 20.0))
    di_s = int(short_adx.get("di_length", 14))
    sm_s = int(short_adx.get("adx_smoothing", 14))
    th_s = float(short_adx.get("threshold", 20.0))

    if di_l == di_s and sm_l == sm_s:
        adx_df = calculate_adx(df["high"], df["low"], df["close"], di_l, sm_l)
        out = pd.concat([out, adx_df], axis=1)
        out["adx_above_threshold_long"] = out["adx"] >= th_l
        out["adx_above_threshold_short"] = out["adx"] >= th_s
    else:
        adx_l = calculate_adx(df["high"], df["low"], df["close"], di_l, sm_l)
        adx_s = calculate_adx(df["high"], df["low"], df["close"], di_s, sm_s)
        out["adx_long"] = adx_l["adx"]
        out["adx_short"] = adx_s["adx"]
        out["adx"] = out["adx_long"]
        out["plus_di"] = adx_l["plus_di"]
        out["minus_di"] = adx_l["minus_di"]
        out["adx_above_threshold_long"] = out["adx_long"] >= th_l
        out["adx_above_threshold_short"] = out["adx_short"] >= th_s

    out["supertrend"] = out["supertrend_long"]
    out["direction"] = out["direction_long"]
    out["st_bull_flip"] = out["st_bull_flip_long"] | out["st_bull_flip_short"]
    out["st_bear_flip"] = out["st_bear_flip_long"] | out["st_bear_flip_short"]
    out["adx_above_threshold"] = out["adx_above_threshold_long"]
    return out


def enrich_with_ema_timeframe(
    primary: pd.DataFrame,
    ema_bars: pd.DataFrame,
    ema_length: int,
    ema_timeframe: str,
    *,
    shift_cross_for_lookahead: bool = True,
) -> pd.DataFrame:
    """
    Map dynamic EMA timeframe values to primary bars with confirmed-cross columns.

    Column names keep the existing `*_1h` convention for compatibility with the
    strategy engine, even when the configured EMA timeframe is not 1H.

    When hourly buckets are **right-labeled** at a fixed minute (e.g. :45 via
    ``resample(..., closed='right', label='right')``), set ``shift_cross_for_lookahead``
    to False so EMA/close crosses are evaluated on the completed hour without an
    extra period shift.
    """
    ema_full = ema_bars.copy()
    ema_full["ema"] = calculate_ema(ema_full["close"], ema_length)

    out = primary.copy()
    ema_label = pd.Series(ema_full.index, index=ema_full.index).reindex(out.index, method="ffill")
    out["ema_1h"] = ema_full["ema"].reindex(out.index, method="ffill")
    out["close_1h"] = ema_full["close"].reindex(out.index, method="ffill")
    out["high_1h"] = ema_full["high"].reindex(out.index, method="ffill")
    out["low_1h"] = ema_full["low"].reindex(out.index, method="ffill")

    out["is_new_1h_candle"] = (ema_label != ema_label.shift(1)).fillna(True)
    out["ema_bull"] = out["close_1h"] > out["ema_1h"]
    out["ema_bear"] = out["close_1h"] < out["ema_1h"]

    cross_avail = ema_full.copy()
    if shift_cross_for_lookahead:
        cross_avail.index = cross_avail.index + timeframe_to_timedelta(ema_timeframe)
    out["close_1h_cross"] = cross_avail["close"].reindex(out.index, method="ffill")
    out["ema_1h_cross"] = cross_avail["ema"].reindex(out.index, method="ffill")

    prev_close = out["close_1h_cross"].shift(1)
    prev_ema = out["ema_1h_cross"].shift(1)
    out["ema_bull_cross"] = (out["close_1h_cross"] > out["ema_1h_cross"]) & (
        prev_close <= prev_ema
    )
    out["ema_bear_cross"] = (out["close_1h_cross"] < out["ema_1h_cross"]) & (
        prev_close >= prev_ema
    )
    return out
