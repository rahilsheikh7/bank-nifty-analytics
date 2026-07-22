"""Log per-primary-bar strategy context and entry/exit decisions for live debugging."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from backtest import BacktestConfig
from strategy import SignalEngine, StateManager
from strategy_runtime import StepBarResult

logger = logging.getLogger("live.decision")


def snapshot_state(state_manager: StateManager) -> Dict[str, Any]:
    s = state_manager.state
    return {
        "position_size": s.position_size,
        "traded_in_bull_trend": s.traded_in_bull_trend,
        "traded_in_bear_trend": s.traded_in_bear_trend,
        "prev_st_direction": s.prev_st_direction,
        "pending_long_ema_wait": s.pending_long_ema_wait,
        "pending_short_ema_wait": s.pending_short_ema_wait,
        "pending_first_hour_long": s.pending_first_hour_long,
        "pending_first_hour_short": s.pending_first_hour_short,
        "pending_adx_long": s.pending_adx_long,
        "pending_adx_short": s.pending_adx_short,
        "adx_wait_bars_left_long": s.adx_wait_bars_left_long,
        "adx_wait_bars_left_short": s.adx_wait_bars_left_short,
        "adx_wait_trigger_long": s.adx_wait_trigger_long,
        "adx_wait_trigger_short": s.adx_wait_trigger_short,
    }


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cmp_relation(close_value: Any, ema_value: Any) -> str:
    if pd.isna(close_value) or pd.isna(ema_value):
        return "n/a"
    try:
        close_f = float(close_value)
        ema_f = float(ema_value)
    except (TypeError, ValueError):
        return "n/a"
    if close_f > ema_f:
        return "above"
    if close_f < ema_f:
        return "below"
    return "equal"


def _bar_flags(bar: pd.Series) -> Dict[str, Any]:
    close_1h = bar.get("close_1h_cross", bar.get("close_1h", np.nan))
    ema_1h = bar.get("ema_1h_cross", bar.get("ema_1h", np.nan))
    ema_ok_long = pd.notna(close_1h) and pd.notna(ema_1h) and float(close_1h) > float(ema_1h)
    ema_ok_short = pd.notna(close_1h) and pd.notna(ema_1h) and float(close_1h) < float(ema_1h)
    adx = bar.get("adx", bar.get("adx_long", np.nan))
    return {
        "close": _fmt_num(bar.get("close")),
        "direction": int(bar.get("direction", 0)),
        "st_bull_flip": bool(bar.get("st_bull_flip_long", bar.get("st_bull_flip", False))),
        "st_bear_flip": bool(bar.get("st_bear_flip_short", bar.get("st_bear_flip", False))),
        "st_bull": bool(bar.get("st_bull_long", False)),
        "st_bear": bool(bar.get("st_bear_short", False)),
        "adx": _fmt_num(adx, 1),
        "adx_ok_long": bool(bar.get("adx_above_threshold_long", bar.get("adx_above_threshold", False))),
        "adx_ok_short": bool(bar.get("adx_above_threshold_short", bar.get("adx_above_threshold", False))),
        "ema_ok_long": ema_ok_long,
        "ema_ok_short": ema_ok_short,
        "close_1h": _fmt_num(close_1h),
        "ema_1h": _fmt_num(ema_1h),
        "close_1h_raw": close_1h,
        "ema_1h_raw": ema_1h,
        "ema_relation": _cmp_relation(close_1h, ema_1h),
        "ema_bull_cross": bool(bar.get("ema_bull_cross", False)),
        "ema_bear_cross": bool(bar.get("ema_bear_cross", False)),
    }


def _st_summary(flags: Dict[str, Any]) -> str:
    if flags["st_bull"] and not flags["st_bear"]:
        state = "bull"
    elif flags["st_bear"] and not flags["st_bull"]:
        state = "bear"
    elif flags["direction"] < 0:
        state = "bull"
    elif flags["direction"] > 0:
        state = "bear"
    else:
        state = "neutral"

    flips: list[str] = []
    if flags["st_bull_flip"]:
        flips.append("bull_flip")
    if flags["st_bear_flip"]:
        flips.append("bear_flip")
    flip_text = ",".join(flips) if flips else "none"
    return f"ST={state} flip={flip_text}"


def _ema_summary(flags: Dict[str, Any]) -> str:
    symbol = {"above": ">", "below": "<", "equal": "="}.get(flags["ema_relation"], "?")
    return (
        f"EMA200={flags['ema_1h']} cmp_close={flags['close_1h']} "
        f"({flags['ema_relation']}, close {symbol} EMA)"
    )


def _pending_ema_summary(state_before: Dict[str, Any], updates: Dict[str, Any], flags: Dict[str, Any]) -> str:
    pending_long = bool(state_before["pending_long_ema_wait"])
    pending_short = bool(state_before["pending_short_ema_wait"])

    if updates.get("set_pending_long_ema_wait"):
        pending_long = True
    if updates.get("clear_pending_long_ema_wait"):
        pending_long = False
    if updates.get("set_pending_short_ema_wait"):
        pending_short = True
    if updates.get("clear_pending_short_ema_wait"):
        pending_short = False

    detail = ""
    if pending_long:
        detail = f" waiting_long(close must be above EMA; now {flags['close_1h']} is {flags['ema_relation']} {flags['ema_1h']})"
    elif pending_short:
        detail = f" waiting_short(close must be below EMA; now {flags['close_1h']} is {flags['ema_relation']} {flags['ema_1h']})"
    return f"pending_ema=L:{pending_long}/S:{pending_short}{detail}"


def _entry_condition(side: str, trigger: str, flags: Dict[str, Any]) -> str:
    if trigger == "st_flip":
        if side == "LONG":
            return f"ST bull flip + close above EMA200 ({flags['close_1h']} > {flags['ema_1h']})"
        return f"ST bear flip + close below EMA200 ({flags['close_1h']} < {flags['ema_1h']})"
    if trigger == "ema_cross":
        if side == "LONG":
            return f"pending long EMA wait confirmed ({flags['close_1h']} > {flags['ema_1h']})"
        return f"pending short EMA wait confirmed ({flags['close_1h']} < {flags['ema_1h']})"
    return trigger


def _interpret_updates(updates: Dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if updates.get("set_adx_wait_long"):
        data = updates["set_adx_wait_long"]
        notes.append(f"ADX wait started (long): {data['bars']} bars, trigger={data['trigger']}")
    if updates.get("set_adx_wait_short"):
        data = updates["set_adx_wait_short"]
        notes.append(f"ADX wait started (short): {data['bars']} bars, trigger={data['trigger']}")
    if updates.get("decrement_adx_wait_long"):
        notes.append("ADX still below threshold (long); decremented wait counter")
    if updates.get("decrement_adx_wait_short"):
        notes.append("ADX still below threshold (short); decremented wait counter")
    if updates.get("clear_adx_wait_long"):
        notes.append("ADX wait cleared (long)")
    if updates.get("clear_adx_wait_short"):
        notes.append("ADX wait cleared (short)")
    if updates.get("set_pending_long_ema_wait"):
        notes.append("EMA not bullish; pending long EMA wait")
    if updates.get("set_pending_short_ema_wait"):
        notes.append("EMA not bearish; pending short EMA wait")
    if updates.get("set_pending_first_hour_long"):
        data = updates["set_pending_first_hour_long"]
        notes.append(f"First-hour gate (long); trigger={data.get('trigger', '?')}")
    if updates.get("set_pending_first_hour_short"):
        data = updates["set_pending_first_hour_short"]
        notes.append(f"First-hour gate (short); trigger={data.get('trigger', '?')}")
    if updates.get("clear_pending_long_ema_wait"):
        notes.append("Cleared pending long EMA wait")
    if updates.get("clear_pending_short_ema_wait"):
        notes.append("Cleared pending short EMA wait")
    return notes


def _no_entry_reason(
    *,
    bar: pd.Series,
    flags: Dict[str, Any],
    state_before: Dict[str, Any],
    result: StepBarResult,
    skip_entry: bool,
    bt_config: BacktestConfig,
) -> str:
    if skip_entry:
        return "entry skipped (final_primary_bar)"
    if state_before["position_size"] != 0:
        if result.exit_signal:
            return f"in position; exit={result.exit_signal.exit_type.value}"
        return "in position; no exit on this bar"
    if result.entry_signal:
        return "entry signal generated"

    notes = _interpret_updates(result.updates)
    if notes:
        return "; ".join(notes)

    if flags["st_bull_flip"] and bt_config.enable_long_entries:
        if state_before["traded_in_bull_trend"]:
            return "ST bull flip but already traded_in_bull_trend"
        if not flags["ema_ok_long"]:
            return "ST bull flip but close <= EMA (not bullish side)"
        if not flags["adx_ok_long"]:
            return "ST bull flip but ADX below threshold"
    if flags["st_bear_flip"] and bt_config.enable_short_entries:
        if state_before["traded_in_bear_trend"]:
            return "ST bear flip but already traded_in_bear_trend"
        if not flags["ema_ok_short"]:
            return "ST bear flip but close >= EMA (not bearish side)"
        if not flags["adx_ok_short"]:
            return "ST bear flip but ADX below threshold"

    if state_before["pending_adx_long"] and flags["st_bull"]:
        if not flags["adx_ok_long"]:
            return (
                f"ADX wait active (long, {state_before['adx_wait_bars_left_long']} bars left, "
                f"trigger={state_before['adx_wait_trigger_long'] or 'st_flip'}); ADX still below threshold"
            )
        return "ADX wait active (long) but other gates blocked entry"

    if state_before["pending_adx_short"] and flags["st_bear"]:
        if not flags["adx_ok_short"]:
            return (
                f"ADX wait active (short, {state_before['adx_wait_bars_left_short']} bars left, "
                f"trigger={state_before['adx_wait_trigger_short'] or 'st_flip'}); ADX still below threshold"
            )
        return "ADX wait active (short) but other gates blocked entry"

    if not flags["st_bull_flip"] and not flags["st_bear_flip"]:
        return "no ST flip on this bar"
    return "no entry conditions met"


def log_primary_bar_decision(
    primary_ts: pd.Timestamp,
    bar: pd.Series,
    *,
    state_before: Dict[str, Any],
    result: StepBarResult,
    skip_entry: bool,
    bt_config: BacktestConfig,
    signal_engine: SignalEngine,
) -> None:
    flags = _bar_flags(bar)
    reason = _no_entry_reason(
        bar=bar,
        flags=flags,
        state_before=state_before,
        result=result,
        skip_entry=skip_entry,
        bt_config=bt_config,
    )

    if result.exit_signal and result.entry_signal:
        sig = result.entry_signal
        side = "LONG" if sig.signal_type.name == "BUY" else "SHORT"
        condition = _entry_condition(side, sig.trigger, flags)
        logger.info(
            "Bar %s | EXIT %s @ %s + SIGNAL %s @ %s condition=%s | close=%s | %s | %s | ADX=%s adx_ok=%s | %s",
            primary_ts,
            result.exit_signal.exit_type.value,
            _fmt_num(result.exit_signal.exit_price),
            side,
            _fmt_num(sig.price),
            condition,
            flags["close"],
            _st_summary(flags),
            _ema_summary(flags),
            flags["adx"],
            flags["adx_ok_long"] if side == "LONG" else flags["adx_ok_short"],
            _pending_ema_summary(state_before, result.updates, flags),
        )
        return

    if result.entry_signal:
        sig = result.entry_signal
        side = "LONG" if sig.signal_type.name == "BUY" else "SHORT"
        condition = _entry_condition(side, sig.trigger, flags)
        logger.info(
            "Bar %s | SIGNAL %s @ %s condition=%s | %s | %s | ADX=%s adx_ok=%s | %s",
            primary_ts,
            side,
            _fmt_num(sig.price),
            condition,
            _st_summary(flags),
            _ema_summary(flags),
            flags["adx"],
            flags["adx_ok_long"] if side == "LONG" else flags["adx_ok_short"],
            _pending_ema_summary(state_before, result.updates, flags),
        )
        return

    if result.exit_signal:
        notes = _interpret_updates(result.updates)
        suffix = f" | {'; '.join(notes)}" if notes else ""
        logger.info(
            "Bar %s | EXIT %s @ %s | close=%s | %s | %s | ADX=%s | %s%s",
            primary_ts,
            result.exit_signal.exit_type.value,
            _fmt_num(result.exit_signal.exit_price),
            flags["close"],
            _st_summary(flags),
            _ema_summary(flags),
            flags["adx"],
            _pending_ema_summary(state_before, result.updates, flags),
            suffix,
        )
        return

    # Default: log decision context (compact + reason)
    logger.info(
        "Bar %s | close=%s | %s | %s | ADX=%s adx_ok=L:%s/S:%s | pos=%s "
        "adx_wait=L:%s(%s)/S:%s(%s) traded=L:%s/S:%s | %s | %s",
        primary_ts,
        flags["close"],
        _st_summary(flags),
        _ema_summary(flags),
        flags["adx"],
        flags["adx_ok_long"],
        flags["adx_ok_short"],
        state_before["position_size"],
        state_before["pending_adx_long"],
        state_before["adx_wait_bars_left_long"],
        state_before["pending_adx_short"],
        state_before["adx_wait_bars_left_short"],
        state_before["traded_in_bull_trend"],
        state_before["traded_in_bear_trend"],
        _pending_ema_summary(state_before, result.updates, flags),
        reason,
    )
