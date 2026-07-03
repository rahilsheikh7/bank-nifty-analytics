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
        "ema_bull_cross": bool(bar.get("ema_bull_cross", False)),
        "ema_bear_cross": bool(bar.get("ema_bear_cross", False)),
    }


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
        return "entry skipped (flat_until_next_bar)"
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

    if result.entry_signal:
        sig = result.entry_signal
        side = "LONG" if sig.signal_type.name == "BUY" else "SHORT"
        logger.info(
            "Bar %s | SIGNAL %s %s @ %s trigger=%s | close=%s ADX=%s ST_dir=%s",
            primary_ts,
            side,
            sig.trigger,
            _fmt_num(sig.price),
            sig.trigger,
            flags["close"],
            flags["adx"],
            flags["direction"],
        )
        return

    if result.exit_signal:
        logger.info(
            "Bar %s | EXIT %s @ %s | close=%s ADX=%s",
            primary_ts,
            result.exit_signal.exit_type.value,
            _fmt_num(result.exit_signal.exit_price),
            flags["close"],
            flags["adx"],
        )
        return

    # Default: log decision context (compact + reason)
    logger.info(
        "Bar %s | close=%s ST_flip=%s/%s ST_bull=%s ADX=%s adx_ok_L=%s ema_ok_L=%s | pos=%s "
        "adx_wait_L=%s(%s) traded_bull=%s | %s",
        primary_ts,
        flags["close"],
        flags["st_bull_flip"],
        flags["st_bear_flip"],
        flags["st_bull"],
        flags["adx"],
        flags["adx_ok_long"],
        flags["ema_ok_long"],
        state_before["position_size"],
        state_before["pending_adx_long"],
        state_before["adx_wait_bars_left_long"],
        state_before["traded_in_bull_trend"],
        reason,
    )
