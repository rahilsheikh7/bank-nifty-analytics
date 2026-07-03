"""Shared per-bar strategy step used by both backtest and live trading.

Extracting this logic guarantees live decisions cannot drift from backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pandas as pd

from strategy import ExitSignal, ExitType, Signal, SignalEngine, SignalType, StateManager

if TYPE_CHECKING:
    from backtest import BacktestConfig


@dataclass
class StepBarResult:
    entry_signal: Optional[Signal]
    exit_signal: Optional[ExitSignal]
    updates: Dict[str, Any]


def apply_state_updates(state_manager: StateManager, updates: Dict[str, Any]) -> None:
    """Apply pending-state mutations returned by evaluate_entry_conditions."""
    sm = state_manager
    if updates.get("set_pending_long_ema_wait"):
        sm.set_pending_long_ema_wait()
    if updates.get("clear_pending_long_ema_wait"):
        sm.clear_pending_long_ema_wait()
    if updates.get("set_pending_short_ema_wait"):
        sm.set_pending_short_ema_wait()
    if updates.get("clear_pending_short_ema_wait"):
        sm.clear_pending_short_ema_wait()
    if updates.get("set_pending_first_hour_long"):
        data = updates["set_pending_first_hour_long"]
        sm.set_pending_first_hour_long(data["trigger"], data.get("deferred"))
    if updates.get("clear_pending_first_hour_long"):
        sm.clear_pending_first_hour_long()
    if updates.get("set_pending_first_hour_short"):
        data = updates["set_pending_first_hour_short"]
        sm.set_pending_first_hour_short(data["trigger"], data.get("deferred"))
    if updates.get("clear_pending_first_hour_short"):
        sm.clear_pending_first_hour_short()
    if updates.get("set_deferred_ema_cross_long"):
        sm.set_deferred_ema_cross_long(updates["set_deferred_ema_cross_long"])
    if updates.get("set_deferred_ema_cross_short"):
        sm.set_deferred_ema_cross_short(updates["set_deferred_ema_cross_short"])
    if updates.get("set_adx_wait_long"):
        data = updates["set_adx_wait_long"]
        sm.set_adx_wait_long(data["bars"], data["trigger"])
    if updates.get("set_adx_wait_short"):
        data = updates["set_adx_wait_short"]
        sm.set_adx_wait_short(data["bars"], data["trigger"])
    if updates.get("clear_adx_wait_long"):
        sm.clear_adx_wait_long()
    if updates.get("clear_adx_wait_short"):
        sm.clear_adx_wait_short()
    if updates.get("decrement_adx_wait_long"):
        sm.decrement_adx_wait_long()
    if updates.get("decrement_adx_wait_short"):
        sm.decrement_adx_wait_short()
    if updates.get("set_volume_wait_long"):
        data = updates["set_volume_wait_long"]
        sm.set_volume_wait_long(data["bars_left"], data["trigger"], data["kind"])
    if updates.get("set_volume_wait_short"):
        data = updates["set_volume_wait_short"]
        sm.set_volume_wait_short(data["bars_left"], data["trigger"], data["kind"])
    if updates.get("clear_volume_wait_long"):
        sm.clear_volume_wait_long()
    if updates.get("clear_volume_wait_short"):
        sm.clear_volume_wait_short()
    if updates.get("decrement_volume_wait_long"):
        sm.decrement_volume_wait_long()
    if updates.get("decrement_volume_wait_short"):
        sm.decrement_volume_wait_short()


def _flip_columns(config: "BacktestConfig") -> Tuple[str, str]:
    if config.enable_long_entries and config.enable_short_entries:
        return "st_bull_flip", "st_bear_flip"
    if config.enable_long_entries:
        return "st_bull_flip_long", "st_bear_flip_long"
    return "st_bull_flip_short", "st_bear_flip_short"


def step_bar(
    *,
    bar: pd.Series,
    bar_index: int,
    df: pd.DataFrame,
    state_manager: StateManager,
    signal_engine: SignalEngine,
    config: "BacktestConfig",
    skip_entry: bool = False,
    skip_bar_sl_tp: bool = False,
) -> StepBarResult:
    """Run one strategy step for a completed primary bar (mirrors BacktestEngine.run)."""
    bull_col, bear_col = _flip_columns(config)
    state_manager.update_supertrend_state(
        st_bull_flip=bool(bar.get(bull_col, False)),
        st_bear_flip=bool(bar.get(bear_col, False)),
        current_direction=int(bar.get("direction", 0)),
    )

    exit_signal: Optional[ExitSignal] = None
    entry_signal: Optional[Signal] = None
    updates: Dict[str, Any] = {}

    state = state_manager.state
    if state.position_size != 0:
        exit_signal = signal_engine.check_exit_conditions(
            bar=bar,
            position_size=state.position_size,
            entry_price=state.entry_price,
            stop_loss=state.stop_loss,
            take_profit=state.take_profit,
            entry_time=state.entry_time,
        )
        if exit_signal and skip_bar_sl_tp and exit_signal.exit_type in {
            ExitType.STOP_LOSS,
            ExitType.TAKE_PROFIT,
        }:
            exit_signal = None

    state = state_manager.state
    if state.position_size == 0 and not skip_entry:
        volume_window = df.iloc[
            bar_index : min(bar_index + signal_engine.volume_candle_lookahead, len(df))
        ]
        entry_signal, updates = signal_engine.evaluate_entry_conditions(
            bar=bar,
            position_size=0,
            traded_in_bull_trend=state.traded_in_bull_trend if config.enable_long_entries else True,
            traded_in_bear_trend=state.traded_in_bear_trend if config.enable_short_entries else True,
            pending_long_ema_wait=state.pending_long_ema_wait if config.enable_long_entries else False,
            pending_short_ema_wait=state.pending_short_ema_wait if config.enable_short_entries else False,
            pending_first_hour_long=state.pending_first_hour_long if config.enable_long_entries else False,
            pending_first_hour_short=state.pending_first_hour_short if config.enable_short_entries else False,
            pending_first_hour_trigger_long=state.pending_first_hour_trigger_long
            if config.enable_long_entries
            else "",
            pending_first_hour_trigger_short=state.pending_first_hour_trigger_short
            if config.enable_short_entries
            else "",
            pending_first_hour_deferred_long=state.pending_first_hour_deferred_long
            if config.enable_long_entries
            else None,
            pending_first_hour_deferred_short=state.pending_first_hour_deferred_short
            if config.enable_short_entries
            else None,
            pending_adx_long=state.pending_adx_long if config.enable_long_entries else False,
            pending_adx_short=state.pending_adx_short if config.enable_short_entries else False,
            adx_wait_bars_left_long=state.adx_wait_bars_left_long if config.enable_long_entries else 0,
            adx_wait_bars_left_short=state.adx_wait_bars_left_short if config.enable_short_entries else 0,
            adx_wait_trigger_long=state.adx_wait_trigger_long if config.enable_long_entries else "",
            adx_wait_trigger_short=state.adx_wait_trigger_short if config.enable_short_entries else "",
            deferred_ema_cross_long=state.deferred_ema_cross_long if config.enable_long_entries else None,
            deferred_ema_cross_short=state.deferred_ema_cross_short if config.enable_short_entries else None,
            volume_window=volume_window,
        )
        apply_state_updates(state_manager, updates)

        if entry_signal and (
            (entry_signal.signal_type == SignalType.BUY and not config.enable_long_entries)
            or (entry_signal.signal_type == SignalType.SELL and not config.enable_short_entries)
        ):
            entry_signal = None
        if skip_entry:
            entry_signal = None

    return StepBarResult(entry_signal=entry_signal, exit_signal=exit_signal, updates=updates)
