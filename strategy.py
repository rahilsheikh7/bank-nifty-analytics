from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# NSE cash session opens 09:15 IST; first 1H candle completes at 10:15.
SESSION_OPEN = time(9, 15)
SESSION_FIRST_HOUR_CLOSE = time(10, 15)


class SignalType(Enum):
    NONE = "none"
    BUY = "buy"
    SELL = "sell"


class ExitType(Enum):
    NONE = "none"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ST_FLIP = "st_flip"
    FORCED_CLOSE = "forced_close"


@dataclass
class Signal:
    signal_type: SignalType
    timestamp: pd.Timestamp
    price: float
    supertrend_value: float
    supertrend_direction: int
    ema_1h: float
    close_1h: float
    trigger: str
    volume_at_entry: Optional[float] = None
    volume_ma_at_entry: Optional[float] = None


@dataclass
class ExitSignal:
    exit_type: ExitType
    timestamp: pd.Timestamp
    exit_price: float
    entry_price: float
    pnl_points: float


@dataclass
class Trade:
    trade_id: int
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_type: Optional[str] = None
    pnl_points: Optional[float] = None
    pnl_value: Optional[float] = None
    contracts: int = 1
    entry_trigger: Optional[str] = None
    ema_at_entry: Optional[float] = None
    volume_at_entry: Optional[float] = None
    volume_ma_at_entry: Optional[float] = None
    max_positive_points: Optional[float] = None
    max_negative_points: Optional[float] = None
    max_positive_pct: Optional[float] = None
    max_negative_pct: Optional[float] = None


@dataclass
class StrategyState:
    position_size: int = 0
    entry_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    traded_in_bull_trend: bool = False
    traded_in_bear_trend: bool = False
    prev_st_direction: int = 0
    pending_long_ema_wait: bool = False
    pending_short_ema_wait: bool = False
    pending_first_hour_long: bool = False
    pending_first_hour_short: bool = False
    pending_first_hour_trigger_long: str = ""
    pending_first_hour_trigger_short: str = ""
    pending_first_hour_deferred_long: Optional[Dict[str, Any]] = None
    pending_first_hour_deferred_short: Optional[Dict[str, Any]] = None
    pending_adx_long: bool = False
    pending_adx_short: bool = False
    adx_wait_bars_left_long: int = 0
    adx_wait_bars_left_short: int = 0
    adx_wait_trigger_long: str = ""
    adx_wait_trigger_short: str = ""
    deferred_ema_cross_long: Optional[Dict[str, Any]] = None
    deferred_ema_cross_short: Optional[Dict[str, Any]] = None
    pending_volume_long: bool = False
    pending_volume_short: bool = False
    volume_wait_bars_left_long: int = 0
    volume_wait_bars_left_short: int = 0
    volume_wait_trigger_long: str = ""
    volume_wait_trigger_short: str = ""
    volume_wait_kind_long: str = ""
    volume_wait_kind_short: str = ""
    trade_count: int = 0


class StateManager:
    def __init__(self, point_value: float = 1.0, contracts_per_trade: int = 1):
        self.state = StrategyState()
        self.point_value = float(point_value)
        self.contracts = int(contracts_per_trade)
        self.trades: List[Trade] = []

    def update_supertrend_state(
        self, st_bull_flip: bool, st_bear_flip: bool, current_direction: int
    ) -> None:
        if st_bull_flip:
            self.state.traded_in_bull_trend = False
            self.state.traded_in_bear_trend = False
            self.state.pending_short_ema_wait = False
            self.clear_pending_first_hour_short()
            self.clear_adx_wait_long()
            self.clear_adx_wait_short()
            self.clear_volume_wait_long()
            self.clear_volume_wait_short()
        if st_bear_flip:
            self.state.traded_in_bear_trend = False
            self.state.traded_in_bull_trend = False
            self.state.pending_long_ema_wait = False
            self.clear_pending_first_hour_long()
            self.clear_adx_wait_long()
            self.clear_adx_wait_short()
            self.clear_volume_wait_long()
            self.clear_volume_wait_short()
        self.state.prev_st_direction = int(current_direction)

    def set_pending_long_ema_wait(self) -> None:
        self.state.pending_long_ema_wait = True

    def clear_pending_long_ema_wait(self) -> None:
        self.state.pending_long_ema_wait = False

    def set_pending_short_ema_wait(self) -> None:
        self.state.pending_short_ema_wait = True

    def clear_pending_short_ema_wait(self) -> None:
        self.state.pending_short_ema_wait = False

    def set_pending_first_hour_long(self, trigger: str, deferred: Optional[Dict[str, Any]] = None) -> None:
        self.state.pending_first_hour_long = True
        self.state.pending_first_hour_trigger_long = trigger
        self.state.pending_first_hour_deferred_long = dict(deferred) if deferred else None

    def clear_pending_first_hour_long(self) -> None:
        self.state.pending_first_hour_long = False
        self.state.pending_first_hour_trigger_long = ""
        self.state.pending_first_hour_deferred_long = None

    def set_pending_first_hour_short(self, trigger: str, deferred: Optional[Dict[str, Any]] = None) -> None:
        self.state.pending_first_hour_short = True
        self.state.pending_first_hour_trigger_short = trigger
        self.state.pending_first_hour_deferred_short = dict(deferred) if deferred else None

    def clear_pending_first_hour_short(self) -> None:
        self.state.pending_first_hour_short = False
        self.state.pending_first_hour_trigger_short = ""
        self.state.pending_first_hour_deferred_short = None

    def set_adx_wait_long(self, bars: int, trigger: str) -> None:
        self.state.pending_adx_long = True
        self.state.adx_wait_bars_left_long = int(bars)
        self.state.adx_wait_trigger_long = trigger

    def set_adx_wait_short(self, bars: int, trigger: str) -> None:
        self.state.pending_adx_short = True
        self.state.adx_wait_bars_left_short = int(bars)
        self.state.adx_wait_trigger_short = trigger

    def clear_adx_wait_long(self) -> None:
        self.state.pending_adx_long = False
        self.state.adx_wait_bars_left_long = 0
        self.state.adx_wait_trigger_long = ""
        self.state.deferred_ema_cross_long = None

    def clear_adx_wait_short(self) -> None:
        self.state.pending_adx_short = False
        self.state.adx_wait_bars_left_short = 0
        self.state.adx_wait_trigger_short = ""
        self.state.deferred_ema_cross_short = None

    def set_deferred_ema_cross_long(self, payload: Dict[str, Any]) -> None:
        self.state.deferred_ema_cross_long = dict(payload)

    def set_deferred_ema_cross_short(self, payload: Dict[str, Any]) -> None:
        self.state.deferred_ema_cross_short = dict(payload)

    def decrement_adx_wait_long(self) -> None:
        self.state.adx_wait_bars_left_long -= 1
        if self.state.adx_wait_bars_left_long <= 0:
            self.clear_adx_wait_long()

    def decrement_adx_wait_short(self) -> None:
        self.state.adx_wait_bars_left_short -= 1
        if self.state.adx_wait_bars_left_short <= 0:
            self.clear_adx_wait_short()

    def set_volume_wait_long(self, bars_left: int, trigger: str, kind: str) -> None:
        self.state.pending_volume_long = True
        self.state.volume_wait_bars_left_long = int(bars_left)
        self.state.volume_wait_trigger_long = trigger
        self.state.volume_wait_kind_long = kind

    def set_volume_wait_short(self, bars_left: int, trigger: str, kind: str) -> None:
        self.state.pending_volume_short = True
        self.state.volume_wait_bars_left_short = int(bars_left)
        self.state.volume_wait_trigger_short = trigger
        self.state.volume_wait_kind_short = kind

    def clear_volume_wait_long(self) -> None:
        self.state.pending_volume_long = False
        self.state.volume_wait_bars_left_long = 0
        self.state.volume_wait_trigger_long = ""
        self.state.volume_wait_kind_long = ""

    def clear_volume_wait_short(self) -> None:
        self.state.pending_volume_short = False
        self.state.volume_wait_bars_left_short = 0
        self.state.volume_wait_trigger_short = ""
        self.state.volume_wait_kind_short = ""

    def decrement_volume_wait_long(self) -> None:
        self.state.volume_wait_bars_left_long -= 1
        if self.state.volume_wait_bars_left_long <= 0:
            self.clear_volume_wait_long()

    def decrement_volume_wait_short(self) -> None:
        self.state.volume_wait_bars_left_short -= 1
        if self.state.volume_wait_bars_left_short <= 0:
            self.clear_volume_wait_short()

    def on_entry(self, signal: Signal, stop_loss: float, take_profit: float) -> None:
        is_long = signal.signal_type == SignalType.BUY
        self.state.position_size = 1 if is_long else -1
        self.state.entry_price = signal.price
        self.state.entry_time = signal.timestamp
        self.state.stop_loss = stop_loss
        self.state.take_profit = take_profit
        self.state.trade_count += 1

        if is_long:
            self.state.traded_in_bull_trend = True
            self.state.pending_long_ema_wait = False
            self.clear_pending_first_hour_long()
            self.clear_adx_wait_long()
            self.clear_volume_wait_long()
        else:
            self.state.traded_in_bear_trend = True
            self.state.pending_short_ema_wait = False
            self.clear_pending_first_hour_short()
            self.clear_adx_wait_short()
            self.clear_volume_wait_short()

        self.trades.append(
            Trade(
                trade_id=self.state.trade_count,
                direction="long" if is_long else "short",
                entry_time=signal.timestamp,
                entry_price=signal.price,
                contracts=self.contracts,
                entry_trigger=signal.trigger,
                ema_at_entry=signal.ema_1h,
                volume_at_entry=signal.volume_at_entry,
                volume_ma_at_entry=signal.volume_ma_at_entry,
            )
        )

    def on_exit(self, exit_signal: ExitSignal) -> Optional[Trade]:
        trade = self.trades[-1] if self.trades else None
        if trade is not None:
            pnl_points = (
                exit_signal.exit_price - trade.entry_price
                if trade.direction == "long"
                else trade.entry_price - exit_signal.exit_price
            )
            trade.exit_time = exit_signal.timestamp
            trade.exit_price = exit_signal.exit_price
            trade.exit_type = exit_signal.exit_type.value
            trade.pnl_points = pnl_points
            trade.pnl_value = pnl_points * self.point_value * trade.contracts

        self.state.position_size = 0
        self.state.entry_price = 0.0
        self.state.entry_time = None
        self.state.stop_loss = 0.0
        self.state.take_profit = 0.0
        return trade


class SignalEngine:
    def __init__(
        self,
        sl_pct_long: float = 0.35,
        tp_pct_long: float = 4.0,
        sl_pct_short: float = 0.35,
        tp_pct_short: float = 4.0,
        use_adx_long: bool = True,
        use_adx_short: bool = True,
        adx_wait_bars_long: int = 5,
        adx_wait_bars_short: int = 5,
        adx_threshold_long: float = 24.0,
        adx_threshold_short: float = 24.0,
        volume_check: bool = False,
        volume_candle_lookahead: int = 1,
        ema_timeframe: str = "1H",
        ema_on_primary: bool = False,
    ):
        self.ema_timeframe = str(ema_timeframe)
        # True when the EMA timeframe equals the primary timeframe: every primary bar
        # is itself an EMA candle, so entries fire at the signal bar's own close/label
        # (like st_flip) with no hourly "first-hour" gate and no lookahead shift.
        self.ema_on_primary = bool(ema_on_primary)
        # Width of one EMA candle, used to label hourly ema_cross entries at the candle
        # START (open) time rather than its right-edge close label (e.g. 15:15 -> 14:15).
        try:
            self.ema_timeframe_delta = pd.Timedelta(self.ema_timeframe)
        except (ValueError, TypeError):
            self.ema_timeframe_delta = pd.Timedelta(hours=1)
        self.sl_pct_long = float(sl_pct_long)
        self.tp_pct_long = float(tp_pct_long)
        self.sl_pct_short = float(sl_pct_short)
        self.tp_pct_short = float(tp_pct_short)
        self.use_adx_long = bool(use_adx_long)
        self.use_adx_short = bool(use_adx_short)
        self.adx_wait_bars_long = max(1, int(adx_wait_bars_long))
        self.adx_wait_bars_short = max(1, int(adx_wait_bars_short))
        self.adx_threshold_long = float(adx_threshold_long)
        self.adx_threshold_short = float(adx_threshold_short)
        self.volume_check = bool(volume_check)
        self.volume_candle_lookahead = max(1, int(volume_candle_lookahead))

    def calculate_exit_levels(self, entry_price: float, is_long: bool) -> Tuple[float, float]:
        sl_pct = self.sl_pct_long if is_long else self.sl_pct_short
        tp_pct = self.tp_pct_long if is_long else self.tp_pct_short
        stop_loss = entry_price * (1 - sl_pct / 100) if is_long else entry_price * (1 + sl_pct / 100)
        take_profit = entry_price * (1 + tp_pct / 100) if is_long else entry_price * (1 - tp_pct / 100)
        return stop_loss, take_profit

    def check_exit_conditions(
        self,
        bar: pd.Series,
        position_size: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_time: Optional[pd.Timestamp] = None,
    ) -> Optional[ExitSignal]:
        if position_size == 0:
            return None
        if entry_time is not None and bar.name < entry_time:
            return None

        is_long = position_size > 0
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # Overnight gap at the session open (09:15): if the market opens beyond our
        # SL/TP we cannot fill at those levels intrabar, so exit at the actual open
        # price and book the extra loss/profit (gaps are outside our control).
        if self._is_session_open_bar(bar):
            if is_long:
                if open_ <= stop_loss:
                    return ExitSignal(ExitType.STOP_LOSS, bar.name, open_, entry_price, open_ - entry_price)
                if open_ >= take_profit:
                    return ExitSignal(ExitType.TAKE_PROFIT, bar.name, open_, entry_price, open_ - entry_price)
            else:
                if open_ >= stop_loss:
                    return ExitSignal(ExitType.STOP_LOSS, bar.name, open_, entry_price, entry_price - open_)
                if open_ <= take_profit:
                    return ExitSignal(ExitType.TAKE_PROFIT, bar.name, open_, entry_price, entry_price - open_)

        if is_long:
            if low <= stop_loss:
                return ExitSignal(ExitType.STOP_LOSS, bar.name, stop_loss, entry_price, stop_loss - entry_price)
            if high >= take_profit:
                return ExitSignal(ExitType.TAKE_PROFIT, bar.name, take_profit, entry_price, take_profit - entry_price)
            if bool(bar.get("st_bear_flip_long_exit", bar.get("st_bear_flip_long", False))):
                return ExitSignal(ExitType.ST_FLIP, bar.name, close, entry_price, close - entry_price)
        else:
            if high >= stop_loss:
                return ExitSignal(ExitType.STOP_LOSS, bar.name, stop_loss, entry_price, entry_price - stop_loss)
            if low <= take_profit:
                return ExitSignal(ExitType.TAKE_PROFIT, bar.name, take_profit, entry_price, entry_price - take_profit)
            if bool(bar.get("st_bull_flip_short_exit", bar.get("st_bull_flip_short", False))):
                return ExitSignal(ExitType.ST_FLIP, bar.name, close, entry_price, entry_price - close)
        return None

    @staticmethod
    def _ema_period_close_bar(bar: pd.Series) -> bool:
        return bool(bar.get("is_ema_period_close", bar.get("is_new_1h_candle", False)))

    @staticmethod
    def _before_first_hour_close(timestamp: pd.Timestamp) -> bool:
        return pd.Timestamp(timestamp).time() < SESSION_FIRST_HOUR_CLOSE

    def _should_first_hour_gate(self, timestamp: pd.Timestamp) -> bool:
        # The "wait until the first 1H candle closes (10:15)" gate only applies when the
        # EMA runs on a higher (hourly) timeframe. With an EMA on the primary timeframe
        # every bar is a valid decision point, including the 09:15 open, so no gate.
        if self.ema_on_primary:
            return False
        return self._before_first_hour_close(timestamp)

    @staticmethod
    def _is_first_hour_close_bar(bar: pd.Series) -> bool:
        ts = pd.Timestamp(bar.name)
        return (ts.hour, ts.minute) == (SESSION_FIRST_HOUR_CLOSE.hour, SESSION_FIRST_HOUR_CLOSE.minute)

    @staticmethod
    def _is_session_open_bar(bar: pd.Series) -> bool:
        ts = pd.Timestamp(bar.name)
        return (ts.hour, ts.minute) == (SESSION_OPEN.hour, SESSION_OPEN.minute)

    @staticmethod
    def _ema_side_ok_long(bar: pd.Series, confirmed_valid: bool, close_confirmed: Any, ema_confirmed: Any) -> bool:
        return confirmed_valid and float(close_confirmed) > float(ema_confirmed)

    @staticmethod
    def _ema_side_ok_short(bar: pd.Series, confirmed_valid: bool, close_confirmed: Any, ema_confirmed: Any) -> bool:
        return confirmed_valid and float(close_confirmed) < float(ema_confirmed)

    @staticmethod
    def _hour_boundary_entry_price(bar: pd.Series) -> float:
        entry_px = bar.get("close_1m_hour_boundary", np.nan)
        if pd.isna(entry_px):
            entry_px = bar.get("close_1h", bar["close"])
        return float(entry_px)

    @staticmethod
    def _hour_boundary_entry_time(bar: pd.Series) -> pd.Timestamp:
        period_end = bar.get("ema_period_end", bar.name)
        return pd.Timestamp(period_end) if pd.notna(period_end) else bar.name

    def _defer_entry_until_first_hour(
        self,
        bar: pd.Series,
        signal: Optional[Signal],
        updates: Dict[str, Any],
        *,
        is_long: bool,
        trigger: str,
        deferred: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Signal], Dict[str, Any]]:
        if signal is None or not self._should_first_hour_gate(bar.name):
            return signal, updates
        key = "set_pending_first_hour_long" if is_long else "set_pending_first_hour_short"
        updates[key] = {"trigger": trigger, "deferred": deferred}
        return None, updates

    def _evaluate_first_hour_open_entry(
        self,
        bar: pd.Series,
        *,
        traded_in_bull_trend: bool,
        traded_in_bear_trend: bool,
        pending_first_hour_long: bool,
        pending_first_hour_short: bool,
        pending_first_hour_trigger_long: str,
        pending_first_hour_trigger_short: str,
        pending_first_hour_deferred_long: Optional[Dict[str, Any]],
        pending_first_hour_deferred_short: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Signal], Dict[str, Any]]:
        updates: Dict[str, Any] = {}
        if not self._is_first_hour_close_bar(bar):
            return None, updates

        close_confirmed = bar.get("close_1h_cross", bar.get("close_1h", np.nan))
        ema_confirmed = bar.get("ema_1h_cross", bar.get("ema_1h", np.nan))
        confirmed_valid = pd.notna(close_confirmed) and pd.notna(ema_confirmed)
        st_bull_long = bool(bar.get("st_bull_long", False))
        st_bear_short = bool(bar.get("st_bear_short", False))
        adx_ok_long = (not self.use_adx_long) or bool(
            bar.get("adx_above_threshold_long", bar.get("adx_above_threshold", False))
        )
        adx_ok_short = (not self.use_adx_short) or bool(
            bar.get("adx_above_threshold_short", bar.get("adx_above_threshold", False))
        )

        if pending_first_hour_long and st_bull_long and not traded_in_bull_trend:
            trigger = pending_first_hour_trigger_long or "st_flip"
            if self._ema_side_ok_long(bar, confirmed_valid, close_confirmed, ema_confirmed):
                updates["clear_pending_first_hour_long"] = True
                if adx_ok_long:
                    if trigger == "ema_cross":
                        if pending_first_hour_deferred_long:
                            deferred = pending_first_hour_deferred_long
                            return (
                                self._signal(
                                    bar,
                                    SignalType.BUY,
                                    "ema_cross",
                                    entry_price=float(deferred["price"]),
                                    timestamp=self._ema_candle_open_time(bar),
                                    ema_1h=float(deferred["ema_1h"]),
                                    close_1h=float(deferred["close_1h"]),
                                ),
                                updates,
                            )
                        return self._ema_cross_signal(bar, SignalType.BUY), updates
                    entry_time = self._hour_boundary_entry_time(bar)
                    entry_price = self._hour_boundary_entry_price(bar)
                    return (
                        self._signal(
                            bar,
                            SignalType.BUY,
                            trigger,
                            entry_price=entry_price,
                            timestamp=entry_time,
                        ),
                        updates,
                    )
                updates["set_adx_wait_long"] = {"bars": self.adx_wait_bars_long, "trigger": trigger}
            else:
                updates["clear_pending_first_hour_long"] = True
                updates["set_pending_long_ema_wait"] = True

        if pending_first_hour_short and st_bear_short and not traded_in_bear_trend:
            trigger = pending_first_hour_trigger_short or "st_flip"
            if self._ema_side_ok_short(bar, confirmed_valid, close_confirmed, ema_confirmed):
                updates["clear_pending_first_hour_short"] = True
                if adx_ok_short:
                    if trigger == "ema_cross":
                        if pending_first_hour_deferred_short:
                            deferred = pending_first_hour_deferred_short
                            return (
                                self._signal(
                                    bar,
                                    SignalType.SELL,
                                    "ema_cross",
                                    entry_price=float(deferred["price"]),
                                    timestamp=self._ema_candle_open_time(bar),
                                    ema_1h=float(deferred["ema_1h"]),
                                    close_1h=float(deferred["close_1h"]),
                                ),
                                updates,
                            )
                        return self._ema_cross_signal(bar, SignalType.SELL), updates
                    entry_time = self._hour_boundary_entry_time(bar)
                    entry_price = self._hour_boundary_entry_price(bar)
                    return (
                        self._signal(
                            bar,
                            SignalType.SELL,
                            trigger,
                            entry_price=entry_price,
                            timestamp=entry_time,
                        ),
                        updates,
                    )
                updates["set_adx_wait_short"] = {"bars": self.adx_wait_bars_short, "trigger": trigger}
            else:
                updates["clear_pending_first_hour_short"] = True
                updates["set_pending_short_ema_wait"] = True

        return None, updates

    def _ema_candle_open_time(self, bar: pd.Series) -> pd.Timestamp:
        """Start (open) timestamp of the EMA candle that confirms an ema_cross.

        ``ema_period_end`` is the candle's right-edge close label (e.g. 15:15 for
        the 14:15-15:14 hourly candle); subtract one EMA timeframe to get the candle
        start (14:15), matching the candle-start convention used by st_flip entries.
        """
        period_end = bar.get("ema_period_end", bar.name)
        ts = pd.Timestamp(period_end) if pd.notna(period_end) else pd.Timestamp(bar.name)
        return ts - self.ema_timeframe_delta

    def _ema_cross_entry_fields(self, bar: pd.Series) -> Tuple[pd.Timestamp, float, float, float]:
        """
        EMA-cross entry at the hourly candle that confirms the cross.

        Entry timestamp = that candle's START/open time (e.g. 14:15 for the
        14:15-15:14 candle) and entry price = the close of the *same* candle (the
        :14 one-minute close, e.g. 15:14). This matches the candle-start labelling
        that st_flip entries already use; the next hour is never used.
        """
        entry_time = self._ema_candle_open_time(bar)
        entry_px = bar.get("close_1m_hour_boundary", np.nan)
        if pd.isna(entry_px):
            entry_px = bar.get("close_1h", bar["close"])
        price = float(entry_px)
        ema_1h = float(bar.get("ema_1h", np.nan))
        close_1h = float(bar.get("close_1h", np.nan))
        return entry_time, price, ema_1h, close_1h

    def _signal(
        self,
        bar: pd.Series,
        signal_type: SignalType,
        trigger: str,
        *,
        entry_price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
        ema_1h: Optional[float] = None,
        close_1h: Optional[float] = None,
    ) -> Signal:
        is_long = signal_type == SignalType.BUY
        if entry_price is not None:
            price = float(entry_price)
        else:
            price = float(bar["close"])
        st_key = "supertrend_long" if is_long else "supertrend_short"
        dir_key = "direction_long" if is_long else "direction_short"
        return Signal(
            signal_type=signal_type,
            timestamp=timestamp if timestamp is not None else bar.name,
            price=price,
            supertrend_value=float(bar.get(st_key, np.nan)),
            supertrend_direction=int(bar.get(dir_key, 0)),
            ema_1h=float(ema_1h if ema_1h is not None else bar.get("ema_1h", np.nan)),
            close_1h=float(close_1h if close_1h is not None else bar.get("close_1h", np.nan)),
            trigger=trigger,
            volume_at_entry=float(bar["volume"]) if pd.notna(bar.get("volume", np.nan)) else None,
            volume_ma_at_entry=float(bar["volume_ma"]) if pd.notna(bar.get("volume_ma", np.nan)) else None,
        )

    def _ema_cross_signal(
        self,
        bar: pd.Series,
        signal_type: SignalType,
        *,
        deferred: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        if deferred:
            return self._signal(
                bar,
                signal_type,
                "ema_cross",
                entry_price=float(deferred["price"]),
                timestamp=deferred["timestamp"],
                ema_1h=float(deferred["ema_1h"]),
                close_1h=float(deferred["close_1h"]),
            )
        if self.ema_on_primary:
            # EMA timeframe == primary: the cross is confirmed on this bar, so enter at
            # this bar's own close and label (same convention as st_flip entries).
            return self._signal(bar, signal_type, "ema_cross")
        ts, price, ema_1h, close_1h = self._ema_cross_entry_fields(bar)
        return self._signal(
            bar,
            signal_type,
            "ema_cross",
            entry_price=price,
            timestamp=ts,
            ema_1h=ema_1h,
            close_1h=close_1h,
        )

    def _deferred_ema_cross_payload(self, bar: pd.Series) -> Dict[str, Any]:
        if self.ema_on_primary:
            return {
                "timestamp": bar.name,
                "price": float(bar["close"]),
                "ema_1h": float(bar.get("ema_1h", np.nan)),
                "close_1h": float(bar.get("close_1h", np.nan)),
            }
        ts, price, ema_1h, close_1h = self._ema_cross_entry_fields(bar)
        return {
            "timestamp": ts,
            "price": price,
            "ema_1h": ema_1h,
            "close_1h": close_1h,
        }

    @staticmethod
    def _row_volume_confirms(row: pd.Series) -> bool:
        volume = row.get("volume", np.nan)
        volume_ma = row.get("volume_ma", np.nan)
        return pd.notna(volume) and pd.notna(volume_ma) and float(volume_ma) > 0 and float(volume) > float(volume_ma)

    def _return_gated_entry(
        self,
        bar: pd.Series,
        signal: Signal,
        updates: Dict[str, Any],
        volume_window: Optional[pd.DataFrame],
        *,
        is_long: bool,
        trigger: str,
        deferred: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Signal], Dict[str, Any]]:
        signal, updates = self._defer_entry_until_first_hour(
            bar,
            signal,
            updates,
            is_long=is_long,
            trigger=trigger,
            deferred=deferred,
        )
        return self._finalize_volume(signal, updates, volume_window)

    def _finalize_volume(
        self,
        signal: Signal,
        updates: Dict[str, Any],
        volume_window: Optional[pd.DataFrame],
    ) -> Tuple[Optional[Signal], Dict[str, Any]]:
        if not self.volume_check:
            return signal, updates
        if volume_window is None or volume_window.empty:
            return None, {}
        for _, row in volume_window.iloc[: self.volume_candle_lookahead].iterrows():
            if self._row_volume_confirms(row):
                adjusted = self._signal(
                    row,
                    signal.signal_type,
                    signal.trigger,
                    entry_price=signal.price,
                    timestamp=signal.timestamp,
                    ema_1h=signal.ema_1h,
                    close_1h=signal.close_1h,
                )
                return adjusted, updates
        return None, {}

    def evaluate_entry_conditions(
        self,
        bar: pd.Series,
        position_size: int,
        traded_in_bull_trend: bool,
        traded_in_bear_trend: bool,
        pending_long_ema_wait: bool = False,
        pending_short_ema_wait: bool = False,
        pending_first_hour_long: bool = False,
        pending_first_hour_short: bool = False,
        pending_first_hour_trigger_long: str = "",
        pending_first_hour_trigger_short: str = "",
        pending_first_hour_deferred_long: Optional[Dict[str, Any]] = None,
        pending_first_hour_deferred_short: Optional[Dict[str, Any]] = None,
        pending_adx_long: bool = False,
        pending_adx_short: bool = False,
        adx_wait_bars_left_long: int = 0,
        adx_wait_bars_left_short: int = 0,
        adx_wait_trigger_long: str = "",
        adx_wait_trigger_short: str = "",
        deferred_ema_cross_long: Optional[Dict[str, Any]] = None,
        deferred_ema_cross_short: Optional[Dict[str, Any]] = None,
        volume_window: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[Signal], Dict[str, Any]]:
        updates: Dict[str, Any] = {}
        if position_size != 0:
            return None, updates

        ema_now = bar.get("ema_1h", np.nan)
        if pd.isna(ema_now) or float(ema_now) <= 0:
            return None, updates

        st_bull_flip_long = bool(bar.get("st_bull_flip_long", bar.get("st_bull_flip", False)))
        st_bear_flip_short = bool(bar.get("st_bear_flip_short", bar.get("st_bear_flip", False)))
        st_bull_long = bool(bar.get("st_bull_long", False))
        st_bear_short = bool(bar.get("st_bear_short", False))

        close_confirmed = bar.get("close_1h_cross", bar.get("close_1h", np.nan))
        ema_confirmed = bar.get("ema_1h_cross", bar.get("ema_1h", np.nan))
        confirmed_valid = pd.notna(close_confirmed) and pd.notna(ema_confirmed)

        is_ema_period_close = self._ema_period_close_bar(bar)
        ema_bull_cross = bool(bar.get("ema_bull_cross", False))
        ema_bear_cross = bool(bar.get("ema_bear_cross", False))

        adx_above_long = bool(bar.get("adx_above_threshold_long", bar.get("adx_above_threshold", False)))
        adx_above_short = bool(bar.get("adx_above_threshold_short", bar.get("adx_above_threshold", False)))
        adx_ok_long = (not self.use_adx_long) or adx_above_long
        adx_ok_short = (not self.use_adx_short) or adx_above_short

        first_hour_signal, first_hour_updates = self._evaluate_first_hour_open_entry(
            bar,
            traded_in_bull_trend=traded_in_bull_trend,
            traded_in_bear_trend=traded_in_bear_trend,
            pending_first_hour_long=pending_first_hour_long,
            pending_first_hour_short=pending_first_hour_short,
            pending_first_hour_trigger_long=pending_first_hour_trigger_long,
            pending_first_hour_trigger_short=pending_first_hour_trigger_short,
            pending_first_hour_deferred_long=pending_first_hour_deferred_long,
            pending_first_hour_deferred_short=pending_first_hour_deferred_short,
        )
        updates.update(first_hour_updates)
        if first_hour_signal is not None:
            return self._finalize_volume(first_hour_signal, updates, volume_window)

        if st_bull_flip_long and not traded_in_bull_trend:
            if self._ema_side_ok_long(bar, confirmed_valid, close_confirmed, ema_confirmed):
                # EMA is already on the bullish side when Supertrend flips -> a clear
                # direct entry at the close of the flip bar. This holds even during the
                # opening hour (e.g. a 09:25 flip): no need to wait for the 10:15 close.
                if adx_ok_long:
                    return self._finalize_volume(
                        self._signal(bar, SignalType.BUY, "st_flip"), updates, volume_window
                    )
                updates["set_adx_wait_long"] = {"bars": self.adx_wait_bars_long, "trigger": "st_flip"}
                return None, updates
            updates["set_pending_long_ema_wait"] = True
            return None, updates

        if st_bear_flip_short and not traded_in_bear_trend:
            if self._ema_side_ok_short(bar, confirmed_valid, close_confirmed, ema_confirmed):
                # EMA already on the bearish side when Supertrend flips -> direct entry
                # at the close of the flip bar, including during the opening hour.
                if adx_ok_short:
                    return self._finalize_volume(
                        self._signal(bar, SignalType.SELL, "st_flip"), updates, volume_window
                    )
                updates["set_adx_wait_short"] = {"bars": self.adx_wait_bars_short, "trigger": "st_flip"}
                return None, updates
            updates["set_pending_short_ema_wait"] = True
            return None, updates

        if pending_adx_long and st_bull_long and not traded_in_bull_trend:
            trigger = adx_wait_trigger_long or "st_flip"
            if trigger == "ema_cross" and deferred_ema_cross_long:
                if adx_ok_long:
                    updates["clear_adx_wait_long"] = True
                    updates["clear_pending_long_ema_wait"] = True
                    return self._return_gated_entry(
                        bar,
                        self._ema_cross_signal(
                            bar, SignalType.BUY, deferred=deferred_ema_cross_long
                        ),
                        updates,
                        volume_window,
                        is_long=True,
                        trigger="ema_cross",
                        deferred=deferred_ema_cross_long,
                    )
                if adx_wait_bars_left_long <= 1:
                    updates["clear_adx_wait_long"] = True
                else:
                    updates["decrement_adx_wait_long"] = True
                return None, updates
            if adx_ok_long:
                updates["clear_adx_wait_long"] = True
                if trigger == "ema_cross":
                    return self._return_gated_entry(
                        bar,
                        self._ema_cross_signal(
                            bar,
                            SignalType.BUY,
                            deferred=deferred_ema_cross_long,
                        ),
                        updates,
                        volume_window,
                        is_long=True,
                        trigger="ema_cross",
                        deferred=deferred_ema_cross_long,
                    )
                return self._return_gated_entry(
                    bar,
                    self._signal(bar, SignalType.BUY, trigger),
                    updates,
                    volume_window,
                    is_long=True,
                    trigger=trigger,
                )
            if adx_wait_bars_left_long <= 1:
                updates["clear_adx_wait_long"] = True
            else:
                updates["decrement_adx_wait_long"] = True
            return None, updates

        if pending_adx_short and st_bear_short and not traded_in_bear_trend:
            trigger = adx_wait_trigger_short or "st_flip"
            if trigger == "ema_cross" and deferred_ema_cross_short:
                if adx_ok_short:
                    updates["clear_adx_wait_short"] = True
                    updates["clear_pending_short_ema_wait"] = True
                    return self._return_gated_entry(
                        bar,
                        self._ema_cross_signal(
                            bar, SignalType.SELL, deferred=deferred_ema_cross_short
                        ),
                        updates,
                        volume_window,
                        is_long=False,
                        trigger="ema_cross",
                        deferred=deferred_ema_cross_short,
                    )
                if adx_wait_bars_left_short <= 1:
                    updates["clear_adx_wait_short"] = True
                else:
                    updates["decrement_adx_wait_short"] = True
                return None, updates
            if adx_ok_short:
                updates["clear_adx_wait_short"] = True
                if trigger == "ema_cross":
                    return self._return_gated_entry(
                        bar,
                        self._ema_cross_signal(
                            bar,
                            SignalType.SELL,
                            deferred=deferred_ema_cross_short,
                        ),
                        updates,
                        volume_window,
                        is_long=False,
                        trigger="ema_cross",
                        deferred=deferred_ema_cross_short,
                    )
                return self._return_gated_entry(
                    bar,
                    self._signal(bar, SignalType.SELL, trigger),
                    updates,
                    volume_window,
                    is_long=False,
                    trigger=trigger,
                )
            if adx_wait_bars_left_short <= 1:
                updates["clear_adx_wait_short"] = True
            else:
                updates["decrement_adx_wait_short"] = True
            return None, updates

        ema_long_confirmed = ema_bull_cross or self._ema_side_ok_long(
            bar, confirmed_valid, close_confirmed, ema_confirmed
        )
        ema_short_confirmed = ema_bear_cross or self._ema_side_ok_short(
            bar, confirmed_valid, close_confirmed, ema_confirmed
        )

        if (
            pending_long_ema_wait
            and st_bull_long
            and not traded_in_bull_trend
            and is_ema_period_close
            and ema_long_confirmed
        ):
            if self._should_first_hour_gate(bar.name):
                updates["clear_pending_long_ema_wait"] = True
                updates["set_pending_first_hour_long"] = {
                    "trigger": "ema_cross",
                    "deferred": self._deferred_ema_cross_payload(bar),
                }
                return None, updates
            if adx_ok_long:
                updates["clear_pending_long_ema_wait"] = True
                return self._return_gated_entry(
                    bar,
                    self._ema_cross_signal(bar, SignalType.BUY),
                    updates,
                    volume_window,
                    is_long=True,
                    trigger="ema_cross",
                    deferred=self._deferred_ema_cross_payload(bar),
                )
            updates["clear_pending_long_ema_wait"] = True
            updates["set_deferred_ema_cross_long"] = self._deferred_ema_cross_payload(bar)
            updates["set_adx_wait_long"] = {"bars": self.adx_wait_bars_long, "trigger": "ema_cross"}
            return None, updates

        if (
            pending_short_ema_wait
            and st_bear_short
            and not traded_in_bear_trend
            and is_ema_period_close
            and ema_short_confirmed
        ):
            if self._should_first_hour_gate(bar.name):
                updates["clear_pending_short_ema_wait"] = True
                updates["set_pending_first_hour_short"] = {
                    "trigger": "ema_cross",
                    "deferred": self._deferred_ema_cross_payload(bar),
                }
                return None, updates
            if adx_ok_short:
                updates["clear_pending_short_ema_wait"] = True
                return self._return_gated_entry(
                    bar,
                    self._ema_cross_signal(bar, SignalType.SELL),
                    updates,
                    volume_window,
                    is_long=False,
                    trigger="ema_cross",
                    deferred=self._deferred_ema_cross_payload(bar),
                )
            updates["clear_pending_short_ema_wait"] = True
            updates["set_deferred_ema_cross_short"] = self._deferred_ema_cross_payload(bar)
            updates["set_adx_wait_short"] = {"bars": self.adx_wait_bars_short, "trigger": "ema_cross"}
            return None, updates

        return None, updates
