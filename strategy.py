from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
    pending_adx_long: bool = False
    pending_adx_short: bool = False
    adx_wait_bars_left_long: int = 0
    adx_wait_bars_left_short: int = 0
    adx_wait_trigger_long: str = ""
    adx_wait_trigger_short: str = ""
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
            self.clear_adx_wait_long()
            self.clear_adx_wait_short()
            self.clear_volume_wait_long()
            self.clear_volume_wait_short()
        if st_bear_flip:
            self.state.traded_in_bear_trend = False
            self.state.traded_in_bull_trend = False
            self.state.pending_long_ema_wait = False
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

    def clear_adx_wait_short(self) -> None:
        self.state.pending_adx_short = False
        self.state.adx_wait_bars_left_short = 0
        self.state.adx_wait_trigger_short = ""

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
            self.clear_adx_wait_long()
            self.clear_volume_wait_long()
        else:
            self.state.traded_in_bear_trend = True
            self.state.pending_short_ema_wait = False
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
    ):
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
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

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

    def _signal(
        self,
        bar: pd.Series,
        signal_type: SignalType,
        trigger: str,
        *,
        entry_price: Optional[float] = None,
    ) -> Signal:
        is_long = signal_type == SignalType.BUY
        if entry_price is not None:
            price = float(entry_price)
        elif trigger == "ema_cross":
            c1h = bar.get("close_1h", np.nan)
            price = float(c1h) if pd.notna(c1h) else float(bar["close"])
        else:
            price = float(bar["close"])
        st_key = "supertrend_long" if is_long else "supertrend_short"
        dir_key = "direction_long" if is_long else "direction_short"
        return Signal(
            signal_type=signal_type,
            timestamp=bar.name,
            price=price,
            supertrend_value=float(bar.get(st_key, np.nan)),
            supertrend_direction=int(bar.get(dir_key, 0)),
            ema_1h=float(bar.get("ema_1h", np.nan)),
            close_1h=float(bar.get("close_1h", np.nan)),
            trigger=trigger,
            volume_at_entry=float(bar["volume"]) if pd.notna(bar.get("volume", np.nan)) else None,
            volume_ma_at_entry=float(bar["volume_ma"]) if pd.notna(bar.get("volume_ma", np.nan)) else None,
        )

    @staticmethod
    def _row_volume_confirms(row: pd.Series) -> bool:
        volume = row.get("volume", np.nan)
        volume_ma = row.get("volume_ma", np.nan)
        return pd.notna(volume) and pd.notna(volume_ma) and float(volume_ma) > 0 and float(volume) > float(volume_ma)

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
        pending_adx_long: bool = False,
        pending_adx_short: bool = False,
        adx_wait_bars_left_long: int = 0,
        adx_wait_bars_left_short: int = 0,
        adx_wait_trigger_long: str = "",
        adx_wait_trigger_short: str = "",
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

        close_h = bar.get("close_1h", np.nan)
        ema_h = bar.get("ema_1h", np.nan)
        confirmed_hour = pd.notna(close_h) and pd.notna(ema_h)
        is_new_ema_candle = bool(bar.get("is_new_1h_candle", False))
        crossed_bull_hour = confirmed_hour and is_new_ema_candle and float(close_h) > float(ema_h)
        crossed_bear_hour = confirmed_hour and is_new_ema_candle and float(close_h) < float(ema_h)

        adx_above_long = bool(bar.get("adx_above_threshold_long", bar.get("adx_above_threshold", False)))
        adx_above_short = bool(bar.get("adx_above_threshold_short", bar.get("adx_above_threshold", False)))
        adx_ok_long = (not self.use_adx_long) or adx_above_long
        adx_ok_short = (not self.use_adx_short) or adx_above_short

        if st_bull_flip_long and not traded_in_bull_trend:
            if confirmed_valid and float(close_confirmed) > float(ema_confirmed):
                if adx_ok_long:
                    return self._finalize_volume(
                        self._signal(bar, SignalType.BUY, "st_flip"), updates, volume_window
                    )
                updates["set_adx_wait_long"] = {"bars": self.adx_wait_bars_long, "trigger": "st_flip"}
                return None, updates
            updates["set_pending_long_ema_wait"] = True
            return None, updates

        if st_bear_flip_short and not traded_in_bear_trend:
            if confirmed_valid and float(close_confirmed) < float(ema_confirmed):
                if adx_ok_short:
                    return self._finalize_volume(
                        self._signal(bar, SignalType.SELL, "st_flip"), updates, volume_window
                    )
                updates["set_adx_wait_short"] = {"bars": self.adx_wait_bars_short, "trigger": "st_flip"}
                return None, updates
            updates["set_pending_short_ema_wait"] = True
            return None, updates

        if pending_adx_long and st_bull_long and not traded_in_bull_trend:
            if adx_ok_long:
                trigger = adx_wait_trigger_long or "st_flip"
                updates["clear_adx_wait_long"] = True
                entry_px = float(bar["close"]) if trigger == "ema_cross" else None
                return self._finalize_volume(
                    self._signal(bar, SignalType.BUY, trigger, entry_price=entry_px),
                    updates,
                    volume_window,
                )
            if adx_wait_bars_left_long <= 1:
                updates["clear_adx_wait_long"] = True
            else:
                updates["decrement_adx_wait_long"] = True
            return None, updates

        if pending_adx_short and st_bear_short and not traded_in_bear_trend:
            if adx_ok_short:
                trigger = adx_wait_trigger_short or "st_flip"
                updates["clear_adx_wait_short"] = True
                entry_px = float(bar["close"]) if trigger == "ema_cross" else None
                return self._finalize_volume(
                    self._signal(bar, SignalType.SELL, trigger, entry_price=entry_px),
                    updates,
                    volume_window,
                )
            if adx_wait_bars_left_short <= 1:
                updates["clear_adx_wait_short"] = True
            else:
                updates["decrement_adx_wait_short"] = True
            return None, updates

        if (
            pending_long_ema_wait
            and st_bull_long
            and not traded_in_bull_trend
            and is_new_ema_candle
            and (bool(bar.get("ema_bull_cross", False)) or crossed_bull_hour)
        ):
            if adx_ok_long:
                updates["clear_pending_long_ema_wait"] = True
                entry_px = float(close_h) if confirmed_hour else float(bar["close"])
                return self._finalize_volume(
                    self._signal(bar, SignalType.BUY, "ema_cross", entry_price=entry_px),
                    updates,
                    volume_window,
                )
            updates["clear_pending_long_ema_wait"] = True
            updates["set_adx_wait_long"] = {"bars": self.adx_wait_bars_long, "trigger": "ema_cross"}
            return None, updates

        if (
            pending_short_ema_wait
            and st_bear_short
            and not traded_in_bear_trend
            and is_new_ema_candle
            and (bool(bar.get("ema_bear_cross", False)) or crossed_bear_hour)
        ):
            if adx_ok_short:
                updates["clear_pending_short_ema_wait"] = True
                entry_px = float(close_h) if confirmed_hour else float(bar["close"])
                return self._finalize_volume(
                    self._signal(bar, SignalType.SELL, "ema_cross", entry_price=entry_px),
                    updates,
                    volume_window,
                )
            updates["clear_pending_short_ema_wait"] = True
            updates["set_adx_wait_short"] = {"bars": self.adx_wait_bars_short, "trigger": "ema_cross"}
            return None, updates

        return None, updates
