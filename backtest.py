from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from indicators import (
    attach_ema_entry_1m_close,
    attach_long_short_indicators,
    enrich_with_ema_timeframe,
    resample_ohlcv,
    timeframe_is_hour_based,
)
from strategy import ExitSignal, ExitType, SignalEngine, SignalType, StateManager, Trade


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "config" / "strategy.yaml"
DEFAULT_DATA = BASE_DIR / "banknifty_1min_from2020.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


@dataclass
class BacktestConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    primary_timeframe: str
    ema_timeframe: str
    ema_length: int
    # If set with hour-based ``ema_timeframe``, hourly OHLC is right-labeled at this minute (e.g. 15 -> …:15 IST).
    hourly_bar_end_minute: Optional[int]
    sl_pct_long: float
    tp_pct_long: float
    sl_pct_short: float
    tp_pct_short: float
    use_adx_long: bool
    use_adx_short: bool
    adx_wait_bars_long: int
    adx_wait_bars_short: int
    adx_threshold_long: float
    adx_threshold_short: float
    volume_check: bool
    volume_candle_lookahead: int
    contracts: int
    point_value: float
    commission_per_trade: float
    slippage_points: float
    initial_capital: float
    independent_books: bool
    enable_long_entries: bool = True
    enable_short_entries: bool = True


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.Series
    signals_df: pd.DataFrame
    metrics: Dict[str, Any]


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_dict_sections(config_root: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Merge ``strategy.<key>`` with root ``<key>``; root keys override nested."""
    merged: Dict[str, Any] = {}
    strategy_inner = config_root.get("strategy")
    if isinstance(strategy_inner, dict):
        section = strategy_inner.get(key)
        if isinstance(section, dict):
            merged.update(section)
    root_section = config_root.get(key)
    if isinstance(root_section, dict):
        merged.update(root_section)
    return merged


def resolve_side_configs(strategy_root: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    legacy_st = strategy_root.get("supertrend") or {}
    legacy_risk = strategy_root.get("risk") or {}
    legacy_adx = strategy_root.get("adx") or {}

    long_st_base = strategy_root.get("long_supertrend") or legacy_st
    short_st_base = strategy_root.get("short_supertrend") or legacy_st
    long_st_entry = strategy_root.get("long_supertrend_entry") or long_st_base
    short_st_entry = strategy_root.get("short_supertrend_entry") or short_st_base
    long_st_exit = strategy_root.get("long_supertrend_exit") or long_st_entry
    short_st_exit = strategy_root.get("short_supertrend_exit") or short_st_entry

    return {
        "long_supertrend_entry": long_st_entry,
        "long_supertrend_exit": long_st_exit,
        "short_supertrend_entry": short_st_entry,
        "short_supertrend_exit": short_st_exit,
        "long_risk": strategy_root.get("long_risk") or legacy_risk,
        "short_risk": strategy_root.get("short_risk") or legacy_risk,
        "long_adx": strategy_root.get("long_adx") or legacy_adx,
        "short_adx": strategy_root.get("short_adx") or legacy_adx,
    }


def build_backtest_config(
    config_root: Dict[str, Any],
    data_index: pd.DatetimeIndex,
    start: Optional[str],
    end: Optional[str],
    ema_timeframe_override: Optional[str],
    primary_timeframe_override: Optional[str],
    contracts_override: Optional[int],
) -> BacktestConfig:
    strategy_cfg = config_root
    sides = resolve_side_configs(strategy_cfg)
    ema_cfg = merge_dict_sections(strategy_cfg, "ema")
    execution_cfg = merge_dict_sections(strategy_cfg, "execution")
    contract_cfg = merge_dict_sections(strategy_cfg, "contract")
    costs_cfg = merge_dict_sections(strategy_cfg, "costs")
    timeframes_cfg = merge_dict_sections(strategy_cfg, "timeframes")

    tz = data_index.tz
    start_ts = _parse_bound(start, data_index.min(), tz, is_end=False)
    end_ts = _parse_bound(end, data_index.max(), tz, is_end=True)

    long_risk = sides["long_risk"]
    short_risk = sides["short_risk"]
    long_adx = sides["long_adx"]
    short_adx = sides["short_adx"]

    ema_tf_resolved = ema_timeframe_override or str(ema_cfg.get("timeframe", "1H"))
    if "hourly_bar_end_minute" in timeframes_cfg and timeframes_cfg.get("hourly_bar_end_minute") is None:
        hourly_bar_end: Optional[int] = None
    elif timeframes_cfg.get("hourly_bar_end_minute") is not None:
        hourly_bar_end = int(timeframes_cfg["hourly_bar_end_minute"])
    elif timeframe_is_hour_based(ema_tf_resolved):
        # NSE cash session opens 09:15 IST; hourly bars close at …:15 (TradingView 1H).
        hourly_bar_end = 15
    else:
        hourly_bar_end = None

    return BacktestConfig(
        start_date=start_ts,
        end_date=end_ts,
        primary_timeframe=primary_timeframe_override or timeframes_cfg.get("primary", "5m"),
        ema_timeframe=ema_tf_resolved,
        ema_length=int(ema_cfg.get("length", 200)),
        hourly_bar_end_minute=hourly_bar_end,
        sl_pct_long=float(long_risk.get("stop_loss_pct", 0.35)),
        tp_pct_long=float(long_risk.get("take_profit_pct", 4.0)),
        sl_pct_short=float(short_risk.get("stop_loss_pct", 0.35)),
        tp_pct_short=float(short_risk.get("take_profit_pct", 4.0)),
        use_adx_long=bool(long_adx.get("use_adx", True)),
        use_adx_short=bool(short_adx.get("use_adx", True)),
        adx_wait_bars_long=max(1, int(long_adx.get("consecutive_candles", 5))),
        adx_wait_bars_short=max(1, int(short_adx.get("consecutive_candles", 5))),
        adx_threshold_long=float(long_adx.get("threshold", 24)),
        adx_threshold_short=float(short_adx.get("threshold", 24)),
        volume_check=bool(strategy_cfg.get("volume_check", False)),
        volume_candle_lookahead=max(1, int(strategy_cfg.get("volume_candle_lookahead", 1))),
        contracts=int(contracts_override or execution_cfg.get("contracts", 1)),
        point_value=float(contract_cfg.get("point_value", contract_cfg.get("lot_size", 1))),
        commission_per_trade=float(costs_cfg.get("commission_per_trade", 0)),
        slippage_points=float(costs_cfg.get("slippage_points", 0)),
        initial_capital=float(execution_cfg.get("initial_capital", 100000)),
        independent_books=bool(execution_cfg.get("independent_books", False)),
    )


def _parse_bound(
    raw: Optional[str],
    default: pd.Timestamp,
    tz,
    *,
    is_end: bool,
) -> pd.Timestamp:
    if not raw:
        return default
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None and tz is not None:
        ts = ts.tz_localize(tz)
    elif ts.tzinfo is not None and tz is not None:
        ts = ts.tz_convert(tz)
    if is_end and len(raw) == 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def load_banknifty_csv(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def prepare_backtest_data(
    df_1m: pd.DataFrame,
    strategy_cfg: Dict[str, Any],
    bt_config: BacktestConfig,
) -> pd.DataFrame:
    sides = resolve_side_configs(strategy_cfg)
    volume_ma_period = max(1, int(strategy_cfg.get("volume_ma_period", 20)))

    primary = resample_ohlcv(df_1m, bt_config.primary_timeframe)
    ema_bars = resample_ohlcv(
        df_1m,
        bt_config.ema_timeframe,
        hourly_end_minute=bt_config.hourly_bar_end_minute,
    )
    hour_right_edge = (
        bt_config.hourly_bar_end_minute is not None
        and timeframe_is_hour_based(bt_config.ema_timeframe)
    )
    prepared = enrich_with_ema_timeframe(
        primary,
        ema_bars,
        ema_length=bt_config.ema_length,
        ema_timeframe=bt_config.ema_timeframe,
        shift_cross_for_lookahead=not hour_right_edge,
    )
    prepared = attach_ema_entry_1m_close(prepared, df_1m, bt_config.ema_timeframe)
    prepared = attach_long_short_indicators(
        prepared,
        sides["long_supertrend_entry"],
        sides["short_supertrend_entry"],
        sides["long_adx"],
        sides["short_adx"],
        long_supertrend_exit=sides["long_supertrend_exit"],
        short_supertrend_exit=sides["short_supertrend_exit"],
    )
    prepared["volume_ma"] = prepared["volume"].rolling(
        window=volume_ma_period,
        min_periods=volume_ma_period,
    ).mean()

    return prepared.dropna(
        subset=[
            "ema_1h",
            "close_1h",
            "supertrend_long",
            "direction_long",
            "supertrend_short",
            "direction_short",
        ]
    )


class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_engine = SignalEngine(
            sl_pct_long=config.sl_pct_long,
            tp_pct_long=config.tp_pct_long,
            sl_pct_short=config.sl_pct_short,
            tp_pct_short=config.tp_pct_short,
            use_adx_long=config.use_adx_long,
            use_adx_short=config.use_adx_short,
            adx_wait_bars_long=config.adx_wait_bars_long,
            adx_wait_bars_short=config.adx_wait_bars_short,
            adx_threshold_long=config.adx_threshold_long,
            adx_threshold_short=config.adx_threshold_short,
            volume_check=config.volume_check,
            volume_candle_lookahead=config.volume_candle_lookahead,
            ema_timeframe=config.ema_timeframe,
        )
        self.state_manager = StateManager(
            point_value=config.point_value,
            contracts_per_trade=config.contracts,
        )
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.signal_log: List[Dict[str, Any]] = []
        self._run_df: Optional[pd.DataFrame] = None

    def run(self, df: pd.DataFrame) -> BacktestResult:
        if self.config.independent_books:
            return self._run_independent_books(df)

        self.state_manager = StateManager(
            point_value=self.config.point_value,
            contracts_per_trade=self.config.contracts,
        )
        self.trades = []
        self.equity_history = []
        self.signal_log = []
        self._run_df = df
        equity = self.config.initial_capital

        for i, (_, bar) in enumerate(df.iterrows()):
            timestamp = bar.name
            if timestamp < self.config.start_date:
                continue
            if timestamp > self.config.end_date:
                break

            if self.config.enable_long_entries and self.config.enable_short_entries:
                st_bull_flip = bool(bar.get("st_bull_flip", False))
                st_bear_flip = bool(bar.get("st_bear_flip", False))
            elif self.config.enable_long_entries:
                st_bull_flip = bool(bar.get("st_bull_flip_long", False))
                st_bear_flip = bool(bar.get("st_bear_flip_long", False))
            else:
                st_bull_flip = bool(bar.get("st_bull_flip_short", False))
                st_bear_flip = bool(bar.get("st_bear_flip_short", False))

            self.state_manager.update_supertrend_state(
                st_bull_flip=st_bull_flip,
                st_bear_flip=st_bear_flip,
                current_direction=int(bar.get("direction", 0)),
            )

            state = self.state_manager.state
            if state.position_size != 0:
                exit_signal = self.signal_engine.check_exit_conditions(
                    bar=bar,
                    position_size=state.position_size,
                    entry_price=state.entry_price,
                    stop_loss=state.stop_loss,
                    take_profit=state.take_profit,
                    entry_time=state.entry_time,
                )
                if exit_signal:
                    trade = self._process_exit(exit_signal, bar)
                    if trade:
                        equity += (trade.pnl_value or 0) - self.config.commission_per_trade
                        self.equity_history.append((timestamp, equity))

            state = self.state_manager.state
            if state.position_size == 0:
                volume_window = df.iloc[i : min(i + self.signal_engine.volume_candle_lookahead, len(df))]
                entry_signal, updates = self.signal_engine.evaluate_entry_conditions(
                    bar=bar,
                    position_size=0,
                    traded_in_bull_trend=state.traded_in_bull_trend if self.config.enable_long_entries else True,
                    traded_in_bear_trend=state.traded_in_bear_trend if self.config.enable_short_entries else True,
                    pending_long_ema_wait=state.pending_long_ema_wait if self.config.enable_long_entries else False,
                    pending_short_ema_wait=state.pending_short_ema_wait if self.config.enable_short_entries else False,
                    pending_first_hour_long=state.pending_first_hour_long
                    if self.config.enable_long_entries
                    else False,
                    pending_first_hour_short=state.pending_first_hour_short
                    if self.config.enable_short_entries
                    else False,
                    pending_first_hour_trigger_long=state.pending_first_hour_trigger_long
                    if self.config.enable_long_entries
                    else "",
                    pending_first_hour_trigger_short=state.pending_first_hour_trigger_short
                    if self.config.enable_short_entries
                    else "",
                    pending_first_hour_deferred_long=state.pending_first_hour_deferred_long
                    if self.config.enable_long_entries
                    else None,
                    pending_first_hour_deferred_short=state.pending_first_hour_deferred_short
                    if self.config.enable_short_entries
                    else None,
                    pending_adx_long=state.pending_adx_long if self.config.enable_long_entries else False,
                    pending_adx_short=state.pending_adx_short if self.config.enable_short_entries else False,
                    adx_wait_bars_left_long=state.adx_wait_bars_left_long if self.config.enable_long_entries else 0,
                    adx_wait_bars_left_short=state.adx_wait_bars_left_short if self.config.enable_short_entries else 0,
                    adx_wait_trigger_long=state.adx_wait_trigger_long if self.config.enable_long_entries else "",
                    adx_wait_trigger_short=state.adx_wait_trigger_short if self.config.enable_short_entries else "",
                    deferred_ema_cross_long=state.deferred_ema_cross_long
                    if self.config.enable_long_entries
                    else None,
                    deferred_ema_cross_short=state.deferred_ema_cross_short
                    if self.config.enable_short_entries
                    else None,
                    volume_window=volume_window,
                )
                self._apply_updates(updates)

                if entry_signal and (
                    (entry_signal.signal_type == SignalType.BUY and not self.config.enable_long_entries)
                    or (entry_signal.signal_type == SignalType.SELL and not self.config.enable_short_entries)
                ):
                    entry_signal = None

                if entry_signal:
                    self._process_entry(entry_signal)
                    self.signal_log.append(
                        {
                            "timestamp": entry_signal.timestamp,
                            "signal": entry_signal.signal_type.value,
                            "price": entry_signal.price,
                            "trigger": entry_signal.trigger,
                        }
                    )

            state = self.state_manager.state
            if not self.equity_history or self.equity_history[-1][0] != timestamp:
                unrealized = self._calculate_unrealized(state, float(bar["close"]))
                self.equity_history.append((timestamp, equity + unrealized))

        if self.state_manager.state.position_size != 0:
            final_bar = df.loc[: self.config.end_date].iloc[-1]
            exit_signal = ExitSignal(
                ExitType.FORCED_CLOSE,
                final_bar.name,
                float(final_bar["close"]),
                self.state_manager.state.entry_price,
                0.0,
            )
            trade = self._process_exit(exit_signal, final_bar)
            if trade:
                equity += (trade.pnl_value or 0) - self.config.commission_per_trade
                self.equity_history.append((final_bar.name, equity))

        equity_curve = pd.Series(
            data=[item[1] for item in self.equity_history],
            index=[item[0] for item in self.equity_history],
            name="equity",
        )
        signals_df = pd.DataFrame(self.signal_log)
        metrics = calculate_metrics(self.trades, equity_curve, self.config.initial_capital)
        return BacktestResult(self.config, self.trades, equity_curve, signals_df, metrics)

    def _run_independent_books(self, df: pd.DataFrame) -> BacktestResult:
        base = replace(self.config, independent_books=False)
        long_result = BacktestEngine(replace(base, enable_long_entries=True, enable_short_entries=False)).run(df)
        short_result = BacktestEngine(replace(base, enable_long_entries=False, enable_short_entries=True)).run(df)

        merged_trades = sorted(
            [*long_result.trades, *short_result.trades],
            key=lambda trade: (trade.exit_time or trade.entry_time, trade.trade_id),
        )
        for trade_id, trade in enumerate(merged_trades, 1):
            trade.trade_id = trade_id

        long_eq_raw = long_result.equity_curve.groupby(level=0).last()
        short_eq_raw = short_result.equity_curve.groupby(level=0).last()
        long_eq = long_eq_raw.reindex(df.index).ffill().fillna(self.config.initial_capital)
        short_eq = short_eq_raw.reindex(df.index).ffill().fillna(self.config.initial_capital)
        equity_curve = (long_eq + short_eq) - self.config.initial_capital
        equity_curve.name = "equity"

        signals_df = pd.concat([long_result.signals_df, short_result.signals_df], ignore_index=True)
        if not signals_df.empty and "timestamp" in signals_df.columns:
            signals_df = signals_df.sort_values("timestamp").reset_index(drop=True)

        metrics = calculate_metrics(merged_trades, equity_curve, self.config.initial_capital)
        self.trades = merged_trades
        return BacktestResult(self.config, merged_trades, equity_curve, signals_df, metrics)

    def _apply_updates(self, updates: Dict[str, Any]) -> None:
        sm = self.state_manager
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

    def _process_entry(self, signal) -> None:
        if signal.signal_type == SignalType.BUY:
            signal.price += self.config.slippage_points
        else:
            signal.price -= self.config.slippage_points
        stop_loss, take_profit = self.signal_engine.calculate_exit_levels(
            signal.price,
            is_long=signal.signal_type == SignalType.BUY,
        )
        self.state_manager.on_entry(signal, stop_loss, take_profit)

    def _process_exit(self, exit_signal: ExitSignal, bar: pd.Series) -> Optional[Trade]:
        if exit_signal.exit_type in {ExitType.ST_FLIP, ExitType.FORCED_CLOSE}:
            is_long = self.state_manager.state.position_size > 0
            if is_long:
                exit_signal.exit_price = float(bar["close"]) - self.config.slippage_points
            else:
                exit_signal.exit_price = float(bar["close"]) + self.config.slippage_points
        trade = self.state_manager.on_exit(exit_signal)
        if trade:
            self._populate_trade_excursions(trade)
            self.trades.append(trade)
        return trade

    def _calculate_unrealized(self, state, current_price: float) -> float:
        if state.position_size == 0:
            return 0.0
        pnl_points = current_price - state.entry_price if state.position_size > 0 else state.entry_price - current_price
        return pnl_points * self.config.point_value * abs(state.position_size) * self.config.contracts

    def _populate_trade_excursions(self, trade: Trade) -> None:
        if self._run_df is None or trade.entry_time is None or trade.exit_time is None:
            return
        trade_slice = self._run_df.loc[trade.entry_time : trade.exit_time]
        if trade_slice.empty or not trade.entry_price:
            return
        highest = float(trade_slice["high"].max())
        lowest = float(trade_slice["low"].min())
        entry = float(trade.entry_price)
        if trade.direction == "long":
            max_pos = highest - entry
            max_neg = lowest - entry
        else:
            max_pos = entry - lowest
            max_neg = entry - highest
        trade.max_positive_points = max_pos
        trade.max_negative_points = max_neg
        trade.max_positive_pct = (max_pos / entry) * 100
        trade.max_negative_pct = (max_neg / entry) * 100

    def get_trade_summary(self) -> pd.DataFrame:
        return trades_to_dataframe(self.trades)


def calculate_metrics(
    trades: List[Trade],
    equity_curve: pd.Series,
    initial_capital: float,
) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "total_points": 0.0,
            "final_equity": float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital,
        }

    winners = [t for t in trades if (t.pnl_value or 0) > 0]
    losers = [t for t in trades if (t.pnl_value or 0) < 0]
    gross_profit = sum(t.pnl_value or 0 for t in winners)
    gross_loss = abs(sum(t.pnl_value or 0 for t in losers))
    net_profit = gross_profit - gross_loss
    total_points = sum(t.pnl_points or 0 for t in trades)

    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) else initial_capital
    max_dd, max_dd_pct = _drawdown(equity_curve)
    longs = [t for t in trades if t.direction == "long"]
    shorts = [t for t in trades if t.direction == "short"]

    return {
        "total_trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": (len(winners) / len(trades)) * 100,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0),
        "avg_trade": net_profit / len(trades),
        "final_equity": final_equity,
        "total_return": final_equity - initial_capital,
        "total_return_pct": ((final_equity - initial_capital) / initial_capital) * 100,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "total_points": total_points,
        "total_profit_points": sum(t.pnl_points or 0 for t in trades if (t.pnl_points or 0) > 0),
        "total_loss_points": sum(t.pnl_points or 0 for t in trades if (t.pnl_points or 0) < 0),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_win_rate": _win_rate(longs),
        "short_win_rate": _win_rate(shorts),
        "long_pnl_points": sum(t.pnl_points or 0 for t in longs),
        "short_pnl_points": sum(t.pnl_points or 0 for t in shorts),
        "long_pnl_value": sum(t.pnl_value or 0 for t in longs),
        "short_pnl_value": sum(t.pnl_value or 0 for t in shorts),
        "tp_exits": len([t for t in trades if t.exit_type == ExitType.TAKE_PROFIT.value]),
        "sl_exits": len([t for t in trades if t.exit_type == ExitType.STOP_LOSS.value]),
        "st_flip_exits": len([t for t in trades if t.exit_type == ExitType.ST_FLIP.value]),
    }


def _win_rate(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return (len([t for t in trades if (t.pnl_value or 0) > 0]) / len(trades)) * 100


def _drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    if equity_curve.empty:
        return 0.0, 0.0
    running_max = equity_curve.expanding().max()
    drawdown = running_max - equity_curve
    drawdown_pct = (drawdown / running_max) * 100
    return float(drawdown.max()), float(drawdown_pct.max())


# (series label, bullish flip column, bearish flip column); see indicators.attach_long_short_indicators
_ST_FLIP_SERIES: Tuple[Tuple[str, str, str], ...] = (
    ("long_entry", "st_bull_flip_long", "st_bear_flip_long"),
    ("short_entry", "st_bull_flip_short", "st_bear_flip_short"),
    ("long_exit", "st_bull_flip_long_exit", "st_bear_flip_long_exit"),
    ("short_exit", "st_bull_flip_short_exit", "st_bear_flip_short_exit"),
)


def collect_supertrend_flips(
    prepared: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """One row per regime flip on a primary bar (long/short entry/exit STs merged when they agree)."""
    rows: List[Dict[str, Any]] = []
    if prepared.empty:
        return pd.DataFrame(columns=["timestamp", "flip", "series", "price"])

    mask = (prepared.index >= start_date) & (prepared.index <= end_date)
    subset = prepared.loc[mask]
    for ts, bar in subset.iterrows():
        price = float(bar["close"])
        for series_label, bull_col, bear_col in _ST_FLIP_SERIES:
            if bool(bar.get(bull_col, False)):
                rows.append(
                    {"timestamp": ts, "flip": "bullish", "series": series_label, "price": price}
                )
            if bool(bar.get(bear_col, False)):
                rows.append(
                    {"timestamp": ts, "flip": "bearish", "series": series_label, "price": price}
                )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "flip", "series", "price"])
    out = pd.DataFrame(rows)
    out = (
        out.groupby(["timestamp", "flip"], as_index=False, sort=False)
        .agg(
            price=("price", "first"),
            series=("series", lambda s: ",".join(sorted(s.unique()))),
        )
    )
    return out.sort_values(["timestamp", "flip"]).reset_index(drop=True)


def append_session_log(log_path: Path, line: str) -> None:
    """Append one line to the session log (UTF-8). Creates parent dirs if needed."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line.rstrip("\r\n") + "\n")


def init_session_log(
    log_path: Path,
    *,
    data_path: Path,
    config_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    primary_timeframe: str,
    ema_timeframe: str,
    hourly_bar_end_minute: Optional[int] = None,
) -> None:
    """Start a new session log file (overwrites if present). Add sections with append_session_log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("=== Bank Nifty session log ===\n")
        handle.write(f"started_at={datetime.now().isoformat(timespec='seconds')}\n")
        handle.write(f"data={data_path}\n")
        handle.write(f"config={config_path}\n")
        handle.write(f"period={start_date} to {end_date}\n")
        handle.write(f"primary_timeframe={primary_timeframe} ema_timeframe={ema_timeframe}\n")
        if hourly_bar_end_minute is not None:
            handle.write(f"hourly_bar_end_minute={hourly_bar_end_minute}\n")


def append_supertrend_flips_to_session_log(
    log_path: Path,
    prepared: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    append_session_log(log_path, "")
    append_session_log(log_path, "=== Supertrend flips ===")
    append_session_log(
        log_path,
        "# format: timestamp | flip | series | price; series is comma-separated when several STs flip together",
    )
    flips = collect_supertrend_flips(prepared, start_date, end_date)
    if flips.empty:
        append_session_log(log_path, "(none in backtest date range)")
        return
    for _, row in flips.iterrows():
        append_session_log(
            log_path,
            f"{row['timestamp']} | {row['flip']} | {row['series']} | {float(row['price']):.2f}",
        )


def append_backtest_summary_to_session_log(
    log_path: Path,
    result: BacktestResult,
    output_csv: Path,
) -> None:
    metrics = result.metrics
    append_session_log(log_path, "")
    append_session_log(log_path, "=== Backtest summary ===")
    append_session_log(log_path, f"result_csv={output_csv}")
    append_session_log(
        log_path,
        f"total_trades={metrics.get('total_trades', 0)} "
        f"win_rate_pct={metrics.get('win_rate', 0):.2f} "
        f"net_profit={metrics.get('net_profit', 0):.2f} "
        f"net_points={metrics.get('total_points', 0):.2f} "
        f"max_drawdown_pct={metrics.get('max_drawdown_pct', 0):.2f}",
    )


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        rows.append(
            {
                "trade_id": trade.trade_id,
                "direction": trade.direction,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "signal_type": trade.entry_trigger,
                "exit_type": trade.exit_type,
                "pnl_points": trade.pnl_points,
                "pnl_value": trade.pnl_value,
                "contracts": trade.contracts,
                "ema_at_entry": trade.ema_at_entry,
                "volume_at_entry": trade.volume_at_entry,
                "volume_ma_at_entry": trade.volume_ma_at_entry,
                "max_positive_points": trade.max_positive_points,
                "max_negative_points": trade.max_negative_points,
                "max_positive_pct": trade.max_positive_pct,
                "max_negative_pct": trade.max_negative_pct,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    result: BacktestResult,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    stamp: Optional[str] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"banknifty_backtest_{stamp}.csv"

    metrics = result.metrics
    report_rows = [
        ("Period", f"{result.config.start_date} to {result.config.end_date}"),
        ("Primary Timeframe", result.config.primary_timeframe),
        ("EMA Timeframe", result.config.ema_timeframe),
        ("EMA Length", result.config.ema_length),
        (
            "Hourly bar end (minute)",
            result.config.hourly_bar_end_minute
            if result.config.hourly_bar_end_minute is not None
            else "legacy (15m offset)",
        ),
        ("Contracts", result.config.contracts),
        ("Initial Capital", result.config.initial_capital),
        ("Final Equity", metrics.get("final_equity", 0)),
        ("Net Profit/Loss", metrics.get("net_profit", 0)),
        ("Total P&L Points", metrics.get("total_points", 0)),
        ("Total Return %", metrics.get("total_return_pct", 0)),
        ("Max Drawdown %", metrics.get("max_drawdown_pct", 0)),
        ("Total Trades", metrics.get("total_trades", 0)),
        ("Winning Trades", metrics.get("winning_trades", 0)),
        ("Losing Trades", metrics.get("losing_trades", 0)),
        ("Win Rate %", metrics.get("win_rate", 0)),
        ("Profit Factor", metrics.get("profit_factor", 0)),
        ("Long Trades", metrics.get("long_trades", 0)),
        ("Long Win Rate %", metrics.get("long_win_rate", 0)),
        ("Long P&L Points", metrics.get("long_pnl_points", 0)),
        ("Short Trades", metrics.get("short_trades", 0)),
        ("Short Win Rate %", metrics.get("short_win_rate", 0)),
        ("Short P&L Points", metrics.get("short_pnl_points", 0)),
        ("Take Profit Exits", metrics.get("tp_exits", 0)),
        ("Stop Loss Exits", metrics.get("sl_exits", 0)),
        ("Supertrend Flip Exits", metrics.get("st_flip_exits", 0)),
    ]
    report_df = pd.DataFrame(report_rows, columns=["metric", "value"])
    trades_df = trades_to_dataframe(result.trades)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        report_df.to_csv(f, index=False)
        f.write("\n")
        trades_df.to_csv(f, index=False, float_format="%.2f")
    return output_path


def print_summary(
    result: BacktestResult,
    output_path: Path,
    session_log_path: Optional[Path] = None,
) -> None:
    metrics = result.metrics
    print("\nBANKNIFTY STANDALONE BACKTEST RESULTS")
    print("=" * 46)
    print(f"Period:      {result.config.start_date} to {result.config.end_date}")
    print(f"Bars TF:     {result.config.primary_timeframe}")
    print(f"EMA TF:      {result.config.ema_timeframe}")
    print(f"Trades:      {metrics.get('total_trades', 0)}")
    print(f"Win Rate:    {metrics.get('win_rate', 0):.2f}%")
    print(f"Net Points:  {metrics.get('total_points', 0):.2f}")
    print(f"Net P&L:     {metrics.get('net_profit', 0):.2f}")
    print(f"Max DD:      {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Output file: {output_path}")
    if session_log_path is not None:
        print(f"Session log: {session_log_path}")


def prompt_date_range_if_needed(
    args: argparse.Namespace,
    data_index: pd.DatetimeIndex,
) -> Tuple[Optional[str], Optional[str]]:
    """Ask for missing CLI date bounds, using full CSV range when left blank."""
    def _validated_bounds(start_raw: str, end_raw: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = _parse_bound(start_raw, data_index.min(), data_index.tz, is_end=False)
        end_ts = _parse_bound(end_raw, data_index.max(), data_index.tz, is_end=True)
        if start_ts > end_ts:
            raise ValueError(
                f"Start date must be before or equal to end date: {start_raw} > {end_raw}"
            )
        return start_ts, end_ts

    start = args.start
    end = args.end
    if start and end:
        _validated_bounds(start, end)
        return start, end

    earliest = data_index.min().strftime("%Y-%m-%d")
    latest = data_index.max().strftime("%Y-%m-%d")
    print("\nBacktest Date Range")
    print("-" * 30)
    print(f"Available data: {earliest} to {latest}")
    print("Press Enter to use the shown default.\n")

    while True:
        if not start:
            entered = input(f"Start date [{earliest}]: ").strip()
            start = entered or earliest
        if not end:
            entered = input(f"End date [{latest}]: ").strip()
            end = entered or latest

        try:
            _validated_bounds(start, end)
            return start, end
        except ValueError as exc:
            print(f"\nInvalid date range: {exc}")
            print("Please enter the dates again.\n")
            if not args.start:
                start = None
            if not args.end:
                end = None


def run_from_args(args: argparse.Namespace) -> BacktestResult:
    data_path = Path(args.data).resolve()
    config_path = Path(args.config).resolve()
    strategy_cfg = load_config(config_path)
    df_1m = load_banknifty_csv(data_path)
    start, end = prompt_date_range_if_needed(args, df_1m.index)
    bt_config = build_backtest_config(
        strategy_cfg,
        df_1m.index,
        start=start,
        end=end,
        ema_timeframe_override=args.ema_timeframe,
        primary_timeframe_override=args.primary_timeframe,
        contracts_override=args.contracts,
    )
    prepared = prepare_backtest_data(df_1m, strategy_cfg, bt_config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir).resolve()
    session_log_path = out_dir / f"banknifty_{stamp}.log"
    init_session_log(
        session_log_path,
        data_path=data_path,
        config_path=config_path,
        start_date=bt_config.start_date,
        end_date=bt_config.end_date,
        primary_timeframe=bt_config.primary_timeframe,
        ema_timeframe=bt_config.ema_timeframe,
        hourly_bar_end_minute=bt_config.hourly_bar_end_minute,
    )
    append_supertrend_flips_to_session_log(
        session_log_path,
        prepared,
        bt_config.start_date,
        bt_config.end_date,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(prepared)
    output_path = write_outputs(result, out_dir, stamp=stamp)
    append_backtest_summary_to_session_log(session_log_path, result, output_path)
    print_summary(result, output_path, session_log_path=session_log_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Bank Nifty Supertrend + EMA backtest")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to Bank Nifty 1-minute CSV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to standalone strategy YAML")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for result CSV and session .log files",
    )
    parser.add_argument("--start", help="Backtest start date, e.g. 2024-01-01. If omitted, CLI asks.")
    parser.add_argument("--end", help="Backtest end date, e.g. 2024-12-31. If omitted, CLI asks.")
    parser.add_argument("--ema-timeframe", help="Override EMA timeframe, e.g. 10m, 20m, 30m, 1H")
    parser.add_argument("--primary-timeframe", help="Override primary execution timeframe")
    parser.add_argument("--contracts", type=int, help="Override contracts/quantity")
    return parser


def main() -> None:
    parser = build_arg_parser()
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
