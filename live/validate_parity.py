"""Validate live signal parity: replay primary bars through step_bar vs backtest.

Compares entry/exit signal timestamps for a given date range on the same 1-min data.
SL/TP intrabar exits are excluded (live-only behaviour).

Usage:
    python -m live.validate_parity --start 2025-06-10 --end 2025-06-13
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import pandas as pd

from backtest import BacktestConfig, BacktestEngine, build_backtest_config, load_banknifty_csv, prepare_backtest_data
from indicators import timeframe_to_rule
from live.config import load_live_config
from strategy import ExitType, SignalEngine, SignalType, StateManager
from strategy_runtime import step_bar

logger = logging.getLogger("live.validate")
IST = "Asia/Kolkata"


@dataclass
class LiveSignal:
    timestamp: pd.Timestamp
    kind: str  # entry | exit
    direction: str
    trigger: str
    price: float


def _collect_live_signals(
    prepared: pd.DataFrame,
    bt_config: BacktestConfig,
) -> List[LiveSignal]:
    """Mirror BacktestEngine.run() using step_bar (same code path as live bot)."""
    signal_engine = SignalEngine(
        sl_pct_long=bt_config.sl_pct_long,
        tp_pct_long=bt_config.tp_pct_long,
        sl_pct_short=bt_config.sl_pct_short,
        tp_pct_short=bt_config.tp_pct_short,
        use_adx_long=bt_config.use_adx_long,
        use_adx_short=bt_config.use_adx_short,
        adx_wait_bars_long=bt_config.adx_wait_bars_long,
        adx_wait_bars_short=bt_config.adx_wait_bars_short,
        adx_threshold_long=bt_config.adx_threshold_long,
        adx_threshold_short=bt_config.adx_threshold_short,
        volume_check=bt_config.volume_check,
        volume_candle_lookahead=bt_config.volume_candle_lookahead,
        ema_timeframe=bt_config.ema_timeframe,
        ema_on_primary=timeframe_to_rule(bt_config.ema_timeframe)
        == timeframe_to_rule(bt_config.primary_timeframe),
    )
    state_manager = StateManager(
        point_value=bt_config.point_value,
        contracts_per_trade=bt_config.contracts,
    )
    signals: List[LiveSignal] = []

    for i, (_, bar) in enumerate(prepared.iterrows()):
        timestamp = bar.name
        if timestamp < bt_config.start_date:
            continue
        if timestamp > bt_config.end_date:
            break

        result = step_bar(
            bar=bar,
            bar_index=i,
            df=prepared,
            state_manager=state_manager,
            signal_engine=signal_engine,
            config=bt_config,
        )

        if result.exit_signal:
            exit_sig = result.exit_signal
            direction = "long" if state_manager.state.position_size > 0 else "short"
            if exit_sig.exit_type in {ExitType.ST_FLIP, ExitType.FORCED_CLOSE}:
                is_long = state_manager.state.position_size > 0
                if is_long:
                    exit_sig.exit_price = float(bar["close"]) - bt_config.slippage_points
                else:
                    exit_sig.exit_price = float(bar["close"]) + bt_config.slippage_points
            signals.append(
                LiveSignal(
                    timestamp,
                    "exit",
                    direction,
                    exit_sig.exit_type.value,
                    float(exit_sig.exit_price),
                )
            )
            state_manager.on_exit(exit_sig)

        if result.entry_signal:
            entry = result.entry_signal
            direction = "long" if entry.signal_type == SignalType.BUY else "short"
            if entry.signal_type == SignalType.BUY:
                entry.price += bt_config.slippage_points
            else:
                entry.price -= bt_config.slippage_points
            sl, tp = signal_engine.calculate_exit_levels(float(entry.price), is_long=direction == "long")
            state_manager.on_entry(entry, sl, tp)
            signals.append(
                LiveSignal(
                    timestamp,
                    "entry",
                    direction,
                    entry.trigger,
                    float(entry.price),
                )
            )

    if state_manager.state.position_size != 0:
        final_bar = prepared.loc[: bt_config.end_date].iloc[-1]
        is_long = state_manager.state.position_size > 0
        price = float(final_bar["close"])
        if is_long:
            price -= bt_config.slippage_points
        else:
            price += bt_config.slippage_points
        signals.append(
            LiveSignal(
                final_bar.name,
                "exit",
                "long" if is_long else "short",
                ExitType.FORCED_CLOSE.value,
                price,
            )
        )

    return signals


def _collect_backtest_signals(result) -> List[LiveSignal]:
    out: List[LiveSignal] = []
    for trade in result.trades:
        out.append(
            LiveSignal(
                pd.Timestamp(trade.entry_time),
                "entry",
                trade.direction,
                trade.entry_trigger,
                float(trade.entry_price),
            )
        )
        out.append(
            LiveSignal(
                pd.Timestamp(trade.exit_time),
                "exit",
                trade.direction,
                trade.exit_type,
                float(trade.exit_price),
            )
        )
    return sorted(out, key=lambda s: s.timestamp)


def compare_signals(live: List[LiveSignal], bt: List[LiveSignal]) -> Tuple[int, List[str]]:
    mismatches: List[str] = []
    n = max(len(live), len(bt))
    for i in range(n):
        ls = live[i] if i < len(live) else None
        bs = bt[i] if i < len(bt) else None
        if ls is None:
            mismatches.append(f"[{i}] backtest only: {bs}")
            continue
        if bs is None:
            mismatches.append(f"[{i}] live only: {ls}")
            continue
        if ls.kind != bs.kind or ls.direction != bs.direction:
            mismatches.append(f"[{i}] kind/dir: live={ls} bt={bs}")
            continue
        if abs(ls.timestamp - bs.timestamp) > pd.Timedelta(minutes=1):
            mismatches.append(f"[{i}] time: live={ls.timestamp} bt={bs.timestamp}")
        if ls.trigger != bs.trigger and not (ls.kind == "exit" and bs.kind == "exit"):
            mismatches.append(f"[{i}] trigger: live={ls.trigger} bt={bs.trigger}")
    return len(mismatches), mismatches


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compare live step_bar signals vs backtest")
    parser.add_argument("--data", default="banknifty_1min_from2020.csv")
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args(argv)

    config_root = load_live_config()
    df_1m = load_banknifty_csv(args.data)
    start = pd.Timestamp(args.start).tz_localize(IST)
    end = pd.Timestamp(args.end).tz_localize(IST) + pd.Timedelta(hours=23, minutes=59)

    bt_config = replace(
        build_backtest_config(
            config_root,
            df_1m.index,
            start=args.start,
            end=args.end,
            ema_timeframe_override=None,
            primary_timeframe_override=None,
            contracts_override=None,
        ),
        independent_books=False,
    )
    prepared = prepare_backtest_data(df_1m, config_root, bt_config)

    live_signals = _collect_live_signals(prepared, bt_config)
    engine = BacktestEngine(bt_config)
    bt_result = engine.run(prepared)
    bt_signals = _collect_backtest_signals(bt_result)

    # Bar-close parity: compare all signals (backtest uses bar-close SL/TP; live uses ticks for SL/TP).
    n_bad, details = compare_signals(live_signals, bt_signals)
    print(f"Live step_bar signals: {len(live_signals)} | Backtest trades signals: {len(bt_signals)}")
    if n_bad == 0:
        print("PASS: bar-close entry/exit signals match backtest")
        return 0
    print(f"FAIL: {n_bad} mismatch(es)")
    for line in details[:20]:
        print(" ", line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
