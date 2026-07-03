"""Replay primary bars for a date and print strategy decisions (matches live step_bar path)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from backtest import prepare_backtest_data
from live.bar_cache import load_bars_csv, resolve_bars_path
from live.bar_decision_log import log_primary_bar_decision, snapshot_state
from live.config import build_backtest_config_for_live, load_live_config
from live.logging_setup import setup_logging
from strategy import SignalEngine
from strategy_runtime import step_bar
from indicators import timeframe_to_rule

IST = "Asia/Kolkata"
logger = logging.getLogger("live.check_day")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay live step_bar decisions for one session day")
    parser.add_argument("--date", required=True, help="Session date YYYY-MM-DD")
    parser.add_argument("--data", default="live/cache/banknifty_1min_neo.csv", help="1-min CSV path")
    parser.add_argument("--around", help="Highlight bars near HH:MM (e.g. 13:40)")
    args = parser.parse_args()

    setup_logging()
    config_root = load_live_config()
    bt_config = build_backtest_config_for_live(config_root)
    live_cfg = config_root.get("live") or {}
    bars_path = resolve_bars_path(live_cfg.get("bars_csv", args.data))

    df_1m = load_bars_csv(bars_path)
    if df_1m.empty:
        print(f"No data in {bars_path}", file=sys.stderr)
        raise SystemExit(1)

    prepared = prepare_backtest_data(df_1m, config_root, bt_config)
    day = pd.Timestamp(args.date, tz=IST)
    day_bars = prepared.loc[(prepared.index >= day) & (prepared.index < day + pd.Timedelta(days=1))]

    if day_bars.empty:
        print(f"No primary bars for {args.date}", file=sys.stderr)
        raise SystemExit(1)

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
    from strategy import StateManager

    state_manager = StateManager(
        point_value=bt_config.point_value,
        contracts_per_trade=bt_config.contracts,
    )

    # Warm up state on all bars before the target day (same as live would have history).
    warmup_end = prepared.index.get_loc(day_bars.index[0])
    for i in range(warmup_end):
        ts = prepared.index[i]
        bar = prepared.iloc[i]
        state_before = snapshot_state(state_manager)
        result = step_bar(
            bar=bar,
            bar_index=i,
            df=prepared,
            state_manager=state_manager,
            signal_engine=signal_engine,
            config=bt_config,
        )
        if result.exit_signal:
            state_manager.on_exit(result.exit_signal)
        if result.entry_signal:
            sig = result.entry_signal
            is_long = sig.signal_type.name == "BUY"
            sl, tp = signal_engine.calculate_exit_levels(float(sig.price), is_long=is_long)
            state_manager.on_entry(sig, sl, tp)

    highlight_minutes: set[tuple[int, int]] = set()
    if args.around:
        hh, mm = map(int, args.around.split(":"))
        for delta in range(-15, 20, 5):
            t = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz=IST) + pd.Timedelta(minutes=delta)
            highlight_minutes.add((t.hour, t.minute))

    print(f"\n=== Decisions for {args.date} ({len(day_bars)} primary bars) ===\n")
    entries = 0
    for ts in day_bars.index:
        i = prepared.index.get_loc(ts)
        bar = prepared.loc[ts]
        state_before = snapshot_state(state_manager)
        result = step_bar(
            bar=bar,
            bar_index=int(i) if not isinstance(i, slice) else int(i.start or 0),
            df=prepared,
            state_manager=state_manager,
            signal_engine=signal_engine,
            config=bt_config,
        )
        log_primary_bar_decision(
            ts,
            bar,
            state_before=state_before,
            result=result,
            skip_entry=False,
            bt_config=bt_config,
            signal_engine=signal_engine,
        )
        if result.entry_signal:
            entries += 1
            sig = result.entry_signal
            is_long = sig.signal_type.name == "BUY"
            sl, tp = signal_engine.calculate_exit_levels(float(sig.price), is_long=is_long)
            state_manager.on_entry(sig, sl, tp)
        if result.exit_signal:
            state_manager.on_exit(result.exit_signal)

        if highlight_minutes and (ts.hour, ts.minute) in highlight_minutes:
            print(f"  >>> highlighted bar {ts}")

    print(f"\nTotal entry signals on {args.date}: {entries}")


if __name__ == "__main__":
    main()
