"""Live trading engine: reuses backtest strategy logic and maps signals to option legs."""
from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest import BacktestConfig, prepare_backtest_data
from indicators import timeframe_to_rule
from live.bar_decision_log import log_primary_bar_decision, snapshot_state
from live.config import LiveConfig
from live.neo_client import BUY, Leg, LegOrder, NeoBroker, SELL
from live.persistence import LivePosition, load_state, save_state
from live.safety import OrderFillWatcher, legs_filled, place_legs_safe, square_off_safe
from strategy import ExitSignal, ExitType, Signal, SignalEngine, SignalType, StateManager, Trade
from strategy_runtime import step_bar

logger = logging.getLogger("live.engine")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
IST = "Asia/Kolkata"


class LiveTrader:
    def __init__(
        self,
        broker: NeoBroker,
        live_cfg: LiveConfig,
        bt_config: BacktestConfig,
        strategy_cfg: Dict[str, Any],
        df_1m: pd.DataFrame,
        *,
        order_fill_watcher: Optional[OrderFillWatcher] = None,
    ):
        self.broker = broker
        self.live_cfg = live_cfg
        self.bt_config = bt_config
        self.strategy_cfg = strategy_cfg
        self.df_1m = df_1m
        self.order_fill_watcher = order_fill_watcher
        self._last_index_price: Optional[float] = None
        self.prepared: Optional[pd.DataFrame] = None

        self.signal_engine = SignalEngine(
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
        self.state_manager = StateManager(
            point_value=bt_config.point_value,
            contracts_per_trade=bt_config.contracts,
        )
        self.live_position: Optional[LivePosition] = None
        self.flat_until_next_bar = False
        self.allow_entries = True
        self.entries_disabled_reason = ""
        self._trade_log_path = RESULTS_DIR / f"live_trades_{datetime.now().strftime('%Y%m%d')}.csv"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self._restore_state()
        logger.info("Trade log: %s", self._trade_log_path)
        logger.info(
            "Engine ready — position_size=%s flat_until_next_bar=%s open_legs=%s",
            self.state_manager.state.position_size,
            self.flat_until_next_bar,
            "yes" if self.live_position else "no",
        )

    def _restore_state(self) -> None:
        ss, lp, flat, last_session_date = load_state()
        if ss is not None:
            self.state_manager.state = ss
            logger.info("Restored strategy state (position_size=%s)", ss.position_size)
        else:
            logger.info("No saved strategy state; starting fresh")
        if lp is not None:
            self.live_position = lp
            logger.info(
                "Restored live position %s strike=%s expiry=%s (%s / %s)",
                lp.direction,
                lp.strike,
                lp.expiry,
                lp.ce_symbol,
                lp.pe_symbol,
            )
        else:
            logger.info("No open live position in state file")

        today = date.today().isoformat()
        if (
            flat
            and self.state_manager.state.position_size == 0
            and last_session_date
            and last_session_date != today
        ):
            logger.info(
                "Clearing stale flat_until_next_bar from prior session (%s)",
                last_session_date,
            )
            flat = False
        self.flat_until_next_bar = flat

    def _persist(self) -> None:
        save_state(
            self.state_manager.state,
            self.live_position,
            flat_until_next_bar=self.flat_until_next_bar,
            last_session_date=date.today().isoformat(),
        )

    def update_df(self, df_1m: pd.DataFrame) -> None:
        self.df_1m = df_1m

    def _reprepare(self) -> pd.DataFrame:
        self.prepared = prepare_backtest_data(self.df_1m, self.strategy_cfg, self.bt_config)
        return self.prepared

    def _leg_qty(self, leg: Leg) -> int:
        return self.live_cfg.lots * leg.lot_size

    def _entry_orders(self, direction: str, legs: Dict[str, Leg]) -> List[LegOrder]:
        qty_ce = self._leg_qty(legs["ce"])
        qty_pe = self._leg_qty(legs["pe"])
        if direction == "long":
            return [
                LegOrder(legs["ce"], BUY, qty_ce),
                LegOrder(legs["pe"], SELL, qty_pe),
            ]
        return [
            LegOrder(legs["ce"], SELL, qty_ce),
            LegOrder(legs["pe"], BUY, qty_pe),
        ]

    def _exit_orders(self, pos: LivePosition) -> List[LegOrder]:
        ce = Leg(
            trading_symbol=pos.ce_symbol,
            token=pos.ce_token,
            lot_size=pos.lot_size,
            strike=float(pos.strike),
            expiry=pos.expiry,
            option_type="CE",
        )
        pe = Leg(
            trading_symbol=pos.pe_symbol,
            token=pos.pe_token,
            lot_size=pos.lot_size,
            strike=float(pos.strike),
            expiry=pos.expiry,
            option_type="PE",
        )
        qty = self.live_cfg.lots * pos.lot_size
        ce_exit = SELL if pos.ce_entry_side == BUY else BUY
        pe_exit = SELL if pos.pe_entry_side == BUY else BUY
        return [LegOrder(ce, ce_exit, qty), LegOrder(pe, pe_exit, qty)]

    def on_primary_bar(self, primary_ts: pd.Timestamp, df_1m: pd.DataFrame) -> None:
        self.df_1m = df_1m
        prepared = self._reprepare()
        if primary_ts not in prepared.index:
            logger.warning("Primary ts %s not in prepared frame; skipping bar", primary_ts)
            return

        bar = prepared.loc[primary_ts]
        bar_index = prepared.index.get_loc(primary_ts)
        if isinstance(bar_index, slice):
            bar_index = int(bar_index.start or 0)

        skip_entry = self.flat_until_next_bar
        if skip_entry:
            self.flat_until_next_bar = False

        state_before = snapshot_state(self.state_manager)
        result = step_bar(
            bar=bar,
            bar_index=int(bar_index),
            df=prepared,
            state_manager=self.state_manager,
            signal_engine=self.signal_engine,
            config=self.bt_config,
            skip_entry=skip_entry,
            skip_bar_sl_tp=self.live_cfg.intrabar_exit,
        )
        log_primary_bar_decision(
            primary_ts,
            bar,
            state_before=state_before,
            result=result,
            skip_entry=skip_entry,
            bt_config=self.bt_config,
            signal_engine=self.signal_engine,
        )

        if result.exit_signal:
            self._handle_exit(result.exit_signal, bar)

        if result.entry_signal:
            self._handle_entry(result.entry_signal)

        self._persist()

    def on_index_tick(self, price: float, ts: Optional[datetime] = None) -> None:
        self._last_index_price = float(price)
        if not self.live_cfg.intrabar_exit:
            return
        state = self.state_manager.state
        if state.position_size == 0 or self.live_position is None:
            return

        is_long = state.position_size > 0
        hit_sl = (is_long and price <= state.stop_loss) or (not is_long and price >= state.stop_loss)
        hit_tp = (is_long and price >= state.take_profit) or (not is_long and price <= state.take_profit)
        if not hit_sl and not hit_tp:
            return

        exit_type = ExitType.STOP_LOSS if hit_sl else ExitType.TAKE_PROFIT
        ts_pd = pd.Timestamp(ts or datetime.now())
        pnl = (price - state.entry_price) if is_long else (state.entry_price - price)
        exit_signal = ExitSignal(exit_type, ts_pd, price, state.entry_price, pnl)
        logger.info("Intrabar %s @ index %.2f (sl=%.2f tp=%.2f)", exit_type.value, price, state.stop_loss, state.take_profit)
        self._handle_exit(exit_signal, None, index_exit_price=price)
        self.flat_until_next_bar = True

    def _handle_entry(self, signal: Signal) -> None:
        if self.state_manager.state.position_size != 0:
            return
        if not self.allow_entries:
            logger.warning("Entry skipped: %s", self.entries_disabled_reason or "entries disabled")
            return

        is_long = signal.signal_type == SignalType.BUY
        direction = "long" if is_long else "short"
        signal_index_price = float(signal.price)
        index_price = self._last_index_price if self._last_index_price is not None else signal_index_price
        if self._last_index_price is not None and abs(index_price - signal_index_price) > 0.01:
            logger.info(
                "Entry index from live tick %.2f (signal close %.2f)",
                index_price,
                signal_index_price,
            )

        if self.bt_config.slippage_points:
            if is_long:
                index_price += self.bt_config.slippage_points
            else:
                index_price -= self.bt_config.slippage_points

        expiry = self.broker.nearest_expiry()
        legs = self.broker.resolve_atm_legs(index_price, expiry)
        strike = self.broker.atm_strike(index_price)
        if int(legs["ce"].strike) != strike or int(legs["pe"].strike) != strike:
            logger.error(
                "Entry aborted: resolved leg strike mismatch expected=%s CE=%s PE=%s",
                strike,
                legs["ce"].strike,
                legs["pe"].strike,
            )
            return
        logger.info(
            "Entry audit %s: index=%.2f strike=%d distance=%.2f expiry=%s lot_size=%d lots=%d",
            direction,
            index_price,
            strike,
            abs(index_price - strike),
            expiry,
            legs["ce"].lot_size,
            self.live_cfg.lots,
        )
        orders = self._entry_orders(direction, legs)

        try:
            refs = place_legs_safe(
                self.broker,
                orders,
                tag=f"entry_{direction}",
                check_margin=True,
                order_watcher=self.order_fill_watcher,
                fill_timeout_sec=self.live_cfg.order_fill_timeout_sec,
            )
        except Exception as exc:
            logger.error("Entry failed: %s", exc)
            return

        if not legs_filled(refs) or any(
            str(r.status or "").lower() == "rejected" for r in refs if r.status
        ):
            logger.error("Entry leg rejected or unfilled; attempting to flatten any filled leg")
            for order, ref in zip(orders, refs):
                if ref.avg_price > 0 and not ref.is_paper:
                    from live.safety import flatten_orphan_leg

                    flatten_orphan_leg(self.broker, order.leg, order.side, order.quantity)
            return

        stop_loss, take_profit = self.signal_engine.calculate_exit_levels(index_price, is_long=is_long)
        signal.price = index_price
        self.state_manager.on_entry(signal, stop_loss, take_profit)

        ce_ref, pe_ref = refs[0], refs[1]
        ce_side = orders[0].side
        pe_side = orders[1].side
        self.live_position = LivePosition(
            direction=direction,
            index_entry=index_price,
            index_sl=stop_loss,
            index_tp=take_profit,
            expiry=expiry,
            strike=strike,
            ce_symbol=legs["ce"].trading_symbol,
            pe_symbol=legs["pe"].trading_symbol,
            ce_token=legs["ce"].token,
            pe_token=legs["pe"].token,
            lot_size=legs["ce"].lot_size,
            ce_entry_side=ce_side,
            pe_entry_side=pe_side,
            ce_entry_price=ce_ref.avg_price,
            pe_entry_price=pe_ref.avg_price,
            entry_time=pd.Timestamp(signal.timestamp).isoformat(),
            entry_trigger=signal.trigger,
            entry_orders=[{"order_id": r.order_id, "symbol": r.trading_symbol, "side": r.side} for r in refs],
        )
        logger.info(
            "ENTER %s index=%.2f CE=%s@%.2f PE=%s@%.2f trigger=%s",
            direction,
            index_price,
            legs["ce"].trading_symbol,
            ce_ref.avg_price,
            legs["pe"].trading_symbol,
            pe_ref.avg_price,
            signal.trigger,
        )
        logger.info("OPEN %s risk: index=%.2f SL=%.2f TP=%.2f strike=%d expiry=%s",
                    direction, index_price, stop_loss, take_profit, strike, expiry)

    def _handle_exit(
        self,
        exit_signal: ExitSignal,
        bar: Optional[pd.Series],
        *,
        index_exit_price: Optional[float] = None,
    ) -> None:
        if self.state_manager.state.position_size == 0 or self.live_position is None:
            return

        pos = self.live_position
        exit_orders = self._exit_orders(pos)
        try:
            exit_refs = square_off_safe(
                self.broker,
                exit_orders,
                tag=f"exit_{exit_signal.exit_type.value}",
                order_watcher=self.order_fill_watcher,
                fill_timeout_sec=self.live_cfg.order_fill_timeout_sec,
            )
        except Exception as exc:
            logger.error("Square-off failed: %s", exc)
            return

        if bar is not None and exit_signal.exit_type in {ExitType.ST_FLIP, ExitType.FORCED_CLOSE}:
            is_long = self.state_manager.state.position_size > 0
            if is_long:
                exit_signal.exit_price = float(bar["close"]) - self.bt_config.slippage_points
            else:
                exit_signal.exit_price = float(bar["close"]) + self.bt_config.slippage_points

        if index_exit_price is not None:
            exit_signal.exit_price = index_exit_price

        trade = self.state_manager.on_exit(exit_signal)
        if trade:
            ce_exit = exit_refs[0].avg_price if exit_refs else 0.0
            pe_exit = exit_refs[1].avg_price if len(exit_refs) > 1 else 0.0
            self._log_trade(trade, pos, ce_exit, pe_exit, exit_signal.exit_type.value)

        self.live_position = None
        self._persist()
        logger.info("EXIT %s @ index %.2f (%s)", pos.direction, exit_signal.exit_price, exit_signal.exit_type.value)

    def flatten_on_shutdown(self) -> None:
        if not self.live_cfg.flatten_on_shutdown or self.live_position is None:
            return
        if self.state_manager.state.position_size == 0:
            return
        logger.warning("Shutdown flatten requested")
        state = self.state_manager.state
        is_long = state.position_size > 0
        price = self.broker.get_index_ltp() or state.entry_price
        pnl = (price - state.entry_price) if is_long else (state.entry_price - price)
        exit_signal = ExitSignal(ExitType.FORCED_CLOSE, pd.Timestamp.now(), price, state.entry_price, pnl)
        self._handle_exit(exit_signal, None, index_exit_price=price)

    def _log_trade(
        self,
        trade: Trade,
        pos: LivePosition,
        ce_exit: float,
        pe_exit: float,
        exit_type: str,
    ) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        index_pnl = (trade.pnl_points or 0.0)
        # Synthetic P&L approximation from leg prices (direction-dependent).
        if pos.direction == "long":
            opt_pnl = (ce_exit - pos.ce_entry_price) - (pe_exit - pos.pe_entry_price)
        else:
            opt_pnl = (pos.ce_entry_price - ce_exit) + (pe_exit - pos.pe_entry_price)
        opt_pnl *= self.live_cfg.lots * pos.lot_size

        row = {
            "trade_id": trade.trade_id,
            "direction": trade.direction,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "signal_type": trade.entry_trigger,
            "exit_type": exit_type,
            "pnl_points": trade.pnl_points,
            "pnl_value": trade.pnl_value,
            "contracts": trade.contracts,
            "ema_at_entry": trade.ema_at_entry,
            "ce_symbol": pos.ce_symbol,
            "pe_symbol": pos.pe_symbol,
            "ce_entry": pos.ce_entry_price,
            "pe_entry": pos.pe_entry_price,
            "ce_exit": ce_exit,
            "pe_exit": pe_exit,
            "net_option_pnl": opt_pnl,
            "index_pnl_points": index_pnl,
            "mode": "live" if self.live_cfg.is_live else "paper",
        }
        write_header = not self._trade_log_path.exists()
        with self._trade_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Logged trade #%s to %s", trade.trade_id, self._trade_log_path)
