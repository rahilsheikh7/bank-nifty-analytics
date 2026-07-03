"""Entry point for the Bank Nifty live options bot (Kotak Neo).

Usage:
    python -m live.run_live

Warmup: daily Kite fetch merged with Neo live bars (config: live.warmup_source).

Prerequisites:
    1. pip install -r requirements.txt -r requirements-live.txt
    2. Set NEO_* and KITE_* credentials in .env
    3. Run generate_token.py each morning for KITE_ACCESS_TOKEN
    4. Set live.mode: paper in config/strategy.yaml (default)
"""
from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Any, Optional

import pandas as pd

from live.bar_cache import BarCache
from live.logging_setup import LOG_FILE, setup_logging
from live.config import (
    build_backtest_config_for_live,
    build_live_config,
    load_live_config,
    load_neo_credentials,
)
from live.engine import LiveTrader
from live.feed import MinuteAggregator, PrimaryBarClock, extract_index_ticks
from live.neo_client import make_broker
from live.neo_session import ensure_trade_session, is_token_stale
from live.persistence import reconcile_with_broker
from live.safety import OrderTracker
from live.warmup import build_warm_1min

IST = "Asia/Kolkata"
logger = logging.getLogger("live.runner")


def _in_session(ts: pd.Timestamp, live_cfg) -> bool:
    t = ts.tz_convert(IST).time() if ts.tzinfo else ts.time()
    return live_cfg.session_start <= t <= live_cfg.session_end


class LiveRunner:
    def __init__(self) -> None:
        self.running = True
        self.trader: Optional[LiveTrader] = None
        self.aggregator: Optional[MinuteAggregator] = None
        self.clock: Optional[PrimaryBarClock] = None
        self.bar_cache: Optional[BarCache] = None
        self.broker = None
        self.live_cfg = None
        self.bt_config = None
        self.creds = None
        self.order_tracker = OrderTracker()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._feed_lock = threading.Lock()
        self._reconnect_scheduled = False
        self._awaiting_reconnect_tick = False
        self._needs_relogin = False
        self._logged_first_tick = False
        self._logged_ws_sample = False
        self._tick_count = 0
        self._last_tick_price: Optional[float] = None
        self._last_tick_time: Optional[pd.Timestamp] = None
        self._heartbeat_counter = 0
        self._started_at = pd.Timestamp.now(tz=IST)
        self._last_ws_problem_key = ""
        self._last_ws_problem_at = 0.0
        self._ws_problem_suppressed = 0
        self._stale_feed_active = False
        self._last_stale_log_at = 0.0
        self._last_rest_fallback_at = 0.0
        self._warmup_ready = True

    def _log_bar_range(self, df_1m: pd.DataFrame, label: str) -> None:
        if df_1m.empty:
            logger.info("%s: no bars", label)
            return
        start = df_1m.index.min()
        end = df_1m.index.max()
        logger.info("%s: %d bars from %s to %s", label, len(df_1m), start, end)

    def _log_session_status(self) -> None:
        now = pd.Timestamp.now(tz=IST)
        if _in_session(now, self.live_cfg):
            logger.info(
                "Market session OPEN (now %s IST) — ticks and bars will be processed",
                now.strftime("%H:%M:%S"),
            )
        else:
            logger.info(
                "Market session CLOSED (now %s IST) — waiting for %s–%s IST",
                now.strftime("%H:%M:%S"),
                self.live_cfg.session_start,
                self.live_cfg.session_end,
            )

    def _on_minute_bar(self, _bar: pd.Series) -> None:
        if self.aggregator is None or self.clock is None or self.trader is None:
            return
        self.trader.update_df(self.aggregator.frame)
        self.clock.check(self.aggregator.frame)

    def _on_primary_bar(self, primary_ts: pd.Timestamp, df_1m: pd.DataFrame) -> None:
        if self.trader is None or self.live_cfg is None:
            return
        if not _in_session(primary_ts, self.live_cfg):
            return
        logger.info("Primary bar %s", primary_ts)
        self.trader.on_primary_bar(primary_ts, df_1m)

    def _handle_order_feed(self, message: Any) -> None:
        data = message
        if isinstance(message, dict) and message.get("type") == "order_feed":
            data = message.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return
        if not isinstance(data, dict):
            return
        oid = str(data.get("nOrdNo", data.get("order_id", "")))
        if oid and self.order_tracker.is_new(oid):
            logger.info("Order update %s status=%s", oid, data.get("ordSt", data.get("status")))

    def _on_ws_message(self, message: Any) -> None:
        if not self.running:
            return
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                return

        if not self._logged_ws_sample and message is not None:
            self._logged_ws_sample = True
            preview = str(message)
            if len(preview) > 400:
                preview = preview[:400] + "..."
            logger.info("First WS message: %s", preview)

        if isinstance(message, dict):
            msg_type = message.get("type")
            if msg_type in ("order", "order_feed"):
                self._handle_order_feed(message)
                return

        if self.aggregator is None or self.trader is None or self.live_cfg is None:
            return

        for price, ts in extract_index_ticks(message):
            self._ingest_index_tick(price, ts, source="ws")

    def _ingest_index_tick(self, price: float, ts: Any, *, source: str = "ws") -> None:
        if not self.running:
            return
        if self.aggregator is None or self.trader is None or self.live_cfg is None:
            return
        now = pd.Timestamp(ts, tz=IST) if getattr(ts, "tzinfo", None) is None else pd.Timestamp(ts).tz_convert(IST)
        wall_now = pd.Timestamp.now(tz=IST)
        if source == "ws" and abs((wall_now - now).total_seconds()) > 300:
            logger.debug("Ignoring stale websocket timestamp %s; using local receive time", now)
            now = wall_now
            ts = wall_now.to_pydatetime()
        if not _in_session(now, self.live_cfg):
            return
        self._tick_count += 1
        self._last_tick_price = price
        self._last_tick_time = now
        if not self._logged_first_tick:
            self._logged_first_tick = True
            logger.info("First index tick (%s): price=%.2f at %s", source, price, now.strftime("%H:%M:%S"))
        if self._awaiting_reconnect_tick:
            self._awaiting_reconnect_tick = False
            self._stale_feed_active = False
            logger.info(
                "Market feed healthy: first tick after reconnect (%s) price=%.2f at %s",
                source, price, now.strftime("%H:%M:%S"),
            )
        elif self._tick_count % 100 == 0:
            logger.info("Index tick #%d: price=%.2f at %s", self._tick_count, price, now.strftime("%H:%M:%S"))
        self.aggregator.on_tick(price, ts)
        self.trader.on_index_tick(price, ts)

    def _log_heartbeat(self) -> None:
        now = pd.Timestamp.now(tz=IST)
        in_sess = _in_session(now, self.live_cfg) if self.live_cfg else False
        feed_age = self._feed_age_seconds(now)
        parts = [
            f"session={'OPEN' if in_sess else 'CLOSED'}",
            f"ticks={self._tick_count}",
            f"feed_age={feed_age:.0f}s",
        ]
        if self._last_tick_price is not None:
            parts.append(f"last={self._last_tick_price:.2f}")
        if self._last_tick_time is not None:
            parts.append(f"last_tick={self._last_tick_time.strftime('%H:%M:%S')}")
        if self.trader is not None:
            state = self.trader.state_manager.state
            parts.append(f"position={state.position_size}")
            if state.position_size != 0:
                parts.append(f"SL={state.stop_loss:.2f}")
                parts.append(f"TP={state.take_profit:.2f}")
        if self.aggregator is not None:
            parts.append(f"bars={len(self.aggregator.frame)}")
        if in_sess and self._tick_count == 0:
            parts.append("waiting for index ticks")
        logger.info("Heartbeat | %s", " | ".join(parts))

    def _maybe_refresh_daily_session(self) -> None:
        if not self.broker or not self.creds:
            return
        if not is_token_stale():
            return
        logger.info("Trade session expired (new IST day) — refreshing token.json and SDK login")
        try:
            self.broker.refresh_trade_session(force=True)
            self.broker.login()
        except Exception as exc:
            logger.error("Daily trade session refresh failed: %s", exc)

    def _feed_age_seconds(self, now: Optional[pd.Timestamp] = None) -> float:
        now = now or pd.Timestamp.now(tz=IST)
        last = self._last_tick_time or self._started_at
        return max(0.0, (now - last).total_seconds())

    def _has_open_position(self) -> bool:
        if self.trader is None:
            return False
        return self.trader.state_manager.state.position_size != 0 and self.trader.live_position is not None

    def _check_feed_health(self) -> None:
        if self.live_cfg is None or self.broker is None:
            return
        now = pd.Timestamp.now(tz=IST)
        if not _in_session(now, self.live_cfg):
            self._stale_feed_active = False
            return

        feed_age = self._feed_age_seconds(now)
        if feed_age <= 30:
            if self._stale_feed_active:
                logger.info("Market feed recovered (last tick %.0fs ago)", feed_age)
            self._stale_feed_active = False
            return

        monotonic_now = time.monotonic()
        if not self._stale_feed_active or monotonic_now - self._last_stale_log_at >= 60:
            logger.warning(
                "Market feed stale: no index tick for %.0fs (position_open=%s)",
                feed_age,
                self._has_open_position(),
            )
            self._last_stale_log_at = monotonic_now
        self._stale_feed_active = True

        if not self._has_open_position() or monotonic_now - self._last_rest_fallback_at < 5:
            return

        self._last_rest_fallback_at = monotonic_now
        try:
            ltp = self.broker.get_index_ltp()
        except Exception as exc:
            logger.warning("REST fallback index check failed while feed stale: %s", exc)
            return
        if ltp is None:
            logger.warning("REST fallback index check returned no LTP while feed stale")
            return
        logger.warning("REST fallback index check while feed stale: %.2f", ltp)
        if self.trader is not None:
            self.trader.on_index_tick(float(ltp), now.to_pydatetime())

    def _on_ws_close(self, *_args: Any) -> None:
        if not self.running:
            return
        self._log_ws_problem(logging.WARNING, "WebSocket closed")
        self._schedule_reconnect("closed")

    def _on_ws_error(self, error: Any) -> None:
        err = str(error).lower()
        self._log_ws_problem(logging.ERROR, f"WebSocket error: {error}")
        if "403" in err or "unauthorized" in err or "token" in err:
            self._needs_relogin = True
        self._schedule_reconnect("error")

    def _log_ws_problem(self, level: int, message: str) -> None:
        now = time.monotonic()
        if message == self._last_ws_problem_key and now - self._last_ws_problem_at < 30:
            self._ws_problem_suppressed += 1
            return
        if self._ws_problem_suppressed:
            logger.log(level, "Suppressed %d repeated WebSocket messages", self._ws_problem_suppressed)
            self._ws_problem_suppressed = 0
        logger.log(level, message)
        self._last_ws_problem_key = message
        self._last_ws_problem_at = now

    def _schedule_reconnect(self, reason: str) -> None:
        with self._feed_lock:
            if self._reconnect_scheduled or not self.running:
                return
            self._reconnect_scheduled = True
            delay = self._reconnect_delay
        logger.warning("Scheduling feed reconnect in %.1fs (%s)", delay, reason)
        threading.Thread(target=self._reconnect_feeds, args=(delay,), daemon=True).start()

    def _reconnect_feeds(self, delay: float) -> None:
        time.sleep(delay)
        if not self.running:
            return
        try:
            if self._needs_relogin and self.broker and self.creds and self.live_cfg:
                if is_token_stale():
                    self.broker.refresh_trade_session(force=True)
                logger.info("Re-authenticating Neo session...")
                self.broker.login()
                self._needs_relogin = False
            if self.broker:
                self.broker.set_callbacks(
                    on_message=self._on_ws_message,
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error,
                )
                self.broker.start_market_feed()
                self.broker.start_order_feed()
                self._awaiting_reconnect_tick = True
                self._reconnect_delay = 1.0
                logger.info("Feed resubscribe sent; waiting for next market tick")
        except Exception as exc:
            logger.warning("Reconnect attempt failed: %s", exc)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            with self._feed_lock:
                self._reconnect_scheduled = False
            self._schedule_reconnect("retry")
            return

        with self._feed_lock:
            self._reconnect_scheduled = False

    def _persist_bars(self) -> None:
        if self.aggregator is not None:
            self.aggregator.flush()
        if self.bar_cache is not None and self.aggregator is not None:
            self.bar_cache.merge_frame(self.aggregator.frame, persist=True)

    def _shutdown(self, *_args: Any) -> None:
        if not self.running:
            return
        logger.info("Shutting down...")
        self.running = False
        self._persist_bars()
        if self.trader:
            self.trader.flatten_on_shutdown()
        if self.broker:
            try:
                self.broker.logout()
            except Exception:
                pass

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._shutdown)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._shutdown)

        config_root = load_live_config()
        self.live_cfg = build_live_config(config_root)
        self.bt_config = build_backtest_config_for_live(config_root)
        self.creds = load_neo_credentials()

        logger.info("=== Bank Nifty live bot starting ===")
        logger.info("Log file: %s", LOG_FILE)
        logger.info(
            "Mode: %s | EMA TF: %s | Primary TF: %s | Warmup: %s | Session %s-%s",
            self.live_cfg.mode,
            self.bt_config.ema_timeframe,
            self.bt_config.primary_timeframe,
            self.live_cfg.warmup_source,
            self.live_cfg.session_start,
            self.live_cfg.session_end,
        )

        try:
            ensure_trade_session(self.creds)
        except RuntimeError as exc:
            logger.error("Neo trade session setup failed: %s", exc)
            raise

        df_1m, warmup_status = build_warm_1min(
            self.live_cfg.warmup_sessions,
            bars_csv=self.live_cfg.bars_csv,
            ema_length=self.bt_config.ema_length,
            ema_timeframe=self.bt_config.ema_timeframe,
            primary_timeframe=self.bt_config.primary_timeframe,
            warmup_source=self.live_cfg.warmup_source,
            kite_fallback_on_error=self.live_cfg.kite_fallback_on_error,
            session_start=self.live_cfg.session_start,
        )
        logger.info(
            "Warmup %s: %d sessions, %d bars, ready=%s — %s",
            warmup_status.source,
            warmup_status.session_count,
            warmup_status.bar_count,
            warmup_status.ready,
            warmup_status.message,
        )
        if not warmup_status.ready:
            logger.warning("Trading signals may be skipped until warmup is sufficient")
        self._warmup_ready = warmup_status.ready

        self._log_bar_range(df_1m, "Merged 1-min history")

        self.bar_cache = BarCache(
            self.live_cfg.bars_csv,
            max_sessions=self.live_cfg.warmup_sessions,
        )
        self.bar_cache.df = df_1m.copy()
        logger.info("Bar cache: %s", self.bar_cache.path)

        logger.info("Connecting to Kotak Neo (client=%s)...", self.creds.client_code)
        self.broker = make_broker(self.creds, self.live_cfg)
        self.broker.login()
        try:
            limits = self.broker.limits()
            net = limits.get("Net", limits.get("net", "?")) if isinstance(limits, dict) else "?"
            logger.info("Neo limits OK — available margin (Net): %s", net)
        except Exception as exc:
            logger.warning("Could not fetch limits: %s", exc)

        logger.info("Initializing strategy engine (lots=%d)...", self.live_cfg.lots)
        self.trader = LiveTrader(self.broker, self.live_cfg, self.bt_config, config_root, df_1m)
        if not self._warmup_ready:
            self.trader.allow_entries = False
            self.trader.entries_disabled_reason = f"warmup not ready ({warmup_status.message})"
            logger.warning("Entries disabled until restart with sufficient warmup")
        reconcile_with_broker(self.broker, self.trader.live_position)

        self.aggregator = MinuteAggregator(
            self.live_cfg,
            self._on_minute_bar,
            initial_df=df_1m,
            bar_cache=self.bar_cache,
        )
        self.clock = PrimaryBarClock(
            self.bt_config.primary_timeframe,
            self.bt_config.hourly_bar_end_minute,
            self._on_primary_bar,
        )
        self.clock.seed(df_1m)

        self.broker.set_callbacks(
            on_message=self._on_ws_message,
            on_close=self._on_ws_close,
            on_error=self._on_ws_error,
        )
        logger.info("Starting WebSocket feeds (market + order)...")
        self.broker.start_market_feed()
        self.broker.start_order_feed()
        logger.info("WebSocket feeds connected")
        try:
            ltp = self.broker.get_index_ltp()
            if ltp is not None:
                logger.info("Index LTP (REST snapshot): %.2f", ltp)
            else:
                logger.warning("Index LTP (REST) unavailable — waiting for websocket ticks")
        except Exception as exc:
            logger.warning("Index LTP (REST) check failed: %s", exc)

        self._log_session_status()
        logger.info(
            "=== Startup complete — bot running in %s mode (Ctrl+C to stop) ===",
            self.live_cfg.mode,
        )
        while self.running:
            time.sleep(1)
            self._heartbeat_counter += 1
            self._check_feed_health()
            if self._heartbeat_counter % 60 == 0:
                self._maybe_refresh_daily_session()
                self._log_heartbeat()

        self._persist_bars()


def main() -> None:
    setup_logging()
    try:
        LiveRunner().run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
