"""Live-bot configuration: loads the `live:` block from config/strategy.yaml and
Kotak Neo credentials from the environment (.env), and builds the same
`BacktestConfig` the backtest uses so the strategy logic is shared verbatim.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

# Reuse the backtest config builder so live and backtest read the exact same knobs.
from backtest import (  # type: ignore
    BacktestConfig,
    DEFAULT_CONFIG,
    build_backtest_config,
    load_config,
)

IST = "Asia/Kolkata"

BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_hhmm(value: str, default: time) -> time:
    try:
        hh, mm = str(value).split(":")
        return time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return default


@dataclass
class NeoCredentials:
    consumer_key: str          # NEO_API_KEY (dashboard token / consumer key)
    client_code: str           # NEO_CLIENT_CODE (ucc)
    mpin: str                  # NEO_MPIN
    mobile: str                # NEO_MOBILE (+91...)
    totp_secret: Optional[str] # NEO_TOTP_SECRET (optional base32 seed)

    @property
    def has_totp_secret(self) -> bool:
        return bool(self.totp_secret and self.totp_secret.strip())


@dataclass
class LiveConfig:
    mode: str                  # "paper" | "live"
    underlying_symbol: str
    index_name: str
    index_segment: str
    option_segment: str
    structure: str             # "synthetic" | "single"
    product: str
    order_type: str
    lots: int
    strike_step: int
    expiry_rule: str
    expiry_roll_days: int
    intrabar_exit: bool
    square_off_on_exit: bool
    flatten_on_shutdown: bool
    warmup_sessions: int
    warmup_max_bars: int
    log_retention_days: int
    bars_csv: str
    warmup_source: str
    kite_fallback_on_error: bool
    session_start: time
    session_end: time
    order_fill_timeout_sec: float
    fast_direct_orders: bool
    scrip_symbol_expiry_csv: str
    fast_direct_lot_size: int
    fast_direct_check_margin: bool

    @property
    def is_live(self) -> bool:
        return str(self.mode).strip().lower() == "live"


def load_live_config(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Return the parsed YAML root (so callers can also build the BacktestConfig)."""
    return load_config(config_path)


def build_live_config(config_root: Dict[str, Any]) -> LiveConfig:
    live = config_root.get("live") or {}
    return LiveConfig(
        mode=str(live.get("mode", "paper")),
        underlying_symbol=str(live.get("underlying_symbol", "BANKNIFTY")),
        index_name=str(live.get("index_name", "Nifty Bank")),
        index_segment=str(live.get("index_segment", "nse_cm")),
        option_segment=str(live.get("option_segment", "nse_fo")),
        structure=str(live.get("structure", "synthetic")),
        product=str(live.get("product", "NRML")),
        order_type=str(live.get("order_type", "MKT")),
        lots=int(live.get("lots", 1)),
        strike_step=int(live.get("strike_step", 100)),
        expiry_rule=str(live.get("expiry_rule", "monthly")),
        expiry_roll_days=int(live.get("expiry_roll_days", 1)),
        intrabar_exit=bool(live.get("intrabar_exit", True)),
        square_off_on_exit=bool(live.get("square_off_on_exit", True)),
        flatten_on_shutdown=bool(live.get("flatten_on_shutdown", False)),
        warmup_sessions=int(live.get("warmup_sessions", 35)),
        warmup_max_bars=int(live.get("warmup_max_bars", 12000)),
        log_retention_days=int(live.get("log_retention_days", 3)),
        bars_csv=str(live.get("bars_csv", "live/cache/banknifty_1min_neo.csv")),
        warmup_source=str(live.get("warmup_source", "kite_daily")),
        kite_fallback_on_error=bool(live.get("kite_fallback_on_error", True)),
        session_start=_parse_hhmm(live.get("session_start", "09:15"), time(9, 15)),
        session_end=_parse_hhmm(live.get("session_end", "15:30"), time(15, 30)),
        order_fill_timeout_sec=float(live.get("order_fill_timeout_sec", 8.0)),
        fast_direct_orders=bool(live.get("fast_direct_orders", False)),
        scrip_symbol_expiry_csv=str(
            live.get(
                "scrip_symbol_expiry_csv",
                "live/cache/scrip_master/banknifty_symbol_expiry.csv",
            )
        ),
        fast_direct_lot_size=int(live.get("fast_direct_lot_size", 30)),
        fast_direct_check_margin=bool(live.get("fast_direct_check_margin", False)),
    )


def load_neo_credentials() -> NeoCredentials:
    load_dotenv()
    return NeoCredentials(
        consumer_key=os.environ.get("NEO_API_KEY", "").strip(),
        client_code=os.environ.get("NEO_CLIENT_CODE", "").strip(),
        mpin=os.environ.get("NEO_MPIN", "").strip(),
        mobile=os.environ.get("NEO_MOBILE", "").strip(),
        totp_secret=os.environ.get("NEO_TOTP_SECRET", "").strip() or None,
    )


def build_backtest_config_for_live(config_root: Dict[str, Any]) -> BacktestConfig:
    """Build the BacktestConfig (timeframes, EMA, SL/TP, ADX) used by the strategy.

    Date bounds are irrelevant for live signal generation, so we hand the builder a
    minimal IST-aware index purely to satisfy its timezone handling.
    """
    minimal_index = pd.DatetimeIndex(
        pd.to_datetime(["2020-01-01 09:15", "2020-01-01 15:30"])
    ).tz_localize(IST)
    return replace(
        build_backtest_config(
            config_root,
            minimal_index,
            start=None,
            end=None,
            ema_timeframe_override=None,
            primary_timeframe_override=None,
            contracts_override=None,
        ),
        # Live bot holds at most one synthetic position (long OR short), never both.
        independent_books=False,
    )
