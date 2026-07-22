"""Persist and restore live bot state across restarts."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from strategy import StrategyState

logger = logging.getLogger("live.persistence")

STATE_DIR = Path(__file__).resolve().parent / "state"
POSITION_FILE = STATE_DIR / "position.json"


def _ts_to_str(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    return pd.Timestamp(ts).isoformat()


def _str_to_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    return pd.Timestamp(value)


@dataclass
class LiveLegFill:
    trading_symbol: str
    token: str
    side: str
    quantity: int
    avg_price: float
    order_id: str


@dataclass
class LivePosition:
    direction: str              # long | short
    index_entry: float
    index_sl: float
    index_tp: float
    expiry: str
    strike: int
    ce_symbol: str
    pe_symbol: str
    ce_token: str
    pe_token: str
    lot_size: int
    ce_entry_side: str
    pe_entry_side: str
    ce_entry_price: float
    pe_entry_price: float
    entry_time: str
    entry_trigger: str
    entry_orders: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LivePosition":
        return cls(**data)


@dataclass
class PendingFinalExit:
    """Exit intent captured on the final primary bar, executed next session."""

    exit_type: str
    direction: str
    signal_time: str
    signal_price: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingFinalExit":
        return cls(**data)


@dataclass
class BrokerReconcileResult:
    """Summary of saved-position vs broker-position reconciliation."""

    open_symbols: set[str]
    expected_symbols: set[str]
    missing_symbols: set[str]
    extra_symbols: set[str]
    manual_exit_detected: bool = False
    partial_mismatch: bool = False


def strategy_state_to_dict(state: StrategyState) -> Dict[str, Any]:
    return {
        "position_size": state.position_size,
        "entry_price": state.entry_price,
        "entry_time": _ts_to_str(state.entry_time),
        "stop_loss": state.stop_loss,
        "take_profit": state.take_profit,
        "traded_in_bull_trend": state.traded_in_bull_trend,
        "traded_in_bear_trend": state.traded_in_bear_trend,
        "prev_st_direction": state.prev_st_direction,
        "pending_long_ema_wait": state.pending_long_ema_wait,
        "pending_short_ema_wait": state.pending_short_ema_wait,
        "pending_first_hour_long": state.pending_first_hour_long,
        "pending_first_hour_short": state.pending_first_hour_short,
        "pending_first_hour_trigger_long": state.pending_first_hour_trigger_long,
        "pending_first_hour_trigger_short": state.pending_first_hour_trigger_short,
        "pending_first_hour_deferred_long": state.pending_first_hour_deferred_long,
        "pending_first_hour_deferred_short": state.pending_first_hour_deferred_short,
        "pending_adx_long": state.pending_adx_long,
        "pending_adx_short": state.pending_adx_short,
        "adx_wait_bars_left_long": state.adx_wait_bars_left_long,
        "adx_wait_bars_left_short": state.adx_wait_bars_left_short,
        "adx_wait_trigger_long": state.adx_wait_trigger_long,
        "adx_wait_trigger_short": state.adx_wait_trigger_short,
        "deferred_ema_cross_long": state.deferred_ema_cross_long,
        "deferred_ema_cross_short": state.deferred_ema_cross_short,
        "pending_volume_long": state.pending_volume_long,
        "pending_volume_short": state.pending_volume_short,
        "volume_wait_bars_left_long": state.volume_wait_bars_left_long,
        "volume_wait_bars_left_short": state.volume_wait_bars_left_short,
        "volume_wait_trigger_long": state.volume_wait_trigger_long,
        "volume_wait_trigger_short": state.volume_wait_trigger_short,
        "volume_wait_kind_long": state.volume_wait_kind_long,
        "volume_wait_kind_short": state.volume_wait_kind_short,
        "trade_count": state.trade_count,
    }


def strategy_state_from_dict(data: Dict[str, Any]) -> StrategyState:
    return StrategyState(
        position_size=int(data.get("position_size", 0)),
        entry_price=float(data.get("entry_price", 0.0)),
        entry_time=_str_to_ts(data.get("entry_time")),
        stop_loss=float(data.get("stop_loss", 0.0)),
        take_profit=float(data.get("take_profit", 0.0)),
        traded_in_bull_trend=bool(data.get("traded_in_bull_trend", False)),
        traded_in_bear_trend=bool(data.get("traded_in_bear_trend", False)),
        prev_st_direction=int(data.get("prev_st_direction", 0)),
        pending_long_ema_wait=bool(data.get("pending_long_ema_wait", False)),
        pending_short_ema_wait=bool(data.get("pending_short_ema_wait", False)),
        pending_first_hour_long=bool(data.get("pending_first_hour_long", False)),
        pending_first_hour_short=bool(data.get("pending_first_hour_short", False)),
        pending_first_hour_trigger_long=str(data.get("pending_first_hour_trigger_long", "")),
        pending_first_hour_trigger_short=str(data.get("pending_first_hour_trigger_short", "")),
        pending_first_hour_deferred_long=data.get("pending_first_hour_deferred_long"),
        pending_first_hour_deferred_short=data.get("pending_first_hour_deferred_short"),
        pending_adx_long=bool(data.get("pending_adx_long", False)),
        pending_adx_short=bool(data.get("pending_adx_short", False)),
        adx_wait_bars_left_long=int(data.get("adx_wait_bars_left_long", 0)),
        adx_wait_bars_left_short=int(data.get("adx_wait_bars_left_short", 0)),
        adx_wait_trigger_long=str(data.get("adx_wait_trigger_long", "")),
        adx_wait_trigger_short=str(data.get("adx_wait_trigger_short", "")),
        deferred_ema_cross_long=data.get("deferred_ema_cross_long"),
        deferred_ema_cross_short=data.get("deferred_ema_cross_short"),
        pending_volume_long=bool(data.get("pending_volume_long", False)),
        pending_volume_short=bool(data.get("pending_volume_short", False)),
        volume_wait_bars_left_long=int(data.get("volume_wait_bars_left_long", 0)),
        volume_wait_bars_left_short=int(data.get("volume_wait_bars_left_short", 0)),
        volume_wait_trigger_long=str(data.get("volume_wait_trigger_long", "")),
        volume_wait_trigger_short=str(data.get("volume_wait_trigger_short", "")),
        volume_wait_kind_long=str(data.get("volume_wait_kind_long", "")),
        volume_wait_kind_short=str(data.get("volume_wait_kind_short", "")),
        trade_count=int(data.get("trade_count", 0)),
    )


def save_state(
    strategy_state: StrategyState,
    live_position: Optional[LivePosition],
    *,
    flat_until_next_bar: bool = False,
    last_session_date: Optional[str] = None,
    pending_final_exit: Optional[PendingFinalExit] = None,
) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy_state": strategy_state_to_dict(strategy_state),
        "live_position": live_position.to_dict() if live_position else None,
        "flat_until_next_bar": flat_until_next_bar,
        "last_session_date": last_session_date,
        "pending_final_exit": pending_final_exit.to_dict() if pending_final_exit else None,
    }
    POSITION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug("Saved live state to %s", POSITION_FILE)


def load_state() -> tuple[
    Optional[StrategyState],
    Optional[LivePosition],
    bool,
    Optional[str],
    Optional[PendingFinalExit],
]:
    if not POSITION_FILE.exists():
        return None, None, False, None, None
    try:
        payload = json.loads(POSITION_FILE.read_text(encoding="utf-8"))
        ss = strategy_state_from_dict(payload.get("strategy_state", {}))
        lp = None
        if payload.get("live_position"):
            lp = LivePosition.from_dict(payload["live_position"])
        pending = None
        if payload.get("pending_final_exit"):
            pending = PendingFinalExit.from_dict(payload["pending_final_exit"])
        flat = bool(payload.get("flat_until_next_bar", False))
        last_session = payload.get("last_session_date")
        return ss, lp, flat, last_session if last_session else None, pending
    except Exception as exc:
        logger.warning("Could not load state file: %s", exc)
        return None, None, False, None, None


def clear_state() -> None:
    if POSITION_FILE.exists():
        POSITION_FILE.unlink()


def _broker_open_symbols(positions_resp: Any) -> set[str]:
    """Extract trading symbols with non-zero net qty from a positions() response."""
    symbols: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            sym = node.get("trdSym") or node.get("trading_symbol") or node.get("pTrdSymbol")
            qty = node.get("netQty") or node.get("net_qty") or node.get("flQty") or node.get("quantity")
            try:
                q = int(float(qty)) if qty not in (None, "", "0") else 0
            except (TypeError, ValueError):
                q = 0
            if sym and q != 0:
                symbols.add(str(sym))
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(positions_resp)
    return symbols


def reconcile_with_broker(
    broker: Any,
    live_position: Optional[LivePosition],
) -> Optional[BrokerReconcileResult]:
    """Compare persisted live position with broker open legs.

    A full manual/broker exit is detected only when both expected legs from the
    saved live position are absent at the broker. Partial mismatches are left for
    manual review because auto-clearing state could hide an orphan option leg.
    """
    try:
        resp = broker.positions()
    except Exception as exc:
        logger.warning("Could not fetch broker positions for reconcile: %s", exc)
        return None

    open_syms = _broker_open_symbols(resp)
    if live_position is None:
        if open_syms:
            logger.warning(
                "Broker has open positions %s but no saved live position; manual review required",
                sorted(open_syms),
            )
        return BrokerReconcileResult(
            open_symbols=open_syms,
            expected_symbols=set(),
            missing_symbols=set(),
            extra_symbols=open_syms,
        )

    expected = {live_position.ce_symbol, live_position.pe_symbol}
    missing = expected - open_syms
    extra = open_syms - expected
    manual_exit_detected = missing == expected
    partial_mismatch = bool(missing) and not manual_exit_detected
    if missing or extra:
        logger.warning(
            "Position mismatch: saved=%s broker_open=%s missing=%s extra=%s",
            sorted(expected),
            sorted(open_syms),
            sorted(missing),
            sorted(extra),
        )
    else:
        logger.info("Broker positions reconcile OK for %s", sorted(expected))
    return BrokerReconcileResult(
        open_symbols=open_syms,
        expected_symbols=expected,
        missing_symbols=missing,
        extra_symbols=extra,
        manual_exit_detected=manual_exit_detected,
        partial_mismatch=partial_mismatch,
    )
