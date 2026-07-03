"""Live trading safety helpers: margin checks, order polling, rate limiting."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

from live.neo_client import BUY, Leg, LegOrder, NeoBroker, OrderRef, SELL
from live.neo_place_order import unique_order_tag

logger = logging.getLogger("live.safety")

TERMINAL_STATUSES = {"complete", "traded", "rejected", "cancelled", "canceled"}


class RateLimiter:
    """Simple token bucket for Neo's ~10 req/s limit."""

    def __init__(self, max_per_second: float = 8.0):
        self.min_interval = 1.0 / max_per_second
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class OrderTracker:
    """Dedupe order-feed events by order id."""

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def is_new(self, order_id: str) -> bool:
        if not order_id or order_id in self._seen:
            return False
        self._seen.add(order_id)
        return True


def check_margin_for_legs(broker: NeoBroker, orders: List[LegOrder]) -> bool:
    """Return True if margin_required succeeds for all legs (best-effort)."""
    limiter = RateLimiter()
    for order in orders:
        limiter.wait()
        try:
            resp = broker.margin_required(order.leg, order.side, order.quantity)
            logger.info("Margin check %s %s: %s", order.side, order.leg.trading_symbol, resp)
        except Exception as exc:
            logger.warning("Margin check failed for %s: %s", order.leg.trading_symbol, exc)
            return False
    return True


def _order_status(resp: Any) -> str:
    if isinstance(resp, dict):
        for key in ("ordSt", "status", "stat"):
            if resp.get(key):
                return str(resp[key]).lower()
        data = resp.get("data")
        if isinstance(data, dict) and data.get("ordSt"):
            return str(data["ordSt"]).lower()
    return ""


def wait_for_order_terminal(
    broker: NeoBroker,
    order_ref: OrderRef,
    *,
    timeout_sec: float = 30.0,
    poll_interval: float = 1.0,
) -> OrderRef:
    """Poll order history until terminal state or timeout."""
    if order_ref.is_paper or not order_ref.order_id:
        return order_ref

    deadline = time.monotonic() + timeout_sec
    limiter = RateLimiter()
    while time.monotonic() < deadline:
        limiter.wait()
        try:
            hist = broker.client.order_history(order_id=order_ref.order_id)
            status = _order_status(hist)
            if status in TERMINAL_STATUSES:
                order_ref.status = status
                return order_ref
        except Exception as exc:
            logger.warning("order_history poll failed for %s: %s", order_ref.order_id, exc)
        time.sleep(poll_interval)
    logger.warning("Order %s did not reach terminal state within %.0fs", order_ref.order_id, timeout_sec)
    return order_ref


def place_legs_safe(
    broker: NeoBroker,
    orders: List[LegOrder],
    *,
    tag: Optional[str] = None,
    check_margin: bool = True,
) -> List[OrderRef]:
    """Place legs with optional margin check and fill polling."""
    if check_margin and not broker.cfg.is_live:
        check_margin = False  # paper skips margin API
    if check_margin and not check_margin_for_legs(broker, orders):
        raise RuntimeError("Pre-trade margin check failed; aborting entry")

    refs = broker.place_legs(orders, tag=tag)
    finalized: List[OrderRef] = []
    for ref in refs:
        finalized.append(wait_for_order_terminal(broker, ref))
    return finalized


def flatten_orphan_leg(broker: NeoBroker, leg: Leg, entry_side: str, quantity: int) -> OrderRef:
    """Emergency: reverse a single filled leg to avoid naked exposure."""
    reverse = SELL if entry_side == BUY else BUY
    logger.error("Flattening orphan leg %s %s x%d", reverse, leg.trading_symbol, quantity)
    return broker.place_leg(leg, reverse, quantity, tag=unique_order_tag("orphan", leg))


def square_off_safe(broker: NeoBroker, orders: List[LegOrder], tag: str = "square_off") -> List[OrderRef]:
    return place_legs_safe(broker, orders, tag=tag, check_margin=False)
