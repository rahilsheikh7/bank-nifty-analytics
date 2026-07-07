"""Live trading safety helpers: margin checks, order fill tracking, rate limiting."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from live.neo_client import BUY, Leg, LegOrder, NeoBroker, OrderRef, SELL
from live.neo_place_order import unique_order_tag

logger = logging.getLogger("live.safety")

TERMINAL_STATUSES = {"complete", "traded", "rejected", "cancelled", "canceled"}
REJECT_CANCEL_STATUSES = {"rejected", "cancelled", "canceled"}


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_status(status: Any) -> str:
    return str(status or "").strip().lower()


def _extract_order_id(data: Dict[str, Any]) -> str:
    for key in ("nOrdNo", "orderId", "order_id", "ordNo", "id"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _extract_status(data: Dict[str, Any]) -> str:
    for key in ("ordSt", "status", "stat", "orderStatus"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _extract_avg_price(data: Dict[str, Any]) -> float:
    for key in ("avgPrc", "avgPrice", "avg_prc", "trdPrc", "fillPrice", "prc", "avgP"):
        price = _to_float(data.get(key), default=0.0)
        if price > 0:
            return price
    return 0.0


def _order_status(resp: Any) -> str:
    if isinstance(resp, dict):
        for key in ("ordSt", "status", "stat"):
            if resp.get(key):
                return str(resp[key]).lower()
        data = resp.get("data")
        if isinstance(data, dict) and data.get("ordSt"):
            return str(data["ordSt"]).lower()
    return ""


def _extract_avg_price_from_history(resp: Any) -> float:
    if not isinstance(resp, dict):
        return 0.0
    for key in ("avgPrc", "avgPrice", "trdPrc", "prc"):
        price = _to_float(resp.get(key), default=0.0)
        if price > 0:
            return price
    data = resp.get("data")
    if isinstance(data, dict):
        return _extract_avg_price(data)
    if isinstance(data, list) and data:
        last = data[-1]
        if isinstance(last, dict):
            return _extract_avg_price(last)
    return 0.0


def _is_reject_or_cancel(status: str) -> bool:
    normalized = _normalize_status(status)
    if normalized in REJECT_CANCEL_STATUSES:
        return True
    return "reject" in normalized or "cancel" in normalized


def _is_fill_complete(ref: OrderRef) -> bool:
    if _is_reject_or_cancel(ref.status):
        return True
    return ref.avg_price > 0


def legs_filled(refs: List[OrderRef]) -> bool:
    if len(refs) < 2:
        return False
    for ref in refs:
        if _is_reject_or_cancel(ref.status):
            return False
        if not ref.is_paper and ref.avg_price <= 0:
            return False
    return True


class OrderFillWatcher:
    """Track Kotak Neo order-feed updates and unblock fill waits."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orders: Dict[str, OrderRef] = {}
        self._events: Dict[str, threading.Event] = {}
        self._buffer: Dict[str, Dict[str, Any]] = {}

    def register(self, ref: OrderRef) -> None:
        if ref.is_paper or not ref.order_id:
            return
        with self._lock:
            self._orders[ref.order_id] = ref
            self._events[ref.order_id] = threading.Event()
            buffered = self._buffer.pop(ref.order_id, None)
        if buffered:
            self.ingest(buffered)

    def ingest(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        oid = _extract_order_id(data)
        if not oid:
            return False

        with self._lock:
            ref = self._orders.get(oid)
            if ref is None:
                self._buffer[oid] = dict(data)
                if len(self._buffer) > 128:
                    self._buffer.pop(next(iter(self._buffer)))
                return False
            self._apply_update_locked(oid, ref, data)
            return True

    def _apply_update_locked(self, oid: str, ref: OrderRef, data: Dict[str, Any]) -> None:
        status = _extract_status(data)
        avg_price = _extract_avg_price(data)
        if status:
            ref.status = status
        if avg_price > 0:
            ref.avg_price = avg_price
        if _is_fill_complete(ref):
            event = self._events.get(oid)
            if event is not None:
                event.set()

    def wait_for_orders(
        self,
        refs: List[OrderRef],
        *,
        timeout_sec: float,
        broker: Optional[NeoBroker] = None,
    ) -> List[OrderRef]:
        live_refs = [ref for ref in refs if ref.order_id and not ref.is_paper]
        for ref in live_refs:
            self.register(ref)
        if not live_refs:
            return refs

        deadline = time.monotonic() + timeout_sec

        def _wait_one(ref: OrderRef) -> None:
            event = self._events.get(ref.order_id)
            if event is None:
                return
            remaining = deadline - time.monotonic()
            if remaining > 0:
                event.wait(timeout=remaining)

        with ThreadPoolExecutor(max_workers=max(1, len(live_refs))) as pool:
            futures = [pool.submit(_wait_one, ref) for ref in live_refs]
            for future in futures:
                future.result()

        pending = [ref for ref in live_refs if not _is_fill_complete(ref)]
        if pending and broker is not None:
            _rest_fallback_parallel(broker, pending, timeout_sec=min(2.0, timeout_sec))

        for ref in live_refs:
            if not _is_fill_complete(ref):
                logger.warning(
                    "Order %s fill not confirmed (status=%s avg=%.2f)",
                    ref.order_id,
                    ref.status,
                    ref.avg_price,
                )
            else:
                logger.info(
                    "Order %s fill confirmed (status=%s avg=%.2f)",
                    ref.order_id,
                    ref.status,
                    ref.avg_price,
                )
        return refs


class OrderTracker:
    """Dedupe order-feed log spam by order id + status."""

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def should_log(self, order_id: str, status: str) -> bool:
        if not order_id:
            return False
        key = f"{order_id}:{status}"
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def check_margin_for_legs(broker: NeoBroker, orders: List[LegOrder]) -> bool:
    """Return True if margin_required succeeds for all legs (best-effort)."""
    limiter = RateLimiter()

    def _check(order: LegOrder) -> bool:
        limiter.wait()
        try:
            resp = broker.margin_required(order.leg, order.side, order.quantity)
            logger.info("Margin check %s %s: %s", order.side, order.leg.trading_symbol, resp)
            return True
        except Exception as exc:
            logger.warning("Margin check failed for %s: %s", order.leg.trading_symbol, exc)
            return False

    if len(orders) <= 1:
        return all(_check(order) for order in orders)

    with ThreadPoolExecutor(max_workers=len(orders)) as pool:
        results = list(pool.map(_check, orders))
    return all(results)


def _rest_fallback_once(broker: NeoBroker, order_ref: OrderRef) -> OrderRef:
    if order_ref.is_paper or not order_ref.order_id:
        return order_ref
    try:
        hist = broker.client.order_history(order_id=order_ref.order_id)
        status = _order_status(hist)
        avg_price = _extract_avg_price_from_history(hist)
        if status:
            order_ref.status = status
        if avg_price > 0:
            order_ref.avg_price = avg_price
    except Exception as exc:
        logger.warning("order_history fallback failed for %s: %s", order_ref.order_id, exc)
    return order_ref


def _rest_fallback_parallel(
    broker: NeoBroker,
    refs: List[OrderRef],
    *,
    timeout_sec: float,
) -> List[OrderRef]:
    if not refs:
        return refs
    with ThreadPoolExecutor(max_workers=max(1, len(refs))) as pool:
        futures = {pool.submit(_rest_fallback_once, broker, ref): ref for ref in refs}
        for future in as_completed(futures, timeout=timeout_sec):
            try:
                future.result()
            except Exception as exc:
                logger.warning("REST fill fallback worker failed: %s", exc)
    return refs


def _place_legs_parallel(broker: NeoBroker, orders: List[LegOrder], tag: Optional[str]) -> List[OrderRef]:
    prefix = tag or "live"
    if len(orders) <= 1:
        return broker.place_legs(orders, tag=tag)

    results: List[Optional[OrderRef]] = [None] * len(orders)
    with ThreadPoolExecutor(max_workers=len(orders)) as pool:
        futures = {
            pool.submit(
                broker.place_leg,
                order.leg,
                order.side,
                order.quantity,
                unique_order_tag(prefix, order.leg),
            ): idx
            for idx, order in enumerate(orders)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return [ref for ref in results if ref is not None]


def place_legs_safe(
    broker: NeoBroker,
    orders: List[LegOrder],
    *,
    tag: Optional[str] = None,
    check_margin: bool = True,
    order_watcher: Optional[OrderFillWatcher] = None,
    fill_timeout_sec: float = 8.0,
) -> List[OrderRef]:
    """Place legs with optional margin check and websocket-confirmed fills."""
    if check_margin and not broker.cfg.is_live:
        check_margin = False
    if check_margin and not check_margin_for_legs(broker, orders):
        raise RuntimeError("Pre-trade margin check failed; aborting entry")

    refs = _place_legs_parallel(broker, orders, tag=tag)
    if order_watcher is not None:
        return order_watcher.wait_for_orders(refs, timeout_sec=fill_timeout_sec, broker=broker)
    return refs


def flatten_orphan_leg(broker: NeoBroker, leg: Leg, entry_side: str, quantity: int) -> OrderRef:
    """Emergency: reverse a single filled leg to avoid naked exposure."""
    reverse = SELL if entry_side == BUY else BUY
    logger.error("Flattening orphan leg %s %s x%d", reverse, leg.trading_symbol, quantity)
    return broker.place_leg(leg, reverse, quantity, tag=unique_order_tag("orphan", leg))


def square_off_safe(
    broker: NeoBroker,
    orders: List[LegOrder],
    tag: str = "square_off",
    *,
    order_watcher: Optional[OrderFillWatcher] = None,
    fill_timeout_sec: float = 8.0,
) -> List[OrderRef]:
    return place_legs_safe(
        broker,
        orders,
        tag=tag,
        check_margin=False,
        order_watcher=order_watcher,
        fill_timeout_sec=fill_timeout_sec,
    )
