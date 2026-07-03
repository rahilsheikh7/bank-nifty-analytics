"""Kotak Neo broker wrapper used by the live bot.

`NeoBroker` is the live adapter over the official `neo_api_client.NeoAPI` SDK:
login (TOTP), option scrip resolution (nearest monthly expiry + ATM CE/PE),
documented REST place-order API, quotes, positions, and WebSocket feeds.

`PaperBroker` subclasses it and overrides only order placement: it simulates
fills at the live option LTP so paper runs use real market prices without
sending real orders. Market data (index feed, quotes) is always real.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from live.config import LiveConfig, NeoCredentials
from live.neo_place_order import (
    is_place_order_ok,
    is_session_expired_error,
    place_order_doc,
    resolve_doc_session,
    unique_order_tag,
)
from live.neo_session import ensure_trade_session, is_token_stale

logger = logging.getLogger("live.neo")

# Transaction-type codes used by the Neo place_order API.
BUY = "B"
SELL = "S"

_MONTHS = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
}


@dataclass
class Leg:
    """A single option contract (one side of the synthetic future)."""
    trading_symbol: str
    token: str
    lot_size: int
    strike: float
    expiry: str            # DDMMMYYYY (e.g. 24JUN2026)
    option_type: str       # CE | PE
    segment: str = "nse_fo"


@dataclass
class OrderRef:
    """Uniform result of an order placement (live or paper)."""
    order_id: str
    trading_symbol: str
    side: str              # B | S
    quantity: int
    status: str            # "complete", "rejected", "paper", ...
    avg_price: float = 0.0
    is_paper: bool = False
    raw: Any = None


@dataclass
class LegOrder:
    """An order intent for one leg: which contract, which side, how many units."""
    leg: Leg
    side: str
    quantity: int


@dataclass
class BrokerPosition:
    """Open synthetic position as the broker sees it (used for reconciliation)."""
    ce: Leg
    pe: Leg
    ce_side: str
    pe_side: str
    quantity: int
    entry_orders: List[OrderRef] = field(default_factory=list)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_STRIKE_FROM_SYMBOL = re.compile(r"(\d{5})(?:CE|PE)$", re.IGNORECASE)


def _strike_from_symbol(trading_symbol: str) -> float:
    match = _STRIKE_FROM_SYMBOL.search(str(trading_symbol or "").strip().upper())
    return float(match.group(1)) if match else 0.0


def _normalize_strike(raw: Any, trading_symbol: str = "") -> float:
    """Return strike in index points (rupees). Neo scrip master uses paise."""
    strike = _to_float(raw, default=0.0)
    if strike <= 0:
        strike = _strike_from_symbol(trading_symbol)
    if strike <= 0:
        return 0.0
    # e.g. 5720000 paise -> 57200 INR (Bank Nifty strikes are well below 100k INR).
    if strike >= 100_000:
        strike /= 100.0
    return strike


def _parse_expiry_to_date(value: Any) -> Optional[date]:
    """Parse an expiry field from the scrip master into a date.

    Handles epoch seconds (int/float/str of digits) and common string formats
    like ``28JUN2026``, ``2026-06-28``, ``28-Jun-2026``.
    """
    if value is None or value == "" or value == -1:
        return None
    # Epoch seconds (Neo often returns large numeric timestamps).
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        try:
            num = float(value)
            if num > 10_000_000:  # plausibly an epoch, not a tiny code
                return datetime.fromtimestamp(num).date()
        except (TypeError, ValueError, OSError):
            pass
    text = str(value).strip().upper()
    for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _fmt_expiry(d: date) -> str:
    return f"{d.day:02d}{_MONTHS[d.month]}{d.year}"


def _is_last_expiry_of_month(target: date, all_dates: List[date]) -> bool:
    same_month = [d for d in all_dates if d.year == target.year and d.month == target.month]
    return bool(same_month) and target == max(same_month)


def _extract_ltp(resp: Any) -> Optional[float]:
    """Best-effort pull of a last-traded-price from a quotes() response."""
    keys = ("ltp", "last_traded_price", "lastPrice", "iv", "lp")

    def _search(node: Any) -> Optional[float]:
        if isinstance(node, dict):
            for k in keys:
                if k in node and node[k] not in (None, "", "0", 0):
                    val = _to_float(node[k], default=float("nan"))
                    if not math.isnan(val):
                        return val
            for v in node.values():
                found = _search(v)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = _search(v)
                if found is not None:
                    return found
        return None

    return _search(resp)


def _extract_order_id(resp: Any) -> Tuple[str, str]:
    """Return (order_id, status) from a place_order response, best-effort."""
    order_id = ""
    status = "unknown"
    if isinstance(resp, dict):
        for k in ("nOrdNo", "orderId", "order_id", "ordNo", "id"):
            if resp.get(k):
                order_id = str(resp[k])
                break
        data = resp.get("data")
        if not order_id and isinstance(data, dict):
            for k in ("nOrdNo", "orderId", "order_id", "ordNo"):
                if data.get(k):
                    order_id = str(data[k])
                    break
        status = str(resp.get("stat") or resp.get("status") or resp.get("ordSt") or status)
    return order_id, status


class NeoBroker:
    """Live Kotak Neo adapter. Market data is real; orders are real."""

    def __init__(self, creds: NeoCredentials, live_cfg: LiveConfig):
        self.creds = creds
        self.cfg = live_cfg
        self.client: Any = None
        self._scrip_cache: Dict[Tuple[str, str, str], Leg] = {}
        self._expiry_cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._doc_session: Any = None
        self._doc_session_source: str = ""

    def _invalidate_doc_session(self) -> None:
        self._doc_session = None
        self._doc_session_source = ""

    def refresh_trade_session(self, *, force: bool = False) -> None:
        """Regenerate token.json (tradeApiLogin + tradeApiValidate) and clear order session cache."""
        ensure_trade_session(self.creds, force=force, totp_provider=lambda: self._resolve_totp(None))
        self._invalidate_doc_session()

    def _ensure_daily_trade_session(self) -> None:
        if is_token_stale():
            logger.info("Trade session stale (new IST trading day) — refreshing token.json")
            self.refresh_trade_session(force=True)

    # ------------------------------------------------------------------ login
    def login(self, totp_provider: Optional[Callable[[], str]] = None) -> None:
        """Authenticate via the SDK: NeoAPI -> totp_login -> totp_validate."""
        try:
            from neo_api_client import NeoAPI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "neo_api_client is not installed. Run: "
                "pip install -r requirements-live.txt"
            ) from exc

        if not self.creds.consumer_key:
            raise RuntimeError("NEO_API_KEY (consumer_key) missing in .env")
        if not self.creds.mobile:
            raise RuntimeError("NEO_MOBILE missing in .env (required for totp_login)")

        self.client = NeoAPI(
            environment="prod",
            access_token=None,
            neo_fin_key=None,
            consumer_key=self.creds.consumer_key,
        )

        totp = self._resolve_totp(totp_provider)
        self._totp_login(self.creds.mobile, self.creds.client_code, totp)
        self.client.totp_validate(mpin=self.creds.mpin)
        self._invalidate_doc_session()
        logger.info("Neo login successful for ucc=%s", self.creds.client_code)

    def _get_doc_session(self):
        self._ensure_daily_trade_session()
        if self._doc_session is None:
            session, source = resolve_doc_session(self.client, prefer_login=False)
            self._doc_session = session
            self._doc_session_source = source
            logger.info("Neo doc API session: %s", source)
        return self._doc_session

    def _resolve_totp(self, totp_provider: Optional[Callable[[], str]]) -> str:
        if self.creds.has_totp_secret:
            try:
                import pyotp  # type: ignore

                return pyotp.TOTP(self.creds.totp_secret).now()
            except ImportError:
                logger.warning("pyotp not installed; falling back to manual TOTP entry")
        if totp_provider is not None:
            return totp_provider()
        return input("Enter the 6-digit TOTP from your authenticator: ").strip()

    def _totp_login(self, mobile: str, ucc: str, totp: str) -> None:
        # The SDK has used both `mobilenumber` and `mobile_number` across versions.
        try:
            self.client.totp_login(mobilenumber=mobile, ucc=ucc, totp=totp)
        except TypeError:
            self.client.totp_login(mobile_number=mobile, ucc=ucc, totp=totp)

    # ----------------------------------------------------------------- quotes
    def get_ltp(self, token: str, segment: str) -> Optional[float]:
        resp = self.client.quotes(
            instrument_tokens=[{"instrument_token": str(token), "exchange_segment": segment}],
            quote_type="ltp",
        )
        ltp = _extract_ltp(resp)
        if ltp is None:
            logger.warning("Could not parse LTP for %s|%s from %s", segment, token, resp)
        return ltp

    def get_index_ltp(self) -> Optional[float]:
        return self.get_ltp(self.cfg.index_name, self.cfg.index_segment)

    # -------------------------------------------------------- scrip resolution
    def _search(
        self, option_type: str, strike: Optional[int] = None, expiry: str = ""
    ) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            "exchange_segment": self.cfg.option_segment,
            "symbol": self.cfg.underlying_symbol,
            "option_type": option_type,
            "expiry": expiry,
            "strike_price": str(strike) if strike is not None else "",
        }
        resp = self.client.search_scrip(**kwargs)
        if isinstance(resp, dict) and "data" in resp:
            resp = resp["data"]
        return resp if isinstance(resp, list) else []

    def nearest_expiry(self, today: Optional[date] = None) -> str:
        """Pick the nearest non-expired expiry per config (monthly/weekly)."""
        today = today or datetime.now().date()
        cache_key = f"{self.cfg.expiry_rule}:{today.isoformat()}"
        if cache_key in self._expiry_cache:
            return self._expiry_cache[cache_key]

        rows = self._search("CE")  # all strikes/expiries for the underlying CE
        dates: List[date] = []
        for row in rows:
            d = _parse_expiry_to_date(row.get("pExpiryDate") or row.get("lExpiryDate"))
            if d is not None:
                dates.append(d)
        dates = sorted(set(dates))

        cutoff_ordinal = today.toordinal() + max(0, self.cfg.expiry_roll_days)
        future = [d for d in dates if d.toordinal() >= cutoff_ordinal]

        chosen: Optional[date] = None
        if self.cfg.expiry_rule.lower() == "weekly":
            chosen = future[0] if future else None
        else:  # monthly: nearest expiry that is the last expiry of its month
            monthlies = [d for d in future if _is_last_expiry_of_month(d, dates)]
            chosen = monthlies[0] if monthlies else (future[0] if future else None)

        if chosen is None:
            chosen = self._computed_monthly_expiry(today)
            logger.warning("No expiry from scrip search; using computed %s", _fmt_expiry(chosen))

        result = _fmt_expiry(chosen)
        self._expiry_cache[cache_key] = result
        logger.info("Selected %s expiry: %s", self.cfg.expiry_rule, result)
        return result

    @staticmethod
    def _computed_monthly_expiry(today: date) -> date:
        """Fallback: last Thursday of the month (rolls to next month if passed)."""
        import calendar

        def last_thursday(year: int, month: int) -> date:
            last_day = calendar.monthrange(year, month)[1]
            d = date(year, month, last_day)
            while d.weekday() != 3:  # Thursday
                d = d.replace(day=d.day - 1)
            return d

        exp = last_thursday(today.year, today.month)
        if exp < today:
            ny, nm = (today.year + (today.month // 12), (today.month % 12) + 1)
            exp = last_thursday(ny, nm)
        return exp

    def _leg_from_row(self, row: Dict[str, Any], option_type: str, expiry: str) -> Leg:
        trading_symbol = str(row.get("pTrdSymbol") or row.get("pSymbolName"))
        raw_strike = row.get("dStrikePrice") or row.get("dStrikePrice;")
        return Leg(
            trading_symbol=trading_symbol,
            token=str(row.get("pSymbol") or row.get("pScripRefKey")),
            lot_size=int(row.get("lLotSize") or row.get("iLotSize") or 0),
            strike=_normalize_strike(raw_strike, trading_symbol),
            expiry=expiry,
            option_type=option_type,
            segment=self.cfg.option_segment,
        )

    def _resolve_leg(self, option_type: str, strike: int, expiry: str) -> Leg:
        cache_key = (option_type, str(strike), expiry)
        if cache_key in self._scrip_cache:
            return self._scrip_cache[cache_key]
        rows = self._search(option_type, strike=strike, expiry=expiry)
        match: Optional[Dict[str, Any]] = None
        for row in rows:
            sym = str(row.get("pTrdSymbol") or row.get("pSymbolName") or "")
            row_strike = _normalize_strike(
                row.get("dStrikePrice") or row.get("dStrikePrice;"),
                sym,
            )
            if abs(row_strike - strike) < 0.5:
                match = row
                break
        if match is None and rows:
            match = rows[0]
        if match is None:
            raise RuntimeError(
                f"No {self.cfg.underlying_symbol} {option_type} at strike {strike} expiry {expiry}"
            )
        leg = self._leg_from_row(match, option_type, expiry)
        if leg.lot_size <= 0:
            raise RuntimeError(f"Resolved {leg.trading_symbol} has invalid lot size {leg.lot_size}")
        self._scrip_cache[cache_key] = leg
        return leg

    def atm_strike(self, index_price: float) -> int:
        step = self.cfg.strike_step
        return int(round(index_price / step) * step)

    def resolve_atm_legs(
        self, index_price: float, expiry: Optional[str] = None
    ) -> Dict[str, Leg]:
        """Resolve the ATM CE and PE legs for the given index price + expiry."""
        expiry = expiry or self.nearest_expiry()
        strike = self.atm_strike(index_price)
        ce = self._resolve_leg("CE", strike, expiry)
        pe = self._resolve_leg("PE", strike, expiry)
        logger.info("ATM legs @ index=%.2f strike=%d expiry=%s: CE=%s PE=%s",
                    index_price, strike, expiry, ce.trading_symbol, pe.trading_symbol)
        return {"ce": ce, "pe": pe}

    # ------------------------------------------------------------------ orders
    def place_leg(self, leg: Leg, side: str, quantity: int, tag: Optional[str] = None) -> OrderRef:
        leg_tag = tag or unique_order_tag("live", leg)
        resp: Dict[str, Any] = {}
        for attempt in range(2):
            session = self._get_doc_session()
            resp = place_order_doc(self.cfg, leg, side, quantity, session, tag=leg_tag)
            if is_place_order_ok(resp):
                break
            if attempt == 0 and is_session_expired_error(resp):
                logger.warning(
                    "Trade session expired during order; regenerating token.json and retrying"
                )
                self.refresh_trade_session(force=True)
                continue
            break

        order_id, status = _extract_order_id(resp)
        if not is_place_order_ok(resp):
            logger.error(
                "LIVE order REJECTED %s %s x%d -> status=%s id=%s resp=%s",
                side, leg.trading_symbol, quantity, status, order_id or "(none)", resp,
            )
        else:
            logger.info(
                "LIVE order %s %s x%d -> id=%s status=%s (doc API, %s)",
                side, leg.trading_symbol, quantity, order_id, status, self._doc_session_source,
            )
        return OrderRef(
            order_id=order_id,
            trading_symbol=leg.trading_symbol,
            side=side,
            quantity=quantity,
            status=status,
            raw=resp,
        )

    def place_legs(self, orders: List[LegOrder], tag: Optional[str] = None) -> List[OrderRef]:
        prefix = tag or "live"
        return [
            self.place_leg(o.leg, o.side, o.quantity, tag=unique_order_tag(prefix, o.leg))
            for o in orders
        ]

    def square_off(self, orders: List[LegOrder], tag: Optional[str] = None) -> List[OrderRef]:
        """Place the reversing orders (caller passes already-reversed sides)."""
        return self.place_legs(orders, tag=tag)

    def positions(self) -> Any:
        return self.client.positions()

    def limits(self) -> Any:
        return self.client.limits(segment="ALL", exchange="ALL", product="ALL")

    def margin_required(self, leg: Leg, side: str, quantity: int) -> Any:
        return self.client.margin_required(
            exchange_segment=leg.segment,
            price="0",
            order_type=self.cfg.order_type,
            product=self.cfg.product,
            quantity=str(quantity),
            instrument_token=str(leg.token),
            transaction_type=side,
        )

    # ------------------------------------------------------------------- feeds
    def set_callbacks(
        self,
        on_message: Callable[[Any], None],
        on_open: Optional[Callable[[Any], None]] = None,
        on_close: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self.client.on_message = on_message
        if on_open:
            self.client.on_open = on_open
        if on_close:
            self.client.on_close = on_close
        if on_error:
            self.client.on_error = on_error

    def start_market_feed(self) -> None:
        self.client.subscribe(
            instrument_tokens=[
                {"instrument_token": self.cfg.index_name, "exchange_segment": self.cfg.index_segment}
            ],
            isIndex=True,
        )
        logger.info("Subscribed to index feed: %s|%s", self.cfg.index_segment, self.cfg.index_name)

    def start_order_feed(self) -> None:
        self.client.subscribe_to_orderfeed()
        logger.info("Subscribed to order feed")

    def logout(self) -> None:
        try:
            if self.client is not None:
                self.client.logout()
        except Exception as exc:  # pragma: no cover
            logger.warning("logout failed: %s", exc)


class PaperBroker(NeoBroker):
    """Paper adapter: real market data, simulated fills at the live option LTP."""

    def __init__(self, creds: NeoCredentials, live_cfg: LiveConfig):
        super().__init__(creds, live_cfg)
        self._paper_seq = 0

    def place_leg(self, leg: Leg, side: str, quantity: int, tag: Optional[str] = None) -> OrderRef:
        ltp = self.get_ltp(leg.token, leg.segment) or 0.0
        self._paper_seq += 1
        order_id = f"PAPER-{self._paper_seq:06d}"
        logger.info("PAPER order %s %s x%d @ %.2f (id=%s)",
                    side, leg.trading_symbol, quantity, ltp, order_id)
        return OrderRef(
            order_id=order_id,
            trading_symbol=leg.trading_symbol,
            side=side,
            quantity=quantity,
            status="paper",
            avg_price=ltp,
            is_paper=True,
        )


def make_broker(creds: NeoCredentials, live_cfg: LiveConfig) -> NeoBroker:
    """Factory: PaperBroker unless live.mode == 'live'."""
    if live_cfg.is_live:
        logger.warning("LIVE MODE: real orders will be placed on Kotak Neo")
        return NeoBroker(creds, live_cfg)
    logger.info("PAPER MODE: orders are simulated; market data is real")
    return PaperBroker(creds, live_cfg)
