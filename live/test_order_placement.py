"""Standalone Neo order-placement test — does NOT change live bot order code.

Uses NeoBroker for login, quotes, scrip resolution, and margin checks.
Doc API orders use ``token.json`` from ``generate_neo_session.py`` by default.

Refresh session:
  python generate_neo_session.py

Usage:
  python -m live.test_order_placement --place --yes
  python -m live.test_order_placement --place --yes --round-trip
  python -m live.test_order_placement --place --yes --round-trip --wait-seconds 15

  # Fast path: token.json only — no login, parallel place
  python -m live.test_order_placement --direct --expiry 28JUL2026 --direction short --place --yes
  python -m live.test_order_placement --direct --expiry 28JUL2026 --direction short --place --yes --round-trip
  python -m live.test_order_placement --direct --expiry 28JUL2026 --direction short --strike 57400 --exit-only --place --yes

  # Cancel open orders by nest order number (nOrdNo from place response)
  python -m live.test_order_placement --cancel 260708000632115 260708000632119 --yes

Inspect search_scrip (no orders):
  python -m live.inspect_scrip
  python -m live.inspect_scrip --strike 57900 --expiry 28JUL2026 --option-type BOTH
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from live.config import LiveConfig, build_live_config, load_live_config, load_neo_credentials
from live.config import NeoCredentials
from live.neo_client import BUY, Leg, LegOrder, NeoBroker, OrderRef, SELL, _extract_ltp, _extract_order_id
from live.safety import check_margin_for_legs

logger = logging.getLogger("live.test_order")

NEO_FIN_KEY = "neotradeapi"
DEFAULT_SESSION_FILE = Path(__file__).resolve().parent.parent / "api.txt"
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent.parent / "token.json"

_EXPIRY_RE = re.compile(r"^(\d{1,2})([A-Za-z]{3})(\d{4})$")


def _timing(label: str, seconds: float) -> None:
    print(f"  [timing] {label}: {seconds:.3f}s")


@contextmanager
def _timed(label: str):
    t0 = time.perf_counter()
    yield
    _timing(label, time.perf_counter() - t0)


def _atm_strike(index_price: float, step: int) -> int:
    return int(round(index_price / step) * step)


def _load_doc_session_standalone(
    *,
    token_file: Path,
    session_file: Path,
    use_session_file: bool,
) -> tuple[ApiSession, str]:
    if use_session_file:
        file_session = load_api_session(session_file)
        if file_session is None:
            raise RuntimeError(f"Could not parse {session_file}")
        return file_session, f"api.txt ({session_file})"

    token_loaded = load_token_session(token_file)
    if token_loaded:
        session, meta = token_loaded
        when = meta.get("generated_at") or "unknown time"
        return session, f"token.json ({token_file}, generated {when})"

    raise RuntimeError("No session: run python generate_neo_session.py")


def _fetch_index_ltp_standalone(creds: NeoCredentials, live_cfg: LiveConfig) -> Optional[float]:
    """Index LTP via quotes API — consumer_key only, no totp login."""
    if not creds.consumer_key:
        return None
    try:
        from neo_api_client import NeoAPI  # type: ignore
        from neo_api_client.urls import BASE_URL
    except ImportError:
        return None
    client = NeoAPI(environment="prod", consumer_key=creds.consumer_key)
    client.configuration.base_url = BASE_URL
    resp = client.quotes(
        instrument_tokens=[
            {"instrument_token": live_cfg.index_name, "exchange_segment": live_cfg.index_segment}
        ],
        quote_type="ltp",
    )
    return _extract_ltp(resp)


def _use_fast_path(args: argparse.Namespace) -> bool:
    if args.use_login_session or args.dump_session:
        return False
    if args.method != "doc":
        return False
    if args.verify_scrip:
        return False
    if args.cancel:
        return True
    if args.exit_only:
        return True
    return bool(args.direct)


def _normalize_expiry(expiry: str) -> str:
    """Normalize to DDMMMYYYY uppercase, e.g. 28JUL2026."""
    raw = str(expiry or "").strip().upper().replace(" ", "")
    m = _EXPIRY_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid expiry {expiry!r}; use e.g. 28JUL2026 or 25AUG2026")
    day, mon, year = m.groups()
    return f"{int(day):02d}{mon}{year}"


def build_option_symbol(underlying: str, expiry: str, strike: int, option_type: str) -> str:
    """Build Neo trading symbol without search_scrip.

    Pattern from scrip master (e.g. BANKNIFTY26JUL53000CE for 28JUL2026 @ 53000).
    """
    norm = _normalize_expiry(expiry)
    m = _EXPIRY_RE.match(norm)
    assert m is not None
    _day, mon, year = m.groups()
    yy = year[-2:]
    ot = str(option_type).strip().upper()
    if ot not in {"CE", "PE"}:
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")
    return f"{underlying.upper()}{yy}{mon}{int(strike)}{ot}"


def build_direct_legs(
    live_cfg: LiveConfig,
    *,
    strike: int,
    expiry: str,
    lot_size: int,
) -> Dict[str, Leg]:
    """Synthetic CE+PE legs from strike/expiry only (place order needs ts, not token)."""
    expiry_norm = _normalize_expiry(expiry)
    ce_sym = build_option_symbol(live_cfg.underlying_symbol, expiry_norm, strike, "CE")
    pe_sym = build_option_symbol(live_cfg.underlying_symbol, expiry_norm, strike, "PE")
    segment = live_cfg.option_segment
    return {
        "ce": Leg(
            trading_symbol=ce_sym,
            token="",
            lot_size=int(lot_size),
            strike=float(strike),
            expiry=expiry_norm,
            option_type="CE",
            segment=segment,
        ),
        "pe": Leg(
            trading_symbol=pe_sym,
            token="",
            lot_size=int(lot_size),
            strike=float(strike),
            expiry=expiry_norm,
            option_type="PE",
            segment=segment,
        ),
    }


def _verify_symbols_with_scrip(
    broker: NeoBroker,
    legs: Dict[str, Leg],
    expiry: str,
) -> bool:
    """Optional: compare built symbols against search_scrip (slow)."""
    strike = int(legs["ce"].strike)
    expiry_norm = _normalize_expiry(expiry)
    ok = True
    for ot in ("CE", "PE"):
        key = ot.lower()
        built = legs[key].trading_symbol
        rows = broker._search(ot, strike=strike, expiry=expiry_norm)
        api_sym = None
        if rows:
            api_sym = rows[0].get("pTrdSymbol") or rows[0].get("pSymbolName")
        match = api_sym == built
        print(f"  {ot}: built={built}  scrip={api_sym or '(not found)'}  {'OK' if match else 'MISMATCH'}")
        ok = ok and match
    return ok


def _pp(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _unique_order_tag(prefix: str, leg: Leg) -> str:
    """Unique client order id (jData ig) — Kotak rejects duplicate ig across orders."""
    ts = int(time.time() * 1000) % 1_000_000
    tag = f"{prefix}_{leg.option_type}_{ts}"
    return tag[:20]


@dataclass
class ApiSession:
    """Session from tradeApiValidate (paste into api.txt)."""
    auth_token: str   # data.token -> header Auth
    sid: str          # data.sid  -> header Sid
    rid: str          # data.rid  (logged only; not sent on place order)
    base_url: str     # data.baseUrl
    server_id: str = ""
    data_center: str = ""
    ucc: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "sid": self.sid,
            "rid": self.rid,
            "has_auth_token": bool(self.auth_token),
            "server_id": self.server_id or "(empty)",
            "data_center": self.data_center,
            "ucc": self.ucc,
        }


def load_api_session(path: Path) -> Optional[ApiSession]:
    """Parse tradeApiValidate JSON from api.txt (supports # comment lines)."""
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    # Try bare JSON first, then strip leading '#' from each line (api.txt format).
    candidates = [raw.strip()]
    uncommented = "\n".join(
        line[1:].strip() if line.lstrip().startswith("#") else line
        for line in raw.splitlines()
    ).strip()
    if uncommented:
        candidates.append(uncommented)
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidates.append(match.group(0))

    data: Optional[Dict[str, Any]] = None
    for blob in candidates:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
                break
        except json.JSONDecodeError:
            continue
    if not data:
        return None

    token = data.get("token")
    sid = data.get("sid")
    base_url = data.get("baseUrl")
    if not token or not sid or not base_url:
        return None
    return ApiSession(
        auth_token=str(token),
        sid=str(sid),
        rid=str(data.get("rid") or ""),
        base_url=str(base_url).rstrip("/"),
        server_id=str(data.get("hsServerId") or ""),
        data_center=str(data.get("dataCenter") or ""),
        ucc=str(data.get("ucc") or ""),
    )


def load_token_session(path: Path) -> Optional[tuple[ApiSession, Dict[str, Any]]]:
    """Load trade session from token.json (output of generate_neo_session.py)."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None

    meta: Dict[str, Any] = {"generated_at": payload.get("generated_at")}
    block = payload.get("session")
    if not isinstance(block, dict):
        validate = payload.get("validate")
        block = validate.get("data") if isinstance(validate, dict) else None
    if not isinstance(block, dict):
        return None

    auth = block.get("auth_token") or block.get("token")
    sid = block.get("sid")
    base_url = block.get("baseUrl") or block.get("base_url")
    if not auth or not sid or not base_url:
        return None

    meta["kType"] = block.get("kType", "Trade")
    session = ApiSession(
        auth_token=str(auth),
        sid=str(sid),
        rid=str(block.get("rid") or ""),
        base_url=str(base_url).rstrip("/"),
        server_id=str(block.get("hsServerId") or ""),
        data_center=str(block.get("dataCenter") or ""),
        ucc=str(block.get("ucc") or ""),
    )
    return session, meta


def _normalize_order_type(order_type: str) -> str:
    key = str(order_type or "MKT").strip().upper()
    return {"MKT": "MKT", "MARKET": "MKT", "L": "L", "LIMIT": "L"}.get(key, key)


def _order_error(resp: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(resp, dict):
        return {"raw": resp}
    for key in ("stat", "status", "emsg", "errMsg", "message", "stCode", "_http_status"):
        if resp.get(key) not in (None, ""):
            out[key] = resp.get(key)
    return out


def _is_order_ok(status: str, order_id: str, raw: Any) -> bool:
    status_l = (status or "").lower()
    if status_l in {"not_ok", "not ok", "rejected", "error", "unknown"}:
        return False
    err = _order_error(raw)
    if str(err.get("stat", "")).lower() == "not_ok":
        return False
    if err.get("emsg") or err.get("errMsg"):
        return False
    return bool(order_id) or status_l == "ok"


def _session_info(broker: NeoBroker) -> Dict[str, Any]:
    cfg = broker.client.configuration
    return {
        "source": "login",
        "base_url": cfg.base_url,
        "sid": cfg.edit_sid,
        "has_auth_token": bool(cfg.edit_token),
        "server_id": cfg.serverId,
        "data_center": cfg.data_center,
    }


def _place_order_url_from_session(session: ApiSession) -> str:
    return f"{session.base_url.rstrip('/')}/quick/order/rule/ms/place"


def _place_order_url(broker: NeoBroker) -> str:
    cfg = broker.client.configuration
    base = str(cfg.base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("Neo baseUrl missing after login")
    session = ApiSession(
        auth_token=str(cfg.edit_token or ""),
        sid=str(cfg.edit_sid or ""),
        rid="",
        base_url=base,
        server_id=str(cfg.serverId or ""),
        data_center=str(cfg.data_center or ""),
    )
    return _place_order_url_from_session(session)


def _place_order_headers_from_session(session: ApiSession) -> Dict[str, str]:
    if not session.auth_token or not session.sid:
        raise RuntimeError("Session missing token (Auth) or sid")
    return {
        "accept": "application/json",
        "Sid": session.sid,
        "Auth": session.auth_token,
        "neo-fin-key": NEO_FIN_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _place_order_headers(broker: NeoBroker) -> Dict[str, str]:
    cfg = broker.client.configuration
    if not cfg.edit_token or not cfg.edit_sid:
        raise RuntimeError("Neo session not ready: missing Auth or Sid")
    return _place_order_headers_from_session(
        ApiSession(
            auth_token=str(cfg.edit_token),
            sid=str(cfg.edit_sid),
            rid="",
            base_url=str(cfg.base_url or ""),
            server_id=str(cfg.serverId or ""),
        )
    )


def build_place_order_jdata(
    live_cfg: LiveConfig,
    leg: Leg,
    side: str,
    quantity: int,
    *,
    tag: Optional[str] = None,
) -> Dict[str, str]:
    """jData fields per Kotak Place Order API documentation."""
    jdata: Dict[str, str] = {
        "am": "NO",
        "dq": "0",
        "es": leg.segment,
        "mp": "0",
        "pc": live_cfg.product,
        "pf": "N",
        "pr": "0",
        "pt": _normalize_order_type(live_cfg.order_type),
        "qt": str(quantity),
        "rt": "DAY",
        "tp": "0",
        "ts": leg.trading_symbol,
        "tt": side,
    }
    if tag:
        jdata["ig"] = str(tag)[:20]
    return jdata


def place_order_doc(
    live_cfg: LiveConfig,
    leg: Leg,
    side: str,
    quantity: int,
    session: ApiSession,
    *,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    jdata = build_place_order_jdata(live_cfg, leg, side, quantity, tag=tag)
    url = _place_order_url_from_session(session)
    headers = _place_order_headers_from_session(session)
    body = {"jData": json.dumps(jdata, separators=(",", ":"))}
    response = requests.post(url, headers=headers, data=body, timeout=30)
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        payload = {"stat": "Not_Ok", "emsg": response.text, "stCode": response.status_code}
    if not isinstance(payload, dict):
        payload = {"stat": "Not_Ok", "emsg": str(payload), "stCode": response.status_code}
    payload["_http_status"] = response.status_code
    payload["_request"] = {
        "url": url,
        "jData": jdata,
        "session_source": session.summary(),
        "headers": {**headers, "Auth": "(hidden)"},
    }
    return payload


def cancel_order_doc(session: ApiSession, order_no: str, *, amo: str = "NO") -> Dict[str, Any]:
    """POST {baseUrl}/quick/order/cancel per Kotak documentation."""
    jdata = {"am": amo, "on": str(order_no)}
    url = f"{session.base_url.rstrip('/')}/quick/order/cancel"
    headers = _place_order_headers_from_session(session)
    body = {"jData": json.dumps(jdata, separators=(",", ":"))}
    response = requests.post(url, headers=headers, data=body, timeout=30)
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        payload = {"stat": "Not_Ok", "emsg": response.text, "stCode": response.status_code}
    if not isinstance(payload, dict):
        payload = {"stat": "Not_Ok", "emsg": str(payload), "stCode": response.status_code}
    payload["_http_status"] = response.status_code
    payload["_request"] = {
        "url": url,
        "jData": jdata,
        "session_source": session.summary(),
        "headers": {**headers, "Auth": "(hidden)"},
    }
    return payload


def _is_cancel_ok(resp: Dict[str, Any]) -> bool:
    status_l = str(resp.get("stat") or resp.get("status") or "").lower()
    if status_l in {"not_ok", "not ok", "rejected", "error", "unknown"}:
        return False
    if resp.get("emsg") or resp.get("errMsg"):
        return False
    st_code = resp.get("stCode")
    if st_code is not None and int(st_code) not in (200,):
        return False
    return status_l == "ok" or bool(resp.get("nOrdNo"))


def _cancel_order_filled(resp: Dict[str, Any]) -> bool:
    err = str(resp.get("errMsg") or resp.get("emsg") or "").lower()
    if "completed" in err or "traded" in err or "fill" in err:
        return True
    try:
        return int(resp.get("stCode") or 0) in (1021,)
    except (TypeError, ValueError):
        return False


def _cancel_orders(session: ApiSession, order_nos: List[str]) -> tuple[bool, bool]:
    """Return (all_cancelled_ok, any_already_filled)."""
    print("\n--- Cancelling orders (documented REST API) ---")
    all_ok = True
    any_filled = False
    t_total = time.perf_counter()
    for on in order_nos:
        t0 = time.perf_counter()
        resp = cancel_order_doc(session, on)
        _timing(f"cancel {on}", time.perf_counter() - t0)
        ok = _is_cancel_ok(resp)
        if not ok and _cancel_order_filled(resp):
            any_filled = True
        all_ok = all_ok and ok
        print(f"\n  Order: {on}")
        print(f"    success?      : {ok}")
        err = _order_error(resp)
        if err:
            print(f"    response      : {_pp(err)}")
        print(f"    raw response  :\n{_pp(resp)}")
    _timing("cancel total", time.perf_counter() - t_total)
    return all_ok, any_filled


def _square_off_legs(
    live_cfg: LiveConfig,
    entry_orders: List[LegOrder],
    *,
    doc_session: ApiSession,
    tag_prefix: str,
) -> tuple[bool, List[OrderRef]]:
    exit_orders = _exit_orders_from_entry(entry_orders)
    print("\n--- Square-off (reverse entry legs) ---")
    for o in exit_orders:
        side = "BUY" if o.side == BUY else "SELL"
        print(f"  {side} {o.leg.trading_symbol} qty={o.quantity}")
    return _place_legs(
        None,
        live_cfg,
        exit_orders,
        method="doc",
        tag_prefix=tag_prefix,
        doc_session=doc_session,
        parallel=True,
    )


def place_order_sdk(
    broker: NeoBroker,
    leg: Leg,
    side: str,
    quantity: int,
    *,
    tag: Optional[str] = None,
) -> Any:
    return broker.client.place_order(
        exchange_segment=leg.segment,
        product=broker.cfg.product,
        price="0",
        order_type=broker.cfg.order_type,
        quantity=str(quantity),
        validity="DAY",
        trading_symbol=leg.trading_symbol,
        transaction_type=side,
        amo="NO",
        disclosed_quantity="0",
        market_protection="0",
        pf="N",
        trigger_price="0",
        tag=tag,
    )


def _to_order_ref(leg: Leg, side: str, quantity: int, resp: Any) -> OrderRef:
    order_id, status = _extract_order_id(resp)
    return OrderRef(
        order_id=order_id,
        trading_symbol=leg.trading_symbol,
        side=side,
        quantity=quantity,
        status=status,
        raw=resp,
    )


def _entry_orders(direction: str, legs: Dict[str, Leg], lots: int) -> List[LegOrder]:
    qty_ce = lots * legs["ce"].lot_size
    qty_pe = lots * legs["pe"].lot_size
    if direction == "long":
        return [LegOrder(legs["ce"], BUY, qty_ce), LegOrder(legs["pe"], SELL, qty_pe)]
    return [LegOrder(legs["ce"], SELL, qty_ce), LegOrder(legs["pe"], BUY, qty_pe)]


def _exit_orders_from_entry(entry: List[LegOrder]) -> List[LegOrder]:
    """Square off: reverse each entry leg (same as live engine exit)."""
    flip = {BUY: SELL, SELL: BUY}
    return [LegOrder(o.leg, flip[o.side], o.quantity) for o in entry]


def _strike_audit(broker: NeoBroker, legs: Dict[str, Leg], index_price: float) -> bool:
    strike = broker.atm_strike(index_price)
    ce_strike, pe_strike = int(legs["ce"].strike), int(legs["pe"].strike)
    ok = ce_strike == strike and pe_strike == strike
    print("\n--- Strike audit (engine check) ---")
    print(f"  index_price     : {index_price:.2f}")
    print(f"  atm_strike      : {strike}")
    print(f"  CE leg strike   : {ce_strike} ({legs['ce'].trading_symbol})")
    print(f"  PE leg strike   : {pe_strike} ({legs['pe'].trading_symbol})")
    print(f"  audit result    : {'PASS' if ok else 'FAIL'}")
    return ok


def _report_leg(side_label: str, ref: OrderRef) -> bool:
    ok = _is_order_ok(ref.status, ref.order_id, ref.raw)
    print(f"\n  Leg: {side_label} {ref.trading_symbol} x{ref.quantity}")
    print(f"    parsed status : {ref.status}")
    print(f"    order_id      : {ref.order_id or '(empty)'}")
    print(f"    success?      : {ok}")
    err = _order_error(ref.raw)
    if err:
        print(f"    error fields  : {_pp(err)}")
    if isinstance(ref.raw, dict) and ref.raw.get("_request"):
        print(f"    request       : {_pp(ref.raw['_request'])}")
    print(f"    raw response  :\n{_pp(ref.raw)}")
    return ok


def _place_one_doc(
    live_cfg: LiveConfig,
    order: LegOrder,
    *,
    tag_prefix: str,
    doc_session: ApiSession,
) -> tuple[str, OrderRef, float]:
    side_label = "BUY" if order.side == BUY else "SELL"
    tag = _unique_order_tag(tag_prefix, order.leg)
    t0 = time.perf_counter()
    resp = place_order_doc(live_cfg, order.leg, order.side, order.quantity, doc_session, tag=tag)
    elapsed = time.perf_counter() - t0
    ref = _to_order_ref(order.leg, order.side, order.quantity, resp)
    return side_label, ref, elapsed


def _place_legs(
    broker: Optional[NeoBroker],
    live_cfg: LiveConfig,
    orders: List[LegOrder],
    *,
    method: str,
    tag_prefix: str,
    doc_session: ApiSession,
    parallel: bool = False,
) -> tuple[bool, List[OrderRef]]:
    label = "documented REST API" if method == "doc" else "neo_api_client SDK"
    mode = "parallel" if parallel and method == "doc" and len(orders) > 1 else "sequential"
    print(f"\n--- Placing orders ({label}, {mode}) ---")
    all_ok = True
    refs: List[OrderRef] = []
    t_total = time.perf_counter()

    if parallel and method == "doc" and len(orders) > 1:
        results: List[Optional[tuple[str, OrderRef, float]]] = [None] * len(orders)
        with ThreadPoolExecutor(max_workers=len(orders)) as pool:
            futures = {
                pool.submit(
                    _place_one_doc,
                    live_cfg,
                    order,
                    tag_prefix=tag_prefix,
                    doc_session=doc_session,
                ): idx
                for idx, order in enumerate(orders)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        for item in results:
            if item is None:
                continue
            side_label, ref, leg_elapsed = item
            refs.append(ref)
            all_ok = _report_leg(side_label, ref) and all_ok
            _timing(f"place {side_label} {ref.trading_symbol}", leg_elapsed)
    else:
        for o in orders:
            side_label = "BUY" if o.side == BUY else "SELL"
            tag = _unique_order_tag(tag_prefix, o.leg)
            t0 = time.perf_counter()
            if method == "doc":
                resp = place_order_doc(live_cfg, o.leg, o.side, o.quantity, doc_session, tag=tag)
            else:
                if broker is None:
                    raise RuntimeError("SDK place requires logged-in broker")
                resp = place_order_sdk(broker, o.leg, o.side, o.quantity, tag=tag)
            leg_elapsed = time.perf_counter() - t0
            ref = _to_order_ref(o.leg, o.side, o.quantity, resp)
            refs.append(ref)
            all_ok = _report_leg(side_label, ref) and all_ok
            _timing(f"place {side_label} {ref.trading_symbol}", leg_elapsed)

    _timing("place total (all legs)", time.perf_counter() - t_total)
    return all_ok, refs


def session_from_broker(broker: NeoBroker) -> ApiSession:
    cfg = broker.client.configuration
    return ApiSession(
        auth_token=str(cfg.edit_token or ""),
        sid=str(cfg.edit_sid or ""),
        rid=str(getattr(cfg, "edit_rid", "") or ""),
        base_url=str(cfg.base_url or "").rstrip("/"),
        server_id=str(cfg.serverId or ""),
        data_center=str(cfg.data_center or ""),
    )


def dump_session_file(broker: NeoBroker, path: Path) -> ApiSession:
    """Write a fresh tradeApiValidate-style JSON block to api.txt after login."""
    session = session_from_broker(broker)
    if not session.auth_token or not session.sid or not session.base_url:
        raise RuntimeError("Login did not return token/sid/baseUrl")
    payload = {
        "data": {
            "token": session.auth_token,
            "sid": session.sid,
            "rid": session.rid,
            "baseUrl": session.base_url,
            "hsServerId": session.server_id,
            "dataCenter": session.data_center,
            "ucc": session.ucc,
            "status": "success",
        }
    }
    lines = ["", "# Paste from tradeApiValidate — refresh with: python -m live.test_order_placement --dump-session", ""]
    for line in json.dumps(payload, indent=4).splitlines():
        lines.append("# " + line)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return session


def _resolve_doc_session(
    broker: NeoBroker,
    *,
    token_file: Path,
    session_file: Path,
    use_login_session: bool,
    use_session_file: bool,
) -> tuple[ApiSession, str]:
    login_session = session_from_broker(broker)

    if use_login_session:
        if not login_session.auth_token or not login_session.sid or not login_session.base_url:
            raise RuntimeError("Login session incomplete; check .env TOTP/MPIN and Neo API key")
        return login_session, "login (fresh — from this run)"

    if use_session_file:
        file_session = load_api_session(session_file) if session_file else None
        if file_session is None:
            raise RuntimeError(f"Could not parse {session_file}")
        return file_session, f"api.txt ({session_file})"

    token_loaded = load_token_session(token_file)
    if token_loaded:
        session, meta = token_loaded
        when = meta.get("generated_at") or "unknown time"
        return session, f"token.json ({token_file}, generated {when})"

    if login_session.auth_token and login_session.sid and login_session.base_url:
        return login_session, "login (token.json missing — fallback)"

    raise RuntimeError(
        f"No session: run python generate_neo_session.py or pass --use-login-session"
    )


def _run_fast_path(
    args: argparse.Namespace,
    live_cfg: LiveConfig,
    creds: NeoCredentials,
    lots: int,
) -> int:
    """token.json only: index -> symbol -> parallel place. No login, no limits."""
    t_run = time.perf_counter()
    print("=== Neo fast direct (token.json — no login, no limits) ===")
    print(f"  product={live_cfg.product} order_type={live_cfg.order_type} lots={lots}")
    print(f"  direction={args.direction}")
    if args.place_cancel:
        print(f"  place_cancel: wait {args.cancel_wait_seconds}s -> cancel (or square-off if MKT filled)")
    if args.round_trip:
        print(f"  round_trip: wait {args.wait_seconds}s -> square-off")
    if args.exit_only:
        print(f"  exit_only: square-off existing {args.direction} legs (no new entry)")

    try:
        t0 = time.perf_counter()
        doc_session, doc_source = _load_doc_session_standalone(
            token_file=args.token_file,
            session_file=args.session_file,
            use_session_file=args.use_session_file,
        )
        _timing("load session", time.perf_counter() - t0)

        if args.cancel:
            print(f"  session: {doc_source}")
            if not args.yes:
                print("\nDRY RUN — pass --yes to send cancel requests")
                for on in args.cancel:
                    print(f"  would cancel: {on}")
                return 0
            cancel_ok, _ = _cancel_orders(doc_session, args.cancel)
            _timing("run total (cancel)", time.perf_counter() - t_run)
            print("\n=== Cancel complete ===" if cancel_ok else "\n=== Cancel had failures ===")
            return 0 if cancel_ok else 1

        if not args.expiry:
            print("--direct requires --expiry (e.g. 28JUL2026)", file=sys.stderr)
            return 1

        if args.index_price is not None:
            index_price = args.index_price
            print(f"  [timing] index LTP: skipped (--index-price {index_price:.2f})")
        else:
            t0 = time.perf_counter()
            index_price = _fetch_index_ltp_standalone(creds, live_cfg)
            _timing("index LTP fetch", time.perf_counter() - t0)

        if args.strike is not None:
            strike = int(args.strike)
        elif index_price is not None:
            strike = _atm_strike(index_price, live_cfg.strike_step)
        else:
            print("--direct needs --strike or --index-price (or live index LTP)", file=sys.stderr)
            return 1

        expiry_norm = _normalize_expiry(args.expiry)
        t0 = time.perf_counter()
        legs = build_direct_legs(
            live_cfg,
            strike=strike,
            expiry=expiry_norm,
            lot_size=args.lot_size,
        )
        _timing("direct symbol build", time.perf_counter() - t0)

        print(f"\n--- Fast direct ---")
        print(f"  session   : {doc_source}")
        print(f"  expiry    : {expiry_norm}  strike={strike}  lot_size={args.lot_size}")
        if index_price is not None:
            print(f"  index LTP : {index_price:.2f}")
        print(f"  CE        : {legs['ce'].trading_symbol}")
        print(f"  PE        : {legs['pe'].trading_symbol}")

        orders = _entry_orders(args.direction, legs, lots)
        tag_prefix = f"test_{args.direction}"
        print(f"\n--- {args.direction} entry legs ---")
        for o in orders:
            side = "BUY" if o.side == BUY else "SELL"
            print(f"  {side} {o.leg.trading_symbol} qty={o.quantity}")

        if not args.place:
            if args.place_cancel:
                print("\n  --place-cancel: place -> wait -> cancel (square-off if MKT filled)")
            if args.round_trip:
                print("\n  --round-trip: place -> wait -> square-off")
            if args.exit_only:
                print("\n  --exit-only: square-off only (use --strike if not ATM)")
            _timing("run total (dry-run)", time.perf_counter() - t_run)
            print("\n=== DRY RUN complete. Use --place --yes to send real orders. ===")
            return 0

        if args.exit_only:
            if not args.yes:
                prompt = "Type YES to square off (reverse legs): "
                if input(prompt).strip() != "YES":
                    print("Cancelled.")
                    return 0
            exit_ok, _ = _square_off_legs(
                live_cfg,
                orders,
                doc_session=doc_session,
                tag_prefix=f"exit_{args.direction}",
            )
            if exit_ok:
                _timing("run total (exit-only)", time.perf_counter() - t_run)
                print("\n=== Square-off complete. ===")
            else:
                _timing("run total (exit-only failed)", time.perf_counter() - t_run)
                print("\n=== EXIT FAILED — position may still be open. ===", file=sys.stderr)
            return 0 if exit_ok else 2

        if args.round_trip and args.place_cancel:
            print("Use either --round-trip or --place-cancel, not both", file=sys.stderr)
            return 1

        if args.exit_only and (args.round_trip or args.place_cancel):
            print("--exit-only cannot be combined with --round-trip or --place-cancel", file=sys.stderr)
            return 1

        if args.round_trip and args.method == "both":
            print("--round-trip does not support --method both; use doc", file=sys.stderr)
            return 1

        if not args.yes:
            if args.place_cancel:
                prompt = (
                    f"Type YES to place entry, wait {args.cancel_wait_seconds}s, then cancel: "
                )
            elif args.round_trip:
                prompt = (
                    f"Type YES to place entry, wait {args.wait_seconds}s, then square off: "
                )
            else:
                prompt = "Type YES to place both MKT legs (parallel): "
            if input(prompt).strip() != "YES":
                print("Cancelled.")
                return 0

        ok, entry_refs = _place_legs(
            None,
            live_cfg,
            orders,
            method="doc",
            tag_prefix=tag_prefix,
            doc_session=doc_session,
            parallel=True,
        )

        if ok and args.place_cancel:
            order_nos = [r.order_id for r in entry_refs if r.order_id]
            if not order_nos:
                print("Place succeeded but no nOrdNo returned — cannot cancel", file=sys.stderr)
                _timing("run total (place-cancel, no ids)", time.perf_counter() - t_run)
                return 2
            print(f"\n--- Waiting {args.cancel_wait_seconds}s before cancel ---")
            time.sleep(args.cancel_wait_seconds)
            cancel_ok, any_filled = _cancel_orders(doc_session, order_nos)
            if cancel_ok:
                _timing("run total (place-cancel)", time.perf_counter() - t_run)
                print("\n=== Place + cancel complete. ===")
            elif any_filled:
                print("\n--- MKT orders filled — squaring off instead of cancel ---")
                exit_ok, _ = _square_off_legs(
                    live_cfg,
                    orders,
                    doc_session=doc_session,
                    tag_prefix=f"exit_{args.direction}",
                )
                ok = exit_ok
                if exit_ok:
                    _timing("run total (place + square-off)", time.perf_counter() - t_run)
                    print("\n=== Place + square-off complete (filled MKT exit). ===")
                else:
                    _timing("run total (square-off failed)", time.perf_counter() - t_run)
                    print("\n=== SQUARE-OFF FAILED — position may still be open. ===", file=sys.stderr)
            else:
                ok = False
                _timing("run total (cancel failed)", time.perf_counter() - t_run)
                print("\n=== CANCEL FAILED — see errors above. ===", file=sys.stderr)
        elif ok and args.round_trip:
            print(f"\n--- Waiting {args.wait_seconds}s before square-off ---")
            time.sleep(args.wait_seconds)
            exit_ok, _ = _square_off_legs(
                live_cfg,
                orders,
                doc_session=doc_session,
                tag_prefix=f"exit_{args.direction}",
            )
            ok = exit_ok
            if exit_ok:
                _timing("run total (round-trip)", time.perf_counter() - t_run)
                print("\n=== Round trip complete (entry + exit). ===")
            else:
                _timing("run total (exit failed)", time.perf_counter() - t_run)
                print("\n=== EXIT FAILED — position may still be open. ===", file=sys.stderr)
        elif ok:
            _timing("run total (entry only)", time.perf_counter() - t_run)
            print("\n=== Order(s) accepted — check Neo order book. ===")
        else:
            _timing("run total (failed)", time.perf_counter() - t_run)
            print("\n=== ORDER FAILED — see stat/emsg/errMsg/stCode above. ===", file=sys.stderr)
        return 0 if ok else 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Kotak Neo order placement (standalone)")
    parser.add_argument("--direction", choices=("long", "short"), default="long")
    parser.add_argument("--index-price", type=float, default=None)
    parser.add_argument("--lots", type=int, default=None)
    parser.add_argument("--place", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="After entry fills, wait then square off the same synthetic legs",
    )
    parser.add_argument(
        "--place-cancel",
        action="store_true",
        help="Place entry, wait, cancel; if MKT already filled, square-off instead",
    )
    parser.add_argument(
        "--exit-only",
        action="store_true",
        help="Square off only — reverse legs for --direction/--strike/--expiry (no new entry)",
    )
    parser.add_argument(
        "--cancel-wait-seconds",
        type=int,
        default=10,
        help="Seconds to wait before cancel when using --place-cancel (default 10)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=10,
        help="Seconds between entry and square-off when using --round-trip (default 10)",
    )
    parser.add_argument(
        "--method",
        choices=("doc", "sdk", "both"),
        default="doc",
        help="doc=official REST (default), sdk=SDK, both=try both",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help="token.json from generate_neo_session.py (default doc API session)",
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        default=DEFAULT_SESSION_FILE,
        help="Legacy api.txt path (with --use-session-file)",
    )
    parser.add_argument(
        "--use-login-session",
        action="store_true",
        help="Use login session for doc API instead of token.json",
    )
    parser.add_argument(
        "--use-session-file",
        action="store_true",
        help="Use api.txt for doc API instead of token.json",
    )
    parser.add_argument(
        "--dump-session",
        action="store_true",
        help="Login and write session to api.txt (legacy), then exit",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Build CE/PE trading symbols from --strike + --expiry (skip search_scrip)",
    )
    parser.add_argument(
        "--strike",
        type=int,
        default=None,
        help="Strike in index points (required with --direct unless --index-price given)",
    )
    parser.add_argument(
        "--expiry",
        default=None,
        help="Expiry e.g. 28JUL2026 (required with --direct)",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=30,
        help="Lot size per leg when using --direct (default 30 for Bank Nifty)",
    )
    parser.add_argument(
        "--skip-margin",
        action="store_true",
        help="Skip margin_required API before place (faster test)",
    )
    parser.add_argument(
        "--verify-scrip",
        action="store_true",
        help="With --direct, also call search_scrip to verify built symbols match",
    )
    parser.add_argument(
        "--cancel",
        nargs="+",
        metavar="ORDER_NO",
        help="Cancel open order(s) by nest order number (nOrdNo from place response)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config_root = load_live_config()
    live_cfg = build_live_config(config_root)
    creds = load_neo_credentials()
    if not creds.consumer_key:
        print("NEO_API_KEY required in .env", file=sys.stderr)
        return 1

    lots = args.lots if args.lots is not None else live_cfg.lots

    if args.dump_session:
        if not creds.mobile:
            print("NEO_MOBILE required in .env for --dump-session", file=sys.stderr)
            return 1
        broker = NeoBroker(creds, live_cfg)
        t_run = time.perf_counter()
        with _timed("neo login"):
            broker.login()
        try:
            session = dump_session_file(broker, args.session_file)
            print(f"Wrote fresh session to {args.session_file}")
            print(_pp(session.summary()))
            print("\nSession tokens expire. For orders, refresh token.json: python generate_neo_session.py")
            _timing("run total (dump-session)", time.perf_counter() - t_run)
            return 0
        finally:
            broker.logout()

    if _use_fast_path(args):
        return _run_fast_path(args, live_cfg, creds, lots)

    if not creds.mobile:
        print("NEO_MOBILE required in .env", file=sys.stderr)
        return 1

    broker = NeoBroker(creds, live_cfg)

    print("=== Neo order placement test (standalone — live bot unchanged) ===")
    if args.cancel:
        print(f"  mode=cancel  orders={args.cancel}")
    else:
        print(f"  product={live_cfg.product} order_type={live_cfg.order_type} lots={lots}")
        mode = "direct-symbol" if args.direct else "scrip-api"
        print(f"  direction={args.direction}  mode={mode}  {'dry-run' if not args.place else f'LIVE method={args.method}'}")
    if args.round_trip:
        print(f"  round_trip: entry -> wait {args.wait_seconds}s -> exit")

    t_run = time.perf_counter()
    with _timed("neo login"):
        broker.login()
    try:
        if args.cancel:
            doc_session, doc_source = _resolve_doc_session(
                broker,
                token_file=args.token_file,
                session_file=args.session_file,
                use_login_session=args.use_login_session,
                use_session_file=args.use_session_file,
            )
            print(f"\n--- Doc API will use: {doc_source} ---")
            print(_pp(doc_session.summary()))
            print("  Cancel order URL:", f"{doc_session.base_url.rstrip('/')}/quick/order/cancel")
            if not args.yes:
                print("\nDRY RUN — pass --yes to send cancel requests")
                for on in args.cancel:
                    print(f"  would cancel: {on}")
                return 0
            cancel_ok, _ = _cancel_orders(doc_session, args.cancel)
            print("\n=== Cancel complete ===" if cancel_ok else "\n=== Cancel had failures (order may already be filled) ===")
            return 0 if cancel_ok else 1

        if args.direct and not args.expiry:
            print("--direct requires --expiry (e.g. 28JUL2026)", file=sys.stderr)
            return 1

        with _timed("account limits"):
            limits = broker.limits()
        print("\n--- Account limits ---")
        print(_pp(limits))
        if isinstance(limits, dict) and limits.get("Net") is not None:
            print(f"  Available margin (Net): {limits['Net']}")

        print("\n--- Neo session (login) ---")
        print(_pp(_session_info(broker)))

        token_loaded = load_token_session(args.token_file)
        if token_loaded:
            tok_sess, tok_meta = token_loaded
            print(f"\n--- Neo session (token.json: {args.token_file}) ---")
            summary = tok_sess.summary()
            summary["generated_at"] = tok_meta.get("generated_at")
            summary["kType"] = tok_meta.get("kType")
            print(_pp(summary))
        else:
            print(f"\n--- token.json ---")
            print(f"  Not found or invalid: {args.token_file}")
            print("  Run: python generate_neo_session.py")

        if args.use_session_file:
            file_session = load_api_session(args.session_file)
            if file_session:
                print(f"\n--- Neo session (api.txt: {args.session_file}) ---")
                print(_pp(file_session.summary()))

        doc_session, doc_source = _resolve_doc_session(
            broker,
            token_file=args.token_file,
            session_file=args.session_file,
            use_login_session=args.use_login_session,
            use_session_file=args.use_session_file,
        )
        print(f"\n--- Doc API will use: {doc_source} ---")
        print(_pp(doc_session.summary()))
        print("  Place order URL:", _place_order_url_from_session(doc_session))

        if args.index_price is not None:
            index_price = args.index_price
            print(f"\n  [timing] index LTP: skipped (--index-price {index_price:.2f})")
        else:
            t0 = time.perf_counter()
            index_price = broker.get_index_ltp()
            _timing("index LTP fetch", time.perf_counter() - t0)

        if args.direct:
            if args.strike is not None:
                strike = int(args.strike)
            elif index_price is not None:
                strike = broker.atm_strike(index_price)
            else:
                print("--direct needs --strike or --index-price (or live index LTP)", file=sys.stderr)
                return 1
            expiry_norm = _normalize_expiry(args.expiry)
            t0 = time.perf_counter()
            legs = build_direct_legs(
                live_cfg,
                strike=strike,
                expiry=expiry_norm,
                lot_size=args.lot_size,
            )
            _timing("direct symbol build", time.perf_counter() - t0)
            print(f"\n--- Direct symbol build (no search_scrip) ---")
            print(f"  expiry={expiry_norm}  strike={strike}  lot_size={args.lot_size}")
            print(f"  CE ts={legs['ce'].trading_symbol}")
            print(f"  PE ts={legs['pe'].trading_symbol}")
            if args.verify_scrip:
                print("\n--- Scrip verify (slow) ---")
                t0 = time.perf_counter()
                verify_ok = _verify_symbols_with_scrip(broker, legs, expiry_norm)
                _timing("scrip verify", time.perf_counter() - t0)
                if not verify_ok:
                    print("Scrip verify failed — built symbol may be wrong for this expiry/strike", file=sys.stderr)
                    return 1
            if index_price is not None:
                print(f"\n--- Index ---\n  {live_cfg.index_name} LTP: {index_price:.2f}")
                _strike_audit(broker, legs, index_price)
        else:
            if index_price is None:
                print("Could not fetch index LTP; pass --index-price", file=sys.stderr)
                return 1

            print(f"\n--- Index ---\n  {live_cfg.index_name} LTP: {index_price:.2f}")

            t0 = time.perf_counter()
            legs = broker.resolve_atm_legs(index_price, broker.nearest_expiry())
            _timing("scrip resolve_atm_legs", time.perf_counter() - t0)
            if not _strike_audit(broker, legs, index_price):
                return 1

        orders = _entry_orders(args.direction, legs, lots)
        print(f"\n--- Synthetic {args.direction} entry legs ---")
        for o in orders:
            side = "BUY" if o.side == BUY else "SELL"
            print(f"  {side} {o.leg.trading_symbol} qty={o.quantity} token={o.leg.token or '(not used for place)'}")

        print("\n--- Place Order jData preview ---")
        tag_prefix = f"test_{args.direction}"
        for o in orders:
            side = "BUY" if o.side == BUY else "SELL"
            tag = _unique_order_tag(tag_prefix, o.leg)
            print(f"  {side} (ig={tag}): {_pp(build_place_order_jdata(live_cfg, o.leg, o.side, o.quantity, tag=tag))}")

        if args.skip_margin:
            print("\n  check_margin_for_legs: SKIPPED (--skip-margin)")
            margin_ok = True
        else:
            t0 = time.perf_counter()
            margin_ok = check_margin_for_legs(broker, orders)
            _timing("margin check", time.perf_counter() - t0)
            print(f"\n  check_margin_for_legs: {'OK' if margin_ok else 'FAILED'}")

        if args.round_trip:
            exit_orders = _exit_orders_from_entry(orders)
            print(f"\n--- Synthetic {args.direction} exit legs (square off) ---")
            for o in exit_orders:
                side = "BUY" if o.side == BUY else "SELL"
                print(f"  {side} {o.leg.trading_symbol} qty={o.quantity} token={o.leg.token}")

        if not args.place:
            if args.place_cancel:
                print("\n  --place-cancel: place -> wait -> cancel (square-off if MKT filled)")
            if args.round_trip:
                print("\n  --round-trip: place -> wait -> square-off")
            if args.exit_only:
                print("\n  --exit-only: square-off only")
            _timing("run total (dry-run)", time.perf_counter() - t_run)
            print("\n=== DRY RUN complete. Use --place --yes to send real orders. ===")
            return 0

        if args.exit_only:
            if not args.yes:
                if input("Type YES to square off (reverse legs): ").strip() != "YES":
                    print("Cancelled.")
                    return 0
            exit_ok, _ = _square_off_legs(
                live_cfg,
                orders,
                doc_session=doc_session,
                tag_prefix=f"exit_{args.direction}",
            )
            if exit_ok:
                _timing("run total (exit-only)", time.perf_counter() - t_run)
                print("\n=== Square-off complete. ===")
            else:
                _timing("run total (exit-only failed)", time.perf_counter() - t_run)
                print("\n=== EXIT FAILED — position may still be open. ===", file=sys.stderr)
            return 0 if exit_ok else 2

        if not margin_ok and not args.skip_margin:
            print("Margin check failed; use --skip-margin to force place in test", file=sys.stderr)
            return 1

        if args.round_trip and args.place_cancel:
            print("Use either --round-trip or --place-cancel, not both", file=sys.stderr)
            return 1

        if args.round_trip and args.method == "both":
            print("--round-trip does not support --method both; use doc or sdk", file=sys.stderr)
            return 1

        if not args.yes:
            if args.place_cancel:
                prompt = (
                    f"Type YES to place entry, wait {args.cancel_wait_seconds}s, then cancel: "
                )
            elif args.round_trip:
                prompt = (
                    f"Type YES to place entry, wait {args.wait_seconds}s, then square off: "
                )
            else:
                prompt = "Type YES to place both MKT legs: "
            if input(prompt).strip() != "YES":
                print("Cancelled.")
                return 0

        entry_refs: List[OrderRef] = []
        if args.method == "both":
            ok_doc, refs_doc = _place_legs(
                broker, live_cfg, orders, method="doc", tag_prefix=tag_prefix, doc_session=doc_session
            )
            ok_sdk, _ = _place_legs(
                broker, live_cfg, orders, method="sdk", tag_prefix=tag_prefix, doc_session=doc_session
            )
            ok = ok_doc or ok_sdk
            entry_refs = refs_doc
            print(f"\n  doc: {'OK' if ok_doc else 'FAIL'} | sdk: {'OK' if ok_sdk else 'FAIL'}")
        else:
            ok, entry_refs = _place_legs(
                broker,
                live_cfg,
                orders,
                method=args.method,
                tag_prefix=tag_prefix,
                doc_session=doc_session,
                parallel=args.direct and args.method == "doc",
            )

        if ok and args.place_cancel:
            order_nos = [r.order_id for r in entry_refs if r.order_id]
            if not order_nos:
                print("Place succeeded but no nOrdNo returned — cannot cancel", file=sys.stderr)
                _timing("run total (place-cancel, no ids)", time.perf_counter() - t_run)
                return 2
            print(f"\n--- Waiting {args.cancel_wait_seconds}s before cancel ---")
            time.sleep(args.cancel_wait_seconds)
            cancel_ok, any_filled = _cancel_orders(doc_session, order_nos)
            if cancel_ok:
                _timing("run total (place-cancel)", time.perf_counter() - t_run)
                print("\n=== Place + cancel complete. ===")
            elif any_filled:
                print("\n--- MKT orders filled — squaring off instead of cancel ---")
                exit_ok, _ = _square_off_legs(
                    live_cfg,
                    orders,
                    doc_session=doc_session,
                    tag_prefix=f"exit_{args.direction}",
                )
                ok = exit_ok
                if exit_ok:
                    _timing("run total (place + square-off)", time.perf_counter() - t_run)
                    print("\n=== Place + square-off complete (filled MKT exit). ===")
                else:
                    _timing("run total (square-off failed)", time.perf_counter() - t_run)
                    print("\n=== SQUARE-OFF FAILED — position may still be open. ===", file=sys.stderr)
            else:
                ok = False
                _timing("run total (cancel failed)", time.perf_counter() - t_run)
                print("\n=== CANCEL FAILED — see errors above. ===", file=sys.stderr)
        elif ok and args.round_trip:
            print(f"\n--- Waiting {args.wait_seconds}s before square-off ---")
            time.sleep(args.wait_seconds)
            exit_ok, _ = _square_off_legs(
                live_cfg,
                orders,
                doc_session=doc_session,
                tag_prefix=f"exit_{args.direction}",
            )
            ok = exit_ok
            if exit_ok:
                _timing("run total (round-trip)", time.perf_counter() - t_run)
                print("\n=== Round trip complete (entry + exit). ===")
            else:
                _timing("run total (exit failed)", time.perf_counter() - t_run)
                print("\n=== EXIT FAILED — position may still be open. ===", file=sys.stderr)
        elif ok:
            _timing("run total (entry only)", time.perf_counter() - t_run)
            print("\n=== Order(s) accepted — check Neo order book. ===")
        else:
            _timing("run total (failed)", time.perf_counter() - t_run)
            print("\n=== ORDER FAILED — see stat/emsg/errMsg/stCode above. ===", file=sys.stderr)
        return 0 if ok else 2
    finally:
        broker.logout()


if __name__ == "__main__":
    raise SystemExit(main())
