"""Kotak Neo documented REST place-order API (used by live bot and test script)."""
from __future__ import annotations

import json
import logging
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import requests

from live.config import LiveConfig

logger = logging.getLogger("live.neo")

NEO_FIN_KEY = "neotradeapi"
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parent.parent / "token.json"


@dataclass
class ApiSession:
    auth_token: str
    sid: str
    rid: str = ""
    base_url: str = ""
    server_id: str = ""
    data_center: str = ""
    ucc: str = ""


@contextmanager
def ipv6_only() -> Iterator[None]:
    """Temporarily force IPv6 only for the REST order request."""
    orig_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv6(host, port, family=0, type=0, proto=0, flags=0):
        return orig_getaddrinfo(host, port, socket.AF_INET6, type, proto, flags)

    socket.getaddrinfo = getaddrinfo_ipv6  # type: ignore[assignment]
    urllib3_restore = None
    try:
        import urllib3.util.connection as urllib3_connection

        urllib3_restore = urllib3_connection.allowed_gai_family
        urllib3_connection.allowed_gai_family = lambda: socket.AF_INET6
    except ImportError:
        pass
    try:
        yield
    finally:
        socket.getaddrinfo = orig_getaddrinfo
        if urllib3_restore is not None:
            import urllib3.util.connection as urllib3_connection

            urllib3_connection.allowed_gai_family = urllib3_restore


def load_token_session(path: Path = DEFAULT_TOKEN_FILE) -> Optional[tuple[ApiSession, Dict[str, Any]]]:
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


def session_from_client(client: Any) -> ApiSession:
    cfg = client.configuration
    return ApiSession(
        auth_token=str(getattr(cfg, "edit_token", None) or ""),
        sid=str(getattr(cfg, "edit_sid", None) or ""),
        rid=str(getattr(cfg, "edit_rid", None) or ""),
        base_url=str(getattr(cfg, "base_url", None) or "").rstrip("/"),
        server_id=str(getattr(cfg, "serverId", None) or ""),
        data_center=str(getattr(cfg, "data_center", None) or ""),
        ucc="",
    )


def is_session_expired_error(resp: Dict[str, Any]) -> bool:
    """Detect Kotak session token expiry / invalid session on place-order."""
    st_code = resp.get("stCode")
    try:
        if int(st_code) in {100022}:
            return True
    except (TypeError, ValueError):
        pass
    text = " ".join(
        str(resp.get(k, "")) for k in ("stat", "emsg", "errMsg", "message")
    ).lower()
    phrases = (
        "invalid session",
        "session token",
        "session expired",
        "token expired",
        "session invalid",
    )
    return any(p in text for p in phrases)


def resolve_doc_session(
    client: Any,
    *,
    token_file: Path = DEFAULT_TOKEN_FILE,
    prefer_login: bool = False,
) -> tuple[ApiSession, str]:
    """Pick session for doc place-order API. Bot uses token.json from tradeApiValidate."""
    token_loaded = load_token_session(token_file)
    if token_loaded:
        session, meta = token_loaded
        when = meta.get("generated_at") or "unknown time"
        return session, f"token.json (generated {when})"

    if prefer_login and client is not None:
        login_session = session_from_client(client)
        if login_session.auth_token and login_session.sid and login_session.base_url:
            return login_session, "login"

    login_session = session_from_client(client)
    if login_session.auth_token and login_session.sid and login_session.base_url:
        return login_session, "login (fallback)"

    raise RuntimeError(
        "No Neo trade session for place order; run python generate_neo_session.py"
    )


def unique_order_tag(prefix: str, leg: Any) -> str:
    """Unique jData ig per leg — Kotak rejects duplicate client order ids."""
    option_type = str(getattr(leg, "option_type", "X") or "X")
    ts = int(time.time() * 1000) % 1_000_000
    tag = f"{prefix}_{option_type}_{ts}"
    return tag[:20]


def _normalize_order_type(order_type: str) -> str:
    key = str(order_type or "MKT").strip().upper()
    return {"MKT": "MKT", "MARKET": "MKT", "L": "L", "LIMIT": "L"}.get(key, key)


def build_place_order_jdata(
    live_cfg: LiveConfig,
    leg: Any,
    side: str,
    quantity: int,
    *,
    tag: Optional[str] = None,
) -> Dict[str, str]:
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


def _place_order_headers(session: ApiSession) -> Dict[str, str]:
    if not session.auth_token or not session.sid:
        raise RuntimeError("Neo session missing Auth token or Sid")
    return {
        "accept": "application/json",
        "Sid": session.sid,
        "Auth": session.auth_token,
        "neo-fin-key": NEO_FIN_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
    }


def place_order_doc(
    live_cfg: LiveConfig,
    leg: Any,
    side: str,
    quantity: int,
    session: ApiSession,
    *,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """POST {baseUrl}/quick/order/rule/ms/place per Kotak documentation."""
    jdata = build_place_order_jdata(live_cfg, leg, side, quantity, tag=tag)
    url = f"{session.base_url.rstrip('/')}/quick/order/rule/ms/place"
    headers = _place_order_headers(session)
    body = {"jData": json.dumps(jdata, separators=(",", ":"))}
    with ipv6_only():
        response = requests.post(url, headers=headers, data=body, timeout=30)
    try:
        payload: Dict[str, Any] = response.json()
    except ValueError:
        payload = {"stat": "Not_Ok", "emsg": response.text, "stCode": response.status_code}
    if not isinstance(payload, dict):
        payload = {"stat": "Not_Ok", "emsg": str(payload), "stCode": response.status_code}
    payload["_http_status"] = response.status_code
    return payload


def is_place_order_ok(resp: Dict[str, Any]) -> bool:
    status_l = str(resp.get("stat") or resp.get("status") or "").lower()
    if status_l in {"not_ok", "not ok", "rejected", "error", "unknown"}:
        return False
    if resp.get("emsg") or resp.get("errMsg"):
        return False
    for key in ("nOrdNo", "orderId", "order_id", "ordNo"):
        if resp.get(key):
            return True
    data = resp.get("data")
    if isinstance(data, dict):
        for key in ("nOrdNo", "orderId", "order_id", "ordNo"):
            if data.get(key):
                return True
    return status_l == "ok"


def build_cancel_order_jdata(order_no: str, *, amo: str = "NO") -> Dict[str, str]:
    return {"am": amo, "on": str(order_no)}


def cancel_order_doc(session: ApiSession, order_no: str, *, amo: str = "NO") -> Dict[str, Any]:
    """POST {baseUrl}/quick/order/cancel per Kotak documentation."""
    jdata = build_cancel_order_jdata(order_no, amo=amo)
    url = f"{session.base_url.rstrip('/')}/quick/order/cancel"
    headers = _place_order_headers(session)
    body = {"jData": json.dumps(jdata, separators=(",", ":"))}
    with ipv6_only():
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
        "headers": {**headers, "Auth": "(hidden)"},
    }
    return payload


def is_cancel_order_ok(resp: Dict[str, Any]) -> bool:
    status_l = str(resp.get("stat") or resp.get("status") or "").lower()
    if status_l in {"not_ok", "not ok", "rejected", "error", "unknown"}:
        return False
    if resp.get("emsg") or resp.get("errMsg"):
        return False
    st_code = resp.get("stCode")
    if st_code is not None and int(st_code) not in (200,):
        return False
    return status_l == "ok" or bool(resp.get("nOrdNo"))
