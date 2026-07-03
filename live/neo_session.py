"""Kotak Neo trade session (tradeApiLogin + tradeApiValidate) for token.json."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

from live.config import NeoCredentials
from live.neo_place_order import DEFAULT_TOKEN_FILE, ApiSession, load_token_session

logger = logging.getLogger("live.neo")

LOGIN_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiLogin"
VALIDATE_URL = "https://mis.kotaksecurities.com/login/1.0/tradeApiValidate"
NEO_FIN_KEY = "neotradeapi"
try:
    from zoneinfo import ZoneInfo

    _IST = ZoneInfo("Asia/Kolkata")
except ImportError:  # pragma: no cover
    _IST = timezone.utc


def _login_headers(authorization: str, *, sid: str = "", auth: str = "") -> Dict[str, str]:
    headers = {
        "Authorization": authorization,
        "neo-fin-key": NEO_FIN_KEY,
        "Content-Type": "application/json",
    }
    if sid:
        headers["sid"] = sid
    if auth:
        headers["Auth"] = auth
    return headers


def resolve_totp(
    creds: NeoCredentials,
    totp_provider: Optional[Callable[[], str]] = None,
) -> str:
    if totp_provider is not None:
        totp = totp_provider().strip()
        if totp:
            return totp
    if creds.has_totp_secret:
        try:
            import pyotp

            return pyotp.TOTP(creds.totp_secret).now()
        except ImportError as exc:
            raise RuntimeError("pyotp required for automated TOTP (NEO_TOTP_SECRET)") from exc
    raise RuntimeError(
        "Neo TOTP required: set NEO_TOTP_SECRET in .env or provide totp_provider"
    )


def trade_api_login(creds: NeoCredentials, totp: str) -> Dict[str, Any]:
    body = {"mobileNumber": creds.mobile, "ucc": creds.client_code, "totp": totp}
    resp = requests.post(
        LOGIN_URL,
        headers=_login_headers(creds.consumer_key),
        json=body,
        timeout=30,
    )
    try:
        payload = resp.json()
    except ValueError:
        payload = {"status": "error", "message": resp.text, "http_status": resp.status_code}
    if resp.status_code >= 400:
        raise RuntimeError(f"tradeApiLogin HTTP {resp.status_code}: {payload}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not data.get("token") or not data.get("sid"):
        raise RuntimeError(f"tradeApiLogin missing token/sid: {payload}")
    return payload


def trade_api_validate(creds: NeoCredentials, view_token: str, view_sid: str) -> Dict[str, Any]:
    body = {"mpin": creds.mpin}
    resp = requests.post(
        VALIDATE_URL,
        headers=_login_headers(creds.consumer_key, sid=view_sid, auth=view_token),
        json=body,
        timeout=30,
    )
    try:
        payload = resp.json()
    except ValueError:
        payload = {"status": "error", "message": resp.text, "http_status": resp.status_code}
    if resp.status_code >= 400:
        raise RuntimeError(f"tradeApiValidate HTTP {resp.status_code}: {payload}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not data.get("token") or not data.get("sid"):
        raise RuntimeError(f"tradeApiValidate missing trade token/sid: {payload}")
    return payload


def build_token_payload(login_resp: Dict[str, Any], validate_resp: Dict[str, Any]) -> Dict[str, Any]:
    trade_data = validate_resp["data"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "login": login_resp,
        "validate": validate_resp,
        "session": {
            "auth_token": trade_data.get("token"),
            "sid": trade_data.get("sid"),
            "rid": trade_data.get("rid"),
            "baseUrl": trade_data.get("baseUrl"),
            "hsServerId": trade_data.get("hsServerId"),
            "dataCenter": trade_data.get("dataCenter"),
            "ucc": trade_data.get("ucc"),
            "kType": trade_data.get("kType"),
        },
    }


def save_token_file(payload: Dict[str, Any], path: Path = DEFAULT_TOKEN_FILE) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def token_generated_at(path: Path = DEFAULT_TOKEN_FILE) -> Optional[datetime]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("generated_at")
        if not raw:
            return None
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None


def is_token_stale(path: Path = DEFAULT_TOKEN_FILE) -> bool:
    """True if token.json is missing or was generated before today (IST trading day)."""
    generated = token_generated_at(path)
    if generated is None:
        return True
    gen_ist = generated.astimezone(_IST)
    now_ist = datetime.now(_IST)
    return gen_ist.date() < now_ist.date()


def generate_trade_session(
    creds: NeoCredentials,
    *,
    totp_provider: Optional[Callable[[], str]] = None,
    token_file: Path = DEFAULT_TOKEN_FILE,
) -> ApiSession:
    """Run tradeApiLogin + tradeApiValidate and write token.json."""
    missing = [
        name
        for name, val in [
            ("NEO_API_KEY", creds.consumer_key),
            ("NEO_MOBILE", creds.mobile),
            ("NEO_CLIENT_CODE", creds.client_code),
            ("NEO_MPIN", creds.mpin),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing Neo credentials: {', '.join(missing)}")

    totp = resolve_totp(creds, totp_provider)
    login_resp = trade_api_login(creds, totp)
    view_data = login_resp["data"]
    validate_resp = trade_api_validate(creds, view_data["token"], view_data["sid"])
    payload = build_token_payload(login_resp, validate_resp)
    save_token_file(payload, token_file)
    loaded = load_token_session(token_file)
    if not loaded:
        raise RuntimeError(f"Failed to load session after writing {token_file}")
    session, meta = loaded
    logger.info(
        "Neo trade session generated (kType=%s, baseUrl=%s, generated_at=%s)",
        meta.get("kType"),
        session.base_url,
        meta.get("generated_at"),
    )
    return session


def ensure_trade_session(
    creds: NeoCredentials,
    *,
    token_file: Path = DEFAULT_TOKEN_FILE,
    force: bool = False,
    totp_provider: Optional[Callable[[], str]] = None,
) -> ApiSession:
    """Return today's trade session; regenerate token.json if missing or stale."""
    if not force and not is_token_stale(token_file):
        loaded = load_token_session(token_file)
        if loaded:
            session, meta = loaded
            logger.info("Using existing trade session from %s (generated %s)", token_file, meta.get("generated_at"))
            return session
    logger.info("Refreshing Neo trade session (%s)", "forced" if force else "stale or missing")
    return generate_trade_session(creds, totp_provider=totp_provider, token_file=token_file)
