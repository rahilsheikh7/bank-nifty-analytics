"""Fast local BANKNIFTY option symbol resolution for live orders."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Set

from live.config import BASE_DIR, LiveConfig
from live.neo_client import Leg

try:
    from zoneinfo import ZoneInfo

    _IST = ZoneInfo("Asia/Kolkata")
except ImportError:  # pragma: no cover
    _IST = timezone.utc

_MONTHS = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


@dataclass(frozen=True)
class DirectExpiryInfo:
    expiry_date: date
    expiry: str


def resolve_symbol_csv_path(config_path: str) -> Path:
    """Resolve configured CSV path relative to the project root."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def format_expiry_for_neo(expiry_date: date) -> str:
    """Return DDMMMYYYY, e.g. 28JUL2026."""
    return f"{expiry_date.day:02d}{_MONTHS[expiry_date.month]}{expiry_date.year}"


def build_option_symbol(
    underlying: str,
    expiry_date: date,
    strike: int,
    option_type: str,
) -> str:
    """Build Neo trading symbol, e.g. BANKNIFTY26JUL57300CE."""
    ot = str(option_type).strip().upper()
    if ot not in {"CE", "PE"}:
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")
    yy = str(expiry_date.year)[-2:]
    return f"{underlying.strip().upper()}{yy}{_MONTHS[expiry_date.month]}{int(strike)}{ot}"


class DirectSymbolResolver:
    """Resolve ATM CE/PE legs from the slim local symbol/expiry CSV."""

    def __init__(self, live_cfg: LiveConfig, *, csv_path: Optional[Path] = None) -> None:
        self.live_cfg = live_cfg
        self.csv_path = csv_path or resolve_symbol_csv_path(live_cfg.scrip_symbol_expiry_csv)
        self._symbols_by_expiry: Dict[date, Set[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Fast direct symbol CSV not found: {self.csv_path}. "
                "Run: python -m live.download_scrip_master --segments nse_fo --underlying BANKNIFTY"
            )

        symbols_by_expiry: Dict[date, Set[str]] = {}
        with self.csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = str(row.get("symbol") or "").strip().upper()
                raw_expiry = str(row.get("expiry_date") or "").strip()
                if not symbol or not raw_expiry:
                    continue
                try:
                    expiry_date = datetime.strptime(raw_expiry, "%d-%m-%Y").date()
                except ValueError:
                    continue
                symbols_by_expiry.setdefault(expiry_date, set()).add(symbol)

        if not symbols_by_expiry:
            raise RuntimeError(f"No symbol/expiry rows loaded from {self.csv_path}")
        self._symbols_by_expiry = symbols_by_expiry

    @property
    def expiries(self) -> list[date]:
        return sorted(self._symbols_by_expiry)

    @property
    def symbol_count(self) -> int:
        return sum(len(symbols) for symbols in self._symbols_by_expiry.values())

    def selected_expiry(self, *, today: Optional[date] = None) -> DirectExpiryInfo:
        today = today or datetime.now(_IST).date()
        cutoff = today + timedelta(days=max(0, self.live_cfg.expiry_roll_days))
        future = [expiry for expiry in self.expiries if expiry >= cutoff]
        if not future:
            raise RuntimeError(
                f"No future {self.live_cfg.underlying_symbol} expiries in {self.csv_path} "
                f"on/after {cutoff.isoformat()}"
            )
        expiry_date = future[0]
        return DirectExpiryInfo(
            expiry_date=expiry_date,
            expiry=format_expiry_for_neo(expiry_date),
        )

    def resolve_legs(
        self,
        *,
        strike: int,
        expiry_info: Optional[DirectExpiryInfo] = None,
    ) -> Dict[str, Leg]:
        expiry_info = expiry_info or self.selected_expiry()
        ce_symbol = build_option_symbol(
            self.live_cfg.underlying_symbol,
            expiry_info.expiry_date,
            strike,
            "CE",
        )
        pe_symbol = build_option_symbol(
            self.live_cfg.underlying_symbol,
            expiry_info.expiry_date,
            strike,
            "PE",
        )
        listed = self._symbols_by_expiry.get(expiry_info.expiry_date, set())
        missing = [symbol for symbol in (ce_symbol, pe_symbol) if symbol not in listed]
        if missing:
            raise RuntimeError(
                f"Direct symbols not listed in {self.csv_path.name} for expiry "
                f"{expiry_info.expiry}: {', '.join(missing)}"
            )

        lot_size = int(self.live_cfg.fast_direct_lot_size)
        if lot_size <= 0:
            raise RuntimeError(f"Invalid fast_direct_lot_size={self.live_cfg.fast_direct_lot_size}")

        segment = self.live_cfg.option_segment
        return {
            "ce": Leg(
                trading_symbol=ce_symbol,
                token="",
                lot_size=lot_size,
                strike=float(strike),
                expiry=expiry_info.expiry,
                option_type="CE",
                segment=segment,
            ),
            "pe": Leg(
                trading_symbol=pe_symbol,
                token="",
                lot_size=lot_size,
                strike=float(strike),
                expiry=expiry_info.expiry,
                option_type="PE",
                segment=segment,
            ),
        }
