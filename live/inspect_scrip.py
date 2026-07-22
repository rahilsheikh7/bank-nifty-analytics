"""Inspect Kotak Neo ``search_scrip`` response for Bank Nifty options.

Shows the instrument-master fields the live bot actually uses so you can
decide what is safe to cache (symbol, token, lot size, strike, expiry).

Usage:
  python -m live.inspect_scrip
  python -m live.inspect_scrip --strike 57900 --expiry 28JUL2026
  python -m live.inspect_scrip --option-type BOTH --strike 57900 --expiry 28JUL2026
  python -m live.inspect_scrip --strike 57900 --expiry 28JUL2026 --raw
  python -m live.inspect_scrip --atm --band 300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from live.config import build_live_config, load_live_config, load_neo_credentials
from live.neo_client import NeoBroker, make_broker


def _pp(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only fields consume by NeoBroker / place-order."""
    trading_symbol = row.get("pTrdSymbol") or row.get("pSymbolName")
    token = row.get("pSymbol") or row.get("pScripRefKey")
    lot_size = row.get("lLotSize") or row.get("iLotSize")
    strike = row.get("dStrikePrice") or row.get("dStrikePrice;")
    expiry = row.get("pExpiryDate") or row.get("lExpiryDate")
    return {
        "trading_symbol": trading_symbol,
        "token": token,
        "lot_size": lot_size,
        "strike_raw": strike,
        "expiry_raw": expiry,
        "option_type": row.get("pOptionType") or row.get("option_type"),
    }


def _print_rows(rows: List[Dict[str, Any]], *, limit: int, raw: bool) -> None:
    print(f"rows={len(rows)}")
    if not rows:
        return
    if raw:
        print(_pp(rows[:limit]))
        if len(rows) > limit:
            print(f"... {len(rows) - limit} more rows omitted (--limit {limit})")
        return

    print(
        "bot fields: trading_symbol | token | lot_size | strike_raw | expiry_raw"
    )
    for i, row in enumerate(rows[:limit]):
        s = _row_summary(row)
        print(
            f"[{i}] {s['trading_symbol']}  token={s['token']}  "
            f"lot={s['lot_size']}  strike={s['strike_raw']}  expiry={s['expiry_raw']}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows (--limit {limit})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump Neo search_scrip for BANKNIFTY options (inspect / cache planning)",
    )
    parser.add_argument(
        "--option-type",
        default="CE",
        choices=["CE", "PE", "BOTH"],
        help="Option type to search (default CE)",
    )
    parser.add_argument(
        "--strike",
        type=int,
        default=None,
        help="Strike in index points (omit = all / broad search)",
    )
    parser.add_argument(
        "--expiry",
        default="",
        help="Expiry string e.g. 28JUL2026 (omit = all expiries in response)",
    )
    parser.add_argument(
        "--atm",
        action="store_true",
        help="Resolve ATM strike from index LTP (overrides --strike if LTP available)",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=0,
        help="With --atm or --strike, also dump CE+PE for strikes ATM±band (step 100)",
    )
    parser.add_argument("--limit", type=int, default=30, help="Max rows to print per search")
    parser.add_argument("--raw", action="store_true", help="Print full raw JSON rows")
    parser.add_argument(
        "--keys",
        action="store_true",
        help="Print union of all keys seen in the first page of rows",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_root = load_live_config()
    live_cfg = build_live_config(config_root)
    creds = load_neo_credentials()
    if not creds.consumer_key or not creds.mobile:
        print("NEO_API_KEY / NEO_MOBILE missing in .env", file=sys.stderr)
        return 1

    broker: NeoBroker = make_broker(creds, live_cfg)
    broker.login()

    strike: Optional[int] = args.strike
    if args.atm:
        ltp = broker.get_index_ltp()
        if ltp is None:
            print("Could not fetch index LTP for --atm", file=sys.stderr)
            broker.logout()
            return 1
        strike = broker.atm_strike(ltp)
        print(f"index LTP={ltp:.2f} -> ATM strike={strike} (step={live_cfg.strike_step})")

    expiry = args.expiry
    if not expiry and (args.atm or strike is not None):
        try:
            expiry = broker.nearest_expiry()
            print(f"using nearest_expiry() -> {expiry}")
        except Exception as exc:
            print(f"nearest_expiry failed ({exc}); searching without expiry filter")

    types = ["CE", "PE"] if args.option_type == "BOTH" else [args.option_type]
    strikes: List[Optional[int]]
    if strike is not None and args.band > 0:
        step = live_cfg.strike_step
        band = abs(args.band)
        strikes = list(range(strike - band, strike + band + 1, step))
        print(f"strike band: {strikes[0]} .. {strikes[-1]} step={step}")
    else:
        strikes = [strike]

    try:
        for ot in types:
            for st in strikes:
                print(
                    f"\n=== search_scrip underlying={live_cfg.underlying_symbol} "
                    f"option_type={ot} strike={st} expiry={expiry!r} ==="
                )
                rows = broker._search(ot, strike=st, expiry=expiry or "")
                if args.keys and rows:
                    keys: set[str] = set()
                    for row in rows[: min(len(rows), args.limit)]:
                        if isinstance(row, dict):
                            keys.update(row.keys())
                    print("keys:", ", ".join(sorted(keys)))
                _print_rows(rows, limit=args.limit, raw=args.raw)
    finally:
        try:
            broker.logout()
        except Exception:
            pass

    print(
        "\nTip: cache trading_symbol + token + lot_size per (expiry, strike, CE/PE). "
        "Refresh daily or when monthly expiry rolls — do not call this on every entry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
