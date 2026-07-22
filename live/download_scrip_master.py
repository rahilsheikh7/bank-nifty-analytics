"""Download Kotak Neo scrip master CSVs for local symbol lookups.

API: GET {base}/script-details/1.0/masterscrip/file-paths
Auth: NEO_API_KEY (plain Authorization header)

Usage:
  python -m live.download_scrip_master
  python -m live.download_scrip_master --segments nse_fo
  python -m live.download_scrip_master --segments nse_fo --underlying BANKNIFTY
  python -m live.download_scrip_master --segments nse_fo --underlying BANKNIFTY
  python -m live.download_scrip_master --extract-only --underlying BANKNIFTY --segment nse_fo
  python -m live.download_scrip_master --lookup --underlying BANKNIFTY --strike 57300 --expiry 28JUL2026
  live/scripts/daily_scrip_master_refresh.sh   # cron @ 08:00 IST
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse

import requests

from live.config import build_live_config, load_live_config, load_neo_credentials

logger = logging.getLogger("live.scrip_master")

MASTER_URL = "https://mis.kotaksecurities.com/script-details/1.0/masterscrip/file-paths"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "scrip_master"
MANIFEST_NAME = "manifest.json"
NSE_FO_EXPIRY_OFFSET = 315511200
_EXPIRY_RE = re.compile(r"^(\d{1,2})([A-Za-z]{3})(\d{4})$")

try:
    from zoneinfo import ZoneInfo

    _IST = ZoneInfo("Asia/Kolkata")
except ImportError:  # pragma: no cover
    _IST = timezone.utc


def _segment_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    if name.endswith(".csv"):
        name = name[: -len(".csv")]
    return name.lower()


def fetch_file_paths(consumer_key: str, *, timeout: float = 30.0) -> Dict[str, Any]:
    """Call scrip master file-paths API. Returns parsed JSON data block."""
    if not consumer_key:
        raise RuntimeError("NEO_API_KEY missing in .env")
    headers = {"Authorization": consumer_key}
    resp = requests.get(MASTER_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected scrip master response: {payload!r}")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        raise RuntimeError(f"No data in scrip master response: {payload!r}")
    return data


def _normalize_expiry_str(expiry: str) -> str:
    raw = str(expiry or "").strip().upper().replace(" ", "")
    m = _EXPIRY_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid expiry {expiry!r}; use e.g. 28JUL2026")
    day, mon, year = m.groups()
    return f"{int(day):02d}{mon}{year}"


def _expiry_str_to_date(expiry: str) -> date:
    norm = _normalize_expiry_str(expiry)
    m = _EXPIRY_RE.match(norm)
    assert m is not None
    day, mon, year = m.groups()
    return datetime.strptime(f"{day}{mon}{year}", "%d%b%Y").date()


def _row_expiry_date(row: Dict[str, str], segment: str) -> Optional[date]:
    raw = row.get("lexpirydate") or row.get("pexpirydate") or ""
    if not raw:
        return None
    try:
        epoch = int(float(raw))
    except (TypeError, ValueError):
        return None
    seg = segment.lower()
    if seg in {"nse_fo", "cde_fo"}:
        epoch += NSE_FO_EXPIRY_OFFSET
    try:
        return datetime.fromtimestamp(epoch, tz=_IST).date()
    except (OSError, OverflowError, ValueError):
        return None


def _row_get(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        val = row.get(key.lower())
        if val not in (None, ""):
            return str(val).strip()
    return ""


def _iter_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                str(k).strip().lower(): str(v).strip()
                for k, v in row.items()
                if k is not None
            }


def download_scrip_master(
    consumer_key: str,
    cache_dir: Path,
    *,
    segments: Optional[List[str]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Download scrip master CSVs into cache_dir. Returns manifest dict."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = fetch_file_paths(consumer_key, timeout=timeout)
    urls = data.get("filesPaths") or data.get("filespaths") or []
    if not isinstance(urls, list):
        raise RuntimeError(f"filesPaths missing in response: {data!r}")

    want = {s.lower() for s in segments} if segments else None
    files_meta: List[Dict[str, Any]] = []
    downloaded = 0

    for url in urls:
        if not isinstance(url, str) or not url.strip():
            continue
        segment = _segment_from_url(url)
        if want is not None and segment not in want:
            continue
        dest = cache_dir / f"{segment}.csv"
        logger.info("Downloading %s -> %s", segment, dest.name)
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        dest.write_bytes(r.content)
        files_meta.append(
            {
                "segment": segment,
                "url": url,
                "file": dest.name,
                "bytes": dest.stat().st_size,
            }
        )
        downloaded += 1

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "base_folder": data.get("baseFolder") or data.get("basefolder"),
        "api_url": MASTER_URL,
        "cache_dir": str(cache_dir),
        "file_count": downloaded,
        "files": files_meta,
    }
    (cache_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    path = cache_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _filtered_csv_name(underlying: str, segment: str) -> str:
    return f"{underlying.strip().lower()}_{segment.lower()}.csv"


def _symbol_expiry_csv_name(underlying: str) -> str:
    return f"{underlying.strip().lower()}_symbol_expiry.csv"


def _format_expiry_dd_mm_yyyy(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _csv_path_for_segment(
    cache_dir: Path,
    segment: str,
    *,
    underlying: Optional[str] = None,
) -> Path:
    if underlying:
        filtered = cache_dir / _filtered_csv_name(underlying, segment)
        if filtered.is_file():
            return filtered
    return cache_dir / f"{segment.lower()}.csv"


def _row_matches_underlying(row: Dict[str, str], underlying: str) -> bool:
    und = underlying.strip().upper()
    sym = _row_get(row, "psymbolname", "psymbol").upper()
    trd = _row_get(row, "ptrdsymbol").upper()
    inst = _row_get(row, "pinstname", "pinsttype").upper()
    return sym == und or trd.startswith(und) or und in sym or und in inst


def extract_underlying_csv(
    source_csv: Path,
    dest_csv: Path,
    underlying: str,
) -> int:
    """Filter segment CSV to one underlying (e.g. BANKNIFTY). Returns row count."""
    if not source_csv.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    written = 0
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with source_csv.open(newline="", encoding="utf-8", errors="replace") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty or invalid CSV: {source_csv}")
        fieldnames = list(reader.fieldnames)
        with dest_csv.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if _row_matches_underlying(
                    {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k},
                    underlying,
                ):
                    writer.writerow(row)
                    written += 1
    return written


def extract_symbol_expiry_csv(
    source_csv: Path,
    dest_csv: Path,
    underlying: str,
    *,
    segment: str = "nse_fo",
) -> int:
    """Filter to underlying; write symbol + expiry_date (DD-MM-YYYY) only."""
    if not source_csv.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    rows_out: List[Dict[str, str]] = []
    for row in _iter_csv_rows(source_csv):
        if not _row_matches_underlying(row, underlying):
            continue
        symbol = _row_get(row, "ptrdsymbol")
        if not symbol:
            continue
        row_exp = _row_expiry_date(row, segment)
        if row_exp is None:
            continue
        rows_out.append(
            {
                "symbol": symbol.upper(),
                "expiry_date": _format_expiry_dd_mm_yyyy(row_exp),
                "_sort_exp": row_exp,
            }
        )

    rows_out.sort(key=lambda r: (r["_sort_exp"], r["symbol"]))
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with dest_csv.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=["symbol", "expiry_date"])
        writer.writeheader()
        for row in rows_out:
            writer.writerow({"symbol": row["symbol"], "expiry_date": row["expiry_date"]})
    return len(rows_out)


def lookup_by_trading_symbol(
    cache_dir: Path,
    trading_symbol: str,
    *,
    segment: str = "nse_fo",
    underlying: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Exact match on pTrdSymbol in cached segment CSV."""
    csv_path = _csv_path_for_segment(cache_dir, segment, underlying=underlying)
    if not csv_path.is_file():
        return None
    target = trading_symbol.strip().upper()
    for row in _iter_csv_rows(csv_path):
        trd = _row_get(row, "ptrdsymbol").upper()
        if trd != target:
            continue
        strike_raw = _row_get(row, "dstrikeprice", "dstrikeprice;")
        try:
            row_strike = float(strike_raw)
            if row_strike > 100000:
                row_strike = row_strike / 100.0
        except (TypeError, ValueError):
            row_strike = 0.0
        row_exp = _row_expiry_date(row, segment)
        return {
            "trading_symbol": trd,
            "token": _row_get(row, "psymbol", "pscriprefkey"),
            "lot_size": _row_get(row, "llotsize", "ilotsize"),
            "segment": _row_get(row, "pexchseg") or segment,
            "strike": row_strike,
            "expiry": row_exp.isoformat() if row_exp else "",
            "option_type": _row_get(row, "poptiontype").upper(),
        }
    return None


def lookup_option(
    cache_dir: Path,
    *,
    underlying: str,
    expiry: str,
    strike: int,
    option_type: str,
    segment: str = "nse_fo",
    underlying_filter: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Find one option row in cached nse_fo CSV by underlying/expiry/strike/CE|PE."""
    csv_path = _csv_path_for_segment(cache_dir, segment, underlying=underlying_filter)
    if not csv_path.is_file():
        return None

    target_expiry = _expiry_str_to_date(expiry)
    target_strike = float(strike)
    ot = option_type.upper()
    und = underlying.upper()

    for row in _iter_csv_rows(csv_path):
        row_exp = _row_expiry_date(row, segment)
        if row_exp != target_expiry:
            continue

        row_ot = _row_get(row, "poptiontype").upper()
        if row_ot and row_ot != ot:
            continue

        strike_raw = _row_get(row, "dstrikeprice", "dstrikeprice;")
        try:
            row_strike = float(strike_raw)
        except (TypeError, ValueError):
            continue
        # CSV strike often scaled (e.g. 5730000.0 for 57300)
        if row_strike > 100000:
            row_strike = row_strike / 100.0
        if int(round(row_strike)) != int(strike):
            continue

        sym_name = _row_get(row, "psymbolname", "psymbol")
        trd = _row_get(row, "ptrdsymbol")
        if not trd.upper().startswith(und) and und not in sym_name.upper():
            continue

        return {
            "trading_symbol": trd,
            "token": _row_get(row, "psymbol", "pscriprefkey"),
            "lot_size": _row_get(row, "llotsize", "ilotsize"),
            "segment": _row_get(row, "pexchseg") or segment,
            "strike": row_strike,
            "expiry": row_exp.isoformat() if row_exp else "",
            "option_type": row_ot or ot,
        }
    return None


def lookup_options(
    cache_dir: Path,
    *,
    underlying: str,
    expiry: str,
    strike: int,
    option_types: Iterable[str],
    segment: str = "nse_fo",
    underlying_filter: Optional[str] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for ot in option_types:
        out[ot.upper()] = lookup_option(
            cache_dir,
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=ot,
            segment=segment,
            underlying_filter=underlying_filter or underlying,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Kotak Neo scrip master CSVs")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for CSV cache (default {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=None,
        help="Only download these segments (e.g. nse_fo nse_cm-v1). Default: all",
    )
    parser.add_argument(
        "--underlying",
        default=None,
        help="Extract only this underlying (e.g. BANKNIFTY) into a smaller CSV",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only filter existing segment CSV; do not call download API",
    )
    parser.add_argument(
        "--lookup",
        action="store_true",
        help="Lookup strike/expiry in cached CSV (no download)",
    )
    parser.add_argument("--strike", type=int, default=None)
    parser.add_argument("--expiry", default=None, help="e.g. 28JUL2026")
    parser.add_argument(
        "--option-type",
        default="BOTH",
        choices=["CE", "PE", "BOTH"],
    )
    parser.add_argument(
        "--segment",
        default="nse_fo",
        help="CSV segment for lookup (default nse_fo)",
    )
    parser.add_argument(
        "--trading-symbol",
        default=None,
        help="Lookup exact pTrdSymbol (e.g. BANKNIFTY26JUL57300CE)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache_dir = args.cache_dir.resolve()

    if args.lookup:
        und_filter = args.underlying
        if args.trading_symbol:
            row = lookup_by_trading_symbol(
                cache_dir,
                args.trading_symbol,
                segment=args.segment,
                underlying=und_filter,
            )
            print(f"cache: {cache_dir}")
            print(f"csv: {_csv_path_for_segment(cache_dir, args.segment, underlying=und_filter)}")
            if row:
                print(json.dumps(row, indent=2))
                return 0
            print(f"not found: {args.trading_symbol}", file=sys.stderr)
            return 1
        if args.strike is None or not args.expiry:
            print("--lookup requires --strike and --expiry", file=sys.stderr)
            return 1
        live_cfg = build_live_config(load_live_config())
        und = args.underlying or live_cfg.underlying_symbol
        types = ["CE", "PE"] if args.option_type == "BOTH" else [args.option_type]
        rows = lookup_options(
            cache_dir,
            underlying=und,
            expiry=args.expiry,
            strike=args.strike,
            option_types=types,
            segment=args.segment,
            underlying_filter=und_filter or und,
        )
        print(f"cache: {cache_dir}")
        print(f"csv: {_csv_path_for_segment(cache_dir, args.segment, underlying=und_filter or und)}")
        for ot, row in rows.items():
            if row:
                print(
                    f"  {ot}: {row['trading_symbol']}  token={row['token']}  "
                    f"lot={row['lot_size']}  expiry={row['expiry']}"
                )
            else:
                print(f"  {ot}: (not found)")
        return 0 if all(rows.values()) else 1

    if args.extract_only:
        if not args.underlying:
            print("--extract-only requires --underlying (e.g. BANKNIFTY)", file=sys.stderr)
            return 1
        segment = (args.segments[0] if args.segments else args.segment).lower()
        source = cache_dir / f"{segment}.csv"
        dest = cache_dir / _symbol_expiry_csv_name(args.underlying)
        try:
            count = extract_symbol_expiry_csv(
                source, dest, args.underlying, segment=segment
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            print(f"Download first: python -m live.download_scrip_master --segments {segment}", file=sys.stderr)
            return 1
        print(f"Extracted {count} {args.underlying} symbol/expiry rows -> {dest}")
        print(f"  source: {source} ({source.stat().st_size:,} bytes)")
        print(f"  output: {dest} ({dest.stat().st_size:,} bytes)")
        return 0

    creds = load_neo_credentials()
    if not creds.consumer_key:
        print("NEO_API_KEY required in .env", file=sys.stderr)
        return 1

    manifest = download_scrip_master(
        creds.consumer_key,
        cache_dir,
        segments=args.segments,
    )
    print(f"Downloaded {manifest['file_count']} file(s) -> {cache_dir}")
    print(f"Manifest: {cache_dir / MANIFEST_NAME}")
    for item in manifest.get("files", []):
        print(f"  {item['segment']}: {item['bytes']:,} bytes")

    if args.underlying:
        segment = (args.segments[0] if args.segments else "nse_fo").lower()
        source = cache_dir / f"{segment}.csv"
        dest = cache_dir / _symbol_expiry_csv_name(args.underlying)
        if source.is_file():
            count = extract_symbol_expiry_csv(
                source, dest, args.underlying, segment=segment
            )
            print(f"\nExtracted {count} {args.underlying} symbol/expiry rows -> {dest.name}")
            print(f"  output size: {dest.stat().st_size:,} bytes")
        else:
            print(f"\nWarning: {source.name} missing; could not extract {args.underlying}", file=sys.stderr)

    print("\nLookup example:")
    print(
        "  python -m live.download_scrip_master --lookup --underlying BANKNIFTY "
        "--strike 57300 --expiry 28JUL2026 --option-type BOTH"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
