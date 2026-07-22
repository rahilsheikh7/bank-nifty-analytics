# Direct order & cancel test commands

Quick reference for `live/test_order_placement.py` and `live/inspect_scrip.py`.  
Run from the project root with the venv active:

```bash
cd /home/trader/bank-nifty-analytics
source .venv/bin/activate
```

Symbol pattern (no `search_scrip`): `BANKNIFTY` + `YY` + `MON` + strike + `CE`/`PE`  
Example: expiry `28JUL2026` + strike `56700` → `BANKNIFTY26JUL56700CE`

Strike must be a valid exchange step (Bank Nifty: typically multiples of **100**). Invalid strikes return 0 scrip rows and will fail at place.

---

## Enter + exit cleanly (MKT) — recommended

**`--round-trip`** places entry, waits, then **squares off** (reverse legs). Use this for filled MKT tests.

```bash
# Enter short @ ATM, wait 10s, exit (square-off)
python -m live.test_order_placement \
  --direct --expiry 28JUL2026 --direction short \
  --round-trip --place --yes
```

Custom wait:
```bash
python -m live.test_order_placement \
  --direct --expiry 28JUL2026 --direction short \
  --round-trip --wait-seconds 5 --place --yes
```

---

## Close an existing position (no new entry)

If you already have an open short from a prior test, use **`--exit-only`** with the **same strike** you entered at:

```bash
# You entered SHORT @ 57400 — square off only
python -m live.test_order_placement \
  --direct --expiry 28JUL2026 --direction short \
  --strike 57400 --exit-only --place --yes
```

`--direction short` means “the position I opened was a short” → script sends BUY CE + SELL PE to flatten.

---

## Enter only (no exit)

```bash
python -m live.test_order_placement \
  --direct --expiry 28JUL2026 --direction short --place --yes
```

---

## Place + cancel (with auto square-off if filled)

**`--place-cancel`** tries to cancel open orders. If MKT already filled (`order is completed`), it **automatically square-offs** instead.

```bash
python -m live.test_order_placement \
  --direct --expiry 28JUL2026 --direction short \
  --place-cancel --place --yes
```

---

## Inspect scrip (no orders)

```bash
python -m live.inspect_scrip
python -m live.inspect_scrip --strike 57400 --expiry 28JUL2026 --option-type BOTH
```

---

## Cancel by order number (open orders only)

```bash
python -m live.test_order_placement --cancel 260709000455370 260709000455369 --yes
```

---

## Scrip master CSV (local symbol cache)

Download daily instrument master files (no login — only `NEO_API_KEY`):

```bash
# All segments (large download)
python -m live.download_scrip_master

# Bank Nifty options only (~26MB full nse_fo, then filter)
python -m live.download_scrip_master --segments nse_fo --underlying BANKNIFTY

# Or filter an already-downloaded nse_fo.csv (no API call)
python -m live.download_scrip_master --extract-only --underlying BANKNIFTY --segment nse_fo
```

Saves `live/cache/scrip_master/banknifty_nse_fo.csv` (~2k rows vs full file).

Lookup using the smaller file:

```bash
python -m live.download_scrip_master --lookup --underlying BANKNIFTY --strike 57300 --expiry 28JUL2026 --option-type BOTH
python -m live.download_scrip_master --lookup --trading-symbol BANKNIFTY26JUL57300CE --underlying BANKNIFTY
```

Refresh daily (scrip master updates each trading day).

---

## Session

```bash
python generate_neo_session.py
```

Fast path uses `token.json` — no login, no account limits.

---

## Flags cheat sheet

| Flag | Meaning |
|------|---------|
| `--direct` | Build `ts` from strike + expiry (no scrip lookup) |
| `--round-trip` | Entry → wait → **square-off** (clean MKT exit) |
| `--exit-only` | **Square-off only** — no new entry |
| `--place-cancel` | Cancel try; **auto square-off** if MKT filled |
| `--strike` | Strike in index points (required for `--exit-only` if not ATM) |
| `--expiry` | e.g. `28JUL2026` |
| `--direction` | `long` or `short` (entry type; exit-only = position you hold) |
| `--wait-seconds` | Wait before square-off in round-trip (default **10**) |
| `--cancel-wait-seconds` | Wait before cancel in place-cancel (default **10**) |
| `--place --yes` | Send real orders |

---

## Timing

Each run prints `[timing]` lines: `load session`, `index LTP fetch`, `place` per leg, `place total`, `run total`.

Typical fast path entry: **~2s** (index + parallel place).
