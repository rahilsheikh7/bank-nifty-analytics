# Bank Nifty Live Options Bot (Kotak Neo)

Real-time trading bot that reuses the **exact same** strategy logic as `backtest.py`
(`SignalEngine` + `StateManager` via `strategy_runtime.step_bar`), computes signals on
the **BANKNIFTY index**, and executes them as a **2-leg synthetic future** on options.

Default mode is **paper** (simulated fills at live option LTP). Set `live.mode: live`
in `config/strategy.yaml` only after validating paper runs.

## Prerequisites

1. **Python 3.10+**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-live.txt
   ```
3. **Kotak Neo** credentials in `.env` (trading + live feed):
   ```
   NEO_API_KEY=...
   NEO_CLIENT_CODE=...
   NEO_MPIN=...
   NEO_MOBILE=+91...
   NEO_TOTP_SECRET=       # optional; auto-TOTP if set
   ```
4. **Zerodha Kite** credentials for daily warmup history:
   ```
   KITE_API_KEY=...
   KITE_ACCESS_TOKEN=...  # run generate_token.py each morning
   ```

## Warmup: Kite daily + Neo live sync

| Source | Role |
|--------|------|
| **Kite** | Fresh ~80 sessions of 1-min Bank Nifty history at each startup (EMA200 seed) |
| **Neo** | Live index ticks during 09:15–15:30; appends today's bars to cache |
| **bars_csv** | Single merged store (`live/cache/banknifty_1min_neo.csv`) |

On duplicate timestamps for **today's session**, Neo/cache bars win over Kite.

Config (`config/strategy.yaml`):

```yaml
live:
  warmup_source: kite_daily
  kite_fallback_on_error: true   # use bars_csv if Kite fails
  warmup_sessions: 80
  bars_csv: live/cache/banknifty_1min_neo.csv
```

Overnight open positions and pending strategy flags survive in `live/state/position.json`
independently of the OHLC refresh.

## Daily workflow

1. `python generate_token.py` → update `KITE_ACCESS_TOKEN` in `.env`
2. Optional pre-fetch: `python -m live.fetch_warmup`
3. `python -m live.smoke_login` (optional Neo login check)
4. `python -m live.run_live` (fetches Kite again at startup, then Neo live feed)
5. Trades logged to `results/live_trades_YYYYMMDD.csv`

## Architecture

```
Kite fetch (startup) ──merge──► bars_csv ◄── Neo ticks (session)
                                      │
                                      ▼
              prepare_backtest_data → step_bar → paper/live orders
```

- **Entries/exits (ST flip, EMA):** completed primary bars
- **SL/TP:** intrabar on index ticks (`intrabar_exit: true`)
- **State:** `live/state/position.json` (position + pending flags across days)

## Switching to live trading

1. Run paper sessions; review `results/live_trades_*.csv`
2. Verify margin via Neo (`limits` at startup)
3. Set `live.mode: live` in `config/strategy.yaml`

## Backtest (unchanged)

Backtest uses `banknifty_1min_from2020.csv` and `backtest.py` only. Live changes do not
modify backtest code paths.

## Files

| File | Purpose |
|------|---------|
| `live/run_live.py` | Main entry point |
| `live/fetch_warmup.py` | Pre-market Kite fetch only |
| `live/kite_warmup.py` | Kite 1-min history fetch |
| `live/bar_cache.py` | CSV persistence + Kite/Neo merge |
| `live/warmup.py` | Startup warmup orchestration |

## Notes

- Live uses a **single combined book** (one position at a time)
- `kite_fallback_on_error: true` allows startup with stale cache if Kite token expired
- Backtest `independent_books` in YAML does not affect live (forced combined book)
