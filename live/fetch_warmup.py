"""Pre-market warmup: fetch Kite history and merge into bars_csv (no Neo login).

Usage:
    python -m live.fetch_warmup

Run after ``generate_token.py`` each morning, or rely on ``run_live`` which
fetches Kite automatically at startup.
"""
from __future__ import annotations

import logging
import sys

from live.config import build_backtest_config_for_live, build_live_config, load_live_config
from live.warmup import build_warm_1min


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_root = load_live_config()
    live_cfg = build_live_config(config_root)
    bt_config = build_backtest_config_for_live(config_root)

    _, status = build_warm_1min(
        live_cfg.warmup_sessions,
        warmup_max_bars=live_cfg.warmup_max_bars,
        bars_csv=live_cfg.bars_csv,
        ema_length=bt_config.ema_length,
        ema_timeframe=bt_config.ema_timeframe,
        primary_timeframe=bt_config.primary_timeframe,
        warmup_source=live_cfg.warmup_source,
        kite_fallback_on_error=live_cfg.kite_fallback_on_error,
        session_start=live_cfg.session_start,
    )
    print(status.message)
    return 0 if status.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
