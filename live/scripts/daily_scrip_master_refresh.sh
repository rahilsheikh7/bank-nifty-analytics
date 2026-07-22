#!/usr/bin/env bash
# Refresh Kotak Neo nse_fo scrip master + BANKNIFTY symbol/expiry CSV.
# Intended for cron at 08:00 IST, e.g.:
#   0 8 * * * /home/trader/bank-nifty-analytics/live/scripts/daily_scrip_master_refresh.sh >> /home/trader/bank-nifty-analytics/logs/scrip_master_refresh.log 2>&1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export TZ=Asia/Kolkata

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv python not found at $PYTHON" >&2
  exit 1
fi

echo "=== scrip master refresh $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

"$PYTHON" -m live.download_scrip_master \
  --segments nse_fo \
  --underlying BANKNIFTY

echo "Done. Output: ${ROOT}/live/cache/scrip_master/banknifty_symbol_expiry.csv"
