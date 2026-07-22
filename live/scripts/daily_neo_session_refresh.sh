#!/usr/bin/env bash
# Refresh Kotak Neo trade session (token.json) before market open.
# Intended for cron at 08:30 IST, e.g.:
#   30 8 * * * /home/trader/bank-nifty-analytics/live/scripts/daily_neo_session_refresh.sh >> /home/trader/bank-nifty-analytics/logs/neo_session_refresh.log 2>&1
#
# Requires in .env: NEO_API_KEY, NEO_MOBILE, NEO_CLIENT_CODE, NEO_MPIN, NEO_TOTP_SECRET
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

if [[ ! -f "${ROOT}/.env" ]]; then
  echo "ERROR: .env not found at ${ROOT}/.env" >&2
  exit 1
fi

echo "=== Neo session refresh $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

"$PYTHON" "${ROOT}/generate_neo_session.py"

echo "Done. Output: ${ROOT}/token.json"
