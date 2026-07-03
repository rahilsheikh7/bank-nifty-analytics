#!/bin/bash
# Non-interactive Bank Nifty live bot (VPS / systemd).
# Same as: python -m live.run_live
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/trader/bank-nifty-analytics}"

cd "$PROJECT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
exec python -m live.run_live
