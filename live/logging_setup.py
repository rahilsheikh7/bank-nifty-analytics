"""Configure live bot logging to console and trading.log."""
from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "trading.log"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_LINE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def prune_trading_log(path: Path = LOG_FILE, *, retention_days: int = 3) -> int:
    """Drop log lines older than ``retention_days`` (keeps recent troubleshooting history)."""
    if retention_days <= 0 or not path.is_file():
        return 0
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return 0
    if not lines:
        return 0

    cutoff = datetime.now() - timedelta(days=retention_days)
    kept: list[str] = []
    for line in lines:
        match = _LOG_LINE_TS.match(line)
        if not match:
            kept.append(line)
            continue
        try:
            ts = datetime.strptime(match.group(1), DATE_FORMAT)
        except ValueError:
            kept.append(line)
            continue
        if ts >= cutoff:
            kept.append(line)

    removed = len(lines) - len(kept)
    if removed > 0:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        tmp.replace(path)
    return removed


def setup_logging(level: int = logging.INFO, *, log_retention_days: int = 3) -> Path:
    """Attach console + file handlers to the root logger (idempotent)."""
    prune_trading_log(LOG_FILE, retention_days=log_retention_days)
    root = logging.getLogger()
    if root.handlers:
        return LOG_FILE

    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Neo SDK uses websocket-client; keep connection noise out of trading.log.
    logging.getLogger("websocket").setLevel(logging.WARNING)

    logging.getLogger("live.logging").info("Logging to %s and console", LOG_FILE)
    return LOG_FILE
