"""Configure live bot logging to console and trading.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "trading.log"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> Path:
    """Attach console + file handlers to the root logger (idempotent)."""
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
