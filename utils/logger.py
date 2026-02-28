"""
Structured logging + trade journal.

All modules should obtain their logger via get_logger(__name__).
Trade events are additionally written to a CSV trade journal.
"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

import config

_LOG_DIR = Path("logs")
_INITIALIZED = False


def _init_logging() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (rotating-safe via plain FileHandler)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _init_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Trade journal (CSV append)
# ---------------------------------------------------------------------------

_JOURNAL_COLUMNS = [
    "timestamp", "action", "strategy", "symbol",
    "signal", "entry_price", "stop_loss", "take_profit",
    "quantity", "bear_regime", "dryrun",
]


def log_trade(trade_data: dict[str, Any]) -> None:
    """
    Append a trade event to the CSV trade journal.

    Accepts any dict — missing fields default to empty string.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    journal_path = Path(config.TRADE_JOURNAL_FILE)

    write_header = not journal_path.exists()

    with open(journal_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_JOURNAL_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        row = {col: trade_data.get(col, "") for col in _JOURNAL_COLUMNS}
        if not row.get("timestamp"):
            from datetime import datetime, timezone
            row["timestamp"] = datetime.now(timezone.utc).isoformat()

        writer.writerow(row)
