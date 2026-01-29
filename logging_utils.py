"""
Logging utilities for Forex Trading Bot.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    console_level: str = 'INFO',
    log_file: str = 'forex_bot.log',
    file_level: str = 'DEBUG',
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """Configure production-friendly logging for console and file.

    - Console: concise format at requested level (default INFO).
    - File: rotating file at file_level (default DEBUG).
    - Suppress noisy third-party debug (e.g., asyncio proactor message).
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicates on reload
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler: concise
    ch = logging.StreamHandler()
    ch.setLevel(level_map.get(str(console_level).upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    root.addHandler(ch)

    # File handler: rotating, always available for debug investigation
    try:
        log_path = Path(log_file).expanduser()
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        fh.setLevel(level_map.get(str(file_level).upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root.addHandler(fh)
    except Exception as e:
        root.warning(f"File logging disabled: {e}")

    # Tame noisy libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('MetaTrader5').setLevel(logging.WARNING)
