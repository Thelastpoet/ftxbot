"""
Logging utilities for Forex Trading Bot.
"""

import logging
from logging.handlers import RotatingFileHandler


def configure_logging(console_level: str = 'INFO') -> None:
    """Configure production-friendly logging for console and file.

    - Console: concise format at requested level (default INFO).
    - File: rolling INFO+ logs in forex_bot.log.
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

    # File handler: rolling logs to cap size
    fh = RotatingFileHandler(
        'forex_bot.log',
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler: concise
    ch = logging.StreamHandler()
    ch.setLevel(level_map.get(str(console_level).upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    root.addHandler(fh)
    root.addHandler(ch)

    # Tame noisy libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('MetaTrader5').setLevel(logging.WARNING)
