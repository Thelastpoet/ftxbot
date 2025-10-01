from datetime import datetime, time, timedelta, timezone
import logging

from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SymbolPrecision:
    symbol: str
    normalized_symbol: str
    digits: int
    tick_size: float
    pip_size: float
    tick_value: float
    pip_value: float


def _normalize_symbol_name(name: str) -> str:
    base = (name or "").upper()
    for sep in (".", "_", "-"):
        if sep in base:
            base = base.split(sep, 1)[0]
    return base


def _resolve_pip_override(overrides: Optional[Dict[str, float]]) -> float:
    if not overrides:
        return 0.0
    for key in ("pip_size", "pip", "display_pip", "pipValue"):
        value = overrides.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
            if value_f > 0:
                return value_f
        except (TypeError, ValueError):
            continue
    return 0.0


def _infer_fx_pip_size(normalized_symbol: str) -> float:
    quote = normalized_symbol[3:]
    if quote == "JPY":
        return 0.01
    return 0.0001


def get_symbol_precision(symbol_info: Any, overrides: Optional[Dict[str, float]] = None) -> SymbolPrecision:
    if not symbol_info:
        return SymbolPrecision(
            symbol="",
            normalized_symbol="",
            digits=0,
            tick_size=0.0,
            pip_size=0.0,
            tick_value=0.0,
            pip_value=0.0,
        )

    name = getattr(symbol_info, "name", "") or ""
    normalized = _normalize_symbol_name(name)
    digits = int(getattr(symbol_info, "digits", 0) or 0)
    tick_size = float(getattr(symbol_info, "trade_tick_size", 0.0) or 0.0)
    if tick_size <= 0:
        tick_size = float(getattr(symbol_info, "point", 0.0) or 0.0)

    pip_override = _resolve_pip_override(overrides)

    if pip_override > 0:
        pip_size = pip_override
    else:
        pip_size = 0.0
        if normalized.startswith(("XAU", "XAG", "XPT", "XPD")):
            pip_size = 0.10
        elif normalized.startswith(("XBR", "XTI", "UKO", "USO", "WTI", "BRENT")):
            pip_size = 0.10
        elif len(normalized) == 6 and normalized.isalpha():
            pip_size = _infer_fx_pip_size(normalized)
        elif digits in (3, 5) and tick_size > 0:
            pip_size = tick_size * 10.0

        if pip_size <= 0 and tick_size > 0:
            pip_size = tick_size

    tick_value = float(getattr(symbol_info, "trade_tick_value", 0.0) or 0.0)

    pip_value = 0.0
    if tick_size > 0 and tick_value > 0 and pip_size > 0:
        pip_value = tick_value * (pip_size / tick_size)

    return SymbolPrecision(
        symbol=name,
        normalized_symbol=normalized,
        digits=digits,
        tick_size=tick_size,
        pip_size=pip_size,
        tick_value=tick_value,
        pip_value=pip_value,
    )

def get_tick_size(symbol_info: any) -> float:
    """
    Return the broker-defined minimum tick size for a symbol.
    This is the only safe value for risk/SL/TP math.
    """
    if not symbol_info:
        return 0.0
    if getattr(symbol_info, "trade_tick_size", 0) > 0:
        return symbol_info.trade_tick_size
    return getattr(symbol_info, "point", 0.0)

def get_display_pip_size(symbol_info: any, overrides: Optional[Dict[str, float]] = None) -> float:
    """
    Human-friendly pip size for logs and reporting only.
    Derived from symbol precision to keep strategy and risk manager aligned.
    """
    precision = get_symbol_precision(symbol_info, overrides=overrides)
    return precision.pip_size


def get_pip_value(symbol_info: any, volume: float = 1.0, overrides: Optional[Dict[str, float]] = None) -> float:
    """
    Returns the monetary value of one display pip for the given volume.
    """
    precision = get_symbol_precision(symbol_info, overrides=overrides)
    if precision.pip_value <= 0:
        return 0.0
    return precision.pip_value * float(volume)


# All times are in UTC.
# This list defines time windows when trading should be paused.
#
# Structure:
# (
#   day_of_week (0=Monday, 6=Sunday, None=Any Day),
#   time_of_day (UTC time object),
#   pre_buffer_minutes (pause this many minutes BEFORE the time),
#   post_buffer_minutes (pause this many minutes AFTER the time),
#   event_name (for logging)
# )
#
NEWS_BLACKOUT_PERIODS = [
    # General high-volatility market opens
    # London Open (occurs around 8:00 UTC)
    (None, time(8, 0), 5, 20, "London Open"),
    # New York Open (occurs around 13:00 UTC)
    (None, time(13, 0), 5, 20, "New York Open"),

    # Major US News Releases (typically 8:30 AM ET or 10:00 AM ET)
    # Corresponds to ~13:30 UTC and ~15:00 UTC (without DST)
    # This is a broad window for events like CPI, PPI, Retail Sales etc.
    (None, time(13, 30), 10, 20, "US Morning News Slot"),
    (None, time(15, 0), 10, 20, "US Late Morning News Slot"),

    # FOMC Statement (typically 2:00 PM ET on a Wednesday)
    # Corresponds to ~19:00 UTC (without DST)
    (2, time(19, 0), 5, 60, "FOMC Statement"),
]

def is_trading_paused(current_utc_time: datetime) -> tuple[bool, str]:
    """
    Checks if the current time falls within a blackout period for news releases 
    or high volatility market opens.

    Args:
        current_utc_time: The current time in UTC (must be timezone-aware).

    Returns:
        A tuple containing:
        - bool: True if trading should be paused, False otherwise.
        - str: A message indicating the reason for the pause, or an empty string.
    """
    current_day = current_utc_time.weekday()

    for day_of_week, event_time, pre_buffer, post_buffer, event_name in NEWS_BLACKOUT_PERIODS:
        
        # Check if the event applies to the current day of the week
        if day_of_week is not None and day_of_week != current_day:
            continue

        # Create a timezone-aware datetime for the event on the current day
        event_dt = datetime.combine(current_utc_time.date(), event_time, tzinfo=timezone.utc)

        # Calculate the start and end of the blackout window
        start_datetime = event_dt - timedelta(minutes=pre_buffer)
        end_datetime = event_dt + timedelta(minutes=post_buffer)

        # Check if the current time falls within the window
        if start_datetime <= current_utc_time <= end_datetime:
            return (True, f"Paused for {event_name}")

    return (False, "")
