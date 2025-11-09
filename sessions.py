"""
Session utilities for AMD context.
Classifies trading sessions based on UTC hours.
"""

from datetime import datetime, timezone
from typing import Literal

SessionType = Literal["ASIA", "LONDON", "NY", "UNKNOWN"]


def get_current_session(dt: datetime = None,
                       asia_hours: tuple = (0, 8),
                       london_hours: tuple = (8, 16),
                       ny_hours: tuple = (13, 22)) -> SessionType:
    """
    Classify current session based on UTC hour.

    Default sessions (UTC):
    - Asia: 00:00-08:00 (Tokyo + Hong Kong + Singapore)
    - London: 08:00-16:00 (European session)
    - NY: 13:00-22:00 (US session, overlaps with London)

    Note: Sessions can overlap (e.g., London/NY overlap 13:00-16:00 UTC).
    Priority: NY > London > Asia for overlapping hours.

    Args:
        dt: datetime object (must be timezone-aware UTC). If None, uses current UTC time.
        asia_hours: (start_hour, end_hour) for Asia session
        london_hours: (start_hour, end_hour) for London session
        ny_hours: (start_hour, end_hour) for NY session

    Returns:
        SessionType: "ASIA", "LONDON", "NY", or "UNKNOWN"
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Ensure timezone-aware
    if dt.tzinfo is None:
        raise ValueError("DateTime must be timezone-aware (use timezone.utc)")

    # Convert to UTC if not already
    dt_utc = dt.astimezone(timezone.utc)
    hour = dt_utc.hour

    # Priority: NY > London > Asia (for overlapping periods)
    if ny_hours[0] <= hour < ny_hours[1]:
        return "NY"
    elif london_hours[0] <= hour < london_hours[1]:
        return "LONDON"
    elif asia_hours[0] <= hour < asia_hours[1]:
        return "ASIA"
    else:
        return "UNKNOWN"


def is_session_open(session: SessionType, dt: datetime = None,
                   asia_hours: tuple = (0, 8),
                   london_hours: tuple = (8, 16),
                   ny_hours: tuple = (13, 22)) -> bool:
    """
    Check if a specific session is currently open.

    Args:
        session: Session to check ("ASIA", "LONDON", "NY")
        dt: datetime object (timezone-aware UTC). If None, uses current UTC time.
        asia_hours: (start_hour, end_hour) for Asia session
        london_hours: (start_hour, end_hour) for London session
        ny_hours: (start_hour, end_hour) for NY session

    Returns:
        bool: True if session is open
    """
    current = get_current_session(dt, asia_hours, london_hours, ny_hours)
    return current == session


def get_session_start_hour(session: SessionType,
                           asia_hours: tuple = (0, 8),
                           london_hours: tuple = (8, 16),
                           ny_hours: tuple = (13, 22)) -> int:
    """
    Get the UTC hour when a session starts.

    Args:
        session: Session type
        asia_hours: (start_hour, end_hour) for Asia session
        london_hours: (start_hour, end_hour) for London session
        ny_hours: (start_hour, end_hour) for NY session

    Returns:
        int: Start hour (UTC)
    """
    session_map = {
        "ASIA": asia_hours[0],
        "LONDON": london_hours[0],
        "NY": ny_hours[0]
    }
    return session_map.get(session, 0)


def get_session_end_hour(session: SessionType,
                         asia_hours: tuple = (0, 8),
                         london_hours: tuple = (8, 16),
                         ny_hours: tuple = (13, 22)) -> int:
    """
    Get the UTC hour when a session ends.

    Args:
        session: Session type
        asia_hours: (start_hour, end_hour) for Asia session
        london_hours: (start_hour, end_hour) for London session
        ny_hours: (start_hour, end_hour) for NY session

    Returns:
        int: End hour (UTC)
    """
    session_map = {
        "ASIA": asia_hours[1],
        "LONDON": london_hours[1],
        "NY": ny_hours[1]
    }
    return session_map.get(session, 24)


def is_near_session_open(session: SessionType, dt: datetime = None,
                         window_minutes: int = 15,
                         asia_hours: tuple = (0, 8),
                         london_hours: tuple = (8, 16),
                         ny_hours: tuple = (13, 22)) -> bool:
    """
    Check if current time is within N minutes after a session open.
    Useful for detecting manipulation at London/NY opens.

    Args:
        session: Session to check
        dt: datetime object (timezone-aware UTC). If None, uses current UTC time.
        window_minutes: Minutes after session open to consider "near open"
        asia_hours: (start_hour, end_hour) for Asia session
        london_hours: (start_hour, end_hour) for London session
        ny_hours: (start_hour, end_hour) for NY session

    Returns:
        bool: True if within window_minutes of session open
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        raise ValueError("DateTime must be timezone-aware (use timezone.utc)")

    dt_utc = dt.astimezone(timezone.utc)
    start_hour = get_session_start_hour(session, asia_hours, london_hours, ny_hours)

    # Check if we're in the session
    if not is_session_open(session, dt, asia_hours, london_hours, ny_hours):
        return False

    # Check if within window of session start
    session_start = dt_utc.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    minutes_since_open = (dt_utc - session_start).total_seconds() / 60

    return 0 <= minutes_since_open <= window_minutes
