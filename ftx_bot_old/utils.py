from datetime import datetime, time, timedelta, timezone

def get_pip_size(symbol_info: any) -> float:
    """
    Calculates the pip size for a given symbol, handling both dicts and objects.
    Handles standard FX, JPY pairs, and others based on digits.
    """
    # Use getattr to safely access attributes from an object or fallback
    point = getattr(symbol_info, 'point', 0.00001)
    digits = getattr(symbol_info, 'digits', 5)

    # For 2-digit (e.g., indices) or 3-digit (e.g., USDJPY) instruments
    if digits in (2, 3):
        # For JPY pairs, one pip is 0.01
        return 0.01
    
    # For 4 or 5-digit forex pairs, one pip is 10 points
    return point * 10

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
