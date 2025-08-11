import pandas as pd
import pytz
from datetime import time, datetime

class SessionManager:
    """
    A single, authoritative source for market session times.
    This object is created once and shared across the entire application
    to ensure consistent time-based logic.
    """
    def __init__(self, config: dict):
        self.config = config
        self.sessions = {
            "asian": {
                "start": time(0, 0),   # 00:00 UTC
                "end": time(9, 0)      # 09:00 UTC
            },
            "london": {
                "start": time(7, 0),   # 07:00 UTC
                "end": time(16, 0)     # 16:00 UTC
            },
            "ny": {
                "start": time(13, 0),  # 13:00 UTC
                "end": time(22, 0)     # 22:00 UTC
            }
        }
        # Allow overrides from config if necessary
        # For example, you could add a "sessions" key to your config.json
        # self.sessions.update(config.get("sessions", {}))

    def get_current_session(self, now_utc: datetime = None) -> str:
        """Determines the current active trading session."""
        if now_utc is None:
            now_utc = datetime.now(pytz.utc)
        
        current_time = now_utc.time()

        if self._is_time_between(current_time, self.sessions['ny']['start'], self.sessions['ny']['end']):
            return "ny"
        if self._is_time_between(current_time, self.sessions['london']['start'], self.sessions['london']['end']):
            return "london"
        if self._is_time_between(current_time, self.sessions['asian']['start'], self.sessions['asian']['end']):
            return "asian"
        
        return "out_of_session"

    def get_asian_session_range_timestamps(self, current_day_utc: datetime) -> tuple[datetime, datetime]:
        """
        Calculates the start and end timestamps for the most recently
        completed or currently active Asian session.
        """
        asian_start_time = self.sessions['asian']['start']
        asian_end_time = self.sessions['asian']['end']

        # Combine the current date with the session start/end times
        start_dt = datetime.combine(current_day_utc.date(), asian_start_time, tzinfo=pytz.utc)
        end_dt = datetime.combine(current_day_utc.date(), asian_end_time, tzinfo=pytz.utc)

        # If we are already past the Asian session for today, these are the correct timestamps.
        # If we are *in* the Asian session, these are also correct.
        # If it's before the Asian session has started for today (e.g. day just rolled over),
        # we need the Asian session from the *previous* day.
        if current_day_utc < start_dt:
            yesterday = current_day_utc.date() - pd.Timedelta(days=1)
            start_dt = datetime.combine(yesterday, asian_start_time, tzinfo=pytz.utc)
            end_dt = datetime.combine(yesterday, asian_end_time, tzinfo=pytz.utc)
            
        return start_dt, end_dt
        
    def _is_time_between(self, check_time: time, start_time: time, end_time: time) -> bool:
        """Helper function to check if a time is within a range."""
        if start_time < end_time:
            return start_time <= check_time < end_time
        else:  # Handles overnight sessions like Sydney that cross midnight
            return check_time >= start_time or check_time < end_time