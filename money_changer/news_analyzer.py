import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

class NewsAnalyzer:
    def __init__(self, csv_filepath='news_calendar.csv', config=None):
        self.csv_filepath = csv_filepath
        self.news_data = pd.DataFrame() # Initialize as empty DataFrame
        self.last_load_time = None
        
        # Safely get config settings
        self.config = config.get('news_filter_settings', {}) if config else {}
        self.minutes_before = timedelta(minutes=self.config.get('minutes_before_news', 30))
        self.minutes_after = timedelta(minutes=self.config.get('minutes_after_news', 30))
        refresh_hours = self.config.get('news_refresh_interval_hours', 24) # Default to 24 hours
        self.refresh_interval = timedelta(hours=refresh_hours)

        # Perform the initial load
        self.load_news_data()

    def _is_refresh_needed(self) -> bool:
        """Checks if the time since the last data load exceeds the refresh interval."""
        if not self.last_load_time:
            return True
        if (datetime.now(timezone.utc) - self.last_load_time) > self.refresh_interval:
            return True
        return False

    def load_news_data(self):
        """Loads or reloads news data from the CSV file if needed."""
        try:
            self.news_data = pd.read_csv(self.csv_filepath, parse_dates=['datetime_utc'])
            # Ensure the datetime column is timezone-aware (UTC)
            self.news_data['datetime_utc'] = self.news_data['datetime_utc'].dt.tz_localize('UTC')
            self.last_load_time = datetime.now(timezone.utc)
            logging.info(f"Successfully loaded {len(self.news_data)} news events. Next refresh in {self.refresh_interval}.")
        except FileNotFoundError:
            logging.warning(f"News calendar file not found at '{self.csv_filepath}'. Trading will proceed without a news filter.")
            self.news_data = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading news calendar: {e}")
            self.news_data = pd.DataFrame()

    def check_for_high_impact_news(self, symbol: str) -> bool:
        """
        Checks for high-impact news. It will automatically reload the news file
        from disk if the refresh interval has been exceeded.
        """
        if self._is_refresh_needed():
            logging.info("News data refresh interval reached. Reloading news calendar...")
            self.load_news_data()

        if self.news_data.empty:
            return False

        base_currency = symbol[:3].upper()
        quote_currency = symbol[3:].upper()
        
        now_utc = datetime.now(timezone.utc)
        start_of_zone = now_utc - self.minutes_after
        end_of_zone = now_utc + self.minutes_before

        relevant_news = self.news_data[
            (self.news_data['currency'].isin([base_currency, quote_currency])) &
            (self.news_data['impact'] == 'High') &
            (self.news_data['datetime_utc'] >= start_of_zone) &
            (self.news_data['datetime_utc'] <= end_of_zone)
        ]

        if not relevant_news.empty:
            for _, row in relevant_news.iterrows():
                logging.warning(f"[{symbol}] Halting trading due to high-impact news: {row['event_name']} ({row['currency']}) at {row['datetime_utc'].strftime('%H:%M')} UTC.")
            return True

        return False