import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
import os

class NewsAnalyzer:
    def __init__(self, csv_filepath='news_calendar.csv', config=None):
        self.csv_filepath = csv_filepath
        self.news_data = pd.DataFrame() # Start with an empty DataFrame
        self.last_load_time = None
        self.file_exists = False
        self.config = config['news_filter_settings'] if config else {}
        self.load_news_data() # Initial attempt to load

    def load_news_data(self):
        """Loads or reloads news data from the CSV file."""
        try:
            self.news_data = pd.read_csv(self.csv_filepath, parse_dates=['datetime_utc'])
            self.news_data['datetime_utc'] = self.news_data['datetime_utc'].dt.tz_localize('UTC')
            self.last_load_time = datetime.now(timezone.utc)
            self.file_exists = True
            logging.info(f"Successfully loaded {len(self.news_data)} news events from {self.csv_filepath}")
        except FileNotFoundError:
            # This is expected on first run before the scraper finishes
            if not self.file_exists: # Log this only once
                 logging.warning(f"News calendar not found at '{self.csv_filepath}'. Awaiting background scraper. Trading continues without news filter.")
            self.news_data = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading news calendar: {e}")
            self.news_data = pd.DataFrame()

    def check_for_high_impact_news(self, symbol: str) -> bool:
        """
        Checks if there is a high-impact news event for the given symbol within the configured time window.
        Returns True if there is a conflicting event, False otherwise.
        """
        # --- NEW: INTELLIGENT RELOADING ---
        # If the file didn't exist, but now it does, load it for the first time.
        if not self.file_exists and os.path.exists(self.csv_filepath):
            logging.info("News calendar has been created by the background scraper. Loading data now.")
            self.load_news_data()

        if self.news_data.empty:
            return False

        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        minutes_before = timedelta(minutes=self.config.get('minutes_before_news', 30))
        minutes_after = timedelta(minutes=self.config.get('minutes_after_news', 30))
        
        now_utc = datetime.now(timezone.utc)
        start_of_zone = now_utc - minutes_after
        end_of_zone = now_utc + minutes_before

        relevant_news = self.news_data[
            (self.news_data['currency'].isin([base_currency, quote_currency])) &
            (self.news_data['impact'] == 'High') &
            (self.news_data['datetime_utc'] >= start_of_zone) &
            (self.news_data['datetime_utc'] <= end_of_zone)
        ]

        if not relevant_news.empty:
            for _, row in relevant_news.iterrows():
                logging.warning(f"[{symbol}] High-impact news detected: {row['event_name']} ({row['currency']}) at {row['datetime_utc'].strftime('%H:%M')} UTC.")
            return True

        return False