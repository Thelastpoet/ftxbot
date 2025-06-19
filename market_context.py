"""
Market Context Analysis Module
"""
import csv
from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path

class MarketContext:
    def __init__(self, calendar_file='news_calendar.csv', auto_update=True):
        self.calendar_file = calendar_file
        self.news_events = []
        self._loaded = False
        
        # Check if file exists, create if not
        if not Path(calendar_file).exists():
            logging.warning(f"Calendar file '{calendar_file}' not found. Creating new calendar...")
            self._create_calendar()
        elif auto_update:
            self._check_and_update_calendar()
        
        self.news_events = self._load_news_calendar(calendar_file)
        self._build_currency_index()

    def _check_and_update_calendar(self):
        """Update calendar if it's older than 24 hours"""
        file_modified = datetime.fromtimestamp(Path(self.calendar_file).stat().st_mtime)
        if datetime.now() - file_modified > timedelta(hours=24):
            self._update_calendar()
    
    def _create_calendar(self):
        """Create a new calendar file"""
        try:
            from forex_factory_scraper import ForexFactoryScraper
            logging.info("Creating new news calendar...")
            scraper = ForexFactoryScraper(headless=True)
            scraper.update_calendar(days_ahead=7, days_back=1)
            logging.info("Calendar created successfully")
        except Exception as e:
            logging.error(f"Failed to create calendar: {e}")
            # Create empty file to prevent repeated attempts
            with open(self.calendar_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['datetime_utc', 'currency', 'impact', 'event_name'])
    
    def _update_calendar(self):
        """Run ForexFactory scraper to update calendar"""
        try:
            from forex_factory_scraper import ForexFactoryScraper
            logging.info("Updating news calendar...")
            scraper = ForexFactoryScraper(headless=True)
            scraper.update_calendar(days_ahead=7, days_back=1)
        except Exception as e:
            logging.error(f"Failed to update calendar: {e}")

    def _load_news_calendar(self, calendar_file):
        events = []
        try:
            with open(calendar_file, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    try:
                        # Only load high impact events
                        if row.get('impact', '').lower() != 'high':
                            continue
                            
                        event_time = datetime.strptime(row['datetime_utc'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                        events.append({
                            'time': event_time,
                            'currency': row['currency'].upper(),
                            'impact': row['impact'],
                            'name': row['event_name']
                        })
                    except (ValueError, KeyError) as e:
                        continue  # Skip without logging
            
            events.sort(key=lambda x: x['time'])
            
            # Only log once
            if not self._loaded:
                logging.info(f"Loaded {len(events)} high-impact events from {calendar_file}")
                self._loaded = True
                
            return events
            
        except FileNotFoundError:
            logging.warning(f"Calendar file '{calendar_file}' not found.")
            return []

    def _build_currency_index(self):
        """Build an index of events by currency for faster lookup"""
        self.events_by_currency = {}
        for event in self.news_events:
            currency = event['currency']
            if currency not in self.events_by_currency:
                self.events_by_currency[currency] = []
            self.events_by_currency[currency].append(event)
    
    def is_news_time(self, symbol, minutes_before=45, minutes_after=45, min_impact='High', current_time=None):
        if not self.news_events:
            return {'is_news': False}

        # Use passed-in time for analysis, or live time as a fallback
        now = current_time or datetime.now(timezone.utc)
        
        base_currency = symbol[:3].upper()
        quote_currency = symbol[3:6].upper()
        
        relevant_events = []
        for currency in [base_currency, quote_currency]:
            relevant_events.extend(self.events_by_currency.get(currency, []))
        
        for event in relevant_events:
            time_to_event = (event['time'] - now).total_seconds() / 60
            
            if -minutes_after <= time_to_event <= minutes_before:
                return {
                    'is_news': True,
                    'currency': event['currency'],
                    'impact': event['impact'],
                    'name': event['name'],
                    'minutes_to_event': round(time_to_event)
                }
        
        return {'is_news': False}

    @staticmethod
    def get_trading_session(current_time=None):
        """Determines the trading session. Uses provided time or live time."""
        now = current_time or datetime.now(timezone.utc)
        hour = now.hour
        
        # Define sessions based on actual forex market hours
        # Note: Sessions overlap in reality
        sessions = [
            # Peak volatility periods
            (13, 16, 'LONDON_NY_OVERLAP', 1.3, 'Peak Volatility'),  # Both London & NY open
            (7, 9, 'LONDON_TOKYO_OVERLAP', 1.1, 'High Volatility'),  # Both London & Tokyo open
            
            # Main sessions (non-overlapping portions)
            (9, 13, 'LONDON_MAIN', 1.1, 'Medium-High Volatility'),  # London only
            (16, 22, 'NY_MAIN', 1.0, 'Medium Volatility'),  # NY only
            (0, 7, 'TOKYO_MAIN', 0.8, 'Low-Medium Volatility'),  # Tokyo (& Sydney)
            
            # Low activity periods
            (22, 24, 'SYDNEY_OPEN', 0.7, 'Low Volatility'),  # Sydney alone
        ]
        
        # Handle sessions that cross midnight
        if hour >= 22 or hour < 6:
            # We're in Sydney session (21:00-06:00 UTC)
            if hour >= 22:
                return {'name': 'SYDNEY_OPEN', 'volatility_multiplier': 0.7, 'state': 'Low Volatility'}
            else:  # hour < 6
                # After midnight, Tokyo is also active
                return {'name': 'TOKYO_MAIN', 'volatility_multiplier': 0.8, 'state': 'Low-Medium Volatility'}
        
        # Check other sessions
        for start, end, name, mult, state in sessions:
            if start <= hour < end:
                return {
                    'name': name,
                    'volatility_multiplier': mult,
                    'state': state
                }
        
        # Fallback (shouldn't reach here)
        return {
            'name': 'OFF_HOURS',
            'volatility_multiplier': 0.5,
            'state': 'Market Closed'
        }
        
    @staticmethod
    def get_amd_session(current_time=None):
        """
        Determines the trading session based on the AMD (Accumulation, Manipulation, Distribution) model.
        - Asian Session (Accumulation): 22:00 - 09:00 UTC
        - London Session (Manipulation): 09:00 - 13:00 UTC
        - NY Session (Distribution): 13:00 - 22:00 UTC
        """
        now = current_time or datetime.now(timezone.utc)
        hour = now.hour

        if hour >= 22 or hour < 9:
            # Asian Session (crosses midnight)
            return {'name': 'ASIAN_SESSION'}
        elif 9 <= hour < 13:
            # London Session
            return {'name': 'LONDON_SESSION'}
        else: # Covers 13 <= hour < 22
            # New York Session
            return {'name': 'NY_SESSION'}
    
    def get_news_summary(self, hours_ahead=4, current_time=None):
        """Get upcoming high impact news within specified hours"""
        if not self.news_events:
            return []
            
        # Use passed-in time for analysis, or live time as a fallback
        now = current_time or datetime.now(timezone.utc)
        upcoming = []
        
        for event in self.news_events:
            time_diff = (event['time'] - now).total_seconds() / 3600
            if 0 <= time_diff <= hours_ahead and event['impact'] == 'High':
                upcoming.append({
                    'time': event['time'].strftime('%H:%M'),
                    'currency': event['currency'],
                    'event': event['name'],
                    'hours_until': round(time_diff, 1)
                })
        
        return upcoming
    
    def reload_calendar(self):
        """Force reload calendar from file"""
        self.news_events = self._load_news_calendar(self.calendar_file)