"""
ForexFactory Economic Calendar Scraper
Scrapes economic events and saves them in CSV format for trading bot consumption
"""
import csv
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import random
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ForexFactoryScraper:
    """Scrapes ForexFactory calendar with anti-detection measures"""
    
    def __init__(self, headless=False):
        self.headless = headless
        self.base_url = "https://www.forexfactory.com/calendar"
        self.events = []
        self.driver = None
        
        # Set up timezone objects
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC
        
    def setup_driver(self):
        """Setup undetected Chrome driver with anti-detection features"""
        options = uc.ChromeOptions()
        
        # Anti-detection measures
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        if self.headless:
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
        # Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = uc.Chrome(options=options)
            logging.info("Chrome driver initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Chrome driver: {e}")
            raise
            
    def random_delay(self, min_seconds=1, max_seconds=3):
        """Add random delay to mimic human behavior"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)
        
    def get_impact_level(self, element):
        """Extract impact level from the impact cell"""
        try:
            # Look for impact icon classes
            if element.find_elements(By.CLASS_NAME, "icon--ff-impact-red"):
                return "High"
            elif element.find_elements(By.CLASS_NAME, "icon--ff-impact-ora"):
                return "Medium"
            elif element.find_elements(By.CLASS_NAME, "icon--ff-impact-yel"):
                return "Low"
            else:
                return "Holiday"
        except:
            return "Unknown"
            
    def scrape_date_range(self, start_date, end_date):
        """Scrape events for a date range"""
        self.setup_driver()
        
        try:
            current_date = start_date
            
            while current_date <= end_date:
                self.scrape_single_day(current_date)
                current_date += timedelta(days=1)
                self.random_delay(2, 4)  # Delay between days
                
        finally:
            if self.driver:
                self.driver.quit()
                
    def scrape_single_day(self, target_date):
        """Scrape events for a specific date"""
        date_str = target_date.strftime("%b%d.%Y").lower()
        url = f"{self.base_url}?day={date_str}"
        
        logging.info(f"Scraping {url}")
        
        try:
            self.driver.get(url)
            self.random_delay(2, 4)
            
            # Wait for calendar to load
            wait = WebDriverWait(self.driver, 15)
            calendar_table = wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "calendar__table"))
            )
            
            # Find all calendar rows
            rows = calendar_table.find_elements(By.TAG_NAME, "tr")
            
            current_time = None
            
            for row in rows:
                try:
                    # Skip non-event rows
                    if "calendar__row" not in row.get_attribute("class"):
                        continue
                        
                    # Extract time if present
                    time_cell = row.find_elements(By.CLASS_NAME, "calendar__time")
                    if time_cell and time_cell[0].text.strip():
                        time_text = time_cell[0].text.strip()
                        if time_text and not any(icon in time_text for icon in ['🔄', '📅']):
                            current_time = time_text
                            
                    # Skip if no time is set yet
                    if not current_time:
                        continue
                        
                    # Extract currency
                    currency_cell = row.find_elements(By.CLASS_NAME, "calendar__currency")
                    if not currency_cell:
                        continue
                    currency = currency_cell[0].text.strip()
                    
                    # Extract impact
                    impact_cell = row.find_elements(By.CLASS_NAME, "calendar__impact")
                    impact = self.get_impact_level(impact_cell[0]) if impact_cell else "Unknown"
                    
                    # Extract event name
                    event_cell = row.find_elements(By.CLASS_NAME, "calendar__event")
                    if not event_cell:
                        continue
                    event_name = event_cell[0].text.strip()
                    
                    # Extract values
                    actual_cell = row.find_elements(By.CLASS_NAME, "calendar__actual")
                    actual = actual_cell[0].text.strip() if actual_cell else ""
                    
                    forecast_cell = row.find_elements(By.CLASS_NAME, "calendar__forecast")
                    forecast = forecast_cell[0].text.strip() if forecast_cell else ""
                    
                    previous_cell = row.find_elements(By.CLASS_NAME, "calendar__previous")
                    previous = previous_cell[0].text.strip() if previous_cell else ""
                    
                    # Convert to UTC using proper timezone handling
                    event_datetime = self.parse_datetime(target_date, current_time)
                    
                    event = {
                        'datetime_utc': event_datetime.strftime('%Y-%m-%d %H:%M'),
                        'currency': currency,
                        'impact': impact,
                        'event_name': event_name,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous
                    }
                    
                    self.events.append(event)
                    
                except Exception as e:
                    logging.warning(f"Error parsing row: {e}")
                    continue
                    
        except TimeoutException:
            logging.error(f"Timeout loading page for {target_date}")
        except Exception as e:
            logging.error(f"Error scraping {target_date}: {e}")
            
    def parse_datetime(self, date, time_str):
        """Parse ForexFactory time to UTC datetime using proper timezone conversion"""
        if time_str.lower() == "all day":
            time_str = "11:59pm"
            
        try:
            time_parts = time_str.replace('am', ' AM').replace('pm', ' PM')
            datetime_str = f"{date.strftime('%Y-%m-%d')} {time_parts}"
            
            naive_dt = datetime.strptime(datetime_str, '%Y-%m-%d %I:%M %p')
            eastern_dt = self.eastern_tz.localize(naive_dt)
            utc_dt = eastern_dt.astimezone(self.utc_tz)
            
            return utc_dt
            
        except Exception as e:
            logging.warning(f"Error parsing time '{time_str}': {e}")
            return date.replace(hour=12, minute=0, tzinfo=timezone.utc)
            
    def save_to_csv(self, filename='news_calendar.csv'):
        """Save scraped events to CSV"""
        self.events.sort(key=lambda x: x['datetime_utc'])
        
        fieldnames = ['datetime_utc', 'currency', 'impact', 'event_name', 'actual', 'forecast', 'previous']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
                
        logging.info(f"Saved {len(self.events)} events to {filename}")
        
    def update_calendar(self, days_ahead=7, days_back=1):
        """Update calendar with recent and upcoming events"""
        end_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back) 
        
        logging.info(f"Updating calendar from {start_date.date()} to {end_date.date()}")
        
        self.scrape_date_range(start_date, end_date)
        self.save_to_csv()

def daily_update():
    """Function to be called daily to update the calendar"""
    scraper = ForexFactoryScraper(headless=True)
    try:
        scraper.update_calendar(days_ahead=7, days_back=1)
    except Exception as e:
        logging.error(f"Failed to update calendar: {e}")

if __name__ == "__main__":
    daily_update()