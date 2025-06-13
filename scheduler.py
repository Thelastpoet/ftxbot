"""
Schedule daily updates of the ForexFactory calendar
"""
import schedule
import time
import logging
from forex_factory_scraper import daily_update

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('calendar_updates.log'),
        logging.StreamHandler()
    ]
)

def run_update():
    """Wrapper for daily update with error handling"""
    try:
        logging.info("Starting scheduled calendar update...")
        daily_update()
        logging.info("Calendar update completed successfully")
    except Exception as e:
        logging.error(f"Calendar update failed: {e}")

# Schedule daily update at 3 AM UTC (less traffic)
schedule.every().day.at("03:00").do(run_update)

# Also run on startup
run_update()

logging.info("Calendar update scheduler started. Running daily at 03:00 UTC")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute