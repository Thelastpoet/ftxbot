
import logging
import logging.handlers
import pandas as pd
from pathlib import Path

# --- Constants ---
LOG_DIR = Path(__file__).parent
OPERATIONS_LOG_FILE = LOG_DIR / "ict_trading_bot.log"
TRADE_JOURNAL_FILE = LOG_DIR / "trade_journal.csv"
LOG_LEVEL = "DEBUG"
MAX_LOG_SIZE_MB = 5
BACKUP_COUNT = 3

# --- Setup for Operations Log ---
def setup_operations_logger():
    """Sets up the main logger for bot operations."""
    op_logger = logging.getLogger("bot_operations")
    op_logger.setLevel(LOG_LEVEL)
    
    # Prevent duplicate handlers if called multiple times
    if op_logger.hasHandlers():
        op_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler (Rotating)
    file_handler = logging.handlers.RotatingFileHandler(
        OPERATIONS_LOG_FILE, 
        maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024, 
        backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    op_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    op_logger.addHandler(console_handler)
    
    return op_logger

# --- Functions for Trade Journal (CSV) ---
def initialize_trade_journal():
    """
    Creates the trade journal CSV with headers if it doesn't exist or is empty.
    """
    headers = [
        "TimestampUTC", "Ticket", "Symbol", "Type", "Volume", 
        "Price", "SL", "TP", "MagicNumber", "Comment", "Status",
        "Narrative", "EntryModel", "DailyBias", "PO3Phase",
        "ClosingPrice", "Profit", "MaxFavorableExcursion", "MaxAdverseExcursion"
    ]
    # Check if file exists and is not empty
    if not TRADE_JOURNAL_FILE.exists() or TRADE_JOURNAL_FILE.stat().st_size == 0:
        pd.DataFrame(columns=headers).to_csv(TRADE_JOURNAL_FILE, index=False)
        logging.getLogger("bot_operations").info(f"Initialized trade journal: {TRADE_JOURNAL_FILE}")

def log_trade_event(trade_data: dict):
    """
    Logs a trade event to the CSV journal.
    'trade_data' should be a dictionary matching the journal headers.
    """
    op_logger = logging.getLogger("bot_operations")
    try:
        # Define headers internally to avoid reading the file if it's empty
        headers = [
            "TimestampUTC", "Ticket", "Symbol", "Type", "Volume", 
            "Price", "SL", "TP", "MagicNumber", "Comment", "Status",
            "Narrative", "EntryModel", "DailyBias", "PO3Phase",
            "ClosingPrice", "Profit", "MaxFavorableExcursion", "MaxAdverseExcursion"
        ]
        
        # Ensure all headers are present in the trade_data dictionary
        for header in headers:
            if header not in trade_data:
                trade_data[header] = None
        
        # Create a DataFrame from the single trade event, ensuring column order
        trade_df = pd.DataFrame([trade_data], columns=headers)
        
        # Check if the file is empty to decide whether to write headers
        file_is_empty = not TRADE_JOURNAL_FILE.exists() or TRADE_JOURNAL_FILE.stat().st_size == 0
        
        # Append to the CSV file
        trade_df.to_csv(TRADE_JOURNAL_FILE, mode='a', header=file_is_empty, index=False)
        
    except Exception as e:
        # Fallback to the operations logger if CSV logging fails
        op_logger.error(f"Failed to write to trade_journal.csv: {e}")
        op_logger.error(f"Trade Data: {trade_data}")

# --- Initial Setup ---
# Initialize the logger and journal on module import
operations_logger = setup_operations_logger()
initialize_trade_journal()
