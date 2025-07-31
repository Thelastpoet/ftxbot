import pytz
from datetime import time

# Trading Parameters
SYMBOLS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'NZDUSD', 'XAUUSD']
TIMEFRAME_STR = "M15"

# Risk Management
RISK_PER_TRADE_PERCENT = 1.0
MAX_TRADES_PER_DAY = 100
MAX_CORRELATION_EXPOSURE = 2

# Structure Parameters
SWING_LOOKBACK = 20
STRUCTURE_LOOKBACK = 50
DATA_LOOKBACK = 1000

# Stop Loss and Take Profit
SL_ATR_MULTIPLIER = 2.0
TP_RR_RATIO = 1.5
MIN_TARGET_RR = 1.0

# Kill Zone Times (New York Time)
ICT_ASIAN_RANGE = {'start': time(20, 0), 'end': time(2, 0)}
ICT_LONDON_KILLZONE = {'start': time(2, 0), 'end': time(5, 0)}
ICT_NEW_YORK_KILLZONE = {'start': time(7, 0), 'end': time(10, 0)} 
ICT_LONDON_CLOSE_KILLZONE = {'start': time(10, 0), 'end': time(12, 0)}

# Timezone settings
NY_TIMEZONE = pytz.timezone('America/New_York')
BROKER_TIMEZONE = 'Etc/GMT-3'

# Bot Operation
LOOP_SLEEP_SECONDS = 60
LOG_FILE = "ict_trading_bot.log"
LOG_LEVEL = "INFO"
MAGIC_NUMBER_PREFIX = 2025

# Entry Configuration
REQUIRE_KILLZONE = False
REQUIRE_ENTRY_CONFIRMATION = True
ALLOW_MANIPULATION_PHASE_ENTRY = True

# Broader Session Times
ICT_SESSIONS = {
    'Asian': {'start': 0, 'end': 5},
    'London': {'start': 3, 'end': 5},
    'NewYork': {'start': 7, 'end': 10},
    'LondonClose': {'start': 14, 'end': 17}
}

# Maximum Acceptable Spread in Points
MAX_SPREAD_POINTS = {
    'AUDUSD': 20,
    'EURUSD': 15,
    'GBPUSD': 20,
    'USDCAD': 25,
    'USDCHF': 20,
    'USDJPY': 15,
    'NZDUSD': 25,
    'XAUUSD': 35, 
    'DEFAULT': 50
}
