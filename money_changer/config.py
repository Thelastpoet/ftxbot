import pytz
from datetime import time

"""
ICT/SMC Configuration - FIXED with correct ICT timings
"""

# Trading Parameters
SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
           'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY',
           'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD',
           'GBPNZD', 'NZDUSD', 'GOLD', 'SILVER']

TIMEFRAME_STR = "M15"  # Primary timeframe

# Risk Management
RISK_PER_TRADE_PERCENT = 1.0
MAX_SPREAD_POINTS = 50
MAX_TRADES_PER_DAY = 100  # For demo testing
MAX_CORRELATION_EXPOSURE = 2

# ICT Structure Parameters
SWING_LOOKBACK = 5 # For swing high/low detection
STRUCTURE_LOOKBACK = 50  # For market structure analysis
DATA_LOOKBACK = 1000  # Total candles to fetch

# Stop Loss and Take Profit
SL_ATR_MULTIPLIER = 2.0  # ATR multiplier for stop loss
TP_RR_RATIO = 1.5  # Default risk:reward
MIN_TARGET_RR = 1.0  # Minimum acceptable R:R

# ICT Kill Zone Times (in NY time) - CORRECT ICT TIMINGS
ICT_ASIAN_RANGE = {
    'start': time(19, 0),   # 7 PM NY time
    'end': time(22, 0)      # 10 PM NY time
}

ICT_LONDON_KILLZONE = {
    'start': time(2, 0),    # 2 AM NY time
    'end': time(5, 0)       # 5 AM NY time
}

ICT_NEWYORK_KILLZONE = {
    'start': time(8, 30),   # 8:30 AM NY time
    'end': time(11, 0)      # 11 AM NY time
}

ICT_LONDON_CLOSE_KILLZONE = {
    'start': time(15, 0),   # 3 PM NY time
    'end': time(17, 0)      # 5 PM NY time
}

# Timezone settings
NY_TIMEZONE = pytz.timezone('America/New_York')
BROKER_TIMEZONE = 'Etc/GMT-3'  # For UTC+3 broker

# Bot Operation
LOOP_SLEEP_SECONDS = 60  # Check every minute during active sessions
LOG_FILE = "ict_trading_bot.log"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
MAGIC_NUMBER_PREFIX = 2025

# Entry Configuration
REQUIRE_KILLZONE = True  # Only trade during kill zones
REQUIRE_ENTRY_CONFIRMATION = True  # Require rejection patterns
MIN_CONFIRMATIONS_REQUIRED = 1  # Number of confirmations needed
ENTRY_CONFIRMATION_LOOKBACK = 3  # Candles to check for confirmation

# Retracement and Market Phase
RETRACEMENT_THRESHOLD_PERCENT = 10.0  # Minimum retracement for entry
ALLOW_MANIPULATION_PHASE_ENTRY = True  # Allow entries during manipulation phase

# Note: The following ICT_SESSIONS can be used for reference but are not the same as kill zones
# Kill zones are specific high-probability windows, while these are broader session times
ICT_SESSIONS = {
    'Asian': {'start': 0, 'end': 5},       # Midnight-5 AM NY (includes Asian range)
    'London': {'start': 3, 'end': 5},      # 3-5 AM NY (London Kill Zone)
    'NewYork': {'start': 7, 'end': 10},    # 7-10 AM NY (New York Kill Zone)
    'LondonClose': {'start': 14, 'end': 17}  # 2-5 PM NY (London Close)
}

# Position Management Configuration
ENABLE_POSITION_MANAGEMENT = True
ENABLE_SCALING = True                    # Allow adding to positions
ENABLE_PARTIAL_EXITS = True              # Allow taking partial profits

# Risk Management Ratios (Based on ICT Methodology)
BREAK_EVEN_RR = 0.5                     # Move to break-even at 0.5R
RISK_FREE_RR = 1.0                      # Move to risk-free at 1.0R  
FIRST_TARGET_RR = 1.5                   # First profit target at 1.5R
RUNNER_PERCENTAGE = 25                  # Keep 25% as runner position

# Position Scaling Settings
MAX_POSITIONS_PER_SYMBOL = 1            # Max positions per symbol
MAX_TOTAL_RISK_PERCENT = 5.0            # Max total account risk
MAX_SCALE_INS = 2                       # Max number of scale-ins
SCALE_VOLUME_RATIO = 0.5                # Scale-in size as ratio of initial

# Advanced Settings
TRAILING_STOP_ENABLED = True            # Enable trailing stops for runners
STRUCTURE_EXIT_ENABLED = True           # Exit on adverse structure breaks
SESSION_BASED_MANAGEMENT = True         # Adjust management based on sessions
LIQUIDITY_TARGET_PRIORITY = True        # Prioritize liquidity-based exits

# Position Management Update Frequency
POSITION_UPDATE_INTERVAL = 30           # Update positions every 30 seconds