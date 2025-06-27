"""
ICT/SMC Configuration - UPDATED with proper ICT Kill Zone times
"""

# Trading Parameters
SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
 'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY',
'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD',
'GBPNZD', 'NZDUSD']
TIMEFRAME_STR = "M15"  # Primary timeframe
HIGHER_TIMEFRAME = "H4"  # For bias and key levels

# Risk Management
RISK_PER_TRADE_PERCENT = 1.0
MAX_SPREAD_POINTS = 50
MAX_TRADES_PER_DAY = 100 # Edited for demo
MAX_CORRELATION_EXPOSURE = 2

# ICT Structure Parameters
SWING_LOOKBACK = 5  # For swing high/low detection
STRUCTURE_LOOKBACK = 50  # For market structure analysis
DATA_LOOKBACK = 1000  # Total candles to fetch

# Stop Loss and Take Profit
SL_ATR_MULTIPLIER = 2.0  # ATR multiplier for stop loss
TP_RR_RATIO = 1.5 # Default risk:reward

# ICT Session Times (UTC) - ALIGNED WITH STANDARD ICT KILL ZONES
# These are the TRUE ICT Kill Zones as taught by Michael Huddleston
ICT_SESSIONS = {
    'Asian': {'start': 0, 'end': 6},      # UTC 00:00-04:00 (Asian Kill Zone)
    'London': {'start': 10, 'end': 13},     # UTC 06:00-09:00 (London Open Kill Zone) 
    'NewYork': {'start': 15, 'end': 18},  # UTC 11:00-14:00 (New York Kill Zone)
    'LondonClose': {'start': 17, 'end': 19}  # UTC 14:00-16:00 (London Close Kill Zone)
}

# Full Trading Sessions for reference (not kill zones)
FULL_SESSIONS = {
    'Sydney': {'start': 21, 'end': 6},    # UTC 21:00-06:00 (crosses midnight)
    'Tokyo': {'start': 0, 'end': 9},      # UTC 00:00-09:00
    'London': {'start': 7, 'end': 16},    # UTC 07:00-16:00
    'NewYork': {'start': 13, 'end': 22}   # UTC 13:00-22:00
}

# Bot Operation
LOOP_SLEEP_SECONDS = 60  # Check every minute during active sessions
LOG_FILE = "ict_trading_bot.log"
LOG_LEVEL = "INFO"
MAGIC_NUMBER_PREFIX = 2025

# Narrative Building Parameters
MIN_STRUCTURE_LOOKBACK = 20  # Minimum candles for structure analysis
JUDAS_SWING_LOOKBACK = 10  # Candles to check for Judas pattern
PO3_ANALYSIS_CANDLES = 24  # Candles for Power of Three analysis (6 hours on M15)

REQUIRE_STRUCTURE_ALIGNMENT = True
REQUIRE_KILLZONE = True

# ICT Entry Parameters
MIN_RETRACEMENT_PERCENT = 38.2  # Minimum retracement before entry
REQUIRE_ENTRY_CONFIRMATION = True  # Require rejection patterns
MIN_CONFIRMATIONS_REQUIRED = 1 
ALLOW_PRE_MANIPULATION_LEVELS = True
MAX_BARS_SINCE_MANIPULATION = 50  # Don't trade setups too old
ENTRY_CONFIRMATION_LOOKBACK = 3  # Candles to check for confirmation
RETRACEMENT_THRESHOLD_PERCENT = 25.0 
ALLOW_MANIPULATION_PHASE_ENTRY = True
MICRO_STRUCTURE_LOOKBACK = 15  # For structure continuation
APPROACHING_LEVEL_TOLERANCE = 0.0005  # 5 pips

MIN_TARGET_RR = 1.0