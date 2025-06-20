"""
Simplified ICT/SMC Configuration
Only includes parameters that are actively used by the trading system.
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

# ICT Session Times (UTC)
ICT_SESSIONS = {
    'asian': {'start': 0, 'end': 6},      # 00:00-03:00 UTC but in EET and EAT for Broker
    'london': {'start': 10, 'end': 13},    # 07:00-10:00 UTC but in EET and EAT for Broker
    'ny': {'start': 15, 'end': 18}        # 12:00-15:00 UTC but in EET and EAT for Broker
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