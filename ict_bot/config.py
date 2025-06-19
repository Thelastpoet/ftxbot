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
SWING_LOOKBACK = 10  # For swing high/low detection
STRUCTURE_LOOKBACK = 50  # For market structure analysis
DATA_LOOKBACK = 200  # Total candles to fetch

# Stop Loss and Take Profit
SL_ATR_MULTIPLIER = 1.5  # ATR multiplier for stop loss
TP_RR_RATIO = 2.0  # Default risk:reward

# ICT Session Times (UTC)
ICT_SESSIONS = {
    'asian': {'start': 0, 'end': 6},      # 00:00-03:00 UTC but in EET and EAT for Broker
    'london': {'start': 10, 'end': 13},    # 07:00-10:00 UTC but in EET and EAT for Broker
    'ny': {'start': 15, 'end': 18}        # 12:00-15:00 UTC but in EET and EAT for Broker
}

# Bot Operation
LOOP_SLEEP_SECONDS = 300  # Check every minute during active sessions
LOG_FILE = "ict_trading_bot.log"
LOG_LEVEL = "INFO"
MAGIC_NUMBER_PREFIX = 2025

# Narrative Building Parameters
MIN_STRUCTURE_LOOKBACK = 20  # Minimum candles for structure analysis
JUDAS_SWING_LOOKBACK = 10  # Candles to check for Judas pattern
PO3_ANALYSIS_CANDLES = 24  # Candles for Power of Three analysis (6 hours on M15)

REQUIRE_STRUCTURE_ALIGNMENT = True
REQUIRE_KILLZONE = True