"""
Configuration settings for the MT5 SMC Bot.
This bot attempts to connect to an already running and logged-in MetaTrader 5 terminal
by calling mt5.initialize() without any arguments.
Ensure your MT5 terminal is open, logged in, and "Allow algorithmic trading" is enabled.
"""

# Trading Parameters
SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
 'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY',
'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD',
'GBPNZD', 'NZDUSD']
TIMEFRAME_STR = "M15" # M1, M5, M15, M30, H1, H4, D1
RISK_PER_TRADE_PERCENT = 1.0  # e.g., 1.0 for 1% risk
MAX_SPREAD_POINTS = 50 # Maximum allowed spread in points
SL_POINTS_DEFAULT = 15 # Default Stop Loss buffer in points
TP_RR_RATIO = 2.0 # Risk/Reward Ratio

# SMC Parameters - FIXED FOR M15 TIMEFRAME
SMC_SWING_LOOKBACK = 6
SMC_STRUCTURE_LOOKBACK = 50 # Keep at 50 for broader structure view
PREMIUM_DISCOUNT_FIB_LEVEL = 0.5

# === PROFESSIONAL SMC ENHANCEMENTS ===

# Confluence Requirements (minimum factors needed for signal)
MIN_CONFLUENCE_SCORE = 1

# Order Block Settings
OB_LOOKBACK_CANDLES = 20    # How far back to look for fresh Order Blocks
OB_PROXIMITY_TOLERANCE = 0.001  # 0.1% tolerance for OB entry zones

# Fair Value Gap Settings  
FVG_LOOKBACK_CANDLES = 50   # How far back to look for open FVGs
FVG_JOIN_CONSECUTIVE = True # Merge consecutive FVGs for cleaner signals

# Liquidity Settings
LIQUIDITY_RANGE_PERCENT = 0.001  # 1% range for liquidity clusters (tight for forex)

# Risk Management Enhancements
MAX_RISK_PER_SESSION = 3.0    # Max 3% risk per session (prevents overtrading)
MAX_TRADES_PER_SESSION = 2    # Max 2 trades per killzone
CORRELATION_FILTER = True     # Avoid trading correlated pairs simultaneously

# Bot Operation Parameters
LOOP_SLEEP_SECONDS = 300  # 5 minutes
LOG_FILE = "mt5_smc_bot_professional.log"
LOG_LEVEL = "INFO"  # Switch back to INFO once debugging is done
MAGIC_NUMBER_PREFIX = 202400

# Higher timeframe
HIGHER_TIMEFRAME = "4h"

SL_ATR_MULTIPLIER = 1.2

# === TIMEFRAME-SPECIFIC SWING LOOKBACK GUIDE ===
# M1:  swing_lookback = 3-5   (looks at 6-10 candles total)
# M5:  swing_lookback = 4-6   (looks at 8-12 candles total)
# M15: swing_lookback = 5-8   (looks at 10-16 candles total)
# M30: swing_lookback = 8-12  (looks at 16-24 candles total)
# H1:  swing_lookback = 10-15 (looks at 20-30 candles total)
# H4:  swing_lookback = 15-20 (looks at 30-40 candles total)
# D1:  swing_lookback = 20-30 (looks at 40-60 candles total)