import MetaTrader5 as mt5

# --- TRADING SETTINGS ---
SYMBOLS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'EURCAD', 'AUDCHF', 'AUDCAD', 'EURGBP', 'EURAUD', 
           'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDUSD']

TIME_FRAMES = (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)

# --- RISK MANAGEMENT SETTINGS ---
RISK_PER_TRADE_PERCENT = 1.0  # Risk 1% of account balance per trade
RISK_REWARD_RATIO = 2.0       # Aim for a 2:1 reward for our risk

# --- STRATEGY PARAMETERS ---
# Swing Point Detection
SWING_LOOKBACK_PERIOD = 20

# H1 Setup Zone Tolerance (as a factor of ATR)
# A value of 1.0 means the price must be within 1x ATR of the level.
PULLBACK_ZONE_ATR_FACTOR = 1.0 

# M15 Entry Trigger SL Buffer (in points)
# How many points below/above the M15 swing to place the stop loss.
STOP_LOSS_BUFFER_POINTS = 10

# --- SYSTEM SETTINGS ---
MAX_OPEN_POSITIONS_TOTAL = 10
MAX_OPEN_POSITIONS_PER_SYMBOL = 1
LOG_FILE_NAME = 'forex_trader_log.csv'
TRADE_LOG_FILE_NAME = 'trade_log.csv'
CHECK_INTERVAL_SECONDS = 300  # 5 minutes