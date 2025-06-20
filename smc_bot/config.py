"""
Smart Money Concepts (SMC) Configuration
Simplified configuration focusing on core SMC principles without ICT complexity.
"""

# Trading Symbols - Major forex pairs with good liquidity
SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 'GBPAUD', 'AUDCAD', 'EURCAD'
]

# Timeframes
TIMEFRAME_STR = "M15"  # Primary timeframe for entry
DATA_LOOKBACK = 500    # Candles to fetch for analysis

# Risk Management
RISK_PER_TRADE_PERCENT = 1.0      # Risk 1% per trade
MAX_SPREAD_POINTS = 30            # Maximum spread in points
MAX_TRADES_PER_DAY = 20            # Daily trade limit
MAX_CORRELATION_EXPOSURE = 2      # Max positions per currency
MIN_RR_RATIO = 1.5               # Minimum risk-reward ratio
MIN_CONFIDENCE = 0.7             # Minimum setup confidence score

# SMC Structure Parameters
SWING_LOOKBACK = 5               # Candles for swing high/low detection

# Trade Management
MIN_MINUTES_BETWEEN_TRADES = 30  # Avoid overtrading same symbol

# Bot Operation
LOOP_SLEEP_SECONDS = 60         # Check every minute
LOG_FILE = "smc_trading_bot.log"
LOG_LEVEL = "DEBUG"
MAGIC_NUMBER_PREFIX = 2025       # For identifying our trades

ATR_PERIOD = 14