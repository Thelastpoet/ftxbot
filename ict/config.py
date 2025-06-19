"""
Enhanced ICT/SMC Configuration for Professional Trading
Implements true Inner Circle Trader concepts with Market Maker Models,
Power of Three, Optimal Trade Entry, and Judas Swing detection.

--- DEMO TESTING CONFIGURATION ---
This version is modified to run continuously with relaxed risk and trade limits.
"""

# Trading Parameters
SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY',
           'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY',
           'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD',
           'GBPNZD', 'NZDUSD']

TIMEFRAME_STR = "M15"  # Primary timeframe for entry
RISK_PER_TRADE_PERCENT = 1.0  # Risk per trade
MAX_SPREAD_POINTS = 50  # Maximum allowed spread in points

# === ICT SPECIFIC PARAMETERS ===

# Optimal Trade Entry (OTE) Fibonacci Levels
OTE_LEVELS = {
    'shallow': 0.62,    # 62% retracement
    'sweet': 0.705,     # 70.5% - ICT sweet spot
    'deep': 0.79        # 79% retracement
}

# Premium/Discount Zones (Multiple Fibonacci levels)
FIBONACCI_ZONES = {
    'deep_discount': 0.20,      # Below 20% - extreme discount
    'discount': 0.25,           # 25% level
    'equilibrium': 0.50,        # 50% - fair value
    'ote_zone': [0.62, 0.79],   # OTE zone range
    'premium': 0.75,            # 75% level  
    'deep_premium': 0.80        # Above 80% - extreme premium
}

# ICT Kill Zones (High probability time windows in UTC)
ICT_KILL_ZONES = {
    'asian_open': {
        'start': 0,     # 00:00 UTC
        'end': 3,       # 03:00 UTC
        'priority': 'medium'
    },
    'london_open': {
        'start': 7,     # 07:00 UTC (08:00 London time)
        'end': 10,      # 10:00 UTC (11:00 London time)
        'priority': 'high'
    },
    'ny_open': {
        'start': 12,    # 12:00 UTC (08:00 NY time)
        'end': 15,      # 15:00 UTC (11:00 NY time)
        'priority': 'high'
    },
    'london_close': {
        'start': 15,    # 15:00 UTC
        'end': 17,      # 17:00 UTC
        'priority': 'medium'
    }
}

# Market Maker Model Parameters
MM_MODEL_CONFIG = {
    'asian_range_hours': 7,         # Hours to consider for Asian range
    'manipulation_threshold': 0.5,   # Multiplier of Asian range for manipulation
    'distribution_target': 1.0       # Multiplier for distribution projection
}

# Power of Three (PO3) Settings
PO3_CONFIG = {
    'lookback_candles': 20,         # Candles to analyze for PO3
    'accumulation_threshold': 0.3,   # Max range for accumulation phase (30% of total)
    'segments': 3                    # Divide range into 3 segments
}

# Judas Swing Parameters
JUDAS_SWING_CONFIG = {
    'initial_range_candles': 2,      # First X candles to establish initial range
    'reversal_strength': 1.0,        # Minimum reversal candle size vs sweep distance
    'lookback': 10                   # Candles to look for Judas pattern
}

# === ENHANCED SMC PARAMETERS ===

# Structure Analysis
SMC_SWING_LOOKBACK = 6               # For M15 timeframe
SMC_STRUCTURE_LOOKBACK = 50          # Broader structure context
PREMIUM_DISCOUNT_FIB_LEVEL = 0.5     # Primary equilibrium level

# Order Block Refinement
OB_REFINEMENT = {
    'require_fvg': True,             # Require FVG before OB
    'volume_threshold': 1.5,         # Volume must be 1.5x average
    'lookback_candles': 50,          # How far to look for OBs
    'fresh_only': True               # Prioritize unmitigated OBs
}

# Fair Value Gap Settings
FVG_CONFIG = {
    'join_consecutive': True,        # Merge consecutive FVGs
    'lookback_candles': 50,          # FVG search range
    'mitigation_type': 'wick'        # Use wicks for mitigation
}

# Liquidity Detection
LIQUIDITY_CONFIG = {
    'range_percent': 0.001,          # Base range for liquidity clusters
    'min_touches': 2,                # Minimum touches to form liquidity
    'sweep_reversal_candles': 2      # Candles to confirm sweep reversal
}

# Stop Loss and Take Profit
SL_POINTS_DEFAULT = 15               # Fallback SL buffer in points
SL_ATR_MULTIPLIER = 1.2             # ATR multiplier for dynamic SL
TP_RR_RATIO = 2.0                   # Base Risk:Reward ratio

# ICT Target Methods
TP_METHODS = {
    'fixed_rr': 2.0,                # Fixed R:R
    'next_liquidity': True,         # Target next liquidity zone
    'mm_projection': True,          # Use Market Maker projections
    'structure_based': True         # Target next structure level
}

# === RISK MANAGEMENT ===

# Session Limits
MAX_RISK_PER_SESSION = 99.0         # Max 99% risk per session (effectively disabled) # <<< CHANGED FOR DEMO TESTING
MAX_TRADES_PER_SESSION = 99         # Max trades per kill zone (effectively disabled)  # <<< CHANGED FOR DEMO TESTING
MAX_TRADES_PER_DAY = 99             # Daily trade limit (effectively disabled)        # <<< CHANGED FOR DEMO TESTING

# Correlation Management
CORRELATION_FILTER = False          # Avoid correlated pairs (disabled for testing) # <<< CHANGED FOR DEMO TESTING
MAX_CORRELATION_EXPOSURE = 10       # Max positions in correlated pairs (set high)  # <<< CHANGED FOR DEMO TESTING

# === BOT OPERATION ===

LOOP_SLEEP_SECONDS = 900             # 1 minute between iterations for faster testing # <<< CHANGED FOR DEMO TESTING
LOG_FILE = "mt5_ict_smc_professional.log"
LOG_LEVEL = "INFO"
MAGIC_NUMBER_PREFIX = 202401        # Updated prefix for ICT version

# Higher Timeframe Analysis
HIGHER_TIMEFRAME = "4h"             # For bias and key levels

# === ENTRY MODEL PRIORITIES ===
# Which ICT entry models to use (in order of priority)
ENTRY_MODELS = [
    'judas_swing',          # Highest priority - clear stop hunt reversal
    'market_maker_model',   # Daily bias model
    'ote_refined_ob',       # OTE with refined order block
    'enhanced_sweep'        # Liquidity sweep with reversal
]

# === TRADE FILTERS ===

# Only trade if these conditions are met
TRADE_FILTERS = {
    'require_killzone': False,       # Must be in kill zone (False = 24/5 trading)
    'require_session': False,        # Must be in active session (disabled for 24/5) # <<< CHANGED FOR DEMO TESTING
    'require_structure': True,       # Must have recent BOS/CHoCH (Core logic, keep enabled)
    'require_pd_zone': True,         # Must be in premium/discount (Core logic, keep enabled)
    'min_ict_confluence': 2          # Minimum ICT factors needed
}

# === TIMEFRAME-SPECIFIC ADJUSTMENTS ===
# Fine-tuning for M15 timeframe
M15_ADJUSTMENTS = {
    'judas_initial_candles': 2,      # First 30 mins of session
    'po3_segments': 12,              # 3 hours divided into 3 segments
    'ote_precision': 0.0005          # 5 pip precision for OTE levels
}