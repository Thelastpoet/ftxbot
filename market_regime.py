"""
Market Regime Detection Module

This standalone module is responsible for classifying the current market state
based on key metrics like trend strength (ADX), volatility (ATR), and price momentum.

It is designed to be imported by a trading bot to allow for adaptive strategies
without cluttering the main execution file. The logic is encapsulated in a
static method for easy, state-free use.
"""

class MarketRegimeDetector:
    """
    Analyzes market data to determine the current regime and returns a set of
    adaptive parameters suitable for that regime.
    """
    
    @staticmethod
    def classify_market(adx, atr_ratio, price_position):
        
        
        # --- PRIORITY 1: Check for Extreme/Climactic Conditions ---
        
        # Regime: TREND_EXHAUSTION
        if adx > 60:
            return 'TREND_EXHAUSTION', {
                'fib_levels': [0.5, 0.618],   
                'confluence_required': 0.95,
                'entry_methods': [],
                'risk_multiplier': 0.6,
                'comment': 'Trend is over-extended, high risk of reversal.'
            }
        
        # Regime: VOLATILE_EXPANSION
        if atr_ratio > 1.6:
            return 'VOLATILE_EXPANSION', {
                'fib_levels': [],  
                'confluence_required': 1.0,
                'entry_methods': [],
                'risk_multiplier': 0.5,
                'comment': 'Sudden volatility spike. Market is unpredictable.'
            }

        # --- PRIORITY 2: Check for Standard Trend Conditions ---

        # Regime: STRONG_TREND
        if adx > 40 and (price_position > 0.8 or price_position < 0.2):
            return 'STRONG_TREND', {
                'fib_levels': [0.236, 0.382],    # Expect shallow retracements
                'confluence_required': 0.6,     # Less confluence needed as trend is strong
                'entry_methods': ['momentum', 'shallow_pullback', 'ma_bounce'],
                'risk_multiplier': 1.2,         # Increase risk to capitalize on strong moves
                'comment': 'Strong, healthy trend. Look for continuation.'
            }

        # Regime: NORMAL_TREND
        elif adx > 25:
            return 'NORMAL_TREND', {
                'fib_levels': [0.382, 0.5, 0.618],
                'confluence_required': 0.8, 
                'entry_methods': ['fibonacci', 'structure_break', 'ma_bounce'],
                'risk_multiplier': 1.0,
                'comment': 'Clear directional trend. Look for pullbacks.'
            }

        # --- PRIORITY 3: Check for Non-Trending Conditions ---

        # Regime: RANGING
        elif adx < 20:
            return 'RANGING', {
                'fib_levels': [0.5, 0.618, 0.786],
                'confluence_required': 0.9,
                'entry_methods': ['fibonacci', 'range_extreme'],
                'risk_multiplier': 0.8,
                'comment': 'Non-directional market. Trade between extremes.'
            }

        # --- PRIORITY 4: Fallback Condition ---
        
        # Regime: UNCLEAR
        else:
            return 'UNCLEAR', {
                'fib_levels': [],               # No reliable patterns
                'confluence_required': 1.0,     # Effectively disables trading
                'entry_methods': [],            # No valid entry methods
                'risk_multiplier': 0.7,         # Reduce risk
                'comment': 'Market direction is unclear. Best to wait.'
            }
