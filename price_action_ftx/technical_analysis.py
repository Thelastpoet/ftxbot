import traceback
from talib import (
    CDLENGULFING, CDLDOJI, CDLSHOOTINGSTAR, CDLHAMMER, 
    CDLMORNINGSTAR, CDLPIERCING, CDLHARAMI, CDLHARAMICROSS,
    CDL3WHITESOLDIERS, CDLMORNINGDOJISTAR,
    CDLEVENINGSTAR, CDLDARKCLOUDCOVER, CDL3BLACKCROWS, CDLEVENINGDOJISTAR,
    MFI
)
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self, config=None):
        self.config = config or {}
        mfi_params = self.config.get('mfi', {})
        self.mfi_period = mfi_params.get('period', 14)

    def calculate_candlestick_patterns(self, data):
        try:
            if data is None or data.empty:
                logging.error("No data provided for candlestick pattern calculation")
                return None
            df = data.copy()
            
            if 'tick_volume' in df.columns:
                df['mfi'] = MFI(df['high'], df['low'], df['close'], df['tick_volume'].astype(float), timeperiod=self.mfi_period)
            else:
                logging.warning("Tick volume not found in data, MFI will not be calculated")
            
            # Calculate patterns
            df['cdl_engulfing'] = CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_doji'] = CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_shooting_star'] = CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_hammer'] = CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morning_star'] = CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_piercing'] = CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_harami'] = CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_harami_cross'] = CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3_white_soldiers'] = CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morning_doji_star'] = CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_evening_star'] = CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_dark_cloud'] = CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3_black_crows'] = CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_evening_doji_star'] = CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
            return df
        except Exception as e:
            logging.error(f"Error in calculate_candlestick_patterns: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    def check_entry_patterns(self, data_with_indicators, direction, pullback_level=None, atr=None):
        """
        Checks for reversal patterns with context filters:
        - MFI 50-level cross for momentum confirmation.
        - Proximity to H1 pullback level.
        - Candlestick pattern must exist.
        - Candlestick must show a strong close in the intended direction.
        """
        # We need at least 3 bars to check the MFI cross on the last closed bar
        if data_with_indicators is None or len(data_with_indicators) < self.mfi_period + 2:
            return False

        last_closed = data_with_indicators.iloc[-2]
        
        candle_range = last_closed['high'] - last_closed['low']
        if candle_range == 0: return False
        
        if atr is None or atr <= 0:
            atr = candle_range * 0.5  
            
        if pullback_level is not None:
            near_level = abs((last_closed['low'] if direction == 'buy' else last_closed['high']) - pullback_level) <= atr
            if not near_level:
                return False
        
        # 3. Candlestick Pattern & Strong Close Confirmation
        is_pattern_valid = False
        
        if direction == 'buy':
            is_pattern_valid = (
                last_closed['cdl_engulfing'] == 100 or
                last_closed['cdl_hammer'] == 100 or
                last_closed['cdl_morning_star'] == 100 or
                last_closed['cdl_piercing'] == 100 or
                last_closed['cdl_harami'] == 100 or
                last_closed['cdl_3_white_soldiers'] == 100
            )

        elif direction == 'sell':
            is_pattern_valid = (
                last_closed['cdl_engulfing'] == -100 or
                last_closed['cdl_shooting_star'] == 100 or
                last_closed['cdl_evening_star'] == -100 or
                last_closed['cdl_dark_cloud'] == -100 or
                last_closed['cdl_harami'] == -100 or
                last_closed['cdl_3_black_crows'] == -100
            )

        return is_pattern_valid

    def identify_support_resistance_levels(self, swing_points):
        """
        Identifies support and resistance levels from pre-calculated swing points.
        This is moved from the main.py stub for consolidation.
        """
        if swing_points:
            highs = swing_points['highs']['high'].unique()
            lows = swing_points['lows']['low'].unique()
            levels = np.concatenate([highs, lows])
            return np.unique(levels) # Return unique levels
        return np.array([])