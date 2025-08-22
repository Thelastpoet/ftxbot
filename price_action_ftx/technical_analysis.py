import traceback
from talib import (
    CDLENGULFING, CDLDOJI, CDLSHOOTINGSTAR, CDLHAMMER, 
    CDLMORNINGSTAR, CDLPIERCING, CDLHARAMI, CDLHARAMICROSS,
    CDL3WHITESOLDIERS, CDLMORNINGDOJISTAR,
    CDLEVENINGSTAR, CDLDARKCLOUDCOVER, CDL3BLACKCROWS, CDLEVENINGDOJISTAR,
    MFI
)
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self, config=None):
        self.config = config or {}
        
        # Load MFI parameters
        mfi_params = self.config.get('mfi', {})
        self.mfi_period = mfi_params.get('period', 14)
        
        # Load Price Action parameters
        pa_settings = self.config.get('price_action_settings', {})
        self.proximity_atr_multiplier = pa_settings.get('proximity_atr_multiplier', 1.5) # Default to 1.5 if not found

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
        if data_with_indicators is None or len(data_with_indicators) < 2:
            return False

        last_closed = data_with_indicators.iloc[-2]
        
        # Proximity Filter: Check if the candle is within a multiplier of the ATR from the pullback level
        if pullback_level is not None and atr is not None and atr > 0:
            proximity_zone = atr * self.proximity_atr_multiplier
            
            price_to_check = last_closed['low'] if direction == 'buy' else last_closed['high']
            distance = abs(price_to_check - pullback_level)
            
            if distance > proximity_zone:
                return False
        
        # Candlestick Pattern Logic
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
        if swing_points:
            highs = swing_points['highs']['high'].unique()
            lows = swing_points['lows']['low'].unique()
            levels = np.concatenate([highs, lows])
            return np.unique(levels)
        return np.array([])