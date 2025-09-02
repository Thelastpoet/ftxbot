import traceback
from talib import EMA, RSI, ATR, MFI  # keep only what you need
import numpy as np
import logging

class IndicatorCalculator:
    def __init__(self, config=None):
        self.config = config or {}

        # Parameters
        self.rsi_period = self.config.get('rsi', {}).get('period', 14)
        self.ema_fast = self.config.get('ema', {}).get('fast', 20)
        self.ema_slow = self.config.get('ema', {}).get('slow', 50)
        self.atr_period = self.config.get('risk_management', {}).get('atr_period', 14)
        self.proximity_atr_multiplier = self.config.get('price_action_settings', {}).get('proximity_atr_multiplier', 1.0)

    def calculate_indicators(self, df):
        try:
            df['ema_fast'] = EMA(df['close'], timeperiod=self.ema_fast)
            df['ema_slow'] = EMA(df['close'], timeperiod=self.ema_slow)
            df['rsi'] = RSI(df['close'], timeperiod=self.rsi_period)
            df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
            return df
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}\n{traceback.format_exc()}")
            return None
        
    def check_entry_patterns(self, data, direction, pullback_level=None, atr=None):
        if data is None or len(data) < max(self.ema_slow, self.rsi_period) + 2:
            return False

        last_closed = data.iloc[-2]

        # --- Proximity filter ---
        if pullback_level is not None and atr is not None and atr > 0:
            proximity_zone = atr * self.proximity_atr_multiplier
            price_to_check = last_closed['low'] if direction == 'buy' else last_closed['high']
            if abs(price_to_check - pullback_level) > proximity_zone:
                return False

        # --- EMA filter ---
        if direction == 'buy' and not (last_closed['ema_fast'] > last_closed['ema_slow']):
            return False
        if direction == 'sell' and not (last_closed['ema_fast'] < last_closed['ema_slow']):
            return False

        # --- RSI momentum filter ---
        if direction == 'buy' and last_closed['rsi'] <= 50:
            return False
        if direction == 'sell' and last_closed['rsi'] >= 50:
            return False

        return True
    
    def identify_support_resistance_levels(self, swing_points):
        if swing_points:
            highs = swing_points['highs']['high'].unique()
            lows = swing_points['lows']['low'].unique()
            levels = np.concatenate([highs, lows])
            return np.unique(levels)
        return np.array([])

