import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from talib import ATR

class PriceActionAnalyzer:
    """
    Analyzes price data to identify market structure and patterns.
    This class is stateless and has no knowledge of trading.
    """
    def __init__(self, swing_lookback_period=20):
        self.swing_lookback_period = swing_lookback_period

    def get_analysis(self, data: pd.DataFrame):
        """Performs a full analysis on a single dataframe."""
        if data is None or data.empty or len(data) < self.swing_lookback_period:
            return None
        
        swing_points = self._find_swing_points(data)
        structure = self._analyze_market_structure(swing_points)
        atr_series = ATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        return {
            'swing_points': swing_points,
            'structure': structure,
            'atr': atr_series,
            'reversal_candle': self._detect_reversal_candle(data),
            'data': data
        }

    def _find_swing_points(self, data: pd.DataFrame):
        """Finds swing high and swing low points in the data."""
        try:
            high_indices = argrelextrema(data['high'].values, np.greater_equal, order=self.swing_lookback_period)[0]
            low_indices = argrelextrema(data['low'].values, np.less_equal, order=self.swing_lookback_period)[0]
            return {'highs': data.iloc[high_indices], 'lows': data.iloc[low_indices]}
        except:
            return {'highs': pd.DataFrame(), 'lows': pd.DataFrame()}

    def _analyze_market_structure(self, swing_points: dict):
        """Analyzes the sequence of swing points to determine trend."""
        if not swing_points or swing_points['highs'].empty or swing_points['lows'].empty or len(swing_points['highs']) < 2 or len(swing_points['lows']) < 2:
            return {'trend': 'Insufficient Data'}
        
        highs, lows = swing_points['highs']['high'], swing_points['lows']['low']
        
        is_bullish = highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]
        is_bearish = highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]

        if is_bullish: return {'trend': 'bullish'}
        elif is_bearish: return {'trend': 'bearish'}
        else: return {'trend': 'ranging'}

    def _detect_reversal_candle(self, data: pd.DataFrame):
        """
        Detects a Bullish or Bearish Engulfing pattern on the most recent candle.
        This is a flexible entry trigger.
        """
        if len(data) < 2:
            return None
            
        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Bullish Engulfing: Last candle is green, prev is red. Last body engulfs prev body.
        is_bullish_engulfing = (last['close'] > last['open'] and 
                                prev['close'] < prev['open'] and
                                last['close'] > prev['open'] and
                                last['open'] < prev['close'])

        # Bearish Engulfing: Last candle is red, prev is green. Last body engulfs prev body.
        is_bearish_engulfing = (last['close'] < last['open'] and
                                prev['close'] > prev['open'] and
                                last['open'] > prev['close'] and
                                last['close'] < prev['open'])

        if is_bullish_engulfing: return 'bullish_engulfing'
        if is_bearish_engulfing: return 'bearish_engulfing'
        return None