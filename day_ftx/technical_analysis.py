# --- START OF FILE technical_analysis.py (Updated) ---

import traceback
from talib import RSI, ADX, MACD, ATR, MOM, EMA, PLUS_DI, MINUS_DI
import talib 
import pandas as pd
import numpy as np
from numpy import select
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self):
        pass

    def calculate_indicators(self, data, rsi_period=14, adx_period=14, atr_period=14,
                             ichimoku_tenkan_sen=9, ichimoku_kijun_sen=26, ichimoku_senkou_span_b=52,
                             macd_fastperiod=12, macd_slowperiod=26, macd_signal_period=9,
                             chandelier_period=14, chandelier_multiplier=3, momentum_period=10,
                             dmi_period=14, ema_short_period=10, ema_long_period=50):
        """
        Calculate technical indicators with NaN handling
        """
        try:
            if data is None or data.empty:
                logging.error("No data provided for indicator calculation")
                return None
            
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # --- Standard Indicators ---
            df['ema_short'] = EMA(df['close'], timeperiod=ema_short_period)
            df['ema_long'] = EMA(df['close'], timeperiod=ema_long_period)
            df['ema_short_slope'] = df['ema_short'].diff()
            df['ema_long_slope'] = df['ema_long'].diff()
            df['rsi'] = RSI(df['close'], timeperiod=rsi_period)
            df['adx'] = ADX(df['high'], df['low'], df['close'], timeperiod=adx_period)
            df['plus_di'] = PLUS_DI(df['high'], df['low'], df['close'], timeperiod=dmi_period)
            df['minus_di'] = MINUS_DI(df['high'], df['low'], df['close'], timeperiod=dmi_period)        
            df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)     
            df['ichimoku_tenkan_sen'] = (df['high'].rolling(window=ichimoku_tenkan_sen).max() + df['low'].rolling(window=ichimoku_tenkan_sen).min()) / 2
            df['ichimoku_kijun_sen'] = (df['high'].rolling(window=ichimoku_kijun_sen).max() + df['low'].rolling(window=ichimoku_kijun_sen).min()) / 2
            df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(ichimoku_kijun_sen)
            df['ichimoku_senkou_span_b'] = ((df['high'].rolling(window=ichimoku_senkou_span_b).max() + df['low'].rolling(window=ichimoku_senkou_span_b).min()) / 2).shift(ichimoku_kijun_sen)
            df['ichimoku_chikou_span'] = df['close'].shift(-ichimoku_kijun_sen)
            df['macd'], df['macd_signal'], df['macd_hist'] = MACD(df['close'], fastperiod=macd_fastperiod, slowperiod=macd_slowperiod, signalperiod=macd_signal_period)    
            df['chandelier_exit'] = self.calculate_chandelier_exit(df, period=chandelier_period, multiplier=chandelier_multiplier)        
            df['momentum'] = MOM(df['close'], timeperiod=momentum_period)

            # --- START: NEW CANDLESTICK PATTERN INDICATORS ---

            # 1. Confirmation Candles (for breakouts)
            df['cdl_marubozu'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
            df['cdl_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])

            # 2. Indecision/Reversal Candles (to AVOID breakouts)
            df['cdl_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_spinning_top'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
            df['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            
            # --- END: NEW CANDLESTICK PATTERN INDICATORS ---

            # Calculate the maximum required lookback period
            max_period = max(
                ema_long_period,
                adx_period + dmi_period,
                macd_slowperiod + macd_signal_period,
                ichimoku_senkou_span_b + ichimoku_kijun_sen,
                chandelier_period,
                momentum_period
            )

            # Drop rows with NaN values but keep enough data for calculations
            # Candlestick patterns don't require a long lookback, so this is safe
            df = df.iloc[max_period:]
            
            # Verify we still have data after removing NaN values
            if df.empty:
                logging.error("No data remaining after removing NaN values")
                return None
                
            # Final NaN check for critical indicators
            critical_indicators = ['ema_short', 'ema_long', 'rsi', 'adx', 'atr']
            if df[critical_indicators].isna().any().any():
                logging.error("Critical indicators still contain NaN values")
                return None

            return df

        except Exception as e:
            logging.error(f"Error in calculate_indicators: {str(e)}")
            logging.error(traceback.format_exc())
            return None
    
    def calculate_chandelier_exit(self, data, period=22, multiplier=3):
        atr = ATR(data['high'], data['low'], data['close'], timeperiod=period)
        chandelier_long = data['high'].rolling(window=period).max() - (atr * multiplier)
        chandelier_short = data['low'].rolling(window=period).min() + (atr * multiplier)
        
        return pd.Series(np.where(data['close'] > data['close'].shift(1), chandelier_long, chandelier_short),
                         index=data.index, name='chandelier_exit')
                    
    def identify_support_resistance_levels(self, data):  
        if data is None or data.empty:
            logging.info("No data provided for SR level identification")
            return []
            
        pivot_calculator = PivotPointsCalculator(data)
        pivot_data = pivot_calculator.calculate_pivots(data, method='fibonacci')
        
        levels = []
        for column in ['p', 's1', 's2', 's3', 'r1', 'r2', 'r3']:
            if column in pivot_data.columns:
                levels.append(pivot_data[column].iloc[-1])
        
        fib_data = pivot_calculator.calculate_fibonacci_retracement(data)
        for column in ['fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8']:
            if column in fib_data.columns:
                levels.append(fib_data[column].iloc[-1])
        
        return sorted(set(levels))
        
class PivotPointsCalculator:
    def __init__(self, data):
        self.data = data
                
    def calculate_fibonacci_retracement(self, data):
        high = data['high'].max()
        low = data['low'].min()
        diff = high - low
        
        data['fib_0.0'] = high
        data['fib_23.6'] = high - 0.236 * diff
        data['fib_38.2'] = high - 0.382 * diff
        data['fib_50.0'] = high - 0.5 * diff
        data['fib_61.8'] = high - 0.618 * diff
        data['fib_100.0'] = low
        
        self.data = data
        return data

    def calculate_pivots(self, data, anchor='D', method='traditional'):        
        _open = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        anchor = anchor.upper() if anchor and isinstance(anchor, str) and len(anchor) >= 1 else "D"
        method_list = ["traditional", "fibonacci", "woodie", "classic", "demark", "camarilla"]
        method = method if method in method_list else "traditional"
        
        if method == "traditional":
            data["p"] = (high + low + close) / 3
            data["s1"] = (2 * data["p"]) - high
            data["s2"] = data["p"] - (high - low)
            data["s3"] = data["p"] - 2 * (high - low)
            data["r1"] = (2 * data["p"]) - low
            data["r2"] = data["p"] + (high - low)
            data["r3"] = data["p"] + 2 * (high - low)

        elif method == "fibonacci":
            data["p"] = (high + low + close) / 3
            pivot_range = high - low
            data["s1"] = data["p"] - 0.382 * pivot_range
            data["s2"] = data["p"] - 0.618 * pivot_range
            data["s3"] = data["p"] - 1.0 * pivot_range
            data["r1"] = data["p"] + 0.382 * pivot_range
            data["r2"] = data["p"] + 0.618 * pivot_range
            data["r3"] = data["p"] + 1.0 * pivot_range

        elif method == "woodie":
            data["p"] = (high + low + _open * 2) / 4
            pivot_range = high - low
            data["s1"] = data["p"] * 2 - high
            data["s2"] = data["p"] - pivot_range
            data["s3"] = low - 2 * (high - data["p"])
            data["r1"] = data["p"] * 2 - low
            data["r2"] = data["p"] + pivot_range
            data["r3"] = high + 2 * (data["p"] - low)

        elif method == "classic":
            data["p"] = (high + low + close) / 3
            pivot_range = high - low
            data["s1"] = data["p"] * 2 - high
            data["s2"] = data["p"] - pivot_range
            data["s3"] = data["p"] - 2 * pivot_range
            data["s4"] = data["p"] - 3 * pivot_range
            data["r1"] = data["p"] * 2 - low
            data["r2"] = data["p"] + pivot_range
            data["r3"] = data["p"] + 2 * pivot_range
            data["r4"] = data["p"] + 3 * pivot_range

        elif method == "demark":
            conds = [close == _open, close > _open]
            vals = [high + low + close * 2, high * 2 + low + close]
            p = select(conds, vals, default=(high + low * 2 + close))
            data["p"] = p / 4
            data["s1"] = p / 2 - high
            data["r1"] = p / 2 - low

        elif method == "camarilla":
            pivot_range = high - low
            data["p"] = (high + low + close) / 3
            data["s1"] = close - pivot_range * 1.1 / 12
            data["s2"] = close - pivot_range * 1.1 / 6
            data["s3"] = close - pivot_range * 1.1 / 4
            data["s4"] = close - pivot_range * 1.1 / 2
            data["r1"] = close + pivot_range * 1.1 / 12
            data["r2"] = close + pivot_range * 1.1 / 6
            data["r3"] = close + pivot_range * 1.1 / 4
            data["r4"] = close + pivot_range * 1.1 / 2

        else:
            raise ValueError("Invalid method")

        self.data = data
        return data