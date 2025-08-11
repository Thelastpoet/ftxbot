import traceback
import talib
import pandas as pd
import numpy as np
from numpy import select
import logging

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndicatorCalculator:
    def __init__(self, config=None):
        self.params = config or {}

    def calculate_indicators(self, data):
        try:
            if data is None or data.empty:
                logging.error("No data provided for indicator calculation")
                return None
            
            df = data.copy()
            params = self.params
            
            # Standard Indicators
            df['ema_short'] = talib.EMA(df['close'], timeperiod=params['ema_short_period'])
            df['ema_long'] = talib.EMA(df['close'], timeperiod=params['ema_long_period'])
            df['ema_short_slope'] = df['ema_short'].diff()
            df['ema_long_slope'] = df['ema_long'].diff()
            df['rsi'] = talib.RSI(df['close'], timeperiod=params['rsi_period'])
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=params['adx_period'])
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=params['dmi_period'])
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=params['dmi_period'])        
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=params['atr_period'])     
            
            # Ichimoku Cloud
            df['ichimoku_tenkan_sen'] = (df['high'].rolling(window=params['ichimoku_tenkan_sen']).max() + df['low'].rolling(window=params['ichimoku_tenkan_sen']).min()) / 2
            df['ichimoku_kijun_sen'] = (df['high'].rolling(window=params['ichimoku_kijun_sen']).max() + df['low'].rolling(window=params['ichimoku_kijun_sen']).min()) / 2
            df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(params['ichimoku_kijun_sen'])
            df['ichimoku_senkou_span_b'] = ((df['high'].rolling(window=params['ichimoku_senkou_span_b']).max() + df['low'].rolling(window=params['ichimoku_senkou_span_b']).min()) / 2).shift(params['ichimoku_kijun_sen'])
            df['ichimoku_chikou_span'] = df['close'].shift(-params['ichimoku_kijun_sen'])
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=params['macd_fastperiod'], slowperiod=params['macd_slowperiod'], signalperiod=params['macd_signal_period'])    
            
            # Other Indicators
            df['momentum'] = talib.MOM(df['close'], timeperiod=params['momentum_period'])

            bb_period = self.params.get('bollinger_band_period', 20)
            bb_std = self.params.get('bollinger_band_std_dev', 2)
            df['bb_upper'], df['bb_mavg'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std, matype=0)

            kc_period = self.params.get('keltner_channel_period', 20)
            kc_atr_period = self.params.get('keltner_channel_atr_period', 10)
            kc_multiplier = self.params.get('keltner_channel_multiplier', 1.5)
            kc_ema = talib.EMA(df['close'], timeperiod=kc_period)
            kc_atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=kc_atr_period)
            df['kc_upper'] = kc_ema + (kc_atr * kc_multiplier)
            df['kc_lower'] = kc_ema - (kc_atr * kc_multiplier)
            # --- END OF NEW INDICATORS ---
            
            df['volume_avg'] = df['tick_volume'].rolling(window=self.params.get('volume_period', 20)).mean()

            # Candlestick Patterns
            df['cdl_marubozu'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
            df['cdl_engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_spinning_top'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
            df['cdl_shootingstar'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_hangingman'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_eveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_morningdojistar'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_eveningdojistar'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
            df['cdl_piercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_darkcloudcover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3whitesoldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_3blackcrows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
            df['cdl_dragonflydoji'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_gravestone'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
            df['cdl_invertedhammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['cdl_kicking'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
            df['cdl_longleggeddoji'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
            
            df.dropna(inplace=True)

            if df.empty:
                logging.error("No data remaining after removing NaN values")
                return None
                
            return df

        except Exception as e:
            logging.error(f"Error in calculate_indicators: {str(e)}")
            logging.error(traceback.format_exc())
            return None
                        
    def identify_support_resistance_levels(self, data):  
        if data is None or data.empty:
            logging.info("No data provided for SR level identification")
            return []
            
        pivot_calculator = PivotPointsCalculator()
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

        return data