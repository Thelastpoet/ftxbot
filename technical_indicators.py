from talib import ATR, EMA, RSI, ADX, PLUS_DI, MINUS_DI
import pandas as pd
import numpy as np
import logging
import traceback

class IndicatorCalculator:
    def __init__(self):
        # Define default periods
        self.default_periods = {
            'atr': 14,
            'adx': 14,
            'rsi': 14,
            'ema_fast': 8,
            'ema_slow': 21,
            'ema_trend': 50,
            'bb_period': 20,
            'bb_std': 2
        }
        
    def calculate_volatility_factor(self, data, atr_period=14, ma_period=30):
        """Calculate volatility factor using standard deviations and normalized ATR"""
        try:
            # ATR from talib
            atr = ATR(data['high'], data['low'], data['close'], timeperiod=atr_period)
            if atr is None:
                return 1.0
            
            atr_percentage = (atr / data['close']) * 100                
            atrp_ma = atr_percentage.rolling(window=ma_period).mean()
            atrp_std = atr_percentage.rolling(window=ma_period).std()
            if pd.isna(atrp_std.iloc[-1]) or atrp_std.iloc[-1] < 0.0001:
                return 1.0
            current_zscore = (atr_percentage.iloc[-1] - atrp_ma.iloc[-1]) / atrp_std.iloc[-1]
            
            volatility_factor = 1 + np.tanh(current_zscore / 2) * 0.3            
            return round(volatility_factor, 2)
            
        except Exception as e:
            logging.error(f"Error calculating volatility factor: {e}")
            return 1.0
    
    def calculate_indicators(self, data, periods=None):
        """Calculate all technical indicators using TALib"""
        try:
            periods = periods or self.default_periods
            df = data.copy()
            
            # Calculate EMAs
            df['ema_fast'] = EMA(df['close'], timeperiod=periods['ema_fast']) 
            df['ema_slow'] = EMA(df['close'], timeperiod=periods['ema_slow'])   
            
            # Momentum Indicators
            df['rsi'] = RSI(df['close'], timeperiod=periods['rsi'])    
            
            # Trend Indicators
            df['adx'] = ADX(df['high'], df['low'], df['close'], timeperiod=periods['adx'])
            df['plus_di'] = PLUS_DI(df['high'], df['low'], df['close'], timeperiod=periods['adx'])
            df['minus_di'] = MINUS_DI(df['high'], df['low'], df['close'], timeperiod=periods['adx'])      
                        
            # Volatility Indicators
            df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=periods['atr'])
            df['atr_dynamic'] = df['atr'].rolling(window=5).mean()
            
            # Calculate volatility factor
            df['volatility_factor'] = self.calculate_volatility_factor(df)   
                                    
            return df
            
        except Exception as e:
            logging.error(f"Error in calculate_indicators: {str(e)}")
            logging.error(traceback.format_exc())
            return None