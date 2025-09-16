"""
Market Data Module
Handles fetching and processing of market data from MT5
"""

import pandas as pd
import numpy as np
import logging
from talib import ATR, LINEARREG_ANGLE, LINEARREG_SLOPE, STDDEV
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketData:
    """Manages market data fetching and processing"""
    
    def __init__(self, mt5_client, config):
        self.mt5_client = mt5_client
        self.config = config
        self.data_cache = {}
        self.cache_expiry = 60  # seconds
        
    async def fetch_data(self, symbol: str, timeframe: str, num_candles: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, H1, etc.)
            num_candles: Number of candles to fetch
            
        Returns:
            DataFrame with OHLC data or None
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.data_cache:
                cached_data, timestamp = self.data_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_expiry:
                    logger.debug(f"Using cached data for {symbol} {timeframe}")
                    return cached_data
            
            # Fetch fresh data from MT5
            rates = self.mt5_client.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data available for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = self._process_rates(rates)
            
            # Cache the data
            self.data_cache[cache_key] = (df, datetime.now())
            
            # Clean old cache entries
            max_cache_entries = len(self.config.symbols) * 3  # Allow 3 timeframes per symbol
            if len(self.data_cache) > max_cache_entries:
                oldest_key = min(self.data_cache.keys(), 
                                key=lambda k: self.data_cache[k][1])
                del self.data_cache[oldest_key]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _process_rates(self, rates: Any) -> pd.DataFrame:
        """
        Process raw rates into pandas DataFrame
        
        Args:
            rates: Raw rates from MT5
            
        Returns:
            Processed DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set time as index
        df.set_index('time', inplace=True)
        
        # Rename columns for consistency
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        # Filter out periods with zero volume (market closed)
        df = df[df['volume'] > 0]
        
        # Ensure data is sorted by time
        df.sort_index(inplace=True)
        
        return df
    
    async def fetch_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of DataFrames keyed by timeframe
        """
        symbol_config = next((s for s in self.config.symbols if s['name'] == symbol), None)
        if not symbol_config:
            return {}
        
        data = {}
        for timeframe in symbol_config['timeframes']:
            df = await self.fetch_data(symbol, timeframe, self.config.max_period * 2)
            if df is not None:
                data[timeframe] = df
        
        return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        """
        return ATR(data['high'], data['low'], data['close'], timeperiod=period)
    
    def identify_trend(self, data: pd.DataFrame, period: int = 20, symbol: str = None) -> str:
        """
        Adaptive trend detection using linear regression + ATR scaling.
        Uses self.calculate_atr() for consistency.
        """
        if len(data) < period:
            return 'ranging'

        try:
            # Linear regression angle (degrees) and slope
            lr_angle = LINEARREG_ANGLE(data['close'].values, timeperiod=period)
            lr_slope = LINEARREG_SLOPE(data['close'].values, timeperiod=period)

            angle = lr_angle[-1] if not np.isnan(lr_angle[-1]) else 0.0
            slope = lr_slope[-1] if not np.isnan(lr_slope[-1]) else 0.0

            # ATR (via helper)
            atr_series = self.calculate_atr(data, period)
            atr = atr_series.iloc[-1] if len(atr_series) > 0 else 0.0

            # --- Adaptive thresholds ---
            base_angle = 3.0     # more sensitive than 5.0
            slope_threshold = max(0.00002, 0.15 * atr / period)  

            # --- Classification ---
            if angle > base_angle and slope > slope_threshold:
                return 'bullish'
            elif angle < -base_angle and abs(slope) > slope_threshold:
                return 'bearish'
            else:
                return 'ranging'

        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            return 'ranging'

    
    def calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate price volatility
        
        Args:
            data: DataFrame with OHLC data
            period: Period for volatility calculation
            
        Returns:
            Volatility value
        """
        if len(data) < period:
            return 0.0
        
        # Calculate standard deviation of returns
        volatility = STDDEV(data['close'].pct_change().dropna(), timeperiod=period).iloc[-1]
        
        return volatility
    
    def get_recent_high_low(self, data: pd.DataFrame, period: int = 20) -> Tuple[float, float]:
        """
        Get recent high and low prices
        
        Args:
            data: DataFrame with OHLC data
            period: Period to look back
            
        Returns:
            Tuple of (recent_high, recent_low)
        """
        if len(data) < period:
            return None, None
        
        recent_data = data.tail(period)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        
        return recent_high, recent_low
    
    def clear_cache(self):
        """Clear data cache"""
        self.data_cache = {}
        logger.info("Market data cache cleared")