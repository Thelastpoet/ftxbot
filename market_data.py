"""
Market Data - Core version
Fetches candle data from MT5 and returns a DataFrame.
"""

import logging
from typing import Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


class MarketData:
    def __init__(self, mt5_client, config):
        self.mt5_client = mt5_client
        self.config = config

    async def fetch_data(self, symbol: str, timeframe: str, num_candles: int) -> Optional[pd.DataFrame]:
        try:
            rates = self.mt5_client.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None
            return self._process_rates(rates)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _process_rates(self, rates: Any) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        # Normalize columns
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread',
            'real_volume': 'real_volume'
        }, inplace=True)
        df = df[df['volume'] > 0]
        df.sort_index(inplace=True)
        return df
