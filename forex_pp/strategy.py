"""
Pure Price Action Strategy Module
Implements the core trading strategy based on support/resistance and breakouts
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
try:
    import talib
except Exception:
    talib = None
from scipy.signal import argrelextrema
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    type: int  # 0 for BUY, 1 for SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_loss_pips: float
    reason: str
    confidence: float
    timestamp: datetime


class PurePriceActionStrategy:
    """Pure price action trading strategy implementation"""

    def __init__(self, config):
        self.config = config
        self.lookback_period = config.lookback_period
        self.swing_window = config.swing_window
        self.breakout_threshold = config.breakout_threshold  # in pips (config expects pips)
        self.min_peak_rank = config.min_peak_rank
        self.proximity_threshold = config.proximity_threshold  # in pips
        self.risk_reward_ratio = config.risk_reward_ratio
        # optional minimum SL in pips to avoid ultra-tight stops in live trading
        self.min_stop_loss_pips = getattr(config, "min_stop_loss_pips", 10)

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find swing highs and lows using argrelextrema
        """
        if len(data) < self.swing_window * 2:
            return np.array([]), np.array([])

        try:
            swing_highs = argrelextrema(data['high'].values, np.greater_equal, order=self.swing_window)[0]
            swing_lows = argrelextrema(data['low'].values, np.less_equal, order=self.swing_window)[0]
            return swing_highs, swing_lows
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return np.array([]), np.array([])

    def calculate_support_resistance(self,
                                     data: pd.DataFrame,
                                     swing_highs: np.ndarray,
                                     swing_lows: np.ndarray,
                                     symbol: str) -> Tuple[List[float], List[float]]:
        """
        Calculate dynamic support and resistance levels from swing points
        """
        resistance_levels: List[float] = []
        support_levels: List[float] = []

        if len(swing_highs) > 0:
            resistance_prices = data.iloc[swing_highs]['high'].values
            resistance_levels = self._cluster_levels(resistance_prices, symbol)

        if len(swing_lows) > 0:
            support_prices = data.iloc[swing_lows]['low'].values
            support_levels = self._cluster_levels(support_prices, symbol)

        return resistance_levels, support_levels

    def _cluster_levels(self, prices: np.ndarray, symbol: str) -> List[float]:
        """
        Cluster nearby price levels using peak ranking, scaled by symbol pip size
        """
        if len(prices) == 0:
            return []

        symbol_info = self._get_symbol_info(symbol)
        point = symbol_info.get('point', 0.00001)
        # define pip size: 1 pip = 10 points for typical 5-digit brokers; for 3-digit JPY pairs pip is 0.01
        pip_size = 0.01 if symbol_info.get('digits', 5) in (2, 3) else point * 10

        proximity = self.proximity_threshold * pip_size  # convert pips to price units

        ranked_prices = []
        for price in prices:
            rank = sum(1 for p in prices if abs(p - price) <= proximity)
            if rank >= self.min_peak_rank:
                ranked_prices.append((price, rank))

        # sort by rank descending
        ranked_prices.sort(key=lambda x: x[1], reverse=True)

        consolidated_levels: List[float] = []
        for price, _rank in ranked_prices:
            is_close = any(abs(price - level) <= proximity for level in consolidated_levels)
            if not is_close:
                consolidated_levels.append(price)

        return sorted(consolidated_levels)

    def detect_breakout(self,
                       current_candle: pd.Series,
                       resistance_levels: List[float],
                       support_levels: List[float],
                       symbol: str) -> Optional[Dict]:
        """
        Detect breakout above resistance or below support
        """
        symbol_info = self._get_symbol_info(symbol)
        point = symbol_info.get('point', 0.00001)
        pip_size = 0.01 if symbol_info.get('digits', 5) in (2, 3) else point * 10
        breakout_threshold_price = self.breakout_threshold * pip_size

        # bullish breakout
        for resistance in resistance_levels:
            if (current_candle['high'] > resistance and
                    current_candle['close'] > resistance + breakout_threshold_price):
                strength = (current_candle['close'] - resistance) / max(resistance, 1e-9)
                return {'type': 'bullish', 'level': resistance, 'strength': float(strength)}

        # bearish breakout
        for support in support_levels:
            if (current_candle['low'] < support and
                    current_candle['close'] < support - breakout_threshold_price):
                strength = (support - current_candle['close']) / max(support, 1e-9)
                return {'type': 'bearish', 'level': support, 'strength': float(strength)}

        return None

    def confirm_signal(self, data: pd.DataFrame, breakout: Dict) -> bool:
        """
        Confirm breakout signal using TA-Lib patterns when available with fallback
        """
        # require enough candles for pattern recognition
        if len(data) < 20:
            return False

        try:
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values

            patterns_found = []

            if talib:
                if breakout['type'] == 'bullish':
                    patterns = {
                        'ENGULFING': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                        'HAMMER': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
                        'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices),
                        'PIERCING': talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices),
                        'MORNING_STAR': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices),
                        'BULLISH_HARAMI': talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices),
                        'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices),
                        'MARUBOZU': talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)
                    }
                    for pattern_name, pattern_result in patterns.items():
                        # check last 3 candles for positive value
                        for i in range(1, 4):
                            if -i >= -len(pattern_result) and pattern_result[-i] > 0:
                                patterns_found.append(pattern_name)
                                logger.debug(f"Bullish pattern found: {pattern_name}")
                                break

                else:  # bearish
                    patterns = {
                        'ENGULFING': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                        'SHOOTING_STAR': talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices),
                        'HANGING_MAN': talib.CDLHANGINGMAN(open_prices, high_prices, low_prices, close_prices),
                        'DARK_CLOUD': talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices),
                        'EVENING_STAR': talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices),
                        'BEARISH_HARAMI': talib.CDLHARAMI(open_prices, high_prices, low_prices, close_prices),
                        'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices),
                        'MARUBOZU': talib.CDLMARUBOZU(open_prices, high_prices, low_prices, close_prices)
                    }
                    for pattern_name, pattern_result in patterns.items():
                        for i in range(1, 4):
                            if -i >= -len(pattern_result) and pattern_result[-i] < 0:
                                patterns_found.append(pattern_name)
                                logger.debug(f"Bearish pattern found: {pattern_name}")
                                break

            # fallback to strong candle if no TA-Lib or no patterns found
            if not patterns_found:
                current_candle = data.iloc[-1]
                if breakout['type'] == 'bullish':
                    is_strong_bullish = (
                        current_candle['close'] > current_candle['open'] and
                        (current_candle['close'] - current_candle['open']) >
                        (current_candle['high'] - current_candle['low']) * 0.6
                    )
                    if is_strong_bullish:
                        patterns_found.append('STRONG_BULLISH_CANDLE')
                else:
                    is_strong_bearish = (
                        current_candle['close'] < current_candle['open'] and
                        (current_candle['open'] - current_candle['close']) >
                        (current_candle['high'] - current_candle['low']) * 0.6
                    )
                    if is_strong_bearish:
                        patterns_found.append('STRONG_BEARISH_CANDLE')

            if patterns_found:
                logger.info(f"Signal confirmed with patterns: {patterns_found}")
                return True
            else:
                logger.debug("No confirming candlestick patterns found")
                return False

        except Exception as e:
            logger.error(f"Error in candlestick pattern confirmation: {e}")
            return self._basic_pattern_confirmation(data, breakout)

    def _basic_pattern_confirmation(self, data: pd.DataFrame, breakout: Dict) -> bool:
        """Basic confirmation when TA-Lib is missing or errors"""
        if len(data) < 2:
            return False

        current_candle = data.iloc[-1]
        previous_candle = data.iloc[-2]

        if breakout['type'] == 'bullish':
            is_bullish_engulfing = (
                current_candle['close'] > current_candle['open'] and
                previous_candle['close'] < previous_candle['open'] and
                current_candle['close'] > previous_candle['open'] and
                current_candle['open'] < previous_candle['close']
            )
            is_strong_bullish = (
                current_candle['close'] > current_candle['open'] and
                (current_candle['close'] - current_candle['open']) >
                (current_candle['high'] - current_candle['low']) * 0.6
            )
            return is_bullish_engulfing or is_strong_bullish

        else:
            is_bearish_engulfing = (
                current_candle['close'] < current_candle['open'] and
                previous_candle['close'] > previous_candle['open'] and
                current_candle['close'] < previous_candle['open'] and
                current_candle['open'] > previous_candle['close']
            )
            is_strong_bearish = (
                current_candle['close'] < current_candle['open'] and
                (current_candle['open'] - current_candle['close']) >
                (current_candle['high'] - current_candle['low']) * 0.6
            )
            return is_bearish_engulfing or is_strong_bearish

    def calculate_stop_loss(self, breakout: Dict,
                            support_levels: List[float],
                            resistance_levels: List[float],
                            current_price: float,
                            symbol: str) -> float:
        """
        Calculate stop loss based on support/resistance levels and minimum SL
        """
        symbol_info = self._get_symbol_info(symbol)
        point = symbol_info.get('point', 0.00001)
        pip_size = 0.01 if symbol_info.get('digits', 5) in (2, 3) else point * 10

        buffer = 5 * pip_size  # 5 pip buffer
        if breakout['type'] == 'bullish':
            stop_loss = breakout['level'] - buffer
            # pick nearest support below current price
            for support in reversed(support_levels):
                if support < current_price - buffer:
                    stop_loss = support - buffer
                    break
        else:
            stop_loss = breakout['level'] + buffer
            for resistance in resistance_levels:
                if resistance > current_price + buffer:
                    stop_loss = resistance + buffer
                    break

        # enforce minimum stop loss distance (in price units)
        min_sl_price_distance = self.min_stop_loss_pips * pip_size
        if abs(current_price - stop_loss) < min_sl_price_distance:
            if breakout['type'] == 'bullish':
                stop_loss = current_price - min_sl_price_distance
            else:
                stop_loss = current_price + min_sl_price_distance

        # adjust for spread using live broker data
        spread_points = symbol_info["spread"]
        spread = spread_points * symbol_info["point"]
        if breakout['type'] == 'bullish':
            stop_loss -= spread
        else:
            stop_loss += spread
            
        logger.info(
            f"Adjusted SL for {symbol}: raw SL={stop_loss:.5f}, "
            f"spread={spread:.5f}, spread_points={spread_points}, "
            f"pip_size={pip_size}"
        )

        return float(stop_loss)

    def calculate_take_profit(self, entry_price: float, stop_loss: float, breakout: Dict) -> float:
        """
        Calculate take profit based on risk-reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio
        if breakout['type'] == 'bullish':
            return float(entry_price + reward)
        else:
            return float(entry_price - reward)

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on price action analysis (live, latest candle)
        """
        if data is None or len(data) < max(self.lookback_period, self.swing_window * 2):
            return None

        try:
            analysis_data = data.tail(self.lookback_period).copy()

            swing_highs, swing_lows = self.find_swing_points(analysis_data)
            if len(swing_highs) == 0 and len(swing_lows) == 0:
                return None

            resistance_levels, support_levels = self.calculate_support_resistance(
                analysis_data, swing_highs, swing_lows, symbol
            )

            if not resistance_levels and not support_levels:
                return None

            current_candle = analysis_data.iloc[-1]
            breakout = self.detect_breakout(current_candle, resistance_levels, support_levels, symbol)
            if not breakout:
                return None

            if not self.confirm_signal(analysis_data, breakout):
                return None

            entry_price = float(current_candle['close'])
            stop_loss = self.calculate_stop_loss(breakout, support_levels, resistance_levels, entry_price, symbol)
            take_profit = self.calculate_take_profit(entry_price, stop_loss, breakout)

            symbol_info = self._get_symbol_info(symbol)
            point = symbol_info.get('point', 0.00001)
            pip_size = 0.01 if symbol_info.get('digits', 5) in (2, 3) else point * 10
            stop_loss_pips = abs(entry_price - stop_loss) / pip_size

            # normalize confidence to 0-1
            confidence = max(0.0, min(float(breakout.get('strength', 0.0)), 1.0))

            signal = TradingSignal(
                type=0 if breakout['type'] == 'bullish' else 1,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_pips=float(stop_loss_pips),
                reason=f"{breakout['type']}_breakout",
                confidence=confidence,
                timestamp=datetime.utcnow()
            )

            logger.info(f"Signal generated for {symbol}: {asdict(signal)}")
            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _get_symbol_info(self, symbol: str) -> Dict:
        """
        Get live symbol information from MetaTrader 5
        """
        if not mt5.initialize():
            logger.error("MT5 initialize() failed")
            raise RuntimeError("MT5 connection failed")

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            raise ValueError(f"Symbol {symbol} not found in MT5")

        return {
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread  # live spread from broker
        }
