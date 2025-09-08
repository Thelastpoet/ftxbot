"""
Pure Price Action Strategy Module
Implements the core trading strategy based on support/resistance and breakouts
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from utils import get_pip_size

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
        self.min_stop_loss_pips = getattr(config, "min_stop_loss_pips", 10)
        self.pending_pullbacks = {}

    def calculate_support_resistance(self, data: pd.DataFrame, symbol: str) -> Tuple[List[float], List[float]]:
        """
        Uses Linear Regression + Standard Deviation for support/resistance when TA-Lib is available.
        Fallback: recent highs/lows if TA-Lib not available.
        """
        if data is None or len(data) < 20:
            return [], []

        close_prices = data['close'].values

        resistance_levels = []
        support_levels = []

        if TALIB_AVAILABLE and talib is not None:
            try:
                lr = talib.LINEARREG(close_prices, timeperiod=20)
                stddev = talib.STDDEV(close_prices, timeperiod=20)
                # Add statistically significant bands
                for multiplier in [1.0, 1.5, 2.0]:
                    upper_band = lr + (stddev * multiplier)
                    lower_band = lr - (stddev * multiplier)

                    if any(data['high'].values[-20:] >= upper_band[-1]):
                        resistance_levels.append(float(upper_band[-1]))
                    if any(data['low'].values[-20:] <= lower_band[-1]):
                        support_levels.append(float(lower_band[-1]))

                # Add recent pivot based on linear regression slope if market flat
                lr_slope = talib.LINEARREG_SLOPE(close_prices, timeperiod=14)
                if not np.isnan(lr_slope[-1]) and abs(lr_slope[-1]) < 0.0001:
                    resistance_levels.append(float(data['high'].tail(20).max()))
                    support_levels.append(float(data['low'].tail(20).min()))
            except Exception as e:
                logger.exception("TA-Lib failure in calculate_support_resistance, falling back to simple S/R: %s", e)
                # fallback to simple S/R
                resistance_levels.append(float(data['high'].tail(20).max()))
                support_levels.append(float(data['low'].tail(20).min()))
        else:
            # Fallback: recent high/low pivots
            resistance_levels.append(float(data['high'].tail(20).max()))
            support_levels.append(float(data['low'].tail(20).min()))

        # dedupe & sort (resistance high->low, support low->high)
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))

        logger.debug(f"{symbol}: Statistical S/R - Resistance: {resistance_levels}, Support: {support_levels}")

        return resistance_levels, support_levels

    def _cluster_levels(self, prices: np.ndarray, symbol: str) -> List[float]:
        """
        Cluster nearby price levels using peak ranking, scaled by symbol pip size
        """
        if len(prices) == 0:
            return []

        symbol_info = self._get_symbol_info(symbol)
        pip_size = get_pip_size(symbol_info)

        proximity = self.proximity_threshold * pip_size  # convert pips to price units

        ranked_prices = []
        for price in prices:
            rank = sum(1 for p in prices if abs(p - price) <= proximity)
            if rank >= self.min_peak_rank:
                ranked_prices.append((price, rank))

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
        Detect breakouts
        """
        symbol_info = self._get_symbol_info(symbol)
        pip_size = get_pip_size(symbol_info)
        
        # Use your existing threshold
        threshold = self.breakout_threshold * pip_size
        
        logger.debug(f"{symbol} - Checking for breakouts. Close: {current_candle['close']:.5f}")
        
        # Check for bullish breakout
        for resistance in resistance_levels:
            if current_candle['close'] > resistance:
                distance_above = current_candle['close'] - resistance
                if distance_above >= threshold:
                    logger.info(f"Bullish breakout detected for {symbol} above {resistance:.5f}")
                    
                    # Mark this as pending pullback opportunity
                    return {
                        'type': 'bullish',
                        'level': resistance,
                        'detection_price': current_candle['close'],
                        'pullback_zone_start': resistance - (2 * pip_size),
                        'pullback_zone_end': resistance + (3 * pip_size),
                        'timestamp': current_candle.name if hasattr(current_candle, 'name') else datetime.now()
                    }
        
        # Check for bearish breakout
        for support in support_levels:
            if current_candle['close'] < support:
                distance_below = support - current_candle['close']
                if distance_below >= threshold:
                    logger.info(f"Bearish breakout detected for {symbol} below {support:.5f}")
                    
                    return {
                        'type': 'bearish',
                        'level': support,
                        'detection_price': current_candle['close'],
                        'pullback_zone_start': support + (2 * pip_size),
                        'pullback_zone_end': support - (3 * pip_size),
                        'timestamp': current_candle.name if hasattr(current_candle, 'name') else datetime.now()
                    }
        
        return None
    
    def check_for_pullback_entry(self,
                             current_candle: pd.Series,
                             pending_breakout: Dict,
                             symbol: str) -> bool:
        """
        Check if price has pulled back to entry zone
        """
        current_price = current_candle['close']
        
        if pending_breakout['type'] == 'bullish':
            # For bullish, we want price to pull back DOWN to the zone
            in_zone = (pending_breakout['pullback_zone_start'] <= current_price <= 
                    pending_breakout['pullback_zone_end'])
            
            if in_zone:
                logger.info(f"{symbol}: Price pulled back to entry zone {current_price:.5f}")
                return True
                
        else:  # bearish
            # For bearish, we want price to pull back UP to the zone
            in_zone = (pending_breakout['pullback_zone_end'] <= current_price <= 
                    pending_breakout['pullback_zone_start'])
            
            if in_zone:
                logger.info(f"{symbol}: Price pulled back to entry zone {current_price:.5f}")
                return True
        
        return False

    def confirm_signal(self, data: pd.DataFrame, breakout: Dict) -> Tuple[bool, List[str]]:
        """
        Confirm breakout signal using TA-Lib patterns when available with fallback.
        Always returns (is_confirmed: bool, patterns_found: List[str]).
        """
        patterns_found: List[str] = []

        if data is None or len(data) < 2:
            return False, patterns_found

        try:
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values

            if TALIB_AVAILABLE and talib is not None:
                # Choose patterns depending on direction
                if breakout['type'] == 'bullish':
                    patterns_to_check = {
                        'ENGULFING': talib.CDLENGULFING, 'HAMMER': talib.CDLHAMMER,
                        'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER, 'PIERCING': talib.CDLPIERCING,
                        'MORNING_STAR': talib.CDLMORNINGSTAR, 'BULLISH_HARAMI': talib.CDLHARAMI,
                        'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS, 'MARUBOZU': talib.CDLMARUBOZU
                    }
                    is_signal = lambda val: val > 0
                else:
                    patterns_to_check = {
                        'ENGULFING': talib.CDLENGULFING, 'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
                        'HANGING_MAN': talib.CDLHANGINGMAN, 'DARK_CLOUD': talib.CDLDARKCLOUDCOVER,
                        'EVENING_STAR': talib.CDLEVENINGSTAR, 'BEARISH_HARAMI': talib.CDLHARAMI,
                        'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS, 'MARUBOZU': talib.CDLMARUBOZU
                    }
                    is_signal = lambda val: val < 0

                for name, func in patterns_to_check.items():
                    try:
                        result = func(open_prices, high_prices, low_prices, close_prices)
                        # check last up-to-3 candles for pattern
                        for i in range(1, min(4, len(result) + 1)):
                            if is_signal(result[-i]):
                                patterns_found.append(name)
                                logger.debug(f"{breakout['type'].capitalize()} pattern found: {name} on candle {-i}")
                                break
                    except Exception as e:
                        logger.debug(f"Pattern {name} computation failed: {e}")
            # Fallback: evaluate last candle(s)
            if not patterns_found:
                current_candle = data.iloc[-1]
                previous_candle = data.iloc[-2] if len(data) >= 2 else None

                if breakout['type'] == 'bullish':
                    is_strong_bullish = (
                        current_candle['close'] > current_candle['open'] and
                        (current_candle['close'] - current_candle['open']) >
                        (current_candle['high'] - current_candle['low']) * 0.6
                    )
                    is_bullish_engulfing = False
                    if previous_candle is not None:
                        is_bullish_engulfing = (
                            current_candle['close'] > current_candle['open'] and
                            previous_candle['close'] < previous_candle['open'] and
                            current_candle['close'] > previous_candle['open'] and
                            current_candle['open'] < previous_candle['close']
                        )
                    if is_strong_bullish:
                        patterns_found.append('STRONG_BULLISH_CANDLE')
                    if is_bullish_engulfing:
                        patterns_found.append('BULLISH_ENGULFING')

                else:  # bearish
                    is_strong_bearish = (
                        current_candle['close'] < current_candle['open'] and
                        (current_candle['open'] - current_candle['close']) >
                        (current_candle['high'] - current_candle['low']) * 0.6
                    )
                    is_bearish_engulfing = False
                    if previous_candle is not None:
                        is_bearish_engulfing = (
                            current_candle['close'] < current_candle['open'] and
                            previous_candle['close'] > previous_candle['open'] and
                            current_candle['close'] < previous_candle['open'] and
                            current_candle['open'] > previous_candle['close']
                        )
                    if is_strong_bearish:
                        patterns_found.append('STRONG_BEARISH_CANDLE')
                    if is_bearish_engulfing:
                        patterns_found.append('BEARISH_ENGULFING')

            is_confirmed = len(patterns_found) > 0
            if is_confirmed:
                logger.info(f"Signal confirmed with patterns: {list(set(patterns_found))}")
            else:
                logger.debug("No confirming candlestick patterns found")

            return is_confirmed, patterns_found

        except Exception as e:
            logger.exception(f"Error in candlestick pattern confirmation: {e}")
            return False, []

    def _basic_pattern_confirmation(self, data: pd.DataFrame, breakout: Dict) -> bool:
        """Basic confirmation when TA-Lib is missing or errors - kept for backwards compatibility"""
        if data is None or len(data) < 2:
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

    def _check_trend_alignment(self, breakout_type: str, trend: str) -> bool:
        """
        ENHANCEMENT: Check if breakout aligns with trend direction
        """
        if trend == 'ranging':
            return True  # Allow trades in ranging market

        return (breakout_type == 'bullish' and trend == 'bullish') or (breakout_type == 'bearish' and trend == 'bearish')

    def _calculate_confidence(self,
                              breakout: Dict,
                              patterns_found: List[str],
                              resistance_levels: List[float],
                              support_levels: List[float],
                              trend_aligned: bool) -> float:
        """
        ENHANCEMENT: Multi-factor confidence calculation
        """
        base_confidence = min(float(breakout.get('strength', 0.0)), 0.4)

        pattern_weight = 0.0
        if patterns_found:
            pattern_weight = min(len(patterns_found) * 0.15, 0.3)
            strong_patterns = ['ENGULFING', 'HAMMER', 'MORNING_STAR', 'EVENING_STAR', 'BULLISH_ENGULFING', 'BEARISH_ENGULFING']
            if any(any(sp in pf for pf in patterns_found) for sp in strong_patterns):
                pattern_weight += 0.1

        total_levels = len(resistance_levels) + len(support_levels)
        structure_weight = min(total_levels * 0.05, 0.2)

        trend_weight = 0.2 if trend_aligned else -0.1

        confidence = base_confidence + pattern_weight + structure_weight + trend_weight
        confidence = max(0.0, min(confidence, 1.0))

        logger.debug(f"Confidence calculation: base={base_confidence:.2f}, pattern={pattern_weight:.2f}, "
                     f"structure={structure_weight:.2f}, trend={trend_weight:.2f}, total={confidence:.2f}")

        return confidence

    def calculate_stop_loss(self, breakout: Dict,
                            support_levels: List[float],
                            resistance_levels: List[float],
                            current_price: float,
                            symbol: str) -> float:
        """
        Calculate stop loss based on support/resistance levels and minimum SL
        """
        symbol_info = self._get_symbol_info(symbol)
        pip_size = get_pip_size(symbol_info)

        buffer_pips = getattr(self.config, 'stop_loss_buffer_pips', 15)
        buffer = buffer_pips * pip_size

        if breakout['type'] == 'bullish':
            # default: slightly below breakout level
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

        # adjust for spread using live broker data (spread in points)
        spread_points = getattr(symbol_info, 'spread', 0)
        spread = spread_points * getattr(symbol_info, 'point', 0)
        if breakout['type'] == 'bullish':
            # push the SL further down to account for spread: conservative
            stop_loss -= spread
        else:
            stop_loss += spread

        logger.info(
            f"Adjusted SL for {symbol}: raw SL={stop_loss:.5f}, "
            f"spread={spread:.5f}, spread_points={spread_points}, "
            f"pip_size={pip_size}"
        )

        return float(stop_loss)

    def calculate_take_profit(self,
                              entry_price: float,
                              stop_loss: float,
                              breakout: Dict,
                              data: pd.DataFrame,
                              symbol: str) -> float:
        """
        Use Time Series Forecast for statistically valid take profit targets with ATR fallback.
        """
        projected_price = None
        try:
            if TALIB_AVAILABLE and talib is not None and data is not None and len(data) >= 14:
                close_prices = data['close'].values
                tsf = talib.TSF(close_prices, timeperiod=14)
                projected_price = float(tsf[-1]) if not np.isnan(tsf[-1]) else None
        except Exception as e:
            logger.debug(f"{symbol}: TSF compute failed: {e}")
            projected_price = None

        sl_distance = abs(entry_price - stop_loss)
        min_tp_distance = sl_distance * self.risk_reward_ratio

        if breakout['type'] == 'bullish':
            min_tp = entry_price + min_tp_distance
            if projected_price is not None and projected_price > min_tp:
                tp = projected_price
                logger.info(f"{symbol}: TP using TSF projection: {tp:.5f}")
            else:
                if TALIB_AVAILABLE and talib is not None and data is not None and len(data) >= 14:
                    try:
                        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)[-1]
                        tp = entry_price + (atr * 3.0)
                    except Exception as e:
                        logger.debug("ATR failed for TP, using min TP: %s", e)
                        tp = min_tp
                else:
                    tp = min_tp

                if projected_price is not None and projected_price < entry_price:
                    logger.warning(
                        f"{symbol}: TSF projects {projected_price:.5f} < entry {entry_price:.5f}. Trade against statistical trend!"
                    )
        else:
            min_tp = entry_price - min_tp_distance
            if projected_price is not None and projected_price < min_tp:
                tp = projected_price
                logger.info(f"{symbol}: TP using TSF projection: {tp:.5f}")
            else:
                if TALIB_AVAILABLE and talib is not None and data is not None and len(data) >= 14:
                    try:
                        atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)[-1]
                        tp = entry_price - (atr * 3.0)
                    except Exception as e:
                        logger.debug("ATR failed for TP, using min TP: %s", e)
                        tp = min_tp
                else:
                    tp = min_tp

                if projected_price is not None and projected_price > entry_price:
                    logger.warning(
                        f"{symbol}: TSF projects {projected_price:.5f} > entry {entry_price:.5f}. Trade against statistical trend!"
                    )

        return float(tp)

    def generate_signal(self, data: pd.DataFrame, symbol: str, trend: str = 'ranging') -> Optional[TradingSignal]:
        """
        Generate signal using pullback strategy
        """
        required_candles = max(self.lookback_period, self.swing_window * 2 + 1)
        if data is None or len(data) < required_candles:
            return None
        
        try:
            # Get data
            completed_data = data.iloc[:-1].tail(self.lookback_period).copy()
            current_candle = data.iloc[-1]
            
            # Calculate S/R levels
            resistance_levels, support_levels = self.calculate_support_resistance(completed_data, symbol)
            
            if not resistance_levels and not support_levels:
                logger.debug(f"No support/resistance levels found for {symbol}")
                return None
            
            # First, check if we have a pending pullback setup
            if symbol in self.pending_pullbacks:
                pending = self.pending_pullbacks[symbol]
                
                # Check if pullback has occurred
                if self.check_for_pullback_entry(current_candle, pending, symbol):
                    
                    # Validate with TSF - it should now SUPPORT our entry
                    close_prices = completed_data['close'].values
                    tsf = talib.TSF(close_prices, timeperiod=14)
                    projected_price = tsf[-1] if not np.isnan(tsf[-1]) else None
                    
                    current_price = float(current_candle['close'])
                    
                    # TSF validation
                    tsf_supports = False
                    if projected_price:
                        if pending['type'] == 'bullish':
                            # For pullback buy, TSF should project higher
                            if projected_price >= current_price:
                                tsf_supports = True
                                logger.info(f"{symbol}: TSF supports pullback entry - projects {projected_price:.5f}")
                        else:
                            # For pullback sell, TSF should project lower
                            if projected_price <= current_price:
                                tsf_supports = True
                                logger.info(f"{symbol}: TSF supports pullback entry - projects {projected_price:.5f}")
                    
                    if not tsf_supports and projected_price:
                        logger.info(f"{symbol}: TSF doesn't support pullback yet - waiting")
                        return None
                    
                    # Confirm with patterns
                    breakout_dict = {'type': pending['type'], 'level': pending['level']}
                    confirmation_result = self.confirm_signal(completed_data, breakout_dict)
                    
                    if isinstance(confirmation_result, tuple):
                        is_confirmed, patterns_found = confirmation_result
                    else:
                        is_confirmed, patterns_found = confirmation_result, []
                    
                    if is_confirmed:
                        logger.info(f"{symbol}: Pullback entry confirmed with patterns: {patterns_found}")
                        
                        # Generate signal at pullback price
                        entry_price = current_price
                        
                        # Calculate stops
                        symbol_info = self._get_symbol_info(symbol)
                        pip_size = get_pip_size(symbol_info)
                        
                        if pending['type'] == 'bullish':
                            stop_loss = pending['level'] - (self.min_stop_loss_pips * pip_size)
                        else:
                            stop_loss = pending['level'] + (self.min_stop_loss_pips * pip_size)
                        
                        # Calculate TP based on risk/reward
                        sl_distance = abs(entry_price - stop_loss)
                        if pending['type'] == 'bullish':
                            take_profit = entry_price + (sl_distance * self.risk_reward_ratio)
                        else:
                            take_profit = entry_price - (sl_distance * self.risk_reward_ratio)
                        
                        # Clear pending setup
                        del self.pending_pullbacks[symbol]
                        
                        # Create signal
                        signal = TradingSignal(
                            type=0 if pending['type'] == 'bullish' else 1,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            stop_loss_pips=sl_distance / pip_size,
                            reason=f"{pending['type']}_pullback",
                            confidence=0.85,  # Pullback entries are high probability
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        logger.info(
                            f"PULLBACK SIGNAL for {symbol}: "
                            f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                            f"Entry={entry_price:.5f}, "
                            f"SL={stop_loss:.5f}, "
                            f"TP={take_profit:.5f}"
                        )
                        
                        return signal
                
                # Check if setup is stale (>20 bars old)
                else:
                    bars_since = len(data) - data.index.get_loc(pending['timestamp'])
                    if bars_since > 20:
                        logger.info(f"{symbol}: Removing stale pullback setup")
                        del self.pending_pullbacks[symbol]
            
            # If no pending pullback, check for new breakout
            else:
                breakout = self.detect_breakout(
                    current_candle,
                    resistance_levels,
                    support_levels,
                    symbol
                )
                
                if breakout:
                    # Store as pending pullback opportunity
                    self.pending_pullbacks[symbol] = breakout
                    logger.info(
                        f"{symbol}: Breakout detected at {breakout['level']:.5f}, "
                        f"waiting for pullback to zone {breakout['pullback_zone_start']:.5f} - "
                        f"{breakout['pullback_zone_end']:.5f}"
                    )
                    
                    # Don't generate signal yet - wait for pullback
                    return None
            
            return None
            
        except Exception as e:
            logger.exception(f"Error generating signal for {symbol}: {e}")
            return None

    def _get_symbol_info(self, symbol: str):
        """
        Get live symbol information from MetaTrader 5 with error handling.
        Returns an object with attributes similar to mt5.symbol_info result (point, spread, digits).
        """
        try:
            # initialize only if not initialized; mt5.initialize may already be called externally
            try:
                initialized = mt5.initialize()
            except Exception:
                initialized = False

            # try to retrieve symbol info
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Failed to get symbol info for {symbol}, using defaults")
                return self._get_default_symbol_info(symbol)
            return info
        except Exception as e:
            logger.exception(f"Error getting symbol info: {e}")
            return self._get_default_symbol_info(symbol)

    def _get_default_symbol_info(self, symbol: str):
        """
        Provide default symbol info as fallback with attributes .point, .spread, .digits
        """
        # reasonable defaults
        if 'JPY' in symbol:
            point = 0.001  # many brokers use 0.001 for 5-digit JPY (adjust as needed)
            digits = 3 if 'JPY' in symbol else 5
        else:
            # default to 5-digit pip (0.00001)
            point = 0.00001
            digits = 5

        class DefaultInfo:
            def __init__(self, point, digits):
                self.point = point
                self.spread = 10  # default spread in points
                self.digits = digits

        return DefaultInfo(point, digits)
