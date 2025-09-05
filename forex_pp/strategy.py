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
    TALIB_AVAILABLE = False
from scipy.signal import argrelextrema
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
        pip_size = get_pip_size(symbol_info)

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
                   symbol: str,
                   data: pd.DataFrame = None) -> Optional[Dict]:
        """
        Detect breakout above resistance or below support
        """
        symbol_info = self._get_symbol_info(symbol)
        pip_size = get_pip_size(symbol_info)        
        
        if TALIB_AVAILABLE and data is not None and len(data) >= 14:
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=14)[-1]
            dynamic_threshold = max(self.breakout_threshold * pip_size, atr * 0.3)  # Min 5 pips or 30% of ATR
            logger.debug(f"{symbol} - Using dynamic threshold: {dynamic_threshold/pip_size:.1f} pips (ATR: {atr/pip_size:.1f} pips)")
        else:
            dynamic_threshold = self.breakout_threshold * pip_size
        
        logger.debug(f"{symbol} - Checking for breakouts. Close: {current_candle['close']:.5f}")

        # Check for bullish breakout
        for resistance in resistance_levels:
            # Check if price broke above resistance
            if current_candle['close'] > resistance:
                # Calculate how far above resistance we closed
                distance_above = current_candle['close'] - resistance
                
                logger.debug(f"{symbol} - Close {current_candle['close']:.5f} vs Resistance {resistance:.5f}, "
                            f"distance: {distance_above:.5f}, threshold: {dynamic_threshold:.5f}")

                if distance_above >= dynamic_threshold:
                    strength = distance_above / pip_size / 10  # Normalize strength
                    logger.info(f"Bullish breakout detected for {symbol} above {resistance:.5f}")
                    return {'type': 'bullish', 'level': resistance, 'strength': float(strength)}

        # Check for bearish breakout
        for support in support_levels:
            # Check if price broke below support
            if current_candle['close'] < support:
                # Calculate how far below support we closed
                distance_below = support - current_candle['close']
                
                logger.debug(f"{symbol} - Close {current_candle['close']:.5f} vs Support {support:.5f}, "
                            f"distance: {distance_below:.5f}, threshold: {dynamic_threshold:.5f}")

                if distance_below >= dynamic_threshold:
                    strength = distance_below / pip_size / 10  # Normalize strength
                    logger.info(f"Bearish breakout detected for {symbol} below {support:.5f}")
                    return {'type': 'bearish', 'level': support, 'strength': float(strength)}

        return None

    def confirm_signal(self, data: pd.DataFrame, breakout: Dict) -> bool:
        """
        Confirm breakout signal using TA-Lib patterns when available with fallback.
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
                # Determine which set of patterns to check
                if breakout['type'] == 'bullish':
                    patterns_to_check = {
                        'ENGULFING': talib.CDLENGULFING, 'HAMMER': talib.CDLHAMMER,
                        'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER, 'PIERCING': talib.CDLPIERCING,
                        'MORNING_STAR': talib.CDLMORNINGSTAR, 'BULLISH_HARAMI': talib.CDLHARAMI,
                        'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS, 'MARUBOZU': talib.CDLMARUBOZU
                    }
                    # Bullish patterns return a positive value (e.g., 100)
                    is_signal = lambda val: val > 0
                else:  # bearish
                    patterns_to_check = {
                        'ENGULFING': talib.CDLENGULFING, 'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
                        'HANGING_MAN': talib.CDLHANGINGMAN, 'DARK_CLOUD': talib.CDLDARKCLOUDCOVER,
                        'EVENING_STAR': talib.CDLEVENINGSTAR, 'BEARISH_HARAMI': talib.CDLHARAMI,
                        'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS, 'MARUBOZU': talib.CDLMARUBOZU
                    }
                    # Bearish patterns return a negative value (e.g., -100)
                    is_signal = lambda val: val < 0

                # Calculate and check patterns
                for name, func in patterns_to_check.items():
                    result = func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Check the last 3 candles for a signal using negative indexing
                    # Ensure we don't go out of bounds if there are fewer than 3 results
                    for i in range(1, min(4, len(result) + 1)):
                        if is_signal(result[-i]):
                            patterns_found.append(name)
                            logger.debug(f"{breakout['type'].capitalize()} pattern found: {name} on candle {-i}")
                            # Break the inner loop once a pattern is found for this type
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
                logger.info(f"Signal confirmed with patterns: {list(set(patterns_found))}")
                return True, patterns_found
            else:
                logger.debug("No confirming candlestick patterns found")
                return False

        except Exception as e:
            logger.error(f"Error in candlestick pattern confirmation: {e}")
            # Fallback to basic confirmation on any error
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
        
    def _check_trend_alignment(self, breakout_type: str, trend: str) -> bool:
        """
        ENHANCEMENT: Check if breakout aligns with trend direction
        """
        if trend == 'ranging':
            return True  # Allow trades in ranging market
        
        if breakout_type == 'bullish' and trend == 'bullish':
            return True
        if breakout_type == 'bearish' and trend == 'bearish':
            return True
            
        return False
    
    def _calculate_confidence(self, 
                                     breakout: Dict, 
                                     patterns_found: List[str],
                                     resistance_levels: List[float],
                                     support_levels: List[float],
                                     trend_aligned: bool) -> float:
        """
        ENHANCEMENT: Multi-factor confidence calculation
        """
        # Base confidence from breakout strength (capped to prevent overweight)
        base_confidence = min(float(breakout.get('strength', 0.0)), 0.4)
        
        # Pattern strength component
        pattern_weight = 0.0
        if patterns_found:
            # Multiple patterns = stronger signal
            pattern_weight = min(len(patterns_found) * 0.15, 0.3)
            
            # Bonus for strong patterns
            strong_patterns = ['ENGULFING', 'HAMMER', 'MORNING_STAR', 'EVENING_STAR']
            if any(pattern in strong_patterns for pattern in patterns_found):
                pattern_weight += 0.1
        
        # Structure quality component (number of S/R levels indicates good structure)
        total_levels = len(resistance_levels) + len(support_levels)
        structure_weight = min(total_levels * 0.05, 0.2)
        
        # Trend alignment bonus
        trend_weight = 0.2 if trend_aligned else -0.1  # Penalty for counter-trend
        
        # Combined confidence
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
        spread_points = symbol_info.spread
        spread = spread_points * symbol_info.point
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

    def calculate_take_profit(self,
                      entry_price: float,
                      breakout: Dict,
                      support_levels: List[float],
                      resistance_levels: List[float],
                      data: pd.DataFrame,
                      symbol: str) -> float:
        """
        Calculate take profit with enhanced methods for better risk-reward
        """
        symbol_info = self._get_symbol_info(symbol)
        pip_size = get_pip_size(symbol_info)
        
        # Get the stop loss that was calculated earlier to ensure minimum RR
        # We'll need to pass this or calculate the distance
        
        # Detect current trading session for session-specific adjustments
        current_hour = datetime.now(timezone.utc).hour
        
        if 21 <= current_hour or current_hour <= 6:  # Asian session
            buffer_pips = getattr(self.config, 'stop_loss_buffer_pips', 15) * 0.7
        else:  # London/NY sessions - more volatility
            buffer_pips = getattr(self.config, 'stop_loss_buffer_pips', 15)
        
        buffer = buffer_pips * pip_size
        
        # Now use the session-adjusted buffer]
    
        session_multipliers = {
            'asian': 3.0,    # 21:00-06:00 UTC
            'london': 4.5,   # 07:00-15:00 UTC  
            'newyork': 4.0,  # 12:00-20:00 UTC
            'overlap': 5.0   # London/NY overlap
        }
        
        # Determine session
        if 21 <= current_hour or current_hour <= 6:
            session = 'asian'
        elif 7 <= current_hour <= 11:
            session = 'london'
        elif 12 <= current_hour <= 15:
            session = 'overlap'  # Best volatility
        else:
            session = 'newyork'
        
        # Method 1: Enhanced Range Projection (Primary)
        if support_levels and resistance_levels:
            range_high = max(resistance_levels)
            range_low = min(support_levels)
            range_height = range_high - range_low
            
            min_range_pips = getattr(self.config, "min_range_pips", 10)  # Increased from 5
            if range_height >= min_range_pips * pip_size:
                # Use range projection with session adjustment
                projection_multiplier = 1.5 if session == 'asian' else 2.0
                
                if breakout['type'] == 'bullish':
                    tp = entry_price + (range_height * projection_multiplier)
                else:
                    tp = entry_price - (range_height * projection_multiplier)
                    
                logger.info(
                    f"{symbol}: TP set using enhanced range projection. "
                    f"Range={range_height/pip_size:.1f} pips, "
                    f"Multiplier={projection_multiplier}, TP={tp:.5f}"
                )
                
                # Ensure minimum risk-reward ratio
                min_tp = self._ensure_minimum_rr(entry_price, tp, breakout, pip_size)
                return float(max(tp, min_tp) if breakout['type'] == 'bullish' else min(tp, min_tp))
        
        # Method 2: Enhanced ATR with Proper Multipliers
        atr_period = getattr(self.config, "atr_period", 14)
        
        if TALIB_AVAILABLE and len(data) >= atr_period:
            # Calculate ATR
            atr = talib.ATR(data['high'].values,
                            data['low'].values, 
                            data['close'].values,
                            timeperiod=atr_period)[-1]
            
            # Detect volatility regime
            atr_ma = talib.SMA(talib.ATR(data['high'].values,
                                        data['low'].values,
                                        data['close'].values,
                                        timeperiod=atr_period), 
                            timeperiod=50)[-1] if len(data) >= 50 else atr
            
            volatility_ratio = atr / atr_ma if atr_ma > 0 else 1.0
            
            # Dynamic multiplier based on volatility and session
            base_multiplier = session_multipliers[session]
            
            # Adjust for volatility regime
            if volatility_ratio > 1.3:  # High volatility
                volatility_adj = 0.8  # Reduce multiplier slightly
            elif volatility_ratio < 0.7:  # Low volatility
                volatility_adj = 1.2  # Increase multiplier
            else:
                volatility_adj = 1.0
            
            # Final ATR multiplier (ensure it's never less than risk_reward_ratio)
            atr_multiplier = max(
                base_multiplier * volatility_adj,
                self.risk_reward_ratio * 1.5  # At least 1.5x the configured RR
            )
            
            tp_distance = atr * atr_multiplier
            
            if breakout['type'] == 'bullish':
                tp = entry_price + tp_distance
            else:
                tp = entry_price - tp_distance
                
            logger.info(
                f"{symbol}: TP using enhanced ATR. "
                f"ATR={atr/pip_size:.1f} pips, Session={session}, "
                f"Volatility ratio={volatility_ratio:.2f}, "
                f"Final multiplier={atr_multiplier:.1f}, TP={tp:.5f}"
            )
            
            return float(tp)
        
        # Method 3: Fallback - Use configured risk_reward_ratio
        # This ensures we ALWAYS have at least the minimum RR
        logger.warning(f"{symbol}: Using fallback RR-based TP calculation")
        
        # We need to know the stop loss distance to calculate proper TP
        # Estimate based on minimum stop loss
        estimated_sl_distance = self.min_stop_loss_pips * pip_size
        tp_distance = estimated_sl_distance * self.risk_reward_ratio
        
        if breakout['type'] == 'bullish':
            tp = entry_price + tp_distance
        else:
            tp = entry_price - tp_distance
        
        return float(tp)

    def _ensure_minimum_rr(self, entry_price: float, calculated_tp: float, 
                        breakout: Dict, pip_size: float) -> float:
        """
        Helper method to ensure minimum risk-reward ratio
        """
        # Calculate the stop loss distance (approximate)
        sl_buffer_pips = getattr(self.config, 'stop_loss_buffer_pips', 15)
        min_sl_distance = max(self.min_stop_loss_pips, sl_buffer_pips) * pip_size
        
        # Ensure TP gives us at least the configured risk_reward_ratio
        min_tp_distance = min_sl_distance * self.risk_reward_ratio
        
        if breakout['type'] == 'bullish':
            min_tp = entry_price + min_tp_distance
            return min_tp
        else:
            min_tp = entry_price - min_tp_distance
            return min_tp

    def generate_signal(self, data: pd.DataFrame, symbol: str, trend: str = 'ranging') -> Optional[TradingSignal]:
        """
        Generate trading signal based on price action analysis
        """
        if data is None or len(data) < max(self.lookback_period, self.swing_window * 2 + 1):
            logger.debug(f"Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
            return None

        try:            
            # Get trading data
            completed_data = data.iloc[:-1].tail(self.lookback_period).copy()
            current_candle = data.iloc[-1]
            
            # Find swing points in the analysis window
            swing_highs, swing_lows = self.find_swing_points(completed_data)
            
            logger.debug(f"{symbol}: Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
            
            if len(swing_highs) == 0 and len(swing_lows) == 0:
                logger.debug(f"No swing points found for {symbol}")
                return None
    
            # Calculate support and resistance levels
            resistance_levels, support_levels = self.calculate_support_resistance(
                completed_data, swing_highs, swing_lows, symbol
            )

            logger.debug(f"{symbol}: {len(resistance_levels)} resistance levels, {len(support_levels)} support levels")
            
            if not resistance_levels and not support_levels:
                logger.debug(f"No support/resistance levels found for {symbol}")
                return None          
           
            # Log resistance and support levels for debugging
            if resistance_levels:
                logger.debug(f"{symbol} - Resistance levels: {[f'{r:.5f}' for r in resistance_levels[:3]]}")
            if support_levels:
                logger.debug(f"{symbol} - Support levels: {[f'{s:.5f}' for s in support_levels[:3]]}")
            
            # Detect breakout on the last completed candle
            breakout = self.detect_breakout(
                current_candle, 
                resistance_levels, 
                support_levels, 
                symbol,
                data=completed_data
            )
            
            if not breakout:
                logger.debug(f"No breakout detected for {symbol}")
                return None
            
            logger.info(f"Breakout detected for {symbol}: {breakout['type']} at level {breakout['level']:.5f}")
            
            trend_aligned = self._check_trend_alignment(breakout['type'], trend)
            if not trend_aligned:
                logger.info(f"Breakout {breakout['type']} against {trend} trend for {symbol} - SKIPPING")
                return None

            # Confirm signal using completed candles only 
            confirmation_result = self.confirm_signal(completed_data, breakout)
            if isinstance(confirmation_result, tuple):
                is_confirmed, patterns_found = confirmation_result
            else:
                is_confirmed, patterns_found = confirmation_result, []
                
            if not is_confirmed:
                logger.debug(f"Breakout not confirmed for {symbol}")
                return None

            # Confirm signal using completed candles only
            if not self.confirm_signal(completed_data, breakout):
                logger.debug(f"Breakout not confirmed for {symbol}")
                return None
            
            # Get symbol info for pip calculations
            symbol_info = self._get_symbol_info(symbol)
            pip_size = get_pip_size(symbol_info)         
            
            reference_price = float(current_candle['close'])
            
            # Calculate stop loss and take profit
            stop_loss = self.calculate_stop_loss(
                breakout, 
                support_levels, 
                resistance_levels, 
                reference_price, 
                symbol
            )
            
            take_profit = self.calculate_take_profit(reference_price, breakout, support_levels, resistance_levels, completed_data, symbol)

            # Calculate stop loss in pips for position sizing
            stop_loss_pips = abs(reference_price - stop_loss) / pip_size

            # Normalize confidence to 0-1
            confidence = self._calculate_confidence(
                breakout, 
                patterns_found, 
                resistance_levels, 
                support_levels, 
                trend_aligned
            )

            # Create the trading signal
            signal = TradingSignal(
                type=0 if breakout['type'] == 'bullish' else 1,
                entry_price=reference_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_pips=float(stop_loss_pips),
                reason=f"{breakout['type']}_breakout",
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

            logger.info(
                f"SIGNAL GENERATED for {symbol}: "
                f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                f"Entry={reference_price:.5f}, "
                f"SL={stop_loss:.5f} ({stop_loss_pips:.1f} pips), "
                f"TP={take_profit:.5f}, "
                f"Confidence={confidence:.2f}"
            )
            
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def _get_symbol_info(self, symbol: str) -> Dict:
        """
        Get live symbol information from MetaTrader 5 with error handling
        """
        try:
            if not mt5.initialize():
                # Try to initialize if not already done
                if not mt5.initialize():
                    logger.error("MT5 initialize() failed")
                    # Return a default structure to prevent crashes
                    return self._get_default_symbol_info(symbol)
            
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return self._get_default_symbol_info(symbol)
            
            return info
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return self._get_default_symbol_info(symbol)

    def _get_default_symbol_info(self, symbol: str) -> Dict:
        """
        Provide default symbol info as fallback
        """
        # Basic defaults for common pairs
        if 'JPY' in symbol:
            point = 0.001
        else:
            point = 0.00001
        
        class DefaultInfo:
            def __init__(self):
                self.point = point
                self.spread = 10
                self.digits = 3 if 'JPY' in symbol else 5
                
        return DefaultInfo()
