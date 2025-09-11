"""
Pure Price Action Strategy Module - LIVE FOREX TRADING
Real-time breakout detection with smart confirmation
"""
import math
import MetaTrader5 as mt5
from scipy.signal import argrelextrema
import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    talib = None
    TALIB_AVAILABLE = False

import logging
from typing import List, Optional, Tuple, NamedTuple
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

class BreakoutInfo(NamedTuple):
    """Breakout detection result"""
    type: str  # 'bullish' or 'bearish'
    level: float
    entry_price: float
    distance: float
    distance_pips: float
    strength_score: float  # 0-1 normalized

class PurePriceActionStrategy:
    """Pure price action trading strategy - LIVE FOREX implementation"""

    def __init__(self, config):
        self.config = config
        self.lookback_period = config.lookback_period
        self.swing_window = config.swing_window
        self.breakout_threshold_pips = config.breakout_threshold  
        self.risk_reward_ratio = config.risk_reward_ratio
        self.min_stop_loss_pips = getattr(config, "min_stop_loss_pips", 15)
        self.stop_loss_buffer_pips = getattr(config, "stop_loss_buffer_pips", 10)
        
        # Trading thresholds
        self.max_spread_atr_ratio = 0.3
        self.max_spread_pips = 3
        self.max_signal_age_seconds = 30
        self.min_candle_time_remaining = 5  # Don't enter in last 5 seconds
        self.max_extension_atr = 1.5
        self.min_extension_atr = 0.3
        self.min_body_ratio = 0.3
        self.min_confidence = 0.6
        self.proximity_threshold = getattr(config, "proximity_threshold", 20)  # in pips
        self.min_peak_rank = getattr(config, "min_peak_rank", 2)  # min confirmations

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

    def _detect_breakout(self, 
                        tick: any,
                        resistance_levels: List[float],
                        support_levels: List[float],
                        atr: Optional[float],
                        pip_size: float) -> Optional[BreakoutInfo]:
        """
        Detect and score breakout - single place for breakout logic
        """
        # Dynamic threshold based on ATR or fixed
        if atr:
            threshold = max(self.breakout_threshold_pips * pip_size, 
                          atr * self.min_extension_atr)
            max_extension = atr * self.max_extension_atr
        else:
            threshold = self.breakout_threshold_pips * pip_size
            max_extension = 20 * pip_size
            logger.warning("Running without ATR - using fixed thresholds")
        
        # Check bullish breakout (use ASK for buys)
        for resistance in resistance_levels:
            if tick.ask > resistance + threshold:
                distance = tick.ask - resistance
                
                if distance > max_extension:
                    continue  # Too extended
                
                # Score breakout strength (0-1)
                if atr:
                    strength = min(distance / atr, 1.0)
                else:
                    strength = min(distance / (10 * pip_size), 1.0)
                
                return BreakoutInfo(
                    type='bullish',
                    level=resistance,
                    entry_price=tick.ask,
                    distance=distance,
                    distance_pips=distance / pip_size,
                    strength_score=strength
                )
        
        # Check bearish breakout (use BID for sells)
        for support in support_levels:
            if tick.bid < support - threshold:
                distance = support - tick.bid
                
                if distance > max_extension:
                    continue
                
                if atr:
                    strength = min(distance / atr, 1.0)
                else:
                    strength = min(distance / (10 * pip_size), 1.0)
                
                return BreakoutInfo(
                    type='bearish',
                    level=support,
                    entry_price=tick.bid,
                    distance=distance,
                    distance_pips=distance / pip_size,
                    strength_score=strength
                )
        
        return None

    def _calculate_confidence(self,
                        breakout: BreakoutInfo,
                        candle_body: float,
                        candle_range: float,
                        is_bullish_candle: bool,
                        atr: Optional[float],
                        spread: float,  # Changed from spread_pips to actual spread value
                        pip_size: float,  # Pass pip_size directly
                        trend: str) -> Tuple[float, bool]:
        """
        Calculate confidence and check if trade should be taken
        Returns: (confidence_score, should_trade)
        """
        confidence = 0.3  # Base
        
        # 1. Breakout strength (already normalized 0-1)
        confidence += breakout.strength_score * 0.2
        
        # 2. Candle momentum
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        if atr and atr > 0:
            momentum_score = min(candle_body / atr, 1.0)
        else:
            momentum_score = min(body_ratio * 2, 1.0)
        
        confidence += momentum_score * 0.3
        
        # 3. Direction alignment (HARD CHECK)
        direction_match = (
            (breakout.type == 'bullish' and is_bullish_candle) or
            (breakout.type == 'bearish' and not is_bullish_candle)
        )
        if not direction_match:
            logger.debug(f"Direction mismatch: {breakout.type} breakout but {'bullish' if is_bullish_candle else 'bearish'} candle")
            return confidence, False  # HARD STOP
        
        confidence += 0.1
        
        # 4. Trend alignment (HARD CHECK for strong trends)
        if trend in ['bullish', 'bearish']:  # Strong trend
            trend_aligned = (
                (breakout.type == 'bullish' and trend == 'bullish') or
                (breakout.type == 'bearish' and trend == 'bearish')
            )
            if not trend_aligned:
                logger.info(f"Against strong {trend} trend - skipping {breakout.type} breakout")
                return confidence, False  # HARD STOP against strong trend
        
        # Bonus for aligned trades
        if trend != 'ranging':
            if (breakout.type == 'bullish' and trend == 'bullish') or \
            (breakout.type == 'bearish' and trend == 'bearish'):
                confidence += 0.2
        
        # 5. Spread penalty
        if atr:
            spread_impact = spread / atr  # spread is already in price units
            if spread_impact > 0.2:
                confidence -= 0.1
        else:
            spread_pips = spread / pip_size
            if spread_pips > 2:
                confidence -= 0.1
        
        final_confidence = max(0.1, min(confidence, 1.0))
        should_trade = final_confidence >= self.min_confidence
        
        return final_confidence, should_trade

    def generate_signal(self, data: pd.DataFrame, symbol: str, trend: str = 'ranging') -> Optional[TradingSignal]:
        """
        Generate trading signal using existing MarketData infrastructure
        """
        if data is None or len(data) < max(self.lookback_period, 20):
            return None

        try:
            # Get tick ONCE
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick for {symbol}")
                return None
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None
            
            pip_size = get_pip_size(symbol_info)
            spread_pips = (tick.ask - tick.bid) / pip_size
            
            # Get forming candle
            forming_candle = data.iloc[-1]
            completed_data = data.iloc[:-1].tail(self.lookback_period)
            
            swing_highs, swing_lows = self.find_swing_points(completed_data)
            
            if len(swing_highs) == 0 and len(swing_lows) == 0:
                logger.debug(f"No swing points found for {symbol}")
                return None
            
            # Apply symbol-specific overrides from config
            min_stop_loss_pips = self.min_stop_loss_pips
            risk_reward_ratio = self.risk_reward_ratio
            for sym_config in getattr(self.config, "symbols", []):
                if sym_config.get('name') == symbol:
                    min_stop_loss_pips = sym_config.get("min_stop_loss_pips", min_stop_loss_pips)
                    risk_reward_ratio = sym_config.get("risk_reward_ratio", risk_reward_ratio)
                    break           
            
            # Calculate ATR from completed data (using existing MarketData method if available)
            atr = None
            if TALIB_AVAILABLE and len(completed_data) >= 14:
                atr = talib.ATR(
                    completed_data['high'].values,
                    completed_data['low'].values,
                    completed_data['close'].values,
                    timeperiod=14
                )[-1]
                logger.debug(f"{symbol}: ATR={atr/pip_size:.1f} pips")
            else:
                logger.warning(f"{symbol}: Running without ATR - TA-Lib not available or insufficient data")
            
            # QUICK FILTERS
            # 1. Spread filter
            max_spread = (atr * self.max_spread_atr_ratio) if atr else (self.max_spread_pips * pip_size)
            if tick.ask - tick.bid > max_spread:
                logger.debug(f"{symbol}: Spread too high ({spread_pips:.1f} pips)")
                return None
            
            # 2. Candle age filter
            candle_age = (datetime.now(timezone.utc) - 
                         pd.to_datetime(forming_candle.name).replace(tzinfo=timezone.utc))
            candle_age_seconds = candle_age.total_seconds()
            
            if candle_age_seconds > self.max_signal_age_seconds:
                logger.debug(f"{symbol}: Signal too old ({candle_age_seconds:.0f}s)")
                return None
            
            # 3. Don't enter near candle close (prevent fakeouts)
            # Assuming M15 candles (900 seconds), M5 (300 seconds), M1 (60 seconds)
            timeframe_seconds = {
                'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600
            }
            # Get timeframe from config (assuming it's stored)
            current_timeframe = 'M15'  # Default, should come from config
            for sym_config in self.config.symbols:
                if sym_config['name'] == symbol:
                    current_timeframe = sym_config['timeframes'][0]  # Primary timeframe
                    break
            
            max_candle_seconds = timeframe_seconds.get(current_timeframe, 900)
            time_remaining = max_candle_seconds - (candle_age_seconds % max_candle_seconds)
            
            if time_remaining < self.min_candle_time_remaining:
                logger.debug(f"{symbol}: Too close to candle close ({time_remaining:.0f}s remaining)")
                return None
            
            # 4. Momentum filter
            candle_body = abs(forming_candle['close'] - forming_candle['open'])
            candle_range = forming_candle['high'] - forming_candle['low']
            body_ratio = candle_body / candle_range if candle_range > 0 else 0
            
            min_body_ratio = self.min_body_ratio
            if atr:
                atr_pips = atr / pip_size
                if atr_pips < 10:  # Low volatility
                    min_body_ratio *= 0.8
            
            if body_ratio < min_body_ratio:
                logger.debug(f"{symbol}: Insufficient momentum (body_ratio={body_ratio:.2f})")
                return None
            
            # Calculate S/R levels (could use MarketData.calculate_support_resistance)
            resistance_levels, support_levels = self.calculate_support_resistance(completed_data, swing_highs, swing_lows, symbol)
            
            if not resistance_levels and not support_levels:
                logger.debug(f"{symbol}: No S/R levels found")
                return None
            
            # Detect breakout (passing tick once)
            breakout = self._detect_breakout(tick, resistance_levels, support_levels, atr, pip_size)
            
            if not breakout:
                return None
            
            logger.info(
                f"{symbol}: Breakout detected - {breakout.type} @ {breakout.entry_price:.5f}, "
                f"distance={breakout.distance_pips:.1f}p, strength={breakout.strength_score:.2f}"
            )
            
            # -------------------------
            # Structure distance check
            # -------------------------
            # Ensure there's space to run toward TP before taking the trade.
            # Build list copies so we can examine the 'next' structure beyond the broken level.
            # We already computed resistance_levels, support_levels above in generate_signal.

            # Normalize lists (already present but safe)
            res_levels = sorted(resistance_levels)
            sup_levels = sorted(support_levels)

            # Find nearest structure in the direction of the breakout beyond the entry
            next_structure = None
            if breakout.type == 'bullish':
                # Next resistance above the entry
                candidates = [r for r in res_levels if r > breakout.entry_price]
                if candidates:
                    next_structure = min(candidates)
            else:
                # Next support below the entry
                candidates = [s for s in sup_levels if s < breakout.entry_price]
                if candidates:
                    next_structure = max(candidates)

            # If there's a nearby structure, require a minimum buffer between entry and that structure
            base_min_sl = min_stop_loss_pips * pip_size
            atr_min_sl = (atr * 0.8) if atr else 0.0
            min_sl_distance = max(base_min_sl, atr_min_sl)

            if next_structure is not None:
                distance_to_next = abs(next_structure - breakout.entry_price)
                # require at least 'room_req' pips (scaled by ATR if available)
                room_req_pips = getattr(self.config, "min_room_after_breakout_pips", None)
                if room_req_pips is None:
                    # default: require at least 1.5 * min_sl (so TP has space) or 0.8 * ATR (if ATR exists)
                    if atr:
                        room_req = max(1.5 * min_sl_distance, 0.8 * atr)
                    else:
                        room_req = 1.5 * min_sl_distance
                else:
                    room_req = room_req_pips * pip_size

                if distance_to_next < room_req:
                    logger.info(
                        f"{symbol}: Not enough room after breakout (distance_to_next={distance_to_next/pip_size:.1f}p < required={room_req/pip_size:.1f}p). Skipping."
                    )
                    return None
            
            # Calculate confidence with HARD CHECKS
            is_bullish_candle = forming_candle['close'] > forming_candle['open']
            spread = tick.ask - tick.bid
            confidence, should_trade = self._calculate_confidence(
                breakout, candle_body, candle_range, is_bullish_candle,
                atr, spread, pip_size, trend
            )
            
            if not should_trade:
                logger.info(f"{symbol}: Trade filtered out (confidence={confidence:.2f})")
                return None
            
            # Calculate SL/TP
            # Min SL: at least min_stop_loss_pips, but scale with ATR if available
            base_min_sl = min_stop_loss_pips * pip_size
            atr_min_sl = (atr * 0.8) if atr else 0.0
            min_sl_distance = max(base_min_sl, atr_min_sl)

            # Spread buffer: ensure SL is not inside spread
            spread = tick.ask - tick.bid
            spread_buffer = spread + pip_size
            configured_buffer = self.stop_loss_buffer_pips * pip_size
            safety_buffer = max(spread_buffer, configured_buffer)

            epsilon = pip_size / 10.0  # tiny safety margin

            if breakout.type == 'bullish':
                preferred_sl = breakout.level - safety_buffer
                fallback_sl = breakout.entry_price - min_sl_distance
                stop_loss = min(preferred_sl, fallback_sl)
                if stop_loss >= breakout.entry_price - epsilon:
                    stop_loss = breakout.entry_price - min_sl_distance - epsilon
            else:
                preferred_sl = breakout.level + safety_buffer
                fallback_sl = breakout.entry_price + min_sl_distance
                stop_loss = max(preferred_sl, fallback_sl)
                if stop_loss <= breakout.entry_price + epsilon:
                    stop_loss = breakout.entry_price + min_sl_distance + epsilon

            sl_distance = abs(breakout.entry_price - stop_loss)

            # Safety checks
            #  - Ensure sl_distance > 0
            if sl_distance <= 0:
                logger.error(f"{symbol}: Computed sl_distance <= 0 (entry={breakout.entry_price}, sl={stop_loss}) -> rejecting")
                return None

            #  - Reject if SL unrealistically close (< k * ATR)
            min_sl_atr_mult = getattr(self.config, "min_sl_atr_mult", 0.5)
            if atr and sl_distance < min_sl_atr_mult * atr:
                logger.info(f"{symbol}: SL {sl_distance/pip_size:.1f}p < {min_sl_atr_mult:.2f} ATR -> rejecting trade")
                return None

            # Log adjustments if preferred != final
            if (breakout.type == 'bullish' and not math.isclose(preferred_sl, stop_loss, rel_tol=1e-9)) or \
            (breakout.type == 'bearish' and not math.isclose(preferred_sl, stop_loss, rel_tol=1e-9)):
                logger.debug(f"{symbol}: SL adjusted from preferred {preferred_sl:.8f} to {stop_loss:.8f} (entry={breakout.entry_price:.8f})")
            
            # TP based on R:R
            tp_distance = sl_distance * risk_reward_ratio
            
            if breakout.type == 'bullish':
                take_profit = breakout.entry_price + tp_distance
            else:
                take_profit = breakout.entry_price - tp_distance
                
            # Round SL/TP to symbol precision
            point = getattr(symbol_info, 'point', None)
            digits = getattr(symbol_info, 'digits', None)

            if point:
                # round to nearest tick
                stop_loss = round(round(stop_loss / point) * point, int(digits) if digits is not None else 8)
                take_profit = round(round(take_profit / point) * point, int(digits) if digits is not None else 8)
            elif digits is not None:
                stop_loss = round(stop_loss, int(digits))
                take_profit = round(take_profit, int(digits))
            
            # Create signal
            signal = TradingSignal(
                type=0 if breakout.type == 'bullish' else 1,
                entry_price=breakout.entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_pips=sl_distance / pip_size,
                reason=f"live_{breakout.type}_breakout",
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(
                f"*** SIGNAL: {symbol} {'BUY' if signal.type == 0 else 'SELL'} @ {breakout.entry_price:.5f}\n"
                f"  SL: {stop_loss:.5f} ({sl_distance/pip_size:.1f}p)\n"
                f"  TP: {take_profit:.5f} ({tp_distance/pip_size:.1f}p)\n"
                f"  R:R: {self.risk_reward_ratio:.1f}\n"
                f"  Confidence: {confidence:.2f}\n"
                f"  Time remaining in candle: {time_remaining:.0f}s"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: np.ndarray, swing_lows: np.ndarray, symbol: str) -> Tuple[List[float], List[float]]:
        """
        Calculate support and resistance levels
        Could delegate to MarketData class if preferred
        """
        if data is None or len(data) < 20:
            return [], []

        resistance_levels = []
        support_levels = []
        
        if len(swing_highs) > 0:
            resistance_prices = data.iloc[swing_highs]['high'].values
            resistance_levels = self._cluster_levels(resistance_prices, symbol)

        if len(swing_lows) > 0:
            support_prices = data.iloc[swing_lows]['low'].values
            support_levels = self._cluster_levels(support_prices, symbol)

        if TALIB_AVAILABLE and talib is not None:
            try:
                close_prices = data['close'].values
                lr = talib.LINEARREG(close_prices, timeperiod=20)
                stddev = talib.STDDEV(close_prices, timeperiod=20)
                
                lr_val = lr[-1]
                std_val = stddev[-1]
                
                for multiplier in [1.5, 2.0]:
                    upper = lr[-1] + (stddev[-1] * multiplier)
                    lower = lr[-1] - (stddev[-1] * multiplier)
                    
                    if any(data['high'].values[-20:] >= upper):
                        resistance_levels.append(float(upper))
                    if any(data['low'].values[-20:] <= lower):
                        support_levels.append(float(lower))
            except Exception as e:
                logger.error(f"TA-Lib error: {e}")
        
        # Always add recent pivots
        resistance_levels.append(float(data['high'].tail(20).max()))
        support_levels.append(float(data['low'].tail(20).min()))

        # Clean and limit
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:3]
        support_levels = sorted(list(set(support_levels)))[:3]

        return resistance_levels, support_levels
    
    def _cluster_levels(self, prices: np.ndarray, symbol: str) -> List[float]:
        """
        Cluster nearby price levels using peak ranking, scaled by symbol pip size
        """
        if len(prices) == 0:
            return []

        symbol_info = mt5.symbol_info(symbol)
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