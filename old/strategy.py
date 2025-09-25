"""
Pure Price Action Strategy Module - LIVE FOREX TRADING
Real-time breakout detection with smart confirmation
"""
import time
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
from typing import List, Optional, Tuple, NamedTuple, TYPE_CHECKING, Dict
from dataclasses import dataclass
from datetime import datetime, timezone

from utils import get_pip_size

# Only import for type checking to avoid runtime dependency cycles
if TYPE_CHECKING:  # pragma: no cover
    from market_session import SessionPhase

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
    breakout_level: float = None
    phase: Optional[str] = None
    session: Optional[str] = None
    phase_weight: float = 0.0
    ttl_minutes: Optional[int] = None
    features: Optional[Dict[str, float]] = None  # model-ready features for calibrator

class BreakoutInfo(NamedTuple):
    """Breakout detection result"""
    type: str  # 'bullish' or 'bearish'
    level: float
    entry_price: float
    distance: float
    distance_pips: float
    strength_score: float  # 0-1 normalized

class PriceActionStrategy:
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
        self.max_spread_atr_ratio = getattr(config, "max_spread_atr_ratio", 0.35)
        self.max_spread_pips = getattr(config, "max_spread_pips", 4)
        self.max_signal_age_seconds = getattr(config, "max_signal_age_seconds", None)
        self.min_candle_time_remaining = getattr(config, "min_candle_time_remaining", 5)  # seconds
        self.max_extension_atr = 1.5
        self.min_extension_atr = 0.3
        self.min_body_ratio = getattr(config, "min_body_ratio", 0.25)
        self.min_confidence = getattr(config, "min_confidence", 0.5)
        self.proximity_threshold = getattr(config, "proximity_threshold", 20)  # in pips
        self.min_peak_rank = getattr(config, "min_peak_rank", 2)  # min confirmations
        self.volume_zscore_threshold = getattr(config, "volume_zscore_threshold", 1.0)

        # M1 confirmation settings
        self.m1_confirmation_enabled = getattr(config, "m1_confirmation_enabled", False)
        self.m1_confirmation_candles = getattr(config, "m1_confirmation_candles", 1)  # number of closed M1 candles beyond level
        self.m1_confirmation_buffer_pips = getattr(config, "m1_confirmation_buffer_pips", 0.5)  # small buffer beyond level

        # Backtest mode: disables real-time age/close filters
        self.backtest_mode = getattr(config, "backtest_mode", False)

        # Optional: require last closed candle to confirm beyond the level
        self.require_close_breakout = getattr(config, "require_close_breakout", False)
        self.close_breakout_buffer_pips = getattr(config, "close_breakout_buffer_pips", 0.2)

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
                        resistance_levels: List[Tuple[float, int]],
                        support_levels: List[Tuple[float, int]],
                        atr: Optional[float],
                        pip_size: float) -> Optional[BreakoutInfo]:
        """
        Detect and score breakout 
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
        for level, rank in resistance_levels:
            if tick.ask > level + threshold:
                distance = tick.ask - level
                
                if distance > max_extension:
                    continue  # Too extended
                
                # Score breakout strength (0-1)
                if atr:
                    strength = min(distance / (1.5 * atr), 1.0)
                else:
                    strength = min(distance / (10 * pip_size), 1.0)
                
                return BreakoutInfo(
                    type='bullish',
                    level=level,
                    entry_price=tick.ask,
                    distance=distance,
                    distance_pips=distance / pip_size,
                    strength_score=strength
                )
        
        # Check bearish breakout (use BID for sells)
        for level, rank in support_levels:
            if tick.bid < level - threshold:
                distance = level - tick.bid
                
                if distance > max_extension:
                    continue
                
                if atr:
                    strength = min(distance / (1.5 * atr), 1.0)
                else:
                    strength = min(distance / (10 * pip_size), 1.0)
                
                return BreakoutInfo(
                    type='bearish',
                    level=level,
                    entry_price=tick.bid,
                    distance=distance,
                    distance_pips=distance / pip_size,
                    strength_score=strength
                )
        
        return None

    def _confirm_breakout_m1(self, symbol: str, breakout: BreakoutInfo, pip_size: float, tick: any) -> Tuple[bool, str]:
        """
        Non-blocking M1 confirmation:
        - Uses last N closed M1 candles; no waiting loops to avoid blocking the bot.
        - Requires those closes beyond level (+/- buffer) and current tick beyond level.

        Returns: (is_confirmed, reason)
        """
        try:
            try:
                mt5.symbol_select(symbol, True)
            except Exception:
                pass

            need_closed = max(1, int(self.m1_confirmation_candles))
            buffer = self.m1_confirmation_buffer_pips * pip_size

            now_utc = datetime.now(timezone.utc)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, need_closed + 5)
            if rates is None or len(rates) == 0:
                return False, "m1_no_data"

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

            closed_df = df[df['time'] + pd.Timedelta(minutes=1) <= now_utc]
            if len(closed_df) < need_closed:
                return False, "m1_insufficient_closed"

            recent_closed = closed_df.tail(need_closed)
            if breakout.type == 'bullish':
                cond_closed = (recent_closed['close'] > (breakout.level + buffer)).all()
                cond_tick = tick.ask > (breakout.level + buffer)
                if cond_closed and cond_tick:
                    return True, "m1_confirmed"
                return False, "m1_not_confirmed_bullish"
            else:
                cond_closed = (recent_closed['close'] < (breakout.level - buffer)).all()
                cond_tick = tick.bid < (breakout.level - buffer)
                if cond_closed and cond_tick:
                    return True, "m1_confirmed"
                return False, "m1_not_confirmed_bearish"
        except Exception as e:
            logger.error(f"Error during M1 confirmation for {symbol}: {e}", exc_info=True)
            return False, "m1_error"

    def _calculate_confidence(self,
                        breakout: BreakoutInfo,
                        candle_body: float,
                        candle_range: float,
                        is_bullish_candle: bool,
                        atr: Optional[float],
                        spread: float,
                        pip_size: float,
                        trend: str,
                        level_rank: int) -> Tuple[float, bool]:
        """
        Calculate confidence and check if trade should be taken
        Returns: (confidence_score, should_trade)
        """
        confidence = 0.3  # Base
        
        # 1. Breakout strength
        confidence += breakout.strength_score * 0.2
        
        # 2. Candle momentum
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        if atr and atr > 0:
            momentum_score = min(candle_body / (1.5 * atr), 1.0)
        else:
            momentum_score = min(body_ratio * 2, 1.0)
        
        confidence += momentum_score * 0.3
        
        # 3. Direction alignment (SOFT)
        direction_match = (
            (breakout.type == 'bullish' and is_bullish_candle) or
            (breakout.type == 'bearish' and not is_bullish_candle)
        )
        if direction_match:
            confidence += 0.12
        else:
            # Soft penalty instead of hard stop; allow strong breakouts to pass
            confidence -= 0.15
            logger.debug(
                f"Direction mismatch: {breakout.type} breakout but {'bullish' if is_bullish_candle else 'bearish'} candle"
            )
        
        # 4. Trend alignment (SOFT)
        if trend in ['bullish', 'bearish']:
            trend_aligned = (
                (breakout.type == 'bullish' and trend == 'bullish') or
                (breakout.type == 'bearish' and trend == 'bearish')
            )
            if trend_aligned:
                confidence += 0.2
            else:
                confidence -= 0.2
                logger.debug(f"Against {trend} trend: applying penalty, not rejecting outright")
        
        # 5. Level Strength Penalty
        # Penalize breakouts of historically strong levels.
        # A rank of 2 is the minimum, so we start penalizing above that.
        if level_rank > 2:
            # For now, a simple linear penalty. This can be made more complex or configurable.
            # Example: rank 3 -> -0.1, rank 4 -> -0.15, etc.
            penalty = (level_rank - 2) * 0.05 
            confidence -= penalty
            logger.debug(f"Applied level strength penalty of {penalty:.2f} for rank {level_rank}")

        # 6. Spread penalty (was 5)
        if atr and atr > 0:
            spread_impact = spread / atr  # price units
            if spread_impact > 0.35:
                confidence -= 0.15
            elif spread_impact > 0.2:
                confidence -= 0.05
        else:
            spread_pips = spread / pip_size if pip_size > 0 else 0
            if spread_pips > 3:
                confidence -= 0.1
        
        final_confidence = max(0.1, min(confidence, 1.0))
        should_trade = final_confidence >= self.min_confidence
        
        return final_confidence, should_trade

    def _calculate_stop_loss(self, breakout: BreakoutInfo, atr: Optional[float], pip_size: float, tick: any, symbol: str) -> Optional[float]:
        """
        Calculates the stop loss for a given breakout signal, ensuring it's logical and safe.
        The stop loss is determined by the most conservative (widest) position based on:
        1. The broken S/R level (structural_sl).
        2. A distance based on volatility (ATR).

        It now returns None if the calculated "natural" SL is smaller than the configured
        minimum, effectively filtering out low-quality signals.
        """
        # Get symbol-specific min stop loss from config, which now acts as a quality filter
        min_stop_loss_pips = self.min_stop_loss_pips
        for sym_config in getattr(self.config, "symbols", []):
            if sym_config.get('name') == symbol:
                min_stop_loss_pips = sym_config.get("min_stop_loss_pips", min_stop_loss_pips)
                break

        # 1. Calculate the 'natural' volatility distance without the config fallback
        natural_volatility_dist = (atr * 0.8) if atr and atr > 0 else 0.0

        # 2. Calculate a safety buffer for placing SL behind structure
        spread = tick.ask - tick.bid
        spread_buffer = spread + pip_size
        configured_buffer = self.stop_loss_buffer_pips * pip_size
        safety_buffer = max(spread_buffer, configured_buffer)

        # 3. Calculate the two potential 'natural' SL levels
        entry_price = breakout.entry_price
        broken_level = breakout.level

        if breakout.type == 'bullish':
            structural_sl = broken_level - safety_buffer
            volatility_sl = entry_price - natural_volatility_dist
            # The natural SL is the more conservative of the two
            natural_sl = min(structural_sl, volatility_sl)
        else:  # Bearish
            structural_sl = broken_level + safety_buffer
            volatility_sl = entry_price + natural_volatility_dist
            # The natural SL is the more conservative of the two
            natural_sl = max(structural_sl, volatility_sl)

        # 4. THE NEW FILTER: Check if the natural SL is too tight
        natural_sl_pips = abs(entry_price - natural_sl) / pip_size if pip_size > 0 else 0
        
        if natural_sl_pips < min_stop_loss_pips:
            logger.debug(
                f"{symbol}: Trade rejected. Natural SL ({natural_sl_pips:.1f} pips) is below "
                f"the quality filter minimum ({min_stop_loss_pips} pips)."
            )
            return None  # Reject the trade by returning None

        # 5. Final safety check to prevent SL from being on the wrong side of entry
        epsilon = pip_size / 10.0
        if breakout.type == 'bullish' and natural_sl >= entry_price - epsilon:
            logger.warning(f"{symbol}: Calculated SL ({natural_sl:.5f}) was on wrong side of entry ({entry_price:.5f}). Rejecting.")
            return None
        elif breakout.type == 'bearish' and natural_sl <= entry_price + epsilon:
            logger.warning(f"{symbol}: Calculated SL ({natural_sl:.5f}) was on wrong side of entry ({entry_price:.5f}). Rejecting.")
            return None

        return natural_sl

    def _calculate_take_profit(self, entry_price: float, stop_loss: float, risk_reward_ratio: float, breakout_type: str) -> float:
        """
        Calculates the take profit level based on the stop loss distance and risk-reward ratio.
        """
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * risk_reward_ratio
        
        if breakout_type == 'bullish':
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def generate_signal(self, data: pd.DataFrame, symbol: str, trend: str = 'ranging', phase: Optional["SessionPhase"] = None) -> Optional[TradingSignal]:
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

              # Apply symbol-specific overrides from config for R:R ratio
              risk_reward_ratio = self.risk_reward_ratio
              for sym_config in getattr(self.config, "symbols", []):
                  if sym_config.get('name') == symbol:
                      risk_reward_ratio = sym_config.get("risk_reward_ratio", risk_reward_ratio)
                      break

              # Calculate ATR from completed data
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

              # Prepare time_remaining for optional logging
              time_remaining = None

              # 2. Time-based filters (only avoid the last few seconds of the candle)
              if not self.backtest_mode:
                  # Don't enter near candle close
                  timeframe_seconds = {'M1': 60, 'M5': 300, 'M15': 900, 'M30': 1800, 'H1': 3600}
                  current_timeframe = 'M15'
                  for sym_config in self.config.symbols:
                      if sym_config['name'] == symbol:
                          current_timeframe = sym_config['timeframes'][0]
                          break
                  max_candle_seconds = timeframe_seconds.get(current_timeframe, 900)
                  # Compute age based on index time if available
                  candle_age = (datetime.now(timezone.utc) -
                                pd.to_datetime(forming_candle.name).replace(tzinfo=timezone.utc))
                  candle_age_seconds = candle_age.total_seconds()
                  time_remaining = max_candle_seconds - (candle_age_seconds % max_candle_seconds)
                  if time_remaining < self.min_candle_time_remaining:
                      logger.debug(f"{symbol}: Too close to candle close ({time_remaining:.0f}s remaining)")
                      return None

              # 4. Momentum filter (either body is decent OR range expanded vs ATR)
              candle_body = abs(forming_candle['close'] - forming_candle['open'])
              candle_range = max(forming_candle['high'] - forming_candle['low'], 1e-12)
              body_ratio = candle_body / candle_range

              min_body_ratio = self.min_body_ratio
              if atr and (atr / pip_size) < 10:
                  min_body_ratio *= 0.8

              range_ok = (atr is not None and atr > 0 and (candle_range / atr) >= 0.8)
              if body_ratio < min_body_ratio and not range_ok:
                  logger.debug(f"{symbol}: Insufficient momentum (body_ratio={body_ratio:.2f}, range/ATR={(candle_range/atr) if atr else 0:.2f})")
                  return None

              # Calculate S/R levels
              resistance_levels_with_rank, support_levels_with_rank = self.calculate_support_resistance(completed_data, swing_highs, swing_lows, symbol)

              if not resistance_levels_with_rank and not support_levels_with_rank:
                  logger.debug(f"{symbol}: No S/R levels found")
                  return None

              # Detect breakout
              breakout = self._detect_breakout(tick, resistance_levels_with_rank, support_levels_with_rank, atr, pip_size)

              if not breakout:
                  return None

              # Find the rank of the broken level
              level_rank = 1 # Default rank
              if breakout.type == 'bullish':
                  for level, rank in resistance_levels_with_rank:
                      if level == breakout.level:
                          level_rank = rank
                          break
              else: # bearish
                  for level, rank in support_levels_with_rank:
                      if level == breakout.level:
                          level_rank = rank
                          break

              logger.debug(
                  f"{symbol}: Breakout detected - {breakout.type} @ {breakout.entry_price:.5f}, "
                  f"distance={breakout.distance_pips:.1f}p, strength={breakout.strength_score:.2f}, "
                  f"level_rank={level_rank}"
              )

              # M1 confirmation gate (prevents false signals on forming M15 candle)
              if self.m1_confirmation_enabled:
                  confirmed, reason = self._confirm_breakout_m1(symbol, breakout, pip_size, tick)
                  if not confirmed:
                      logger.info(f"{symbol}: Breakout not confirmed on M1 ({reason}), skipping.")
                      return None
                  else:
                      # Use last closed M1 candle for momentum/direction in confidence calc
                      try:
                          rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 3)
                          if rates_m1 is not None and len(rates_m1) >= 2:
                              m1_df = pd.DataFrame(rates_m1)
                              m1_df.sort_values(by="time", inplace=True)
                              last_closed = m1_df.iloc[-2]
                              m1_body = abs(last_closed['close'] - last_closed['open'])
                              m1_range = max(last_closed['high'] - last_closed['low'], 1e-12)
                              m1_bullish = last_closed['close'] > last_closed['open']
                              # Override local variables used later in confidence calculation
                              candle_body = float(m1_body)
                              candle_range = float(m1_range)
                              is_bullish_candle = bool(m1_bullish)
                      except Exception as e:
                          logger.warning(f"{symbol}: Failed to derive M1 momentum candle: {e}")

              # Structure distance check
              res_levels = [level for level, rank in resistance_levels_with_rank]
              sup_levels = [level for level, rank in support_levels_with_rank]
              next_structure = None
              if breakout.type == 'bullish':
                  candidates = [r for r in res_levels if r > breakout.entry_price]
                  if candidates: next_structure = min(candidates)
              else:
                  candidates = [s for s in sup_levels if s < breakout.entry_price]
                  if candidates: next_structure = max(candidates)

              if next_structure is not None:
                  min_sl_pips = self.min_stop_loss_pips
                  min_sl_distance = max(min_sl_pips * pip_size, (atr * 0.8) if atr else 0.0)
                  distance_to_next = abs(next_structure - breakout.entry_price)

                  room_req_pips = getattr(self.config, "min_room_after_breakout_pips", None)
                  if room_req_pips:
                      room_req = room_req_pips * pip_size
                  else:
                      room_req = max(1.0 * min_sl_distance, 0.5 * atr if atr else 0.0)

                  if distance_to_next < room_req:
                      logger.debug(
                          f"{symbol}: Limited room after breakout ({distance_to_next/pip_size:.1f}p < {room_req/pip_size:.1f}p). Skipping.")
                      return None

              # Calculate confidence
              is_bullish_candle = forming_candle['close'] > forming_candle['open']
              spread = tick.ask - tick.bid
              # Derive model features prior to scoring for downstream calibrator usage
              # 1) strength
              strength = breakout.strength_score
              # 2) momentum (same formulation as confidence block)
              if atr and atr > 0:
                  momentum = min(candle_body / (1.5 * atr), 1.0)
              else:
                  body_ratio_local = candle_body / candle_range if candle_range > 0 else 0.0
                  momentum = min(body_ratio_local * 2.0, 1.0)
              # 3) direction match
              dir_match = 1.0 if ((breakout.type == 'bullish' and is_bullish_candle) or (breakout.type == 'bearish' and not is_bullish_candle)) else 0.0
              # 4) trend match (only if trending)
              trend_match = 1.0 if (trend in ['bullish', 'bearish'] and ((breakout.type == 'bullish' and trend == 'bullish') or (breakout.type == 'bearish' and
  trend == 'bearish'))) else 0.0
              # 5) spread impact
              if atr and atr > 0:
                  spread_impact = min(max(spread / atr, 0.0), 1.0)
              else:
                  spread_pips_local = spread / pip_size if pip_size > 0 else 0.0
                  # rough normalization to 0..1 using a 10-pip cap
                  spread_impact = min(max(spread_pips_local / 10.0, 0.0), 1.0)

              confidence, should_trade = self._calculate_confidence(
                  breakout, candle_body, candle_range, is_bullish_candle,
                  atr, spread, pip_size, trend, level_rank
              )

              # Session phase adjustment (additive weight; minimal, not restrictive)
              if phase is not None:
                  try:
                      phase_weight = getattr(phase, 'weight', 0.0) or 0.0
                      confidence = max(0.1, min(1.0, confidence + float(phase_weight)))
                      if not should_trade and confidence >= self.min_confidence:
                          should_trade = True
                  except Exception:
                      pass

              if not should_trade:
                  logger.debug(f"{symbol}: Trade filtered out (confidence={confidence:.2f})")
                  return None

              # Calculate SL/TP
              stop_loss = self._calculate_stop_loss(breakout, atr, pip_size, tick, symbol)
              if stop_loss is None:
                  return None # Signal rejected by the SL quality filter

              take_profit = self._calculate_take_profit(breakout.entry_price, stop_loss, risk_reward_ratio, breakout.type)
              sl_distance = abs(breakout.entry_price - stop_loss)

              # Safety checks for SL distance
              if sl_distance <= 0:
                  logger.error(f"{symbol}: Computed sl_distance <= 0 (entry={breakout.entry_price}, sl={stop_loss}) -> rejecting")
                  return None

              min_sl_atr_mult = getattr(self.config, "min_sl_atr_mult", 0.5)
              if atr and sl_distance < min_sl_atr_mult * atr:
                  logger.info(f"{symbol}: SL {sl_distance/pip_size:.1f}p < {min_sl_atr_mult:.2f} ATR -> rejecting trade")
                  return None

              # Round SL/TP to symbol precision
              point = getattr(symbol_info, 'point', None)
              digits = getattr(symbol_info, 'digits', None)
              if point and digits is not None:
                  stop_loss = round(round(stop_loss / point) * point, int(digits))
                  take_profit = round(round(take_profit / point) * point, int(digits))
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
                  timestamp=datetime.now(timezone.utc),
                  breakout_level=breakout.level,
                  phase=(getattr(phase, 'name', None) if phase is not None else None),
                  session=(getattr(phase, 'session', None) if phase is not None else None),
                  phase_weight=(getattr(phase, 'weight', 0.0) if phase is not None else 0.0),
                  ttl_minutes=(getattr(phase, 'ttl_minutes', None) if phase is not None else None)
                  ,
                  features={
                      'strength': float(strength),
                      'momentum': float(momentum),
                      'dir_match': float(dir_match),
                      'trend_match': float(trend_match),
                      'spread_impact': float(spread_impact),
                  }
              )

              time_remaining_str = (
                  f"  Time remaining in candle: {time_remaining:.0f}s" if time_remaining is not None else ""
              )
              logger.debug(
                  f"*** SIGNAL: {symbol} {'BUY' if signal.type == 0 else 'SELL'} @ {signal.entry_price:.5f}"
                  f"  SL: {signal.stop_loss:.5f} ({signal.stop_loss_pips:.1f}p)"
                  f"  TP: {signal.take_profit:.5f} ({(abs(signal.take_profit - signal.entry_price)/pip_size):.1f}p)"
                  f"  R:R: {risk_reward_ratio:.1f}"
                  f"  Confidence: {signal.confidence:.2f}"
                  f"  Phase: {signal.phase or 'n/a'}"
                  f"{time_remaining_str}"
              )

              return signal

          except Exception as e:
              logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
              return None

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: np.ndarray, swing_lows: np.ndarray, symbol: str) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Calculate significant support and resistance levels based on clustered swing points.
        Now returns levels with their rank (strength).
        """
        if data is None or len(data) < self.swing_window * 2:
            return [], []

        resistance_levels_with_rank = []
        support_levels_with_rank = []

        # 1. Find levels from swing points using clustering
        if len(swing_highs) > 0:
            resistance_prices = data.iloc[swing_highs]['high'].values
            resistance_levels_with_rank = self._cluster_levels(resistance_prices, symbol)

        if len(swing_lows) > 0:
            support_prices = data.iloc[swing_lows]['low'].values
            support_levels_with_rank = self._cluster_levels(support_prices, symbol)

        # 2. Add recent extreme high/low as a fallback level with a default rank of 1
        recent_high = float(data['high'].tail(20).max())
        recent_low = float(data['low'].tail(20).min())
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol} in S/R calculation")
            return [], []
        pip_size = get_pip_size(symbol_info)
        proximity = self.proximity_threshold * pip_size

        # Add recent high if it's not already close to an existing resistance level
        if not any(abs(recent_high - level) <= proximity for level, _ in resistance_levels_with_rank):
            resistance_levels_with_rank.append((recent_high, 1))

        # Add recent low if it's not already close to an existing support level
        if not any(abs(recent_low - level) <= proximity for level, _ in support_levels_with_rank):
            support_levels_with_rank.append((recent_low, 1))

        # 3. Clean, sort by price, and limit the number of levels
        resistance_levels_with_rank.sort(key=lambda x: x[0], reverse=True)
        support_levels_with_rank.sort(key=lambda x: x[0])

        return resistance_levels_with_rank[:3], support_levels_with_rank[:3]

    def _cluster_levels(self, prices: np.ndarray, symbol: str) -> List[Tuple[float, int]]:
        """
        Cluster nearby price levels using peak ranking, scaled by symbol pip size.
        Now returns a list of tuples (level, rank) to preserve the strength score.
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

        consolidated_levels: List[Tuple[float, int]] = []
        for price, rank in ranked_prices:
            # Check if a similar price level is already in our list
            is_close = any(abs(price - level) <= proximity for level, _ in consolidated_levels)
            if not is_close:
                consolidated_levels.append((price, rank))

        # Sort by price, but return both price and rank
        consolidated_levels.sort(key=lambda x: x[0])
        return consolidated_levels
