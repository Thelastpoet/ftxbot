"""
Pure Price Action Strategy Module - LIVE FOREX TRADING
Real-time breakout detection with smart confirmation
"""
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
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from utils import SymbolPrecision, get_symbol_precision

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
    features: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)

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
        self.stop_loss_atr_multiplier = getattr(config, "stop_loss_atr_multiplier", 0.8)
        
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

        # M1 confirmation settings (default off to avoid over-restriction)
        self.m1_confirmation_enabled = getattr(config, "m1_confirmation_enabled", False)
        self.m1_confirmation_candles = getattr(config, "m1_confirmation_candles", 1)
        self.m1_confirmation_buffer_pips = getattr(config, "m1_confirmation_buffer_pips", 0.5)

        # Backtest mode: disables real-time age/close filters
        self.backtest_mode = getattr(config, "backtest_mode", False)

        # Optional: require last closed candle to confirm beyond the level
        self.require_close_breakout = getattr(config, "require_close_breakout", 0.0)
        self.breakout_confirmation = {
            "enabled": True,
            "modes": ["time_above", "retest_hold", "body_close"],
            "min_votes": 1,
            "zone_atr_mult": 0.15,
            "min_zone_pips": 2.0,
            "instant_buffer_mult": 0.10,
            "close_buffer_mult": 0.12,
            "retest_tol_mult": 0.08,
            "min_buffer_pips": 0.3,
            "time_above_seconds": 45,
            "retest_max_lookahead": 5,
            "min_excursion_mult": 0.30,
            "entry_style": "market_on_confirm"
        }
        bc = getattr(config, "breakout_confirmation", None)
        if isinstance(bc, dict):
            self.breakout_confirmation.update(bc)
        self._brk_state = {}  # per-symbol FSM state
        
        self.min_clv_no_pattern = getattr(config, "min_clv_no_pattern", 0.55)  # optional sanity check

        # Candlestick patterns influence (configurable, soft)
        self.pattern_indecision_penalty_per_hit = getattr(config, "pattern_indecision_penalty_per_hit", 0.06)
        self.pattern_indecision_penalty_cap = getattr(config, "pattern_indecision_penalty_cap", 0.12)
        self.pattern_mitigate_direction = getattr(config, "pattern_mitigate_direction", True)
        self.pattern_only_momentum = getattr(config, "pattern_only_momentum", False)
        self.pattern_strength_map = getattr(config, "pattern_strength_map", None)

        self.active_symbol = None
        self.active_symbol_params: Dict[str, float] = {}
    
    def set_symbol_params(self, symbol: str, params: Dict[str, float]) -> None:
        """Set dynamic parameter overrides for the active symbol."""
        self.active_symbol = symbol
        self.active_symbol_params = params or {}

    def _symbol_config(self, symbol: str) -> Dict[str, float]:
        for sym_config in getattr(self.config, "symbols", []):
            if sym_config.get('name') == symbol:
                return sym_config
        return {}

    def _param(self, key: str, default: float) -> float:
        """Return active parameter override if available."""
        if isinstance(self.active_symbol_params, dict) and key in self.active_symbol_params:
            try:
                return float(self.active_symbol_params[key])
            except (TypeError, ValueError):
                return default
        return default
    
    def _flag(self, key: str, default: bool) -> bool:
        """Return boolean override from active params (treat >=0.5 as True)."""
        try:
            val = self._param(key, 1.0 if default else 0.0)
            return bool(val >= 0.5)
        except Exception:
            return bool(default)
        
        # ---------- Breakout Confirmation (FSM) helpers ----------
    def _bcfg(self, key, default=None):
        """Read breakout_confirmation config key with a default."""
        return self.breakout_confirmation.get(key, default)

    def _zone_bounds(self, level: float, atr: float, pip_size: float) -> Tuple[float, float]:
        """Compute breakout acceptance zone around the S/R level."""
        min_zone = self._bcfg("min_zone_pips", 2.0) * pip_size if pip_size > 0 else 0.0
        atr_zone = (atr or 0.0) * self._bcfg("zone_atr_mult", 0.15)
        w = max(min_zone, atr_zone)
        return level - w / 2.0, level + w / 2.0

    def _buf(self, kind: str, spread: float, atr: float, pip_size: float) -> float:
        """
        Get adaptive buffer for 'instant', 'close', or 'retest_tol' checks.
        Uses max(min_buffer_pips, spread, atr * mult).
        """
        if kind == "instant_buffer":
            mult = self._bcfg("instant_buffer_mult", 0.10)
        elif kind == "close_buffer":
            mult = self._bcfg("close_buffer_mult", 0.12)
        elif kind == "retest_tol":
            mult = self._bcfg("retest_tol_mult", 0.08)
        else:
            mult = 0.10

        min_buf = self._bcfg("min_buffer_pips", 0.3) * (pip_size if pip_size > 0 else 1.0)
        return max(min_buf, float(spread or 0.0), float(atr or 0.0) * float(mult))

    def _arm_breakout(
        self,
        symbol: str,
        direction: str,           # "bullish" or "bearish"
        level: float,
        price: float,
        spread: float,
        atr: float,
        pip_size: float,
        now_ts: float,
    ) -> bool:
        """Arm the FSM when price trades beyond the zone + instant buffer."""
        lo, hi = self._zone_bounds(level, atr, pip_size)
        buf = self._buf("instant_buffer", spread, atr, pip_size)

        if direction == "bullish" and price >= hi + buf:
            self._brk_state[symbol] = {
                "armed": {"dir": "long", "level": level, "lo": lo, "hi": hi, "max_exc": 0.0, "armed_ts": now_ts},
                "dwell_start": None,
                "last_touch_ts": None,
            }
            logger.info(f"{symbol}: FSM ARMED long zone=({lo:.5f},{hi:.5f}) price={price:.5f}")
            return True

        if direction == "bearish" and price <= lo - buf:
            self._brk_state[symbol] = {
                "armed": {"dir": "short", "level": level, "lo": lo, "hi": hi, "max_exc": 0.0, "armed_ts": now_ts},
                "dwell_start": None,
                "last_touch_ts": None,
            }
            logger.info(f"{symbol}: FSM ARMED short zone=({lo:.5f},{hi:.5f}) price={price:.5f}")
            return True

        return False

    def _update_acceptance(
        self,
        symbol: str,
        completed_data: pd.DataFrame,  # closed candles (last row = last closed)
        forming_row: pd.Series,        # current forming candle
        spread: float,
        atr: float,
        pip_size: float,
        now_ts: float,
    ) -> Optional[Dict[str, float]]:
        """
        Accrue acceptance 'votes' after arming, then confirm if votes >= min_votes
        and minimum excursion is satisfied.
        """
        st = self._brk_state.get(symbol)
        if not st or not st.get("armed"):
            return None

        a = st["armed"]
        lo, hi = a["lo"], a["hi"]

        # Track excursion using forming candle
        if a["dir"] == "long":
            a["max_exc"] = max(a["max_exc"], max(0.0, float(forming_row["high"]) - hi))
        else:
            a["max_exc"] = max(a["max_exc"], max(0.0, lo - float(forming_row["low"])))

        # Voting modes
        votes = 0
        modes = set(self._bcfg("modes", []))
        min_votes = int(self._bcfg("min_votes", 1))
        close_buf = self._buf("close_buffer", spread, atr, pip_size)

        # 1) body_close: any last closed candle beyond zone + buffer
        if "body_close" in modes and len(completed_data) > 0:
            lc = float(completed_data.iloc[-1]["close"])
            if (a["dir"] == "long" and lc > hi + close_buf) or (a["dir"] == "short" and lc < lo - close_buf):
                votes += 1

        # 2) time_above: dwell time beyond the zone on the forming bar
        if "time_above" in modes:
            in_ctrl = (a["dir"] == "long" and float(forming_row["low"]) >= hi) or \
                      (a["dir"] == "short" and float(forming_row["high"]) <= lo)
            if in_ctrl and st["dwell_start"] is None:
                st["dwell_start"] = now_ts
            if not in_ctrl:
                st["dwell_start"] = None
            dwell_needed = float(self._bcfg("time_above_seconds", 45))
            if st["dwell_start"] is not None and (now_ts - st["dwell_start"]) >= dwell_needed:
                votes += 1

        # 3) retest_hold: first pullback touches the zone but does NOT close back inside
        if "retest_hold" in modes and len(completed_data) > 0:
            last = completed_data.iloc[-1]
            retest_ok = False
            if a["dir"] == "long":
                touched = float(last["low"]) <= hi and float(last["low"]) >= lo
                held = float(last["close"]) >= hi  # close remains above zone
                retest_ok = touched and held
            else:
                touched = float(last["high"]) >= lo and float(last["high"]) <= hi
                held = float(last["close"]) <= lo  # close remains below zone
                retest_ok = touched and held
            if retest_ok:
                votes += 1

        # Guard: require a minimum excursion away from the zone to avoid stop-runs
        min_exc = float(self._bcfg("min_excursion_mult", 0.30)) * float(atr or 0.0)
        if atr is not None and atr > 0 and a["max_exc"] < min_exc:
            return None

        if votes >= min_votes:
            self._brk_state[symbol]["confirmed"] = {"dir": a["dir"], "level": a["level"], "lo": lo, "hi": hi}
            logger.info(f"{symbol}: FSM CONFIRMED dir={a['dir']} votes={votes} exc={a['max_exc']:.5f}")
            return self._brk_state[symbol]["confirmed"]

        return None


    def _build_calibrator_features(
        self,
        breakout: BreakoutInfo,
        body_ratio: float,
        is_bullish_candle: bool,
        trend: str,
        spread: float,
        tick_size: float,
        pip_size: float,
        atr: Optional[float],
        pattern_strength: Optional[float] = None,
    ) -> Dict[str, float]:
        """Assemble calibrator feature vector from the current signal context."""
        strength = float(np.clip(getattr(breakout, 'strength_score', 0.0), 0.0, 1.0))
        if self.pattern_only_momentum:
            momentum = float(np.clip(pattern_strength if pattern_strength is not None else 0.0, 0.0, 1.0))
        else:
            momentum = float(np.clip(pattern_strength if pattern_strength is not None else body_ratio, 0.0, 1.0))
        dir_match = 1.0 if ((breakout.type == 'bullish') == is_bullish_candle) else 0.0
        if trend in ('bullish', 'bearish'):
            trend_match = 1.0 if trend == breakout.type else 0.0
        else:
            trend_match = 0.5
        if atr and atr > 0:
            spread_norm = spread / atr
        else:
            scale = pip_size if pip_size > 0 else tick_size
            spread_norm = (spread / (scale * 5.0)) if scale > 0 else 0.0
        spread_impact = float(np.clip(spread_norm, 0.0, 1.0))
        return {
            'strength': strength,
            'momentum': momentum,
            'dir_match': dir_match,
            'trend_match': trend_match,
            'spread_impact': spread_impact,
        }

    def _detect_candlestick_patterns(self, data: pd.DataFrame, breakout: BreakoutInfo) -> List[str]:
        """
        Detect a curated set of candlestick patterns around the breakout context.

        Uses TA-Lib pattern recognition on the last few CLOSED candles only,
        returning a list of detected pattern names that are directionally
        consistent with the breakout, plus direction-agnostic indecision patterns.

        The detection is conservative and only inspects the last up to 3
        closed candles to avoid stale context.
        """
        patterns_found: List[str] = []

        # Basic guards
        if data is None or len(data) < 20:
            return patterns_found
        if not TALIB_AVAILABLE or talib is None:
            return patterns_found

        try:
            open_prices = data['open'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            close_prices = data['close'].values

            if breakout.type == 'bullish':
                patterns_to_check = {
                    'ENGULFING': talib.CDLENGULFING,
                    'HAMMER': talib.CDLHAMMER,
                    'INVERTED_HAMMER': talib.CDLINVERTEDHAMMER,
                    'PIERCING': talib.CDLPIERCING,
                    'MORNING_STAR': talib.CDLMORNINGSTAR,
                    'BULLISH_HARAMI': talib.CDLHARAMI,
                    'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
                    'MARUBOZU': talib.CDLMARUBOZU,
                }
                is_signal = lambda v: v > 0
            else:
                patterns_to_check = {
                    'ENGULFING': talib.CDLENGULFING,
                    'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
                    'HANGING_MAN': talib.CDLHANGINGMAN,
                    'DARK_CLOUD': talib.CDLDARKCLOUDCOVER,
                    'EVENING_STAR': talib.CDLEVENINGSTAR,
                    'BEARISH_HARAMI': talib.CDLHARAMI,
                    'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
                    'MARUBOZU': talib.CDLMARUBOZU,
                }
                is_signal = lambda v: v < 0

            # Indecision/neutral patterns (direction-agnostic)
            neutral_patterns = {
                'DOJI': talib.CDLDOJI,
                'SPINNING_TOP': talib.CDLSPINNINGTOP,
                'HIGH_WAVE': talib.CDLHIGHWAVE,
                'DOJI_STAR': talib.CDLDOJISTAR,
                'LONG_LEGGED_DOJI': talib.CDLLONGLEGGEDDOJI,
                'RICKSHAWMAN': talib.CDLRICKSHAWMAN,
            }

            # Check last up to 3 closed candles only
            lookback = min(3, len(close_prices))

            for name, func in patterns_to_check.items():
                try:
                    result = func(open_prices, high_prices, low_prices, close_prices)
                except Exception:
                    continue
                for i in range(1, lookback + 1):
                    if is_signal(result[-i]):
                        patterns_found.append(name)
                        break

            # Add indecision if found
            for name, func in neutral_patterns.items():
                try:
                    result = func(open_prices, high_prices, low_prices, close_prices)
                except Exception:
                    continue
                for i in range(1, lookback + 1):
                    if result[-i] != 0:
                        patterns_found.append(name)
                        break

            return patterns_found
        except Exception:
            # Be silent here to avoid noisy logs during live trading
            return patterns_found

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

    def _detect_breakout(
        self,
        tick: any,
        resistance_levels: List[float],
        support_levels: List[float],
        atr: Optional[float],
        precision: SymbolPrecision,
    ) -> Optional[BreakoutInfo]:
        """
        Detect and score breakout
        """
        tick_size = precision.tick_size or 0.0
        pip_size = precision.pip_size or tick_size

        if atr:
            pip_threshold = self._param('breakout_threshold_pips', self.breakout_threshold_pips) * pip_size
            vol_threshold = atr * self.min_extension_atr
            threshold = max(pip_threshold, vol_threshold)
            max_extension = atr * self.max_extension_atr
        else:
            threshold = self._param('breakout_threshold_pips', self.breakout_threshold_pips) * pip_size
            scale = pip_size if pip_size > 0 else (tick_size if tick_size > 0 else 1.0)
            max_extension = 20 * scale
            logger.warning("Running without ATR - using fixed thresholds")

        scale = pip_size if pip_size > 0 else (tick_size if tick_size > 0 else 1.0)

        for resistance in resistance_levels:
            if tick.ask > resistance + threshold:
                distance = tick.ask - resistance

                if distance > max_extension:
                    continue  # Too extended

                if atr:
                    strength = min(distance / atr, 1.0)
                else:
                    strength = min(distance / (10 * scale), 1.0)

                distance_pips = distance / scale

                return BreakoutInfo(
                    type='bullish',
                    level=resistance,
                    entry_price=tick.ask,
                    distance=distance,
                    distance_pips=distance_pips,
                    strength_score=strength
                )

        for support in support_levels:
            if tick.bid < support - threshold:
                distance = support - tick.bid

                if distance > max_extension:
                    continue

                if atr:
                    strength = min(distance / atr, 1.0)
                else:
                    strength = min(distance / (10 * scale), 1.0)

                distance_pips = distance / scale

                return BreakoutInfo(
                    type='bearish',
                    level=support,
                    entry_price=tick.bid,
                    distance=distance,
                    distance_pips=distance_pips,
                    strength_score=strength
                )

        return None   

    def _calculate_confidence(
        self,
        breakout,
        candles,      # expect closed candles (last row = last closed candle)
        atr,
        spread,
        pip_size,
        min_confidence=0.6,
    ):
        """
        Minimal confidence scorer:
        - base score
        - plus normalized breakout.strength_score contribution
        - minus a single ATR-normalized spread penalty (light)
        Returns: (confidence: float, should_trade: bool)
        """

        # require at least one closed candle (safety)
        if candles is None or len(candles) == 0:
            return 0.0, False

        # --- simple, explicit constants (easy to tune) ---
        base = 0.30
        breakout_weight = 0.65       # how much the normalized breakout strength moves the score
        spread_penalty_max = 0.25    # maximum penalty subtracted for large spread
        spread_ratio_cap = 0.5       # spread/atr ratio that maps to the max penalty

        # normalized breakout strength (ensure 0..1)
        strength = 0.0
        try:
            strength = float(getattr(breakout, "strength_score", 0.0) or 0.0)
        except Exception:
            strength = 0.0
        strength = max(0.0, min(1.0, strength))

        # base + breakout contribution
        confidence = base + strength * breakout_weight

        # simple spread penalty (ATR-normalized when possible)
        try:
            if atr and float(atr) > 0:
                spread_ratio = abs(float(spread)) / float(atr)
                # linear mapping: 0 -> 0 penalty, spread_ratio >= spread_ratio_cap -> spread_penalty_max
                penalty = (spread_ratio / spread_ratio_cap) * spread_penalty_max if spread_ratio > 0 else 0.0
                penalty = max(0.0, min(spread_penalty_max, penalty))
                confidence -= penalty
            else:
                # fallback: mild pip-based penalty if pip_size provided and spread is large in pips
                if pip_size and float(pip_size) > 0:
                    spread_pips = abs(float(spread)) / float(pip_size)
                    if spread_pips > 3:
                        # scale into penalty range but keep it small
                        penalty = min(spread_penalty_max, ((spread_pips - 3) / 10.0) * spread_penalty_max)
                        confidence -= penalty
        except Exception:
            # on any numeric error, do not crash the scorer â€” keep current confidence
            pass

        # clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        return float(confidence), (confidence >= float(min_confidence))


    def _calculate_stop_loss(
        self,
        breakout: BreakoutInfo,
        atr: Optional[float],
        precision: SymbolPrecision,
        tick: any,
        symbol: str,
        min_stop_loss_pips: float,
        stop_loss_buffer_pips: float,
        stop_loss_atr_multiplier: float,
    ) -> Optional[float]:
        """
        Calculates the stop loss for a given breakout signal, ensuring it's logical and safe.
        The stop loss is determined by the most conservative (widest) position based on:
        1. The broken S/R level (structural_sl).
        2. A distance based on volatility (ATR).

        It now returns None if the calculated "natural" SL is smaller than the configured
        minimum, effectively filtering out low-quality signals.
        """
        tick_size = precision.tick_size or 0.0
        pip_size = precision.pip_size or tick_size

        if pip_size <= 0:
            logger.warning(f"{symbol}: Unable to resolve pip size for stop-loss calculation.")
            return None

        atr_multiplier = max(float(stop_loss_atr_multiplier), 0.0)
        natural_volatility_dist = (atr * atr_multiplier) if atr and atr > 0 else 0.0

        spread = tick.ask - tick.bid
        spread_buffer = spread + (tick_size if tick_size > 0 else 0.0)
        configured_buffer = stop_loss_buffer_pips * pip_size
        safety_buffer = max(spread_buffer, configured_buffer)

        entry_price = breakout.entry_price
        broken_level = breakout.level

        if breakout.type == 'bullish':
            structural_sl = broken_level - safety_buffer
            volatility_sl = entry_price - natural_volatility_dist
            natural_sl = min(structural_sl, volatility_sl)
        else:  # Bearish
            structural_sl = broken_level + safety_buffer
            volatility_sl = entry_price + natural_volatility_dist
            natural_sl = max(structural_sl, volatility_sl)

        natural_sl_pips = abs(entry_price - natural_sl) / pip_size if pip_size > 0 else 0.0

        atr_based_min_pips = (atr * atr_multiplier) / pip_size if (atr and pip_size > 0) else 0.0
        min_allowed_sl_pips = max(float(min_stop_loss_pips), float(atr_based_min_pips))

        if natural_sl_pips < min_allowed_sl_pips:
            logger.debug(
                f"{symbol}: Trade rejected. Natural SL ({natural_sl_pips:.1f} pips) is below "
                f"the minimum allowed ({min_allowed_sl_pips:.1f} pips; configured floor={min_stop_loss_pips} pips; "
                f"ATR-based minimum={atr_based_min_pips:.2f} pips)."
            )
            return None

        epsilon_base = tick_size if tick_size > 0 else pip_size
        epsilon = epsilon_base / 10.0 if epsilon_base > 0 else 0.0
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
            
            symbol_overrides = self._symbol_config(symbol)
            precision = get_symbol_precision(symbol_info, overrides=symbol_overrides)
            tick_size = precision.tick_size or 0.0
            pip_size = precision.pip_size or tick_size
            if pip_size <= 0:
                logger.error(f"{symbol}: Unable to determine pip size for signal generation")
                return None
            spread_ticks = (tick.ask - tick.bid) / tick_size if tick_size > 0 else 0.0
            
            # Get forming candle
            forming_candle = data.iloc[-1]
            completed_data = data.iloc[:-1].tail(self.lookback_period)
            
            swing_highs, swing_lows = self.find_swing_points(completed_data)
            
            if len(swing_highs) == 0 and len(swing_lows) == 0:
                logger.debug(f"No swing points found for {symbol}")
                return None
            
            default_min_sl = symbol_overrides.get('min_stop_loss_pips', self.min_stop_loss_pips)
            default_buffer = symbol_overrides.get('stop_loss_buffer_pips', self.stop_loss_buffer_pips)
            default_atr_mult = symbol_overrides.get('stop_loss_atr_multiplier', self.stop_loss_atr_multiplier)
            default_rr = symbol_overrides.get('risk_reward_ratio', self.risk_reward_ratio)
            default_min_conf = symbol_overrides.get('min_confidence', self.min_confidence)

            min_stop_loss_pips = self._param('min_stop_loss_pips', default_min_sl)
            stop_loss_buffer_pips = self._param('stop_loss_buffer_pips', default_buffer)
            stop_loss_atr_multiplier = self._param('stop_loss_atr_multiplier', default_atr_mult)
            risk_reward_ratio = self._param('risk_reward_ratio', default_rr)
            min_confidence = self._param('min_confidence', default_min_conf)
            
            # Calculate ATR from completed data
            atr = None
            if TALIB_AVAILABLE and len(completed_data) >= 14:
                atr = talib.ATR(
                    completed_data['high'].values,
                    completed_data['low'].values,
                    completed_data['close'].values,
                    timeperiod=14
                )[-1]
                atr_ticks = atr / tick_size if tick_size > 0 else 0.0
                atr_pips = atr / pip_size if pip_size > 0 else 0.0
                logger.debug(f"{symbol}: ATR={atr_ticks:.1f} ticks (~{atr_pips:.1f} pips)")
            else:
                logger.warning(f"{symbol}: Running without ATR - TA-Lib not available or insufficient data")
            
            # QUICK FILTERS
            # 1. Spread filter
            max_spread = (atr * self.max_spread_atr_ratio) if atr else (self.max_spread_pips * pip_size)
            if tick.ask - tick.bid > max_spread:
                logger.debug(f"{symbol}: Spread too high ({spread_ticks:.1f} ticks)")
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
            if atr and (atr / pip_size if pip_size > 0 else 0.0) < 10:
                min_body_ratio *= 0.8
            
            range_ok = (atr is not None and atr > 0 and (candle_range / atr) >= 0.8)
            if body_ratio < min_body_ratio and not range_ok:
                logger.debug(f"{symbol}: Insufficient momentum (body_ratio={body_ratio:.2f}, range/ATR={(candle_range/atr) if atr else 0:.2f})")
                return None
            
            # Calculate S/R levels
            resistance_levels, support_levels = self.calculate_support_resistance(completed_data, swing_highs, swing_lows, symbol)
            
            if not resistance_levels and not support_levels:
                logger.debug(f"{symbol}: No S/R levels found")
                return None
            
            # Detect breakout
            breakout = self._detect_breakout(tick, resistance_levels, support_levels, atr, precision)
            
            if not breakout:
                return None

            logger.info(
                f"{symbol}: Breakout detected - {breakout.type} @ {breakout.entry_price:.5f}, "
                f"distance={breakout.distance_pips:.1f}p, strength={breakout.strength_score:.2f}"
            )

            use_fsm = float(self._flag('require_close_breakout', self.require_close_breakout) or 0.0) >= 0.5
            if use_fsm and self._bcfg("enabled", True):
                spread = (tick.ask - tick.bid) if tick else 0.0
                pip_size = precision.pip_size or (precision.tick_size or 0.0001)
                now_ts = float(getattr(tick, "time", 0))
                if symbol not in self._brk_state or not self._brk_state[symbol].get("armed"):
                    self._arm_breakout(symbol,
                                    "bullish" if breakout.type == "bullish" else "bearish",
                                    breakout.level, breakout.entry_price,
                                    spread, atr or 0.0, pip_size, now_ts)
                    if not self._brk_state.get(symbol, {}).get("armed"):
                        return None
                confirmed = self._update_acceptance(symbol, completed_data, forming_candle,
                                                    spread, atr or 0.0, pip_size, now_ts)
                if not confirmed:
                    return None
                if self._bcfg("entry_style") == "limit_on_retest":
                    breakout = (breakout._replace(entry_price=confirmed["hi"])
                                if breakout.type == "bullish" else
                                breakout._replace(entry_price=confirmed["lo"]))
            
            # Structure distance check
            res_levels = sorted(resistance_levels)
            sup_levels = sorted(support_levels)
            next_structure = None
            if breakout.type == 'bullish':
                candidates = [r for r in res_levels if r > breakout.entry_price]
                if candidates: next_structure = min(candidates)
            else:
                candidates = [s for s in sup_levels if s < breakout.entry_price]
                if candidates: next_structure = max(candidates)

            if next_structure is not None:
                min_sl_pips = min_stop_loss_pips
                min_sl_distance = max(min_sl_pips * pip_size, (atr * 0.8) if atr else 0.0)
                distance_to_next = abs(next_structure - breakout.entry_price)
                
                room_req_pips = getattr(self.config, "min_room_after_breakout_pips", None)
                if room_req_pips:
                    room_req = room_req_pips * pip_size
                else:
                    room_req = max(1.0 * min_sl_distance, 0.5 * atr if atr else 0.0)

                if distance_to_next < room_req:
                    logger.debug(
                        f"{symbol}: Limited room after breakout. Skipping.")
                    return None
            
            # Calculate confidence
            is_bullish_candle = forming_candle['close'] > forming_candle['open']
            spread = tick.ask - tick.bid
            # Candlestick patterns from last closed candles (M15 context)
            patterns_found: List[str] = []
            try:
                patterns_found = self._detect_candlestick_patterns(completed_data, breakout)
            except Exception:
                patterns_found = []

            # If no confirming candlestick pattern was found, perform a CLV check on the
            # forming candle as a final momentum filter before calculating confidence.
            if not patterns_found:
                rng = max(forming_candle['high'] - forming_candle['low'], 1e-9)
                if breakout.type == 'bullish':
                    clv = (forming_candle['close'] - forming_candle['low']) / rng
                else: # bearish
                    clv = (forming_candle['high'] - forming_candle['close']) / rng
                    
                if clv < self.min_clv_no_pattern:
                    logger.info(f"{symbol}: Trade rejected by CLV filter (clv={clv:.2f} < {self.min_clv_no_pattern})")
                    return None

            # Summarize patterns for momentum features/logging
            pattern_strength = 0.0
            pattern_conf_hits = 0
            pattern_indec_hits = 0
            try:
                pats = set([p.upper() for p in patterns_found])
                confirming = {
                    'ENGULFING', 'HAMMER', 'INVERTED_HAMMER', 'PIERCING', 'MORNING_STAR',
                    'BULLISH_HARAMI', 'BEARISH_HARAMI', 'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS', 'MARUBOZU',
                    'EVENING_STAR', 'DARK_CLOUD', 'SHOOTING_STAR', 'HANGING_MAN'
                }
                indecision = {'DOJI', 'SPINNING_TOP'}
                pattern_conf_hits = len(pats.intersection(confirming))
                pattern_indec_hits = len(pats.intersection(indecision))
                # Build strength map with overrides
                strength_map = {
                    'MARUBOZU': 1.0,
                    'THREE_WHITE_SOLDIERS': 0.95,
                    'THREE_BLACK_CROWS': 0.95,
                    'ENGULFING': 0.90,
                    'MORNING_STAR': 0.85,
                    'EVENING_STAR': 0.85,
                    'PIERCING': 0.75,
                    'DARK_CLOUD': 0.75,
                    'HAMMER': 0.65,
                    'INVERTED_HAMMER': 0.65,
                    'SHOOTING_STAR': 0.65,
                    'HANGING_MAN': 0.65,
                    'BULLISH_HARAMI': 0.60,
                    'BEARISH_HARAMI': 0.60,
                }
                if isinstance(self.pattern_strength_map, dict) and self.pattern_strength_map:
                    try:
                        custom = {str(k).upper(): float(v) for k, v in self.pattern_strength_map.items()}
                        strength_map.update(custom)
                    except Exception:
                        pass
                scores = [float(strength_map[p]) for p in pats if p in strength_map]
                if scores:
                    base = max(scores)
                    extra = 0.10 * max(0, len(scores) - 1)
                    pattern_strength = float(np.clip(base + extra, 0.0, 1.0))
            except Exception:
                pattern_strength = 0.0

            confidence, should_trade = self._calculate_confidence(
                breakout, completed_data, atr, spread, pip_size, min_confidence
            )            
            
            if not should_trade:
                logger.info(f"{symbol}: Trade filtered out (confidence={confidence:.2f})")
                return None
            
            # Calculate SL/TP
            stop_loss = self._calculate_stop_loss(
                breakout, atr, precision, tick, symbol, min_stop_loss_pips, stop_loss_buffer_pips,
                stop_loss_atr_multiplier
            )
            if stop_loss is None:
                return None  # Signal rejected by the SL quality filter

            take_profit = self._calculate_take_profit(breakout.entry_price, stop_loss, risk_reward_ratio, breakout.type)
            sl_distance = abs(breakout.entry_price - stop_loss)

            # Safety checks for SL distance
            if sl_distance <= 0:
                logger.error(f"{symbol}: Computed sl_distance <= 0 (entry={breakout.entry_price}, sl={stop_loss}) -> rejecting")
                return None

            min_sl_atr_mult = getattr(self.config, "min_sl_atr_mult", 0.5)
            if atr and sl_distance < min_sl_atr_mult * atr:
                logger.info(f"{symbol}: SL {sl_distance/tick_size:.1f}t < {min_sl_atr_mult:.2f} ATR -> rejecting trade")
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
            
            features = self._build_calibrator_features(
                breakout=breakout,
                body_ratio=body_ratio,
                is_bullish_candle=is_bullish_candle,
                trend=trend,
                spread=spread,
                tick_size=tick_size,
                pip_size=pip_size,
                atr=atr,
                pattern_strength=pattern_strength,
            )
            # Add pattern-derived diagnostics to features
            features['pattern_strength'] = float(pattern_strength)
            features['pattern_conf_hits'] = float(pattern_conf_hits)
            features['pattern_indecision_hits'] = float(pattern_indec_hits)
            parameter_snapshot = {
                'min_stop_loss_pips': float(min_stop_loss_pips),
                'stop_loss_buffer_pips': float(stop_loss_buffer_pips),
                'stop_loss_atr_multiplier': float(stop_loss_atr_multiplier),
                'risk_reward_ratio': float(risk_reward_ratio),
                'min_confidence': float(min_confidence),
                'breakout_threshold_pips': float(self._param('breakout_threshold_pips', self.breakout_threshold_pips)),
                'min_peak_rank': float(self._param('min_peak_rank', self.min_peak_rank)),
                'proximity_threshold_pips': float(self._param('proximity_threshold_pips', self.proximity_threshold)),
                'require_close_breakout': float(self._param('require_close_breakout', self.require_close_breakout)),
            }
            # Include patterns and settings for logging/traceability
            parameter_snapshot['patterns'] = list(patterns_found) if patterns_found else []
            parameter_snapshot['pattern_only_momentum'] = bool(self.pattern_only_momentum)
            # Create signal
            signal = TradingSignal(
                type=0 if breakout.type == 'bullish' else 1,
                entry_price=breakout.entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                stop_loss_pips=sl_distance / pip_size if pip_size > 0 else 0.0,
                reason=f"live_{breakout.type}_breakout",
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                breakout_level=breakout.level,
                features=features,
                parameters=parameter_snapshot,
            )
            
            time_remaining_str = (
                f"  Time remaining in candle: {time_remaining:.0f}s" if time_remaining is not None else ""
            )
            patterns_str = ", ".join(patterns_found) if patterns_found else "none"
            logger.info(
                f"*** SIGNAL: {symbol} {'BUY' if signal.type == 0 else 'SELL'} @ {signal.entry_price:.5f}"
                f"  SL: {signal.stop_loss:.5f} ({sl_distance/tick_size:.0f} ticks, ~{signal.stop_loss_pips:.1f} pips)"
                f"  TP: {signal.take_profit:.5f} ({abs(signal.take_profit - signal.entry_price)/tick_size:.0f} ticks)"
                f"  R:R: {risk_reward_ratio:.1f}"
                f"  Confidence: {signal.confidence:.2f}"
                f"  Patterns: {patterns_str}"
                f"{time_remaining_str}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: np.ndarray, swing_lows: np.ndarray, symbol: str) -> Tuple[List[float], List[float]]:
        """
        Calculate support and resistance levels based on clustered swing points.
        """
        if data is None or len(data) < self.swing_window * 2:
            return [], []

        resistance_levels = []
        support_levels = []

        if len(swing_highs) > 0:
            resistance_prices = data.iloc[swing_highs]['high'].values
            resistance_levels = self._cluster_levels(resistance_prices, symbol)

        if len(swing_lows) > 0:
            support_prices = data.iloc[swing_lows]['low'].values
            support_levels = self._cluster_levels(support_prices, symbol)

        recent_high = float(data['high'].tail(20).max())
        recent_low = float(data['low'].tail(20).min())

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol} in S/R calculation")
            return [], []

        symbol_overrides = self._symbol_config(symbol)
        precision = get_symbol_precision(symbol_info, overrides=symbol_overrides)
        pip_size = precision.pip_size or (precision.tick_size or 0.0)
        if pip_size <= 0:
            return resistance_levels[:3], support_levels[:3]

        proximity = self._param('proximity_threshold_pips', self.proximity_threshold) * pip_size

        if not any(abs(recent_high - level) <= proximity for level in resistance_levels):
            resistance_levels.append(recent_high)

        if not any(abs(recent_low - level) <= proximity for level in support_levels):
            support_levels.append(recent_low)

        resistance = sorted(list(set(resistance_levels)), reverse=True)[:3]
        support = sorted(list(set(support_levels)))[:3]

        return resistance, support

    def _cluster_levels(self, prices: np.ndarray, symbol: str) -> List[float]:
        """
        Cluster nearby price levels using peak ranking, scaled by symbol pip size
        """
        if len(prices) == 0:
            return []

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return []

        precision = get_symbol_precision(symbol_info, overrides=self._symbol_config(symbol))
        pip_size = precision.pip_size or (precision.tick_size or 0.0)
        if pip_size <= 0:
            return sorted(set(float(p) for p in prices))

        proximity = self.proximity_threshold * pip_size  # convert pips to price units

        ranked_prices = []
        for price in prices:
            rank = sum(1 for p in prices if abs(p - price) <= proximity)
            if rank >= int(self._param('min_peak_rank', self.min_peak_rank)):
                ranked_prices.append((price, rank))

        ranked_prices.sort(key=lambda x: x[1], reverse=True)

        consolidated_levels: List[float] = []
        for price, _rank in ranked_prices:
            is_close = any(abs(price - level) <= proximity for level in consolidated_levels)
            if not is_close:
                consolidated_levels.append(price)

        return sorted(consolidated_levels)
