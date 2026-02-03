"""
Core Price Action Breakout Strategy
"""

from dataclasses import dataclass
from datetime import datetime, timezone, time
from typing import List, Optional, Tuple, NamedTuple

import MetaTrader5 as mt5
import pandas as pd
import logging

from utils import resolve_pip_size

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    type: int  # 0 = BUY, 1 = SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_loss_pips: float
    reason: str
    timestamp: datetime
    breakout_level: float

class BreakoutInfo(NamedTuple):
    type: str      # 'bullish' or 'bearish'
    level: float
    entry_price: float

class PurePriceActionStrategy:
    """
    Breakout strategy based on swing levels.

    Core Logic:
    1. Find swing highs/lows to identify S/R levels
    2. Detect when price breaks through S/R by threshold
    3. Place SL beyond opposing structure
    4. Target based on risk-reward ratio
    """

    def __init__(self, config, mt5_client=None):
        self.config = config
        self.mt5_client = mt5_client
        self.mt5 = getattr(mt5_client, "mt5", mt5)

        # Core parameters
        self.lookback_period = getattr(config, 'lookback_period', 20)
        self.swing_window = getattr(config, 'swing_window', 5)
        self.breakout_threshold_pips = getattr(config, 'breakout_threshold', 0)
        self.breakout_window_bars = int(getattr(config, 'breakout_window_bars', 1) or 1)
        self.min_stop_loss_pips = getattr(config, 'min_stop_loss_pips', 20)
        self.stop_loss_buffer_pips = getattr(config, 'stop_loss_buffer_pips', 15)
        self.risk_reward_ratio = getattr(config, 'risk_reward_ratio', 2.0)
        self.min_rr = getattr(config, 'min_rr', 1.0)
        self.max_extension_pips = getattr(config, 'max_extension_pips', None)
        self.max_extension_atr_mult = getattr(config, 'max_extension_atr_mult', None)
        self.sr_lookback_period = int(getattr(config, 'sr_lookback_period', 80))
        self.sr_proximity_pips = float(getattr(config, 'sr_proximity_pips', 10.0))
        self.tp_buffer_pips = float(getattr(config, 'tp_buffer_pips', 2.0))
        self.structure_min_touches = int(getattr(config, 'structure_min_touches', 2))
        self.structure_atr_band_mult = float(getattr(config, 'structure_atr_band_mult', 0.25))
        # Extension cap (market-adaptive)
        self.extension_atr_ratio_period = int(getattr(config, 'extension_atr_ratio_period', 200))
        self.extension_atr_min_mult = float(getattr(config, 'extension_atr_min_mult', 1.0))
        self.extension_atr_max_mult = float(getattr(config, 'extension_atr_max_mult', 3.0))
        self.extension_atr_sensitivity = float(getattr(config, 'extension_atr_sensitivity', 1.5))

        # ATR for dynamic SL buffer and breakout threshold
        self.atr_period = int(getattr(config, 'atr_period', 14))
        self.atr_sl_k = float(getattr(config, 'atr_sl_k', 0.6))
        self.min_sl_buffer_pips = float(getattr(config, 'min_sl_buffer_pips', 10))
        self.max_sl_pips = getattr(config, 'max_sl_pips', None)

        # ATR-based breakout threshold (if set, overrides fixed pips)
        self.breakout_threshold_atr_mult = getattr(config, 'breakout_threshold_atr_mult', None)
        # Spread-anchored breakout threshold (if set)
        self.breakout_threshold_spread_mult = getattr(config, 'breakout_threshold_spread_mult', 2.0)

        # Trend filter (EMA alignment)
        self.use_trend_filter = getattr(config, 'use_trend_filter', True)
        self.trend_ema_period = int(getattr(config, 'trend_ema_period', 50))

        # EMA slope filter (reject trades when EMA is flat/ranging)
        self.use_ema_slope_filter = getattr(config, 'use_ema_slope_filter', True)
        self.ema_slope_period = int(getattr(config, 'ema_slope_period', 20))
        self.min_ema_slope_pips_per_bar = float(getattr(config, 'min_ema_slope_pips_per_bar', 0.1))
        self.min_ema_slope_atr_per_bar = getattr(config, 'min_ema_slope_atr_per_bar', None)

        # Spread guard
        self.spread_guard_pips_default = getattr(config, 'spread_guard_pips_default', None)

        # Breakout confirmations
        self.require_structure_confirmation = bool(getattr(config, 'require_structure_confirmation', True))
        self.require_two_bar_confirmation = bool(getattr(config, 'require_two_bar_confirmation', True))
        self.require_fresh_breakout = bool(getattr(config, 'require_fresh_breakout', True))

        # Momentum filter (breakout strength)
        self.use_momentum_filter = bool(getattr(config, 'use_momentum_filter', False))
        self.momentum_atr_mult = float(getattr(config, 'momentum_atr_mult', 1.0))
        self.momentum_close_percent = float(getattr(config, 'momentum_close_percent', 0.7))

        # Entry pacing
        self.entry_cooldown_bars = int(getattr(config, 'entry_cooldown_bars', 0))
        self.entry_window_bars = int(getattr(config, 'entry_window_bars', 0))

        # Take-profit mode
        self.tp_mode = str(getattr(config, 'tp_mode', 'structure')).lower()
        self.tp_r_multiple = float(getattr(config, 'tp_r_multiple', 1.5))
        self.tp_use_structure_cap = bool(getattr(config, 'tp_use_structure_cap', False))

        # Max SL by ATR (adaptive cap)
        self.max_sl_atr_mult = getattr(config, 'max_sl_atr_mult', None)

        # Session filter (UTC)
        self.use_session_filter = bool(getattr(config, 'use_session_filter', False))
        self.session_start_utc = getattr(config, 'session_start_utc', "07:00")
        self.session_end_utc = getattr(config, 'session_end_utc', "20:00")
        self._session_start = self._parse_time(self.session_start_utc)
        self._session_end = self._parse_time(self.session_end_utc)

        # Duplicate signal prevention (one per bar per direction)
        self._last_breakout_bar = {}
        self._last_signal_time = {}

    def _record_reject(self, symbol: str, reason: str) -> None:
        """Record rejection reason if diagnostics are enabled."""
        try:
            diag = getattr(self.config, "diagnostics", None)
            if diag and hasattr(diag, "record_reject"):
                diag.record_reject(reason, symbol)
        except Exception:
            pass

    def _parse_time(self, value: str) -> Optional[time]:
        try:
            parts = value.split(":")
            if len(parts) != 2:
                return None
            return time(int(parts[0]), int(parts[1]))
        except Exception:
            return None

    def _in_session(self, ts: datetime) -> bool:
        if not self._session_start or not self._session_end:
            return True
        t = ts.time()
        if self._session_start <= self._session_end:
            return self._session_start <= t <= self._session_end
        # Session wraps midnight
        return t >= self._session_start or t <= self._session_end

    # ----- MT5 helpers -----
    def _get_tick(self, symbol: str):
        if self.mt5_client:
            return self.mt5_client.get_symbol_info_tick(symbol)
        return mt5.symbol_info_tick(symbol)

    def _get_symbol_info(self, symbol: str):
        if self.mt5_client:
            return self.mt5_client.get_symbol_info(symbol)
        return mt5.symbol_info(symbol)

    # ----- Swing Points and S/R -----
    def _find_swing_indices(self, series: pd.Series, window: int) -> List[int]:
        """Find swing high indices where value is strictly greater than surrounding bars."""
        idxs: List[int] = []
        values = series.values
        n = len(values)
        for i in range(window, n - window):
            left = values[i - window:i]
            right = values[i + 1:i + window + 1]
            if len(left) > 0 and len(right) > 0:
                if values[i] > left.max() and values[i] > right.max():
                    idxs.append(i)
        return idxs

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        if data is None or len(data) < self.swing_window * 2 + 1:
            return [], []
        highs = self._find_swing_indices(data['high'], self.swing_window)
        lows = self._find_swing_indices(-data['low'], self.swing_window)
        return highs, lows

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: List[int],
                                     swing_lows: List[int], symbol: str, pip: float,
                                     proximity_pips: float = 10.0,
                                     atr_last: Optional[float] = None,
                                     min_touches: int = 2,
                                     atr_band_mult: float = 0.25) -> Tuple[List[float], List[float]]:
        """Extract recent distinct swing highs/lows as resistance/support."""
        proximity = float(proximity_pips) * pip  # proximity filter in pips
        atr_band = float(atr_last) * float(atr_band_mult) if atr_last and atr_last > 0 else None

        res: List[float] = []
        sup: List[float] = []
        max_levels = 5

        # Take recent swings
        for i in reversed(swing_highs[-50:]):
            level = float(data.iloc[i]['high'])
            if atr_band is not None and min_touches > 1:
                touches = int(((data['high'] - level).abs() <= atr_band).sum())
                if touches < min_touches:
                    continue
            if not any(abs(level - x) <= proximity for x in res):
                res.append(level)
            if len(res) >= max_levels:
                break

        for i in reversed(swing_lows[-50:]):
            level = float(data.iloc[i]['low'])
            if atr_band is not None and min_touches > 1:
                touches = int(((data['low'] - level).abs() <= atr_band).sum())
                if touches < min_touches:
                    continue
            if not any(abs(level - x) <= proximity for x in sup):
                sup.append(level)
            if len(sup) >= max_levels:
                break

        return sorted(res), sorted(sup)

    # ----- Breakout Detection -----
    def _detect_breakout(self, last_close: float, resistance: List[float],
                         support: List[float], threshold: float) -> Optional[BreakoutInfo]:
        """Detect breakout by finding the highest R or lowest S broken."""
        # BUY: Find highest resistance broken
        broken_resistances = [level for level in resistance if last_close > level + threshold]
        if broken_resistances:
            highest_broken = max(broken_resistances)
            return BreakoutInfo('bullish', highest_broken, last_close)

        # SELL: Find lowest support broken
        broken_supports = [level for level in support if last_close < level - threshold]
        if broken_supports:
            lowest_broken = min(broken_supports)
            return BreakoutInfo('bearish', lowest_broken, last_close)

        return None

    def _find_breakout_age(self, closes: pd.Series, level: float, threshold: float,
                            breakout_type: str, window_bars: int) -> Optional[int]:
        """Find how many bars ago the breakout first occurred within a window."""
        if closes is None or len(closes) < 2:
            return None
        max_age = max(0, int(window_bars))
        # Ensure we don't index beyond available data
        max_age = min(max_age, len(closes) - 2)
        for age in range(0, max_age + 1):
            idx = -1 - age
            prev_idx = idx - 1
            close_now = float(closes.iloc[idx])
            close_prev = float(closes.iloc[prev_idx])
            if breakout_type == 'bullish':
                if close_now > level + threshold and close_prev <= level + threshold:
                    return age
            else:
                if close_now < level - threshold and close_prev >= level - threshold:
                    return age
        return None

    def _passes_momentum_filter(self, breakout_type: str, bar: pd.Series, atr_last: Optional[float],
                                pip: float, atr_mult: float, close_percent: float) -> bool:
        """Require a strong impulse candle for breakout confirmation."""
        try:
            if atr_last is None or atr_last <= 0 or pip <= 0:
                return True  # Skip if ATR unavailable
            high = float(bar['high'])
            low = float(bar['low'])
            close = float(bar['close'])
            rng = high - low
            if rng <= 0:
                return False
            if rng < (atr_last * float(atr_mult)):
                return False
            pos = (close - low) / rng  # 0..1
            cp = max(0.5, min(0.95, float(close_percent)))
            if breakout_type == 'bullish':
                return pos >= cp
            return pos <= (1.0 - cp)
        except Exception:
            return False

    # ----- ATR Calculation -----
    def _compute_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int) -> Optional[pd.Series]:
        """Compute ATR using True Range with SMA."""
        try:
            if len(closes) < period + 2:
                return None
            prev_close = closes.shift(1)
            tr1 = (highs - lows).abs()
            tr2 = (highs - prev_close).abs()
            tr3 = (lows - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=period, min_periods=period).mean()
        except Exception:
            return None

    def _infer_bar_delta(self, index: pd.Index) -> Optional[pd.Timedelta]:
        """Infer typical bar duration from index deltas."""
        try:
            if index is None or len(index) < 3:
                return None
            diffs = index.to_series().diff().dropna()
            if diffs.empty:
                return None
            return diffs.median()
        except Exception:
            return None

    def _compute_atr_ratio(self, highs: pd.Series, lows: pd.Series, closes: pd.Series,
                           period: int, ratio_period: int) -> Optional[float]:
        """Compute ATR ratio versus rolling median ATR (dimensionless)."""
        try:
            atr_series = self._compute_atr(highs, lows, closes, period)
            if atr_series is None or atr_series.empty:
                return None
            if len(atr_series) < max(period + 1, ratio_period):
                return None
            atr_med = atr_series.rolling(window=ratio_period, min_periods=ratio_period).median()
            if atr_med is None or atr_med.empty:
                return None
            atr_last = float(atr_series.iloc[-1])
            med_last = float(atr_med.iloc[-1])
            if med_last <= 0:
                return None
            return atr_last / med_last
        except Exception:
            return None

    # ----- EMA Calculation for Trend Filter -----
    def _compute_ema(self, closes: pd.Series, period: int) -> Optional[pd.Series]:
        """Compute Exponential Moving Average for trend filter."""
        try:
            if len(closes) < period:
                return None
            return closes.ewm(span=period, adjust=False).mean()
        except Exception:
            return None

    def _check_trend_alignment(self, data: pd.DataFrame, breakout_type: str,
                                trend_data: Optional[pd.DataFrame] = None,
                                pip: float = 0.0001,
                                trend_label: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if breakout aligns with EMA trend on higher timeframe.
        Returns (is_aligned, trend_direction).

        Rules:
        - BUY signals: price must be above EMA (uptrend)
        - SELL signals: price must be below EMA (downtrend)
        - EMA slope must be above minimum threshold (not ranging)

        Uses higher timeframe data (trend_data) if provided for more stable trend anchor.
        Falls back to trading timeframe data if higher timeframe not available.
        """
        if not self.use_trend_filter:
            return True, "DISABLED"

        # Use higher timeframe data for trend if available, otherwise fall back to trading TF
        has_trend_data = trend_data is not None and len(trend_data) > 0
        trend_df = trend_data if has_trend_data else data

        ema = self._compute_ema(trend_df['close'], self.trend_ema_period)
        if ema is None or len(ema) < 1:
            return False, "NO_DATA"

        last_close = float(trend_df.iloc[-1]['close'])
        last_ema = float(ema.iloc[-1])

        # Determine trend direction suffix for logging
        tf_label = trend_label if has_trend_data and trend_label else "TF"

        # EMA slope filter - reject trades when EMA is flat (ranging market)
        if self.use_ema_slope_filter and len(ema) >= (self.ema_slope_period + 1):
            ema_now = float(ema.iloc[-1])
            ema_past = float(ema.iloc[-(self.ema_slope_period + 1)])
            slope_per_bar = (ema_now - ema_past) / float(self.ema_slope_period)

            # Prefer ATR-normalized slope (dimensionless)
            atr_series = self._compute_atr(trend_df['high'], trend_df['low'], trend_df['close'], self.atr_period)
            atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else None

            if atr_last and atr_last > 0 and self.min_ema_slope_atr_per_bar is not None:
                slope_atr_per_bar = slope_per_bar / atr_last
                if abs(slope_atr_per_bar) < float(self.min_ema_slope_atr_per_bar):
                    return False, f"FLAT_{tf_label}"
            else:
                # Fallback to pip-based slope if ATR not available
                slope_pips_per_bar = slope_per_bar / pip
                if abs(slope_pips_per_bar) < self.min_ema_slope_pips_per_bar:
                    return False, f"FLAT_{tf_label}"

        if last_close > last_ema:
            trend = f"BULLISH_{tf_label}"
            aligned = (breakout_type == 'bullish')
        else:
            trend = f"BEARISH_{tf_label}"
            aligned = (breakout_type == 'bearish')

        return aligned, trend

    # ----- Stop Loss Calculation -----
    def _calculate_stop_loss(self, breakout: BreakoutInfo, pip: float, symbol: str,
                             atr_last: Optional[float], support: List[float],
                             resistance: List[float],
                             entry_price: Optional[float] = None) -> Optional[float]:
        """
        Place SL beyond nearest opposing S/R level.
        For bullish: SL below nearest support
        For bearish: SL above nearest resistance

        Returns None if no valid structure exists (no structure = no trade).
        """
        # Get per-symbol overrides
        buf_pips = float(self.stop_loss_buffer_pips)
        min_sl_pips = float(self.min_stop_loss_pips)
        atr_k = float(self.atr_sl_k)
        min_buf_pips = float(self.min_sl_buffer_pips)

        for sc in getattr(self.config, 'symbols', []) or []:
            if sc.get('name') == symbol:
                buf_pips = float(sc.get('stop_loss_buffer_pips', buf_pips))
                min_sl_pips = float(sc.get('min_stop_loss_pips', min_sl_pips))
                break

        # Dynamic buffer: max of config buffer, min buffer, ATR*K
        buf_price = buf_pips * pip
        min_buf_price = min_buf_pips * pip
        atr_price = float(atr_last) if atr_last else 0.0
        dyn_extra = max(buf_price, min_buf_price, atr_k * atr_price)

        min_sl = min_sl_pips * pip
        entry = breakout.entry_price if entry_price is None else float(entry_price)
        level = breakout.level

        if breakout.type == 'bullish':
            # Find nearest support below broken resistance
            supports_below = [s for s in support if s < level]
            if not supports_below:
                logger.debug(f"No support structure below {level:.5f} - rejecting trade")
                return None  # NO STRUCTURE = NO TRADE
            nearest_support = max(supports_below)
            sl_struct = nearest_support - dyn_extra
            sl_min = entry - min_sl
            return min(sl_struct, sl_min)
        else:  # bearish
            # Find nearest resistance above broken support
            resistances_above = [r for r in resistance if r > level]
            if not resistances_above:
                logger.debug(f"No resistance structure above {level:.5f} - rejecting trade")
                return None  # NO STRUCTURE = NO TRADE
            nearest_resistance = min(resistances_above)
            sl_struct = nearest_resistance + dyn_extra
            sl_min = entry + min_sl
            return max(sl_struct, sl_min)

    def _calculate_structure_take_profit(self, breakout: BreakoutInfo, support: List[float],
                                         resistance: List[float], pip: float,
                                         tp_buffer_pips: float) -> Optional[float]:
        """Calculate TP at the next structure level in the trade direction."""
        buffer_price = float(tp_buffer_pips) * pip
        if breakout.type == 'bullish':
            targets = [r for r in resistance if r > breakout.level]
            if not targets:
                return None
            target = min(targets)
            return target - buffer_price
        else:
            targets = [s for s in support if s < breakout.level]
            if not targets:
                return None
            target = max(targets)
            return target + buffer_price

    # ----- Main Signal Generation -----
    def generate_signal(self, data: pd.DataFrame, symbol: str,
                        trend_data: Optional[pd.DataFrame] = None,
                        structure_data: Optional[pd.DataFrame] = None,
                        trend_timeframe: Optional[str] = None) -> Optional[TradingSignal]:
        """
        Generate trading signal based on breakout of S/R levels.

        Args:
            data: Trading timeframe candles (e.g., M15)
            symbol: Symbol name
            trend_data: Higher timeframe candles for trend filter (e.g., H4)
            structure_data: Higher timeframe candles for S/R structure (e.g., H1)
            trend_timeframe: Label for trend timeframe (used in logs)

        Returns TradingSignal if valid breakout detected, None otherwise.
        """
        try:
            if data is None or len(data) < max(20, self.lookback_period):
                return None

            tick = self._get_tick(symbol)
            info = self._get_symbol_info(symbol)
            if not tick or not info:
                return None

            pip = resolve_pip_size(symbol, info, self.config)
            if pip <= 0:
                return None

            # Spread guard
            current_spread_pips = abs(float(tick.ask) - float(tick.bid)) / pip
            spread_guard = self.spread_guard_pips_default
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    spread_guard = sc.get('spread_guard_pips', spread_guard)
                    break

            if spread_guard and current_spread_pips > float(spread_guard):
                logger.debug(f"{symbol}: Spread {current_spread_pips:.1f}p > guard {spread_guard}p - skipping")
                self._record_reject(symbol, "REJECT_SPREAD_GUARD")
                return None

            # Defaults (can be overridden per-symbol)
            thr_pips = float(self.breakout_threshold_pips)
            thr_atr_mult = self.breakout_threshold_atr_mult
            thr_spread_mult = self.breakout_threshold_spread_mult
            rr = float(self.risk_reward_ratio)
            min_rr = float(self.min_rr)
            max_sl_pips = self.max_sl_pips
            breakout_window_bars = int(self.breakout_window_bars)
            max_ext_pips = self.max_extension_pips
            max_ext_atr = self.max_extension_atr_mult
            sr_lookback = int(self.sr_lookback_period)
            sr_proximity_pips = float(self.sr_proximity_pips)
            tp_buffer_pips = float(self.tp_buffer_pips)
            structure_min_touches = int(self.structure_min_touches)
            structure_atr_band_mult = float(self.structure_atr_band_mult)
            ext_ratio_period = int(self.extension_atr_ratio_period)
            ext_min_mult = float(self.extension_atr_min_mult)
            ext_max_mult = float(self.extension_atr_max_mult)
            ext_sensitivity = float(self.extension_atr_sensitivity)
            require_structure_conf = bool(self.require_structure_confirmation)
            require_two_bar_conf = bool(self.require_two_bar_confirmation)
            require_fresh_breakout = bool(self.require_fresh_breakout)
            use_momentum_filter = bool(self.use_momentum_filter)
            momentum_atr_mult = float(self.momentum_atr_mult)
            momentum_close_percent = float(self.momentum_close_percent)
            entry_cooldown_bars = int(self.entry_cooldown_bars)
            entry_window_bars = int(self.entry_window_bars)
            tp_mode = str(self.tp_mode).lower()
            tp_r_multiple = float(self.tp_r_multiple)
            tp_use_structure_cap = bool(self.tp_use_structure_cap)
            max_sl_atr_mult = self.max_sl_atr_mult

            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    thr_pips = float(sc.get('breakout_threshold_pips', thr_pips))
                    thr_atr_mult = sc.get('breakout_threshold_atr_mult', thr_atr_mult)
                    thr_spread_mult = sc.get('breakout_threshold_spread_mult', thr_spread_mult)
                    rr = float(sc.get('risk_reward_ratio', rr))
                    max_sl_pips = sc.get('max_sl_pips', max_sl_pips)
                    breakout_window_bars = int(sc.get('breakout_window_bars', breakout_window_bars) or breakout_window_bars)
                    max_ext_pips = sc.get('max_extension_pips', max_ext_pips)
                    max_ext_atr = sc.get('max_extension_atr_mult', max_ext_atr)
                    sr_lookback = int(sc.get('sr_lookback_period', sr_lookback) or sr_lookback)
                    sr_proximity_pips = float(sc.get('sr_proximity_pips', sr_proximity_pips) or sr_proximity_pips)
                    tp_buffer_pips = float(sc.get('tp_buffer_pips', tp_buffer_pips) or tp_buffer_pips)
                    structure_min_touches = int(sc.get('structure_min_touches', structure_min_touches) or structure_min_touches)
                    structure_atr_band_mult = float(sc.get('structure_atr_band_mult', structure_atr_band_mult) or structure_atr_band_mult)
                    ext_ratio_period = int(sc.get('extension_atr_ratio_period', ext_ratio_period) or ext_ratio_period)
                    ext_min_mult = float(sc.get('extension_atr_min_mult', ext_min_mult) or ext_min_mult)
                    ext_max_mult = float(sc.get('extension_atr_max_mult', ext_max_mult) or ext_max_mult)
                    ext_sensitivity = float(sc.get('extension_atr_sensitivity', ext_sensitivity) or ext_sensitivity)
                    require_structure_conf = bool(sc.get('require_structure_confirmation', require_structure_conf))
                    require_two_bar_conf = bool(sc.get('require_two_bar_confirmation', require_two_bar_conf))
                    require_fresh_breakout = bool(sc.get('require_fresh_breakout', require_fresh_breakout))
                    use_momentum_filter = bool(sc.get('use_momentum_filter', use_momentum_filter))
                    momentum_atr_mult = float(sc.get('momentum_atr_mult', momentum_atr_mult) or momentum_atr_mult)
                    momentum_close_percent = float(sc.get('momentum_close_percent', momentum_close_percent) or momentum_close_percent)
                    entry_cooldown_bars = int(sc.get('entry_cooldown_bars', entry_cooldown_bars) or entry_cooldown_bars)
                    entry_window_bars = int(sc.get('entry_window_bars', entry_window_bars) or entry_window_bars)
                    tp_mode = str(sc.get('tp_mode', tp_mode)).lower()
                    tp_r_multiple = float(sc.get('tp_r_multiple', tp_r_multiple) or tp_r_multiple)
                    tp_use_structure_cap = bool(sc.get('tp_use_structure_cap', tp_use_structure_cap))
                    max_sl_atr_mult = sc.get('max_sl_atr_mult', max_sl_atr_mult)
                    break

            # If structure confirmation is enabled, drop the extra two-bar confirmation
            if require_structure_conf:
                require_two_bar_conf = False

            # Use completed candles only (exclude current forming bar)
            completed = data.iloc[:-1].tail(self.lookback_period)
            if len(completed) < max(20, self.lookback_period):
                return None
            bar_time = completed.index[-1]
            entry_delta = self._infer_bar_delta(completed.index)

            if self.use_session_filter and not self._in_session(bar_time):
                self._record_reject(symbol, "REJECT_SESSION")
                return None

            # Longer lookback for structure (S/R) using structure timeframe data
            if structure_data is None or len(structure_data) == 0:
                logger.debug(f"{symbol}: No structure data available")
                self._record_reject(symbol, "REJECT_NO_STRUCTURE_DATA")
                return None

            structure_completed = structure_data.iloc[:-1]
            sr_df = structure_completed.tail(sr_lookback)
            if len(sr_df) < self.swing_window * 2 + 1:
                return None

            # Find swing points and calculate S/R on structure window
            highs, lows = self.find_swing_points(sr_df)
            if not highs and not lows:
                self._record_reject(symbol, "REJECT_NO_SWINGS")
                return None

            struct_atr_series = self._compute_atr(
                structure_completed['high'],
                structure_completed['low'],
                structure_completed['close'],
                self.atr_period,
            )
            struct_atr_last = float(struct_atr_series.iloc[-1]) if struct_atr_series is not None and not pd.isna(struct_atr_series.iloc[-1]) else None

            resistance, support = self.calculate_support_resistance(
                sr_df,
                highs,
                lows,
                symbol,
                pip,
                proximity_pips=sr_proximity_pips,
                atr_last=struct_atr_last,
                min_touches=structure_min_touches,
                atr_band_mult=structure_atr_band_mult,
            )

            last_close = float(completed.iloc[-1]['close'])
            prev_close = float(completed.iloc[-2]['close']) if len(completed) > 1 else None

            # Compute ATR early (needed for both breakout threshold and SL buffer)
            atr_series = self._compute_atr(completed['high'], completed['low'],
                                           completed['close'], self.atr_period)
            atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else None

            # Detect breakout - use max of fixed, ATR, and spread-anchored thresholds
            threshold_candidates: List[float] = []
            if thr_pips is not None and thr_pips > 0:
                threshold_candidates.append(float(thr_pips) * pip)
            if thr_atr_mult is not None and atr_last is not None and atr_last > 0:
                threshold_candidates.append(float(thr_atr_mult) * atr_last)
            if thr_spread_mult is not None and current_spread_pips > 0:
                threshold_candidates.append(float(thr_spread_mult) * current_spread_pips * pip)

            threshold = max(threshold_candidates) if threshold_candidates else 0.0
            if threshold <= 0:
                logger.debug(f"{symbol}: No valid threshold (ATR/spread unavailable)")
                self._record_reject(symbol, "REJECT_NO_THRESHOLD")
                return None

            atr_pips = (atr_last / pip) if atr_last else 0.0

            breakout = self._detect_breakout(last_close, resistance, support, threshold)
            if not breakout:
                # Only log at debug level when no breakout (most common case)
                dist_to_r = min((r - last_close)/pip for r in resistance) if resistance else float('inf')
                dist_to_s = min((last_close - s)/pip for s in support) if support else float('inf')
                logger.debug(f"{symbol}: price={last_close:.5f} | R dist={dist_to_r:.1f}p, S dist={dist_to_s:.1f}p, need={threshold/pip:.1f}p")
                return None

            # Fresh breakout check (avoid late entries)
            if require_fresh_breakout and prev_close is not None:
                if breakout.type == 'bullish' and prev_close > breakout.level + threshold:
                    self._record_reject(symbol, "REJECT_BREAKOUT_OLD")
                    return None
                if breakout.type == 'bearish' and prev_close < breakout.level - threshold:
                    self._record_reject(symbol, "REJECT_BREAKOUT_OLD")
                    return None

            # BREAKOUT DETECTED - now log detailed analysis
            logger.debug(
                f"{symbol}: === BREAKOUT {breakout.type.upper()} === level={breakout.level:.5f} price={last_close:.5f} "
                f"spread={current_spread_pips:.1f}p ATR={atr_pips:.1f}p thr={threshold/pip:.1f}p"
            )
            logger.debug(f"{symbol}: S/R levels R={[round(r,5) for r in resistance[:3]]} S={[round(s,5) for s in support[:3]]}")

            # Momentum filter: require strong impulse on breakout bar
            if use_momentum_filter:
                if not self._passes_momentum_filter(
                    breakout.type,
                    completed.iloc[-1],
                    atr_last,
                    pip,
                    momentum_atr_mult,
                    momentum_close_percent,
                ):
                    self._record_reject(symbol, "REJECT_MOMENTUM")
                    return None

            # Structure confirmation: last completed structure bar must close beyond level by threshold
            struct_close = None
            struct_threshold = None
            if require_structure_conf:
                if structure_completed is None or len(structure_completed) < 1:
                    logger.debug(f"{symbol}: No completed structure data for confirmation")
                    self._record_reject(symbol, "REJECT_NO_STRUCT_DATA")
                    return None
                struct_threshold_candidates: List[float] = []
                if thr_pips is not None and thr_pips > 0:
                    struct_threshold_candidates.append(float(thr_pips) * pip)
                if thr_atr_mult is not None and struct_atr_last is not None and struct_atr_last > 0:
                    struct_threshold_candidates.append(float(thr_atr_mult) * struct_atr_last)
                if thr_spread_mult is not None and current_spread_pips > 0:
                    struct_threshold_candidates.append(float(thr_spread_mult) * current_spread_pips * pip)

                struct_threshold = max(struct_threshold_candidates) if struct_threshold_candidates else 0.0
                if struct_threshold <= 0:
                    logger.debug(f"{symbol}: Invalid structure threshold (structure ATR unavailable)")
                    self._record_reject(symbol, "REJECT_STRUCT_THRESHOLD")
                    return None

                struct_close = float(structure_completed.iloc[-1]['close'])
                if breakout.type == 'bullish':
                    if struct_close <= breakout.level + struct_threshold:
                        logger.debug(f"{symbol}: H1 close {struct_close:.5f} not beyond {breakout.level + struct_threshold:.5f} - awaiting confirmation")
                        self._record_reject(symbol, "REJECT_STRUCT_CONFIRM")
                        return None
                else:
                    if struct_close >= breakout.level - struct_threshold:
                        logger.debug(f"{symbol}: H1 close {struct_close:.5f} not beyond {breakout.level - struct_threshold:.5f} - awaiting confirmation")
                        self._record_reject(symbol, "REJECT_STRUCT_CONFIRM")
                        return None
                logger.debug(f"{symbol}: Structure CONFIRMED (H1 close={struct_close:.5f})")

                # Freshness aligned to structure confirmation: allow entry within N entry bars after close
                if entry_window_bars and entry_window_bars > 0 and entry_delta is not None:
                    struct_delta = self._infer_bar_delta(structure_completed.index)
                    if struct_delta is not None:
                        struct_close_time = structure_completed.index[-1] + struct_delta
                        last_entry_time = completed.index[-1] + entry_delta
                        window_end = struct_close_time + (entry_delta * entry_window_bars)
                        if not (struct_close_time <= last_entry_time <= window_end):
                            logger.debug(f"{symbol}: Outside entry window (struct_close={struct_close_time}, entry={last_entry_time}, window_end={window_end})")
                            self._record_reject(symbol, "REJECT_ENTRY_WINDOW")
                            return None
                        logger.debug(f"{symbol}: Entry window OK")

            # Two-bar confirmation on entry timeframe
            if require_two_bar_conf:
                if len(completed) < 2:
                    return None
                prev_close = float(completed.iloc[-2]['close'])
                if breakout.type == 'bullish':
                    if not (last_close > breakout.level + threshold and prev_close > breakout.level + threshold):
                        logger.debug(f"{symbol}: Two-bar confirmation failed for bullish breakout")
                        self._record_reject(symbol, "REJECT_TWO_BAR")
                        return None
                else:
                    if not (last_close < breakout.level - threshold and prev_close < breakout.level - threshold):
                        logger.debug(f"{symbol}: Two-bar confirmation failed for bearish breakout")
                        self._record_reject(symbol, "REJECT_TWO_BAR")
                        return None

            # Breakout window: only allow recent breakouts when structure confirmation is disabled
            if not require_structure_conf and not require_fresh_breakout:
                breakout_age = self._find_breakout_age(
                    completed['close'],
                    breakout.level,
                    threshold,
                    breakout.type,
                    breakout_window_bars,
                )
                if breakout_age is None:
                    logger.debug(f"{symbol}: Breakout too old (window={breakout_window_bars} bars)")
                    self._record_reject(symbol, "REJECT_BREAKOUT_OLD")
                    return None

            # Trend filter: only take trades aligned with EMA on higher timeframe
            trend_completed = None
            if trend_data is not None and len(trend_data) > 1:
                # Use completed higher-timeframe bars only
                trend_completed = trend_data.iloc[:-1]

            trend_aligned, trend_dir = self._check_trend_alignment(
                completed,
                breakout.type,
                trend_completed,
                pip,
                trend_timeframe,
            )
            if not trend_aligned:
                logger.debug(f"{symbol}: Trend filter REJECT - {breakout.type} vs trend={trend_dir}")
                self._record_reject(symbol, "REJECT_TREND")
                return None
            logger.debug(f"{symbol}: Trend OK ({trend_dir})")

            # Use actual execution price
            entry_eff = float(tick.ask) if breakout.type == 'bullish' else float(tick.bid)

            # Anti-chase: reject if price already too far from breakout level (or confirmation close)
            ext_limit = None
            if max_ext_pips is not None:
                try:
                    ext_limit = float(max_ext_pips)
                except Exception:
                    ext_limit = None
            if max_ext_atr is not None:
                try:
                    atr_for_extension = atr_last
                    ratio_source = completed
                    if require_structure_conf and struct_atr_last is not None and struct_atr_last > 0:
                        atr_for_extension = struct_atr_last
                        ratio_source = structure_completed
                    if atr_for_extension is not None and atr_for_extension > 0:
                        # Market-adaptive multiplier based on ATR ratio
                        ratio = self._compute_atr_ratio(
                            ratio_source['high'],
                            ratio_source['low'],
                            ratio_source['close'],
                            self.atr_period,
                            ext_ratio_period,
                        )
                        mult = float(max_ext_atr)
                        if ratio is not None:
                            try:
                                raw = float(max_ext_atr) + float(ext_sensitivity) * (float(ratio) - 1.0)
                                mult = max(ext_min_mult, min(ext_max_mult, raw))
                            except Exception:
                                pass
                        atr_limit = float(mult) * (atr_for_extension / pip)
                        ext_limit = atr_limit if ext_limit is None else min(ext_limit, atr_limit)
                except Exception:
                    pass
            if ext_limit is not None and ext_limit > 0:
                # When using structure confirmation, anchor extension to the confirming close
                # to avoid double-penalizing late entries caused by higher-TF confirmation.
                ext_ref = struct_close if (require_structure_conf and struct_close is not None) else breakout.level
                ext_pips = abs(entry_eff - ext_ref) / pip
                if ext_pips > ext_limit:
                    logger.debug(f"{symbol}: Extension REJECT {ext_pips:.1f}p > limit {ext_limit:.1f}p")
                    self._record_reject(symbol, "REJECT_EXTENSION")
                    return None
                logger.debug(f"{symbol}: Extension OK ({ext_pips:.1f}p <= {ext_limit:.1f}p)")

            # Calculate SL (ATR already computed above)
            sl = self._calculate_stop_loss(
                breakout,
                pip,
                symbol,
                atr_last,
                support,
                resistance,
                entry_price=entry_eff,
            )
            if sl is None:
                logger.debug(f"{symbol}: No opposing structure for SL - REJECT")
                self._record_reject(symbol, "REJECT_NO_SL")
                return None

            sl_pips = abs(entry_eff - sl) / pip
            if sl_pips <= 0:
                return None

            # Max SL check (use actual entry and raw SL)
            if max_sl_pips and sl_pips > float(max_sl_pips):
                logger.debug(f"{symbol}: SL {sl_pips:.1f}p > max {max_sl_pips}p - REJECT")
                self._record_reject(symbol, "REJECT_MAX_SL")
                return None
            if max_sl_atr_mult is not None and atr_last is not None and atr_last > 0:
                max_sl_atr_pips = float(max_sl_atr_mult) * (atr_last / pip)
                if sl_pips > max_sl_atr_pips:
                    logger.debug(f"{symbol}: SL {sl_pips:.1f}p > ATR cap {max_sl_atr_pips:.1f}p - REJECT")
                    self._record_reject(symbol, "REJECT_MAX_SL_ATR")
                    return None

            # Calculate TP (mode: r_multiple or structure)
            tp = None
            tp_from_structure = False
            if tp_mode == "r_multiple":
                target_pips = sl_pips * tp_r_multiple
                if breakout.type == 'bullish':
                    tp = entry_eff + (target_pips * pip)
                else:
                    tp = entry_eff - (target_pips * pip)
                if tp_use_structure_cap:
                    struct_tp = self._calculate_structure_take_profit(breakout, support, resistance, pip, tp_buffer_pips)
                    if struct_tp is not None:
                        if breakout.type == 'bullish' and struct_tp < tp:
                            tp = struct_tp
                            tp_from_structure = True
                        elif breakout.type == 'bearish' and struct_tp > tp:
                            tp = struct_tp
                            tp_from_structure = True
            else:
                tp = self._calculate_structure_take_profit(breakout, support, resistance, pip, tp_buffer_pips)
                tp_from_structure = True
                if tp is None:
                    logger.debug(f"{symbol}: No target structure for TP - REJECT")
                    self._record_reject(symbol, "REJECT_NO_TP")
                    return None

            logger.debug(f"{symbol}: SL={sl:.5f} TP={tp:.5f} entry={entry_eff:.5f}")

            # Round to broker precision
            point = getattr(info, 'point', None)
            digits = getattr(info, 'digits', None)
            if point and digits is not None:
                sl = round(round(sl / point) * point, int(digits))
                tp = round(round(tp / point) * point, int(digits))

            # Ensure TP is on correct side of entry after rounding
            if breakout.type == 'bullish':
                if tp <= entry_eff:
                    logger.debug(f"{symbol}: TP {tp:.5f} not above entry {entry_eff:.5f}")
                    self._record_reject(symbol, "REJECT_TP_SIDE")
                    return None
            else:
                if tp >= entry_eff:
                    logger.debug(f"{symbol}: TP {tp:.5f} not below entry {entry_eff:.5f}")
                    self._record_reject(symbol, "REJECT_TP_SIDE")
                    return None

            # Recalculate pips after rounding
            sl_pips = abs(entry_eff - sl) / pip
            tp_pips = abs(tp - entry_eff) / pip
            if sl_pips <= 0 or tp_pips <= 0:
                return None

            required_rr = max(min_rr, rr)

            # Minimum TP distance: must clear the market-driven breakout threshold
            try:
                min_tp_pips = float(threshold) / pip if threshold and pip > 0 else 0.0
            except Exception:
                min_tp_pips = 0.0
            if min_tp_pips > 0 and tp_pips < min_tp_pips:
                logger.debug(f"{symbol}: TP {tp_pips:.1f}p < min {min_tp_pips:.1f}p - REJECT")
                self._record_reject(symbol, "REJECT_MIN_TP")
                return None

            actual_rr = tp_pips / sl_pips
            # Minimum entry-based RR check (market-driven via existing required_rr)
            if actual_rr < required_rr:
                logger.debug(f"{symbol}: RR_entry {actual_rr:.2f} < min {required_rr:.2f} - REJECT")
                self._record_reject(symbol, "REJECT_RR_ENTRY")
                return None
            # Structure-based RR check only when TP is structure-derived
            structure_rr = None
            if tp_from_structure:
                if breakout.type == 'bullish':
                    struct_risk = (breakout.level - sl) / pip
                    struct_reward = (tp - breakout.level) / pip
                else:
                    struct_risk = (sl - breakout.level) / pip
                    struct_reward = (breakout.level - tp) / pip
                if struct_risk <= 0 or struct_reward <= 0:
                    logger.debug(f"{symbol}: Invalid struct risk/reward - REJECT")
                    self._record_reject(symbol, "REJECT_RR_STRUCT")
                    return None
                structure_rr = struct_reward / struct_risk
                if structure_rr < required_rr:
                    logger.debug(f"{symbol}: RR_struct {structure_rr:.2f} < min {required_rr:.2f} - REJECT")
                    self._record_reject(symbol, "REJECT_RR_STRUCT")
                    return None

            logger.debug(f"{symbol}: RR OK (entry={actual_rr:.2f}, struct={structure_rr if structure_rr is not None else 'n/a'}, min={required_rr:.2f})")

            # Cooldown: avoid rapid re-entry on the same symbol
            if entry_cooldown_bars and entry_cooldown_bars > 0 and entry_delta is not None:
                last_time = self._last_signal_time.get(symbol)
                if last_time is not None:
                    if bar_time < last_time + (entry_delta * entry_cooldown_bars):
                        self._record_reject(symbol, "REJECT_COOLDOWN")
                        return None

            # Duplicate prevention: one signal per bar per direction
            try:
                key = (symbol, breakout.type)
                if self._last_breakout_bar.get(key) == bar_time:
                    return None
                self._last_breakout_bar[key] = bar_time
            except Exception:
                pass

            signal = TradingSignal(
                type=0 if breakout.type == 'bullish' else 1,
                entry_price=entry_eff,
                stop_loss=sl,
                take_profit=tp,
                stop_loss_pips=sl_pips,
                reason=f"{breakout.type}_breakout",
                timestamp=datetime.now(timezone.utc),
                breakout_level=breakout.level,
            )
            try:
                self._last_signal_time[symbol] = bar_time
            except Exception:
                pass

            rr_struct_str = f"{structure_rr:.2f}" if structure_rr is not None else "n/a"
            logger.info(
                f"SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'} @ {entry_eff:.5f} "
                f"SL {sl:.5f} TP {tp:.5f} ({sl_pips:.1f}p, RR_struct={rr_struct_str}, RR_entry={actual_rr:.2f}) [Trend: {trend_dir}]"
            )
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
