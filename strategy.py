"""
Core Price Action Breakout Strategy
"""

from dataclasses import dataclass
from datetime import datetime, timezone
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
        self.breakout_threshold_pips = getattr(config, 'breakout_threshold', 7)
        self.breakout_window_bars = int(getattr(config, 'breakout_window_bars', 1) or 1)
        self.min_stop_loss_pips = getattr(config, 'min_stop_loss_pips', 20)
        self.stop_loss_buffer_pips = getattr(config, 'stop_loss_buffer_pips', 15)
        self.risk_reward_ratio = getattr(config, 'risk_reward_ratio', 2.0)
        self.min_rr = getattr(config, 'min_rr', 1.0)
        self.max_extension_pips = getattr(config, 'max_extension_pips', None)
        self.max_extension_atr_mult = getattr(config, 'max_extension_atr_mult', None)
        self.sr_lookback_period = int(getattr(config, 'sr_lookback_period', 80))
        self.sr_proximity_pips = float(getattr(config, 'sr_proximity_pips', 10.0))

        # ATR for dynamic SL buffer and breakout threshold
        self.atr_period = int(getattr(config, 'atr_period', 14))
        self.atr_sl_k = float(getattr(config, 'atr_sl_k', 0.6))
        self.min_sl_buffer_pips = float(getattr(config, 'min_sl_buffer_pips', 10))
        self.max_sl_pips = getattr(config, 'max_sl_pips', None)

        # ATR-based breakout threshold (if set, overrides fixed pips)
        self.breakout_threshold_atr_mult = getattr(config, 'breakout_threshold_atr_mult', None)

        # Trend filter (200 EMA alignment)
        self.use_trend_filter = getattr(config, 'use_trend_filter', True)
        self.trend_ema_period = int(getattr(config, 'trend_ema_period', 200))

        # EMA slope filter (reject trades when EMA is flat/ranging)
        self.use_ema_slope_filter = getattr(config, 'use_ema_slope_filter', True)
        self.ema_slope_period = int(getattr(config, 'ema_slope_period', 20))
        self.min_ema_slope_pips_per_bar = float(getattr(config, 'min_ema_slope_pips_per_bar', 0.1))

        # Spread guard
        self.spread_guard_pips_default = getattr(config, 'spread_guard_pips_default', None)

        # Duplicate signal prevention (one per bar per direction)
        self._last_breakout_bar = {}

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
                                     proximity_pips: float = 10.0) -> Tuple[List[float], List[float]]:
        """Extract recent distinct swing highs/lows as resistance/support."""
        proximity = float(proximity_pips) * pip  # proximity filter in pips

        res: List[float] = []
        sup: List[float] = []
        max_levels = 5

        # Take recent swings
        for i in reversed(swing_highs[-50:]):
            level = float(data.iloc[i]['high'])
            if not any(abs(level - x) <= proximity for x in res):
                res.append(level)
            if len(res) >= max_levels:
                break

        for i in reversed(swing_lows[-50:]):
            level = float(data.iloc[i]['low'])
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
                                pip: float = 0.0001) -> Tuple[bool, str]:
        """
        Check if breakout aligns with 200 EMA trend on higher timeframe.
        Returns (is_aligned, trend_direction).

        Rules:
        - BUY signals: price must be above 200 EMA (uptrend)
        - SELL signals: price must be below 200 EMA (downtrend)
        - EMA slope must be above minimum threshold (not ranging)

        Uses higher timeframe data (trend_data) if provided for more stable trend anchor.
        Falls back to trading timeframe data if higher timeframe not available.
        """
        if not self.use_trend_filter:
            return True, "DISABLED"

        # Use H1 data for trend if available, otherwise fall back to trading TF
        trend_df = trend_data if trend_data is not None and len(trend_data) > 0 else data

        ema = self._compute_ema(trend_df['close'], self.trend_ema_period)
        if ema is None or len(ema) < 1:
            return False, "NO_DATA"

        last_close = float(trend_df.iloc[-1]['close'])
        last_ema = float(ema.iloc[-1])

        # Determine trend direction suffix for logging
        tf_label = "H1" if trend_data is not None and len(trend_data) > 0 else "TF"

        # EMA slope filter - reject trades when EMA is flat (ranging market)
        if self.use_ema_slope_filter and len(ema) >= self.ema_slope_period:
            ema_now = float(ema.iloc[-1])
            ema_past = float(ema.iloc[-self.ema_slope_period])
            slope_pips_per_bar = (ema_now - ema_past) / (self.ema_slope_period * pip)

            if abs(slope_pips_per_bar) < self.min_ema_slope_pips_per_bar:
                logger.debug(f"EMA slope filter: slope={slope_pips_per_bar:.3f} pips/bar < min {self.min_ema_slope_pips_per_bar}")
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
                             resistance: List[float]) -> Optional[float]:
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
        entry = breakout.entry_price
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

    def _calculate_take_profit(self, entry: float, stop: float, rr: float, side: str) -> float:
        """Calculate TP based on risk-reward ratio."""
        dist = abs(entry - stop) * rr
        return entry + dist if side == 'bullish' else entry - dist

    # ----- Main Signal Generation -----
    def generate_signal(self, data: pd.DataFrame, symbol: str,
                        trend_data: Optional[pd.DataFrame] = None) -> Optional[TradingSignal]:
        """
        Generate trading signal based on breakout of S/R levels.

        Args:
            data: Trading timeframe candles (e.g., M15)
            symbol: Symbol name
            trend_data: Higher timeframe candles for trend filter (e.g., H1)

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
                logger.debug(f"{symbol}: Spread {current_spread_pips:.1f}p > guard {spread_guard}p")
                return None

            # Defaults (can be overridden per-symbol)
            thr_pips = float(self.breakout_threshold_pips)
            thr_atr_mult = self.breakout_threshold_atr_mult
            rr = float(self.risk_reward_ratio)
            min_rr = float(self.min_rr)
            max_sl_pips = self.max_sl_pips
            breakout_window_bars = int(self.breakout_window_bars)
            max_ext_pips = self.max_extension_pips
            max_ext_atr = self.max_extension_atr_mult
            sr_lookback = int(self.sr_lookback_period)
            sr_proximity_pips = float(self.sr_proximity_pips)

            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    thr_pips = float(sc.get('breakout_threshold_pips', thr_pips))
                    thr_atr_mult = sc.get('breakout_threshold_atr_mult', thr_atr_mult)
                    rr = float(sc.get('risk_reward_ratio', rr))
                    max_sl_pips = sc.get('max_sl_pips', max_sl_pips)
                    breakout_window_bars = int(sc.get('breakout_window_bars', breakout_window_bars) or breakout_window_bars)
                    max_ext_pips = sc.get('max_extension_pips', max_ext_pips)
                    max_ext_atr = sc.get('max_extension_atr_mult', max_ext_atr)
                    sr_lookback = int(sc.get('sr_lookback_period', sr_lookback) or sr_lookback)
                    sr_proximity_pips = float(sc.get('sr_proximity_pips', sr_proximity_pips) or sr_proximity_pips)
                    break

            # Use completed candles only (exclude current forming bar)
            completed = data.iloc[:-1].tail(self.lookback_period)
            if len(completed) < max(20, self.lookback_period):
                return None

            # Longer lookback for structure (S/R)
            sr_df = data.iloc[:-1].tail(sr_lookback)
            if len(sr_df) < self.swing_window * 2 + 1:
                return None

            # Find swing points and calculate S/R on structure window
            highs, lows = self.find_swing_points(sr_df)
            if not highs and not lows:
                return None

            resistance, support = self.calculate_support_resistance(
                sr_df, highs, lows, symbol, pip, proximity_pips=sr_proximity_pips
            )

            # Compute ATR early (needed for both breakout threshold and SL buffer)
            atr_series = self._compute_atr(completed['high'], completed['low'],
                                           completed['close'], self.atr_period)
            atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else None

            # Detect breakout - use ATR-based threshold if configured, else fixed pips
            last_close = float(completed.iloc[-1]['close'])
            if thr_atr_mult is not None and atr_last is not None and atr_last > 0:
                threshold = float(thr_atr_mult) * atr_last
                logger.debug(f"{symbol}: Using ATR-based threshold: {threshold/pip:.1f} pips (ATR={atr_last/pip:.1f}p * {thr_atr_mult})")
            else:
                threshold = thr_pips * pip

            breakout = self._detect_breakout(last_close, resistance, support, threshold)
            if not breakout:
                return None

            logger.debug(f"{symbol}: Breakout {breakout.type} @ level {breakout.level:.5f}")

            # Breakout window: only allow recent breakouts (current or recent bar)
            breakout_age = self._find_breakout_age(
                completed['close'],
                breakout.level,
                threshold,
                breakout.type,
                breakout_window_bars,
            )
            if breakout_age is None:
                logger.debug(f"{symbol}: Breakout too old (window={breakout_window_bars} bars)")
                return None

            # Trend filter: only take trades aligned with 200 EMA on higher timeframe
            trend_completed = None
            if trend_data is not None and len(trend_data) > 1:
                # Use completed higher-timeframe bars only
                trend_completed = trend_data.iloc[:-1]

            trend_aligned, trend_dir = self._check_trend_alignment(completed, breakout.type, trend_completed, pip)
            if not trend_aligned:
                logger.debug(f"{symbol}: Breakout {breakout.type} rejected - trend is {trend_dir}")
                return None

            # Duplicate prevention: one signal per bar per direction
            try:
                bar_time = completed.index[-1]
                key = (symbol, breakout.type)
                if self._last_breakout_bar.get(key) == bar_time:
                    return None
                self._last_breakout_bar[key] = bar_time
            except Exception:
                pass

            # Use actual execution price
            entry_eff = float(tick.ask) if breakout.type == 'bullish' else float(tick.bid)

            # Anti-chase: reject if price already too far from breakout level
            ext_limit = None
            if max_ext_pips is not None:
                try:
                    ext_limit = float(max_ext_pips)
                except Exception:
                    ext_limit = None
            if max_ext_atr is not None and atr_last is not None and atr_last > 0:
                try:
                    atr_limit = float(max_ext_atr) * (atr_last / pip)
                    ext_limit = atr_limit if ext_limit is None else min(ext_limit, atr_limit)
                except Exception:
                    pass
            if ext_limit is not None and ext_limit > 0:
                ext_pips = abs(entry_eff - breakout.level) / pip
                if ext_pips > ext_limit:
                    logger.info(f"{symbol}: Extension {ext_pips:.1f}p > limit {ext_limit:.1f}p")
                    return None

            # Calculate SL (ATR already computed above)
            sl = self._calculate_stop_loss(breakout, pip, symbol, atr_last, support, resistance)
            if sl is None:
                logger.info(f"{symbol}: No valid structure for SL - trade rejected")
                return None

            # Calculate TP
            tp = self._calculate_take_profit(entry_eff, sl, rr, breakout.type)

            # Round to broker precision
            point = getattr(info, 'point', None)
            digits = getattr(info, 'digits', None)
            if point and digits is not None:
                sl = round(round(sl / point) * point, int(digits))
                tp = round(round(tp / point) * point, int(digits))

            # Calculate actual RR
            sl_pips = abs(entry_eff - sl) / pip
            tp_pips = abs(tp - entry_eff) / pip

            if sl_pips <= 0:
                return None

            # Max SL check (use actual entry and rounded SL)
            if max_sl_pips and sl_pips > float(max_sl_pips):
                logger.info(f"{symbol}: SL {sl_pips:.1f}p > max {max_sl_pips}p")
                return None

            actual_rr = tp_pips / sl_pips

            # Minimum RR check
            if actual_rr < min_rr:
                logger.info(f"{symbol}: RR {actual_rr:.2f} < min {min_rr}")
                return None

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

            logger.info(
                f"SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'} @ {entry_eff:.5f} "
                f"SL {sl:.5f} TP {tp:.5f} ({sl_pips:.1f}p, RR={actual_rr:.2f}) [Trend: {trend_dir}]"
            )
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
