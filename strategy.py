"""
Core Price Action Strategy.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple, NamedTuple, TYPE_CHECKING, Dict, Any

import MetaTrader5 as mt5
import pandas as pd
import logging

from utils import get_pip_size, resolve_pip_size
from patterns import candlestick_confirm

if TYPE_CHECKING:
    from history_analysis import HistoricalSnapshot

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    type: int  # 0 = BUY, 1 = SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    stop_loss_pips: float
    reason: str
    confidence: float
    timestamp: datetime
    breakout_level: float
    pattern_score: Optional[float] = None
    pattern_dir: Optional[str] = None
    pattern_primary: Optional[str] = None
    pattern_timeframe: Optional[str] = None
    hist_trend_bias: Optional[str] = None
    hist_breakout_rate: Optional[float] = None
    hist_adr_progress: Optional[float] = None

class BreakoutInfo(NamedTuple):
    type: str      # 'bullish' or 'bearish'
    level: float
    entry_price: float

class PurePriceActionStrategy:
    """Minimal breakout strategy based on swing levels and fixed pip thresholds."""

    def __init__(self, config, mt5_client=None):
        self.config = config
        # Prefer injected MT5 client (with reconnect/selection) over raw module access
        self.mt5_client = mt5_client
        self.mt5 = getattr(mt5_client, "mt5", mt5)
        # Core params only
        self.lookback_period = getattr(config, 'lookback_period', 20)
        self.swing_window = getattr(config, 'swing_window', 5)
        self.breakout_threshold_pips = getattr(config, 'breakout_threshold', 7)
        self.min_stop_loss_pips = getattr(config, 'min_stop_loss_pips', 20)
        self.stop_loss_buffer_pips = getattr(config, 'stop_loss_buffer_pips', 15)
        self.risk_reward_ratio = getattr(config, 'risk_reward_ratio', 2.0)
        self.atr_period = int(getattr(config, 'atr_period', 14))
        self.atr_sl_k = float(getattr(config, 'atr_sl_k', 0.6))
        self.min_sl_buffer_pips = float(getattr(config, 'min_sl_buffer_pips', 10))
        self.max_sl_pips = getattr(config, 'max_sl_pips', None)
        self.min_headroom_rr = float(getattr(config, 'min_headroom_rr', 1.2))
        self.max_rr_cap = getattr(config, 'max_rr_cap', None)
        self.context_lookback_period = getattr(config, 'context_lookback_period', max(100, self.lookback_period * 3))
        self.obstacle_buffer_pips = float(getattr(config, 'obstacle_buffer_pips', 3))
        self.min_rr_after_adjustment = float(getattr(config, 'min_rr_after_adjustment', self.risk_reward_ratio))
        self.max_sr_levels = int(getattr(config, 'max_sr_levels', 5))
        # Patterns/ATR integration controls
        self.enable_patterns = bool(getattr(config, 'enable_patterns', True))
        self.pattern_window = int(getattr(config, 'pattern_window', 3))
        self.pattern_score_threshold = float(getattr(config, 'pattern_score_threshold', 0.6))
        self.pattern_strong_threshold = float(getattr(config, 'pattern_strong_threshold', 0.8))
        self.allowed_patterns = getattr(config, 'allowed_patterns', None)
        self.pattern_use_trading_tf = bool(getattr(config, 'pattern_use_trading_tf', False))
        self.htf_pattern_timeframe = getattr(config, 'htf_pattern_timeframe', None)
        self.htf_pattern_window = int(getattr(config, 'htf_pattern_window', self.pattern_window))
        self.htf_pattern_score_threshold = float(getattr(config, 'htf_pattern_score_threshold', self.pattern_score_threshold))
        self.require_htf_pattern_alignment = bool(getattr(config, 'require_htf_pattern_alignment', True))
        self.atr_source = getattr(config, 'atr_source', 'talib')
        self._last_breakout_bar = {}  # One signal per bar per direction
        self._last_block_reason = {} # One log per block reason per symbol
        # Breakout acceptance controls
        self.entry_mode = str(getattr(config, 'entry_mode', 'confirm')).lower()
        self.entry_confirmation_bars = int(getattr(config, 'entry_confirmation_bars', 1))

    # -----------------------------
    # MT5 helpers (fail-safe)
    # -----------------------------
    def _get_tick(self, symbol: str):
        if self.mt5_client:
            return self.mt5_client.get_symbol_info_tick(symbol)
        return mt5.symbol_info_tick(symbol)

    def _get_symbol_info(self, symbol: str):
        if self.mt5_client:
            return self.mt5_client.get_symbol_info(symbol)
        return mt5.symbol_info(symbol)

    # -----------------------------
    # Swing points and S/R levels
    # -----------------------------
    def _find_swing_indices(self, series: pd.Series, window: int) -> List[int]:
        """
        Find swing high indices where value is strictly greater than surrounding bars.

        Uses strict inequality (>) to avoid detecting multiple swing points at flat tops/bottoms.
        """
        idxs: List[int] = []
        values = series.values
        n = len(values)
        for i in range(window, n - window):
            left = values[i - window:i]  # Exclude current bar from left window
            right = values[i + 1:i + window + 1]  # Exclude current bar from right window

            # Strict inequality: must be strictly greater than all neighbors
            if len(left) > 0 and len(right) > 0:
                if values[i] > left.max() and values[i] > right.max():
                    idxs.append(i)
        return idxs

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        if data is None or len(data) < self.swing_window * 2 + 1:
            return [], []
        highs = self._find_swing_indices(data['high'], self.swing_window)
        lows = self._find_swing_indices(-data['low'], self.swing_window)  # reuse max finder on inverted lows
        return highs, lows

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int], symbol: str) -> Tuple[List[float], List[float]]:
        """
        Pick recent distinct swing highs/lows as resistance/support.

        Number of levels is configurable via max_sr_levels (default 5).
        Proximity threshold can be overridden per symbol.
        """
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            return [], []
        pip = get_pip_size(symbol_info)

        # Get proximity threshold (default 10 pips, but configurable per symbol)
        proximity_pips = 10.0
        try:
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    proximity_pips = float(sc.get('level_proximity_pips', proximity_pips))
                    break
        except Exception:
            pass
        proximity = proximity_pips * pip

        res: List[float] = []
        sup: List[float] = []

        max_levels = self.max_sr_levels

        # Take the last 50 swings to avoid ancient levels
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

       
    # -----------------------------
    # Breakout and SL/TP
    # -----------------------------
    def _compute_atr(self, highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int) -> Optional[pd.Series]:
        """Compute ATR via TA-Lib if configured, else fallback to manual."""
        try:
            if len(closes) < period + 2:
                return None
            if str(self.atr_source).lower() == 'talib':
                try:
                    import talib as ta  # type: ignore
                    arr = ta.ATR(highs.values, lows.values, closes.values, timeperiod=int(period))
                    if arr is not None:
                        series = pd.Series(arr, index=closes.index)
                        try:
                            logger.debug(f"ATR via TA-Lib (period={period}) computed")
                        except Exception:
                            pass
                        return series
                except Exception:
                    # fall back to manual below
                    pass
            prev_close = closes.shift(1)
            tr1 = (highs - lows).abs()
            tr2 = (highs - prev_close).abs()
            tr3 = (lows - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            try:
                logger.debug(f"ATR via manual TR SMA (period={period}) computed")
            except Exception:
                pass
            return atr
        except Exception:
            return None

    def _candlestick_confirm(
        self,
        completed: pd.DataFrame,
        bo: 'BreakoutInfo',
        symbol: str,
        window: Optional[int] = None,
        scope: str = "LTF",
    ) -> Tuple[bool, Optional[dict]]:
        """Confirm breakout or bias using TA-Lib candlestick patterns. Fail-open if disabled or TA-Lib missing."""
        if not self.enable_patterns:
            return True, None
        try:
            ok, info = candlestick_confirm(
                completed[['open', 'high', 'low', 'close']],
                direction=bo.type,
                window=int(window if window is not None else self.pattern_window),
                allowed_patterns=self.allowed_patterns,
            )
            try:
                logger.debug(
                    f"{symbol}[{scope}]: pattern_ok={ok} dir={info.get('dir') if info else None} "
                    f"score={info.get('score') if info else None}"
                )
            except Exception:
                pass
            if not ok:
                return False, info
            score = float((info or {}).get('score', 0.0) or 0.0)
            if score < float(self.pattern_score_threshold):
                return False, info
            return True, info
        except Exception:
            return True, None

    def _resolve_htf_pattern_prefs(self, symbol: str) -> Tuple[Optional[str], bool]:
        """Resolve preferred HTF timeframe and requirement flag for candlestick bias."""
        timeframe = self.htf_pattern_timeframe
        require = bool(self.require_htf_pattern_alignment)
        try:
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    timeframe = sc.get('htf_pattern_timeframe', timeframe)
                    if 'require_htf_pattern_alignment' in sc:
                        require = bool(sc.get('require_htf_pattern_alignment'))
                    break
        except Exception:
            pass
        return timeframe, require

    def _resolve_historical_params(self, symbol: str) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(getattr(self.config, 'historical_analysis', {}) or {})
        try:
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    sym_hist = sc.get('historical_analysis', {})
                    if sym_hist:
                        params.update({k: v for k, v in sym_hist.items() if v is not None})
                    break
        except Exception:
            pass
        return params

    def _historical_gate(
        self,
        symbol: str,
        snapshot: Optional['HistoricalSnapshot'],
        params: Dict[str, Any],
        breakout: 'BreakoutInfo',
    ) -> bool:
        if snapshot is None or not params.get('enabled', False):
            return False

        try:
            max_age = float(params.get('max_data_age_minutes', 0) or 0)
            if max_age > 0 and getattr(snapshot, 'generated_at', None):
                age_minutes = (datetime.now(timezone.utc) - snapshot.generated_at).total_seconds() / 60.0
                if age_minutes > max_age:
                    logger.info(f"[INFO] {symbol}: historical snapshot stale ({age_minutes:.1f}m > {max_age:.1f}m); skipping historical gate")
                    return False
        except Exception:
            pass

        if params.get('block_on_adr_exhaustion', True) and getattr(snapshot, 'adr_exhausted', False):
            current_block_state = "HIST_ADR_EXHAUSTED"
            if self._last_block_reason.get(symbol) != current_block_state:
                progress_label = (
                    f"{float(snapshot.adr_progress):.2f}"
                    if getattr(snapshot, 'adr_progress', None) is not None
                    else "n/a"
                )
                logger.info(f"[BLOCK] {symbol}: ADR exhausted (progress={progress_label})")
                self._last_block_reason[symbol] = current_block_state
            return True

        if params.get('enforce_trend_alignment', True):
            trend = getattr(snapshot, 'trend_bias', None)
            if trend in ('bullish', 'bearish') and trend != breakout.type:
                current_block_state = "HIST_TREND_MISMATCH"
                if self._last_block_reason.get(symbol) != current_block_state:
                    logger.info(f"[BLOCK] {symbol}: historical trend bias {trend} opposes breakout {breakout.type}")
                    self._last_block_reason[symbol] = current_block_state
                return True

        min_rate = params.get('min_breakout_success_rate', None)
        snap_rate = getattr(snapshot, 'breakout_success_rate', None)
        if (
            min_rate is not None
            and snap_rate is not None
            and float(snap_rate) < float(min_rate)
        ):
            current_block_state = "HIST_BREAKOUT_RATE_LOW"
            if self._last_block_reason.get(symbol) != current_block_state:
                logger.info(
                    f"[BLOCK] {symbol}: historical breakout success {snap_rate:.2f} < min {float(min_rate):.2f}"
                )
                self._last_block_reason[symbol] = current_block_state
            return True
        return False

    def _detect_breakout_close(self, last_close: float, resistance: List[float], support: List[float], threshold_pips: float, pip: float) -> Optional[BreakoutInfo]:
        """
        Detect breakout by finding the HIGHEST resistance or LOWEST support broken.

        This is critical for correct SL placement - we need the most significant level broken,
        not just any level that was penetrated.
        """
        thr = threshold_pips * pip

        # BUY: Find HIGHEST resistance broken (most significant level)
        broken_resistances = [level for level in resistance if last_close > level + thr]
        if broken_resistances:
            highest_broken = max(broken_resistances)
            return BreakoutInfo('bullish', highest_broken, last_close)

        # SELL: Find LOWEST support broken (most significant level)
        broken_supports = [level for level in support if last_close < level - thr]
        if broken_supports:
            lowest_broken = min(broken_supports)
            return BreakoutInfo('bearish', lowest_broken, last_close)

        return None

    def _stop_loss(self, breakout: BreakoutInfo, pip: float, symbol: str, atr_last: Optional[float],
                   ctx_res: List[float], ctx_sup: List[float]) -> float:
        """
        Place SL beyond nearest opposing S/R level to avoid pullback stops.

        For bullish: SL below nearest support (not just below broken resistance)
        For bearish: SL above nearest resistance (not just above broken support)

        This prevents stop-outs during normal retests of structure.
        """
        # Apply per-symbol overrides if present
        buf_pips = float(self.stop_loss_buffer_pips)
        min_sl_pips = float(self.min_stop_loss_pips)
        atr_k = float(self.atr_sl_k)
        min_buf_pips = float(self.min_sl_buffer_pips)
        try:
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    buf_pips = float(sc.get('stop_loss_buffer_pips', buf_pips))
                    min_sl_pips = float(sc.get('min_stop_loss_pips', min_sl_pips))
                    atr_k = float(sc.get('atr_sl_k', atr_k))
                    min_buf_pips = float(sc.get('min_sl_buffer_pips', min_buf_pips))
                    break
        except Exception:
            pass

        # Dynamic extra buffer beyond structure: max(config buffer, min buffer, ATR*K)
        buf_price = float(buf_pips) * float(pip)
        add_min_buf_price = float(min_buf_pips) * float(pip)
        atr_price = float(atr_last) if atr_last is not None else 0.0
        dyn_extra = max(buf_price, add_min_buf_price, float(atr_k) * atr_price)

        min_sl = float(min_sl_pips) * float(pip)
        entry = breakout.entry_price
        level = breakout.level  # Broken level

        if breakout.type == 'bullish':
            # Find nearest SUPPORT below broken resistance to avoid pullback stops
            supports_below = [s for s in ctx_sup if s < level]

            if supports_below:
                # Place SL below nearest support (not just below broken resistance)
                nearest_support = max(supports_below)  # Closest support to broken level
                sl_struct = nearest_support - dyn_extra
                logger.debug(f"{symbol}: Using support {nearest_support:.5f} for SL (below broken R {level:.5f})")
            else:
                # No support below broken level - use broken level itself
                sl_struct = level - dyn_extra
                logger.debug(f"{symbol}: No support below, SL below broken level {level:.5f}")

            sl_min = entry - min_sl
            # Choose the deeper stop (further from entry) to ensure proper protection
            return min(sl_struct, sl_min)

        else:  # bearish
            # Find nearest RESISTANCE above broken support to avoid pullback stops
            resistances_above = [r for r in ctx_res if r > level]

            if resistances_above:
                # Place SL above nearest resistance (not just above broken support)
                nearest_resistance = min(resistances_above)  # Closest resistance to broken level
                sl_struct = nearest_resistance + dyn_extra
                logger.debug(f"{symbol}: Using resistance {nearest_resistance:.5f} for SL (above broken S {level:.5f})")
            else:
                # No resistance above broken level - use broken level itself
                sl_struct = level + dyn_extra
                logger.debug(f"{symbol}: No resistance above, SL above broken level {level:.5f}")

            sl_min = entry + min_sl
            # Choose the deeper stop above entry
            return max(sl_struct, sl_min)

    def _take_profit(self, entry: float, stop: float, rr: float, side: str) -> float:
        dist = abs(entry - stop) * rr
        return entry + dist if side == 'bullish' else entry - dist

    def _next_obstacle_level(self, side: str, entry_price: float, ctx_res: List[float], ctx_sup: List[float]) -> Optional[float]:
        """Return the next opposing S/R level ahead of price in the breakout direction.
        - For bullish, the next resistance strictly above entry.
        - For bearish, the next support strictly below entry.
        Levels are expected sorted ascending.
        """
        if side == 'bullish':
            ahead = [x for x in ctx_res if x > entry_price]
            return ahead[0] if ahead else None
        else:
            behind = [x for x in ctx_sup if x < entry_price]
            return behind[-1] if behind else None

    # -----------------------------
    # Public: generate signal
    # -----------------------------
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        mtf_context: Optional[dict] = None,
        historical_snapshot: Optional['HistoricalSnapshot'] = None,
    ) -> Optional[TradingSignal]:
        try:
            hist_snapshot = historical_snapshot if historical_snapshot else None
            if data is None or len(data) < max(20, self.lookback_period):
                return None

            tick = self._get_tick(symbol)
            info = self._get_symbol_info(symbol)
            if not tick or not info:
                return None
            pip = resolve_pip_size(symbol, info, self.config)
            if pip <= 0:
                return None

            # Calculate current spread
            current_spread_pips = abs(float(tick.ask) - float(tick.bid)) / float(pip)

            # Spread guard (optional per symbol - absolute threshold)
            try:
                spread_guard_pips = None
                for sc in getattr(self.config, 'symbols', []) or []:
                    if sc.get('name') == symbol:
                        spread_guard_pips = sc.get('spread_guard_pips', None)
                        break
                # Fallback to global default if defined
                if spread_guard_pips is None:
                    spread_guard_pips = getattr(self.config, 'spread_guard_pips_default', None)
                if spread_guard_pips is not None:
                    if current_spread_pips > float(spread_guard_pips):
                        current_block_state = "SPREAD_GUARD"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(f"[BLOCK] {symbol}: spread {current_spread_pips:.1f}p > guard {float(spread_guard_pips):.1f}p")
                            self._last_block_reason[symbol] = current_block_state
                        return None
            except Exception:
                pass

            # Session awareness: Dynamic spread filter (relative threshold)
            try:
                spread_multiplier = float(getattr(self.config, 'spread_multiplier_threshold', 1.5))
                spread_lookback = int(getattr(self.config, 'spread_lookback_bars', 20))

                # Calculate average spread from recent bars
                if len(data) >= spread_lookback:
                    recent_bars = data.iloc[-spread_lookback:]
                    if 'spread' in recent_bars.columns:
                        avg_spread_pips = float(recent_bars['spread'].mean()) / float(pip)
                    else:
                        # Calculate spread from high-low as proxy if not available
                        avg_range_pips = float((recent_bars['high'] - recent_bars['low']).mean()) / float(pip)
                        # Estimate spread as ~10% of average bar range (conservative estimate)
                        avg_spread_pips = avg_range_pips * 0.1

                    # Block if current spread is abnormally wide
                    if avg_spread_pips > 0 and current_spread_pips > (avg_spread_pips * spread_multiplier):
                        current_block_state = "ELEVATED_SPREAD"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(f"[BLOCK] {symbol}: elevated spread ({current_spread_pips:.1f}p > {spread_multiplier}x avg {avg_spread_pips:.1f}p)")
                            self._last_block_reason[symbol] = current_block_state
                        return None
            except Exception:
                pass

            # Use closed candles for structure only (local structure window)
            completed = data.iloc[:-1].tail(self.lookback_period)
            if len(completed) < max(20, self.lookback_period):
                return None

            highs, lows = self.find_swing_points(completed)
            if not highs and not lows:
                logger.debug(f"{symbol}: No swing points found")
                return None

            resistance, support = self.calculate_support_resistance(completed, highs, lows, symbol)

            # Historical context window (broader)
            try:
                context_window = max(int(self.context_lookback_period), int(self.lookback_period))
            except Exception:
                context_window = max(100, self.lookback_period * 3)
            ctx_src = data.iloc[:-1].tail(context_window)
            ctx_res, ctx_sup = [], []
            if len(ctx_src) >= max(20, self.swing_window * 2 + 1):
                c_highs, c_lows = self.find_swing_points(ctx_src)
                if c_highs or c_lows:
                    ctx_res, ctx_sup = self.calculate_support_resistance(ctx_src, c_highs, c_lows, symbol)

            # Merge in multi-timeframe contextual S/R if provided
            proximity = float(getattr(self.config, 'level_merge_proximity_pips', 10.0)) * float(pip)
            def _merge_levels(base: List[float], add: List[float]) -> List[float]:
                out: List[float] = list(base)
                for lvl in add:
                    if not any(abs(lvl - x) <= proximity for x in out):
                        out.append(float(lvl))
                return sorted(out)

            try:
                if isinstance(mtf_context, dict):
                    # For each timeframe df in context, compute its own S/R and merge
                    for tf_name, df_tf in mtf_context.items():
                        if df_tf is None or not hasattr(df_tf, 'iloc'):
                            continue
                        src_tf = df_tf.iloc[:-1].tail(context_window)
                        if len(src_tf) < max(20, self.swing_window * 2 + 1):
                            continue
                        th, tl = self.find_swing_points(src_tf)
                        if not th and not tl:
                            continue
                        r_tf, s_tf = self.calculate_support_resistance(src_tf, th, tl, symbol)
                        if r_tf:
                            ctx_res = _merge_levels(ctx_res, r_tf)
                        if s_tf:
                            ctx_sup = _merge_levels(ctx_sup, s_tf)
                    try:
                        logger.debug(f"{symbol}: MTF context levels res={ctx_res} sup={ctx_sup}")
                    except Exception:
                        pass
            except Exception:
                pass

            # Keep breakout engine independent from external level injectors

            # Debug observability
            logger.debug(f"{symbol}: pip={pip:.5f} thr_pips={self.breakout_threshold_pips} res={resistance} sup={support}")
            logger.debug(f"{symbol}: tick.ask={float(tick.ask):.5f} tick.bid={float(tick.bid):.5f}")

            # Per-symbol overrides for RR and threshold
            thr_pips = float(self.breakout_threshold_pips)
            rr = float(self.risk_reward_ratio)
            atr_period = int(self.atr_period)
            min_headroom_rr = float(self.min_headroom_rr)
            max_rr_cap = self.max_rr_cap
            max_sl_pips = self.max_sl_pips
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    thr_pips = float(sc.get('breakout_threshold_pips', thr_pips))
                    rr = float(sc.get('risk_reward_ratio', rr))
                    atr_period = int(sc.get('atr_period', atr_period))
                    min_headroom_rr = float(sc.get('min_headroom_rr', min_headroom_rr))
                    max_rr_cap = sc.get('max_rr_cap', max_rr_cap)
                    max_sl_pips = sc.get('max_sl_pips', max_sl_pips)
                    break

            # Use last closed candle close for breakout confirmation
            last_close = float(completed.iloc[-1]['close'])

            # Use configured breakout threshold (constant)
            bo = self._detect_breakout_close(last_close, resistance, support, thr_pips, pip)
            if not bo:
                return None

            logger.debug(f"{symbol}: breakout type={bo.type} level={bo.level:.5f} entry={bo.entry_price:.5f}")

            # Zone-size filter: Skip if opposing S/R is too close (whipsaw risk)
            enable_zone_guard = bool(getattr(self.config, 'enable_zone_guard', True))
            min_zone_pips = float(getattr(self.config, 'min_breakout_zone_pips', 20))
            try:
                for sc in getattr(self.config, 'symbols', []) or []:
                    if sc.get('name') == symbol:
                        enable_zone_guard = bool(sc.get('enable_zone_guard', enable_zone_guard))
                        min_zone_pips = float(sc.get('min_breakout_zone_pips', min_zone_pips))
                        break
            except Exception:
                pass

            if enable_zone_guard and min_zone_pips > 0:
                if bo.type == 'bullish':
                    # Check nearest support below broken resistance
                    supports_below = [s for s in ctx_sup if s < bo.level]
                    if supports_below:
                        nearest_support = max(supports_below)
                        zone_size_pips = (bo.level - nearest_support) / pip
                        if zone_size_pips < min_zone_pips:
                            current_block_state = "ZONE_TOO_TIGHT"
                            if self._last_block_reason.get(symbol) != current_block_state:
                                logger.info(f"[BLOCK] {symbol}: breakout zone too tight ({zone_size_pips:.1f}p < {min_zone_pips:.1f}p min); high whipsaw risk")
                                self._last_block_reason[symbol] = current_block_state
                            return None
                else:  # bearish
                    # Check nearest resistance above broken support
                    resistances_above = [r for r in ctx_res if r > bo.level]
                    if resistances_above:
                        nearest_resistance = min(resistances_above)
                        zone_size_pips = (nearest_resistance - bo.level) / pip
                        if zone_size_pips < min_zone_pips:
                            current_block_state = "ZONE_TOO_TIGHT"
                            if self._last_block_reason.get(symbol) != current_block_state:
                                logger.info(f"[BLOCK] {symbol}: breakout zone too tight ({zone_size_pips:.1f}p < {min_zone_pips:.1f}p min); high whipsaw risk")
                                self._last_block_reason[symbol] = current_block_state
                            return None

            hist_params = self._resolve_historical_params(symbol)
            if self._historical_gate(symbol, hist_snapshot, hist_params, bo):
                return None

            # Breakout acceptance (anti-fakeout): require quality breakout bars
            if self.entry_mode == 'confirm':
                need_bars = max(1, int(self.entry_confirmation_bars))

                if need_bars > 1:
                    # Multi-bar confirmation: require N consecutive closes beyond level
                    if len(completed) < need_bars:
                        return None
                    thr_price = float(thr_pips) * float(pip)
                    closes = completed['close'].iloc[-need_bars:]
                    if bo.type == 'bullish':
                        if not (closes > (bo.level + thr_price)).all():
                            current_block_state = "ENTRY_CONFIRMATION_CLOSE"
                            if self._last_block_reason.get(symbol) != current_block_state:
                                logger.info(f"[BLOCK] {symbol}: entry confirmation failed (need {need_bars} close(s) beyond level)")
                                self._last_block_reason[symbol] = current_block_state
                            return None
                    else:
                        if not (closes < (bo.level - thr_price)).all():
                            current_block_state = "ENTRY_CONFIRMATION_CLOSE"
                            if self._last_block_reason.get(symbol) != current_block_state:
                                logger.info(f"[BLOCK] {symbol}: entry confirmation failed (need {need_bars} close(s) beyond level)")
                                self._last_block_reason[symbol] = current_block_state
                            return None
                else:
                    # Single-bar confirmation: require strong close (not weak wick)
                    # Check that close is in top/bottom 30% of bar range for conviction
                    last_bar = completed.iloc[-1]
                    bar_high = float(last_bar['high'])
                    bar_low = float(last_bar['low'])
                    bar_close = float(last_bar['close'])
                    bar_range = bar_high - bar_low

                    if bar_range > 0:
                        if bo.type == 'bullish':
                            # For bullish, close should be in upper 30% of bar
                            close_position = (bar_close - bar_low) / bar_range
                            if close_position < 0.7:  # Close in lower 70% = weak
                                current_block_state = "WEAK_BREAKOUT_CLOSE"
                                if self._last_block_reason.get(symbol) != current_block_state:
                                    logger.info(f"[BLOCK] {symbol}: weak breakout bar (close at {close_position*100:.0f}% of range, need >70%)")
                                    self._last_block_reason[symbol] = current_block_state
                                return None
                        else:  # bearish
                            # For bearish, close should be in lower 30% of bar
                            close_position = (bar_close - bar_low) / bar_range
                            if close_position > 0.3:  # Close in upper 70% = weak
                                current_block_state = "WEAK_BREAKOUT_CLOSE"
                                if self._last_block_reason.get(symbol) != current_block_state:
                                    logger.info(f"[BLOCK] {symbol}: weak breakout bar (close at {close_position*100:.0f}% of range, need <30%)")
                                    self._last_block_reason[symbol] = current_block_state
                                return None

            # TA-Lib candlestick usage
            pattern_info_entry: Optional[dict] = None
            pattern_info_bias: Optional[dict] = None

            def _mtf_df_for(tf_label: Optional[str]) -> Optional[pd.DataFrame]:
                if not tf_label or not isinstance(mtf_context, dict):
                    return None
                for key in (tf_label, str(tf_label).upper(), str(tf_label).lower()):
                    if key in mtf_context:
                        return mtf_context[key]
                return None

            # Optional trading timeframe confirmation (disabled by default to avoid delays)
            if self.enable_patterns and self.pattern_use_trading_tf:
                try:
                    gate_skipped = False
                    pattern_ok, info = self._candlestick_confirm(
                        completed,
                        bo,
                        symbol,
                        window=self.pattern_window,
                        scope="LTF",
                    )
                    pattern_info_entry = info or None

                    if isinstance(pattern_info_entry, dict) and pattern_info_entry.get('skipped'):
                        logger.warning(f"{symbol}: TA-Lib unavailable; skipping trading-tf pattern gate")
                        pattern_info_entry = None
                        gate_skipped = True
                        pattern_ok = True

                    if pattern_info_entry is not None:
                        pattern_info_entry = dict(pattern_info_entry)
                        pattern_info_entry.setdefault('timeframe', 'trading')
                        pattern_info_entry['scope'] = 'entry'

                    if gate_skipped:
                        logger.debug(f"{symbol}: pattern gate skipped (trading timeframe)")
                        pattern_info_entry = None
                    elif not pattern_info_entry or pattern_info_entry.get('score', 0.0) == 0:
                        current_block_state = "NO_PATTERN_CONFIRMATION"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(f"[BLOCK] {symbol}: no LTF pattern confirmation (score=0)")
                            self._last_block_reason[symbol] = current_block_state
                        return None

                    score = float(pattern_info_entry.get('score', 0.0) or 0.0)
                    pattern_dir = pattern_info_entry.get('dir')

                    if pattern_dir and pattern_dir != bo.type:
                        current_block_state = "PATTERN_DIRECTION_MISMATCH"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(
                                f"[BLOCK] {symbol}: LTF pattern opposes breakout (pattern={pattern_dir}, breakout={bo.type}, score={score:.2f})"
                            )
                            self._last_block_reason[symbol] = current_block_state
                        return None

                    if score < float(self.pattern_score_threshold):
                        current_block_state = "PATTERN_SCORE_LOW"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(
                                f"[BLOCK] {symbol}: LTF pattern score too low ({score:.2f} < {self.pattern_score_threshold})"
                            )
                            self._last_block_reason[symbol] = current_block_state
                        return None
                    logger.debug(f"{symbol}: LTF pattern confirmed ({pattern_dir}, score={score:.2f})")
                except Exception as e:
                    current_block_state = "PATTERN_ANALYSIS_ERROR"
                    if self._last_block_reason.get(symbol) != current_block_state:
                        logger.warning(f"[BLOCK] {symbol}: LTF pattern analysis failed ({str(e)})")
                        self._last_block_reason[symbol] = current_block_state
                    pattern_info_entry = None
                    return None

            # Higher timeframe bias confirmation (preferred)
            if self.enable_patterns:
                htf_tf, require_htf = self._resolve_htf_pattern_prefs(symbol)
                htf_df = _mtf_df_for(htf_tf)
                if htf_tf and htf_df is None:
                    if require_htf:
                        current_block_state = "HTF_PATTERN_CONTEXT_MISSING"
                        if self._last_block_reason.get(symbol) != current_block_state:
                            logger.info(f"[BLOCK] {symbol}: HTF ({htf_tf}) data unavailable for bias confirmation")
                            self._last_block_reason[symbol] = current_block_state
                        return None
                if htf_tf and htf_df is not None:
                    htf_completed = htf_df.iloc[:-1] if len(htf_df) > 1 else htf_df
                    if len(htf_completed) < max(1, self.htf_pattern_window):
                        if require_htf:
                            current_block_state = "HTF_PATTERN_DATA_SHORT"
                            if self._last_block_reason.get(symbol) != current_block_state:
                                logger.info(f"[BLOCK] {symbol}: insufficient HTF bars for pattern confirmation ({len(htf_completed)} < {self.htf_pattern_window})")
                                self._last_block_reason[symbol] = current_block_state
                            return None
                    else:
                        try:
                            gate_skipped = False
                            pattern_ok, info = self._candlestick_confirm(
                                htf_completed,
                                bo,
                                symbol,
                                window=self.htf_pattern_window,
                                scope=f"HTF-{htf_tf}",
                            )
                            pattern_info_bias = info or None

                            if isinstance(pattern_info_bias, dict) and pattern_info_bias.get('skipped'):
                                logger.warning(f"{symbol}: TA-Lib unavailable; skipping HTF ({htf_tf}) pattern gate")
                                gate_skipped = True
                                pattern_info_bias = None
                                pattern_ok = True

                            if pattern_info_bias is not None:
                                pattern_info_bias = dict(pattern_info_bias)
                                pattern_info_bias.setdefault('timeframe', htf_tf)
                                pattern_info_bias['scope'] = 'bias'

                            bias_dir = pattern_info_bias.get('dir') if isinstance(pattern_info_bias, dict) else None
                            bias_score = float((pattern_info_bias or {}).get('score', 0.0) or 0.0)

                            if gate_skipped:
                                logger.debug(f"{symbol}: pattern gate skipped (HTF {htf_tf})")
                                pattern_info_bias = None
                            elif not pattern_ok:
                                if not bias_dir:
                                    current_block_state = "HTF_PATTERN_ABSENT"
                                    if self._last_block_reason.get(symbol) != current_block_state:
                                        logger.info(f"[BLOCK] {symbol}: no HTF pattern confirmation (score={bias_score:.2f})")
                                        self._last_block_reason[symbol] = current_block_state
                                else:
                                    current_block_state = "HTF_PATTERN_DIRECTION_MISMATCH"
                                    if self._last_block_reason.get(symbol) != current_block_state:
                                        logger.info(
                                            f"[BLOCK] {symbol}: HTF pattern opposes breakout (pattern={bias_dir}, breakout={bo.type})"
                                        )
                                        self._last_block_reason[symbol] = current_block_state
                                return None

                            if bias_score < float(self.htf_pattern_score_threshold):
                                current_block_state = "HTF_PATTERN_SCORE_LOW"
                                if self._last_block_reason.get(symbol) != current_block_state:
                                    logger.info(
                                        f"[BLOCK] {symbol}: HTF pattern score too low ({bias_score:.2f} < {self.htf_pattern_score_threshold})"
                                    )
                                    self._last_block_reason[symbol] = current_block_state
                                return None
                            logger.debug(f"{symbol}: HTF pattern confirmed ({bias_dir}, score={bias_score:.2f})")
                        except Exception as e:
                            if require_htf:
                                current_block_state = "HTF_PATTERN_ANALYSIS_ERROR"
                                if self._last_block_reason.get(symbol) != current_block_state:
                                    logger.warning(f"[BLOCK] {symbol}: HTF pattern analysis failed ({str(e)})")
                                    self._last_block_reason[symbol] = current_block_state
                                return None
                            pattern_info_bias = None

            pattern_info = pattern_info_bias or pattern_info_entry

            # Emit at most one signal per closed bar for the same side
            try:
                bar_time = completed.index[-1]
                keyb = (symbol, bo.type)
                if self._last_breakout_bar.get(keyb) == bar_time:
                    logger.debug(f"{symbol}: duplicate signal for same bar/direction; skipping")
                    return None
                self._last_breakout_bar[keyb] = bar_time
            except Exception:
                pass

            # Compute ATR on completed window
            atr_series = self._compute_atr(completed['high'], completed['low'], completed['close'], atr_period)
            atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else None

            sl = self._stop_loss(bo, pip, symbol, atr_last, ctx_res, ctx_sup)
            sl_pips = abs(bo.entry_price - sl) / pip
            # Enforce max SL distance if configured
            if max_sl_pips is not None and sl_pips > float(max_sl_pips):
                current_block_state = "MAX_SL_EXCEEDED"
                if self._last_block_reason.get(symbol) != current_block_state:
                    logger.info(f"[BLOCK] {symbol}: SL {sl_pips:.1f}p > max {float(max_sl_pips):.1f}p; skipping")
                    self._last_block_reason[symbol] = current_block_state
                return None


            # Use actual execution entry price (ask/bid) for all calculations
            try:
                entry_eff = float(tick.ask) if bo.type == 'bullish' else float(tick.bid)
            except Exception:
                entry_eff = bo.entry_price

            # Calculate RR with cap
            rr_eff = float(rr)
            if max_rr_cap is not None:
                try:
                    rr_eff = min(rr_eff, float(max_rr_cap))
                except Exception:
                    pass

            # Calculate initial TP based on effective entry
            tp_initial = self._take_profit(entry_eff, sl, rr_eff, bo.type)

            # Find next obstacle ONCE using effective entry
            obstacle = None
            try:
                obstacle = self._next_obstacle_level(bo.type, entry_eff, ctx_res, ctx_sup)
            except Exception:
                obstacle = None

            # Calculate headroom and potentially cap TP
            obstacle_buffer_pips_eff = float(self.obstacle_buffer_pips)
            buf = float(obstacle_buffer_pips_eff) * float(pip)

            enable_headroom_guard = bool(getattr(self.config, 'enable_headroom_guard', True))
            try:
                for sc in getattr(self.config, 'symbols', []) or []:
                    if sc.get('name') == symbol:
                        enable_headroom_guard = bool(sc.get('enable_headroom_guard', enable_headroom_guard))
                        break
            except Exception:
                pass

            if obstacle is not None and enable_headroom_guard:
                # Calculate distance to obstacle
                headroom_price = abs(obstacle - entry_eff) - buf
                headroom_pips = max(0.0, headroom_price / float(pip))

                # Check minimum headroom requirement FIRST (block if path too constrained)
                min_headroom_rr_eff = float(min_headroom_rr)
                required_headroom = min_headroom_rr_eff * sl_pips

                try:
                    logger.debug(f"{symbol}: obstacle={obstacle:.5f} entry_eff={entry_eff:.5f} headroom={headroom_pips:.1f}p req={required_headroom:.1f}p")
                except Exception:
                    pass

                if headroom_pips < required_headroom:
                    current_block_state = "INSUFFICIENT_HEADROOM"
                    if self._last_block_reason.get(symbol) != current_block_state:
                        logger.info(f"[BLOCK] {symbol}: insufficient headroom to obstacle ({headroom_pips:.1f}p < {required_headroom:.1f}p)")
                        self._last_block_reason[symbol] = current_block_state
                    return None

                # Cap TP to just before obstacle if it would exceed
                if bo.type == 'bullish':
                    tp_max = obstacle - buf
                    if tp_initial > tp_max:
                        tp_original = tp_initial
                        tp_initial = tp_max
                        try:
                            logger.debug(f"{symbol}: TP capped from {tp_original:.5f} to {tp_initial:.5f} (obstacle at {obstacle:.5f})")
                        except Exception:
                            pass
                else:  # bearish
                    tp_max = obstacle + buf
                    if tp_initial < tp_max:
                        tp_original = tp_initial
                        tp_initial = tp_max
                        try:
                            logger.debug(f"{symbol}: TP capped from {tp_original:.5f} to {tp_initial:.5f} (obstacle at {obstacle:.5f})")
                        except Exception:
                            pass

            tp = tp_initial

            # Round to broker precision
            point = getattr(info, 'point', None)
            digits = getattr(info, 'digits', None)
            if point and digits is not None:
                sl = round(round(sl / point) * point, int(digits))
                tp = round(round(tp / point) * point, int(digits))

            # Calculate actual RR using effective entry price
            sl_pips = abs(entry_eff - sl) / pip
            tp_pips = abs(tp - entry_eff) / pip

            if sl_pips <= 0:
                return None

            actual_rr = tp_pips / sl_pips if sl_pips > 0 else 0

            # Tiered acceptance based on trade quality
            confidence = 0.5  # Default

            # Tier 1: Full RR achieved (best quality)
            if actual_rr >= rr_eff:
                confidence = 0.8
                logger.debug(f"{symbol}: Tier 1 trade (RR={actual_rr:.2f} >= target {rr_eff:.2f})")

            # Tier 2: Good RR but reduced (good quality)
            elif actual_rr >= float(getattr(self.config, 'min_rr_tier2', 1.2)):
                confidence = 0.6
                logger.debug(f"{symbol}: Tier 2 trade (RR={actual_rr:.2f})")

            # Tier 3: Minimal acceptable RR (acceptable quality)
            elif actual_rr >= float(getattr(self.config, 'min_rr_tier3', 1.0)):
                confidence = 0.5
                logger.debug(f"{symbol}: Tier 3 trade (RR={actual_rr:.2f})")

            # Tier 4: RR too low, reject
            else:
                current_block_state = "INSUFFICIENT_RR"
                if self._last_block_reason.get(symbol) != current_block_state:
                    min_acceptable = float(getattr(self.config, 'min_rr_tier3', 1.0))
                    logger.info(f"[BLOCK] {symbol}: RR too low ({actual_rr:.2f} < {min_acceptable:.2f})")
                    self._last_block_reason[symbol] = current_block_state
                return None

            try:
                # Provide visibility into context decisioning
                logger.debug(f"{symbol}: sl_pips={sl_pips:.2f} rr_cfg={rr:.2f} rr_eff={rr_eff:.2f} tp={tp:.5f}")
            except Exception:
                logger.debug(f"{symbol}: sl_pips={sl_pips:.2f} tp={tp:.5f}")

            signal = TradingSignal(
                type=0 if bo.type == 'bullish' else 1,
                entry_price=entry_eff,
                stop_loss=sl,
                take_profit=tp,
                stop_loss_pips=sl_pips,
                reason=f"core_{bo.type}_breakout",
                confidence=confidence,
                timestamp=datetime.now(timezone.utc),
                breakout_level=bo.level,
                pattern_score=(pattern_info.get('score') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                pattern_dir=(pattern_info.get('dir') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                pattern_primary=(pattern_info.get('primary') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                pattern_timeframe=(pattern_info.get('timeframe') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                hist_trend_bias=getattr(hist_snapshot, 'trend_bias', None) if hist_snapshot else None,
                hist_breakout_rate=getattr(hist_snapshot, 'breakout_success_rate', None) if hist_snapshot else None,
                hist_adr_progress=getattr(hist_snapshot, 'adr_progress', None) if hist_snapshot else None,
            )
            # Enriched INFO log for terminal: includes pattern/ATR source if available
            try:
                pat_label = None
                pat_score = None
                if 'pattern_info' in locals() and isinstance(pattern_info, dict):
                    pat_label = pattern_info.get('primary') or pattern_info.get('dir')
                    tf = pattern_info.get('timeframe')
                    if tf and pat_label:
                        pat_label = f"{tf}:{pat_label}"
                    ps = pattern_info.get('score')
                    pat_score = f"{float(ps):.2f}" if ps is not None else None
                parts = [
                    f"CORE SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'}",
                    f"ref @ {signal.entry_price:.5f}",
                    f"SL {signal.stop_loss:.5f}",
                    f"TP {signal.take_profit:.5f}",
                    f"({sl_pips:.1f}p, RR={actual_rr:.2f}, conf={confidence:.1f})",
                ]
                if pat_label and pat_score:
                    parts.append(f"pat={pat_label}:{pat_score}")
                if signal.hist_trend_bias:
                    parts.append(f"trend={signal.hist_trend_bias}")
                if signal.hist_breakout_rate is not None:
                    parts.append(f"histWin={float(signal.hist_breakout_rate):.2f}")
                if signal.hist_adr_progress is not None:
                    parts.append(f"ADRprog={float(signal.hist_adr_progress):.2f}")
                parts.append(f"ATR={str(self.atr_source).upper()}")
                # Add context if available
                logger.info(" ".join(parts))
            except Exception:
                logger.info(
                    f"CORE SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'} ref @ {signal.entry_price:.5f} "
                    f"SL {signal.stop_loss:.5f} TP {signal.take_profit:.5f} ({sl_pips:.1f}p, RR={actual_rr:.2f}, conf={confidence:.1f})"
                )
            return signal
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
