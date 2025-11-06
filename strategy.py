"""
Core Price Action Strategy.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple, NamedTuple

import MetaTrader5 as mt5
import pandas as pd
import logging

from utils import get_pip_size, resolve_pip_size
from patterns import candlestick_confirm

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
    # Optional pattern context for logging/analytics
    pattern_score: Optional[float] = None
    pattern_dir: Optional[str] = None
    pattern_primary: Optional[str] = None

class BreakoutInfo(NamedTuple):
    type: str      # 'bullish' or 'bearish'
    level: float
    entry_price: float

class PurePriceActionStrategy:
    """Minimal breakout strategy based on swing levels and fixed pip thresholds."""

    def __init__(self, config):
        self.config = config
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
        # Patterns/ATR integration controls
        self.enable_patterns = bool(getattr(config, 'enable_patterns', True))
        self.pattern_window = int(getattr(config, 'pattern_window', 3))
        self.pattern_score_threshold = float(getattr(config, 'pattern_score_threshold', 0.6))
        self.pattern_strong_threshold = float(getattr(config, 'pattern_strong_threshold', 0.8))
        self.allowed_patterns = getattr(config, 'allowed_patterns', None)
        self.dup_relax_on_strong_pattern = bool(getattr(config, 'dup_relax_on_strong_pattern', True))
        self.headroom_relax_pct_on_strong = float(getattr(config, 'headroom_relax_pct_on_strong', 0.25))
        self.obstacle_buffer_relax_pct_on_strong = float(getattr(config, 'obstacle_buffer_relax_pct_on_strong', 0.3))
        self.atr_source = getattr(config, 'atr_source', 'talib')
        self._last_breakout = {}  
        self._last_breakout_bar = {} 

    # -----------------------------
    # Swing points and S/R levels
    # -----------------------------
    def _find_swing_indices(self, series: pd.Series, window: int) -> List[int]:
        idxs: List[int] = []
        values = series.values
        n = len(values)
        for i in range(window, n - window):
            left = values[i - window:i + 1]
            right = values[i:i + window + 1]
            
            if values[i] >= left.max() and values[i] >= right.max():
                idxs.append(i)
        return idxs

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        if data is None or len(data) < self.swing_window * 2 + 1:
            return [], []
        highs = self._find_swing_indices(data['high'], self.swing_window)
        lows = self._find_swing_indices(-data['low'], self.swing_window)  # reuse max finder on inverted lows
        return highs, lows

    def calculate_support_resistance(self, data: pd.DataFrame, swing_highs: List[int], swing_lows: List[int], symbol: str) -> Tuple[List[float], List[float]]:
        """Pick up to 3 recent distinct swing highs/lows as resistance/support."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return [], []
        pip = get_pip_size(symbol_info)
        proximity = 10 * pip  # 10 pips proximity consolidation

        res: List[float] = []
        sup: List[float] = []

        # Take the last 50 swings to avoid ancient levels
        for i in reversed(swing_highs[-50:]):
            level = float(data.iloc[i]['high'])
            if not any(abs(level - x) <= proximity for x in res):
                res.append(level)
            if len(res) >= 3:
                break

        for i in reversed(swing_lows[-50:]):
            level = float(data.iloc[i]['low'])
            if not any(abs(level - x) <= proximity for x in sup):
                sup.append(level)
            if len(sup) >= 3:
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

    def _candlestick_confirm(self, completed: pd.DataFrame, bo: 'BreakoutInfo', symbol: str) -> Tuple[bool, Optional[dict]]:
        """Confirm breakout using TA-Lib candlestick patterns. Fail-open if disabled or TA-Lib missing."""
        if not self.enable_patterns:
            return True, None
        try:
            ok, info = candlestick_confirm(
                completed[['open', 'high', 'low', 'close']],
                direction=bo.type,
                window=int(self.pattern_window),
                allowed_patterns=self.allowed_patterns,
            )
            try:
                logger.debug(f"{symbol}: pattern_ok={ok} dir={info.get('dir') if info else None} score={info.get('score') if info else None}")
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
        
    def _detect_breakout_close(self, last_close: float, resistance: List[float], support: List[float], threshold_pips: float, pip: float) -> Optional[BreakoutInfo]:
        thr = threshold_pips * pip
        # BUY: close > resistance + thr
        for level in resistance:
            if last_close > level + thr:
                return BreakoutInfo('bullish', level, last_close)
        # SELL: close < support - thr
        for level in support:
            if last_close < level - thr:
                return BreakoutInfo('bearish', level, last_close)
        return None

    def _stop_loss(self, breakout: BreakoutInfo, pip: float, symbol: str, atr_last: Optional[float]) -> float:
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
        level = breakout.level
        if breakout.type == 'bullish':
            # behind broken resistance with extra volatility buffer
            sl_struct = level - dyn_extra
            sl_min = entry - min_sl
            # Choose the deeper stop (further from entry) to avoid sweeps
            return min(sl_struct, sl_min)
        else:
            sl_struct = level + dyn_extra
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
    def generate_signal(self, data: pd.DataFrame, symbol: str, mtf_context: Optional[dict] = None) -> Optional[TradingSignal]:
        try:
            if data is None or len(data) < max(20, self.lookback_period):
                return None

            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if not tick or not info:
                return None
            pip = resolve_pip_size(symbol, info, self.config)
            if pip <= 0:
                return None

            # Spread guard (optional per symbol)
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
                    spread_pips = abs(float(tick.ask) - float(tick.bid)) / float(pip)
                    if spread_pips > float(spread_guard_pips):
                        logger.debug(f"{symbol}: spread {spread_pips:.1f}p > guard {float(spread_guard_pips):.1f}p")
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
            try:
                if isinstance(mtf_context, dict):
                    proximity = 10.0 * float(pip)
                    def _merge_levels(base: List[float], add: List[float]) -> List[float]:
                        out: List[float] = list(base)
                        for lvl in add:
                            if not any(abs(lvl - x) <= proximity for x in out):
                                out.append(float(lvl))
                        return sorted(out)

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
            bo = self._detect_breakout_close(last_close, resistance, support, thr_pips, pip)
            if not bo:
                return None

            logger.debug(f"{symbol}: breakout type={bo.type} level={bo.level:.5f} entry={bo.entry_price:.5f}")

            # TA-Lib candlestick confirmation (gate). If disabled/unavailable, pass-through.
            pattern_info = None
            if self.enable_patterns:
                ok, info = self._candlestick_confirm(completed, bo, symbol)
                if not ok:
                    return None
                pattern_info = info or None

            # Emit at most one signal per closed bar for the same side
            try:
                bar_time = completed.index[-1]
                keyb = (symbol, bo.type)
                if self._last_breakout_bar.get(keyb) == bar_time:
                    return None
                self._last_breakout_bar[keyb] = bar_time
            except Exception:
                pass

            # Duplicate breakout suppression (optional)
            try:
                dup_distance_pips = None
                dup_window_sec = None
                for sc in getattr(self.config, 'symbols', []) or []:
                    if sc.get('name') == symbol:
                        dup_distance_pips = sc.get('duplicate_breakout_distance_pips', None)
                        dup_window_sec = sc.get('duplicate_breakout_window_seconds', None)
                        break
                # Fallback to global defaults if defined
                if dup_distance_pips is None:
                    dup_distance_pips = getattr(self.config, 'duplicate_breakout_distance_pips_default', None)
                if dup_window_sec is None:
                    dup_window_sec = getattr(self.config, 'duplicate_breakout_window_seconds_default', None)
                # Relax suppression if a strong confirming pattern exists
                try:
                    strong = False
                    if 'pattern_info' in locals() and pattern_info is not None:
                        strong = float(pattern_info.get('score', 0.0) or 0.0) >= float(self.pattern_strong_threshold)
                    if strong and self.dup_relax_on_strong_pattern and dup_distance_pips is not None and dup_window_sec is not None:
                        dup_distance_pips = max(0.0, float(dup_distance_pips) * 0.6)
                        dup_window_sec = max(60.0, float(dup_window_sec) * 0.5)
                except Exception:
                    pass
                if dup_distance_pips is not None and dup_window_sec is not None:
                    key = (symbol, bo.type)
                    last = self._last_breakout.get(key)
                    if last:
                        dt_now = datetime.now(timezone.utc)
                        if (dt_now - last['time']).total_seconds() <= float(dup_window_sec):
                            dist_pips = abs(float(bo.entry_price) - float(last['price'])) / float(pip)
                            if dist_pips <= float(dup_distance_pips):
                                logger.debug(f"{symbol}: duplicate breakout suppressed ({dist_pips:.1f}p within {dup_window_sec}s)")
                                return None
                    # record current
                    self._last_breakout[key] = {'time': datetime.now(timezone.utc), 'price': float(bo.entry_price)}
            except Exception:
                pass

            # Compute ATR on completed window
            atr_series = self._compute_atr(completed['high'], completed['low'], completed['close'], atr_period)
            atr_last = float(atr_series.iloc[-1]) if atr_series is not None and not pd.isna(atr_series.iloc[-1]) else None

            sl = self._stop_loss(bo, pip, symbol, atr_last)
            sl_pips = abs(bo.entry_price - sl) / pip
            # Enforce max SL distance if configured
            if max_sl_pips is not None and sl_pips > float(max_sl_pips):
                logger.debug((f"{symbol}: SL {sl_pips:.1f}p > max {float(max_sl_pips):.1f}p; skipping"))
                return None

            # If a strong confirming candlestick printed, modestly relax headroom requirement
            try:
                if 'pattern_info' in locals() and pattern_info is not None:
                    _ps = float(pattern_info.get('score', 0.0) or 0.0)
                    if _ps >= float(self.pattern_strong_threshold):
                        _old = float(min_headroom_rr)
                        min_headroom_rr = max(0.25, float(min_headroom_rr) * (1.0 - float(self.headroom_relax_pct_on_strong)))
                        try:
                            logger.debug(f"{symbol}: relax headroom RR on strong pattern (score={_ps:.2f}): {(_old):.2f} -> {float(min_headroom_rr):.2f}")
                        except Exception:
                            pass
            except Exception:
                pass

            # Headroom filter: require sufficient space to next obstacle
            obstacle = None
            try:
                obstacle = self._next_obstacle_level(bo.type, bo.entry_price, ctx_res, ctx_sup)
            except Exception:
                obstacle = None
            if obstacle is not None:
                # Optionally relax obstacle buffer if strong confirming pattern present
                obstacle_buffer_pips_eff = float(self.obstacle_buffer_pips)
                try:
                    if 'pattern_info' in locals() and pattern_info is not None:
                        _ps2 = float(pattern_info.get('score', 0.0) or 0.0)
                        if _ps2 >= float(self.pattern_strong_threshold):
                            _oldb = float(obstacle_buffer_pips_eff)
                            obstacle_buffer_pips_eff = max(0.0, _oldb * (1.0 - float(self.obstacle_buffer_relax_pct_on_strong)))
                            try:
                                logger.debug(f"{symbol}: relax obstacle buffer on strong pattern: {_oldb:.2f} -> {float(obstacle_buffer_pips_eff):.2f} pips")
                            except Exception:
                                pass
                except Exception:
                    pass
                buf = float(obstacle_buffer_pips_eff) * float(pip)
                headroom_price = max(0.0, abs(obstacle - bo.entry_price) - buf)
                headroom_pips = headroom_price / float(pip)
                try:
                    logger.debug(f"{symbol}: obstacle={obstacle:.5f} headroom_pips={headroom_pips:.1f} sl_pips={sl_pips:.1f}")
                except Exception:
                    pass
                if headroom_pips < (float(min_headroom_rr) * sl_pips):
                    logger.debug(f"{symbol}: headroom {headroom_pips:.1f}p < {min_headroom_rr:.2f}×SL {sl_pips:.1f}p; skipping")
                    return None

            # RR and cap for TP
            rr_eff = float(rr)
            if max_rr_cap is not None:
                try:
                    rr_eff = min(rr_eff, float(max_rr_cap))
                except Exception:
                    pass
            tp = self._take_profit(bo.entry_price, sl, rr_eff, bo.type)

            # We no longer cap TP to the obstacle; headroom is used as a filter above

            # Round to precision
            point = getattr(info, 'point', None)
            digits = getattr(info, 'digits', None)
            if point and digits is not None:
                sl = round(round(sl / point) * point, int(digits))
                tp = round(round(tp / point) * point, int(digits))

            # Recompute SL pips after rounding to precision
            sl_pips = abs(bo.entry_price - sl) / pip
            if sl_pips <= 0:
                return None

            try:
                # Provide visibility into context decisioning
                logger.debug(f"{symbol}: sl_pips={sl_pips:.2f} rr_cfg={rr:.2f} rr_eff={rr_eff:.2f} tp={tp:.5f}")
            except Exception:
                logger.debug(f"{symbol}: sl_pips={sl_pips:.2f} tp={tp:.5f}")

            signal = TradingSignal(
                type=0 if bo.type == 'bullish' else 1,
                entry_price=bo.entry_price,
                stop_loss=sl,
                take_profit=tp,
                stop_loss_pips=sl_pips,
                reason=f"core_{bo.type}_breakout",
                confidence=0.5,
                timestamp=datetime.now(timezone.utc),
                breakout_level=bo.level,
                pattern_score=(pattern_info.get('score') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                pattern_dir=(pattern_info.get('dir') if pattern_info else None) if isinstance(pattern_info, dict) else None,
                pattern_primary=(pattern_info.get('primary') if pattern_info else None) if isinstance(pattern_info, dict) else None,
            )
            # Enriched INFO log for terminal: includes pattern/ATR source if available
            try:
                pat_label = None
                pat_score = None
                if 'pattern_info' in locals() and isinstance(pattern_info, dict):
                    pat_label = pattern_info.get('primary') or pattern_info.get('dir')
                    ps = pattern_info.get('score')
                    pat_score = f"{float(ps):.2f}" if ps is not None else None
                parts = [
                    f"CORE SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'}",
                    f"ref @ {signal.entry_price:.5f}",
                    f"SL {signal.stop_loss:.5f}",
                    f"TP {signal.take_profit:.5f}",
                    f"({sl_pips:.1f}p, RR={rr:.2f})",
                ]
                if pat_label and pat_score:
                    parts.append(f"pat={pat_label}:{pat_score}")
                parts.append(f"ATR={str(self.atr_source).upper()}")
                logger.info(" ".join(parts))
            except Exception:
                logger.info(
                    f"CORE SIGNAL {symbol} {'BUY' if signal.type==0 else 'SELL'} ref @ {signal.entry_price:.5f} "
                    f"SL {signal.stop_loss:.5f} TP {signal.take_profit:.5f} ({sl_pips:.1f}p, RR={rr:.2f})"
                )
            return signal
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}", exc_info=True)
            return None
