from __future__ import annotations

"""
Liquidity Context Module
------------------------
Purpose
    Provide short-term market context for stop-loss (SL) placement by detecting
    nearby liquidity pools (equal highs/lows) and recent liquidity sweeps, and
    emitting simple, deterministic guidance for SL adjustment or trade skip.

Design Principles
    - Pure functions; no side effects. Easy to unit test.
    - Works in two modes: "research" (can look ahead for sweep detection) and
      "realtime" (confirms sweeps only on closed bars; no lookahead).
    - Simple, auditable outputs (booleans, pips, indices, reasons).

Dependencies
    numpy, pandas

Typical Integration (advisory/logging-only phase)
    from liquidity_context import build_liquidity_context

    ctx = build_liquidity_context(
        ohlc=df,  # DataFrame with columns ['open','high','low','close']
        swing_highs_lows=shl_df,  # DataFrame with columns ['HighLow','Level']
        side='long' or 'short',
        entry_price=entry,
        broken_level=level,
        structural_sl=structural_sl,
        atr=atr_value,
        pip=pip_size,
        spread_pips=current_spread_pips,
        config=LiquidityConfig(),
        mode='realtime',  # or 'research'
    )

    # Log ctx dict fields for analysis. In enforcement phase, act on ctx.decisions.

Notes
    - This module is intentionally conservative; if caps are exceeded or data is
      insufficient, it recommends skipping rather than forcing unsafe SLs.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Literal, Tuple, Dict
import numpy as np
import pandas as pd

Mode = Literal['realtime', 'research']
Side = Literal['long', 'short']


@dataclass
class LiquidityConfig:
    # Pool detection
    sweep_lookback_bars: int = 40
    recent_sweep_max_age_bars: int = 15
    equal_level_tolerance_pips: Optional[float] = None  # if None, use range_percent
    range_percent: float = 0.01  # fallback tolerance as % of window range
    min_sweep_amplitude_pips: float = 10.0
    wick_to_body_ratio_min: float = 1.5
    use_close_back: bool = True  # require close back inside after poke

    # Cushioning and caps
    cushion_atr_mult: float = 0.6
    cushion_atr_mult_repeat: float = 0.8  # if repeated sweeps on the same side
    min_cushion_pips: float = 4.0
    max_sl_distance_pips: Optional[float] = None  # per-symbol cap; None = no extra cap here

    # Decision toggles
    require_confirmation_on_dual_pools: bool = True


@dataclass
class PoolInfo:
    side: int  # +1 for highs-pool (sell-side liquidity), -1 for lows-pool (buy-side liquidity)
    level_avg: float
    band_low: float
    band_high: float
    start_idx: int
    end_idx: int
    first_swept_idx: Optional[int]
    last_swept_idx: Optional[int]
    sweep_count: int
    swept_extreme_price: Optional[float]


@dataclass
class LiquidityContext:
    # presence & details of pools/sweeps
    has_recent_buy_sweep: bool
    has_recent_sell_sweep: bool
    last_buy_sweep_idx: Optional[int]
    last_sell_sweep_idx: Optional[int]
    last_buy_sweep_low: Optional[float]
    last_sell_sweep_high: Optional[float]
    buy_sweep_count: int
    sell_sweep_count: int

    # proximity / chop
    equal_highs_nearby: bool
    equal_lows_nearby: bool
    dist_to_nearest_pool_pips: Optional[float]

    # retest hints (placeholder for future use)
    retest_confirmed: Optional[bool]
    retest_depth_pips: Optional[float]

    # regime hints (placeholders)
    atr: float
    spread_pips: float

    # decisions
    sl_zone_violation: bool
    sweep_zone_extreme: Optional[float]
    recommended_cushion_pips: float
    should_require_confirmation: bool
    should_skip: bool
    explain: str

    def asdict(self) -> Dict:
        return asdict(self)


def _compute_tolerance(ohlc: pd.DataFrame, pip: float, cfg: LiquidityConfig) -> float:
    if cfg.equal_level_tolerance_pips is not None:
        return cfg.equal_level_tolerance_pips * pip
    rng = float(ohlc['high'].max() - ohlc['low'].min())
    return max(1e-9, rng * cfg.range_percent)


def _wick_body_ratio(o: float, h: float, l: float, c: float, direction: str) -> float:
    body = abs(c - o) + 1e-12
    if direction == 'up':  # upper wick dominance
        wick = h - max(o, c)
    else:  # 'down'
        wick = min(o, c) - l
    return wick / body


def _detect_pools_and_sweeps(
    ohlc: pd.DataFrame,
    shl: pd.DataFrame,
    pip: float,
    cfg: LiquidityConfig,
    mode: Mode,
) -> Tuple[list[PoolInfo], list[PoolInfo]]:
    """Return (highs_pools, lows_pools) with sweep stats.

    shl columns: ['HighLow', 'Level'] where HighLow in {+1 (high), -1 (low)}.
    mode='research' may look ahead; 'realtime' does not confirm future sweeps until seen.
    """
    n = len(ohlc)
    o = ohlc['open'].values
    h = ohlc['high'].values
    l = ohlc['low'].values
    c = ohlc['close'].values

    tol = _compute_tolerance(ohlc, pip, cfg)

    # Work on copies to safely mark consumed candidates within each side
    shl_hl = shl['HighLow'].to_numpy().copy()
    shl_lvl = shl['Level'].to_numpy().copy()

    highs_idx = np.flatnonzero(shl_hl == 1)
    lows_idx = np.flatnonzero(shl_hl == -1)

    highs_pools: list[PoolInfo] = []
    lows_pools: list[PoolInfo] = []

    def build_side_pools(cand_idx: np.ndarray, side_val: int) -> list[PoolInfo]:
        pools: list[PoolInfo] = []
        used = shl_hl.copy()
        for i in cand_idx:
            if used[i] != side_val:
                continue
            base = shl_lvl[i]
            band_low = base - tol
            band_high = base + tol
            group = [base]
            start_idx = i
            end_idx = i

            # Determine sweeps relative to band
            c_start = i + 1
            first_swept_idx = None
            last_swept_idx = None
            sweep_count = 0
            swept_extreme_price = None

            if c_start < n:
                if side_val == 1:
                    # highs pool: a sweep is high >= band_high with close back below band_high (optional)
                    for k in range(c_start, n):
                        hit = h[k] >= band_high
                        if not hit:
                            continue
                        if cfg.use_close_back:
                            if c[k] >= band_high:
                                continue  # not a close-back
                            if _wick_body_ratio(o[k], h[k], l[k], c[k], 'up') < cfg.wick_to_body_ratio_min:
                                continue
                        # amplitude check vs tolerance (approximate pips)
                        amp_ok = (h[k] - band_high) / pip >= max(0.0, cfg.min_sweep_amplitude_pips)
                        if not amp_ok:
                            continue
                        first_swept_idx = k if first_swept_idx is None else first_swept_idx
                        last_swept_idx = k
                        sweep_count += 1
                        swept_extreme_price = h[k] if (swept_extreme_price is None or h[k] > swept_extreme_price) else swept_extreme_price
                        if mode == 'realtime':
                            break  # confirm on first occurrence only
                else:
                    # lows pool: a sweep is low <= band_low with close back above band_low (optional)
                    for k in range(c_start, n):
                        hit = l[k] <= band_low
                        if not hit:
                            continue
                        if cfg.use_close_back:
                            if c[k] <= band_low:
                                continue
                            if _wick_body_ratio(o[k], h[k], l[k], c[k], 'down') < cfg.wick_to_body_ratio_min:
                                continue
                        amp_ok = (band_low - l[k]) / pip >= max(0.0, cfg.min_sweep_amplitude_pips)
                        if not amp_ok:
                            continue
                        first_swept_idx = k if first_swept_idx is None else first_swept_idx
                        last_swept_idx = k
                        sweep_count += 1
                        swept_extreme_price = l[k] if (swept_extreme_price is None or l[k] < swept_extreme_price) else swept_extreme_price
                        if mode == 'realtime':
                            break

            # Group equal levels until (optionally) the first sweep
            for j in cand_idx:
                if j <= i:
                    continue
                if first_swept_idx is not None and j >= first_swept_idx:
                    break
                if used[j] == side_val and (band_low <= shl_lvl[j] <= band_high):
                    group.append(shl_lvl[j])
                    end_idx = j
                    used[j] = 0  # consume

            if len(group) > 1:
                level_avg = float(np.mean(group))
                pools.append(PoolInfo(
                    side=side_val,
                    level_avg=level_avg,
                    band_low=float(band_low),
                    band_high=float(band_high),
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    first_swept_idx=None if first_swept_idx is None else int(first_swept_idx),
                    last_swept_idx=None if last_swept_idx is None else int(last_swept_idx),
                    sweep_count=int(sweep_count),
                    swept_extreme_price=None if swept_extreme_price is None else float(swept_extreme_price),
                ))
        return pools

    highs_pools = build_side_pools(highs_idx, 1)
    lows_pools = build_side_pools(lows_idx, -1)
    return highs_pools, lows_pools


def _nearest_pool_info(
    pools: list[PoolInfo],
    price: float,
    pip: float,
) -> Tuple[Optional[PoolInfo], Optional[float]]:
    if not pools:
        return None, None
    distances = []
    for p in pools:
        # distance to the band
        if price > p.band_high:
            d = price - p.band_high
        elif price < p.band_low:
            d = p.band_low - price
        else:
            d = 0.0
        distances.append(d)
    idx = int(np.argmin(distances))
    nearest = pools[idx]
    dist_pips = distances[idx] / pip if pip else None
    return nearest, None if dist_pips is None else float(dist_pips)


def build_liquidity_context(
    ohlc: pd.DataFrame,
    swing_highs_lows: pd.DataFrame,
    side: Side,
    entry_price: float,
    broken_level: float,
    structural_sl: float,
    atr: float,
    pip: float,
    spread_pips: float,
    config: Optional[LiquidityConfig] = None,
    mode: Mode = 'realtime',
) -> LiquidityContext:
    """Compute a LiquidityContext for the current decision.

    Returns a context with advisory decisions for SL placement and optional skip.
    """
    cfg = config or LiquidityConfig()

    highs_pools, lows_pools = _detect_pools_and_sweeps(
        ohlc=ohlc, shl=swing_highs_lows, pip=pip, cfg=cfg, mode=mode
    )

    # Choose pool side relevant to the stop
    # Long -> risk below (lows pools), Short -> risk above (highs pools)
    relevant_pools = lows_pools if side == 'long' else highs_pools
    other_side_pools = highs_pools if side == 'long' else lows_pools

    # Determine recent sweep stats
    last_buy_sweep_idx = None
    last_sell_sweep_idx = None
    last_buy_sweep_low = None
    last_sell_sweep_high = None
    buy_sweep_count = 0
    sell_sweep_count = 0

    for p in lows_pools:
        if p.last_swept_idx is not None:
            last_buy_sweep_idx = p.last_swept_idx if (last_buy_sweep_idx is None or p.last_swept_idx > last_buy_sweep_idx) else last_buy_sweep_idx
            if p.swept_extreme_price is not None:
                last_buy_sweep_low = p.swept_extreme_price if (last_buy_sweep_low is None or p.swept_extreme_price < last_buy_sweep_low) else last_buy_sweep_low
            buy_sweep_count += p.sweep_count

    for p in highs_pools:
        if p.last_swept_idx is not None:
            last_sell_sweep_idx = p.last_swept_idx if (last_sell_sweep_idx is None or p.last_swept_idx > last_sell_sweep_idx) else last_sell_sweep_idx
            if p.swept_extreme_price is not None:
                last_sell_sweep_high = p.swept_extreme_price if (last_sell_sweep_high is None or p.swept_extreme_price > last_sell_sweep_high) else last_sell_sweep_high
            sell_sweep_count += p.sweep_count

    # Nearby pool distances (for chop awareness)
    nearest_rel_pool, dist_to_nearest_pool_pips = _nearest_pool_info(relevant_pools, broken_level, pip)
    nearest_oth_pool, _ = _nearest_pool_info(other_side_pools, broken_level, pip)

    equal_highs_nearby = nearest_oth_pool is not None if side == 'long' else nearest_rel_pool is not None
    equal_lows_nearby = nearest_rel_pool is not None if side == 'long' else nearest_oth_pool is not None

    # Sweep-zone logic
    sl_zone_violation = False
    sweep_zone_extreme = None

    if side == 'long':
        # zone between broken level and last swept low (or active pool band low if unswept)
        if last_buy_sweep_low is not None:
            zone_low = last_buy_sweep_low
        elif nearest_rel_pool is not None:
            zone_low = nearest_rel_pool.band_low
        else:
            zone_low = None
        if zone_low is not None:
            zone_high = broken_level
            if structural_sl > zone_low and structural_sl < zone_high:
                sl_zone_violation = True
                sweep_zone_extreme = zone_low
    else:  # short
        if last_sell_sweep_high is not None:
            zone_high = last_sell_sweep_high
        elif nearest_rel_pool is not None:
            zone_high = nearest_rel_pool.band_high
        else:
            zone_high = None
        if zone_high is not None:
            zone_low = broken_level
            if structural_sl < zone_high and structural_sl > zone_low:
                sl_zone_violation = True
                sweep_zone_extreme = zone_high

    # Cushion recommendation
    base_k = cfg.cushion_atr_mult
    if (side == 'long' and buy_sweep_count >= 2) or (side == 'short' and sell_sweep_count >= 2):
        base_k = cfg.cushion_atr_mult_repeat
    recommended_cushion_pips = max(cfg.min_cushion_pips, (atr / pip) * base_k if pip else cfg.min_cushion_pips)

    # Require confirmation if both sides have pools near broken level
    should_require_confirmation = cfg.require_confirmation_on_dual_pools and (nearest_rel_pool is not None and nearest_oth_pool is not None)

    # Skip logic only if pushing SL beyond sweep extreme would exceed caps (cap check is done by caller with exact numbers)
    should_skip = False  # advisory here; caller decides after computing final SL

    explain_parts = []
    if sl_zone_violation:
        explain_parts.append('SL is inside sweep zone; recommend placing beyond the sweep extreme with cushion.')
    if should_require_confirmation:
        explain_parts.append('Both sides show nearby liquidity pools; confirmation recommended to avoid chop.')
    if not explain_parts:
        explain_parts.append('No sweep-zone violation detected.')

    ctx = LiquidityContext(
        has_recent_buy_sweep=last_buy_sweep_idx is not None,
        has_recent_sell_sweep=last_sell_sweep_idx is not None,
        last_buy_sweep_idx=last_buy_sweep_idx,
        last_sell_sweep_idx=last_sell_sweep_idx,
        last_buy_sweep_low=last_buy_sweep_low,
        last_sell_sweep_high=last_sell_sweep_high,
        buy_sweep_count=int(buy_sweep_count),
        sell_sweep_count=int(sell_sweep_count),
        equal_highs_nearby=bool(equal_highs_nearby),
        equal_lows_nearby=bool(equal_lows_nearby),
        dist_to_nearest_pool_pips=None if dist_to_nearest_pool_pips is None else float(dist_to_nearest_pool_pips),
        retest_confirmed=None,
        retest_depth_pips=None,
        atr=float(atr),
        spread_pips=float(spread_pips),
        sl_zone_violation=bool(sl_zone_violation),
        sweep_zone_extreme=None if sweep_zone_extreme is None else float(sweep_zone_extreme),
        recommended_cushion_pips=float(recommended_cushion_pips),
        should_require_confirmation=bool(should_require_confirmation),
        should_skip=bool(should_skip),
        explain=' '.join(explain_parts),
    )
    return ctx


# ------- Convenience API for callers that want plain dicts ------------

def build_context_dict(*args, **kwargs) -> Dict:
    return build_liquidity_context(*args, **kwargs).asdict()
