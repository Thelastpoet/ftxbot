"""
SL/TP Optimization Research

Tests different SL and TP combinations to find optimal values
based on actual price behavior after breakouts.

Usage:
    python sl_tp_research.py --symbol EURUSD --start 2023-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BreakoutOutcome:
    """Tracks price behavior after a breakout."""
    direction: str
    entry_price: float
    max_favorable_pips: float
    max_adverse_pips: float
    session: str
    hour_utc: int

    # Outcome at different SL/TP levels
    outcomes: Dict[str, str] = None  # key: "SL_TP", value: "WIN"/"LOSS"/"NEITHER"


def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime):
    """Fetch data from MT5."""
    if not mt5 or not mt5.initialize():
        logger.error("MT5 not available")
        return None

    tf_map = {
        'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
    }

    tf = tf_map.get(timeframe.upper())
    if not mt5.symbol_select(symbol, True):
        return None

    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)

    logger.info(f"Fetched {len(df)} bars")
    return df


def get_pip_size(symbol: str) -> float:
    if 'JPY' in symbol.upper():
        return 0.01
    return 0.0001


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int = 200) -> pd.Series:
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()


def find_swing_points(data: pd.DataFrame, window: int = 5):
    """Find swing highs and lows."""
    highs = data['high'].values
    lows = data['low'].values
    n = len(data)

    swing_highs, swing_lows = [], []

    for i in range(window, n - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append(i)
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append(i)

    return swing_highs, swing_lows


def calculate_sr_levels(data: pd.DataFrame, swing_highs: List[int],
                        swing_lows: List[int], pip: float):
    """Calculate S/R levels."""
    proximity = 10 * pip
    resistance, support = [], []

    for i in reversed(swing_highs[-30:]):
        level = data.iloc[i]['high']
        if not any(abs(level - x) <= proximity for x in resistance):
            resistance.append(level)
        if len(resistance) >= 3:
            break

    for i in reversed(swing_lows[-30:]):
        level = data.iloc[i]['low']
        if not any(abs(level - x) <= proximity for x in support):
            support.append(level)
        if len(support) >= 3:
            break

    return resistance, support


def detect_breakout(close: float, resistance: List[float],
                    support: List[float], threshold: float):
    """Detect breakout."""
    for r in resistance:
        if close > r + threshold:
            return ('bullish', r)
    for s in support:
        if close < s - threshold:
            return ('bearish', s)
    return None


def get_session(hour: int) -> str:
    if 8 <= hour < 12:
        return "LONDON"
    elif 12 <= hour < 17:
        return "OVERLAP"
    elif 17 <= hour < 21:
        return "NY_LATE"
    return "OTHER"


def analyze_breakout_outcomes(data: pd.DataFrame, breakout_idx: int,
                               direction: str, entry_price: float,
                               pip: float, max_bars: int = 100) -> BreakoutOutcome:
    """Analyze what happens after a breakout at different SL/TP levels."""

    bar = data.iloc[breakout_idx]
    outcome = BreakoutOutcome(
        direction=direction,
        entry_price=entry_price,
        max_favorable_pips=0,
        max_adverse_pips=0,
        session=get_session(bar.name.hour),
        hour_utc=bar.name.hour,
        outcomes={}
    )

    # Track price extremes
    max_high = entry_price
    min_low = entry_price

    # SL/TP combinations to test
    sl_levels = [10, 15, 20, 25, 30, 35, 40]  # pips
    tp_levels = [15, 20, 25, 30, 40, 50, 60]  # pips

    # Track which SL/TP combos are still open
    open_trades = {}
    for sl in sl_levels:
        for tp in tp_levels:
            key = f"{sl}_{tp}"
            open_trades[key] = True

    # Walk forward
    for i in range(breakout_idx + 1, min(breakout_idx + max_bars, len(data))):
        future_bar = data.iloc[i]
        max_high = max(max_high, future_bar['high'])
        min_low = min(min_low, future_bar['low'])

        # Check each SL/TP combo
        for sl in sl_levels:
            for tp in tp_levels:
                key = f"{sl}_{tp}"
                if not open_trades[key]:
                    continue

                sl_price = entry_price - sl * pip if direction == 'bullish' else entry_price + sl * pip
                tp_price = entry_price + tp * pip if direction == 'bullish' else entry_price - tp * pip

                if direction == 'bullish':
                    if future_bar['low'] <= sl_price:
                        outcome.outcomes[key] = "LOSS"
                        open_trades[key] = False
                    elif future_bar['high'] >= tp_price:
                        outcome.outcomes[key] = "WIN"
                        open_trades[key] = False
                else:
                    if future_bar['high'] >= sl_price:
                        outcome.outcomes[key] = "LOSS"
                        open_trades[key] = False
                    elif future_bar['low'] <= tp_price:
                        outcome.outcomes[key] = "WIN"
                        open_trades[key] = False

    # Mark remaining as neither
    for key, still_open in open_trades.items():
        if still_open:
            outcome.outcomes[key] = "NEITHER"

    # Calculate max favorable/adverse
    if direction == 'bullish':
        outcome.max_favorable_pips = (max_high - entry_price) / pip
        outcome.max_adverse_pips = (entry_price - min_low) / pip
    else:
        outcome.max_favorable_pips = (entry_price - min_low) / pip
        outcome.max_adverse_pips = (max_high - entry_price) / pip

    return outcome


def run_research(symbol: str, start: str, end: str, atr_mult: float = 0.3, use_trend: bool = True):
    """Run SL/TP optimization research."""

    pip = get_pip_size(symbol)

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Fetch extra data for indicators
    lookback_days = 20
    data_start = start_dt - timedelta(days=lookback_days)
    
    # Use H1 for trend filter if possible, else use M15
    data = fetch_data(symbol, "M15", data_start, end_dt)
    if data is None:
        return

    # Calculate Indicators
    data['atr'] = calculate_atr(data, 14)
    data['ema'] = calculate_ema(data['close'], 200)

    # Trim to requested start date
    data = data[data.index >= start_dt]

    outcomes: List[BreakoutOutcome] = []
    last_breakout_bar = {}

    # Find breakouts and analyze outcomes
    logger.info("Scanning for breakouts...")

    for i in range(50, len(data) - 100):
        analysis_data = data.iloc[i-30:i]
        
        # Get pre-calculated indicators
        current_atr = data.iloc[i-1]['atr']
        if pd.isna(current_atr):
            continue
            
        # Trend Filter
        if use_trend:
            current_close = data.iloc[i-1]['close']
            current_ema = data.iloc[i-1]['ema']
            if pd.isna(current_ema):
                continue
            
            trend_bullish = current_close > current_ema
            trend_bearish = current_close < current_ema
        else:
            trend_bullish = True
            trend_bearish = True

        swing_highs, swing_lows = find_swing_points(analysis_data, 5)
        if not swing_highs and not swing_lows:
            continue

        resistance, support = calculate_sr_levels(analysis_data, swing_highs, swing_lows, pip)

        last_close = analysis_data.iloc[-1]['close']
        
        # ATR-based threshold
        threshold = atr_mult * current_atr

        breakout = detect_breakout(last_close, resistance, support, threshold)
        if not breakout:
            continue

        direction, level = breakout
        
        # Apply Trend Filter
        if direction == 'bullish' and not trend_bullish:
            continue
        if direction == 'bearish' and not trend_bearish:
            continue

        # Duplicate prevention
        bar_time = analysis_data.index[-1]
        key = (direction,)
        if last_breakout_bar.get(key) == bar_time:
            continue
        last_breakout_bar[key] = bar_time

        # Session filter (only analyze London/NY)
        hour = bar_time.hour
        if not (8 <= hour < 17):
            continue

        # Analyze this breakout
        outcome = analyze_breakout_outcomes(data, i-1, direction, last_close, pip)
        outcomes.append(outcome)

    logger.info(f"Analyzed {len(outcomes)} breakouts")

    # Compile statistics
    sl_levels = [10, 15, 20, 25, 30, 35, 40]
    tp_levels = [15, 20, 25, 30, 40, 50, 60]

    results = {}

    for sl in sl_levels:
        for tp in tp_levels:
            key = f"{sl}_{tp}"
            wins = sum(1 for o in outcomes if o.outcomes.get(key) == "WIN")
            losses = sum(1 for o in outcomes if o.outcomes.get(key) == "LOSS")
            neither = sum(1 for o in outcomes if o.outcomes.get(key) == "NEITHER")

            total_resolved = wins + losses
            win_rate = wins / total_resolved if total_resolved > 0 else 0

            rr = tp / sl

            # Expected value per trade (in R)
            ev = (win_rate * rr) - ((1 - win_rate) * 1) if total_resolved > 0 else 0

            # Profit factor
            gross_profit = wins * tp
            gross_loss = losses * sl
            pf = gross_profit / gross_loss if gross_loss > 0 else 0

            results[key] = {
                "sl": sl,
                "tp": tp,
                "rr": round(rr, 2),
                "wins": wins,
                "losses": losses,
                "neither": neither,
                "total_resolved": total_resolved,
                "win_rate": round(win_rate * 100, 1),
                "ev_per_trade": round(ev, 3),
                "profit_factor": round(pf, 2),
                "net_pips": round(wins * tp - losses * sl, 1),
            }

    # Price behavior stats
    favorable_pips = [o.max_favorable_pips for o in outcomes]
    adverse_pips = [o.max_adverse_pips for o in outcomes]

    price_stats = {
        "avg_max_favorable": round(np.mean(favorable_pips), 1),
        "median_max_favorable": round(np.median(favorable_pips), 1),
        "p75_max_favorable": round(np.percentile(favorable_pips, 75), 1),
        "avg_max_adverse": round(np.mean(adverse_pips), 1),
        "median_max_adverse": round(np.median(adverse_pips), 1),
        "p75_max_adverse": round(np.percentile(adverse_pips, 75), 1),
    }

    # Print results
    print("\n" + "=" * 80)
    print("SL/TP OPTIMIZATION RESEARCH")
    print("=" * 80)

    print(f"\nBreakouts Analyzed: {len(outcomes)}")
    print(f"\nPrice Behavior After Breakout:")
    print(f"  Avg Max Favorable: {price_stats['avg_max_favorable']} pips")
    print(f"  Median Max Favorable: {price_stats['median_max_favorable']} pips")
    print(f"  75th Percentile Favorable: {price_stats['p75_max_favorable']} pips")
    print(f"  Avg Max Adverse: {price_stats['avg_max_adverse']} pips")
    print(f"  Median Max Adverse: {price_stats['median_max_adverse']} pips")
    print(f"  75th Percentile Adverse: {price_stats['p75_max_adverse']} pips")

    print("\n" + "-" * 80)
    print("TOP 15 SL/TP COMBINATIONS (by Expected Value)")
    print("-" * 80)
    print(f"{'SL':>4} {'TP':>4} {'R:R':>5} {'WinRate':>8} {'EV/Trade':>9} {'PF':>6} {'NetPips':>10} {'Trades':>7}")
    print("-" * 80)

    # Sort by EV
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ev_per_trade'], reverse=True)

    for key, r in sorted_results[:15]:
        print(f"{r['sl']:>4} {r['tp']:>4} {r['rr']:>5} {r['win_rate']:>7}% {r['ev_per_trade']:>9} {r['profit_factor']:>6} {r['net_pips']:>10} {r['total_resolved']:>7}")

    print("\n" + "-" * 80)
    print("WORST 5 SL/TP COMBINATIONS")
    print("-" * 80)

    for key, r in sorted_results[-5:]:
        print(f"{r['sl']:>4} {r['tp']:>4} {r['rr']:>5} {r['win_rate']:>7}% {r['ev_per_trade']:>9} {r['profit_factor']:>6} {r['net_pips']:>10} {r['total_resolved']:>7}")

    print("\n" + "=" * 80)

    # Save full results
    output = {
        "breakouts_analyzed": len(outcomes),
        "price_behavior": price_stats,
        "sl_tp_results": results,
        "top_10": [{"key": k, **v} for k, v in sorted_results[:10]],
    }

    with open("sl_tp_research_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to sl_tp_research_results.json")

    if mt5:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(description='SL/TP Optimization Research')
    parser.add_argument('--symbol', type=str, default='EURUSD')
    parser.add_argument('--start', type=str, default='2023-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--atr', type=float, default=0.3, help='ATR Multiplier for threshold')
    parser.add_argument('--no-trend', action='store_true', help='Disable Trend Filter')
    args = parser.parse_args()

    run_research(args.symbol, args.start, args.end, args.atr, not args.no_trend)


if __name__ == "__main__":
    main()