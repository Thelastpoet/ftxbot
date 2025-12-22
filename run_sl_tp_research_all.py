"""
Run SL/TP research for all symbols and compile results.

Usage:
    python run_sl_tp_research_all.py
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from sl_tp_research import run_research, fetch_data, get_pip_size, calculate_atr, calculate_ema
from sl_tp_research import find_swing_points, calculate_sr_levels, detect_breakout
from sl_tp_research import analyze_breakout_outcomes, BreakoutOutcome

# All symbols from backtest
SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
    'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY',
    'EURGBP', 'EURAUD', 'EURCAD', 'EURCHF',
    'GBPAUD', 'GBPCAD', 'GBPCHF',
    'AUDNZD', 'AUDCAD',
    'XAUUSD'
]

def run_all_symbols(start: str, end: str, atr_mult: float = 0.3, use_trend: bool = True):
    """Run research for all symbols and compile results."""

    if not mt5 or not mt5.initialize():
        print("ERROR: MT5 not available")
        return

    all_results = {}
    all_price_stats = {}

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    lookback_days = 20
    data_start = start_dt - timedelta(days=lookback_days)

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}...")
        print('='*60)

        pip = get_pip_size(symbol)

        # Fetch data
        data = fetch_data(symbol, "M15", data_start, end_dt)
        if data is None:
            print(f"  SKIPPED - No data for {symbol}")
            continue

        # Calculate indicators
        data['atr'] = calculate_atr(data, 14)
        data['ema'] = calculate_ema(data['close'], 200)
        data = data[data.index >= start_dt]

        outcomes = []
        last_breakout_bar = {}

        # Find breakouts
        for i in range(50, len(data) - 100):
            analysis_data = data.iloc[i-30:i]

            current_atr = data.iloc[i-1]['atr']
            if pd.isna(current_atr):
                continue

            # Trend filter
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
            threshold = atr_mult * current_atr

            breakout = detect_breakout(last_close, resistance, support, threshold)
            if not breakout:
                continue

            direction, level = breakout

            if direction == 'bullish' and not trend_bullish:
                continue
            if direction == 'bearish' and not trend_bearish:
                continue

            bar_time = analysis_data.index[-1]
            key = (direction,)
            if last_breakout_bar.get(key) == bar_time:
                continue
            last_breakout_bar[key] = bar_time

            # Session filter
            hour = bar_time.hour
            if not (8 <= hour < 17):
                continue

            outcome = analyze_breakout_outcomes(data, i-1, direction, last_close, pip)
            outcomes.append(outcome)

        print(f"  Breakouts analyzed: {len(outcomes)}")

        if len(outcomes) < 10:
            print(f"  SKIPPED - Too few breakouts")
            continue

        # Compile stats for this symbol
        sl_levels = [10, 15, 20, 25, 30, 35, 40]
        tp_levels = [15, 20, 25, 30, 40, 50, 60]

        symbol_results = {}
        for sl in sl_levels:
            for tp in tp_levels:
                key = f"{sl}_{tp}"
                wins = sum(1 for o in outcomes if o.outcomes.get(key) == "WIN")
                losses = sum(1 for o in outcomes if o.outcomes.get(key) == "LOSS")

                total_resolved = wins + losses
                win_rate = wins / total_resolved if total_resolved > 0 else 0
                rr = tp / sl
                ev = (win_rate * rr) - ((1 - win_rate) * 1) if total_resolved > 0 else 0

                gross_profit = wins * tp
                gross_loss = losses * sl
                pf = gross_profit / gross_loss if gross_loss > 0 else 0

                symbol_results[key] = {
                    "sl": sl, "tp": tp, "rr": round(rr, 2),
                    "wins": wins, "losses": losses,
                    "total_resolved": total_resolved,
                    "win_rate": round(win_rate * 100, 1),
                    "ev_per_trade": round(ev, 3),
                    "profit_factor": round(pf, 2),
                    "net_pips": round(wins * tp - losses * sl, 1),
                }

        # Price behavior stats
        favorable_pips = [o.max_favorable_pips for o in outcomes]
        adverse_pips = [o.max_adverse_pips for o in outcomes]

        all_price_stats[symbol] = {
            "breakouts": len(outcomes),
            "avg_mfe": round(np.mean(favorable_pips), 1),
            "median_mfe": round(np.median(favorable_pips), 1),
            "p75_mfe": round(np.percentile(favorable_pips, 75), 1),
            "avg_mae": round(np.mean(adverse_pips), 1),
            "median_mae": round(np.median(adverse_pips), 1),
            "p75_mae": round(np.percentile(adverse_pips, 75), 1),
        }

        all_results[symbol] = symbol_results

        # Print top 3 for this symbol
        sorted_res = sorted(symbol_results.items(), key=lambda x: x[1]['ev_per_trade'], reverse=True)
        print(f"  Top 3 SL/TP combos:")
        for k, r in sorted_res[:3]:
            print(f"    SL={r['sl']}, TP={r['tp']} -> WR={r['win_rate']}%, EV={r['ev_per_trade']}, PF={r['profit_factor']}")

    mt5.shutdown()

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY: PRICE BEHAVIOR BY SYMBOL (MFE = Max Favorable, MAE = Max Adverse)")
    print("="*100)
    print(f"{'Symbol':<10} {'Breakouts':<10} {'AvgMFE':<10} {'MedianMFE':<12} {'P75MFE':<10} {'AvgMAE':<10} {'MedianMAE':<12} {'P75MAE':<10}")
    print("-"*100)

    for sym, stats in sorted(all_price_stats.items(), key=lambda x: x[1]['avg_mfe'], reverse=True):
        print(f"{sym:<10} {stats['breakouts']:<10} {stats['avg_mfe']:<10} {stats['median_mfe']:<12} {stats['p75_mfe']:<10} {stats['avg_mae']:<10} {stats['median_mae']:<12} {stats['p75_mae']:<10}")

    # Find optimal SL/TP per symbol
    print("\n" + "="*100)
    print("OPTIMAL SL/TP PER SYMBOL (by Expected Value)")
    print("="*100)
    print(f"{'Symbol':<10} {'Best SL':<8} {'Best TP':<8} {'R:R':<6} {'WinRate':<10} {'EV':<8} {'PF':<8} {'vs Current':<12}")
    print("-"*100)

    current_sl = 20  # approximate current settings
    current_tp = 30  # 1.5 RR

    for sym in SYMBOLS:
        if sym not in all_results:
            continue

        res = all_results[sym]
        sorted_res = sorted(res.items(), key=lambda x: x[1]['ev_per_trade'], reverse=True)
        best = sorted_res[0][1]

        # Get current SL/TP performance
        current_key = f"{current_sl}_{current_tp}"
        current_ev = res.get(current_key, {}).get('ev_per_trade', 0)

        diff = best['ev_per_trade'] - current_ev
        diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"

        print(f"{sym:<10} {best['sl']:<8} {best['tp']:<8} {best['rr']:<6} {best['win_rate']:<9}% {best['ev_per_trade']:<8} {best['profit_factor']:<8} {diff_str:<12}")

    # Save all results
    output_dir = Path("sl_tp_research_results")
    output_dir.mkdir(exist_ok=True)

    output = {
        "price_behavior": all_price_stats,
        "sl_tp_results": all_results,
    }

    # Save combined results
    output_file = output_dir / "all_symbols_summary.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Save per-symbol results for easier analysis
    for sym, res in all_results.items():
        sym_file = output_dir / f"{sym}_sl_tp.json"
        sym_output = {
            "symbol": sym,
            "price_behavior": all_price_stats.get(sym, {}),
            "sl_tp_results": res,
            "top_10": sorted(res.items(), key=lambda x: x[1]['ev_per_trade'], reverse=True)[:10]
        }
        with open(sym_file, "w") as f:
            json.dump(sym_output, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    # Use same date range as backtest
    run_all_symbols("2024-01-01", "2024-12-22")
