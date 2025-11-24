## Overview

This bot trades FX and metals breakouts off recent swing structure on the trading timeframe (default M15) while using higher-timeframe (e.g., H1) candlestick patterns to enforce directional bias. Breakouts trigger immediately from the trading timeframe as soon as closes clear S/R thresholds, so execution stays fast even on XAUUSD spikes.

## Candlestick Confirmation

- **Trading timeframe (entry timing)**: Disabled by default to avoid waiting for low-quality, low-timeframe patterns. Enable by setting `pattern_use_trading_tf` to `true` in `trading_settings` if you explicitly want same-timeframe candlestick gates.
- **Higher timeframe (bias filter)**: Controlled via `htf_pattern_timeframe`. When `require_htf_pattern_alignment` is true, trades only proceed if TA-Lib detects a confirming pattern on that HTF (e.g., H1) and its direction matches the breakout. Missing HTF data or opposite signals block trades to avoid fakeouts.

Key knobs under `trading_settings`:

```json
"enable_patterns": true,
"pattern_use_trading_tf": false,
"htf_pattern_timeframe": "H1",
"htf_pattern_window": 3,
"htf_pattern_score_threshold": 0.6,
"require_htf_pattern_alignment": true
```

Ensure each symbol’s `timeframes` array includes the HTF (e.g., `["M15","H1"]`) so the bot downloads those candles for bias confirmation.

## Historical Analysis Module

`history_analysis.py` introduces a standalone `HistoricalAnalyzer` that can be wired in later to enrich signals with ADR exhaustion, Donchian channel regimes, macro highs/lows, and breakout hit-rate stats per symbol. It pulls defaults from `config.json -> historical_analysis` and merges any `symbols[].historical_analysis` overrides (e.g., unique ADR thresholds for metals vs. majors).

Example defaults:

```json
"historical_analysis": {
  "enabled": true,
  "max_data_age_minutes": 180,
  "refresh_interval_minutes": 60,
  "intraday_timeframe": "M15",
  "intraday_bars": 200,
  "adr_window": 14,
  "adr_exhaustion_pct": 0.9,
  "block_on_adr_exhaustion": true,
  "donchian_window": 55,
  "trend_threshold_pips": 0.2,
  "enforce_trend_alignment": true,
  "breakout_lookback_bars": 180,
  "min_breakout_success_rate": 0.45
}
```

Usage sketch (until it's wired into the main loop):

```python
from history_analysis import HistoricalAnalyzer

analyzer = HistoricalAnalyzer(config_dict)
snapshot = analyzer.refresh_symbol(
    symbol="EURUSD",
    fetcher=lambda sym, tf, bars: market_data.fetch_data_sync(sym, tf, bars),
    pip_size=0.0001,
    intraday_df=current_df,
)
```

The caller can then inspect `snapshot.adr_exhausted`, `snapshot.trend_bias`, etc., before handing off to the strategy. Wiring is intentionally deferred to keep the current live logic unchanged.

When enabled (defaults shown above), the live strategy automatically blocks trades when the historical snapshot reports ADR exhaustion, opposing higher-timeframe trend bias, or a breakout success rate below `min_breakout_success_rate`. Snapshots are refreshed roughly every `refresh_interval_minutes` per symbol, and per-symbol overrides can tweak those thresholds under `symbols[].historical_analysis`. You can also supply a dedicated `intraday_timeframe` (and bar count) so ADR progress uses a consistent session feed even if the trading timeframe differs.
