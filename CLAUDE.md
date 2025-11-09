# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based automated Forex trading bot that connects to MetaTrader 5 (MT5) to execute price-action breakout trades. The bot uses swing-level detection, candlestick pattern confirmation via TA-Lib, and multi-timeframe context to generate trading signals with dynamic stop-loss and take-profit levels.

**Key Technologies**: Python 3.13, MetaTrader5 Python API, pandas, TA-Lib, asyncio

## Running the Bot

### Main Entry Point
```bash
python main.py
```

### Command-Line Options
```bash
# Custom config file
python main.py --config config.json

# Override risk per trade
python main.py --risk-per-trade 0.02

# Trade specific symbol
python main.py --symbol EURUSD --timeframe M15

# Set logging level
python main.py --log-level DEBUG
```

### Configuration
- Main config: `config.json`
- Logs are written to: `forex_bot.log` (full debug) and console (configurable level)
- Trade history: `trades.log`, `trades.json`, `trades.csv`

## Architecture

### Core Flow
The bot operates in an async event loop (`main.py:473-487`):
1. **Initialize** MT5 connection and components (`main.py:174-229`)
2. **Monitor** open trades for closure status (`main.py:372-471`)
3. **Process** each configured symbol/timeframe for signals (`main.py:332-370`)
4. **Execute** trades when signals meet all criteria (`main.py:231-330`)
5. **Sleep** for configured interval (default 5 seconds)

### Module Responsibilities

**`main.py`** - Bot orchestration
- `Config`: Loads `config.json` with CLI overrides
- `TradingBot`: Async main loop, initializes all components, coordinates signal generation and execution
- Handles preflight symbol introspection (filling modes, tick specs)
- Monitors positions and updates trade logger on closure with reason codes (SL/TP/manual/stopout)

**`strategy.py`** - Signal generation (price action + patterns)
- `PurePriceActionStrategy`: Core breakout logic
- **Swing detection**: Finds local highs/lows using rolling window (`_find_swing_indices`)
- **S/R levels**: Consolidates swing points into resistance/support (`calculate_support_resistance`)
- **Breakout confirmation**: Requires close beyond S/R + pip threshold
- **Candlestick patterns**: Optional TA-Lib confirmation via `patterns.py` (`_candlestick_confirm`)
- **Multi-timeframe context**: Merges S/R from higher timeframes if provided (`mtf_context` parameter)
- **ATR-based SL**: Dynamic stop-loss using TA-Lib ATR or manual TR calculation (`_compute_atr`, `_stop_loss`)
- **Headroom filter**: Ensures sufficient space to next obstacle level before signaling (`_next_obstacle_level`)
- **Duplicate suppression**: Prevents re-entry near same level within time/price window
- **Strong pattern relaxation**: Relaxes headroom/duplicate/obstacle requirements if high-confidence pattern detected

**`mt5_client.py`** - MetaTrader 5 interface
- Encapsulates all MT5 API calls with auto-reconnect logic (`_ensure_connected`)
- **Filling mode selection**: Dynamically tries IOC/FOK/RETURN per symbol and caches working mode (`preferred_filling_mode`, `place_order`)
- Handles order placement with retry on unsupported fill modes
- Position management: get, close, modify
- Historical data: deals, orders by position ID

**`market_data.py`** - Candle data fetching
- Simple async wrapper around `mt5.copy_rates_from_pos`
- Returns DataFrame with normalized OHLCV columns

**`risk_manager.py`** - Position sizing and validation
- Calculates lot size based on account equity/balance and risk percentage
- Uses MT5 native `trade_tick_value`/`trade_tick_size` for pip value
- Enforces volume min/max/step, stops level, drawdown limits
- Pre-trade parameter validation

**`patterns.py`** - TA-Lib candlestick confirmation
- Aggregates ~10 curated TA-Lib patterns (engulfing, hammer, morning/evening star, etc.)
- Computes directional score [0,1] for breakout confirmation
- **Fail-open design**: Returns `ok=True` if TA-Lib unavailable (does not block live trading)
- Returns pattern metadata (primary pattern, score, direction)

**`trade_logger.py`** - Trade persistence
- Logs all trade lifecycle events to JSON/CSV
- Updates status on closure with reason codes (SL/TP/manual/expert/stopout)

**`utils.py`** - Pip size utilities
- `get_pip_size`: Default heuristic based on symbol digits
- `resolve_pip_size`: Allows per-symbol `pip_unit` override from config (e.g., XAUUSD)

### Configuration Architecture

The `config.json` uses a three-level override hierarchy:
1. **Global defaults** in `trading_settings` and `risk_management`
2. **Per-symbol overrides** in the `symbols` array (e.g., `XAUUSD` has custom pip_unit, min_stop_loss_pips, etc.)
3. **CLI flags** (e.g., `--symbol`, `--risk-per-trade`) override both

Key per-symbol overrides available:
- `pip_unit`: For non-standard instruments (XAUUSD uses 0.01)
- `min_stop_loss_pips`, `stop_loss_buffer_pips`, `breakout_threshold_pips`
- `risk_reward_ratio`, `min_headroom_rr`, `max_rr_cap`, `max_sl_pips`
- `spread_guard_pips`, `duplicate_breakout_distance_pips`, `duplicate_breakout_window_seconds`
- `drift_max_pips`, `drift_fraction_of_sl`: Execution drift guards
- `atr_period`, `atr_sl_k`: ATR-based SL tuning
- `timeframes`: List for multi-timeframe context (first is traded, rest provide S/R)

### Critical Implementation Details

**Execution Flow Safeguards** (`main.py:231-330`):
- **Drift guard**: Rejects if execution price drifts too far from signal price
- **Minimum SL with slack**: Allows small drift before rejecting tight stops
- **Same-direction check**: Skips signal if position with same magic and direction already open
- **Risk limits**: Checks drawdown and margin before sizing
- **Parameter validation**: Ensures volume, SL/TP meet broker minimums

**MT5 Filling Mode Handling** (`mt5_client.py:240-336`):
- Order placement tries multiple fill modes sequentially (cached → override → symbol → defaults)
- Retries on `TRADE_RETCODE_INVALID_FILL` (10030) with next candidate
- Caches working mode per symbol to optimize future orders
- Same strategy for position close (`close_position:387-440`)

**Position Monitoring** (`main.py:372-471`):
- Reconciles open trades with MT5 positions by ticket or deal
- Fetches historical deals to classify closure reason via `DEAL_REASON_*` enums
- Falls back to comment text parsing if reason code unavailable
- Updates trade logger with close price, profit, status, and timestamp

**Pattern-Driven Relaxation** (`strategy.py:405-473`):
- If pattern score ≥ `pattern_strong_threshold` (0.8):
  - Reduces duplicate suppression distance/window by 40-50%
  - Reduces headroom requirement by 25%
  - Reduces obstacle buffer by 30%
- Ensures high-confidence setups can trade even in tighter conditions

**Multi-Timeframe Context** (`strategy.py:310-341`):
- If `mtf_context` dict provided, merges S/R levels from higher timeframes
- Proximity consolidation prevents duplicate levels within 10 pips
- Used for headroom/obstacle calculations but does not trade higher TFs directly

## Development Notes

### Dependencies
Install via virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install MetaTrader5 pandas numpy ta-lib python-dotenv requests
```

Note: TA-Lib requires binary installation on Windows. See [ta-lib.org](https://ta-lib.org)

### Logging Philosophy
- **File log** (`forex_bot.log`): Full DEBUG detail for post-analysis
- **Console**: Concise INFO (configurable via `--log-level`)
- Noisy libraries (asyncio, MetaTrader5) are suppressed to WARNING
- Signal generation logs include ATR source, pattern score, and headroom for transparency

### Error Handling Patterns
- **Fail-safe**: Pattern confirmation fails open (allows trade if TA-Lib unavailable)
- **Auto-reconnect**: MT5 client attempts reconnection on connection loss (max 3 attempts)
- **Graceful degradation**: Missing optional config falls back to defaults

### Key Constraints
- **No tests**: Codebase lacks unit/integration tests
- **Windows-only**: MetaTrader5 API requires Windows
- **Live-only**: No backtesting framework (historical testing done externally)

## Common Modifications

### Adding a New Symbol
Edit `config.json` `symbols` array:
```json
{
  "name": "GBPJPY",
  "timeframes": ["M15", "H1"],
  "pip_unit": null,  // optional override
  "min_stop_loss_pips": 25
}
```

### Adjusting Strategy Behavior
- **Tighter breakouts**: Reduce `breakout_threshold_pips` (global or per-symbol)
- **Wider stops**: Increase `atr_sl_k` or `stop_loss_buffer_pips`
- **More aggressive patterns**: Lower `pattern_score_threshold`
- **Disable patterns**: Set `enable_patterns: false` in `trading_settings`

### Changing Risk Profile
- `risk_per_trade`: Fraction of equity risked (0.01 = 1%)
- `fixed_lot_size`: Bypass dynamic sizing (set to `null` for auto)
- `max_drawdown_percentage`: Stop trading if drawdown exceeds (0.05 = 5%)
- `use_equity`: If `false`, uses balance instead of equity for sizing

### Debugging a Failed Trade
1. Check `forex_bot.log` for DEBUG logs near signal timestamp
2. Look for rejection reasons: spread, drift, headroom, duplicate, SL/TP validation
3. Verify symbol info (digits, point, stops_level) logged during preflight
4. Check MT5 order result codes in log (retcode, comment)

### Extending Signal Logic
When modifying `strategy.py:generate_signal`:
- All candles except the last are "completed" (used for structure)
- Last candle is incomplete (not used for S/R or pattern detection)
- Always check `pip > 0` before division
- Round SL/TP to symbol's point precision before returning
- Add new filters before final `TradingSignal` construction

## Code Patterns to Follow

### Per-Symbol Config Lookup
```python
# Always iterate config.symbols to find overrides
min_sl = self.config.min_stop_loss_pips  # global default
for sc in getattr(self.config, 'symbols', []) or []:
    if sc.get('name') == symbol:
        min_sl = sc.get('min_stop_loss_pips', min_sl)
        break
```

### MT5 Reconnect Pattern
```python
if not self._ensure_connected():
    return None
result = mt5.some_call()
if result is None and self.auto_reconnect and self.reconnect():
    result = mt5.some_call()
```

### Safe Attribute Access on MT5 Objects
```python
value = getattr(symbol_info, 'attribute_name', default_value)
```

