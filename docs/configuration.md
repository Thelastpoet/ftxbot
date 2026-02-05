# Configuration Guide

This guide explains all the configuration options available in the Forex Trading Bot.

## Main Configuration File

The main configuration is stored in `config.json` and consists of three main sections:

- `trading_settings` - Strategy and market analysis parameters
- `risk_management` - Risk control parameters
- `symbols` - List of symbols to trade with their settings

## Trading Settings

### Basic Timing
- `main_loop_interval_seconds`: How often the bot checks for signals (default: 5 seconds)
- `lookback_period`: Number of bars to analyze for swing detection (default: 80)

### Swing and Breakout Detection
- `swing_window`: Number of bars to compare on each side for swing detection (default: 6)
- `sr_lookback_period`: Number of bars to analyze for support/resistance (default: 120)
- `sr_proximity_pips`: Maximum distance between similar support/resistance levels (default: 8)

### Breakout Thresholds
- `breakout_threshold_pips`: Fixed pip threshold for breakouts (default: 2)
- `breakout_threshold_atr_mult`: ATR multiplier for dynamic thresholds (default: 0.4)
- `breakout_threshold_spread_mult`: Spread multiplier for threshold (default: 2.0)
- `breakout_window_bars`: Maximum bars ago a breakout can occur (default: 0)

### Stop Loss Settings
- `min_stop_loss_pips`: Minimum stop loss distance in pips (default: 15)
- `stop_loss_buffer_pips`: Additional buffer added to stop loss (default: 10)
- `min_sl_buffer_pips`: Minimum stop loss buffer (default: 8)
- `max_sl_pips`: Maximum stop loss distance (default: null)
- `atr_sl_k`: ATR multiplier for dynamic stop loss buffers (default: 0.6)

### Structure Settings
- `structure_min_touches`: Minimum touches for valid S/R level (default: 2)
- `structure_atr_band_mult`: ATR band multiplier for structure validation (default: 0.4)

### Trend and Filter Settings
- `use_trend_filter`: Enable/disable trend filter (default: true)
- `trend_ema_period`: EMA period for trend filter (default: 40)
- `use_ema_slope_filter`: Enable/disable EMA slope filter (default: false)
- `ema_slope_period`: Period for EMA slope calculation (default: 8)

### Confirmation Settings
- `require_structure_confirmation`: Require higher timeframe confirmation (default: false)
- `require_two_bar_confirmation`: Require two-bar confirmation (default: false)
- `require_fresh_breakout`: Only trade fresh breakouts (default: true)
- `fresh_breakout_grace_bars`: Grace period for breakout freshness (default: 10)

### Entry Settings
- `entry_mode`: "momentum" or "retest" (default: "momentum")
- `retest_window_bars`: Bars to wait for retest (default: 8)
- `entry_cooldown_bars`: Minimum bars between signals (default: 3)

### Take Profit Settings
- `tp_mode`: "r_multiple" or "structure" (default: "r_multiple")
- `tp_r_multiple`: Risk-reward ratio for take profit (default: 1.5)
- `tp_use_structure_cap`: Cap take profit at structure level (default: true)

### Session Settings
- `use_session_filter`: Enable/disable session filter (default: true)
- `session_start_utc`: Trading session start time (default: "08:00")
- `session_end_utc`: Trading session end time (default: "18:00")

## Risk Management Settings

- `risk_per_trade`: Risk percentage per trade (e.g., 0.01 = 1%) (default: 0.004)
- `fixed_lot_size`: Fixed lot size (overrides risk calculation) (default: null)
- `max_drawdown_percentage`: Maximum drawdown before stopping (default: 0.04)
- `risk_reward_ratio`: Minimum risk-reward ratio (default: 1.5)
- `min_rr`: Minimum risk-reward ratio (default: 1.2)

## Symbol Configuration

Each symbol can have specific settings:

```json
{
  "name": "EURUSD",
  "entry_timeframe": "M15",
  "structure_timeframe": "H1", 
  "trend_timeframe": "H4",
  "pip_unit": 0.0001,
  "min_stop_loss_pips": 15,
  "stop_loss_buffer_pips": 10
}
```

### Symbol-Specific Parameters

All trading settings can be overridden per symbol:
- `pip_unit`: Pip size for the symbol
- `min_stop_loss_pips`: Symbol-specific minimum stop loss
- `stop_loss_buffer_pips`: Symbol-specific stop loss buffer
- `breakout_threshold_pips`: Symbol-specific breakout threshold
- And many more parameters

## Example Configuration

```json
{
  "trading_settings": {
    "main_loop_interval_seconds": 5,
    "lookback_period": 80,
    "swing_window": 6,
    "breakout_threshold_pips": 2,
    "min_stop_loss_pips": 15,
    "risk_reward_ratio": 1.5
  },
  "risk_management": {
    "risk_per_trade": 0.01,
    "max_drawdown_percentage": 0.05
  },
  "symbols": [
    {
      "name": "EURUSD",
      "entry_timeframe": "M15",
      "structure_timeframe": "H1",
      "trend_timeframe": "H4"
    }
  ]
}
```