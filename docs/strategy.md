# Strategy Documentation

## Overview

The Forex Trading Bot implements a price action breakout strategy that combines multiple analytical approaches to identify trading opportunities.

## Core Components

### 1. Multi-Timeframe Analysis

The strategy operates on three timeframes simultaneously:

- **Entry Timeframe**: Used for entry timing (e.g., M15)
- **Structure Timeframe**: Used for support/resistance levels (e.g., H1)
- **Trend Timeframe**: Used for trend alignment (e.g., H4)

### 2. Swing Point Detection

The bot identifies swing highs and lows using configurable parameters:

- `swing_window`: Number of bars to compare on each side
- `sr_lookback_period`: Number of bars to analyze for swing detection
- `structure_min_touches`: Minimum touches required for a valid support/resistance level

### 3. Breakout Detection

Breakouts are detected when price moves beyond support/resistance levels by a configurable threshold:

- `breakout_threshold_pips`: Fixed pip threshold
- `breakout_threshold_atr_mult`: ATR-based threshold multiplier
- `breakout_threshold_spread_mult`: Spread-based threshold multiplier

### 4. Entry Conditions

The bot requires multiple confirmations before entering a trade:

- Breakout confirmation
- Trend alignment (optional)
- Momentum filter (optional)
- Structure confirmation (optional)
- Two-bar confirmation (optional)

## Risk Management

### Position Sizing

Position size is calculated based on:
- Account balance/equity
- Risk percentage per trade
- Stop loss distance in pips
- Pip value for the instrument

### Stop Loss Placement

Stop losses are placed beyond opposing structure with a buffer:
- Base stop loss at nearest opposing support/resistance level
- Buffer added based on ATR or fixed pips
- Minimum stop loss distance enforced

### Take Profit Calculation

Take profit can be calculated in two modes:
- `r_multiple`: Based on risk-reward ratio
- `structure`: At next structure level in the trade direction

## Trailing Stops

The bot implements R-multiple based trailing stops:
- Break-even triggered when profit reaches 0.4x initial risk
- Trailing begins when profit reaches 0.7x initial risk
- Trail distance is 0.4x initial risk
- Positions can be closed after maximum holding time

## Filters and Safeguards

### Spread Filter
- Trades rejected when spread exceeds `spread_guard_pips`

### Session Filter
- Trading restricted to specific hours (UTC)

### Rollover Protection
- Trading suspended during daily Forex rollover (21:55-23:05 UTC)

### Momentum Filter
- Requires strong impulse candles for breakout confirmation

## Configuration Parameters

### Trading Settings
- `main_loop_interval_seconds`: Frequency of market analysis
- `lookback_period`: Bars to analyze for swing detection
- `swing_window`: Window size for swing detection
- `breakout_threshold_*`: Various breakout threshold settings
- `min_stop_loss_pips`: Minimum stop loss distance
- `atr_period`: ATR calculation period
- `sr_lookback_period`: Structure analysis lookback
- `tp_r_multiple`: Risk-reward ratio for take profit

### Risk Management Settings
- `risk_per_trade`: Risk percentage per trade
- `max_drawdown_percentage`: Maximum drawdown threshold
- `risk_reward_ratio`: Minimum risk-reward ratio

## Signal Generation Process

1. Fetch market data for all required timeframes
2. Identify swing points and calculate support/resistance levels
3. Detect breakouts from support/resistance levels
4. Apply all configured filters and confirmations
5. Calculate stop loss and take profit levels
6. Validate risk-reward ratio
7. Generate trading signal if all conditions met

## State Management

The bot maintains state to prevent duplicate signals:
- Tracks last breakout bar per symbol and direction
- Persists state across restarts
- Prevents multiple entries in the same direction per bar

## Monitoring and Logging

- Detailed logging of all decision points
- Trade execution logging
- Performance metrics tracking
- Error and warning notifications