# Risk Management Documentation

## Overview

The Forex Trading Bot implements risk management to protect capital and maintain trading performance. The risk management system operates on multiple levels to prevent losses and maintain consistent risk exposure.

## Position Sizing

### Risk-Based Position Sizing

The bot calculates position size based on a percentage of account balance or equity:

```
risk_amount = base * risk_per_trade
position_size = risk_amount / (stop_loss_pips * pip_value)
```

Where:
- `base` is either account balance or equity (configurable)
- `risk_per_trade` is the configurable risk percentage per trade
- `stop_loss_pips` is the distance from entry to stop loss in pips
- `pip_value` is the monetary value of one pip for the instrument

### Pip Value Calculation

The pip value is calculated using MT5's native values:

```
pip_value = (trade_tick_value / trade_tick_size) * pip_size
```

## Risk Limits

### Drawdown Protection

The system monitors account drawdown and halts trading if it exceeds the configured threshold:

```
drawdown = max(0, (balance - equity) / balance)
if drawdown >= max_drawdown:
    halt_trading()
```

### Margin Level Monitoring

Trading is suspended if the margin level falls below safe levels:

```
if margin_level < 200:  # 200% is typically considered safe
    halt_trading()
```

## Trade Validation

Before executing any trade, the system validates multiple parameters:

### Volume Validation

```
if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
    reject_trade()
```

### Price Placement Validation

For BUY orders:
```
if not (take_profit > current_price > stop_loss):
    reject_trade()
```

For SELL orders:
```
if not (take_profit < current_price < stop_loss):
    reject_trade()
```

### Minimum Distance Validation

Ensures stop loss and take profit are not too close to market prices:
```
min_stop_distance = trade_stops_level * point
if abs(current_price - stop_loss) < min_stop_distance:
    reject_trade()
```

## Configuration Parameters

### Risk Management Settings

- `risk_per_trade`: Percentage of account balance to risk per trade (e.g., 0.01 = 1%)
- `fixed_lot_size`: Optional fixed lot size (overrides risk calculation)
- `max_drawdown_percentage`: Maximum drawdown threshold before stopping
- `use_equity`: Whether to use equity instead of balance for risk calculation

### Stop Loss Settings

- `min_stop_loss_pips`: Minimum stop loss distance in pips
- `stop_loss_buffer_pips`: Additional buffer added to stop loss
- `max_sl_pips`: Maximum stop loss distance in pips
- `atr_sl_k`: ATR multiplier for dynamic stop loss buffers
- `min_sl_buffer_pips`: Minimum stop loss buffer in pips

## Spread Management

### Spread Guard

The system includes a spread guard to avoid trading during periods of high spreads:

```
current_spread_pips = abs(ask - bid) / pip_size
if current_spread_pips > spread_guard_pips:
    reject_trade()
```

## Trailing Stop Risk Management

The trailing stop system implements additional risk controls:

### R-Multiple Logic

- Break-even trigger: When profit reaches 0.4x initial risk
- Trail trigger: When profit reaches 0.7x initial risk
- Trail distance: 0.4x initial risk behind price

### Time-Based Exit

Positions can be automatically closed after a configurable number of hours to prevent indefinite holding.

## Symbol-Specific Risk Controls

Risk parameters can be overridden per symbol in the configuration:

```json
{
  "name": "XAUUSD",
  "pip_unit": 0.01,
  "min_stop_loss_pips": 250,
  "stop_loss_buffer_pips": 60,
  "breakout_threshold_pips": 30,
  "risk_reward_ratio": 1.5,
  "spread_guard_pips": 40,
  "max_sl_atr_mult": 4.0
}
```

## Circuit Breakers

### Daily Loss Limit

The system can be configured to stop trading after reaching a daily loss threshold.

### Consecutive Losses

Trading can be halted after a certain number of consecutive losing trades.

## Risk Monitoring

### Real-time Monitoring

The system continuously monitors:
- Current drawdown
- Open trade exposure
- Correlation between open positions
- Margin level

### Logging and Alerts

All risk-related decisions are logged for review:
- Trade rejections due to risk limits
- Drawdown warnings
- Margin level warnings
- Position size adjustments

## Best Practices

### Recommended Settings

- Risk per trade: 0.5% - 2% of account balance
- Maximum drawdown: 5% - 10% of account balance
- Minimum risk-reward ratio: 1:1.5 or higher

### Diversification

- Limit the number of concurrent positions
- Avoid high correlation between traded instruments
- Distribute risk across different market conditions

### Regular Review

- Monitor performance metrics regularly
- Adjust risk parameters based on market conditions
- Review and update stop loss and take profit settings