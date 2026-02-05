# Monitoring

This guide explains how to monitor the Forex Trading Bot's performance and activity.

## Real-Time Monitoring

### Console Output
The bot provides real-time information in the console:

- **Connection Status**: Shows MT5 connection status
- **Signal Generation**: Logs when trading signals are detected
- **Trade Execution**: Records when trades are placed
- **Heartbeat Messages**: Every minute, shows open positions and status
- **Error Messages**: Reports any issues or problems

Example heartbeat message:
```
[HEARTBEAT] symbols=5 open_pos=2 total_trades=15 loop=45ms
```

### Log Levels
- **INFO**: Normal operations and trade executions
- **WARNING**: Potential issues that don't stop operation
- **ERROR**: Problems that may affect functionality
- **DEBUG**: Detailed information for troubleshooting

## Log Files

### Location
Logs are written to `forex_bot.log` by default.

### Log Rotation
For long-running bots, consider setting up log rotation to manage file sizes.

### Log Analysis
Regularly review logs for:
- Trade execution details
- Error patterns
- Performance metrics
- Unusual market behavior

## Trade Tracking

### Trade Records
All trades are recorded in `trades.json` with details:
- Timestamp
- Symbol
- Order type (BUY/SELL)
- Entry/exit prices
- Stop loss and take profit levels
- Volume
- Profit/loss
- Status (OPEN/CLOSED)

### Performance Metrics
Monitor:
- Win rate
- Average profit/loss
- Maximum drawdown
- Profit factor
- Number of trades per day

## Position Monitoring

### Open Positions
The bot tracks all open positions and:
- Updates stop losses if trailing is enabled
- Monitors for take profit conditions
- Checks for exit signals
- Validates position health

### Position Sizing
Monitor that position sizes remain within acceptable limits based on risk parameters.

## Risk Monitoring

### Drawdown Tracking
Watch for increasing drawdown that approaches your maximum threshold.

### Margin Levels
Monitor account margin levels to avoid margin calls.

### Exposure
Track total exposure across all positions to ensure diversification.

## Market Conditions

### Volatility
Monitor market volatility which affects:
- Stop loss distances
- Take profit levels
- Signal frequency

### Liquidity
Watch for low liquidity periods that may affect:
- Execution quality
- Spread widening
- Signal reliability

## Bot Health

### Resource Usage
Monitor:
- CPU usage
- Memory consumption
- Network connectivity

### Response Times
Check that the bot responds within expected timeframes.

### Error Rates
Track frequency of errors and connection issues.

## Alerting

### Manual Monitoring
Set up regular checks for:
- Bot status
- Recent trades
- Account balance
- Open positions

### Automated Alerts
Consider setting up:
- Email notifications for major events
- SMS alerts for critical errors
- Dashboard monitoring

## Performance Evaluation

### Daily Review
At the end of each trading day:
- Check all executed trades
- Review any errors or warnings
- Verify position closures
- Assess overall performance

### Weekly Review
Weekly analysis should include:
- Performance metrics
- Strategy effectiveness
- Parameter adjustments needed
- Market condition changes

### Monthly Review
Monthly evaluation should cover:
- Overall profitability
- Risk-adjusted returns
- Drawdown analysis
- Strategy modifications

## Troubleshooting

### Common Monitoring Issues
- **Missing logs**: Check file permissions and disk space
- **Stale positions**: Verify bot is running and connected
- **Unexpected trades**: Review configuration and filters
- **High resource usage**: Check number of monitored symbols

### Performance Optimization
- Reduce monitored symbols if needed
- Adjust polling intervals
- Optimize strategy parameters
- Upgrade hardware if necessary