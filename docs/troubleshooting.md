# Troubleshooting Guide

## Common Issues and Solutions

### Connection Problems

#### MT5 Connection Failure
**Symptoms:**
- "MT5 initialization failed" error
- "MT5 not connected" messages
- Unable to fetch market data

**Solutions:**
1. Ensure MetaTrader 5 terminal is running and logged in
2. Check that the MT5 terminal allows DLL imports (enable in Tools → Options → Expert Advisors)
3. Verify that the trading account is active and not locked
4. Restart the MT5 terminal if connection issues persist
5. Check firewall settings to ensure MT5 can connect to brokers

#### Auto-reconnection Issues
**Symptoms:**
- Frequent disconnections
- Trades not executing during disconnection periods

**Solutions:**
1. Check internet connection stability
2. Verify broker server status
3. Increase `reconnect_delay` in the MT5 client if needed
4. Ensure MT5 terminal remains active (not minimized or suspended)

### Trading Issues

#### No Signals Generated
**Symptoms:**
- Bot runs but no trades are executed
- "No breakout detected" messages in logs

**Solutions:**
1. Verify that configured symbols are available in MT5 Market Watch
2. Check that sufficient historical data is available
3. Review configuration parameters (thresholds might be too strict)
4. Ensure market is open for the configured symbols
5. Check that spread is below `spread_guard_pips` threshold

#### Rejected Trades
**Symptoms:**
- "Trade rejected" messages in logs
- Orders failing to execute

**Solutions:**
1. Check that stop loss and take profit levels meet broker requirements
2. Verify that position size is within minimum/maximum limits
3. Ensure account has sufficient margin
4. Check that prices are properly formatted according to broker requirements
5. Review `stops_level` requirements for the instrument

### Configuration Issues

#### Invalid Configuration
**Symptoms:**
- Errors loading config.json
- Unexpected behavior despite correct configuration

**Solutions:**
1. Validate JSON syntax using a JSON validator
2. Ensure all required fields are present
3. Check that numeric values are properly formatted
4. Verify that timeframes are supported by your broker
5. Confirm that symbol names match exactly with MT5

#### Parameter Tuning
**Symptoms:**
- Too many/few signals
- Poor performance
- Excessive risk

**Solutions:**
1. Start with conservative parameters and adjust gradually
2. Backtest changes before applying to live trading
3. Monitor performance metrics closely
4. Consider market volatility when setting thresholds
5. Adjust risk parameters based on account size

### Performance Issues

#### High CPU/Memory Usage
**Symptoms:**
- Slow execution
- High resource consumption
- Delays in signal processing

**Solutions:**
1. Reduce the number of monitored symbols
2. Increase `main_loop_interval_seconds` to reduce frequency
3. Decrease the amount of historical data requested
4. Close unnecessary MT5 charts and tools
5. Restart the bot periodically to clear memory

#### Delayed Execution
**Symptoms:**
- Late trade entries/exits
- Missed opportunities
- Slippage issues

**Solutions:**
1. Optimize computer performance (close unnecessary programs)
2. Ensure stable internet connection
3. Check broker execution speed
4. Adjust deviation parameters for faster fills
5. Consider using faster broker servers

### Risk Management Issues

#### Unexpected Losses
**Symptoms:**
- Larger than expected losses
- Stop losses not triggering correctly
- Margin calls

**Solutions:**
1. Review risk parameters in configuration
2. Verify stop loss and take profit calculations
3. Check that risk per trade is appropriate for account size
4. Monitor market conditions during volatile periods
5. Consider reducing position sizes during uncertain times

#### Drawdown Protection Not Working
**Symptoms:**
- Drawdown exceeding configured limits
- Continued trading during large losses

**Solutions:**
1. Verify drawdown calculation method (balance vs equity)
2. Check that account information is being retrieved correctly
3. Ensure drawdown monitoring is enabled
4. Review the maximum drawdown percentage setting

### Logging and Monitoring Issues

#### Missing Logs
**Symptoms:**
- Insufficient information in logs
- Difficulty debugging issues

**Solutions:**
1. Set log level to DEBUG for detailed information
2. Check that log file paths are writable
3. Verify logging configuration is correct
4. Monitor both console and file logs

#### Trade History Problems
**Symptoms:**
- Missing trade records
- Inconsistent trade data
- State not persisting across restarts

**Solutions:**
1. Check file permissions for trade log directory
2. Verify state manager is functioning correctly
3. Ensure JSON files are not corrupted
4. Backup trade history regularly

## Diagnostic Commands

### Checking MT5 Connection
```python
from mt5_client import MetaTrader5Client
client = MetaTrader5Client()
print(f"Connected: {client.is_connected()}")
print(f"Account Info: {client.get_account_info()}")
```

### Verifying Symbol Data
```python
symbol_info = client.get_symbol_info("EURUSD")
print(f"Symbol Info: {symbol_info}")
tick = client.get_symbol_info_tick("EURUSD")
print(f"Tick Info: {tick}")
```

### Testing Configuration
```python
from config import Config
config = Config()
print(f"Symbols: {config.symbols}")
print(f"Risk per trade: {config.risk_per_trade}")
```

## Preventive Measures

### Regular Maintenance
1. Update MT5 terminal regularly
2. Monitor account balance and margin levels
3. Review performance metrics weekly
4. Update configuration based on market conditions
5. Backup configuration files regularly

### Monitoring Checklist
- [ ] MT5 terminal is running and connected
- [ ] Sufficient margin available for trading
- [ ] Internet connection is stable
- [ ] Computer resources are adequate
- [ ] Market is open for configured symbols
- [ ] Configuration parameters are appropriate
- [ ] Risk limits are properly set
- [ ] Logs are being written correctly

## When to Stop Trading

Stop trading immediately if:
- Multiple consecutive losses occur
- Market conditions change dramatically
- Technical issues persist despite troubleshooting
- Drawdown approaches maximum limits
- Personal circumstances require attention
- Regulatory changes affect trading

## Support Resources

If issues persist after troubleshooting:
1. Check the GitHub issues page for similar problems
2. Review the documentation thoroughly
3. Consider reaching out to the community for assistance
4. Consult with experienced traders for advice
5. Consider pausing live trading to investigate issues

Remember: When in doubt, stop trading and investigate before continuing.