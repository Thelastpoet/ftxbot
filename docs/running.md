# Running the Bot

This guide explains how to run the Forex Trading Bot in different scenarios.

## Basic Execution

Run the bot with default settings:

```bash
python main.py
```

## Command Line Options

The bot accepts several command line arguments:

### Configuration
- `--config`: Specify a different config file
  ```bash
  python main.py --config my_custom_config.json
  ```

### Symbol Selection
- `--symbol`: Trade only a specific symbol
  ```bash
  python main.py --symbol EURUSD
  ```

### Timeframe Override
- `--timeframe`: Override the entry timeframe for all symbols
  ```bash
  python main.py --timeframe M30
  ```

### Risk Adjustment
- `--risk-per-trade`: Override the risk per trade percentage
  ```bash
  python main.py --risk-per-trade 0.005
  ```

### Logging
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR)
  ```bash
  python main.py --log-level DEBUG
  ```

## Running Modes

### Paper Trading Mode
Currently, the bot doesn't have a built-in paper trading mode. To simulate, consider:
- Using a demo account
- Reducing position sizes significantly
- Monitoring closely without intervention

### Live Trading Mode
When ready for live trading:
- Ensure you're using a live account
- Review all risk parameters
- Start with small position sizes
- Monitor the bot closely initially

## Monitoring the Bot

### Console Output
The bot provides real-time information in the console:
- Connection status
- Signal generation
- Trade executions
- Error messages
- Heartbeat messages every minute

### Log Files
Logs are written to `forex_bot.log` based on your logging configuration.

### Trade Records
All trades are recorded in `trades.json` for review and analysis.

## Managing the Bot

### Starting the Bot
```bash
python main.py
```

### Stopping the Bot
Press `Ctrl+C` to stop the bot gracefully.

### Restarting the Bot
The bot handles restarts gracefully:
- It will reconnect to MT5
- Resume monitoring configured symbols
- Continue from current market conditions

## Automation

### Windows Task Scheduler
Create a scheduled task to start the bot automatically.

### Linux Cron Jobs
Use cron to schedule automatic starts:
```bash
# Start bot at market open Monday-Friday
0 6 * * 1-5 /path/to/venv/bin/python /path/to/bot/main.py
```

### Process Managers
Consider using process managers like systemd (Linux) or NSSM (Windows) for automatic restarts.

## Best Practices

### Before Running
- Verify MT5 is connected and logged in
- Check that all configured symbols are available
- Review risk parameters
- Ensure sufficient margin in your account

### During Operation
- Monitor initial runs closely
- Check logs regularly
- Verify trade executions match expectations
- Watch for error messages

### Maintenance
- Restart periodically to clear memory
- Rotate log files to prevent large files
- Update the bot when new versions are released
- Review and adjust parameters based on performance

## Troubleshooting Common Issues

### Bot Won't Start
- Check Python dependencies are installed
- Verify MT5 terminal is running
- Confirm config.json is valid JSON

### No Trades Executed
- Check that market is open for configured symbols
- Verify spread is below guard thresholds
- Confirm risk parameters are appropriate

### Connection Issues
- Ensure MT5 allows DLL imports
- Check internet connection
- Verify broker server status