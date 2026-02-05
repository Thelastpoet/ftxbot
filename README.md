# Forex Trading Bot

A Forex trading bot built on the MetaTrader 5 platform with price action strategies, risk management, and multi-timeframe analysis.

## Important Warning

This is experimental software. Trading involves substantial risk of financial loss. Past performance does not guarantee future results. Use at your own risk.

## Features

- Multi-timeframe Analysis: Uses different timeframes for entry (e.g., M15), structure (e.g., H1), and trend (e.g., H4)
- Breakout Strategy: Identifies swing highs/lows and detects breakouts with configurable thresholds
- Risk Management: Position sizing based on risk percentage and stop loss distance
- R-Multiple Trailing Stops: Trailing stops based on risk multiples rather than fixed pips
- Multiple Filters: Trend filters, momentum filters, structure confirmation, and more
- State Persistence: Prevents duplicate signals across restarts
- Auto-Reconnection: Automatic handling of MT5 disconnections
- Spread Monitoring: Avoids trading during periods of high spreads
- Session Filtering: Configurable trading hours

## Prerequisites

- Python 3.8+
- MetaTrader 5 Terminal (with active trading account)
- MetaTrader5 Python package
- Pandas
- Additional dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex-trading-bot.git
cd forex-trading-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up pre-commit hooks (recommended for contributors):
```bash
# On Unix/Linux/Mac:
chmod +x setup-pre-commit.sh
./setup-pre-commit.sh

# On Windows:
powershell -ExecutionPolicy Bypass -File setup-pre-commit.ps1
```

4. Ensure your MT5 terminal is running and logged in to your trading account

## Configuration

The bot is configured through `config.json`. Key configuration sections:

### Trading Settings
- `main_loop_interval_seconds`: How often the bot checks for signals (default: 5 seconds)
- `lookback_period`: Number of bars to analyze for swing detection
- `breakout_threshold_pips`: Minimum movement to qualify as a breakout
- `min_stop_loss_pips`: Minimum stop loss distance in pips
- `risk_reward_ratio`: Desired risk-to-reward ratio for trades

### Risk Management
- `risk_per_trade`: Risk percentage per trade (e.g., 0.01 = 1%)
- `max_drawdown_percentage`: Maximum drawdown before stopping
- `fixed_lot_size`: Optional fixed lot size (overrides risk calculation)

### Symbols
Configure which symbols to trade and their specific timeframes:
```json
"symbols": [
  {
    "name": "EURUSD",
    "entry_timeframe": "M15",
    "structure_timeframe": "H1", 
    "trend_timeframe": "H4"
  }
]
```

## Usage

Run the bot with default settings:
```bash
python main.py
```

With command-line arguments:
```bash
python main.py --symbol EURUSD --timeframe M15 --risk-per-trade 0.005
```

Available options:
- `--config`: Path to config file (default: config.json)
- `--symbol`: Specific symbol to trade (overrides config)
- `--timeframe`: Timeframe for analysis (overrides config)
- `--risk-per-trade`: Risk percentage per trade
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Strategy Overview

The bot implements a price action breakout strategy:

1. Swing Detection: Identifies swing highs and lows on higher timeframes
2. Support/Resistance: Creates S/R levels from swing points
3. Breakout Detection: Monitors for price breakouts beyond S/R levels
4. Entry Logic: Enters trades when breakouts are confirmed
5. Stop Loss: Placed beyond opposing structure with buffer
6. Take Profit: Calculated based on risk-reward ratio or structure levels
7. Trailing: Dynamic trailing stops that adapt to market conditions

## Risk Management

- Position sizing based on risk percentage and stop loss distance
- Maximum drawdown limits
- Spread guards to avoid trading during high-spread periods
- Stop loss minimums to avoid invalid orders
- Daily rollover period avoidance (21:55-23:05 UTC)

## Customization

The bot is highly configurable:
- Adjust all parameters in `config.json`
- Modify strategy logic in `strategy.py`
- Customize trailing stops in `trailing_stop.py`
- Add custom risk rules in `risk_manager.py`

## Monitoring

- Real-time logging of all activities
- Trade history saved to `trades.json`
- Heartbeat logs every minute showing open positions and performance
- Detailed signal analysis in logs

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is provided "as is" without warranty of any kind. Trading involves substantial risk of financial loss. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

## Issues

If you encounter any problems, please file an issue on the GitHub repository with:
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information (Python version, OS, MT5 version)

## Documentation

Complete documentation is available in the [documentation directory](docs/index.md).

## Development Setup

For contributors, we recommend setting up pre-commit hooks to ensure code quality and consistency:

1. After installing dependencies, run the pre-commit setup script:
   ```bash
   # On Unix/Linux/Mac:
   chmod +x setup-pre-commit.sh
   ./setup-pre-commit.sh
   
   # On Windows:
   powershell -ExecutionPolicy Bypass -File setup-pre-commit.ps1
   ```

2. The pre-commit hooks will automatically format and lint your code before each commit

## Acknowledgments

- MetaTrader 5 for the trading platform
- The open-source community for various Python packages used in this project