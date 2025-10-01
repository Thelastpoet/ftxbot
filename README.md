# FTXBot - Forex Trading Bot with MT5 Integration

**FTXBot** is a sophisticated Forex trading bot written in Python that implements pure price action strategies with MetaTrader 5 (MT5) integration. Despite its name suggesting FTX, the project actually integrates with MT5 for Forex trading, providing professional-grade algorithmic trading capabilities.

## 🚀 Features

- **Multi-timeframe Analysis**: Uses both M15 and H1 timeframes for trend alignment and signal generation
- **Price Action Strategy**: Pure price action with swing point identification and breakout detection
- **Risk Management**: Comprehensive risk management with position sizing, stop loss, and take profit controls
- **Candlestick Pattern Recognition**: Uses TA-Lib for pattern recognition to enhance signal quality
- **Dynamic Calibration**: Machine learning-based calibration system that learns from trade results
- **Market Session Management**: Trades only during specific market sessions (Asia, London, New York)
- **Live Execution**: Real-time trade execution with dynamic entry price validation
- **Trade Logging**: Comprehensive trade logging and status synchronization

## 🛠️ Prerequisites

Before running the bot, you need:

- Python 3.8+
- MetaTrader 5 terminal installed and running
- Active MT5 trading account (demo or real)
- Required Python packages (see below)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ftxbot.git
cd ftxbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install MetaTrader5 pandas numpy scipy
# Optional: pip install TA-Lib (requires additional installation steps)
```

4. Set up your MT5 terminal with proper account credentials

## ⚙️ Configuration

The bot is configured through `config.json`. Key configuration sections include:

### Trading Settings
- `max_period`: Maximum lookback period for analysis
- `main_loop_interval_seconds`: How often the bot checks for opportunities
- `breakout_threshold_pips`: Minimum pips for breakout confirmation
- `atr_tp_multiplier`: Multiplier for ATR-based take profit

### Risk Management
- `risk_per_trade`: Percentage of account risked per trade
- `max_drawdown_percentage`: Maximum account drawdown before shutdown
- `risk_reward_ratio`: Minimum risk/reward ratio for trades

### Symbols
Configure which Forex pairs to trade and their timeframes:
```json
"symbols": [
  {
    "name": "EURUSD",
    "timeframes": ["M15", "H1"]
  }
]
```

### Trading Sessions
Define when the bot should trade:
```json
"trading_sessions": [
  { "name": "Asia",    "start_time": "03:00", "end_time": "12:00" },
  { "name": "London",  "start_time": "10:00", "end_time": "20:00" },
  { "name": "NewYork", "start_time": "15:00", "end_time": "01:00" }
]
```

## 🚀 Usage

### Basic Execution
```bash
python main.py
```

### Command Line Options
- `--config`: Specify alternative configuration file
- `--risk-per-trade`: Override risk percentage per trade
- `--symbol`: Override trading symbol for single symbol testing
- `--timeframe`: Override primary timeframe
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Example
```bash
python main.py --risk-per-trade 0.02 --symbol EURUSD --log-level DEBUG
```

## 🔧 Architecture

The bot is organized into several key modules:

- `main.py`: Main application orchestrator that manages the trading loop
- `strategy.py`: Core trading strategy implementation with pure price action logic
- `mt5_client.py`: MT5 API wrapper for trading operations
- `risk_manager.py`: Risk management and position sizing calculations
- `market_data.py`: Data fetching and processing from MT5
- `market_session.py`: Trading session and time-based controls
- `calibrator.py`: Machine learning calibrator for signal validation
- `optimizer.py`: Optimization algorithms for strategy parameters
- `symbol_runtime.py`: Per-symbol runtime context and configuration management
- `trade_logger.py`: Trade logging and persistence
- `utils.py`: Utility functions and helper methods

## 📊 Trading Logic

The bot follows a comprehensive multi-step process:

1. **Data Fetching**: Retrieves historical data for configured timeframes
2. **Trend Analysis**: Identifies trend direction on higher timeframes
3. **Breakout Detection**: Finds swing points and potential breakouts
4. **Pattern Recognition**: Validates signals with candlestick patterns
5. **Risk Assessment**: Calculates position size and validates risk parameters
6. **Execution**: Places trades with validated parameters
7. **Monitoring**: Synchronizes trade status and manages open positions

### Breakout Detection
The bot identifies support and resistance levels using swing point analysis and detects breakouts with configurable pip thresholds. It validates breakouts using multiple criteria including trend alignment and pattern confirmation.

### Dynamic Calibration
A machine learning system adjusts signal acceptance based on historical trade results, learning which market conditions and signal characteristics lead to profitable outcomes.

## ⚠️ Risk Controls

The bot implements multiple layers of risk control:
- Maximum drawdown limits
- Per-trade risk percentage
- Position correlation limits between currency pairs
- Stop loss minimums
- News filter to pause trading during high-volatility events
- Cooldown periods after trade closures
- Price drift validation to avoid slippage issues

## 📁 Files and Directories

- `calibrator_states/`: Storage for calibrator state files
- `optimizer_states/`: Storage for optimizer state files
- `symbol_configs/`: Per-symbol configuration files
- `trades.log`: Log of executed trades
- `forex_bot.log`: Application log file
- `bot_memory.json`: Runtime memory state for consumed breakouts and recent closures

## 🔍 Monitoring

The bot outputs detailed logs that include:
- Signal generation and confidence levels
- Trade execution details
- Risk management validation
- Market session status
- Performance metrics

## 🔒 Security Considerations

- Store your MT5 credentials securely
- Use a dedicated trading account for the bot
- Monitor the bot's activity regularly
- Start with demo account before going live
- Implement proper account security measures

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Trading involves substantial risk and may not be suitable for everyone. Past performance does not guarantee future results. This software is provided as-is without warranty. Use at your own risk and only with funds you can afford to lose.**

## 📞 Support

For issues and questions, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- MetaTrader 5 platform for providing the trading infrastructure
- TA-Lib for technical analysis functions
- The open-source community for providing essential Python libraries