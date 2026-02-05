# Quick Start Guide

This guide will help you get the Forex Trading Bot up and running quickly.

## Prerequisites

- Python 3.8+
- MetaTrader 5 terminal with active account
- Stable internet connection

## Step 1: Download and Setup

1. Clone or download the repository:
   ```bash
   git clone https://github.com/yourusername/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Configure MetaTrader 5

1. Open MetaTrader 5
2. Go to Tools → Options → Expert Advisors
3. Enable "Allow DLL imports"
4. Ensure your trading account is connected

## Step 3: Basic Configuration

Edit the `config.json` file to set your preferred settings:

```json
{
  "trading_settings": {
    "risk_per_trade": 0.01,
    "symbols": [
      {"name": "EURUSD", "entry_timeframe": "M15", "structure_timeframe": "H1", "trend_timeframe": "H4"}
    ]
  }
}
```

## Step 4: Run the Bot

Start the bot with default settings:

```bash
python main.py
```

## Step 5: Monitor

Watch the console output for:
- Connection status
- Signal generation
- Trade executions
- Error messages

## Important Notes

- Start with demo account first
- Monitor the bot closely during initial runs
- Adjust parameters based on market conditions
- Check logs regularly for issues