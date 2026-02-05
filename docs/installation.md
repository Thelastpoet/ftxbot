# Installation Guide

This guide will walk you through installing and setting up the Forex Trading Bot.

## Prerequisites

Before installing the Forex Trading Bot, ensure you have:

- Python 3.8 or higher
- MetaTrader 5 terminal with an active trading account
- Internet connection
- Administrative privileges (on Windows, may be required for MT5 API)

## Downloading the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forex-trading-bot.git
   cd forex-trading-bot
   ```

2. Or download the ZIP file from the releases page and extract it.

## Setting Up Python Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## MetaTrader 5 Setup

1. Open your MetaTrader 5 terminal
2. Go to Tools → Options → Expert Advisors
3. Enable "Allow DLL imports"
4. Add your trading account if not already added
5. Ensure the terminal is connected to your broker

## Initial Configuration

1. Open the `config.json` file in a text editor
2. Review and adjust the trading settings according to your preferences
3. Add or modify the symbols you wish to trade
4. Set your risk management parameters appropriately

## Running the Bot

Once everything is set up, you can run the bot:

```bash
python main.py
```

## Verification

After starting, check that:
- The bot connects to MT5 successfully
- Market data is being fetched
- No immediate errors appear in the logs
- The bot begins analyzing the configured symbols