#!/usr/bin/env python3
"""
Forex Trading Bot - Main Application
Pure Price Action Strategy with MetaTrader 5 Integration
"""

import asyncio
import logging
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import asdict, is_dataclass
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the trading bot"""
    
    def __init__(self, config_path: str = 'config.json', args: Optional[argparse.Namespace] = None):
        self.config_path = Path(config_path)
        self.args = args
        self.load_config()
        
    def load_config(self):
        """Load configuration from JSON file and apply CLI overrides"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Trading settings
            trading = config_data.get('trading_settings', {})
            self.max_period = trading.get('max_period', 50)
            self.main_loop_interval = trading.get('main_loop_interval_seconds', 60)
            self.order_retry_attempts = trading.get('order_retry_attempts', 3)
            self.lookback_period = trading.get('lookback_period', 20)
            self.swing_window = trading.get('swing_window', 5)
            self.breakout_threshold = trading.get('breakout_threshold_pips', 5)
            self.min_peak_rank = trading.get('min_peak_rank', 2)
            self.proximity_threshold = trading.get('proximity_threshold_pips', 10)
            
            # Risk management
            risk = config_data.get('risk_management', {})
            self.risk_per_trade = risk.get('risk_per_trade', 0.02)
            self.fixed_lot_size = risk.get('fixed_lot_size', None)
            self.max_drawdown = risk.get('max_drawdown_percentage', 0.1)
            self.risk_reward_ratio = risk.get('risk_reward_ratio', 2.0)
            
            # Symbols and timeframes
            symbols_data = config_data.get('symbols', [])
            self.symbols = []
            for sym in symbols_data:
                self.symbols.append({
                    'name': sym['name'],
                    'timeframes': sym.get('timeframes', ['M15'])
                })
            
            # Trading sessions
            sessions_data = config_data.get('trading_sessions', [])
            self.trading_sessions = []
            for session in sessions_data:
                self.trading_sessions.append({
                    'name': session['name'],
                    'start_time': datetime.strptime(session['start_time'], '%H:%M').time(),
                    'end_time': datetime.strptime(session['end_time'], '%H:%M').time()
                })
            
            # Apply command-line overrides
            if self.args:
                if self.args.risk_per_trade:
                    self.risk_per_trade = self.args.risk_per_trade
                if self.args.symbol:
                    self.symbols = [{'name': self.args.symbol, 'timeframes': ['M15']}]
                if self.args.timeframe:
                    for sym in self.symbols:
                        sym['timeframes'] = [self.args.timeframe]
                        
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            self.set_defaults()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            self.set_defaults()
    
    def set_defaults(self):
        """Set default configuration values"""
        self.max_period = 50
        self.main_loop_interval = 60
        self.order_retry_attempts = 3
        self.lookback_period = 20
        self.swing_window = 5
        self.breakout_threshold = 5
        self.min_peak_rank = 2
        self.proximity_threshold = 10
        self.risk_per_trade = 0.02
        self.fixed_lot_size = None
        self.max_drawdown = 0.1
        self.risk_reward_ratio = 2.0
        self.symbols = [{'name': 'EURUSD', 'timeframes': ['M15']}]
        self.trading_sessions = []

class TradingSession:
    """Manages trading session times"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def is_trading_time(self) -> bool:
        """Check if current time is within any defined trading session"""
        if not self.config.trading_sessions:
            return True  # Trade 24/7 if no sessions defined
        
        current_time = datetime.now().time()
        
        for session in self.config.trading_sessions:
            start = session['start_time']
            end = session['end_time']
            
            # Handle sessions that cross midnight
            if start <= end:
                if start <= current_time <= end:
                    logger.debug(f"Within {session['name']} session")
                    return True
            else:
                if current_time >= start or current_time <= end:
                    logger.debug(f"Within {session['name']} session")
                    return True
        
        return False

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.trade_logger = None
        self.session_manager = TradingSession(config)
        self.running = False
        self.initial_balance = None
        self.last_processed_candle_time: Dict[Tuple[str, str], datetime] = {}
        
    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Import components (delayed to allow for proper module structure)
            from mt5_client import MetaTrader5Client
            from market_data import MarketData
            from strategy import PurePriceActionStrategy
            from risk_manager import RiskManager
            from trade_logger import TradeLogger
            
            # Initialize components
            self.mt5_client = MetaTrader5Client()
            if not self.mt5_client.initialized:
                raise Exception("Failed to initialize MetaTrader5 connection")
            
            self.market_data = MarketData(self.mt5_client, self.config)
            self.strategy = PurePriceActionStrategy(self.config)
            self.risk_manager = RiskManager(self.config, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')
            
            # Get initial account balance
            account_info = self.mt5_client.get_account_info()
            if account_info:
                self.initial_balance = account_info.balance
                logger.info(f"Initial account balance: {self.initial_balance}")
            
            logger.info("Trading bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def check_drawdown(self) -> bool:
        """Check if maximum drawdown has been reached"""
        if not self.initial_balance:
            return False
        
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            return False
        
        current_balance = account_info.balance
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        
        if drawdown >= self.config.max_drawdown:
            logger.warning(f"Maximum drawdown reached: {drawdown:.2%}")
            return True
        
        return False
    
    async def execute_trade(self, signal, symbol: str):
        """Execute a trade based on signal (accepts dataclass or dict)"""
        # allow both dataclass and dict for signal
        if is_dataclass(signal):
            signal = asdict(signal)

        try:
            # First check if we already have a position for this symbol
            existing_positions = self.mt5_client.get_positions(symbol)
            if existing_positions and len(existing_positions) > 2:
                logger.info(f"Position already exists for {symbol}, skipping new signal to let it work")
                return

            # Get symbol info (required for pip calculations & validations)
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}, aborting trade execution")
                return

            # Recompute stop loss pips using RiskManager (ensures consistent pip units)
            entry_price = float(signal.get('entry_price'))
            sl_price = float(signal.get('stop_loss'))
            tp_price = float(signal.get('take_profit'))

            # Use the risk manager helper to compute real pip distance
            computed_sl_pips = self.risk_manager.calculate_stop_loss_pips(entry_price, sl_price, symbol_info)

            # Log both price distance and pip distance clearly
            price_distance = abs(entry_price - sl_price)
            logger.info(
                f"Computed SL distance for {symbol}: price distance={price_distance:.8f}, "
                f"pip distance={computed_sl_pips:.4f} pips"
            )

            # Enforce minimum stop loss if configured
            min_sl_pips = getattr(self.config, 'min_stop_loss_pips', None)
            if min_sl_pips is not None and computed_sl_pips < min_sl_pips:
                logger.warning(
                    f"Stop loss too tight for {symbol}: {computed_sl_pips:.2f} pips < min_stop_loss_pips {min_sl_pips}. Skipping trade."
                )
                return

            # Check risk limits before executing
            if not self.risk_manager.check_risk_limits(symbol):
                logger.info(f"Risk limits prevent trading {symbol}")
                return

            # Calculate position size using the recomputed pip value
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=computed_sl_pips
            )

            if position_size <= 0:
                logger.warning(f"Invalid position size calculated for {symbol}")
                return

            # Validate trade parameters
            if not self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                volume=position_size,
                stop_loss=sl_price,
                take_profit=tp_price
            ):
                logger.warning(f"Trade parameters validation failed for {symbol}")
                return

            # Place order
            result = self.mt5_client.place_order(
                symbol=symbol,
                order_type=signal['type'],
                volume=position_size,
                sl=sl_price,
                tp=tp_price,
                comment=f"PPA_{signal.get('reason')}"
            )

            if result and getattr(result, 'retcode', None) == self.mt5_client.mt5.TRADE_RETCODE_DONE:
                # Log successful trade
                trade_details = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'order_type': 'BUY' if signal['type'] == 0 else 'SELL',
                    'entry_price': result.price,
                    'volume': position_size,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'ticket': result.order
                }
                self.trade_logger.log_trade(trade_details)
                logger.info(f"Trade executed successfully: {trade_details}")
            else:
                logger.error(f"Failed to execute trade: {result}")

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def process_symbol(self, symbol_config: Dict):
        """Process a single symbol for trading opportunities"""
        symbol = symbol_config['name']
        
        for timeframe in symbol_config['timeframes']:
            try:
                # Fetch market data
                num_candles_to_fetch = self.config.lookback_period + self.config.swing_window * 2 + 5
                data = await self.market_data.fetch_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    num_candles=num_candles_to_fetch
                )
                
                if data is None or data.empty:
                    logger.warning(f"No data available for {symbol} {timeframe}")
                    continue
                
                current_completed_candle_time = data.iloc[-2].name
                
                tracker_key = (symbol, timeframe)

                # Check if this completed candle has already been processed
                if tracker_key in self.last_processed_candle_time and \
                   self.last_processed_candle_time[tracker_key] == current_completed_candle_time:
                    logger.debug(f"No new completed candle for {symbol} {timeframe} (last processed: {current_completed_candle_time}). Skipping signal generation.")
                    continue # No new candle, skip to next symbol/timeframe

                # Update the tracker for this symbol/timeframe
                self.last_processed_candle_time[tracker_key] = current_completed_candle_time
                logger.debug(f"New completed candle detected for {symbol} {timeframe}: {current_completed_candle_time}. Proceeding with signal generation.")
                
                # Generate trading signal
                signal = self.strategy.generate_signal(data, symbol)
                
                if signal:
                    logger.info(f"Signal generated for {symbol}: {signal}")
                    
                    # Check if within trading session
                    if self.session_manager.is_trading_time():
                        await self.execute_trade(signal, symbol)
                    else:
                        logger.info("Outside trading session, signal ignored")
                        
            except Exception as e:
                logger.error(f"Error processing {symbol} {timeframe}: {e}")
                
    async def sync_trade_status(self):
        """Sync trade status using MT5's position checking"""
        if not self.trade_logger:
            return
            
        try:
            # Get trades that are marked as OPEN in our records
            open_trades = [t for t in self.trade_logger.trades if t.get('status') == 'OPEN']
            
            if not open_trades:
                return
                
            # Get all current positions from MT5
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {pos.ticket for pos in current_positions}
            
            # Check each open trade
            for trade in open_trades:
                ticket = trade.get('ticket')
                if ticket and ticket not in current_tickets:
                    # Position was closed externally
                    logger.info(f"Detected externally closed position: {ticket} for {trade.get('symbol', 'unknown')}")
                    
                    # Try to get exit details from history
                    end_time = datetime.now()
                    start_time = trade.get('timestamp', end_time - timedelta(days=7))
                    
                    deals = self.mt5_client.get_history_deals(start_time, end_time)
                    exit_price = 0
                    profit = 0
                    
                    # Find the closing deal
                    for deal in deals:
                        if hasattr(deal, 'position_id') and deal.position_id == ticket:
                            if hasattr(deal, 'price'):
                                exit_price = deal.price
                            if hasattr(deal, 'profit'):
                                profit = deal.profit
                            break
                    
                    # Update trade status
                    self.trade_logger.update_trade(ticket, exit_price, profit, 'CLOSED_EXTERNAL')
                    
        except Exception as e:
            logger.error(f"Error syncing trade status: {e}")
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting main trading loop")
        
        while self.running:
            try:
                # Check for maximum drawdown
                if await self.check_drawdown():
                    logger.error("Maximum drawdown reached, stopping bot")
                    break
                
                # Sync trade status with MT5
                await self.sync_trade_status()                
                # Process each symbol
                for symbol_config in self.config.symbols:
                    await self.process_symbol(symbol_config)
                
                # Wait for next iteration
                await asyncio.sleep(self.config.main_loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.config.main_loop_interval)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot")
        self.running = False
        
        # Positions should be managed according to their stop loss and take profit
        if self.mt5_client:
            positions = self.mt5_client.get_all_positions()
            if positions:
                logger.info(f"Note: {len(positions)} positions remain open and will be managed by their SL/TP")
                for position in positions:
                    logger.info(f"Open position: {position.symbol} - Volume: {position.volume} - Profit: {position.profit}")
        
        logger.info("Trading bot shutdown complete")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Forex Trading Bot with Pure Price Action Strategy')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--risk-per-trade', type=float,
                       help='Override risk percentage per trade')
    parser.add_argument('--symbol', type=str,
                       help='Override trading symbol')
    parser.add_argument('--timeframe', type=str,
                       help='Override primary timeframe')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = Config(args.config, args)
    
    # Create and run bot
    bot = TradingBot(config)
    
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)