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
from typing import Dict, Optional
import sys

from strategy import TradingSignal

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
            self.min_stop_loss_pips = trading.get('min_stop_loss_pips', 20)
            self.stop_loss_buffer_pips = trading.get('stop_loss_buffer_pips', 15)
            self.min_range_pips = trading.get('min_range_pips', 10)
            self.atr_period = trading.get('atr_period', 14)
            self.atr_tp_multiplier = trading.get('atr_tp_multiplier', 4.0)
            self.session_based_tp = trading.get('session_based_tp', True)
            self.volatility_adjustment = trading.get('volatility_adjustment', True)
            
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
        self.risk_reward_ratio = 1.25
        self.symbols = [{'name': 'EURUSD', 'timeframes': ['M15']}]
        self.trading_sessions = []
        self.min_stop_loss_pips = 20
        self.stop_loss_buffer_pips = 15
        self.min_range_pips = 10
        self.atr_period = 14
        self.atr_tp_multiplier = 4.0
        self.session_based_tp = True
        self.volatility_adjustment = True

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
    
    async def execute_trade(self, signal: TradingSignal, symbol: str):
        """Execute a trade based on signal"""
        try:
            # Get symbol info (required for pip calculations & validations)
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}, aborting trade execution")
                return

            # Direct attribute access from dataclass
            tick = self.mt5_client.get_symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get live tick for {symbol}")
                return
                
            # Use live prices based on order direction
            live_entry_price = tick.ask if signal.type == 0 else tick.bid
            logger.info(f"Live entry price for {symbol}: {live_entry_price:.5f} (signal was {signal.entry_price:.5f})")
            sl_price = float(signal.stop_loss)
            tp_price = float(signal.take_profit)

            # Use the risk manager helper to compute real pip distance
            computed_sl_pips = self.risk_manager.calculate_stop_loss_pips(
                live_entry_price, sl_price, symbol_info
            )

            # Log both price distance and pip distance clearly
            price_distance = abs(live_entry_price - sl_price)
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
            if not self.risk_manager.check_risk_limits(symbol, signal.type):
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

            # Place order - using dataclass attributes directly
            result = self.mt5_client.place_order(
                symbol=symbol,
                order_type=signal.type,  # Direct attribute access
                volume=position_size,
                sl=sl_price,
                tp=tp_price,
                comment=f"PPA_{signal.reason}"  # Direct attribute access
            )

            if result and getattr(result, 'retcode', None) == self.mt5_client.mt5.TRADE_RETCODE_DONE:
                # Log successful trade with all signal details
                trade_details = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'order_type': 'BUY' if signal.type == 0 else 'SELL',
                    'entry_price': result.price,
                    'volume': position_size,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'ticket': result.order,
                    'reason': signal.reason,  # Add signal reason
                    'confidence': signal.confidence,  # Add confidence
                    'signal_time': signal.timestamp  # Add when signal was generated
                }
                self.trade_logger.log_trade(trade_details)
                logger.info(
                    f"Trade executed successfully: {symbol} "
                    f"{'BUY' if signal.type == 0 else 'SELL'} "
                    f"@ {result.price:.5f}, Volume: {position_size:.2f}, "
                    f"Ticket: {result.order}"
                )
            else:
                error_msg = getattr(result, 'comment', 'Unknown error') if result else 'No result returned'
                logger.error(f"Failed to execute trade for {symbol}: {error_msg}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)
                
    async def process_symbol(self, symbol_config: Dict):
        """
        Process a single symbol for trading opportunities
        Multi-timeframe coordination with trend alignment
        """
        symbol = symbol_config['name']
        
        # Early check: if we already have a position for this symbol, skip all calculations
        existing_positions = self.mt5_client.get_positions(symbol)
        if existing_positions and len(existing_positions) >= 1:
            logger.debug(f"Position already exists for {symbol}, skipping signal generation")
            return
        
        try:
            # Fetch multi-timeframe data
            mtf_data = await self.market_data.fetch_multi_timeframe_data(symbol)
            
            if not mtf_data:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Use H1 for trend, M15 for signal timing
            if 'H1' in mtf_data and 'M15' in mtf_data:
                # Get H1 trend direction using existing function
                h1_trend = self.market_data.identify_trend(mtf_data['H1'])
                logger.debug(f"{symbol} H1 trend: {h1_trend}")
                
                # Generate M15 signal with trend context
                m15_signal = self.strategy.generate_signal(mtf_data['M15'], symbol, trend=h1_trend)
                
                if m15_signal:
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if m15_signal.type == 0 else 'SELL'}, "
                        f"Confidence={m15_signal.confidence:.2f}, "
                        f"H1 Trend={h1_trend}"
                    )
                    
                    # Check if within trading session
                    if self.session_manager.is_trading_time():
                        await self.execute_trade(m15_signal, symbol)
                    else:
                        logger.info("Outside trading session, signal ignored")
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            # FALLBACK: Single timeframe processing (if only one timeframe available)
            elif 'M15' in mtf_data:
                # Use M15 data for both trend and signal
                m15_data = mtf_data['M15']
                m15_trend = self.market_data.identify_trend(m15_data)
                
                signal = self.strategy.generate_signal(m15_data, symbol, trend=m15_trend)
                
                if signal:
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )
                    
                    if self.session_manager.is_trading_time():
                        await self.execute_trade(signal, symbol)
                    else:
                        logger.info("Outside trading session, signal ignored")
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            elif 'H1' in mtf_data:
                # Use H1 data for both trend and signal
                h1_data = mtf_data['H1']
                h1_trend = self.market_data.identify_trend(h1_data)
                
                signal = self.strategy.generate_signal(h1_data, symbol, trend=h1_trend)
                
                if signal:
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )
                    
                    if self.session_manager.is_trading_time():
                        await self.execute_trade(signal, symbol)
                    else:
                        logger.info("Outside trading session, signal ignored")
                else:
                    logger.debug(f"No signal generated for {symbol}")
                        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
    async def sync_trade_status(self):
        """Trade status synchronization"""
        if not self.trade_logger:
            return
            
        try:
            open_trades = [t for t in self.trade_logger.trades 
                        if t.get('status') == 'OPEN']
            
            if not open_trades:
                return
                
            # Get current positions
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {pos.ticket for pos in current_positions}
            
            for trade in open_trades:
                ticket = trade.get('ticket')
                if ticket and ticket not in current_tickets:
                    # Position was closed
                    logger.info(f"Position {ticket} closed, fetching details...")
                    
                    # Get the closing deal information
                    end_time = datetime.now()
                    start_time = datetime.fromisoformat(trade['timestamp'])
                    
                    # Get history of deals
                    deals = self.mt5_client.get_history_deals(start_time, end_time)
                    
                    exit_price = 0
                    profit = 0
                    commission = 0
                    
                    # Find ALL deals related to this position
                    for deal in deals:
                        # Check multiple fields for matching
                        if (hasattr(deal, 'position_id') and deal.position_id == ticket) or \
                        (hasattr(deal, 'order') and deal.order == ticket):
                            
                            # Closing deals have DEAL_ENTRY_OUT
                            if hasattr(deal, 'entry') and deal.entry == 1:  # DEAL_ENTRY_OUT
                                exit_price = deal.price
                                profit += deal.profit
                                commission += getattr(deal, 'commission', 0)
                    
                    # If we couldn't find exit price, calculate from current price
                    if exit_price == 0:
                        current_positions = self.mt5_client.get_all_positions()
                        open_pos = any(pos.ticket == ticket for pos in current_positions)
                        
                        if open_pos:
                            logger.debug(f"Position {ticket} still open, skipping sync")
                            continue
                        
                        extended_end_time = datetime.now() + timedelta(days=1)
                        extended_start_time = datetime.fromisoformat(trade['timestamp']) - timedelta(hours=1)

                        extended_deals = self.mt5_client.get_history_deals(extended_start_time, extended_end_time)
                        
                        for deal in extended_deals:
                            if (hasattr(deal, 'position_id') and deal.position_id == ticket) or \
                            (hasattr(deal, 'order') and deal.order == ticket):
                                if hasattr(deal, 'entry') and deal.entry == 1:  # DEAL_ENTRY_OUT
                                    exit_price = deal.price
                                    profit += deal.profit
                                    commission += getattr(deal, 'commission', 0)
                                    logger.info(f"Found exit for position {ticket} in extended history")
                                    break
                                
                    if exit_price == 0:
                        logger.error(f"Could not find exit price for closed position {ticket}. "
                            f"This trade will not be properly logged. Consider manual review.")
                        
                        # Mark it with a special status so we know it needs review
                        self.trade_logger.update_trade(ticket, 0, 0 - commission, 'CLOSED_UNKNOWN')
                        continue

                    # Update trade record
                    self.trade_logger.update_trade(
                        ticket, 
                        exit_price, 
                        profit - commission, 
                        'CLOSED_SL' if abs(exit_price - trade['stop_loss']) < 0.0001 else 
                        'CLOSED_TP' if abs(exit_price - trade['take_profit']) < 0.0001 else 
                        'CLOSED_MANUAL'
                    )
                    
        except Exception as e:
            logger.error(f"Error syncing trade status: {e}", exc_info=True)
    
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