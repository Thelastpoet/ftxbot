"""
SMC Trading Bot - Main Entry Point
"""

import logging
import time
import MetaTrader5 as mt5
from datetime import datetime
from collections import defaultdict

# Import our modules
from bot_components import (
    MetaTrader5Client, MarketDataProvider, PositionSizer, 
    TradeExecutor, SymbolManager
)
from smc_core import SMCTradingSystem
import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SMCTradingBot:
    """
    Main trading bot implementing core Smart Money Concepts.
    """
    
    def __init__(self):
        # Initialize components
        self.mt5_client = MetaTrader5Client()
        self.market_provider = MarketDataProvider(self.mt5_client)
        self.symbol_manager = SymbolManager(self.mt5_client)
        self.smc_system = None  # Will be initialized after MT5 connection
        self.position_sizer = PositionSizer(self.mt5_client, config.RISK_PER_TRADE_PERCENT)
        self.trade_executor = TradeExecutor(self.mt5_client, config.MAGIC_NUMBER_PREFIX)
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date = None
        self.session_performance = defaultdict(lambda: {'trades': 0, 'pnl': 0})
        self.active_setups = {}  # Track setups for each symbol
        
    def initialize(self):
        """Initialize the bot and all components."""
        logger.info("="*60)
        logger.info("SMC Trading Bot - Initialization")
        logger.info("Smart Money Concepts - Simplified & Optimized")
        logger.info("="*60)
        
        # Connect to MT5
        if not self.mt5_client.initialize():
            logger.error("Failed to initialize MT5 connection")
            return False
        
        # Initialize SMC system after MT5 connection
        self.smc_system = SMCTradingSystem(self.mt5_client, config)
        
        # Initialize symbols
        valid_symbols = self.symbol_manager.initialize_symbols(config.SYMBOLS)
        if not valid_symbols:
            logger.error("No valid symbols to trade")
            return False
        
        logger.info(f"Initialized with {len(valid_symbols)} symbols")
        logger.info(f"Risk per trade: {config.RISK_PER_TRADE_PERCENT}%")
        logger.info(f"Max daily trades: {config.MAX_TRADES_PER_DAY}")
        logger.info(f"Minimum confidence: {config.MIN_CONFIDENCE}")
        logger.info(f"Minimum R:R ratio: {config.MIN_RR_RATIO}")
        logger.info("="*60)
        
        return True
    
    def check_daily_reset(self):
        """Reset daily counters if new trading day."""
        current_date = datetime.now().date()
        
        if self.last_trade_date != current_date:
            logger.info(f"New trading day: {current_date}")
            self.daily_trades = 0
            self.session_performance.clear()
            self.last_trade_date = current_date
    
    def can_trade(self):
        """Check if we can take more trades today."""
        if self.daily_trades >= config.MAX_TRADES_PER_DAY:
            logger.info(f"Daily trade limit reached ({self.daily_trades}/{config.MAX_TRADES_PER_DAY})")
            return False
        return True
    
    def check_correlations(self, trade_symbol, trade_direction):
        """
        Check if a new trade would exceed correlation limits.
        Simplified version focusing on major currency exposure.
        """
        positions = self.mt5_client.get_current_positions()
        
        # Extract base and quote currencies
        base_curr = trade_symbol[:3]
        quote_curr = trade_symbol[3:6]
        
        # Count existing exposure
        base_exposure = 0
        quote_exposure = 0
        
        for pos in positions:
            pos_base = pos.symbol[:3]
            pos_quote = pos.symbol[3:6]
            
            if pos.type == mt5.ORDER_TYPE_BUY:
                if pos_base == base_curr:
                    base_exposure += 1
                if pos_quote == quote_curr:
                    quote_exposure -= 1
            else:  # SELL
                if pos_base == base_curr:
                    base_exposure -= 1
                if pos_quote == quote_curr:
                    quote_exposure += 1
        
        # Check if new trade would exceed limits
        if trade_direction == "BUY":
            if abs(base_exposure + 1) > config.MAX_CORRELATION_EXPOSURE:
                logger.warning(f"{trade_symbol}: Would exceed correlation limit for {base_curr}")
                return False
            if abs(quote_exposure - 1) > config.MAX_CORRELATION_EXPOSURE:
                logger.warning(f"{trade_symbol}: Would exceed correlation limit for {quote_curr}")
                return False
        else:  # SELL
            if abs(base_exposure - 1) > config.MAX_CORRELATION_EXPOSURE:
                logger.warning(f"{trade_symbol}: Would exceed correlation limit for {base_curr}")
                return False
            if abs(quote_exposure + 1) > config.MAX_CORRELATION_EXPOSURE:
                logger.warning(f"{trade_symbol}: Would exceed correlation limit for {quote_curr}")
                return False
        
        return True
    
    def is_active_session(self):
        """
        Check if we're in an active trading session.
        SMC works best during liquid market hours.
        """
        current_hour = datetime.now().hour
        
        # Define liquid trading hours (can be customized)
        # London: 8-16 UTC
        # NY: 13-21 UTC
        # Overlap: 13-16 UTC (best liquidity)
        
        if 8 <= current_hour <= 21:  # Covers both sessions
            return True
        
        return False
    
    def process_symbol(self, symbol):
        """Process a single symbol for SMC setups."""
        logger.debug(f"\nProcessing {symbol}...")
        
        # Check if symbol has position
        self.symbol_manager.update_position_status(symbol)
        symbol_data = self.symbol_manager.get_symbol_data(symbol)
        
        if symbol_data['has_position']:
            logger.debug(f"{symbol}: Position already open ({symbol_data['position_type']})")
            return
        
        # Check if we recently traded this symbol (avoid overtrading)
        if symbol in self.active_setups:
            last_signal_time = self.active_setups[symbol].get('time')
            if last_signal_time:
                time_since_signal = (datetime.now() - last_signal_time).total_seconds() / 60
                if time_since_signal < config.MIN_MINUTES_BETWEEN_TRADES:
                    logger.debug(f"{symbol}: Too soon since last signal ({time_since_signal:.1f} min)")
                    return
        
        # Check spread
        current_spread = self.symbol_manager.check_spread(symbol, config.MAX_SPREAD_POINTS)
        if current_spread is None:
            return
        
        # Get market data
        ohlc_df = self.market_provider.get_ohlc(symbol, config.TIMEFRAME_STR, config.DATA_LOOKBACK)
        if ohlc_df is None or len(ohlc_df) < 100:
            logger.warning(f"{symbol}: Insufficient data")
            return
        
        # Analyze and potentially trade
        result = self.smc_system.analyze_and_trade(
            symbol, ohlc_df, self.position_sizer, self.trade_executor
        )
        
        if result:
            self.daily_trades += 1
            self.active_setups[symbol] = {
                'setup': result['setup'],
                'time': datetime.now()
            }
            
            # Log session performance
            current_hour = datetime.now().hour
            if 8 <= current_hour < 13:
                session = 'London'
            elif 13 <= current_hour < 16:
                session = 'Overlap'
            elif 16 <= current_hour <= 21:
                session = 'NY'
            else:
                session = 'Other'
            
            self.session_performance[session]['trades'] += 1
    
    def monitor_positions(self):
        """
        Monitor open positions for management opportunities.
        This is where we could add trailing stops or partial profits.
        """
        positions = self.mt5_client.get_current_positions()
        
        for pos in positions:
            # Check if this is one of our positions (by magic number)
            if str(pos.magic).startswith(str(config.MAGIC_NUMBER_PREFIX)):
                symbol = pos.symbol
                
                # Get current price
                ticker = self.mt5_client.get_symbol_ticker(symbol)
                if not ticker:
                    continue
                
                current_price = ticker.bid if pos.type == mt5.ORDER_TYPE_BUY else ticker.ask
                
                # Calculate position P&L in pips
                if pos.type == mt5.ORDER_TYPE_BUY:
                    pnl_pips = (current_price - pos.price_open) / self.symbol_manager.symbols[symbol]['point']
                else:
                    pnl_pips = (pos.price_open - current_price) / self.symbol_manager.symbols[symbol]['point']
                
                # Log significant moves
                if abs(pnl_pips) > 100:
                    logger.info(f"{symbol}: Position P&L: {pnl_pips:.0f} pips")
    
    def run_cycle(self):
        """Run one complete trading cycle."""
        self.check_daily_reset()
        
        if not self.can_trade():
            return
        
        # Check if we're in active session
        """
        if not self.is_active_session():
            logger.debug("Outside active trading hours. Monitoring only...")
            self.monitor_positions()
            return
        """
        logger.info(f"Active session - Processing {len(self.symbol_manager.symbols)} symbols")
        
        # Process each symbol
        for symbol in self.symbol_manager.symbols:
            if not self.can_trade():
                break
            
            try:
                self.process_symbol(symbol)
            except Exception as e:
                logger.error(f"{symbol}: Error processing - {e}", exc_info=True)
            
            time.sleep(0.5)  # Small delay between symbols
        
        # Monitor existing positions
        self.monitor_positions()
    
    def log_performance_summary(self):
        """Log daily performance summary."""
        logger.info("\n" + "="*50)
        logger.info("DAILY PERFORMANCE SUMMARY")
        logger.info("="*50)
        logger.info(f"Total trades: {self.daily_trades}")
        logger.info("\nSession breakdown:")
        for session, stats in self.session_performance.items():
            logger.info(f"  {session}: {stats['trades']} trades")
        logger.info("="*50)
    
    def run(self):
        """Main bot loop."""
        if not self.initialize():
            return
        
        logger.info("\n SMC Trading Bot Started\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                start_time = time.time()
                
                logger.info(f"\n{'='*30} Cycle {iteration} {'='*30}")
                logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check connection
                if not self.mt5_client.is_connected():
                    logger.error("MT5 connection lost. Attempting reconnect...")
                    self.mt5_client.shutdown()
                    time.sleep(5)
                    if not self.mt5_client.initialize():
                        logger.error("Failed to reconnect. Exiting.")
                        break
                
                # Run trading cycle
                self.run_cycle()
                
                # Log performance every 10 cycles
                if iteration % 10 == 0:
                    self.log_performance_summary()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, config.LOOP_SLEEP_SECONDS - elapsed)
                
                if sleep_time > 0:
                    logger.info(f"Cycle completed in {elapsed:.1f}s. "
                               f"Sleeping for {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of the bot."""
        logger.info("\n Shutting down SMC Trading Bot...")
        
        # Final performance summary
        self.log_performance_summary()
        
        # Log open positions
        positions = self.mt5_client.get_current_positions()
        if positions:
            logger.info(f"\nOpen positions: {len(positions)}")
            for pos in positions:
                logger.info(f"  {pos.symbol}: {pos.type_description} @ {pos.price_open}")
        
        self.mt5_client.shutdown()
        logger.info("\n SMC Trading Bot Shutdown Complete\n")


if __name__ == "__main__":
    bot = SMCTradingBot()
    bot.run()