"""
ICT Trading Bot - Main Entry Point
Implements the Inner Circle Trader methodology with proper narrative sequencing.
"""

import logging
import time
import MetaTrader5 as mt5
from datetime import datetime, timezone, time as time_obj
from collections import defaultdict

# Import our modules
from bot_components import (
    MetaTrader5Client, MarketDataProvider, PositionSizer, 
    TradeExecutor, SymbolManager
)
from ict_bot import ICTSignalGenerator
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

class ICTTradingBot:
    """
    Main trading bot that coordinates all components and follows ICT methodology.
    """
    
    def __init__(self):
        # Initialize components
        self.mt5_client = MetaTrader5Client()
        self.market_provider = MarketDataProvider(self.mt5_client)
        self.symbol_manager = SymbolManager(self.mt5_client)
        self.signal_generator = ICTSignalGenerator(config)
        self.position_sizer = PositionSizer(self.mt5_client, config.RISK_PER_TRADE_PERCENT)
        self.trade_executor = TradeExecutor(self.mt5_client, config.MAGIC_NUMBER_PREFIX)
        
        # Trading state
        self.daily_trades = 0
        self.last_trade_date = None
        self.session_trades = defaultdict(int)
        self.active_narratives = {}
        
    def initialize(self):
        """Initialize the bot and all components."""
        logger.info("="*60)
        logger.info("ICT Trading Bot - Initialization")
        logger.info("="*60)
        
        # Connect to MT5
        if not self.mt5_client.initialize():
            logger.error("Failed to initialize MT5 connection")
            return False
        
        # Initialize symbols
        valid_symbols = self.symbol_manager.initialize_symbols(config.SYMBOLS)
        if not valid_symbols:
            logger.error("No valid symbols to trade")
            return False
        
        logger.info(f"Initialized with {len(valid_symbols)} symbols")
        logger.info(f"Risk per trade: {config.RISK_PER_TRADE_PERCENT}%")
        logger.info(f"Max daily trades: {config.MAX_TRADES_PER_DAY}")
        logger.info("="*60)
        
        return True
    
    def check_daily_reset(self):
        """Reset daily counters if new trading day."""
        current_date = datetime.now().date()
        
        if self.last_trade_date != current_date:
            logger.info(f"New trading day: {current_date}")
            self.daily_trades = 0
            self.session_trades.clear()
            self.last_trade_date = current_date
    
    def can_trade(self):
        """Check if we can take more trades today."""
        if self.daily_trades >= config.MAX_TRADES_PER_DAY:
            logger.info(f"Daily trade limit reached ({self.daily_trades}/{config.MAX_TRADES_PER_DAY})")
            return False
        return True
    
    def check_correlations(self, trade_symbol, trade_direction):
        """
        Check if a new trade would exceed the maximum exposure for any single currency.
        """
        positions = self.mt5_client.get_current_positions()
        currency_exposure = defaultdict(int)

        # Map existing positions to currency exposure
        for pos in positions:
            base_curr, quote_curr = pos.symbol[:3], pos.symbol[3:]
            # Long position (BUY) = long base, short quote
            if pos.type == mt5.ORDER_TYPE_BUY:
                currency_exposure[base_curr] += 1
                currency_exposure[quote_curr] -= 1
            # Short position (SELL) = short base, long quote
            else:
                currency_exposure[base_curr] -= 1
                currency_exposure[quote_curr] += 1
        
        # Calculate the exposure of the PROPOSED trade
        base_curr, quote_curr = trade_symbol[:3], trade_symbol[3:]
        if trade_direction == "BUY":
            new_exposure = {'base': (base_curr, 1), 'quote': (quote_curr, -1)}
        else: # SELL
            new_exposure = {'base': (base_curr, -1), 'quote': (quote_curr, 1)}

        # Check if adding this trade would breach the limit for either currency
        for leg in ['base', 'quote']:
            currency, change = new_exposure[leg]
            # Check the absolute exposure, as we want to limit both long and short over-exposure
            if abs(currency_exposure[currency] + change) > config.MAX_CORRELATION_EXPOSURE:
                logger.warning(
                    f"Correlation limit for {currency} would be breached. "
                    f"Current exposure: {currency_exposure[currency]}, "
                    f"Proposed change: {change}. Trade on {trade_symbol} rejected."
                )
                return False
        
        return True
    
    def process_symbol(self, symbol):
        """Process a single symbol following ICT methodology."""
        logger.debug(f"\nProcessing {symbol}...")
        
        # Check if symbol has position
        self.symbol_manager.update_position_status(symbol)
        symbol_data = self.symbol_manager.get_symbol_data(symbol)
        
        if symbol_data['has_position']:
            logger.info(f"{symbol}: Position already open ({symbol_data['position_type']})")
            return
        
        # Check spread
        current_spread = self.symbol_manager.check_spread(symbol, config.MAX_SPREAD_POINTS)
        if current_spread is None: # Check if the spread was too high or failed to get
            return
        
        # Get market data - M15, Daily, AND H4 using unified method
        ohlc_df = self.market_provider.get_data(symbol, config.TIMEFRAME_STR, config.DATA_LOOKBACK)
        if ohlc_df is None or len(ohlc_df) < config.STRUCTURE_LOOKBACK:
            logger.warning(f"{symbol}: Insufficient M15 data")
            return
        
        # Get daily data for proper ICT bias analysis
        daily_df = self.market_provider.get_data(symbol, "D1", 50)
        if daily_df is None or len(daily_df) < 20:
            logger.warning(f"{symbol}: Insufficient daily data for bias analysis")
            return
        
        # Get H4 data for HTF analysis
        h4_df = self.market_provider.get_data(symbol, "H4", 100)
        if h4_df is None or len(h4_df) < 10:
            logger.warning(f"{symbol}: Insufficient H4 data for HTF analysis")
            # Don't return - continue without H4 data
            h4_df = None
        
        # Generate signal following ICT narrative with daily AND H4 data
        signal, sl_price, tp_price, narrative = self.signal_generator.generate_signal(
            ohlc_df, symbol, current_spread, daily_df, h4_df 
        )
        
        if signal:
            # Log the narrative
            self._log_narrative(symbol, narrative, signal, sl_price, tp_price)
            
            # Check correlations
            if not self.check_correlations(symbol, signal):
                return
            
            # Calculate position size
            volume = self.position_sizer.calculate_volume(symbol, sl_price, signal)
            if volume is None:
                logger.error(f"{symbol}: Failed to calculate position size")
                return
            
            # Place the trade
            comment = f"Good"
            result = self.trade_executor.place_market_order(
                symbol, signal, volume, sl_price, tp_price, comment
            )
            
            if result:
                logger.info(f"{symbol}: Trade executed successfully!")
                self.daily_trades += 1
                self.active_narratives[symbol] = narrative
                
                # Update session trades
                if narrative.in_killzone:
                    self.session_trades[narrative.killzone_name] += 1
            else:
                logger.error(f"{symbol}: Trade execution failed")
    
    def _log_narrative(self, symbol, narrative, signal, sl, tp):
        """Log the complete ICT narrative for the trade."""
        logger.info(f"\n{'='*50}")
        logger.info(f"ICT SIGNAL GENERATED: {symbol} - {signal}")
        logger.info(f"{'='*50}")
        logger.info(f"NARRATIVE:")
        logger.info(f"  Daily Bias: {narrative.daily_bias.upper()}")
        logger.info(f"  PO3 Phase: {narrative.po3_phase}")
        logger.info(f"  Kill Zone: {narrative.killzone_name or 'None'}")
        logger.info(f"  Entry Model: {narrative.entry_model}")
        logger.info(f"\n TRADE DETAILS:")
        logger.info(f"  Entry: {narrative.current_price:.5f}")
        logger.info(f"  Stop Loss: {sl:.5f}")
        logger.info(f"  Take Profit: {tp:.5f}")
        risk_reward = abs(tp - narrative.current_price) / abs(narrative.current_price - sl)
        logger.info(f"  Risk:Reward = 1:{risk_reward:.1f}")
        logger.info(f"{'='*50}\n")
    
    def run_cycle(self):
        """
        Run one complete trading cycle with proper NY-based kill zone filtering.
        """
        self.check_daily_reset()
        
        if not self.can_trade():
            return
        
        # Determine if we should be active based on Kill Zones
        require_killzone = getattr(config, 'REQUIRE_KILLZONE', True)

        if require_killzone:
            # Convert current UTC time to NY time for kill zone check
            current_utc = datetime.now(timezone.utc)
            current_ny = current_utc.astimezone(config.NY_TIMEZONE)
            current_ny_time = current_ny.time()
            
            in_any_killzone = False
            active_zone = None
            
            # Check London Kill Zone (2-5 AM NY)
            if time_obj(2, 0) <= current_ny_time < time_obj(5, 0):
                in_any_killzone = True
                active_zone = "LONDON KILL ZONE"
                
            # Check New York Kill Zone (8:30-11 AM NY)
            elif time_obj(8, 30) <= current_ny_time < time_obj(11, 0):
                in_any_killzone = True
                active_zone = "NEW YORK KILL ZONE"
                
            # Check London Close Kill Zone (3-5 PM NY)
            elif time_obj(15, 0) <= current_ny_time < time_obj(17, 0):
                in_any_killzone = True
                active_zone = "LONDON CLOSE KILL ZONE"
            
            if in_any_killzone:
                logger.info(f"Active Kill Zone: {active_zone} (NY Time: {current_ny_time.strftime('%H:%M')})")
            else:
                logger.debug(f"Outside kill zones. NY Time: {current_ny_time.strftime('%H:%M')}. Waiting...")
                return
        else:
            logger.debug("Killzone requirement disabled, processing all symbols")

        # If we are active, then process each symbol
        for symbol in self.symbol_manager.symbols:
            if not self.can_trade():
                break
            
            try:
                self.process_symbol(symbol)
            except Exception as e:
                logger.error(f"{symbol}: Error processing - {e}", exc_info=True)
            
            time.sleep(0.5)  # Small delay between symbols
    
    def run(self):
        """Main bot loop."""
        if not self.initialize():
            return
        
        logger.info("\n ICT Trading Bot Started\n")
        
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
        logger.info("\n Final Statistics:")
        logger.info(f"  Total trades today: {self.daily_trades}")
        logger.info(f"  Session breakdown: {dict(self.session_trades)}")
        
        self.mt5_client.shutdown()
        logger.info("\n ICT Trading Bot Shutdown Complete\n")

if __name__ == "__main__":
    bot = ICTTradingBot()
    bot.run()