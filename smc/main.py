import logging
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from collections import defaultdict
from bot_components import MetaTrader5Client, MarketDataProvider, SMCAnalyzer, SignalGenerator, PositionSizer, TradeExecutor, global_config

logging.basicConfig(
    level=global_config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(global_config.LOG_FILE, mode='w'), # mode='w' to overwrite log each run
        logging.StreamHandler()
    ]
)
main_logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, bot_config):
        self.config = bot_config
        self.mt5_client = MetaTrader5Client()
        self.market_provider = MarketDataProvider(self.mt5_client)
        
        self.smc_analyzer = SMCAnalyzer(
            swing_lookback=bot_config.SMC_SWING_LOOKBACK,
            structure_lookback=bot_config.SMC_STRUCTURE_LOOKBACK,
            pd_fib_level=bot_config.PREMIUM_DISCOUNT_FIB_LEVEL
        )
        # SignalGenerator is now initialized without symbol-specific point size
        self.signal_generator = SignalGenerator(
            smc_analyzer=self.smc_analyzer,
            sl_buffer_points=bot_config.SL_POINTS_DEFAULT,
            tp_rr_ratio=bot_config.TP_RR_RATIO,
            higher_timeframe=bot_config.HIGHER_TIMEFRAME,
            sl_atr_multiplier=bot_config.SL_ATR_MULTIPLIER            
        )
        self.position_sizer = PositionSizer(
            mt5_client=self.mt5_client,
            risk_percent_per_trade=bot_config.RISK_PER_TRADE_PERCENT
        )
        self.trade_executor = TradeExecutor(
            mt5_client=self.mt5_client,
            magic_prefix=bot_config.MAGIC_NUMBER_PREFIX
        )
        
        # Per-symbol state and data
        self.symbol_data = {}

    def _initialize_symbol_data(self):
        """Fetches and stores static data like point size for each symbol."""
        for symbol_name in self.config.SYMBOLS:
            symbol_info = self.mt5_client.get_symbol_info(symbol_name)
            if not symbol_info:
                main_logger.error(f"Failed to get symbol info for {symbol_name}. It will be skipped.")
                continue
            
            self.symbol_data[symbol_name] = {
                'is_position_open': False,
                'open_position_type': None,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'description': symbol_info.description
            }
        
        # Filter out symbols for which info could not be fetched
        self.config.SYMBOLS = [s for s in self.config.SYMBOLS if s in self.symbol_data]

    def _update_open_position_status(self, symbol):
        """Updates the open position status for a specific symbol."""
        if symbol not in self.symbol_data: 
            main_logger.warning(f"Symbol {symbol} not in symbol_data")
            return

        positions = self.mt5_client.get_current_positions(symbol=symbol)
        current_symbol_state = self.symbol_data[symbol]
        
        if not positions:
            if current_symbol_state['is_position_open']:
                main_logger.info(f"Position for {symbol} appears closed.")
            current_symbol_state['is_position_open'] = False
            current_symbol_state['open_position_type'] = None
        else:
            # Log all positions for debugging
            for pos in positions:
                main_logger.debug(f"Position found for {symbol}: Ticket={pos.ticket}, Type={pos.type}, Volume={pos.volume}, SL={pos.sl}, TP={pos.tp}")
            
            if not current_symbol_state['is_position_open']:
                 main_logger.info(f"Existing position found for {symbol} (Ticket: {positions[0].ticket}).")
            current_symbol_state['is_position_open'] = True
            current_symbol_state['open_position_type'] = "BUY" if positions[0].type == mt5.ORDER_TYPE_BUY else "SELL"

    def _check_spread(self, symbol):
        """Checks spread for a specific symbol."""
        if symbol not in self.symbol_data: 
            main_logger.warning(f"Symbol {symbol} not in symbol_data for spread check")
            return False

        ticker = self.mt5_client.get_symbol_ticker(symbol)
        symbol_info = self.mt5_client.get_symbol_info(symbol)
        
        if not ticker or not symbol_info:
            main_logger.warning(f"({symbol}) Could not get ticker/symbol_info for spread check.")
            return False 

        spread_points = int(round((ticker.ask - ticker.bid) / symbol_info.point))
        max_spread = self.config.MAX_SPREAD_POINTS 
        
        if spread_points > max_spread:
            main_logger.info(f"({symbol}) Spread {spread_points} points > max {max_spread}. Skipping.")
            return False
            
        main_logger.debug(f"({symbol}) Current spread: {spread_points} points (Ask={ticker.ask:.{self.symbol_data[symbol]['digits']}f}, Bid={ticker.bid:.{self.symbol_data[symbol]['digits']}f}).")
        return True
    
    def check_correlation_limit(self, new_symbol, new_direction):
        """Simple correlation protection"""
        # Define correlation groups
        correlation_groups = {
            'USD_LONGS': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
            'USD_SHORTS': ['USDCAD', 'USDCHF', 'USDJPY'],
            'EUR_LONGS': ['EURUSD', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD'],
            'EUR_SHORTS': ['EURAUD', 'EURCAD', 'EURNZD'],  # Selling these = Short EUR
            'GBP_LONGS': ['GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD'],
            'JPY_SHORTS': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY'],
        }
        
        # Get ALL positions (not just for current symbol)
        current_positions = self.mt5_client.get_current_positions()
        if not current_positions:
            return True  # No positions, no correlation risk
        
        # Count current exposure per group
        group_exposure = defaultdict(int)
        
        for pos in current_positions:
            for group, symbols in correlation_groups.items():
                if pos.symbol in symbols:
                    if 'LONGS' in group and pos.type == mt5.ORDER_TYPE_BUY:
                        group_exposure[group] += 1
                    elif 'SHORTS' in group and pos.type == mt5.ORDER_TYPE_SELL:
                        group_exposure[group] += 1
                    # Handle inverse correlations
                    elif 'LONGS' in group and pos.type == mt5.ORDER_TYPE_SELL:
                        group_exposure[group.replace('LONGS', 'SHORTS')] += 1
                    elif 'SHORTS' in group and pos.type == mt5.ORDER_TYPE_BUY:
                        group_exposure[group.replace('SHORTS', 'LONGS')] += 1
        
        # Check if new trade would exceed limit
        for group, symbols in correlation_groups.items():
            if new_symbol in symbols:
                if new_direction == "BUY" and 'LONGS' in group:
                    if group_exposure[group] >= 2:
                        main_logger.warning(f"({new_symbol}) Blocking BUY signal - Already have {group_exposure[group]} positions in {group}")
                        return False
                elif new_direction == "SELL" and 'SHORTS' in group:
                    if group_exposure[group] >= 2:
                        main_logger.warning(f"({new_symbol}) Blocking SELL signal - Already have {group_exposure[group]} positions in {group}")
                        return False
        
        # Log current exposure
        if group_exposure:
            main_logger.info(f"Current correlation exposure: {dict(group_exposure)}")
        
        return True

    def run_loop_iteration(self):
        """Runs one iteration of the trading loop for all symbols."""
        main_logger.debug(f"Starting loop iteration at {datetime.now()}")
        
        for symbol_name in self.config.SYMBOLS:
            main_logger.debug(f"Processing symbol: {symbol_name}")
            
            if symbol_name not in self.symbol_data or self.symbol_data[symbol_name].get('point') is None:
                main_logger.warning(f"Skipping {symbol_name} due to missing symbol data (point size).")
                continue

            self._update_open_position_status(symbol_name)
            symbol_state = self.symbol_data[symbol_name]

            if symbol_state['is_position_open']:
                main_logger.info(f"Position for {symbol_name} is {symbol_state['open_position_type']}. Waiting...")
                continue

            if not self._check_spread(symbol_name):
                continue

            # Calculate number of candles needed
            num_candles = self.config.SMC_STRUCTURE_LOOKBACK + self.config.SMC_SWING_LOOKBACK + 50
            main_logger.debug(f"({symbol_name}) Requesting {num_candles} candles on {self.config.TIMEFRAME_STR}")
            
            ohlc_df = self.market_provider.get_ohlc(
                symbol_name, self.config.TIMEFRAME_STR, num_candles
            )

            if ohlc_df is None or ohlc_df.empty:
                main_logger.warning(f"({symbol_name}) No OHLC data returned. Skipping.")
                continue
                
            if len(ohlc_df) < self.config.SMC_STRUCTURE_LOOKBACK:
                main_logger.warning(f"({symbol_name}) Insufficient OHLC data. Got {len(ohlc_df)}, need {self.config.SMC_STRUCTURE_LOOKBACK}. Skipping.")
                continue
            
            # Debug OHLC data
            main_logger.debug(f"({symbol_name}) OHLC shape: {ohlc_df.shape}, "
                            f"Columns: {list(ohlc_df.columns)}, "
                            f"Index type: {type(ohlc_df.index)}, "
                            f"Time range: {ohlc_df['time'].min()} to {ohlc_df['time'].max()}")
            
            # Ensure proper index
            if not isinstance(ohlc_df.index, pd.DatetimeIndex):
                main_logger.debug(f"({symbol_name}) Converting time column to DatetimeIndex")
                ohlc_df.set_index('time', inplace=True)

            # Pass the specific symbol's point size to generate method
            point_size = symbol_state['point']
            signal, sl_price, tp_price = self.signal_generator.generate(ohlc_df, point_size, symbol_name)

            if signal:
                main_logger.info(f"*** SIGNAL GENERATED *** {signal} for {symbol_name}, SL={sl_price:.{symbol_state['digits']}f}, TP={tp_price:.{symbol_state['digits']}f}")
                
                # Check correlation limit
                if not self.check_correlation_limit(symbol_name, signal):
                    main_logger.warning(f"({symbol_name}) Signal blocked due to correlation limits")
                    continue
                
                volume = self.position_sizer.calculate_volume(symbol_name, sl_price, signal)
                if volume is None or volume == 0:
                    main_logger.warning(f"({symbol_name}) Invalid volume ({volume}). Skipping trade.")
                    continue
                
                main_logger.info(f"({symbol_name}) Calculated volume: {volume} lots.")
                
                # Final validation before trade
                ticker = self.mt5_client.get_symbol_ticker(symbol_name)
                if ticker:
                    current_price = ticker.ask if signal == "BUY" else ticker.bid
                    main_logger.info(f"({symbol_name}) Final check - Current: {current_price:.{symbol_state['digits']}f}, "
                                   f"SL: {sl_price:.{symbol_state['digits']}f}, TP: {tp_price:.{symbol_state['digits']}f}")
                
                trade_result = self.trade_executor.place_market_order(
                    symbol_name, signal, volume, sl_price, tp_price
                )
                if trade_result:
                    main_logger.info(f"({symbol_name}) *** TRADE EXECUTED *** Order: {trade_result.order}, Deal: {trade_result.deal}")
                    symbol_state['is_position_open'] = True 
                    symbol_state['open_position_type'] = signal
                else:
                    main_logger.error(f"({symbol_name}) *** TRADE FAILED ***")
            
            time.sleep(0.1) # Small delay between symbols

    def run(self):
        main_logger.info("="*60)
        main_logger.info(f"Starting SMC Trading Bot")
        main_logger.info(f"Version: Professional SMC with Order Blocks, FVG, Liquidity")
        main_logger.info(f"Symbols: {len(self.config.SYMBOLS)}")
        main_logger.info(f"Timeframe: {self.config.TIMEFRAME_STR}")
        main_logger.info(f"Risk per trade: {self.config.RISK_PER_TRADE_PERCENT}%")
        main_logger.info("="*60)
        
        if not self.mt5_client.initialize():
            main_logger.error("Halting bot: MT5 initialization failed.")
            return

        self._initialize_symbol_data()
        if not self.config.SYMBOLS:
            main_logger.error("No symbols could be initialized. Halting bot.")
            self.mt5_client.shutdown()
            return
        
        try:
            iteration_count = 0
            while True:
                iteration_count += 1
                loop_start_time = time.time()
                
                main_logger.info(f"\n{'='*40} ITERATION {iteration_count} {'='*40}")
                main_logger.info(f"Loop start time: {datetime.now()}")
                
                # Check MT5 connection
                if not self.mt5_client.is_connected() or not self.mt5_client.get_terminal_info():
                    main_logger.error("MT5 connection lost. Attempting to re-initialize...")
                    self.mt5_client.shutdown() 
                    time.sleep(10) 
                    if not self.mt5_client.initialize():
                        main_logger.error("Re-initialization failed. Exiting.")
                        break
                    self._initialize_symbol_data()
                    if not self.config.SYMBOLS: 
                        main_logger.error("No symbols after re-initialization. Exiting.")
                        break
                
                # Run main trading logic
                self.run_loop_iteration()

                # Calculate sleep time
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.config.LOOP_SLEEP_SECONDS - loop_duration)
                main_logger.info(f"Loop iteration {iteration_count} completed in {loop_duration:.2f}s. "
                               f"Sleeping for {sleep_time:.2f}s until next iteration.")
                
                if sleep_time > 0:
                    main_logger.debug(f"Next iteration will start at approximately {datetime.fromtimestamp(time.time() + sleep_time)}")
                
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            main_logger.info("\nBot stopped by user (Ctrl+C).")
        except Exception as e:
            main_logger.error(f"Unhandled exception in TradingBot run loop: {e}", exc_info=True)
        finally:
            self.mt5_client.shutdown()
            main_logger.info(f"TradingBot has shut down. Total iterations: {iteration_count}")
            main_logger.info("="*60)

if __name__ == "__main__":
    main_logger.info("SMC Trading Bot starting...")
    bot = TradingBot(bot_config=global_config)
    bot.run()