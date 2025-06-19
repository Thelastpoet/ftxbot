import logging
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from collections import defaultdict
from ict_bot import MetaTrader5Client, MarketDataProvider, SMCAnalyzer, SignalGenerator, PositionSizer, TradeExecutor, global_config

logging.basicConfig(
    level=global_config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(global_config.LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
main_logger = logging.getLogger(__name__)

class ICTTradingBot:
    """Enhanced Trading Bot implementing true ICT/SMC concepts."""
    
    def __init__(self, bot_config):
        self.config = bot_config
        self.mt5_client = MetaTrader5Client()
        self.market_provider = MarketDataProvider(self.mt5_client)
        
        self.smc_analyzer = SMCAnalyzer(
            swing_lookback=bot_config.SMC_SWING_LOOKBACK,
            structure_lookback=bot_config.SMC_STRUCTURE_LOOKBACK,
            pd_fib_level=bot_config.PREMIUM_DISCOUNT_FIB_LEVEL
        )
        
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
        
        # ICT-specific tracking
        self.session_trades = defaultdict(int)  # Track trades per session
        self.daily_trades = 0
        self.daily_risk_used = 0.0
        self.last_trade_day = None
        self.killzone_trades = defaultdict(int)  # Track trades per kill zone

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
                'description': symbol_info.description,
                'last_signal_time': None,  # Prevent duplicate signals
                'last_judas_time': None     # Track Judas Swing timing
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
            
        main_logger.debug(f"({symbol}) Current spread: {spread_points} points.")
        return True
    
    def _update_daily_tracking(self):
        """Reset daily tracking at start of new trading day."""
        current_date = datetime.now().date()
        
        if self.last_trade_day != current_date:
            self.daily_trades = 0
            self.daily_risk_used = 0.0
            self.last_trade_day = current_date
            self.session_trades.clear()
            self.killzone_trades.clear()
            main_logger.info(f"New trading day {current_date}: Daily counters reset")
    
    def _check_risk_limits(self):
        """Check if we've hit risk limits."""
        # Daily trade limit
        if self.daily_trades >= self.config.MAX_TRADES_PER_DAY:
            main_logger.info(f"Daily trade limit reached ({self.daily_trades}/{self.config.MAX_TRADES_PER_DAY})")
            return False
            
        # Daily risk limit
        if self.daily_risk_used >= self.config.MAX_RISK_PER_SESSION:
            main_logger.info(f"Daily risk limit reached ({self.daily_risk_used:.1f}%/{self.config.MAX_RISK_PER_SESSION}%)")
            return False
            
        return True
    
    def _check_killzone_limits(self, killzone_name):
        """Check if we've hit kill zone trade limits."""
        if not killzone_name:
            return True
            
        if self.killzone_trades[killzone_name] >= self.config.MAX_TRADES_PER_SESSION:
            main_logger.info(f"Kill zone trade limit reached for {killzone_name} "
                           f"({self.killzone_trades[killzone_name]}/{self.config.MAX_TRADES_PER_SESSION})")
            return False
            
        return True

    def check_correlation_limit(self, new_symbol, new_direction):
        """Enhanced correlation protection for ICT trading."""
        # Define correlation groups
        correlation_groups = {
            'USD_LONGS': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
            'USD_SHORTS': ['USDCAD', 'USDCHF', 'USDJPY'],
            'EUR_LONGS': ['EURUSD', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD'],
            'EUR_SHORTS': ['EURAUD', 'EURCAD', 'EURNZD'],
            'GBP_LONGS': ['GBPUSD', 'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD'],
            'JPY_SHORTS': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'CHFJPY'],
        }
        
        if not self.config.CORRELATION_FILTER:
            return True
        
        current_positions = self.mt5_client.get_current_positions()
        if not current_positions:
            return True
        
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
                    if group_exposure[group] >= self.config.MAX_CORRELATION_EXPOSURE:
                        main_logger.warning(f"({new_symbol}) Blocking BUY - Already have {group_exposure[group]} positions in {group}")
                        return False
                elif new_direction == "SELL" and 'SHORTS' in group:
                    if group_exposure[group] >= self.config.MAX_CORRELATION_EXPOSURE:
                        main_logger.warning(f"({new_symbol}) Blocking SELL - Already have {group_exposure[group]} positions in {group}")
                        return False
        
        if group_exposure:
            main_logger.info(f"Current correlation exposure: {dict(group_exposure)}")
        
        return True

    def _validate_ict_signal(self, signal, symbol, smc_analysis):
        """Validate signal based on ICT trade filters."""
        filters = self.config.TRADE_FILTERS
        
        # Check kill zone requirement
        if filters['require_killzone'] and not smc_analysis.get('in_killzone'):
            main_logger.info(f"({symbol}) Signal rejected: Not in ICT kill zone")
            return False
        
        # Check session requirement
        if filters['require_session'] and not smc_analysis.get('in_session'):
            main_logger.info(f"({symbol}) Signal rejected: No active session")
            return False
        
        # Check structure requirement
        if filters['require_structure']:
            structure_df = smc_analysis.get('structure')
            if structure_df is None or 'BOS' not in structure_df.columns:
                main_logger.info(f"({symbol}) Signal rejected: No market structure")
                return False
        
        # Check premium/discount requirement
        if filters['require_pd_zone']:
            current_price = smc_analysis['ohlc_df']['close'].iloc[-1]
            eq = smc_analysis.get('equilibrium')
            if signal == "BUY" and current_price > eq:
                main_logger.info(f"({symbol}) BUY signal rejected: Price in premium")
                return False
            elif signal == "SELL" and current_price < eq:
                main_logger.info(f"({symbol}) SELL signal rejected: Price in discount")
                return False
        
        return True

    def run_loop_iteration(self):
        """Runs one iteration of the ICT trading loop."""
        main_logger.debug(f"Starting ICT loop iteration at {datetime.now()}")
        
        # Update daily tracking
        self._update_daily_tracking()
        
        # Check risk limits
        if not self._check_risk_limits():
            main_logger.info("Risk limits reached for today. Waiting for next day...")
            return
        
        for symbol_name in self.config.SYMBOLS:
            main_logger.debug(f"Processing symbol: {symbol_name}")
            
            if symbol_name not in self.symbol_data or self.symbol_data[symbol_name].get('point') is None:
                main_logger.warning(f"Skipping {symbol_name} due to missing symbol data.")
                continue

            self._update_open_position_status(symbol_name)
            symbol_state = self.symbol_data[symbol_name]

            if symbol_state['is_position_open']:
                main_logger.info(f"Position for {symbol_name} is {symbol_state['open_position_type']}. Monitoring...")
                continue

            if not self._check_spread(symbol_name):
                continue

            # Calculate number of candles needed for ICT analysis
            num_candles = self.config.SMC_STRUCTURE_LOOKBACK + self.config.SMC_SWING_LOOKBACK + 100
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
            
            # Ensure proper index
            if not isinstance(ohlc_df.index, pd.DatetimeIndex):
                main_logger.debug(f"({symbol_name}) Converting time column to DatetimeIndex")
                ohlc_df.set_index('time', inplace=True)

            # Get comprehensive ICT analysis
            main_logger.debug(f"({symbol_name}) Running comprehensive ICT analysis...")
            smc_analysis = self.smc_analyzer.get_comprehensive_analysis(ohlc_df, higher_timeframe=self.config.HIGHER_TIMEFRAME)
            
            # Log ICT-specific findings
            if smc_analysis.get('power_of_three'):
                main_logger.info(f"({symbol_name}) Power of Three detected: {smc_analysis['power_of_three']['type']}")
            
            if smc_analysis.get('market_maker_model'):
                main_logger.info(f"({symbol_name}) Market Maker Model active: {smc_analysis['market_maker_model']['type']}")
            
            if smc_analysis.get('judas_swing'):
                main_logger.info(f"({symbol_name}) Judas Swing detected: {smc_analysis['judas_swing']['type']}")
            
            # Check kill zone limits
            killzone_name = smc_analysis.get('killzone_name')
            if killzone_name and not self._check_killzone_limits(killzone_name):
                continue

            # Generate signal using ICT methodology
            point_size = symbol_state['point']
            signal, sl_price, tp_price = self.signal_generator.generate(ohlc_df, point_size, symbol_name)

            if signal:
                # Validate signal with ICT filters
                if not self._validate_ict_signal(signal, symbol_name, smc_analysis):
                    continue
                
                # Prevent duplicate signals
                current_time = datetime.now()
                if symbol_state['last_signal_time'] and \
                   (current_time - symbol_state['last_signal_time']).seconds < 900:  # 15 min cooldown
                    main_logger.info(f"({symbol_name}) Signal cooldown active. Skipping.")
                    continue
                
                main_logger.info(f"*** ICT SIGNAL *** {signal} for {symbol_name}, "
                               f"SL={sl_price:.{symbol_state['digits']}f}, TP={tp_price:.{symbol_state['digits']}f}")
                
                # Log ICT entry model used
                if smc_analysis.get('judas_swing'):
                    main_logger.info(f"({symbol_name}) Entry Model: JUDAS SWING REVERSAL")
                elif smc_analysis.get('market_maker_model'):
                    main_logger.info(f"({symbol_name}) Entry Model: MARKET MAKER MODEL")
                elif smc_analysis.get('in_killzone'):
                    main_logger.info(f"({symbol_name}) Entry Model: ICT KILL ZONE - {killzone_name}")
                else:
                    main_logger.info(f"({symbol_name}) Entry Model: OTE/ORDER BLOCK CONFLUENCE")
                
                # Check correlation limit
                if not self.check_correlation_limit(symbol_name, signal):
                    main_logger.warning(f"({symbol_name}) Signal blocked due to correlation limits")
                    continue
                
                # Calculate position size
                volume = self.position_sizer.calculate_volume(symbol_name, sl_price, signal)
                if volume is None or volume == 0:
                    main_logger.warning(f"({symbol_name}) Invalid volume ({volume}). Skipping trade.")
                    continue
                
                main_logger.info(f"({symbol_name}) Calculated volume: {volume} lots.")
                
                # Final validation
                ticker = self.mt5_client.get_symbol_ticker(symbol_name)
                if ticker:
                    current_price = ticker.ask if signal == "BUY" else ticker.bid
                    main_logger.info(f"({symbol_name}) Final check - Current: {current_price:.{symbol_state['digits']}f}, "
                                   f"SL: {sl_price:.{symbol_state['digits']}f}, TP: {tp_price:.{symbol_state['digits']}f}")
                
                # Execute trade
                trade_result = self.trade_executor.place_market_order(
                    symbol_name, signal, volume, sl_price, tp_price
                )
                
                if trade_result:
                    main_logger.info(f"({symbol_name}) *** ICT TRADE EXECUTED *** Order: {trade_result.order}, Deal: {trade_result.deal}")
                    
                    # Update tracking
                    symbol_state['is_position_open'] = True 
                    symbol_state['open_position_type'] = signal
                    symbol_state['last_signal_time'] = current_time
                    
                    self.daily_trades += 1
                    self.daily_risk_used += self.config.RISK_PER_TRADE_PERCENT
                    
                    if killzone_name:
                        self.killzone_trades[killzone_name] += 1
                    
                    # Log trade statistics
                    main_logger.info(f"Daily trades: {self.daily_trades}/{self.config.MAX_TRADES_PER_DAY}, "
                                   f"Daily risk: {self.daily_risk_used:.1f}%/{self.config.MAX_RISK_PER_SESSION}%")
                else:
                    main_logger.error(f"({symbol_name}) *** TRADE FAILED ***")
            
            # Small delay between symbols
            time.sleep(0.1)

    def run(self):
        main_logger.info("="*80)
        main_logger.info(f"Starting ICT/SMC Professional Trading Bot")
        main_logger.info(f"Version: True ICT Implementation with PO3, MM Models, OTE, Judas Swings")
        main_logger.info(f"Symbols: {len(self.config.SYMBOLS)}")
        main_logger.info(f"Timeframe: {self.config.TIMEFRAME_STR}")
        main_logger.info(f"Risk per trade: {self.config.RISK_PER_TRADE_PERCENT}%")
        main_logger.info(f"Max daily trades: {self.config.MAX_TRADES_PER_DAY}")
        main_logger.info(f"ICT Kill Zones: {list(self.config.ICT_KILL_ZONES.keys())}")
        main_logger.info("="*80)
        
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
                
                main_logger.info(f"\n{'='*50} ICT ITERATION {iteration_count} {'='*50}")
                
                # Get current time and check if in kill zone
                current_time = datetime.now()
                current_hour = current_time.hour
                active_killzone = None
                
                for kz_name, kz_config in self.config.ICT_KILL_ZONES.items():
                    if kz_config['start'] <= current_hour < kz_config['end']:
                        active_killzone = kz_name
                        break
                
                main_logger.info(f"Current time: {current_time}, Kill Zone: {active_killzone or 'None'}")
                
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
                
                # Run main ICT trading logic
                self.run_loop_iteration()

                # Calculate sleep time
                loop_duration = time.time() - loop_start_time
                sleep_time = max(0, self.config.LOOP_SLEEP_SECONDS - loop_duration)
                
                main_logger.info(f"Loop iteration {iteration_count} completed in {loop_duration:.2f}s. "
                               f"Sleeping for {sleep_time:.2f}s until next iteration.")
                
                if sleep_time > 0:
                    next_check = datetime.fromtimestamp(time.time() + sleep_time)
                    main_logger.debug(f"Next iteration at {next_check}")
                
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            main_logger.info("\nBot stopped by user (Ctrl+C).")
        except Exception as e:
            main_logger.error(f"Unhandled exception in ICT TradingBot: {e}", exc_info=True)
        finally:
            self.mt5_client.shutdown()
            main_logger.info(f"ICT Trading Bot has shut down. Total iterations: {iteration_count}")
            main_logger.info(f"Total trades today: {self.daily_trades}, Total risk used: {self.daily_risk_used:.1f}%")
            main_logger.info("="*80)

if __name__ == "__main__":
    main_logger.info("ICT/SMC Professional Trading Bot starting...")
    bot = ICTTradingBot(bot_config=global_config)
    bot.run()