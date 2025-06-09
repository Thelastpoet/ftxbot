import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback
from talib import ATR

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
MAGIC_NUMBER = 234000
EA_COMMENT = "MTF EA"
DEFAULT_SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY',
    'EURAUD', 'EURJPY', 'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDUSD']
DEFAULT_TIMEFRAMES = (mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1)
CHECK_INTERVAL = 300 

class MetaTrader5Client:
    def __init__(self):
        self.initialized = mt5.initialize()
        if not self.initialized:
            logging.error("Failed to initialize MT5")

    def is_initialized(self):
        return self.initialized

    def get_account_info(self):
        return mt5.account_info()

class MarketData:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes
        
        # Select symbol
        if not mt5.symbol_select(self.symbol, True):
            raise ValueError(f"Failed to select symbol {self.symbol}")
        
    def fetch_data(self, timeframe):
        """Fetch historical data for the given timeframe"""            
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 1000)
        if rates is None or len(rates) == 0:
            logging.error(f"No rates available for {self.symbol} on timeframe {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        df['tick_volume'] = df['tick_volume'].replace(0, 1)
                        
        return df    
    
    def get_pip_size(self):
        """Get pip size for the symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                return symbol_info.point * 10
            else:
                return symbol_info.point
        return 0.0001  # Default
                                          
class TradeManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.order_manager = OrderManager(client, market_data)
        
        # Initialize timeframe hierarchy
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] 
        self.tf_lower = min(self.timeframes)
        
        # Position limits
        self.max_positions = 10
        self.max_per_symbol = 2
        
    def find_swing_points(self, data, window=10):
        """Find confirmed swing highs and lows using only historical data"""
        df = data.copy()
        
        # Only analyze bars that can be confirmed (exclude last 'window' bars)
        confirmed_length = len(df) - window
        if confirmed_length < window * 2 + 1:
            return pd.DataFrame(), pd.DataFrame()
        
        swing_highs = []
        swing_lows = []
        
        # Only check bars that are fully confirmed
        for i in range(window, confirmed_length):
            is_swing_high = True
            is_swing_low = True
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check surrounding bars (all historical)
            for j in range(1, window + 1):
                # Check left side
                if df.iloc[i - j]['high'] >= current_high:
                    is_swing_high = False
                if df.iloc[i - j]['low'] <= current_low:
                    is_swing_low = False
                    
                # Check right side (now safe because we excluded recent bars)
                if df.iloc[i + j]['high'] >= current_high:
                    is_swing_high = False
                if df.iloc[i + j]['low'] <= current_low:
                    is_swing_low = False
            
            if is_swing_high:
                swing_highs.append({'index': i, 'price': current_high, 'time': df.index[i]})
            if is_swing_low:
                swing_lows.append({'index': i, 'price': current_low, 'time': df.index[i]})
        
        return pd.DataFrame(swing_highs), pd.DataFrame(swing_lows)

    def analyze_higher_timeframe(self, data):
        """Higher Timeframe - Clear trend identification using proper swing analysis"""
        try:
            df = data.copy()
            
            # Find swing points with appropriate window for H1
            swing_highs, swing_lows = self.find_swing_points(df, window=10)
            
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return {'trend': 'unclear', 'strength': 0}
            
            # Get last few swings for analysis
            recent_highs = swing_highs.tail(3)['price'].values
            recent_lows = swing_lows.tail(3)['price'].values
            
            # Analyze trend based on swing progression
            trend = 'unclear'
            strength = 0
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Check for higher highs and higher lows (uptrend)
                hh_count = sum(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
                hl_count = sum(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
                
                # Check for lower highs and lower lows (downtrend)
                lh_count = sum(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
                ll_count = sum(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
                
                # Determine trend with strength
                if hh_count >= 1 and hl_count >= 1:
                    trend = 'uptrend'
                    strength = (hh_count + hl_count) / 4.0  # Normalize to 0-1
                elif lh_count >= 1 and ll_count >= 1:
                    trend = 'downtrend'
                    strength = (lh_count + ll_count) / 4.0
                
                # Additional confirmation with moving averages
                ma50 = df['close'].rolling(50).mean().iloc[-1]
                ma200 = df['close'].rolling(200).mean().iloc[-1]
                current_price = df['close'].iloc[-1]
                
                if trend == 'uptrend' and current_price > ma50 > ma200:
                    strength = min(strength + 0.2, 1.0)
                elif trend == 'downtrend' and current_price < ma50 < ma200:
                    strength = min(strength + 0.2, 1.0)
                    
            return {
                'trend': trend, 
                'strength': strength,
                'last_swing_high': swing_highs.iloc[-1]['price'] if len(swing_highs) > 0 else None,
                'last_swing_low': swing_lows.iloc[-1]['price'] if len(swing_lows) > 0 else None
            }
            
        except Exception as e:
            logging.error(f"Error in analyze_higher_timeframe: {str(e)}")
            return {'trend': 'unclear', 'strength': 0}

    def analyze_medium_timeframe(self, data, htf_context):
        """Detect valid pullback setups in trending markets"""
        try:
            df = data.copy()
            
            if len(df) < 50:  # Need sufficient data for analysis
                return {'valid_setup': False, 'fib_level': None}
            
            trend = htf_context['trend']
            if trend == 'unclear':
                return {'valid_setup': False, 'fib_level': None}
            
            # Find recent swing points for Fibonacci calculation
            swing_highs, swing_lows = self.find_swing_points(df, window=5)  # Smaller window for M15
            
            if len(swing_highs) == 0 or len(swing_lows) == 0:
                return {'valid_setup': False, 'fib_level': None}
            
            current_price = df['close'].iloc[-1]
            spread = self.get_current_spread()
            
            if trend == 'uptrend':
                # Find the most recent significant swing high and low
                if len(swing_highs) == 0:
                    return {'valid_setup': False, 'fib_level': None}
                    
                recent_high = swing_highs.iloc[-1]['price']
                recent_high_index = swing_highs.iloc[-1]['index']
                
                # Find the low before this high
                low_before_high = swing_lows[swing_lows['index'] < recent_high_index]
                
                if len(low_before_high) == 0:
                    return {'valid_setup': False, 'fib_level': None}
                    
                recent_low = low_before_high.iloc[-1]['price']
                swing_range = recent_high - recent_low
                
                # Calculate Fibonacci levels
                fib_levels = {
                    '0.382': recent_high - (swing_range * 0.382),
                    '0.500': recent_high - (swing_range * 0.500),
                    '0.618': recent_high - (swing_range * 0.618)
                }
                
                # Check if we're at a Fibonacci level with dynamic tolerance
                pip_size = self.market_data.get_pip_size()
                tolerance = max(5 * pip_size, swing_range * 0.02)  # 5 pips or 2% of range
                
                for level_name, level_price in fib_levels.items():
                    if abs(current_price - level_price) <= tolerance + spread:
                        # Additional confirmation: price should be pulling back from recent high
                        bars_since_high = len(df) - swing_highs.iloc[-1]['index']
                        if 3 <= bars_since_high <= 20:  # Reasonable pullback duration
                            return {
                                'valid_setup': True, 
                                'fib_level': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance)
                            }
                            
            elif trend == 'downtrend':
                if len(swing_lows) == 0:
                    return {'valid_setup': False, 'fib_level': None}
                    
                recent_low = swing_lows.iloc[-1]['price']
                recent_low_index = swing_lows.iloc[-1]['index']
                
                # Find the high before this low
                high_before_low = swing_highs[swing_highs['index'] < recent_low_index]
                
                if len(high_before_low) == 0:
                    return {'valid_setup': False, 'fib_level': None}
                    
                recent_high = high_before_low.iloc[-1]['price']
                swing_range = recent_high - recent_low
                
                # Calculate Fibonacci levels for downtrend (measured from low)
                fib_levels = {
                    '0.382': recent_low + (swing_range * 0.382),
                    '0.500': recent_low + (swing_range * 0.500),
                    '0.618': recent_low + (swing_range * 0.618)
                }
                
                pip_size = self.market_data.get_pip_size()
                tolerance = max(5 * pip_size, swing_range * 0.02)
                
                for level_name, level_price in fib_levels.items():
                    if abs(current_price - level_price) <= tolerance + spread:
                        bars_since_low = len(df) - swing_lows.iloc[-1]['index']
                        if 3 <= bars_since_low <= 20:
                            return {
                                'valid_setup': True, 
                                'fib_level': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance)
                            }
            
            return {'valid_setup': False, 'fib_level': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_medium_timeframe: {str(e)}")
            return {'valid_setup': False, 'fib_level': None}

    def analyze_lower_timeframe(self, data, trade_direction):
        """Detect Break of Structure (BOS) for precise entries - wait for candle close"""
        try:
            df = data.copy()
            
            if len(df) < 20:
                return {'valid_entry': False, 'entry_type': None}
            
            # Find recent swing points on M5
            swing_highs, swing_lows = self.find_swing_points(df, window=3)  # Tight window for M5
            
            if len(swing_highs) == 0 or len(swing_lows) == 0:
                return {'valid_entry': False, 'entry_type': None}
            
            # Look at the last CLOSED candle (not current)
            last_closed = df.iloc[-2]  # -1 is current, -2 is last closed
            prev_candle = df.iloc[-3]
            
            # Add spread consideration
            spread = self.get_current_spread()
            
            if trade_direction == 'buy':
                # Get the most recent swing high
                last_swing_high = swing_highs.iloc[-1]['price']
                
                # Check if last closed candle broke above swing high BY MORE THAN SPREAD
                break_occurred = (
                    prev_candle['close'] <= last_swing_high and  
                    last_closed['close'] > last_swing_high + spread and  # Must break by more than spread
                    last_closed['close'] > last_closed['open']           # Bullish candle
                )
                
                if break_occurred:
                    # Calculate candle strength
                    candle_range = last_closed['high'] - last_closed['low']
                    if candle_range > 0:
                        body_size = abs(last_closed['close'] - last_closed['open'])
                        candle_strength = body_size / candle_range
                        
                        if candle_strength > 0.6:  # Strong bullish candle
                            # Verify current price still above break level
                            current_price = df.iloc[-1]['close']
                            if current_price > last_swing_high:
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_resistance',
                                    'break_level': last_swing_high
                                }
                                
            else:  # sell
                # Get the most recent swing low
                last_swing_low = swing_lows.iloc[-1]['price']
                
                # Check if last closed candle broke below swing low BY MORE THAN SPREAD
                break_occurred = (
                    prev_candle['close'] >= last_swing_low and   
                    last_closed['close'] < last_swing_low - spread and   # Must break by more than spread
                    last_closed['close'] < last_closed['open']           # Bearish candle
                )
                
                if break_occurred:
                    candle_range = last_closed['high'] - last_closed['low']
                    if candle_range > 0:
                        body_size = abs(last_closed['close'] - last_closed['open'])
                        candle_strength = body_size / candle_range
                        
                        if candle_strength > 0.6:  # Strong bearish candle
                            # Verify current price still below break level
                            current_price = df.iloc[-1]['close']
                            if current_price < last_swing_low:
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_support',
                                    'break_level': last_swing_low
                                }
            
            return {'valid_entry': False, 'entry_type': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_lower_timeframe: {str(e)}")
            return {'valid_entry': False, 'entry_type': None}

    def get_current_spread(self):
        """Get current spread in price units"""
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        if symbol_info:
            return symbol_info.spread * symbol_info.point
        return 0

    def check_spread_acceptable(self):
        """Check if spread is acceptable for trading"""
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        if not symbol_info:
            return False
            
        spread_points = symbol_info.spread
        
        # Define max acceptable spread per symbol type
        symbol = self.market_data.symbol
        if 'JPY' in symbol:
            max_spread = 30  # 3 pips for JPY pairs
        else:
            max_spread = 30  # 3 pips for other pairs
            
        return spread_points <= max_spread

    def check_for_signals(self, symbol):
        """Main signal checking method with synchronized data fetching"""
        try:
            # Check spread first
            if not self.check_spread_acceptable():
                return False, f"[{symbol}] Spread too high", None
                
            # Check position limits
            positions = mt5.positions_get()
            if positions:
                total_positions = len(positions)
                symbol_positions = len([p for p in positions if p.symbol == symbol])
                
                if total_positions >= self.max_positions:
                    return False, f"Max total positions ({self.max_positions}) reached", None
                    
                if symbol_positions >= self.max_per_symbol:
                    return False, f"Max positions for {symbol} ({self.max_per_symbol}) reached", None

            # Fetch all data at once to minimize time gaps
            data = {}
            fetch_time = time.time()
            for tf in self.timeframes:
                data[tf] = self.market_data.fetch_data(tf)
                if data[tf] is None:
                    logging.warning(f"[{symbol}] Failed to fetch data for timeframe {tf}")
                    return False, f"[{symbol}] No data for timeframe {tf}", None
            
            # Ensure data fetching was quick (< 1 second)
            if time.time() - fetch_time > 1:
                logging.warning(f"[{symbol}] Data fetching took too long, skipping")
                return False, f"[{symbol}] Data synchronization issue", None

            # 1. Higher timeframe trend
            htf_analysis = self.analyze_higher_timeframe(data[self.tf_higher])
            if htf_analysis['trend'] == 'unclear' or htf_analysis['strength'] < 0.5:
                return False, f"[{symbol}] No clear/strong trend on HTF", None
            
            logging.info(f"[{symbol}] HTF trend: {htf_analysis['trend']} (strength: {htf_analysis['strength']:.2f})")
                        
            # 2. Medium timeframe setup
            mtf_analysis = self.analyze_medium_timeframe(data[self.tf_medium], htf_analysis)
            if not mtf_analysis['valid_setup']:
                return False, f"[{symbol}] No valid Fibonacci pullback on MTF", None

            logging.info(f"[{symbol}] MTF pullback to {mtf_analysis['fib_level']} level")

            # 3. Lower timeframe entry
            trade_direction = 'buy' if htf_analysis['trend'] == 'uptrend' else 'sell'
            ltf_analysis = self.analyze_lower_timeframe(data[self.tf_lower], trade_direction)
            if not ltf_analysis['valid_entry']:
                return False, f"[{symbol}] No break of structure on LTF", None

            logging.info(f"[{symbol}] LTF {ltf_analysis['entry_type']} confirmed!")

            # All conditions met - prepare trade
            account_info = self.client.get_account_info()
            if account_info is None:
                logging.error(f"[{symbol}] Failed to get account info")
                return False, f"[{symbol}] Could not get account info", None
            
            # Use talib ATR
            high = data[self.tf_higher]['high'].values
            low = data[self.tf_higher]['low'].values
            close = data[self.tf_higher]['close'].values
            
            atr_values = ATR(high, low, close, timeperiod=14)
            atr_value = atr_values[-1]
            
            if np.isnan(atr_value) or atr_value <= 0:
                return False, f"[{symbol}] Invalid ATR calculation", None
            
            entry_analysis = {
                'ltf_break_level': ltf_analysis.get('break_level'),
                'mtf_fib_zone': mtf_analysis.get('entry_zone'),  # (lower, upper) from Fib analysis
                'htf_swings': {
                    'last_swing_high': htf_analysis.get('last_swing_high'),
                    'last_swing_low': htf_analysis.get('last_swing_low')
                },
                'current_session': self.order_manager.get_current_session()
            }
            
            # Calculate trade parameters
            stop_loss, take_profit, stop_loss_pips = self.order_manager.calculate_sl_tp(
                symbol,
                trade_direction,
                atr_value,
                entry_analysis
            )
            
            # Dynamic position sizing based on setup quality
            base_risk = 0.01  # 1% base risk
            
            # Adjust risk based on setup quality
            if htf_analysis['strength'] > 0.7 and mtf_analysis['fib_level'] == '0.618':
                risk_percent = base_risk * 1.2  # Increase risk for high-quality setups
            elif htf_analysis['strength'] < 0.6 or mtf_analysis['fib_level'] == '0.382':
                risk_percent = base_risk * 0.8  # Decrease risk for lower-quality setups
            else:
                risk_percent = base_risk
                
            lots_size = self.order_manager.calculate_lot_size(
                symbol, 
                account_info, 
                stop_loss_pips,
                risk_percent=risk_percent
            )

            logging.info(f"[{symbol}] Risk: {risk_percent*100:.1f}%, Lots: {lots_size}")

            # Execute trade
            result = self.order_manager.place_order(
                symbol,
                lots_size,
                trade_direction,
                stop_loss,
                take_profit
            )
                    
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_info = {
                    'direction': trade_direction,
                    'entry_price': result.price,
                    'lot_size': lots_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'stop_loss_pips': stop_loss_pips,
                    'atr': atr_value,
                    'setup_quality': {
                        'trend_strength': htf_analysis['strength'],
                        'fib_level': mtf_analysis['fib_level'],
                        'entry_type': ltf_analysis['entry_type']
                    }
                }
                logging.info(f"[{symbol}] ✓ Trade executed: {trade_direction} {lots_size} lots at {result.price:.5f}")
                return True, f"[{symbol}] Trade executed successfully", trade_info
            else:
                error_msg = result.comment if result else "Unknown error"
                logging.error(f"[{symbol}] ✗ Order failed: {error_msg}")
                return False, f"[{symbol}] Order failed: {error_msg}", None

        except Exception as e:
            logging.error(f"[{symbol}] Error in check_for_signals: {str(e)}")
            logging.error(traceback.format_exc())
            return False, f"[{symbol}] Error: {str(e)}", None
        
class OrderManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        
    def get_symbol_info(self, symbol):
        """Get symbol info with pip calculation utilities"""
        symbol_tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        
        if not symbol_tick or not symbol_info:
            raise ValueError(f"Could not get info for {symbol}")
        
        pip_size = symbol_info.point * 10 if symbol_info.digits == 5 or symbol_info.digits == 3 else symbol_info.point
        pip_value_per_lot = symbol_info.trade_tick_value * (pip_size / symbol_info.trade_tick_size)
        
        return {
            'tick': symbol_tick,
            'info': symbol_info,
            'pip_size': pip_size,
            'pip_value_per_lot': pip_value_per_lot
        }
        
    def calculate_sl_tp(self, symbol, order_type, atr_value, sl_multiplier=2.1, tp_multiplier=3.0):
        """Calculate SL and TP with proper R:R ratio"""
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        pip_size = symbol_data['pip_size']
        
        # Get current price and spread
        if order_type == 'buy':
            current_price = symbol_tick.ask
            stop_distance = atr_value * sl_multiplier
            profit_distance = atr_value * tp_multiplier
            
            stop_loss = current_price - stop_distance
            take_profit = current_price + profit_distance
        else:  # sell
            current_price = symbol_tick.bid
            stop_distance = atr_value * sl_multiplier
            profit_distance = atr_value * tp_multiplier
            
            stop_loss = current_price + stop_distance
            take_profit = current_price - profit_distance
            
        # Calculate stop loss in pips
        stop_loss_pips = stop_distance / pip_size
                    
        return stop_loss, take_profit, stop_loss_pips
    
    def calculate_sl_tp(self, symbol, order_type, atr_value, entry_analysis):
        """
        Professional SL/TP calculation based on market structure and reality
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell' 
            atr_value: Current ATR value from H1
            entry_analysis: Dict containing structure levels from entry analysis
                - ltf_break_level: The M5 break point that triggered entry
                - mtf_fib_zone: The M15 Fibonacci zone we entered from
                - htf_swings: Recent H1 swing highs/lows
                - current_session: 'asian', 'london', 'newyork'
        """
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        symbol_info = symbol_data['info']
        pip_size = symbol_data['pip_size']
        
        # Get current spread in price units
        spread = symbol_info.spread * symbol_info.point
        
        # Determine pair volatility characteristics
        if 'JPY' in symbol:
            volatility_multiplier = 1.5  # JPY pairs move more
            base_noise_buffer = 15 * pip_size
        elif any(curr in symbol for curr in ['GBP', 'AUD']):
            volatility_multiplier = 1.3  # Commonwealth pairs are volatile
            base_noise_buffer = 12 * pip_size
        else:
            volatility_multiplier = 1.0  # Majors like EURUSD
            base_noise_buffer = 10 * pip_size
        
        # Adjust noise buffer based on session
        session = entry_analysis.get('current_session', 'normal')
        if session == 'london_open':  # 07:00-09:00 GMT
            noise_buffer = base_noise_buffer * 1.5
        elif session == 'newyork_open':  # 13:00-15:00 GMT
            noise_buffer = base_noise_buffer * 1.8
        elif session == 'asian':
            noise_buffer = base_noise_buffer * 0.7
        else:
            noise_buffer = base_noise_buffer
        
        # Get structure levels from entry analysis
        ltf_break = entry_analysis.get('ltf_break_level')
        mtf_fib_zone = entry_analysis.get('mtf_fib_zone')  # (lower, upper)
        htf_swings = entry_analysis.get('htf_swings', {})
        
        if order_type == 'buy':
            current_price = symbol_tick.ask
            
            # STOP LOSS: Structure-based with reality adjustments
            # Option 1: Below the M5 structure that triggered entry
            structure_stop = ltf_break - noise_buffer - spread
            
            # Option 2: Below the M15 Fib zone we entered from
            if mtf_fib_zone:
                fib_stop = mtf_fib_zone[0] - (5 * pip_size) - spread
                structure_stop = min(structure_stop, fib_stop)  # Use the higher/safer stop
            
            # Option 3: ATR-based maximum (disaster prevention)
            atr_stop = current_price - (atr_value * 0.75 * volatility_multiplier)
            
            # Use the tighter stop (closer to entry) but not too tight
            stop_loss = max(structure_stop, atr_stop)
            
            # Ensure minimum stop distance (broker requirement + sanity)
            min_stop_distance = max(
                symbol_info.trade_stops_level * symbol_info.point,  # Broker minimum
                20 * pip_size  # Our minimum (20 pips)
            )
            if current_price - stop_loss < min_stop_distance:
                stop_loss = current_price - min_stop_distance
            
            # TAKE PROFIT: Realistic targets
            # Primary target: Next M15/H1 resistance
            if htf_swings.get('last_swing_high'):
                resistance_target = htf_swings['last_swing_high'] - (3 * pip_size) - spread
            else:
                # Fallback: Use ATR projection
                resistance_target = current_price + (atr_value * 1.2 * volatility_multiplier)
            
            # Ensure minimum 1.5:1 RR ratio
            min_profit = (current_price - stop_loss) * 1.5
            take_profit = max(resistance_target, current_price + min_profit)
            
            # But cap at realistic day trading target
            max_profit = atr_value * 2.0 * volatility_multiplier
            take_profit = min(take_profit, current_price + max_profit)
            
        else:  # SELL
            current_price = symbol_tick.bid
            
            # STOP LOSS: Structure-based with reality adjustments
            structure_stop = ltf_break + noise_buffer + spread
            
            if mtf_fib_zone:
                fib_stop = mtf_fib_zone[1] + (5 * pip_size) + spread
                structure_stop = max(structure_stop, fib_stop)
            
            atr_stop = current_price + (atr_value * 0.75 * volatility_multiplier)
            stop_loss = min(structure_stop, atr_stop)
            
            # Ensure minimum stop distance
            min_stop_distance = max(
                symbol_info.trade_stops_level * symbol_info.point,
                20 * pip_size
            )
            if stop_loss - current_price < min_stop_distance:
                stop_loss = current_price + min_stop_distance
            
            # TAKE PROFIT: Realistic targets
            if htf_swings.get('last_swing_low'):
                support_target = htf_swings['last_swing_low'] + (3 * pip_size) + spread
            else:
                support_target = current_price - (atr_value * 1.2 * volatility_multiplier)
            
            # Ensure minimum 1.5:1 RR ratio
            min_profit = (stop_loss - current_price) * 1.5
            take_profit = min(support_target, current_price - min_profit)
            
            # Cap at realistic target
            max_profit = atr_value * 2.0 * volatility_multiplier
            take_profit = max(take_profit, current_price - max_profit)
        
        # Normalize prices to tick size
        tick_size = symbol_info.trade_tick_size
        stop_loss = round(stop_loss / tick_size) * tick_size
        take_profit = round(take_profit / tick_size) * tick_size
        
        # Calculate actual risk metrics
        stop_distance = abs(current_price - stop_loss)
        profit_distance = abs(take_profit - current_price)
        actual_rr_ratio = profit_distance / stop_distance if stop_distance > 0 else 0
        stop_loss_pips = stop_distance / pip_size
        
        # Log the decision process
        logging.info(f"[{symbol}] SL/TP Calculation:")
        logging.info(f"  - Spread: {spread/pip_size:.1f} pips")
        logging.info(f"  - Noise buffer: {noise_buffer/pip_size:.1f} pips")
        logging.info(f"  - Stop: {stop_loss:.5f} ({stop_loss_pips:.1f} pips)")
        logging.info(f"  - Target: {take_profit:.5f}")
        logging.info(f"  - Actual RR: 1:{actual_rr_ratio:.1f}")
        
        return stop_loss, take_profit, stop_loss_pips

    def get_current_session(self):
        """Determine current trading session for volatility adjustments"""
        from datetime import datetime
        
        current_hour = datetime.now().hour  # This is server time
        
        # Adjust these based on your server timezone
        if 7 <= current_hour < 9:
            return 'london_open'
        elif 13 <= current_hour < 15:
            return 'newyork_open'
        elif 0 <= current_hour < 7:
            return 'asian'
        elif 20 <= current_hour <= 23:
            return 'sydney'
        else:
            return 'normal'
        
    def calculate_lot_size(self, symbol, account_info, stop_loss_pips, risk_percent=0.01):
        """
        Professional lot size calculation with real risk management
        
        Args:
            symbol: Trading symbol
            account_info: MT5 account info
            stop_loss_pips: Stop loss distance in pips
            risk_percent: Base risk percentage (will be adjusted)
        """
        # Get symbol specifics
        symbol_data = self.get_symbol_info(symbol)
        symbol_info = symbol_data['info']
        pip_value_per_lot = symbol_data['pip_value_per_lot']
        
        # 1. ACCOUNT STATE ADJUSTMENT
        equity = account_info.equity
        balance = account_info.balance
        
        # Calculate current drawdown
        drawdown_percent = ((balance - equity) / balance) * 100 if balance > 0 else 0
        
        # Adjust risk based on account state
        if drawdown_percent > 10:
            # In significant drawdown - reduce risk
            risk_multiplier = 0.5
            logging.info(f"[{symbol}] Drawdown {drawdown_percent:.1f}% - Reducing risk")
        elif equity > balance * 1.1:
            # Up over 10% - can be slightly more aggressive
            risk_multiplier = 1.2
        else:
            # Normal state
            risk_multiplier = 1.0
        
        # 2. CORRELATION ADJUSTMENT
        correlation_factor = self.check_correlation_exposure(symbol)
        risk_percent = risk_percent * risk_multiplier * correlation_factor
        
        # 3. PAIR VOLATILITY ADJUSTMENT
        if 'JPY' in symbol or 'GBP' in symbol:
            # High volatility pairs - reduce position size
            risk_percent *= 0.8
        elif symbol in ['EURUSD', 'USDCHF']:
            # Low volatility, high liquidity - can use full size
            risk_percent *= 1.0
        else:
            # Medium volatility pairs
            risk_percent *= 0.9
        
        # 4. SESSION-BASED ADJUSTMENT
        current_hour = self.get_trading_hour()
        if 20 <= current_hour or current_hour < 7:
            # Asian session - reduce size (tighter stops might get hit)
            risk_percent *= 0.8
        elif 7 <= current_hour < 9 or 13 <= current_hour < 15:
            # Major market opens - reduce size (volatility spikes)
            risk_percent *= 0.9
        
        # 5. STOP LOSS QUALITY ADJUSTMENT
        if stop_loss_pips > 30:
            # Wide stop = poor entry, reduce size
            risk_percent *= 0.7
            logging.info(f"[{symbol}] Wide stop {stop_loss_pips:.1f} pips - Reducing size")
        elif stop_loss_pips < 15:
            # Very tight stop = higher chance of stopping out
            risk_percent *= 0.8
            logging.info(f"[{symbol}] Tight stop {stop_loss_pips:.1f} pips - Reducing size")
        
        # 6. MAXIMUM RISK CAPS
        # Never risk more than 2% adjusted (even on best setups)
        risk_percent = min(risk_percent, 0.02)
        
        # During news or high-impact events (you'd need a news calendar)
        if self.is_high_impact_news_soon():
            risk_percent = min(risk_percent, 0.005)  # Max 0.5% during news
        
        # 7. CALCULATE BASE LOT SIZE
        risk_amount = equity * risk_percent  # Use equity, not balance
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # 8. LEVERAGE AND MARGIN CHECK
        # Check if we're over-leveraging
        position_value = lot_size * symbol_info.trade_contract_size
        leverage_used = position_value / equity
        
        max_leverage = 30  # Professional day traders rarely exceed 30:1
        if leverage_used > max_leverage:
            # Reduce to maximum acceptable leverage
            lot_size = (equity * max_leverage) / symbol_info.trade_contract_size
            logging.warning(f"[{symbol}] Leverage cap applied: {leverage_used:.1f}x -> {max_leverage}x")
        
        # 9. BROKER CONSTRAINTS
        # Round to lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply min/max
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # 10. MARGIN REQUIREMENT CHECK
        margin_required = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY,
            symbol,
            lot_size,
            symbol_data['tick'].ask
        )
        
        if margin_required and margin_required > equity * 0.5:
            # Never use more than 50% of equity for margin
            lot_size *= (equity * 0.5) / margin_required
            lot_size = math.floor(lot_size / lot_step) * lot_step
            logging.warning(f"[{symbol}] Margin constraint applied")
        
        # 11. FINAL SAFETY CHECK
        # Ensure the calculated risk doesn't exceed our maximum
        actual_risk = lot_size * stop_loss_pips * pip_value_per_lot
        actual_risk_percent = actual_risk / equity
        
        if actual_risk_percent > 0.02:  # Hard cap at 2%
            lot_size *= 0.02 / actual_risk_percent
            lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Log the decision
        logging.info(f"[{symbol}] Lot calculation:")
        logging.info(f"  - Base risk: {risk_percent*100:.2f}%")
        logging.info(f"  - Risk amount: ${risk_amount:.2f}")
        logging.info(f"  - Final lots: {lot_size:.2f}")
        logging.info(f"  - Actual risk: {actual_risk_percent*100:.2f}%")
        
        return lot_size

    def check_correlation_exposure(self, symbol):
        """
        Check existing positions for correlation risk
        Returns a factor to reduce position size if correlated pairs exist
        """
        positions = mt5.positions_get()
        if not positions:
            return 1.0
        
        # Define correlation groups
        correlation_groups = {
            'USD_LONG': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],  # Sell these = USD long
            'USD_SHORT': ['USDCAD', 'USDCHF', 'USDJPY'],  # Buy these = USD long
            'EUR_CROSS': ['EURUSD', 'EURGBP', 'EURJPY', 'EURAUD'],
            'GBP_CROSS': ['GBPUSD', 'GBPJPY', 'EURGBP', 'GBPAUD'],
            'JPY_CROSS': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY'],
            'COMMODITY': ['AUDUSD', 'NZDUSD', 'USDCAD', 'AUDCAD']
        }
        
        # Check which groups the new symbol belongs to
        symbol_groups = []
        for group, symbols in correlation_groups.items():
            if symbol in symbols:
                symbol_groups.append(group)
        
        # Count existing positions in same groups
        correlated_positions = 0
        for pos in positions:
            if pos.symbol == symbol:
                continue  # Don't count positions in same symbol
                
            for group, symbols in correlation_groups.items():
                if group in symbol_groups and pos.symbol in symbols:
                    correlated_positions += 1
                    break
        
        # Reduce size based on correlation exposure
        if correlated_positions == 0:
            return 1.0
        elif correlated_positions == 1:
            return 0.7  # 30% reduction for one correlated pair
        elif correlated_positions == 2:
            return 0.5  # 50% reduction for two correlated pairs
        else:
            return 0.3  # 70% reduction for three or more
        
    def get_trading_hour(self):
        """Get current trading hour in GMT"""
        from datetime import datetime
        import pytz
        
        # Convert server time to GMT (adjust based on your server)
        gmt = pytz.timezone('GMT')
        current_time = datetime.now(gmt)
        return current_time.hour

    def is_high_impact_news_soon(self):
        """
        Check if high-impact news is coming in next 30 minutes
        In production, this would connect to a news calendar API
        """
        # Simplified version - you'd integrate with ForexFactory or similar
        current_hour = self.get_trading_hour()
        
        # Common high-impact news times (GMT)
        news_times = [
            8,   # European data
            9,   # More European data  
            13,  # US data
            14,  # More US data
            18   # FOMC minutes (Wednesdays)
        ]
        
        if current_hour in news_times or (current_hour + 1) in news_times:
            return True
        
        return False
        
    def place_order(self, symbol, lots_size, order_type, stop_loss, take_profit):
        """Place market order with proper error handling"""
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        symbol_info = symbol_data['info']
            
        # Verify symbol is tradeable
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            logging.error(f"Trading disabled for {symbol}")
            return None
                        
        if order_type == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            entry_price = symbol_tick.ask
        else:  # sell
            mt5_order_type = mt5.ORDER_TYPE_SELL
            entry_price = symbol_tick.bid
                             
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots_size,
            "type": mt5_order_type,
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "magic": MAGIC_NUMBER,
            "comment": EA_COMMENT,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
                            
        # Send the order
        result = mt5.order_send(request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            actual_entry = result.price 
            logging.info(f"[{symbol}] Entry={actual_entry:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")
        else:
            logging.error(f"Order failed for {symbol}: {result.comment}")

        return result

def main():
    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("Failed to connect to MetaTrader 5")
        return
        
    symbols = DEFAULT_SYMBOLS
    timeframes = DEFAULT_TIMEFRAMES

    try:
        # Initialize market data and trade managers
        market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
        trade_managers = {symbol: TradeManager(client, market_data_dict[symbol]) for symbol in symbols}

        while True:
            start_time = time.time()

            # Check each symbol for signals
            for symbol in symbols:
                success, message, trade_info = trade_managers[symbol].check_for_signals(symbol)
                
                if success and trade_info:
                    logging.info(f"{message} - {trade_info}")
                elif success is False and "Max" in message:
                    logging.info(message)

            end_time = time.time()
            cycle_duration = end_time - start_time
            
            # Sleep until next check
            sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
            if sleep_time > 0:
                logging.info(f"Next check in {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        mt5.shutdown()
        logging.info("Trading script terminated")

if __name__ == "__main__":
    main()