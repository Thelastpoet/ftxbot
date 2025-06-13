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
            
            if len(df) < 100:
                return {'valid_setup': False, 'fib_level': None}
            
            trend = htf_context['trend']
            if trend == 'unclear':
                return {'valid_setup': False, 'fib_level': None}
            
            current_price = df['close'].iloc[-1]
            spread = self.get_current_spread()
            pip_size = self.market_data.get_pip_size()
            
            # Use talib ATR for consistency with the rest of your code
            atr_values = ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_atr = atr_values[-1]
            
            if trend == 'uptrend':
                # FIXED: Find the most recent high first
                lookback_start = max(0, len(df) - 50)
                recent_data = df.iloc[lookback_start:]
                
                # Find the highest high in recent data
                trend_end_idx = recent_data['high'].idxmax()
                trend_end_price = recent_data.loc[trend_end_idx, 'high']
                
                # Find the lowest low BEFORE that high
                data_before_high = df.loc[:trend_end_idx]
                search_start = max(0, len(data_before_high) - 50)
                
                trend_start_idx = data_before_high.iloc[search_start:]['low'].idxmin()
                trend_start_price = data_before_high.loc[trend_start_idx, 'low']
                
                # Ensure we have a meaningful trend move (use ATR instead of fixed pips)
                swing_range = trend_end_price - trend_start_price
                if swing_range < current_atr:  # Must be at least 1 ATR
                    return {'valid_setup': False, 'fib_level': None}
                
                # Calculate Fibonacci levels
                fib_levels = {
                    '0.382': trend_end_price - (swing_range * 0.382),
                    '0.500': trend_end_price - (swing_range * 0.500),
                    '0.618': trend_end_price - (swing_range * 0.618)
                }
                
                # Tighter tolerance
                tolerance = max(3 * pip_size, swing_range * 0.01)  # Reduced from 5 pips/2%
                
                for level_name, level_price in fib_levels.items():
                    if abs(current_price - level_price) <= tolerance + spread:
                        bars_since_high = len(df) - df.index.get_loc(trend_end_idx)
                        if bars_since_high >= 5:
                            return {
                                'valid_setup': True, 
                                'fib_level': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance),
                                'trend_start': trend_start_price,
                                'trend_end': trend_end_price
                            }
                            
            else:  # downtrend
                # Find the most recent low first
                lookback_start = max(0, len(df) - 50)
                recent_data = df.iloc[lookback_start:]
                
                trend_end_idx = recent_data['low'].idxmin()
                trend_end_price = recent_data.loc[trend_end_idx, 'low']
                
                # Find the highest high BEFORE that low
                data_before_low = df.loc[:trend_end_idx]
                search_start = max(0, len(data_before_low) - 50)
                
                trend_start_idx = data_before_low.iloc[search_start:]['high'].idxmax()
                trend_start_price = data_before_low.loc[trend_start_idx, 'high']
                
                swing_range = trend_start_price - trend_end_price
                if swing_range < current_atr:
                    return {'valid_setup': False, 'fib_level': None}
                
                fib_levels = {
                    '0.382': trend_end_price + (swing_range * 0.382),
                    '0.500': trend_end_price + (swing_range * 0.500),
                    '0.618': trend_end_price + (swing_range * 0.618)
                }
                
                tolerance = max(3 * pip_size, swing_range * 0.01)
                
                for level_name, level_price in fib_levels.items():
                    if abs(current_price - level_price) <= tolerance + spread:
                        bars_since_low = len(df) - df.index.get_loc(trend_end_idx)
                        if bars_since_low >= 5:
                            return {
                                'valid_setup': True, 
                                'fib_level': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance),
                                'trend_start': trend_start_price,
                                'trend_end': trend_end_price
                            }
            
            return {'valid_setup': False, 'fib_level': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_medium_timeframe: {str(e)}")
            return {'valid_setup': False, 'fib_level': None}
    
    def analyze_lower_timeframe(self, data, trade_direction):
        """
        Detect Break of Structure (BOS) for precise entries - FIXED timing issue
        """
        try:
            df = data.copy()
            
            if len(df) < 20:
                return {'valid_entry': False, 'entry_type': None}
            
            # Find recent swing points on M5
            swing_highs, swing_lows = self.find_swing_points(df, window=3)
            
            if len(swing_highs) == 0 or len(swing_lows) == 0:
                return {'valid_entry': False, 'entry_type': None}
            
            # Look at the last CLOSED candle only
            last_closed = df.iloc[-2]
            prev_candle = df.iloc[-3]
            
            # Get current market data
            symbol_tick = mt5.symbol_info_tick(self.market_data.symbol)
            if not symbol_tick:
                logging.error(f"Failed to get tick data for {self.market_data.symbol}")
                return {'valid_entry': False, 'entry_type': None}
            
            spread = self.get_current_spread()
            pip_size = self.market_data.get_pip_size()
            
            if trade_direction == 'buy':
                last_swing_high = swing_highs.iloc[-1]['price']
                
                # Check if last closed candle broke above swing high
                break_occurred = (
                    prev_candle['close'] <= last_swing_high and  
                    last_closed['close'] > last_swing_high + spread and
                    last_closed['close'] > last_closed['open']
                )
                
                if break_occurred:
                    candle_range = last_closed['high'] - last_closed['low']
                    if candle_range > 0:
                        body_size = abs(last_closed['close'] - last_closed['open'])
                        candle_strength = body_size / candle_range
                        
                        if candle_strength > 0.6:
                            # FIX: Enter at break level + small buffer, not current market price
                            entry_price = last_swing_high + spread + (2 * pip_size)
                            
                            # Only enter if current price hasn't run away
                            if symbol_tick.ask <= entry_price + (3 * pip_size):
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_resistance',
                                    'break_level': last_swing_high,
                                    'entry_price': min(symbol_tick.ask, entry_price),  # Better of the two
                                    'candle_close': last_closed['close'],
                                    'candle_strength': candle_strength,
                                    'spread_at_signal': spread
                                }
                            else:
                                logging.info(f"Price ran away from break: {symbol_tick.ask:.5f} vs {entry_price:.5f}")
                                
            else:  # sell
                last_swing_low = swing_lows.iloc[-1]['price']
                
                break_occurred = (
                    prev_candle['close'] >= last_swing_low and   
                    last_closed['close'] < last_swing_low - spread and
                    last_closed['close'] < last_closed['open']
                )
                
                if break_occurred:
                    candle_range = last_closed['high'] - last_closed['low']
                    if candle_range > 0:
                        body_size = abs(last_closed['close'] - last_closed['open'])
                        candle_strength = body_size / candle_range
                        
                        if candle_strength > 0.6:
                            # FIX: Enter at break level - small buffer, not current market price
                            entry_price = last_swing_low - spread - (2 * pip_size)
                            
                            # Only enter if current price hasn't run away
                            if symbol_tick.bid >= entry_price - (3 * pip_size):
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_support',
                                    'break_level': last_swing_low,
                                    'entry_price': max(symbol_tick.bid, entry_price),  # Better of the two
                                    'candle_close': last_closed['close'],
                                    'candle_strength': candle_strength,
                                    'spread_at_signal': spread
                                }
                            else:
                                logging.info(f"Price ran away from break: {symbol_tick.bid:.5f} vs {entry_price:.5f}")
            
            return {'valid_entry': False, 'entry_type': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_lower_timeframe: {str(e)}")
            return {'valid_entry': False, 'entry_type': None}
    
    def check_m5_momentum(self, data, trade_direction):
        """
        Quick momentum check on M5 - don't need full structure break
        Just avoid entering on extended moves
        """
        df = data.copy()
        if len(df) < 20:
            return False
            
        # Simple momentum check using recent candles
        recent_candles = df.tail(5)
        
        if trade_direction == 'buy':
            # Count consecutive bullish candles properly
            consecutive_bulls = 0
            max_consecutive = 0
            
            for i in range(len(recent_candles)):
                if recent_candles.iloc[i]['close'] > recent_candles.iloc[i]['open']:
                    consecutive_bulls += 1
                    max_consecutive = max(max_consecutive, consecutive_bulls)
                else:
                    consecutive_bulls = 0  # Reset on bearish candle
                    
            # Also check if we're not too extended from recent low
            recent_low = df.tail(10)['low'].min()
            current_price = df.iloc[-1]['close']
            extension = (current_price - recent_low) / self.market_data.get_pip_size()
            
            return max_consecutive < 3 and extension < 30
            
        else:  # sell
            consecutive_bears = 0
            max_consecutive = 0
            
            for i in range(len(recent_candles)):
                if recent_candles.iloc[i]['close'] < recent_candles.iloc[i]['open']:
                    consecutive_bears += 1
                    max_consecutive = max(max_consecutive, consecutive_bears)
                else:
                    consecutive_bears = 0  # Reset on bullish candle
                    
            recent_high = df.tail(10)['high'].max()
            current_price = df.iloc[-1]['close']
            extension = (recent_high - current_price) / self.market_data.get_pip_size()
            
            return max_consecutive < 3 and extension < 30

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
    
    def estimate_stop_distance(self, symbol, trade_direction, ltf_break_level, current_price, atr_value):
        """
        Estimate stop distance BEFORE full trade calculation
        Must match calculate_sl_tp() logic EXACTLY
        """
        pip_size = self.market_data.get_pip_size()
        symbol_info = mt5.symbol_info(symbol)
        spread = symbol_info.spread * symbol_info.point
        
        # EXACT same logic as calculate_sl_tp()
        noise_buffer = atr_value * 0.2  # 20% of H1 ATR
        
        if trade_direction == 'buy':
            estimated_stop = ltf_break_level - noise_buffer - spread
            estimated_stop_distance = current_price - estimated_stop
        else:  # sell
            estimated_stop = ltf_break_level + noise_buffer + spread
            estimated_stop_distance = estimated_stop - current_price
        
        # Ensure minimum 20 pips (broker minimum)
        min_stop = 20 * pip_size
        estimated_stop_distance = max(estimated_stop_distance, min_stop)
        
        return estimated_stop_distance / pip_size  # Return in pips

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
                if self.check_m5_momentum(data[self.tf_lower], trade_direction):
                    # Enter at Fib level with momentum confirmation
                    # Use appropriate side of the Fib zone
                    if trade_direction == 'buy':
                        break_level = mtf_analysis['entry_zone'][0]  # Lower bound for buys
                    else:
                        break_level = mtf_analysis['entry_zone'][1]  # Upper bound for sells
                        
                    symbol_tick = mt5.symbol_info_tick(symbol)
                    current_spread = self.get_current_spread()
                        
                    ltf_analysis = {
                        'valid_entry': True,
                        'entry_type': 'fib_with_momentum',
                        'break_level': break_level,
                        'entry_price': symbol_tick.ask if trade_direction == 'buy' else symbol_tick.bid,
                        'spread_at_signal': current_spread,
                        'candle_strength': 0.0,  # Not applicable for momentum
                        'break_distance_pips': 0.0  # Not applicable
                    }
                else:
                    return False, f"[{symbol}] Waiting for better M5 entry", None

            logging.info(f"[{symbol}] Entry signal: {ltf_analysis['entry_type']}")
            
            # Use talib ATR
            high = data[self.tf_higher]['high'].values
            low = data[self.tf_higher]['low'].values
            close = data[self.tf_higher]['close'].values
            
            atr_values = ATR(high, low, close, timeperiod=14)
            atr_value = atr_values[-1]
            
            if np.isnan(atr_value) or atr_value <= 0:
                return False, f"[{symbol}] Invalid ATR calculation", None
            
            # ========== NEW VALIDATION SECTION ==========
            # 4. CHECK IF THERE'S ROOM FOR PROFIT
            current_price = data[self.tf_lower]['close'].iloc[-1]
            pip_size = self.market_data.get_pip_size()
            
            # Estimate what our stop distance would be
            estimated_stop_pips = self.estimate_stop_distance(
                symbol,
                trade_direction,
                ltf_analysis.get('break_level'),
                current_price,
                atr_value
            )
            
            logging.info(f"[{symbol}] Estimated stop: {estimated_stop_pips:.1f} pips")
            
            # Simple check: Is there enough room to the next major level?
            if trade_direction == 'buy':
                target = mtf_analysis.get('trend_end')
                room_to_target = target - current_price
                room_in_pips = room_to_target / pip_size
                    
                # Need at least 1.5x stop distance for good RR
                min_room_required = estimated_stop_pips * 1.5
                    
                if room_in_pips < min_room_required:
                    actual_rr = room_in_pips / estimated_stop_pips
                    logging.warning(f"[{symbol}] Poor RR - Room: {room_in_pips:.1f} pips, "
                        f"Need: {min_room_required:.1f} pips, "
                            f"Potential RR: 1:{actual_rr:.1f}")
                    return False, f"[{symbol}] Insufficient RR (1:{actual_rr:.1f})", None
                                                
            else:  # sell
                target = mtf_analysis.get('trend_end')
                room_to_target = current_price - target
                room_in_pips = room_to_target / pip_size
                    
                min_room_required = estimated_stop_pips * 1.5
            
                if room_in_pips < min_room_required:
                    actual_rr = room_in_pips / estimated_stop_pips
                    logging.warning(f"[{symbol}] Poor RR - Room: {room_in_pips:.1f} pips, "
                        f"Need: {min_room_required:.1f} pips, "
                        f"Potential RR: 1:{actual_rr:.1f}")
                    return False, f"[{symbol}] Insufficient RR (1:{actual_rr:.1f})", None
                                    
            logging.info(f"[{symbol}] Room check passed - Potential RR: 1:{(room_in_pips/estimated_stop_pips):.1f}")
            # ========== END OF NEW SECTION ==========

            # All conditions met - prepare trade
            account_info = self.client.get_account_info()
            if account_info is None:
                logging.error(f"[{symbol}] Failed to get account info")
                return False, f"[{symbol}] Could not get account info", None
                        
            entry_analysis = {
                'ltf_break_level': ltf_analysis.get('break_level'),
                'mtf_fib_zone': mtf_analysis.get('entry_zone'),
                'htf_swings': {
                    'last_swing_high': htf_analysis.get('last_swing_high'),
                    'last_swing_low': htf_analysis.get('last_swing_low')
                },
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
            
    def calculate_sl_tp(self, symbol, order_type, atr_value, entry_analysis):
        """
        SL/TP calculation based on structure
        """
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        symbol_info = symbol_data['info']
        pip_size = symbol_data['pip_size']
               
        # Get the M5 break level that triggered our entry
        ltf_break = entry_analysis.get('ltf_break_level')
        if not ltf_break:
            raise ValueError("No break level found in entry analysis")
        
        # Get current price for calculations
        if order_type == 'buy':
            current_price = symbol_tick.ask
            
            hunt_buffer = 12 * pip_size 
            
            # Find a cluster of lows (liquidity pool)
            # Then place stop BELOW the entire cluster
            structure_low = entry_analysis.get('structure_low', ltf_break)
            dynamic_buffer = atr_value * 0.3 
            
            # Stop goes below the M5 structure we broke
            stop_loss = structure_low - max(hunt_buffer, dynamic_buffer)
            
            remainder = (stop_loss / pip_size) % 10
            if remainder < 2 or remainder > 8:
                stop_loss -= 3 * pip_size
                
            # Take profit: Use H1 swing high or 2x ATR, whichever is closer
            htf_resistance = entry_analysis.get('htf_swings', {}).get('last_swing_high')
            atr_target = current_price + (atr_value * 2.0)
            
            if htf_resistance and htf_resistance > current_price:
                # Use the closer target
                take_profit = min(htf_resistance - (3 * pip_size), atr_target)
            else:
                take_profit = atr_target
                
        else:  # sell
            current_price = symbol_tick.bid
            
            hunt_buffer = 12 * pip_size
            structure_high = entry_analysis.get('structure_high', ltf_break)
            dynamic_buffer = atr_value * 0.3
            
            stop_loss = structure_high + max(hunt_buffer, dynamic_buffer)
                        
            # Round to ugly price
            remainder = (stop_loss / pip_size) % 10
            if remainder < 2 or remainder > 8:
                stop_loss += 3 * pip_size
                
            # Take profit: Use H1 swing low or 2x ATR, whichever is closer
            htf_support = entry_analysis.get('htf_swings', {}).get('last_swing_low')
            atr_target = current_price - (atr_value * 2.0)
            
            if htf_support and htf_support < current_price:
                take_profit = max(htf_support + (3 * pip_size), atr_target)
            else:
                take_profit = atr_target
        
        # Ensure minimum stop distance (broker requirement)
        min_stop_distance = max(
            symbol_info.trade_stops_level * symbol_info.point,
            20 * pip_size
        )
        
        if order_type == 'buy' and current_price - stop_loss < min_stop_distance:   
            stop_loss = current_price - min_stop_distance
        elif order_type == 'sell' and stop_loss - current_price < min_stop_distance:
            stop_loss = current_price + min_stop_distance
        
        # Ensure minimum 1.5:1 RR
        stop_distance = abs(current_price - stop_loss)
        profit_distance = abs(take_profit - current_price)
        
        if profit_distance < stop_distance * 1.5:
            if order_type == 'buy':
                take_profit = current_price + (stop_distance * 1.5)
            else:
                take_profit = current_price - (stop_distance * 1.5)
        
        # Normalize to tick size
        tick_size = symbol_info.trade_tick_size
        stop_loss = round(stop_loss / tick_size) * tick_size
        take_profit = round(take_profit / tick_size) * tick_size
        
        # Calculate metrics
        stop_loss_pips = abs(current_price - stop_loss) / pip_size
        actual_rr = abs(take_profit - current_price) / abs(current_price - stop_loss)
        
        logging.info(f"[{symbol}] SL: {stop_loss:.5f} ({stop_loss_pips:.1f} pips), "
                    f"TP: {take_profit:.5f}, RR: 1:{actual_rr:.1f}")
        
        return stop_loss, take_profit, stop_loss_pips
        
    def calculate_lot_size(self, symbol, account_info, stop_loss_pips, risk_percent=0.01):
        """
        Professional lot size calculation that actually works in live trading
        
        Args:
            symbol: Trading symbol
            account_info: MT5 account info
            stop_loss_pips: Stop loss distance in pips
            risk_percent: Base risk percentage (default 1%)
        """
        # Get symbol specifics
        symbol_data = self.get_symbol_info(symbol)
        symbol_info = symbol_data['info']
        pip_value_per_lot = symbol_data['pip_value_per_lot']
        
        # CRITICAL: Use equity, not balance
        # If you're in drawdown, equity < balance, so you risk less
        # If you're in profit, equity > balance, compound naturally
        equity = account_info.equity
        
        # Start with base risk - no arbitrary adjustments without data
        adjusted_risk = risk_percent
        
        # TODO: After collecting performance data, implement adjustments based on:
        # - Actual stop-out rates per pair
        # - Win rate at different stop distances  
        # - Real volatility measurements during your trading hours
        # For now, we trade all pairs with same risk % and let results guide us
        
        # Stop loss reality check - based on market mechanics, not arbitrary numbers
        # With structure-based stops + noise buffer:
        # - Stops < 20 pips: Broker minimum or forced entry, higher chance of noise stop-out
        # - Stops 20-40 pips: Normal, healthy distance for day trading
        # - Stops > 40 pips: Either high volatility period or poor entry timing
        
        if stop_loss_pips < 20:
            # Tight stop = higher probability of random stop-out
            # Reduce risk to compensate for lower win rate
            adjusted_risk *= 0.8
            logging.info(f"[{symbol}] Tight stop {stop_loss_pips:.1f} pips - reducing size by 20%")
        elif stop_loss_pips > 40:
            # Wide stop = either poor entry or extreme volatility
            # Both cases warrant risk reduction
            adjusted_risk *= 0.7
            logging.info(f"[{symbol}] Wide stop {stop_loss_pips:.1f} pips - reducing size by 30%")
            
            if stop_loss_pips > 60:
                # Very wide stop = something's wrong with this setup
                adjusted_risk *= 0.5  # Total 35% of original risk
                logging.info(f"[{symbol}] Very wide stop {stop_loss_pips:.1f} pips - using minimal size")
        
        # Calculate risk amount
        risk_amount = equity * adjusted_risk
        
        # Basic lot calculation
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Round to broker's lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply broker constraints
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # CRITICAL: Leverage check - prevent catastrophic over-leveraging
        # Professional day traders think in terms of total exposure
        # With proper stops, 30:1 on a single position is acceptable
        # But this assumes you're not maxing out on every trade
        position_value = lot_size * symbol_info.trade_contract_size
        leverage_used = position_value / equity
        
        # Maximum acceptable leverage per position
        # This isn't arbitrary - it's based on surviving a 3-4 trade losing streak
        max_leverage_per_position = 30
        
        if leverage_used > max_leverage_per_position:
            # Scale down to maximum acceptable leverage
            lot_size = (equity * max_leverage_per_position) / symbol_info.trade_contract_size
            lot_size = math.floor(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, lot_size)
            
            new_leverage = (lot_size * symbol_info.trade_contract_size) / equity
            logging.warning(f"[{symbol}] Leverage cap: {leverage_used:.1f}x -> {new_leverage:.1f}x")
        
        # Margin requirement check - ensure we have enough free margin
        margin_required = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY,
            symbol,
            lot_size,
            symbol_data['tick'].ask
        )
        
        if margin_required:
            free_margin = account_info.margin_free
            # Never use more than 40% of free margin for a single position
            if margin_required > free_margin * 0.4:
                lot_size = (free_margin * 0.4 / margin_required) * lot_size
                lot_size = math.floor(lot_size / lot_step) * lot_step
                lot_size = max(symbol_info.volume_min, lot_size)
                logging.warning(f"[{symbol}] Margin constraint applied")
        
        # Final sanity check - ensure actual risk doesn't exceed 2%
        actual_risk_amount = lot_size * stop_loss_pips * pip_value_per_lot
        actual_risk_percent = (actual_risk_amount / equity) * 100
        
        if actual_risk_percent > 2.0:
            # Hard cap at 2% regardless of calculations
            lot_size = (equity * 0.02) / (stop_loss_pips * pip_value_per_lot)
            lot_size = math.floor(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, lot_size)
            logging.warning(f"[{symbol}] Risk cap applied: {actual_risk_percent:.1f}% -> 2.0%")
        
        # Recalculate final metrics for logging
        final_risk_amount = lot_size * stop_loss_pips * pip_value_per_lot
        final_risk_percent = (final_risk_amount / equity) * 100
        final_leverage = (lot_size * symbol_info.trade_contract_size) / equity
        
        logging.info(f"[{symbol}] Position sizing:")
        logging.info(f"  - Equity: ${equity:.2f}")
        logging.info(f"  - Stop: {stop_loss_pips:.1f} pips")
        logging.info(f"  - Risk: {final_risk_percent:.2f}% (${final_risk_amount:.2f})")
        logging.info(f"  - Lots: {lot_size:.2f}")
        logging.info(f"  - Leverage: {final_leverage:.1f}x")
        
        return lot_size
        
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