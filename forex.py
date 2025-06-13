import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from talib import ATR
from market_behavior_analyzer import MarketBehaviorAnalyzer

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
MAGIC_NUMBER = 234000
EA_COMMENT = "Adaptive MTF EA"
DEFAULT_SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 
                  'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY', 
                  'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 
                  'GBPNZD', 'NZDUSD']
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

class AdaptiveTradeManager:
    """
    Trade manager that learns from market behavior instead of using magic numbers
    """
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.order_manager = AdaptiveOrderManager(client, market_data)
        
        # Initialize behavior analyzer
        self.behavior_analyzer = MarketBehaviorAnalyzer(market_data.symbol)
        self.market_params = None
        self.last_analysis_time = None
        
        # Initialize timeframe hierarchy
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] 
        self.tf_lower = min(self.timeframes)
        
        # Position limits
        self.max_positions = 10
        self.max_per_symbol = 2
        
        # Update market knowledge on initialization
        self.update_market_knowledge()
        
    def update_market_knowledge(self):
        """
        Update our understanding of how this specific market behaves
        """
        # Define staleness thresholds
        full_analysis_threshold = timedelta(hours=24) # Re-run heavy analysis if data is > 24h old
        in_session_refresh_threshold = timedelta(hours=6) # Lightweight refresh while bot is running

        # Check if we need to perform a full, heavy analysis
        run_full_analysis = False
        cached_behavior = self.behavior_analyzer.load_behavior()
        last_analysis_str = cached_behavior.get('last_analysis_timestamp')

        if not last_analysis_str:
            logging.info(f"[{self.market_data.symbol}] No cached behavior file found. Performing initial analysis.")
            run_full_analysis = True
        else:
            last_analysis_time = datetime.fromisoformat(last_analysis_str)
            if datetime.now() - last_analysis_time > full_analysis_threshold:
                logging.info(f"[{self.market_data.symbol}] Cached behavior is stale (older than 24 hours). Re-analyzing.")
                run_full_analysis = True

        # Also check for in-session refresh need (original logic)
        if self.last_analysis_time and (datetime.now() - self.last_analysis_time > in_session_refresh_threshold):
            logging.info(f"[{self.market_data.symbol}] In-session refresh triggered (>6 hours). Re-analyzing.")
            run_full_analysis = True
        
        if run_full_analysis:
            logging.info(f"[{self.market_data.symbol}] Starting full market behavior analysis...")
            self.behavior_analyzer.analyze_historical_data(days_back=60)
            self.market_params = self.behavior_analyzer.get_optimal_parameters()
            self.last_analysis_time = datetime.now()
            
            if self.market_params:
                self.validate_market_context()
            else:
                logging.warning(f"[{self.market_data.symbol}] Analysis completed but no parameters were generated.")

        # If analysis was not run, but params are not loaded yet, load them from cache.
        elif self.market_params is None:
            logging.info(f"[{self.market_data.symbol}] Using fresh, cached behavior data.")
            self.market_params = self.behavior_analyzer.get_optimal_parameters()
            self.last_analysis_time = datetime.now() # Set in-memory time to now
            
            if self.market_params:
                logging.info(f"[{self.market_data.symbol}] Market parameters loaded from cache.")
                # Optional: Log a key parameter to confirm it's loaded
                if 'typical_pullback_depths' in self.market_params:
                    logging.info(f"  - Cached Typical pullbacks: {self.market_params['typical_pullback_depths']}")
                    
    def estimate_quick_stop_distance(self, current_price, trend_start_price):
        """Quick stop distance estimation for pre-validation"""
        pip_size = self.market_data.get_pip_size()
        
        # Use market-specific buffer if available
        if self.market_params and 'stop_buffer_pips' in self.market_params:
            buffer_pips = self.market_params['stop_buffer_pips'].get('safe', 20)
            return buffer_pips
        else:
            # Fallback: use trend range based estimate
            trend_range = abs(current_price - trend_start_price) / pip_size
            return min(max(trend_range * 0.15, 15), 30)  # 15% of trend, min 15, max 30 pips
        
    def validate_market_context(self):
        """Check if current market conditions match historical data context"""
        try:
            # Get current ATR for context
            current_data = self.market_data.fetch_data(self.tf_higher)
            if current_data is None or len(current_data) < 20:
                return
            
            from talib import ATR
            atr_values = ATR(current_data['high'].values, 
                            current_data['low'].values, 
                            current_data['close'].values, 
                            timeperiod=14)
            current_atr = atr_values[-1]
            pip_size = self.market_data.get_pip_size()
            current_atr_pips = current_atr / pip_size
            
            # Compare to stored session volatility
            current_session = self.get_current_session()
            if ('session_volatility' in self.market_params and 
                current_session in self.market_params['session_volatility']):
                
                historical_avg = self.market_params['session_volatility'][current_session]['avg_range']
                volatility_ratio = current_atr_pips / historical_avg if historical_avg > 0 else 1
                
                # If volatility changed significantly, apply conservative adjustments
                if volatility_ratio > 1.4 or volatility_ratio < 0.6:
                    logging.warning(f"[{self.market_data.symbol}] Volatility changed significantly "
                                f"(ratio: {volatility_ratio:.1f}), applying conservative adjustments")
                    
                    # Make pullback requirements deeper when volatility changes
                    if 'typical_pullback_depths' in self.market_params:
                        pullbacks = self.market_params['typical_pullback_depths']
                        for level in pullbacks:
                            pullbacks[level] = max(pullbacks[level], pullbacks[level] * 1.2)
                            
        except Exception as e:
            logging.error(f"Error in validate_market_context: {str(e)}")
    
    def find_swing_points(self, data, window=10):
        """Find confirmed swing highs and lows - now adaptive to pair behavior"""
        df = data.copy()
        
        # Use market-specific window if available
        if self.market_params and 'typical_swing_size' in self.market_params:
            # Adjust window based on typical swing duration
            typical_size = self.market_params['typical_swing_size'].get('medium', 30)
            if typical_size < 20:
                window = 5  # Smaller swings = smaller window
            elif typical_size > 50:
                window = 15  # Larger swings = larger window
        
        # Only analyze bars that can be confirmed
        confirmed_length = len(df) - window
        if confirmed_length < window * 2 + 1:
            return pd.DataFrame(), pd.DataFrame()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, confirmed_length):
            is_swing_high = True
            is_swing_low = True
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check surrounding bars
            for j in range(1, window + 1):
                # Check left side
                if df.iloc[i - j]['high'] >= current_high:
                    is_swing_high = False
                if df.iloc[i - j]['low'] <= current_low:
                    is_swing_low = False
                    
                # Check right side
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
        """Higher Timeframe trend identification using market-specific behavior"""
        try:
            df = data.copy()
            
            # Find swing points
            swing_highs, swing_lows = self.find_swing_points(df)
            
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
                    strength = (hh_count + hl_count) / 4.0
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
                
                # Adjust strength based on market behavior
                if self.market_params and 'session_volatility' in self.market_params:
                    current_session = self.get_current_session()
                    if current_session in self.market_params['session_volatility']:
                        session_data = self.market_params['session_volatility'][current_session]
                        # Higher volatility sessions = potentially stronger trends
                        if session_data['avg_range'] > session_data['median_range'] * 1.2:
                            strength = min(strength + 0.1, 1.0)
                    
            return {
                'trend': trend, 
                'strength': strength,
                'last_swing_high': swing_highs.iloc[-1]['price'] if len(swing_highs) > 0 else None,
                'last_swing_low': swing_lows.iloc[-1]['price'] if len(swing_lows) > 0 else None,
                'swing_count': len(swing_highs) + len(swing_lows)
            }
            
        except Exception as e:
            logging.error(f"Error in analyze_higher_timeframe: {str(e)}")
            return {'trend': 'unclear', 'strength': 0}

    def analyze_medium_timeframe(self, data, htf_context):
        """Detect pullback setups using actual market behavior instead of fixed Fibonacci"""
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
            
            # Get market-specific pullback levels
            if self.market_params and 'typical_pullback_depths' in self.market_params:
                pullback_levels = self.market_params['typical_pullback_depths']
                logging.info(f"[{self.market_data.symbol}] Using learned pullback levels: {pullback_levels}")
            else:
                # Fallback to Fibonacci if no data
                pullback_levels = {
                    '20th': 0.236,
                    '35th': 0.382,
                    '50th': 0.500,
                    '65th': 0.618,
                    '80th': 0.786
                }
                logging.info(f"[{self.market_data.symbol}] Using fallback Fibonacci levels")
            
            # Use talib ATR
            atr_values = ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_atr = atr_values[-1]
            
            if trend == 'uptrend':
                # Find the swing for pullback measurement
                lookback_start = max(0, len(df) - 50)
                recent_data = df.iloc[lookback_start:]
                
                trend_end_idx = recent_data['high'].idxmax()
                trend_end_price = recent_data.loc[trend_end_idx, 'high']
                
                data_before_high = df.loc[:trend_end_idx]
                search_start = max(0, len(data_before_high) - 50)
                
                trend_start_idx = data_before_high.iloc[search_start:]['low'].idxmin()
                trend_start_price = data_before_high.loc[trend_start_idx, 'low']
                
                # Calculate swing range correctly for uptrend
                swing_range = trend_end_price - trend_start_price
                if swing_range < current_atr:
                    return {'valid_setup': False, 'fib_level': None}
                
                # NEW: Pre-check profit potential BEFORE validating pullback
                room_to_target = (trend_end_price - current_price) / pip_size
                estimated_stop_pips = self.estimate_quick_stop_distance(current_price, trend_start_price)
                
                # Minimum RR check BEFORE looking for pullback levels
                min_rr = 1.5
                if self.market_params and 'typical_swing_size' in self.market_params:
                    typical_swing = self.market_params['typical_swing_size'].get('medium', 50)
                    if typical_swing < 30:
                        min_rr = 1.2
                    elif typical_swing > 60:
                        min_rr = 2.0

                required_room = estimated_stop_pips * min_rr
                logging.info(f"[{self.market_data.symbol}] UPTREND - Room to target: {room_to_target:.1f} pips, Required: {required_room:.1f} pips")

                if room_to_target < required_room:
                    return {
                        'valid_setup': False, 
                        'fib_level': None,
                        'reason': f'Insufficient room: {room_to_target:.1f} pips, need {required_room:.1f}'
                    }
                
                # Check against market-specific pullback levels
                for level_name, level_ratio in sorted(pullback_levels.items(), key=lambda x: x[1]):
                    level_price = trend_end_price - (swing_range * level_ratio)
                    
                    # Use market-specific tolerance
                    if self.market_params and 'session_volatility' in self.market_params:
                        current_session = self.get_current_session()
                        if current_session in self.market_params['session_volatility']:
                            typical_range = self.market_params['session_volatility'][current_session]['median_range']
                            tolerance = max(typical_range * 0.1 * pip_size, swing_range * 0.02)
                        else:
                            tolerance = swing_range * 0.03
                    else:
                        tolerance = swing_range * 0.03
                    
                    if abs(current_price - level_price) <= tolerance + spread:
                        bars_since_high = len(df) - df.index.get_loc(trend_end_idx)
                        if bars_since_high >= 5:
                            return {
                                'valid_setup': True, 
                                'fib_level': f"{level_ratio:.3f}",  # Actual market ratio
                                'level_name': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance),
                                'trend_start': trend_start_price,
                                'trend_end': trend_end_price,
                                'swing_range_pips': swing_range / pip_size
                            }
                            
            else:  # downtrend
                lookback_start = max(0, len(df) - 50)
                recent_data = df.iloc[lookback_start:]
                
                trend_end_idx = recent_data['low'].idxmin()
                trend_end_price = recent_data.loc[trend_end_idx, 'low']
                
                data_before_low = df.loc[:trend_end_idx]
                search_start = max(0, len(data_before_low) - 50)
                
                trend_start_idx = data_before_low.iloc[search_start:]['high'].idxmax()
                trend_start_price = data_before_low.loc[trend_start_idx, 'high']
                
                # Calculate swing range correctly for downtrend
                swing_range = trend_start_price - trend_end_price
                if swing_range < current_atr:
                    return {'valid_setup': False, 'fib_level': None}
                
                # NEW: Pre-check profit potential BEFORE validating pullback
                room_to_target = (current_price - trend_end_price) / pip_size
                estimated_stop_pips = self.estimate_quick_stop_distance(current_price, trend_start_price)
                
                # Minimum RR check BEFORE looking for pullback levels
                min_rr = 1.5
                if self.market_params and 'typical_swing_size' in self.market_params:
                    typical_swing = self.market_params['typical_swing_size'].get('medium', 50)
                    if typical_swing < 30:
                        min_rr = 1.2
                    elif typical_swing > 60:
                        min_rr = 2.0

                required_room = estimated_stop_pips * min_rr
                logging.info(f"[{self.market_data.symbol}] DOWNTREND - Room to target: {room_to_target:.1f} pips, Required: {required_room:.1f} pips")

                if room_to_target < required_room:
                    return {
                        'valid_setup': False, 
                        'fib_level': None,
                        'reason': f'Insufficient room: {room_to_target:.1f} pips, need {required_room:.1f}'
                    }
                
                # Check against market-specific pullback levels
                for level_name, level_ratio in sorted(pullback_levels.items(), key=lambda x: x[1]):
                    level_price = trend_end_price + (swing_range * level_ratio)
                    
                    # Use market-specific tolerance
                    if self.market_params and 'session_volatility' in self.market_params:
                        current_session = self.get_current_session()
                        if current_session in self.market_params['session_volatility']:
                            typical_range = self.market_params['session_volatility'][current_session]['median_range']
                            tolerance = max(typical_range * 0.1 * pip_size, swing_range * 0.02)
                        else:
                            tolerance = swing_range * 0.03
                    else:
                        tolerance = swing_range * 0.03
                    
                    if abs(current_price - level_price) <= tolerance + spread:
                        bars_since_low = len(df) - df.index.get_loc(trend_end_idx)
                        if bars_since_low >= 5:
                            return {
                                'valid_setup': True, 
                                'fib_level': f"{level_ratio:.3f}",
                                'level_name': level_name,
                                'entry_zone': (level_price - tolerance, level_price + tolerance),
                                'trend_start': trend_start_price,
                                'trend_end': trend_end_price,
                                'swing_range_pips': swing_range / pip_size
                            }
            
            return {'valid_setup': False, 'fib_level': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_medium_timeframe: {str(e)}")
            return {'valid_setup': False, 'fib_level': None}
    
    def analyze_lower_timeframe(self, data, trade_direction):
        """
        Detect entry signals using market-specific break characteristics
        """
        try:
            df = data.copy()
            
            if len(df) < 20:
                return {'valid_entry': False, 'entry_type': None}
            
            # Get market-specific break requirements
            break_requirements = self.get_break_requirements()
            
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
                    last_closed['close'] > last_swing_high + spread
                )
                
                if break_occurred:
                    # Evaluate break quality using market-specific criteria
                    candle_range = last_closed['high'] - last_closed['low']
                    body_size = abs(last_closed['close'] - last_closed['open'])
                    
                    if candle_range > 0:
                        body_ratio = body_size / candle_range
                        
                        # Calculate volume spike
                        recent_volume = df.iloc[-20:-2]['tick_volume'].mean()
                        volume_spike = last_closed['tick_volume'] / recent_volume if recent_volume > 0 else 1
                        
                        # Check if break meets market-specific criteria
                        min_body_ratio = break_requirements.get('min_body_ratio', 0.6)
                        min_volume_spike = break_requirements.get('min_volume_spike', 1.0)
                        
                        if body_ratio >= min_body_ratio and volume_spike >= min_volume_spike:
                            entry_price = last_swing_high + spread + (2 * pip_size)
                            
                            # Only enter if price hasn't run away
                            max_distance = break_requirements.get('max_entry_distance_pips', 5) * pip_size
                            if symbol_tick.ask <= entry_price + max_distance:
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_resistance',
                                    'break_level': last_swing_high,
                                    'entry_price': min(symbol_tick.ask, entry_price),
                                    'candle_close': last_closed['close'],
                                    'candle_strength': body_ratio,
                                    'volume_spike': volume_spike,
                                    'spread_at_signal': spread
                                }
                                
            else:  # sell
                last_swing_low = swing_lows.iloc[-1]['price']
                
                break_occurred = (
                    prev_candle['close'] >= last_swing_low and   
                    last_closed['close'] < last_swing_low - spread
                )
                
                if break_occurred:
                    candle_range = last_closed['high'] - last_closed['low']
                    body_size = abs(last_closed['close'] - last_closed['open'])
                    
                    if candle_range > 0:
                        body_ratio = body_size / candle_range
                        
                        recent_volume = df.iloc[-20:-2]['tick_volume'].mean()
                        volume_spike = last_closed['tick_volume'] / recent_volume if recent_volume > 0 else 1
                        
                        min_body_ratio = break_requirements.get('min_body_ratio', 0.6)
                        min_volume_spike = break_requirements.get('min_volume_spike', 1.0)
                        
                        if body_ratio >= min_body_ratio and volume_spike >= min_volume_spike:
                            entry_price = last_swing_low - spread - (2 * pip_size)
                            
                            max_distance = break_requirements.get('max_entry_distance_pips', 5) * pip_size
                            if symbol_tick.bid >= entry_price - max_distance:
                                return {
                                    'valid_entry': True, 
                                    'entry_type': 'break_of_support',
                                    'break_level': last_swing_low,
                                    'entry_price': max(symbol_tick.bid, entry_price),
                                    'candle_close': last_closed['close'],
                                    'candle_strength': body_ratio,
                                    'volume_spike': volume_spike,
                                    'spread_at_signal': spread
                                }
            
            return {'valid_entry': False, 'entry_type': None}
            
        except Exception as e:
            logging.error(f"Error in analyze_lower_timeframe: {str(e)}")
            return {'valid_entry': False, 'entry_type': None}
    
    def get_break_requirements(self):
        """Get market-specific requirements for valid breaks"""
        requirements = {
            'min_body_ratio': 0.6,
            'min_volume_spike': 1.0,
            'max_entry_distance_pips': 5
        }
        
        if self.market_params and 'break_success_factors' in self.market_params:
            factors = self.market_params['break_success_factors']
            
            # Use actual success thresholds from market data
            if 'body_ratio' in factors and factors['body_ratio'].get('significant'):
                requirements['min_body_ratio'] = factors['body_ratio'].get('success_threshold', 0.6)
            
            if 'volume_spike' in factors and factors['volume_spike'].get('significant'):
                requirements['min_volume_spike'] = factors['volume_spike'].get('success_threshold', 1.0)
            
            if 'distance_from_level' in factors:
                requirements['max_entry_distance_pips'] = factors['distance_from_level'].get('success_mean', 5)
        
        return requirements
    
    def check_m5_momentum(self, data, trade_direction):
        """
        Check momentum using market-specific patterns
        """
        df = data.copy()
        if len(df) < 20:
            return False
        
        # Get market-specific momentum requirements
        momentum_reqs = self.get_momentum_requirements(trade_direction)
        
        recent_candles = df.tail(5)
        
        if trade_direction == 'buy':
            # Count consecutive bullish candles
            consecutive_bulls = 0
            max_consecutive = 0
            
            for i in range(len(recent_candles)):
                if recent_candles.iloc[i]['close'] > recent_candles.iloc[i]['open']:
                    consecutive_bulls += 1
                    max_consecutive = max(max_consecutive, consecutive_bulls)
                else:
                    consecutive_bulls = 0
            
            # Check extension from recent low
            recent_low = df.tail(10)['low'].min()
            current_price = df.iloc[-1]['close']
            extension = (current_price - recent_low) / self.market_data.get_pip_size()
            
            # Use market-specific limits
            max_consecutive_allowed = momentum_reqs.get('max_consecutive_candles', 3)
            max_extension_allowed = momentum_reqs.get('max_extension_pips', 30)
            
            return max_consecutive < max_consecutive_allowed and extension < max_extension_allowed
            
        else:  # sell
            consecutive_bears = 0
            max_consecutive = 0
            
            for i in range(len(recent_candles)):
                if recent_candles.iloc[i]['close'] < recent_candles.iloc[i]['open']:
                    consecutive_bears += 1
                    max_consecutive = max(max_consecutive, consecutive_bears)
                else:
                    consecutive_bears = 0
            
            recent_high = df.tail(10)['high'].max()
            current_price = df.iloc[-1]['close']
            extension = (recent_high - current_price) / self.market_data.get_pip_size()
            
            max_consecutive_allowed = momentum_reqs.get('max_consecutive_candles', 3)
            max_extension_allowed = momentum_reqs.get('max_extension_pips', 30)
            
            return max_consecutive < max_consecutive_allowed and extension < max_extension_allowed
    
    def get_momentum_requirements(self, trade_direction):
        """Get market-specific momentum requirements"""
        requirements = {
            'max_consecutive_candles': 3,
            'max_extension_pips': 30
        }
        
        if self.market_params and 'momentum_characteristics' in self.market_params:
            momentum_data = self.market_params['momentum_characteristics']
            
            if trade_direction == 'buy' and 'bullish' in momentum_data:
                # Adjust based on what actually works for this pair
                avg_consolidation = momentum_data['bullish'].get('avg_consolidation_bars', 5)
                requirements['max_consecutive_candles'] = max(2, int(avg_consolidation * 0.6))
                
                # Use session-specific extension limits if available
                current_session = self.get_current_session()
                if 'session_volatility' in self.market_params and current_session in self.market_params['session_volatility']:
                    typical_range = self.market_params['session_volatility'][current_session]['percentile_75']
                    requirements['max_extension_pips'] = typical_range * 0.7
                    
            elif trade_direction == 'sell' and 'bearish' in momentum_data:
                avg_consolidation = momentum_data['bearish'].get('avg_consolidation_bars', 5)
                requirements['max_consecutive_candles'] = max(2, int(avg_consolidation * 0.6))
                
                current_session = self.get_current_session()
                if 'session_volatility' in self.market_params and current_session in self.market_params['session_volatility']:
                    typical_range = self.market_params['session_volatility'][current_session]['percentile_75']
                    requirements['max_extension_pips'] = typical_range * 0.7
        
        return requirements
    
    def get_current_session(self):
        """Get current trading session"""
        hour = datetime.now().hour
        if 22 <= hour or hour < 7:
            return 'Asian'
        elif 7 <= hour < 14:
            return 'London'
        elif 14 <= hour < 22:
            return 'NewYork'
        return 'Unknown'

    def get_current_spread(self):
        """Get current spread in price units"""
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        if symbol_info:
            return symbol_info.spread * symbol_info.point
        return 0

    def check_spread_acceptable(self):
        """Check if spread is acceptable based on market-specific data"""
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        if not symbol_info:
            return False
        
        current_spread = symbol_info.spread
        current_hour = datetime.now().hour
        
        # Use market-specific spread data if available
        if (self.market_params and 
            'typical_spreads_by_hour' in self.market_params and 
            str(current_hour) in self.market_params['typical_spreads_by_hour']):
            
            hour_data = self.market_params['typical_spreads_by_hour'][str(current_hour)]
            max_acceptable = hour_data.get('90th', hour_data.get('75th', 30))
            
            return current_spread <= max_acceptable
        else:
            # Fallback to reasonable defaults
            symbol = self.market_data.symbol
            if 'JPY' in symbol:
                max_spread = 30
            else:
                max_spread = 30
            
            return current_spread <= max_spread
    
    def estimate_stop_distance(self, symbol, trade_direction, ltf_break_level, current_price, atr_value):
        """
        Estimate stop distance using market-specific data
        """
        pip_size = self.market_data.get_pip_size()
        symbol_info = mt5.symbol_info(symbol)
        spread = symbol_info.spread * symbol_info.point
        
        # Get market-specific buffer
        if self.market_params and 'stop_buffer_pips' in self.market_params:
            buffer_pips = self.market_params['stop_buffer_pips'].get('normal', 15)
            buffer = buffer_pips * pip_size
        else:
            buffer = atr_value * 0.25
        
        if trade_direction == 'buy':
            estimated_stop = ltf_break_level - buffer - spread
            estimated_stop_distance = current_price - estimated_stop
        else:  # sell
            estimated_stop = ltf_break_level + buffer + spread
            estimated_stop_distance = estimated_stop - current_price
        
        # Ensure minimum distance
        min_stop = max(20 * pip_size, symbol_info.trade_stops_level * symbol_info.point)
        estimated_stop_distance = max(estimated_stop_distance, min_stop)
        
        return estimated_stop_distance / pip_size

    def check_for_signals(self, symbol):
        """Main signal checking method with market-adaptive logic"""
        try:
            # Update market knowledge if needed
            self.update_market_knowledge()
            
            # Check if we have enough data
            if not self.market_params:
                logging.warning(f"[{symbol}] No market behavior data yet")
                return False, f"[{symbol}] Still learning market behavior", None
            
            # Check trading hours
            current_hour = datetime.now().hour
            if 'best_hours' in self.market_params and self.market_params['best_hours']:
                if current_hour not in self.market_params['best_hours']:
                    return False, f"[{symbol}] Not optimal trading hour (best: {self.market_params['best_hours']})", None
            
            # Check spread
            if not self.check_spread_acceptable():
                return False, f"[{symbol}] Spread too high for current hour", None
            
            # Check position limits
            positions = mt5.positions_get()
            if positions:
                total_positions = len(positions)
                symbol_positions = len([p for p in positions if p.symbol == symbol])
                
                if total_positions >= self.max_positions:
                    return False, f"Max total positions ({self.max_positions}) reached", None
                    
                if symbol_positions >= self.max_per_symbol:
                    return False, f"Max positions for {symbol} ({self.max_per_symbol}) reached", None

            # Fetch all data at once
            data = {}
            fetch_time = time.time()
            for tf in self.timeframes:
                data[tf] = self.market_data.fetch_data(tf)
                if data[tf] is None:
                    logging.warning(f"[{symbol}] Failed to fetch data for timeframe {tf}")
                    return False, f"[{symbol}] No data for timeframe {tf}", None
            
            # Ensure data fetching was quick
            if time.time() - fetch_time > 1:
                logging.warning(f"[{symbol}] Data fetching took too long, skipping")
                return False, f"[{symbol}] Data synchronization issue", None

            # 1. Higher timeframe trend
            htf_analysis = self.analyze_higher_timeframe(data[self.tf_higher])
            
            # Use market-specific minimum trend strength
            min_trend_strength = 0.5
            if self.market_params and 'session_volatility' in self.market_params:
                current_session = self.get_current_session()
                if current_session in self.market_params['session_volatility']:
                    # Lower requirements during high volatility sessions
                    session_data = self.market_params['session_volatility'][current_session]
                    if session_data['avg_range'] > session_data['percentile_75']:
                        min_trend_strength = 0.4
            
            if htf_analysis['trend'] == 'unclear' or htf_analysis['strength'] < min_trend_strength:
                return False, f"[{symbol}] No clear/strong trend on HTF (strength: {htf_analysis['strength']:.2f})", None
            
            logging.info(f"[{symbol}] HTF trend: {htf_analysis['trend']} (strength: {htf_analysis['strength']:.2f})")
                        
            # 2. Medium timeframe setup
            mtf_analysis = self.analyze_medium_timeframe(data[self.tf_medium], htf_analysis)
            if not mtf_analysis['valid_setup']:
                return False, f"[{symbol}] No valid pullback on MTF", None
            
            logging.info(f"[{symbol}] MTF analysis: {mtf_analysis}")
            if not mtf_analysis['valid_setup']:
                logging.info(f"[{symbol}] No valid pullback on MTF (fib_level={mtf_analysis.get('fib_level')})")
                return False, f"[{symbol}] No valid pullback on MTF", None                

            logging.info(f"[{symbol}] MTF pullback to {mtf_analysis['level_name']} ({mtf_analysis['fib_level']})")

            # 3. Lower timeframe entry
            trade_direction = 'buy' if htf_analysis['trend'] == 'uptrend' else 'sell'
            ltf_analysis = self.analyze_lower_timeframe(data[self.tf_lower], trade_direction)
            
            logging.info(f"[{symbol}] LTF analysis: {ltf_analysis}")
            if not ltf_analysis['valid_entry']:
                logging.info(f"[{symbol}] LTF no entry: {ltf_analysis.get('reason')}")
                if self.check_m5_momentum(data[self.tf_lower], trade_direction):
                    # Momentum entry at pullback level
                    if trade_direction == 'buy':
                        break_level = mtf_analysis['entry_zone'][0]
                    else:
                        break_level = mtf_analysis['entry_zone'][1]
                        
                    symbol_tick = mt5.symbol_info_tick(symbol)
                    current_spread = self.get_current_spread()
                        
                    ltf_analysis = {
                        'valid_entry': True,
                        'entry_type': 'pullback_with_momentum',
                        'break_level': break_level,
                        'entry_price': symbol_tick.ask if trade_direction == 'buy' else symbol_tick.bid,
                        'spread_at_signal': current_spread,
                        'candle_strength': 0.0,
                        'volume_spike': 0.0
                    }
                else:
                    return False, f"[{symbol}] Waiting for M5 entry signal", None

            logging.info(f"[{symbol}] Entry signal: {ltf_analysis['entry_type']}")
            
            # Calculate ATR
            atr_values = ATR(data[self.tf_higher]['high'].values, 
                           data[self.tf_higher]['low'].values, 
                           data[self.tf_higher]['close'].values, 
                           timeperiod=14)
            atr_value = atr_values[-1]
            
            if np.isnan(atr_value) or atr_value <= 0:
                return False, f"[{symbol}] Invalid ATR calculation", None
            
            # 4. Risk/Reward validation
            current_price = data[self.tf_lower]['close'].iloc[-1]
            pip_size = self.market_data.get_pip_size()
            
            estimated_stop_pips = self.estimate_stop_distance(
                symbol,
                trade_direction,
                ltf_analysis.get('break_level'),
                current_price,
                atr_value
            )
            
            logging.info(f"[{symbol}] Estimated stop: {estimated_stop_pips:.1f} pips")
            
            # Check room to target
            if trade_direction == 'buy':
                target = mtf_analysis.get('trend_end')
                room_to_target = target - current_price
            else:
                target = mtf_analysis.get('trend_end')
                room_to_target = current_price - target
            
            room_in_pips = room_to_target / pip_size
            
            # Use market-specific minimum RR
            min_rr_required = 1.5
            if self.market_params and 'typical_swing_size' in self.market_params:
                typical_swing = self.market_params['typical_swing_size'].get('medium', 50)
                if typical_swing < 30:
                    min_rr_required = 1.2  # Smaller swings = accept lower RR
                elif typical_swing > 60:
                    min_rr_required = 2.0  # Larger swings = require better RR
            
            min_room_required = estimated_stop_pips * min_rr_required
            
            if room_in_pips < min_room_required:
                actual_rr = room_in_pips / estimated_stop_pips
                logging.warning(f"[{symbol}] Poor RR - Room: {room_in_pips:.1f} pips, "
                              f"Need: {min_room_required:.1f} pips, "
                              f"Potential RR: 1:{actual_rr:.1f}")
                return False, f"[{symbol}] Insufficient RR (1:{actual_rr:.1f})", None
                                    
            logging.info(f"[{symbol}] Room check passed - Potential RR: 1:{(room_in_pips/estimated_stop_pips):.1f}")

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
                entry_analysis,
                self.market_params
            )
            
            # Dynamic position sizing based on setup quality and market behavior
            base_risk = 0.01
            
            # Adjust risk based on multiple factors
            risk_multiplier = 1.0
            
            # Factor 1: Trend strength
            if htf_analysis['strength'] > 0.7:
                risk_multiplier *= 1.1
            elif htf_analysis['strength'] < 0.5:
                risk_multiplier *= 0.9
            
            # Factor 2: Pullback quality (deeper is often better)
            pullback_ratio = float(mtf_analysis['fib_level'])
            if pullback_ratio > 0.5:
                risk_multiplier *= 1.1
            elif pullback_ratio < 0.35:
                risk_multiplier *= 0.8
            
            # Factor 3: Entry quality
            if ltf_analysis['entry_type'] == 'break_of_resistance' or ltf_analysis['entry_type'] == 'break_of_support':
                if ltf_analysis.get('volume_spike', 1) > 1.5:
                    risk_multiplier *= 1.1
            
            # Factor 4: Market conditions
            if self.market_params and 'break_success_factors' in self.market_params:
                current_session = self.get_current_session()
                session_success = self.market_params['break_success_factors'].get('session_success_rate', {})
                if current_session in session_success:
                    if session_success[current_session] > 0.65:
                        risk_multiplier *= 1.1
                    elif session_success[current_session] < 0.45:
                        risk_multiplier *= 0.7
            
            # Apply risk multiplier with bounds
            risk_percent = base_risk * max(0.5, min(1.5, risk_multiplier))
                
            lots_size = self.order_manager.calculate_lot_size(
                symbol, 
                account_info, 
                stop_loss_pips,
                self.market_params,
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
                        'pullback_level': mtf_analysis['fib_level'],
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


class AdaptiveOrderManager:
    """Order manager that uses market-specific parameters"""
    
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
            
    def calculate_sl_tp(self, symbol, order_type, atr_value, entry_analysis, market_params):
        """
        Calculate stop loss and take profit using market-specific behavior
        """
        symbol_data = self.get_symbol_info(symbol)
        symbol_tick = symbol_data['tick']
        symbol_info = symbol_data['info']
        pip_size = symbol_data['pip_size']
                
        # Get the entry level
        ltf_break = entry_analysis.get('ltf_break_level')
        if not ltf_break:
            raise ValueError("No break level found in entry analysis")
        
        # Get current price
        if order_type == 'buy':
            current_price = symbol_tick.ask
            
            # Use market-specific stop buffer
            if market_params and 'stop_buffer_pips' in market_params:
                buffer_pips = market_params['stop_buffer_pips'].get('safe', 20)
                buffer = buffer_pips * pip_size
            else:
                # Fallback to ATR-based
                buffer = atr_value * 0.3
            
            # Place stop below structure
            stop_loss = ltf_break - buffer
            
            # Round to psychological level if market data suggests it helps
            if market_params and 'stop_hunt_distances' in market_params:
                # Make stop less obvious
                remainder = (stop_loss / pip_size) % 10
                if remainder < 2 or remainder > 8:
                    stop_loss -= 3 * pip_size
            
            # Take profit calculation
            htf_resistance = entry_analysis.get('htf_swings', {}).get('last_swing_high')
            
            # Use market-specific swing size for targets
            if market_params and 'typical_swing_size' in market_params:
                typical_swing = market_params['typical_swing_size'].get('large', 50) * pip_size
                atr_target = current_price + min(atr_value * 2.0, typical_swing)
            else:
                atr_target = current_price + (atr_value * 2.0)
            
            if htf_resistance and htf_resistance > current_price:
                take_profit = min(htf_resistance - (3 * pip_size), atr_target)
            else:
                take_profit = atr_target
                
        else:  # sell
            current_price = symbol_tick.bid
            
            # Market-specific stop buffer
            if market_params and 'stop_buffer_pips' in market_params:
                buffer_pips = market_params['stop_buffer_pips'].get('safe', 20)
                buffer = buffer_pips * pip_size
            else:
                buffer = atr_value * 0.3
            
            stop_loss = ltf_break + buffer
            
            # Round to psychological level
            if market_params and 'stop_hunt_distances' in market_params:
                remainder = (stop_loss / pip_size) % 10
                if remainder < 2 or remainder > 8:
                    stop_loss += 3 * pip_size
            
            # Take profit
            htf_support = entry_analysis.get('htf_swings', {}).get('last_swing_low')
            
            if market_params and 'typical_swing_size' in market_params:
                typical_swing = market_params['typical_swing_size'].get('large', 50) * pip_size
                atr_target = current_price - min(atr_value * 2.0, typical_swing)
            else:
                atr_target = current_price - (atr_value * 2.0)
            
            if htf_support and htf_support < current_price:
                take_profit = max(htf_support + (3 * pip_size), atr_target)
            else:
                take_profit = atr_target
        
        # Ensure minimum stop distance
        min_stop_distance = max(
            symbol_info.trade_stops_level * symbol_info.point,
            20 * pip_size
        )
        
        if order_type == 'buy' and current_price - stop_loss < min_stop_distance:   
            stop_loss = current_price - min_stop_distance
        elif order_type == 'sell' and stop_loss - current_price < min_stop_distance:
            stop_loss = current_price + min_stop_distance
        
        # Ensure minimum RR based on market behavior
        min_rr = 1.5
        if market_params and 'typical_swing_size' in market_params:
            # Adjust minimum RR based on typical market moves
            typical_swing = market_params['typical_swing_size'].get('medium', 50)
            if typical_swing < 30:
                min_rr = 1.2  # Accept lower RR for pairs with smaller moves
            elif typical_swing > 60:
                min_rr = 2.0  # Require better RR for pairs with larger moves
        
        stop_distance = abs(current_price - stop_loss)
        profit_distance = abs(take_profit - current_price)
        
        if profit_distance < stop_distance * min_rr:
            if order_type == 'buy':
                take_profit = current_price + (stop_distance * min_rr)
            else:
                take_profit = current_price - (stop_distance * min_rr)
        
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
        
    def calculate_lot_size(self, symbol, account_info, stop_loss_pips, market_params, risk_percent=0.01):
        """
        Calculate lot size with market-specific adjustments
        """
        # Get symbol specifics
        symbol_data = self.get_symbol_info(symbol)
        symbol_info = symbol_data['info']
        pip_value_per_lot = symbol_data['pip_value_per_lot']
                
        # Use equity for compounding
        equity = account_info.equity
        
        # Adjust risk based on market behavior
        adjusted_risk = risk_percent
        
        # Adjustment 1: Stop distance relative to typical
        if market_params and 'stop_buffer_pips' in market_params:
            typical_stop = market_params['stop_buffer_pips'].get('normal', 25)
            
            if stop_loss_pips < typical_stop * 0.8:
                # Tighter than usual stop = higher risk of noise
                adjusted_risk *= 0.8
                logging.info(f"[{symbol}] Tight stop {stop_loss_pips:.1f} vs typical {typical_stop:.1f} - reducing risk")
            elif stop_loss_pips > typical_stop * 1.5:
                # Wider than usual = poor entry or high volatility
                adjusted_risk *= 0.7
                logging.info(f"[{symbol}] Wide stop {stop_loss_pips:.1f} vs typical {typical_stop:.1f} - reducing risk")
        
        # Adjustment 2: Current session success rate
        if market_params and 'break_success_factors' in market_params:
            current_hour = datetime.now().hour
            current_session = 'Asian' if (22 <= current_hour or current_hour < 7) else 'London' if (7 <= current_hour < 14) else 'NewYork'
            
            session_success = market_params['break_success_factors'].get('session_success_rate', {})
            if current_session in session_success:
                success_rate = session_success[current_session]
                if success_rate < 0.4:
                    adjusted_risk *= 0.7
                    logging.info(f"[{symbol}] Poor {current_session} session success rate - reducing risk")
                elif success_rate > 0.7:
                    adjusted_risk *= 1.2
                    logging.info(f"[{symbol}] Strong {current_session} session success rate - increasing risk")
        
        # Calculate risk amount
        risk_amount = equity * adjusted_risk
        
        # Basic lot calculation
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        
        # Round to broker's lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply broker constraints
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # Leverage check
        position_value = lot_size * symbol_info.trade_contract_size
        leverage_used = position_value / equity
        
        max_leverage_per_position = 30
        
        if leverage_used > max_leverage_per_position:
            lot_size = (equity * max_leverage_per_position) / symbol_info.trade_contract_size
            lot_size = math.floor(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, lot_size)
            
            new_leverage = (lot_size * symbol_info.trade_contract_size) / equity
            logging.warning(f"[{symbol}] Leverage cap: {leverage_used:.1f}x -> {new_leverage:.1f}x")
        
        # Margin check
        margin_required = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY,
            symbol,
            lot_size,
            symbol_data['tick'].ask
        )
        
        if margin_required:
            free_margin = account_info.margin_free
            if margin_required > free_margin * 0.4:
                lot_size = (free_margin * 0.4 / margin_required) * lot_size
                lot_size = math.floor(lot_size / lot_step) * lot_step
                lot_size = max(symbol_info.volume_min, lot_size)
                logging.warning(f"[{symbol}] Margin constraint applied")
        
        # Final risk check
        actual_risk_amount = lot_size * stop_loss_pips * pip_value_per_lot
        actual_risk_percent = (actual_risk_amount / equity) * 100
        
        if actual_risk_percent > 2.0:
            lot_size = (equity * 0.02) / (stop_loss_pips * pip_value_per_lot)
            lot_size = math.floor(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, lot_size)
            logging.warning(f"[{symbol}] Risk cap applied: {actual_risk_percent:.1f}% -> 2.0%")
        
        # Final metrics
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
        # Initialize market data and adaptive trade managers
        market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
        trade_managers = {symbol: AdaptiveTradeManager(client, market_data_dict[symbol]) for symbol in symbols}
        
        logging.info("Adaptive trading system initialized")
        logging.info("Learning market behavior for all symbols...")

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
        logging.info("Adaptive trading system terminated")

if __name__ == "__main__":
    main()