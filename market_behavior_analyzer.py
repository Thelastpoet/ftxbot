import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import json
import os
import logging
from scipy import stats

class MarketBehaviorAnalyzer:
    """
    Learns how each forex pair actually behaves instead of using magic numbers
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.behavior_file = f"market_behavior_{symbol}.json"
        self.behavior = self.load_behavior()
        
    def load_behavior(self):
        """Load existing behavior data or start fresh"""
        if os.path.exists(self.behavior_file):
            with open(self.behavior_file, 'r') as f:
                return json.load(f)
        return {
            'pullback_depths': [],
            'stop_hunt_distances': [],
            'successful_break_characteristics': [],
            'failed_break_characteristics': [],
            'trend_durations': [],
            'reversal_patterns': [],
            'session_volatility': {},
            'typical_swing_sizes': [],
            'spread_patterns': [],
            'momentum_patterns': []
        }
    
    def analyze_historical_data(self, days_back=60):
        """
        Scan historical data to learn how this specific pair behaves
        """
        logging.info(f"[{self.symbol}] Starting behavior analysis for {days_back} days...")
        
        # Fetch H1 data for trend analysis
        rates_h1 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, days_back * 24)
        if rates_h1 is None or len(rates_h1) == 0:
            logging.error(f"[{self.symbol}] Failed to fetch H1 data")
            return
            
        df_h1 = pd.DataFrame(rates_h1)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        
        # Fetch M15 data for more detailed analysis
        rates_m15 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M15, 0, days_back * 24 * 4)
        if rates_m15 is None or len(rates_m15) == 0:
            logging.error(f"[{self.symbol}] Failed to fetch M15 data")
            return
            
        df_m15 = pd.DataFrame(rates_m15)
        df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
        
        # Fetch M5 data for entry analysis
        rates_m5 = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, days_back * 24 * 12)
        if rates_m5 is None or len(rates_m5) == 0:
            logging.error(f"[{self.symbol}] Failed to fetch M5 data")
            return
            
        df_m5 = pd.DataFrame(rates_m5)
        df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
        
        # Clear old data to start fresh
        self.behavior = {
            'pullback_depths': [],
            'stop_hunt_distances': [],
            'successful_break_characteristics': [],
            'failed_break_characteristics': [],
            'trend_durations': [],
            'reversal_patterns': [],
            'session_volatility': {},
            'typical_swing_sizes': [],
            'spread_patterns': [],
            'momentum_patterns': []
        }
        
        # Run all analyses
        self.measure_pullback_behavior(df_h1, df_m15)
        self.measure_stop_hunt_behavior(df_h1, df_m5)
        self.measure_break_behavior(df_m15, df_m5)
        self.measure_session_behavior(df_h1)
        self.measure_swing_sizes(df_h1)
        self.measure_spread_patterns()
        self.measure_momentum_patterns(df_m5)
        
        self.save_behavior()
        logging.info(f"[{self.symbol}] Behavior analysis complete")
        
    def measure_pullback_behavior(self, df_h1, df_m15):
        """
        Learn how deep pullbacks typically go in trends for THIS pair
        """
        pip_size = self.get_pip_size()
        
        # Find all significant trends on H1
        for i in range(20, len(df_h1) - 20):
            window = df_h1.iloc[i-20:i]
            
            # Identify uptrend
            if self.is_trending_up(window):
                trend_start_idx = i - 20
                trend_end_idx = i
                
                trend_low = window['low'].min()
                trend_high = window['high'].max()
                trend_range = trend_high - trend_low
                
                if trend_range < 20 * pip_size:  # Skip tiny moves
                    continue
                
                # Find pullback in next 20 H1 bars
                future = df_h1.iloc[i:min(i+20, len(df_h1))]
                
                for j in range(1, len(future)):
                    if future.iloc[j]['low'] < future.iloc[j-1]['low']:
                        # Found a pullback
                        pullback_low = future.iloc[j]['low']
                        pullback_depth = (trend_high - pullback_low) / trend_range
                        
                        # Get more detailed info from M15
                        time_start = df_h1.iloc[trend_start_idx]['time']
                        time_end = future.iloc[j]['time']
                        
                        m15_data = df_m15[(df_m15['time'] >= time_start) & (df_m15['time'] <= time_end)]
                        
                        if len(m15_data) > 0:
                            # Check if pullback held (price went back up)
                            remaining_future = future.iloc[j+1:]
                            if len(remaining_future) > 0 and remaining_future['high'].max() > trend_high * 0.995:
                                # This was a successful pullback
                                self.behavior['pullback_depths'].append({
                                    'depth_ratio': pullback_depth,
                                    'trend_size_pips': trend_range / pip_size,
                                    'hour': window.iloc[-1]['time'].hour,
                                    'day_of_week': window.iloc[-1]['time'].dayofweek,
                                    'pullback_bars': j,
                                    'trend_bars': 20,
                                    'session': self.get_session(window.iloc[-1]['time'])
                                })
                        break
            
            # Identify downtrend
            elif self.is_trending_down(window):
                trend_start_idx = i - 20
                trend_end_idx = i
                
                trend_high = window['high'].max()
                trend_low = window['low'].min()
                trend_range = trend_high - trend_low
                
                if trend_range < 20 * pip_size:
                    continue
                
                # Find pullback in next 20 H1 bars
                future = df_h1.iloc[i:min(i+20, len(df_h1))]
                
                for j in range(1, len(future)):
                    if future.iloc[j]['high'] > future.iloc[j-1]['high']:
                        # Found a pullback
                        pullback_high = future.iloc[j]['high']
                        pullback_depth = (pullback_high - trend_low) / trend_range
                        
                        # Check if pullback held
                        remaining_future = future.iloc[j+1:]
                        if len(remaining_future) > 0 and remaining_future['low'].min() < trend_low * 1.005:
                            self.behavior['pullback_depths'].append({
                                'depth_ratio': pullback_depth,
                                'trend_size_pips': trend_range / pip_size,
                                'hour': window.iloc[-1]['time'].hour,
                                'day_of_week': window.iloc[-1]['time'].dayofweek,
                                'pullback_bars': j,
                                'trend_bars': 20,
                                'session': self.get_session(window.iloc[-1]['time'])
                            })
                        break
    
    def measure_stop_hunt_behavior(self, df_h1, df_m5):
        """
        Learn how far beyond swings price typically goes before reversing
        """
        pip_size = self.get_pip_size()
        
        # Find swing points on H1
        swings = self.find_swing_points(df_h1, window=5)
        
        for swing in swings:
            swing_time = swing['time']
            swing_type = swing['type']
            swing_price = swing['price']
            
            # Get M5 data around this swing
            start_time = swing_time - timedelta(hours=2)
            end_time = swing_time + timedelta(hours=6)
            
            m5_window = df_m5[(df_m5['time'] >= start_time) & (df_m5['time'] <= end_time)]
            
            if len(m5_window) == 0:
                continue
            
            # Find the swing point in M5 data
            swing_idx = m5_window[m5_window['time'] >= swing_time].index[0] if len(m5_window[m5_window['time'] >= swing_time]) > 0 else None
            
            if swing_idx is None:
                continue
            
            # Look for stop hunts in the next 30 M5 bars
            future_m5 = m5_window.loc[swing_idx:].iloc[1:31]
            
            if swing_type == 'high':
                # Look for price going above swing high
                max_hunt = future_m5['high'].max() if len(future_m5) > 0 else swing_price
                
                if max_hunt > swing_price:
                    hunt_distance = max_hunt - swing_price
                    
                    # Check if it was indeed a hunt (price came back down)
                    remaining = m5_window.loc[swing_idx:].iloc[31:61]
                    if len(remaining) > 0 and remaining['close'].iloc[-1] < swing_price:
                        self.behavior['stop_hunt_distances'].append({
                            'distance_pips': hunt_distance / pip_size,
                            'swing_type': 'high',
                            'session': self.get_session(swing_time),
                            'hour': swing_time.hour,
                            'bars_to_hunt': len(future_m5[future_m5['high'] <= swing_price]) + 1,
                            'bars_to_return': len(future_m5) + len(remaining[remaining['close'] > swing_price])
                        })
            
            else:  # swing low
                # Look for price going below swing low
                min_hunt = future_m5['low'].min() if len(future_m5) > 0 else swing_price
                
                if min_hunt < swing_price:
                    hunt_distance = swing_price - min_hunt
                    
                    # Check if it was indeed a hunt
                    remaining = m5_window.loc[swing_idx:].iloc[31:61]
                    if len(remaining) > 0 and remaining['close'].iloc[-1] > swing_price:
                        self.behavior['stop_hunt_distances'].append({
                            'distance_pips': hunt_distance / pip_size,
                            'swing_type': 'low',
                            'session': self.get_session(swing_time),
                            'hour': swing_time.hour,
                            'bars_to_hunt': len(future_m5[future_m5['low'] >= swing_price]) + 1,
                            'bars_to_return': len(future_m5) + len(remaining[remaining['close'] < swing_price])
                        })
    
    def measure_break_behavior(self, df_m15, df_m5):
        """
        Learn what distinguishes successful breaks from failures
        """
        pip_size = self.get_pip_size()
        
        # Find potential support/resistance levels on M15
        for i in range(20, len(df_m15) - 40):
            # Look for a clear level that was tested multiple times
            window = df_m15.iloc[i-20:i]
            
            # Find recent high that might act as resistance
            recent_high = window['high'].max()
            recent_high_count = len(window[abs(window['high'] - recent_high) < 5 * pip_size])
            
            if recent_high_count >= 2:  # Level was tested at least twice
                # Check for a break
                future = df_m15.iloc[i:i+40]
                
                for j in range(len(future)):
                    if future.iloc[j]['close'] > recent_high and (j == 0 or future.iloc[j-1]['close'] <= recent_high):
                        # Found a break
                        break_time = future.iloc[j]['time']
                        break_candle = future.iloc[j]
                        
                        # Get detailed M5 data around the break
                        m5_start = break_time - timedelta(hours=1)
                        m5_end = break_time + timedelta(hours=2)
                        m5_data = df_m5[(df_m5['time'] >= m5_start) & (df_m5['time'] <= m5_end)]
                        
                        if len(m5_data) == 0:
                            continue
                        
                        # Analyze break characteristics
                        break_idx = m5_data[m5_data['time'] >= break_time].index[0] if len(m5_data[m5_data['time'] >= break_time]) > 0 else None
                        
                        if break_idx is None:
                            continue
                        
                        pre_break = m5_data.loc[:break_idx].iloc[-12:] if break_idx > 12 else m5_data.loc[:break_idx]
                        post_break = m5_data.loc[break_idx:].iloc[1:25] if len(m5_data.loc[break_idx:]) > 25 else m5_data.loc[break_idx:].iloc[1:]
                        
                        # Determine if break was successful
                        success = False
                        if len(post_break) >= 12:
                            # Success = stays above level for at least 12 M5 bars
                            success = all(post_break.iloc[:12]['low'] > recent_high - 2 * pip_size)
                        
                        # Calculate break characteristics
                        break_data = {
                            'success': success,
                            'break_candle_range': (break_candle['high'] - break_candle['low']) / pip_size,
                            'break_candle_body': abs(break_candle['close'] - break_candle['open']) / pip_size,
                            'break_candle_type': 'bullish' if break_candle['close'] > break_candle['open'] else 'bearish',
                            'volume_spike': break_candle['tick_volume'] / window['tick_volume'].mean() if window['tick_volume'].mean() > 0 else 1,
                            'pre_break_momentum': (pre_break['close'].iloc[-1] - pre_break['close'].iloc[0]) / pip_size if len(pre_break) > 0 else 0,
                            'time_of_day': break_time.hour,
                            'session': self.get_session(break_time),
                            'attempts_before_break': recent_high_count,
                            'distance_from_level': (break_candle['close'] - recent_high) / pip_size
                        }
                        
                        if success:
                            self.behavior['successful_break_characteristics'].append(break_data)
                        else:
                            self.behavior['failed_break_characteristics'].append(break_data)
                        
                        break  # Only analyze first break of each level
    
    def measure_session_behavior(self, df):
        """
        Measure volatility and behavior patterns by trading session
        """
        pip_size = self.get_pip_size()
        
        sessions = {
            'Asian': {'hours': list(range(22, 24)) + list(range(0, 7)), 'data': []},
            'London': {'hours': list(range(7, 14)), 'data': []},
            'NewYork': {'hours': list(range(14, 22)), 'data': []},
            'Overlap': {'hours': list(range(14, 17)), 'data': []}  # London/NY overlap
        }
        
        for i in range(len(df)):
            bar = df.iloc[i]
            hour = bar['time'].hour
            bar_range = (bar['high'] - bar['low']) / pip_size
            
            for session_name, session_info in sessions.items():
                if hour in session_info['hours']:
                    session_info['data'].append({
                        'range': bar_range,
                        'volume': bar['tick_volume'],
                        'hour': hour,
                        'day': bar['time'].dayofweek
                    })
        
        # Calculate statistics for each session
        for session_name, session_info in sessions.items():
            if session_info['data']:
                ranges = [d['range'] for d in session_info['data']]
                volumes = [d['volume'] for d in session_info['data']]
                
                self.behavior['session_volatility'][session_name] = {
                    'avg_range': np.mean(ranges),
                    'median_range': np.median(ranges),
                    'std_range': np.std(ranges),
                    'percentile_75': np.percentile(ranges, 75),
                    'percentile_90': np.percentile(ranges, 90),
                    'avg_volume': np.mean(volumes),
                    'samples': len(ranges)
                }
    
    def measure_swing_sizes(self, df):
        """
        Measure typical swing sizes for this pair
        """
        pip_size = self.get_pip_size()
        swings = self.find_swing_points(df, window=5)
        
        # Pair consecutive swings to measure swing sizes
        for i in range(1, len(swings)):
            prev_swing = swings[i-1]
            curr_swing = swings[i]
            
            if prev_swing['type'] != curr_swing['type']:  # Opposite swings
                swing_size = abs(curr_swing['price'] - prev_swing['price']) / pip_size
                
                self.behavior['typical_swing_sizes'].append({
                    'size_pips': swing_size,
                    'duration_bars': abs(curr_swing['index'] - prev_swing['index']),
                    'start_type': prev_swing['type'],
                    'session': self.get_session(prev_swing['time']),
                    'hour': prev_swing['time'].hour
                })
    
    def measure_spread_patterns(self):
        """
        Measure spread patterns throughout the day
        """
        # This would need tick data, so we'll store a framework for it
        # In live trading, you'd collect this data over time
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            current_spread = symbol_info.spread
            current_hour = datetime.now().hour
            
            if 'spread_by_hour' not in self.behavior:
                self.behavior['spread_by_hour'] = {}
            
            if str(current_hour) not in self.behavior['spread_by_hour']:
                self.behavior['spread_by_hour'][str(current_hour)] = []
            
            self.behavior['spread_by_hour'][str(current_hour)].append(current_spread)
    
    def measure_momentum_patterns(self, df_m5):
        """
        Measure how momentum develops before successful moves
        """
        pip_size = self.get_pip_size()
        
        for i in range(20, len(df_m5) - 20):
            # Look for strong moves
            future_move = df_m5.iloc[i:i+20]
            move_size = (future_move['close'].iloc[-1] - future_move['close'].iloc[0]) / pip_size
            
            if abs(move_size) > 15:  # Significant move
                # Analyze momentum leading up to it
                lookback = df_m5.iloc[i-20:i]
                
                momentum_data = {
                    'move_size': move_size,
                    'direction': 'up' if move_size > 0 else 'down',
                    'pre_move_trend': self.calculate_momentum_score(lookback),
                    'consolidation_bars': self.count_consolidation_bars(lookback),
                    'session': self.get_session(df_m5.iloc[i]['time']),
                    'hour': df_m5.iloc[i]['time'].hour
                }
                
                self.behavior['momentum_patterns'].append(momentum_data)
    
    # Helper methods
    
    def is_trending_up(self, window):
        """Check if price window shows uptrend"""
        sma_fast = window['close'].rolling(5).mean()
        sma_slow = window['close'].rolling(10).mean()
        
        if len(sma_fast.dropna()) < 5 or len(sma_slow.dropna()) < 5:
            return False
        
        # Price making higher highs and higher lows
        highs = window['high'].values
        lows = window['low'].values
        
        higher_highs = sum(highs[i] > highs[i-1] for i in range(1, len(highs))) > len(highs) * 0.5
        higher_lows = sum(lows[i] > lows[i-1] for i in range(1, len(lows))) > len(lows) * 0.4
        
        return higher_highs and higher_lows and sma_fast.iloc[-1] > sma_slow.iloc[-1]
    
    def is_trending_down(self, window):
        """Check if price window shows downtrend"""
        sma_fast = window['close'].rolling(5).mean()
        sma_slow = window['close'].rolling(10).mean()
        
        if len(sma_fast.dropna()) < 5 or len(sma_slow.dropna()) < 5:
            return False
        
        highs = window['high'].values
        lows = window['low'].values
        
        lower_highs = sum(highs[i] < highs[i-1] for i in range(1, len(highs))) > len(highs) * 0.5
        lower_lows = sum(lows[i] < lows[i-1] for i in range(1, len(lows))) > len(lows) * 0.4
        
        return lower_highs and lower_lows and sma_fast.iloc[-1] < sma_slow.iloc[-1]
    
    def find_swing_points(self, df, window=5):
        """Find swing highs and lows"""
        swings = []
        
        for i in range(window, len(df) - window):
            # Check for swing high
            if df.iloc[i]['high'] == df.iloc[i-window:i+window+1]['high'].max():
                swings.append({
                    'index': i,
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['high'],
                    'type': 'high'
                })
            
            # Check for swing low
            if df.iloc[i]['low'] == df.iloc[i-window:i+window+1]['low'].min():
                swings.append({
                    'index': i,
                    'time': df.iloc[i]['time'],
                    'price': df.iloc[i]['low'],
                    'type': 'low'
                })
        
        return swings
    
    def calculate_momentum_score(self, window):
        """Calculate momentum score for a price window"""
        if len(window) < 2:
            return 0
        
        closes = window['close'].values
        
        # Count consecutive moves in same direction
        ups = sum(closes[i] > closes[i-1] for i in range(1, len(closes)))
        downs = sum(closes[i] < closes[i-1] for i in range(1, len(closes)))
        
        # Directional score
        if ups > downs:
            return ups / len(closes)
        else:
            return -downs / len(closes)
    
    def count_consolidation_bars(self, window):
        """Count bars in consolidation before move"""
        if len(window) < 5:
            return 0
        
        pip_size = self.get_pip_size()
        avg_range = window['high'].values - window['low'].values
        avg_range = np.mean(avg_range)
        
        # Count bars with below-average range
        consolidation = 0
        for i in range(len(window)):
            bar_range = window.iloc[i]['high'] - window.iloc[i]['low']
            if bar_range < avg_range * 0.7:
                consolidation += 1
        
        return consolidation
    
    def get_session(self, time):
        """Determine market session"""
        hour = time.hour
        if 22 <= hour or hour < 7:
            return 'Asian'
        elif 7 <= hour < 14:
            return 'London'
        elif 14 <= hour < 22:
            return 'NewYork'
    
    def get_pip_size(self):
        """Get pip size for the symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                return symbol_info.point * 10
            else:
                return symbol_info.point
        return 0.0001
    
    def save_behavior(self):
        """Save learned behavior to file"""
        # Keep only recent data to prevent file bloat
        max_records = 1000
        
        for key in ['pullback_depths', 'stop_hunt_distances', 'successful_break_characteristics', 
                    'failed_break_characteristics', 'typical_swing_sizes', 'momentum_patterns']:
            if key in self.behavior and len(self.behavior[key]) > max_records:
                self.behavior[key] = self.behavior[key][-max_records:]
        
        # Clean up spread data
        if 'spread_by_hour' in self.behavior:
            for hour in self.behavior['spread_by_hour']:
                if len(self.behavior['spread_by_hour'][hour]) > 100:
                    self.behavior['spread_by_hour'][hour] = self.behavior['spread_by_hour'][hour][-100:]
        
        # Add a timestamp to record when this analysis was run
        self.behavior['last_analysis_timestamp'] = datetime.now().isoformat()
        
        with open(self.behavior_file, 'w') as f:
            json.dump(self.behavior, f, default=str, indent=2)
    
    def get_optimal_parameters(self):
        """
        Based on learned behavior, return optimal parameters for this pair
        """
        if not self.behavior.get('pullback_depths'):
            return None  # Need to analyze first
        
        params = {}
        
        # Pullback depths
        if self.behavior['pullback_depths']:
            pullback_data = self.behavior['pullback_depths']
    
            # Sample size validation
            if len(pullback_data) < 20:
                logging.warning(f"[{self.symbol}] Insufficient pullback samples ({len(pullback_data)}), using conservative defaults")
                params['typical_pullback_depths'] = {
                    '20th': 0.30,
                    '35th': 0.40,
                    '50th': 0.50,
                    '65th': 0.618,
                    '80th': 0.786
                }
            else:
                pullback_depths = [p['depth_ratio'] for p in pullback_data]
                
                # Apply minimum thresholds to prevent shallow pullbacks
                params['typical_pullback_depths'] = {
                    '20th': max(np.percentile(pullback_depths, 20), 0.25),  # Never below 25%
                    '35th': max(np.percentile(pullback_depths, 35), 0.35),  # Never below 35%
                    '50th': max(np.percentile(pullback_depths, 50), 0.45),  # Never below 45%
                    '65th': max(np.percentile(pullback_depths, 65), 0.60),  # Never below 60%
                    '80th': max(np.percentile(pullback_depths, 80), 0.75)   # Never below 75%
                }
                
                logging.info(f"[{self.symbol}] Pullback thresholds: 20th={params['typical_pullback_depths']['20th']:.1%}")
        
        # Stop hunt distances
        if self.behavior['stop_hunt_distances']:
            stop_hunts = [s['distance_pips'] for s in self.behavior['stop_hunt_distances']]
            params['stop_buffer_pips'] = {
                'minimal': np.percentile(stop_hunts, 70),
                'normal': np.percentile(stop_hunts, 85),
                'safe': np.percentile(stop_hunts, 95),
                'max_observed': max(stop_hunts)
            }
        
        # Break characteristics
        params['break_success_factors'] = self.analyze_break_success_factors()
        
        # Session volatility
        params['session_volatility'] = self.behavior.get('session_volatility', {})
        
        # Best trading hours
        params['best_hours'] = self.get_best_trading_hours()
        
        # Typical swing sizes
        if self.behavior['typical_swing_sizes']:
            swing_sizes = [s['size_pips'] for s in self.behavior['typical_swing_sizes']]
            params['typical_swing_size'] = {
                'small': np.percentile(swing_sizes, 25),
                'medium': np.percentile(swing_sizes, 50),
                'large': np.percentile(swing_sizes, 75)
            }
        
        # Momentum patterns
        if self.behavior['momentum_patterns']:
            params['momentum_characteristics'] = self.analyze_momentum_patterns()
        
        # Spread patterns
        if 'spread_by_hour' in self.behavior:
            params['typical_spreads_by_hour'] = {}
            for hour, spreads in self.behavior['spread_by_hour'].items():
                if spreads:
                    params['typical_spreads_by_hour'][hour] = {
                        'median': np.median(spreads),
                        '75th': np.percentile(spreads, 75),
                        '90th': np.percentile(spreads, 90)
                    }
        
        return params
    
    def analyze_break_success_factors(self):
        """
        Determine what factors actually predict successful breaks
        """
        if len(self.behavior.get('successful_break_characteristics', [])) < 5:
            return None
        
        successful = pd.DataFrame(self.behavior['successful_break_characteristics'])
        failed = pd.DataFrame(self.behavior.get('failed_break_characteristics', []))
        
        if len(failed) < 5:
            return None
        
        factors = {}
        
        # Analyze each factor
        numeric_factors = ['break_candle_range', 'break_candle_body', 'volume_spike', 
                          'pre_break_momentum', 'distance_from_level']
        
        for factor in numeric_factors:
            if factor in successful.columns and factor in failed.columns:
                # Statistical test to see if there's a significant difference
                success_values = successful[factor].dropna()
                fail_values = failed[factor].dropna()
                
                if len(success_values) > 0 and len(fail_values) > 0:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(success_values, fail_values)
                    
                    factors[factor] = {
                        'success_mean': success_values.mean(),
                        'success_std': success_values.std(),
                        'fail_mean': fail_values.mean(),
                        'fail_std': fail_values.std(),
                        'significant': p_value < 0.05,
                        'success_threshold': success_values.quantile(0.25)  # Bottom 25% of successful
                    }
        
        # Success rate by session
        if 'session' in successful.columns:
            total_by_session = pd.concat([successful, failed])['session'].value_counts()
            success_by_session = successful['session'].value_counts()
            
            factors['session_success_rate'] = {}
            for session in total_by_session.index:
                success_count = success_by_session.get(session, 0)
                total_count = total_by_session.get(session, 0)
                if total_count > 0:
                    factors['session_success_rate'][session] = success_count / total_count
        
        # Success rate by hour
        if 'time_of_day' in successful.columns:
            factors['hourly_success_rate'] = {}
            for hour in range(24):
                total_hour = len(successful[successful['time_of_day'] == hour]) + len(failed[failed['time_of_day'] == hour])
                success_hour = len(successful[successful['time_of_day'] == hour])
                if total_hour >= 3:  # Minimum sample size
                    factors['hourly_success_rate'][hour] = success_hour / total_hour
        
        return factors
    
    def analyze_momentum_patterns(self):
        """
        Analyze what momentum patterns precede big moves
        """
        if not self.behavior.get('momentum_patterns'):
            return None
        
        momentum_df = pd.DataFrame(self.behavior['momentum_patterns'])
        
        # Separate bullish and bearish moves
        bullish = momentum_df[momentum_df['direction'] == 'up']
        bearish = momentum_df[momentum_df['direction'] == 'down']
        
        patterns = {}
        
        if len(bullish) > 5:
            patterns['bullish'] = {
                'avg_pre_move_momentum': bullish['pre_move_trend'].mean(),
                'avg_consolidation_bars': bullish['consolidation_bars'].mean(),
                'min_momentum_for_big_move': bullish['pre_move_trend'].quantile(0.25)
            }
        
        if len(bearish) > 5:
            patterns['bearish'] = {
                'avg_pre_move_momentum': bearish['pre_move_trend'].mean(),
                'avg_consolidation_bars': bearish['consolidation_bars'].mean(),
                'max_momentum_for_big_move': bearish['pre_move_trend'].quantile(0.75)
            }
        
        return patterns
    
    def get_best_trading_hours(self):
        """
        Determine when this pair actually trends well based on break success
        """
        success_by_hour = {}
        
        # Analyze successful breaks by hour
        for break_data in self.behavior.get('successful_break_characteristics', []):
            hour = break_data.get('time_of_day', -1)
            if hour >= 0:
                if hour not in success_by_hour:
                    success_by_hour[hour] = {'success': 0, 'total': 0}
                success_by_hour[hour]['success'] += 1
                success_by_hour[hour]['total'] += 1
        
        # Analyze failed breaks by hour
        for break_data in self.behavior.get('failed_break_characteristics', []):
            hour = break_data.get('time_of_day', -1)
            if hour >= 0:
                if hour not in success_by_hour:
                    success_by_hour[hour] = {'success': 0, 'total': 0}
                success_by_hour[hour]['total'] += 1
        
        # Find hours with good success rate AND enough samples
        good_hours = []
        for hour, stats in success_by_hour.items():
            if stats['total'] >= 5:  # Minimum sample size
                success_rate = stats['success'] / stats['total']
                if success_rate >= 0.6:  # 60% or better success rate
                    good_hours.append(hour)
        
        return sorted(good_hours)