import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone, timedelta
from talib import ATR, EMA, RSI, ADX, PLUS_DI, MINUS_DI
from collections import deque

from market_regime import MarketRegimeDetector
from ftx_market_context import MarketContext

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AMDAnalyzer:
    
    def __init__(self, market_data, market_context):
        self.market_data = market_data
        self.pip_size = market_data.get_pip_size()
        self.asian_range_cache = {}
        self.market_context = market_context
        
    def get_session_range(self, df, session_name, current_day):
        """Calculate high/low range for a given session"""
        
        if session_name == 'ASIAN_SESSION':
            # Asian session: 22:00 previous day to 09:00 current day
            start = datetime.combine(current_day - timedelta(days=1), 
                                datetime.min.time()).replace(hour=22, tzinfo=timezone.utc)
            end = datetime.combine(current_day, 
                            datetime.min.time()).replace(hour=9, tzinfo=timezone.utc)
            
            # Special handling for incomplete current day data
            if current_day == df.index[-1].date():
                current_time = datetime.now(timezone.utc)
                
                # If Asian session is still ongoing, we have incomplete data
                if current_time < end:
                    logging.debug(f"Asian session still ongoing until {end}, current time {current_time}")
                    return None  # Wait for complete session data
                                
        elif session_name in ['LONDON_SESSION']:
            start = datetime.combine(current_day, datetime.min.time()).replace(hour=9, tzinfo=timezone.utc)
            end = datetime.combine(current_day, datetime.min.time()).replace(hour=13, tzinfo=timezone.utc)
            
        elif session_name in ['NY_SESSION']:
            start = datetime.combine(current_day, datetime.min.time()).replace(hour=13, tzinfo=timezone.utc)
            end = datetime.combine(current_day, datetime.min.time()).replace(hour=22, tzinfo=timezone.utc)
        else:
            return None
            
        # Filter the dataframe for the session
        session_data = df[(df.index >= start) & (df.index < end)]
        
        if session_data.empty:
            return None
            
        return {
            'high': session_data['high'].max(),
            'low': session_data['low'].min(),
            'range': session_data['high'].max() - session_data['low'].min()
        }
    
    def identify_amd_context(self, df, current_time, current_session, market_structure):
        today = current_time.date()
        
        # Get AMD session grouping
        amd_session_info = self.market_context.get_amd_session(current_time)
        amd_session = amd_session_info['name']
        
        # Get Asian range
        asian_range = self.asian_range_cache.get(today)
        if not asian_range and current_time.hour >= 9:  # After Asian session ends
            asian_range = self.get_session_range(df, 'ASIAN_SESSION', today)
            if asian_range:
                self.asian_range_cache[today] = asian_range
        
        amd_context = {
            'phase': None,
            'asian_range': asian_range,
            'setup_bias': None,
            'key_levels': []
        }
        
        current_price = df['close'].iloc[-1]
        
        # ACCUMULATION PHASE (Asian Session: 22:00-09:00 UTC)
        if amd_session == 'ASIAN_SESSION':
            amd_context['phase'] = 'ACCUMULATION'
            if market_structure['regime'] == 'RANGING':
                amd_context['setup_bias'] = None  # No directional bias
            else:
                # Even in trending markets, Asian often consolidates
                amd_context['setup_bias'] = 'avoid_momentum'
                
        # MANIPULATION PHASE (London Session: 09:00-13:00 UTC)
        elif amd_session == 'LONDON_SESSION':
            if asian_range:
                if current_price > asian_range['high'] or current_price < asian_range['low']:
                    recent_bars = df.iloc[-10:]
                    # Check for false breakout (manipulation)
                    if current_price > asian_range['high'] and any(recent_bars['close'] < asian_range['high']):
                        amd_context['phase'] = 'MANIPULATION_CONFIRMED'
                        amd_context['setup_bias'] = 'sell'
                        amd_context['key_levels'].append(asian_range['low'])
                    elif current_price < asian_range['low'] and any(recent_bars['close'] > asian_range['low']):
                        amd_context['phase'] = 'MANIPULATION_CONFIRMED'
                        amd_context['setup_bias'] = 'buy'
                        amd_context['key_levels'].append(asian_range['high'])
                else:
                    amd_context['phase'] = 'AWAITING_MANIPULATION'
                    
        # DISTRIBUTION PHASE (NY Session: 13:00-22:00 UTC)
        elif amd_session == 'NY_SESSION':
            amd_context['phase'] = 'DISTRIBUTION'
            if market_structure['regime'] in ['STRONG_TREND', 'NORMAL_TREND']:
                # Convert trend direction to buy/sell
                trend_dir = market_structure['trend']['direction']
                if trend_dir == 'buy':
                    amd_context['setup_bias'] = 'buy'
                elif trend_dir == 'sell':
                    amd_context['setup_bias'] = 'sell'
                else:
                    amd_context['setup_bias'] = None
            elif asian_range:
                # Check if we've extended beyond Asian range
                if current_price > asian_range['high'] + (asian_range['range'] * 0.5):
                    amd_context['setup_bias'] = 'buy'
                elif current_price < asian_range['low'] - (asian_range['range'] * 0.5):
                    amd_context['setup_bias'] = 'sell'
                    
        return amd_context

class PriceActionZone:
    """Manages price zones instead of exact levels"""
    
    def __init__(self, center_price, atr, zone_type='fibonacci'):
        self.center = center_price
        self.zone_type = zone_type
        
        # Dynamic zone size based on ATR
        if zone_type == 'fibonacci':
            self.upper = center_price + (atr * 0.15)  # 15% of ATR
            self.lower = center_price - (atr * 0.15)
        elif zone_type == 'structure':
            self.upper = center_price + (atr * 0.2)
            self.lower = center_price - (atr * 0.2)
            
    def contains_price(self, price):
        """Check if price is within zone"""
        return self.lower <= price <= self.upper
        
    def touched_recently(self, df, lookback=10):
        """Check if zone was touched in recent bars"""
        recent_bars = df.iloc[-lookback:]
        for _, bar in recent_bars.iterrows():
            if bar['low'] <= self.upper and bar['high'] >= self.lower:
                return True
        return False
        
    def get_rejection_strength(self, df, lookback=5):
        """Calculate how strongly price rejected from this zone"""
        rejection_score = 0
        recent_bars = df.iloc[-lookback:]
        
        for idx, (_, bar) in enumerate(recent_bars.iterrows()):
            # Check for bullish rejection (lower wick)
            if bar['low'] <= self.upper and bar['close'] > self.center:
                wick_size = min(bar['open'], bar['close']) - bar['low']
                body_size = abs(bar['close'] - bar['open'])
                if body_size > 1e-10:
                    rejection_score += min(wick_size / body_size, 3.0) * ((idx + 1) / lookback)
                    
            # Check for bearish rejection (upper wick)
            elif bar['high'] >= self.lower and bar['close'] < self.center:
                wick_size = bar['high'] - max(bar['open'], bar['close'])
                body_size = abs(bar['close'] - bar['open'])
                if body_size > 1e-10:
                    rejection_score += min(wick_size / body_size, 3.0) * ((idx + 1) / lookback)
                    
        return rejection_score

class AdaptiveTradeManager:
    """Enhanced trade manager that adapts to market conditions"""
    
    def __init__(self, client, market_data, market_context=None):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.order_manager = OrderManager(client, market_data)
        self.indicator_calc = IndicatorCalculator()
        self.market_context = market_context or MarketContext()
        self.regime_detector = MarketRegimeDetector()
        
        # --- NEW: Instantiate AMDAnalyzer once as a component ---
        self.amd_analyzer = AMDAnalyzer(self.market_data, self.market_context)
        
        self.setup_history = deque(maxlen=100)
        self.last_trade_time = {}
        
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] 
        self.tf_lower = min(self.timeframes)
        
    # --- Analysis Pipeline ---
    def analyze_market_structure(self, data):
        """
        Comprehensive market structure analysis including AMD context.
        Returns a single, complete structure dictionary.
        """
        try:
            df = data.copy()
            
            # --- STEP 1: Calculate base indicators ---
            df = self.indicator_calc.calculate_indicators(df)
            if df is None or 'atr' not in df.columns or pd.isna(df['atr'].iloc[-1]):
                return None
            
            # --- STEP 2: Get basic metrics and classify regime/trend ---
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            adx = df['adx'].iloc[-1]
            
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            range_20 = high_20 - low_20
            price_position = (current_price - low_20) / range_20 if range_20 > 0 else 0.5
            
            atr_ma = df['atr'].rolling(20).mean().iloc[-1]
            atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0
            
            regime, params = self.regime_detector.classify_market(adx, atr_ratio, price_position)
            structure_levels = self.find_key_levels(df)
            trend_info = self.analyze_trend_comprehensive(df)
            
            # --- STEP 3: Build the initial structure dictionary ---
            structure = {
                'regime': regime,
                'regime_params': params,
                'trend': trend_info,
                'structure_levels': structure_levels,
                'current_price': current_price,
                'atr': atr,
                'adx': adx,
                'price_position': price_position
            }
            
            # --- STEP 4: Get session and AMD context ---
            # Use the last candle's timestamp for accurate analysis
            current_time = df.index[-1]
            session_info = self.market_context.get_trading_session(current_time)
            amd_context = self.amd_analyzer.identify_amd_context(df, current_time, session_info, structure)
            
            # --- STEP 5: Enhance the structure dict with session and AMD info ---
            structure['session'] = session_info
            structure['amd'] = amd_context
            
            # Add AMD-driven trading biases
            if amd_context['phase'] == 'ACCUMULATION':
                structure['trading_approach'] = 'range_bound'
                structure['avoid_setups'] = ['momentum', 'breakout']
            elif amd_context['phase'] == 'MANIPULATION_CONFIRMED':
                structure['trading_approach'] = 'reversal'
                structure['preferred_direction'] = amd_context['setup_bias']
            elif amd_context['phase'] == 'DISTRIBUTION':
                structure['trading_approach'] = 'trend_following'
                if amd_context['setup_bias']:
                    structure['preferred_direction'] = amd_context['setup_bias']
            
            return structure
            
        except Exception as e:
            logging.error(f"Error in analyze_market_structure: {e}")
            traceback.print_exc()
            return None
            
    def analyze_trend_comprehensive(self, df):
        """Enhanced trend analysis"""
        try:
            # Basic trend via EMAs
            ema_fast = df['ema_fast'].iloc[-1]
            ema_slow = df['ema_slow'].iloc[-1]
            price = df['close'].iloc[-1]
            
            # Trend direction
            if price > ema_fast > ema_slow:
                direction = 'buy'
                strength_base = 1.0
            elif price < ema_fast < ema_slow:
                direction = 'sell'
                strength_base = 1.0
            else:
                direction = 'unclear'
                strength_base = 0.3
                
            # Calculate pullback depth if in trend
            if direction != 'unclear':
                # Find recent extreme
                if direction == 'buy':
                    recent_high = df['high'].iloc[-20:].max()
                    recent_low = df['low'].iloc[-20:].min()
                    pullback_depth = (recent_high - price) / (recent_high - recent_low) if recent_high > recent_low else 0
                else:
                    recent_high = df['high'].iloc[-20:].max()
                    recent_low = df['low'].iloc[-20:].min()
                    pullback_depth = (price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0
            else:
                pullback_depth = 0
                
            # Momentum analysis
            rsi = df['rsi'].iloc[-1]
            rsi_ma = df['rsi'].rolling(10).mean().iloc[-1]
            momentum_aligned = (direction == 'buy' and rsi > 50) or (direction == 'sell' and rsi < 50)
            
            # Final strength calculation
            strength = strength_base
            if momentum_aligned:
                strength *= 1.1
            if 0.2 < pullback_depth < 0.7:  # Healthy pullback
                strength *= 1.1
                
            return {
                'direction': direction,
                'strength': min(strength, 1.0),
                'pullback_depth': pullback_depth,
                'momentum_aligned': momentum_aligned,
                'ema_fast': ema_fast,
                'ema_slow': ema_slow
            }
            
        except Exception as e:
            logging.error(f"Error in analyze_trend_comprehensive: {e}")
            return {'direction': 'unclear', 'strength': 0}
            
    def find_key_levels(self, df, lookback=100):
        """Find all key price levels including Fib, S/R, and psychological"""
        levels = []
        pip_size = self.market_data.get_pip_size()
        
        # 1. Recent swing highs/lows
        swing_highs, swing_lows = self.find_swing_points(df, window=10)
        
        for _, high in swing_highs.iterrows():
            levels.append({
                'price': high['price'],
                'type': 'resistance',
                'strength': 0.8
            })
            
        for _, low in swing_lows.iterrows():
            levels.append({
                'price': low['price'],
                'type': 'support',
                'strength': 0.8
            })
            
        # 2. Round numbers (psychological)
        current_price = df['close'].iloc[-1]
        if 'JPY' in self.market_data.symbol:
            round_interval = 0.50  # 50 pips for JPY
        else:
            round_interval = 0.0050  # 50 pips for non-JPY
        
        for i in range(-5, 6):
            round_level = round(current_price / round_interval) * round_interval + (i * round_interval)
            if abs(round_level - current_price) < 200 * pip_size:  # Within 200 pips
                levels.append({
                    'price': round_level,
                    'type': 'psychological',
                    'strength': 0.6
                })
                
        # 3. Moving averages as dynamic levels
        if len(df) >= 50:
            ma50 = df['close'].rolling(50).mean().iloc[-1]
            levels.append({
                'price': ma50,
                'type': 'dynamic_support' if current_price > ma50 else 'dynamic_resistance',
                'strength': 0.7
            })
            
        return levels
        
    def find_entry_setups(self, market_structure, timeframe_data):
        """
        Find entry setups based on market regime
        Returns list of potential setups
        """
        setups = []
        regime = market_structure['regime']
        params = market_structure['regime_params']
        
        if regime in ['TREND_EXHAUSTION', 'VOLATILE_EXPANSION']:
            logging.info(f"Blocking all setups due to {regime} regime")
            return []  # Return empty list, no setups allowed
        
        # NEW: Check if we should avoid certain setups
        avoid_setups = market_structure.get('avoid_setups', [])
        
        # Get appropriate entry methods for current regime
        entry_methods = params['entry_methods']
        
        # Filter out avoided setups
        entry_methods = [m for m in entry_methods if m not in avoid_setups]
                
        # 1. Check Fibonacci setups (if applicable)
        if 'fibonacci' in entry_methods:
            fib_setups = self.find_fibonacci_setups(
                timeframe_data, 
                market_structure,
                params['fib_levels']
            )
            setups.extend(fib_setups)
            
        # 2. Check momentum setups (for strong trends)
        if 'momentum' in entry_methods:
            momentum_setups = self.find_momentum_setups(
                timeframe_data,
                market_structure
            )
            setups.extend(momentum_setups)
            
        # 3. Check MA bounce setups
        if 'ma_bounce' in entry_methods:
            ma_setups = self.find_ma_bounce_setups(
                timeframe_data,
                market_structure
            )
            setups.extend(ma_setups)
            
        # 4. Check range extreme setups (for ranging markets)
        if 'range_extreme' in entry_methods:
            range_setups = self.find_range_extreme_setups(
                timeframe_data,
                market_structure
            )
            setups.extend(range_setups)
            
        # 5. NEW: Check shallow pullback setups (for strong trends)
        if 'shallow_pullback' in entry_methods:
            shallow_setups = self.find_shallow_pullback_setups(
                timeframe_data,
                market_structure
            )
            setups.extend(shallow_setups)
        
        # 6. NEW: Check structure break setups
        if 'structure_break' in entry_methods:
            break_setups = self.find_structure_break_setups(
                timeframe_data,
                market_structure
            )
            setups.extend(break_setups)
            
        # Score and filter setups
        scored_setups = self.score_setups(setups, market_structure, timeframe_data)
        
        # Apply preferred direction filter
        preferred_dir = market_structure.get('preferred_direction')
        if preferred_dir:
            # Filter setups to only preferred direction
            scored_setups = [s for s in scored_setups if s['direction'] == preferred_dir]
        
        # Return only high-quality setups
        min_score = params['confluence_required']
        return [s for s in scored_setups if s['score'] >= min_score]
        
    def find_fibonacci_setups(self, df, market_structure, fib_levels):
        """Enhanced Fibonacci setup detection using zones"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['buy', 'sell']:
            return setups
            
        # Get recent swing points
        swing_highs, swing_lows = self.find_swing_points(df, window=15)
        if swing_highs.empty or swing_lows.empty:
            return setups
            
        # Determine impulse move
        if trend == 'buy':
            last_high = swing_highs.iloc[-1]
            # Find swing lows that occurred BEFORE this high
            valid_lows = swing_lows[swing_lows.index < swing_highs.index[-1]]
            if valid_lows.empty:
                return setups
            impulse_start = valid_lows.iloc[-1]['price']
            impulse_end = last_high['price']
        else:
            last_low = swing_lows.iloc[-1]
            # Find swing highs that occurred BEFORE this low
            valid_highs = swing_highs[swing_highs.index < swing_lows.index[-1]]
            if valid_highs.empty:
                return setups
            impulse_start = valid_highs.iloc[-1]['price']
            impulse_end = last_low['price']
            
        swing_range = abs(impulse_end - impulse_start)
        atr = market_structure['atr']
        current_price = df['close'].iloc[-1]
        
        # Check each Fib level
        for ratio in fib_levels:
            if trend == 'buy':
                fib_price = impulse_end - (swing_range * ratio)
            else:
                fib_price = impulse_start + (swing_range * ratio)
                
            # Create zone instead of exact level
            zone = PriceActionZone(fib_price, atr, 'fibonacci')
            
            # Check if we're in or near the zone
            if zone.contains_price(current_price) or zone.touched_recently(df, lookback=10):
                rejection_strength = zone.get_rejection_strength(df)
                
                if rejection_strength > 0.5:  # Some rejection detected
                    setup = {
                        'type': 'fibonacci',
                        'level': f'{int(ratio*100)}%',
                        'zone': zone,
                        'direction': 'buy' if trend == 'buy' else 'sell',
                        'entry_price': current_price,
                        'rejection_strength': rejection_strength,
                        'base_score': 0.7
                    }
                    setups.append(setup)
                    
        return setups
        
    def find_momentum_setups(self, df, market_structure):
        """Find momentum continuation setups for strong trends (Corrected)"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['buy', 'sell'] or market_structure['regime'] != 'STRONG_TREND':
            return setups
            
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        
        # Look for shallow pullback to EMA
        ema_fast = market_structure['trend']['ema_fast']
        zone = PriceActionZone(ema_fast, atr, 'structure')
        
        # Check if price is currently touching the dynamic zone
        if zone.contains_price(current_price):
            # Check the momentum on the two most recent CLOSED candles.
            last_closed_bar = df.iloc[-2]
            prev_closed_bar = df.iloc[-3]
            momentum_confirmed = False
            
            if trend == 'buy':
                # Bullish momentum: The last two closed candles made a higher low,
                # and the most recent closed candle was bullish.
                if (last_closed_bar['low'] > prev_closed_bar['low'] and 
                    last_closed_bar['close'] > last_closed_bar['open']):
                    momentum_confirmed = True
            else: # sell
                # Bearish momentum: The last two closed candles made a lower high,
                # and the most recent closed candle was bearish.
                if (last_closed_bar['high'] < prev_closed_bar['high'] and
                    last_closed_bar['close'] < last_closed_bar['open']):
                    momentum_confirmed = True
                    
            if momentum_confirmed:
                setup = {
                    'type': 'momentum',
                    'zone': zone,
                    'direction': 'buy' if trend == 'buy' else 'sell',
                    'entry_price': current_price,
                    'base_score': 0.8
                }
                setups.append(setup)
                
        return setups
        
    def find_ma_bounce_setups(self, df, market_structure):
        """
        Finds robust moving average bounce setups based on recent price action.
        This pattern looks for a test of the MA followed by a confirmation of rejection.
        """
        setups = []
        trend = market_structure['trend']['direction']
        if trend not in ['buy', 'sell']:
            return setups

        df_copy = df.copy()
        current_price = df_copy['close'].iloc[-1]
        atr = market_structure['atr']
        ma_periods = [20, 50]  # Check both 20 and 50 MAs

        for period in ma_periods:
            if len(df_copy) < period + 10:  # Ensure enough data for MA and pattern
                continue

            ma_col_name = f'ma_{period}'
            df_copy[ma_col_name] = df_copy['close'].rolling(window=period).mean()

            if pd.isna(df_copy[ma_col_name].iloc[-1]):
                continue

            # We look back over the last ~5 bars to find the bounce structure.
            lookback_window = 5
            recent_bars = df_copy.iloc[-lookback_window:]
            ma_on_last_bar = recent_bars[ma_col_name].iloc[-1]

            # Find the absolute low/high point of the recent test
            test_low_point = recent_bars['low'].min()
            test_high_point = recent_bars['high'].max()

            bounce_confirmed = False
            setup_direction = ''
            structural_level_for_sl = None

            # buy: Looking for a bounce off the MA from above
            if trend == 'buy':
                # 1. Price must have touched or dipped slightly below the MA recently.
                is_test = any(recent_bars['low'] <= recent_bars[ma_col_name])
                # 2. The most recent CLOSED candle must show rejection (closed bullishly above the MA).
                confirmation_candle = df_copy.iloc[-2]
                is_confirmation = (confirmation_candle['close'] > confirmation_candle['open'] and
                                confirmation_candle['close'] > confirmation_candle[ma_col_name])
                # 3. The MA itself should be sloping upwards.
                ma_slope = df_copy[ma_col_name].iloc[-1] - df_copy[ma_col_name].iloc[-lookback_window]
                is_ma_sloping_up = ma_slope > 0

                if is_test and is_confirmation and is_ma_sloping_up:
                    # 4. Don't enter if the price has already run too far from the MA.
                    if abs(current_price - ma_on_last_bar) < atr * 1.5:
                        bounce_confirmed = True
                        setup_direction = 'buy'
                        structural_level_for_sl = test_low_point  # The low of the bounce pattern

            # sell: Looking for a bounce off the MA from below
            elif trend == 'sell':
                # 1. Price must have touched or spiked slightly above the MA recently.
                is_test = any(recent_bars['high'] >= recent_bars[ma_col_name])
                # 2. The most recent CLOSED candle must show rejection (closed bearishly below the MA).
                confirmation_candle = df_copy.iloc[-2]
                is_confirmation = (confirmation_candle['close'] < confirmation_candle['open'] and
                                confirmation_candle['close'] < confirmation_candle[ma_col_name])
                # 3. The MA itself should be sloping downwards.
                ma_slope = df_copy[ma_col_name].iloc[-1] - df_copy[ma_col_name].iloc[-lookback_window]
                is_ma_sloping_down = ma_slope < 0

                if is_test and is_confirmation and is_ma_sloping_down:
                    # 4. Don't enter if the price has already run too far from the MA.
                    if abs(current_price - ma_on_last_bar) < atr * 1.5:
                        bounce_confirmed = True
                        setup_direction = 'sell'
                        structural_level_for_sl = test_high_point  # The high of the bounce pattern

            if bounce_confirmed:
                zone = PriceActionZone(ma_on_last_bar, atr, 'structure')
                setup = {
                    'type': 'ma_bounce',
                    'ma_period': period,
                    'zone': zone,
                    'direction': setup_direction,
                    'entry_price': current_price,
                    'stop_loss_level': structural_level_for_sl,
                    'base_score': 0.75  # Higher base score for this more robust pattern
                }
                setups.append(setup)
                # Once a bounce is found on one MA, we don't need to check others
                break
                
        return setups
        
    def find_range_extreme_setups(self, df, market_structure):
        """Find setups at range extremes for ranging markets"""
        setups = []
        
        if market_structure['regime'] != 'RANGING':
            return setups
            
        # Define range
        lookback = 50
        range_high = df['high'].iloc[-lookback:].max()
        range_low = df['low'].iloc[-lookback:].min()
        range_size = range_high - range_low
        
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        
        # Check if at range extremes
        upper_zone = PriceActionZone(range_high - range_size * 0.1, atr, 'structure')
        lower_zone = PriceActionZone(range_low + range_size * 0.1, atr, 'structure')
        
        if upper_zone.contains_price(current_price):
            rejection = upper_zone.get_rejection_strength(df)
            if rejection > 1.0:
                setup = {
                    'type': 'range_extreme',
                    'extreme': 'upper',
                    'zone': upper_zone,
                    'direction': 'sell',
                    'entry_price': current_price,
                    'base_score': 0.8
                }
                setups.append(setup)
                
        elif lower_zone.contains_price(current_price):
            rejection = lower_zone.get_rejection_strength(df)
            if rejection > 1.0:
                setup = {
                    'type': 'range_extreme',
                    'extreme': 'lower',
                    'zone': lower_zone,
                    'direction': 'buy',
                    'entry_price': current_price,
                    'base_score': 0.8
                }
                setups.append(setup)
                
        return setups
    
    def find_shallow_pullback_setups(self, df, market_structure):
        """Find shallow retracements in strong trends (23.6%-38.2%)"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['buy', 'sell'] or market_structure['regime'] != 'STRONG_TREND':
            return setups
        
        # Find recent trend extreme (last 20 bars)
        lookback = 20
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        
        if trend == 'buy':
            # In uptrend, find recent high and low
            recent_high_idx = df['high'].iloc[-lookback:].idxmax()
            recent_high = df['high'].iloc[-lookback:].max()
            
            # Find the low BEFORE this high
            bars_before_high = df[df.index < recent_high_idx].iloc[-lookback:]
            if len(bars_before_high) == 0:
                return setups
                
            recent_low = bars_before_high['low'].min()
            trend_range = recent_high - recent_low
            
            if trend_range < atr * 2:  # Trend too small
                return setups
            
            # Shallow pullback zones (23.6% and 38.2%)
            pullback_236 = recent_high - (trend_range * 0.236)
            pullback_382 = recent_high - (trend_range * 0.382)
            
            # Check if we're in shallow pullback zone
            if pullback_382 <= current_price <= pullback_236:
                # Look for bullish reversal in last 3 closed bars
                last_three_closed = df.iloc[-4:-1]
                
                # Check for reversal pattern
                reversal_found = False
                for i in range(len(last_three_closed)):
                    bar = last_three_closed.iloc[i]
                    # Bullish reversal: close > open and low near/below zone
                    if (bar['close'] > bar['open'] and 
                        bar['low'] <= pullback_236):
                        reversal_found = True
                        break
                
                if reversal_found:
                    # Create zone around the shallow pullback area
                    zone_center = (pullback_236 + pullback_382) / 2
                    zone = PriceActionZone(zone_center, atr * 0.1, 'shallow')
                    
                    setup = {
                        'type': 'shallow_pullback',
                        'pullback_level': '23.6%-38.2%',
                        'zone': zone,
                        'direction': 'buy',
                        'entry_price': current_price,
                        'trend_high': recent_high,
                        'trend_low': recent_low,
                        'base_score': 0.75
                    }
                    setups.append(setup)
                    
        else:  # trend == 'sell'
            # In downtrend, find recent low and high
            recent_low_idx = df['low'].iloc[-lookback:].idxmin()
            recent_low = df['low'].iloc[-lookback:].min()
            
            # Find the high BEFORE this low
            bars_before_low = df[df.index < recent_low_idx].iloc[-lookback:]
            if len(bars_before_low) == 0:
                return setups
                
            recent_high = bars_before_low['high'].max()
            trend_range = recent_high - recent_low
            
            if trend_range < atr * 2:  # Trend too small
                return setups
            
            # Shallow pullback zones (23.6% and 38.2%)
            pullback_236 = recent_low + (trend_range * 0.236)
            pullback_382 = recent_low + (trend_range * 0.382)
            
            # Check if we're in shallow pullback zone
            if pullback_236 <= current_price <= pullback_382:
                # Look for bearish reversal in last 3 closed bars
                last_three_closed = df.iloc[-4:-1]
                
                # Check for reversal pattern
                reversal_found = False
                for i in range(len(last_three_closed)):
                    bar = last_three_closed.iloc[i]
                    # Bearish reversal: close < open and high near/above zone
                    if (bar['close'] < bar['open'] and 
                        bar['high'] >= pullback_236):
                        reversal_found = True
                        break
                
                if reversal_found:
                    # Create zone around the shallow pullback area
                    zone_center = (pullback_236 + pullback_382) / 2
                    zone = PriceActionZone(zone_center, atr * 0.1, 'shallow')
                    
                    setup = {
                        'type': 'shallow_pullback',
                        'pullback_level': '23.6%-38.2%',
                        'zone': zone,
                        'direction': 'sell',
                        'entry_price': current_price,
                        'trend_high': recent_high,
                        'trend_low': recent_low,
                        'base_score': 0.75
                    }
                    setups.append(setup)
        
        return setups

    def find_structure_break_setups(self, df, market_structure):
        """Find breaks of key horizontal levels with confirmation"""
        setups = []
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        
        # Only look for structure breaks in trending markets
        if market_structure['regime'] not in ['NORMAL_TREND', 'STRONG_TREND']:
            return setups
        
        # Get key S/R levels from market structure
        key_levels = [level for level in market_structure['structure_levels'] 
                    if level['type'] in ['resistance', 'support'] and level['strength'] >= 0.7]
        
        if not key_levels:
            return setups
        
        # Look at recent price action (last 20 bars)
        lookback = 20
        recent_bars = df.iloc[-lookback:]
        
        for level in key_levels:
            level_price = level['price']
            
            # Skip levels too far from current price (more than 2 ATR away)
            if abs(current_price - level_price) > atr * 2:
                continue
            
            # RESISTANCE BREAK (Bullish)
            if level['type'] == 'resistance' and current_price > level_price:
                # Check if this is a recent break (within last 10 bars)
                bars_below_level = recent_bars[recent_bars['high'] < level_price]
                
                if len(bars_below_level) >= 5:  # Was below for at least 5 bars
                    # Find when the break occurred
                    break_bar_idx = None
                    for i in range(len(recent_bars) - 1):
                        if (recent_bars.iloc[i]['high'] < level_price and 
                            recent_bars.iloc[i + 1]['close'] > level_price):
                            break_bar_idx = i + 1
                            break
                    
                    if break_bar_idx and break_bar_idx >= len(recent_bars) - 10:
                        # Recent break found, now check for confirmation
                        bars_since_break = recent_bars.iloc[break_bar_idx:]
                        
                        # Confirmation criteria:
                        # 1. Price stayed above level for at least 3 bars
                        # 2. OR had a successful retest (came back to level and bounced)
                        
                        stayed_above = all(bar['low'] > level_price - (atr * 0.1) 
                                        for _, bar in bars_since_break.iterrows())
                        
                        # Check for retest
                        retest_found = False
                        for _, bar in bars_since_break.iterrows():
                            if (bar['low'] <= level_price + (atr * 0.1) and 
                                bar['close'] > level_price):
                                retest_found = True
                                break
                        
                        if stayed_above or retest_found:
                            zone = PriceActionZone(level_price, atr * 0.15, 'structure')
                            
                            setup = {
                                'type': 'structure_break',
                                'break_type': 'resistance_break',
                                'zone': zone,
                                'direction': 'buy',
                                'entry_price': current_price,
                                'level_strength': level['strength'],
                                'retest': retest_found,
                                'base_score': 0.8 if retest_found else 0.7
                            }
                            setups.append(setup)
            
            # SUPPORT BREAK (Bearish)
            elif level['type'] == 'support' and current_price < level_price:
                # Check if this is a recent break (within last 10 bars)
                bars_above_level = recent_bars[recent_bars['low'] > level_price]
                
                if len(bars_above_level) >= 5:  # Was above for at least 5 bars
                    # Find when the break occurred
                    break_bar_idx = None
                    for i in range(len(recent_bars) - 1):
                        if (recent_bars.iloc[i]['low'] > level_price and 
                            recent_bars.iloc[i + 1]['close'] < level_price):
                            break_bar_idx = i + 1
                            break
                    
                    if break_bar_idx and break_bar_idx >= len(recent_bars) - 10:
                        # Recent break found, now check for confirmation
                        bars_since_break = recent_bars.iloc[break_bar_idx:]
                        
                        # Confirmation criteria:
                        # 1. Price stayed below level for at least 3 bars
                        # 2. OR had a successful retest (came back to level and rejected)
                        
                        stayed_below = all(bar['high'] < level_price + (atr * 0.1) 
                                        for _, bar in bars_since_break.iterrows())
                        
                        # Check for retest
                        retest_found = False
                        for _, bar in bars_since_break.iterrows():
                            if (bar['high'] >= level_price - (atr * 0.1) and 
                                bar['close'] < level_price):
                                retest_found = True
                                break
                        
                        if stayed_below or retest_found:
                            zone = PriceActionZone(level_price, atr * 0.15, 'structure')
                            
                            setup = {
                                'type': 'structure_break',
                                'break_type': 'support_break',
                                'zone': zone,
                                'direction': 'sell',
                                'entry_price': current_price,
                                'level_strength': level['strength'],
                                'retest': retest_found,
                                'base_score': 0.8 if retest_found else 0.7
                            }
                            setups.append(setup)
        
        return setups
        
    def score_setups(self, setups, market_structure, df):
        """Score setups based on multiple confluence factors"""
        if not setups:
            return []

        scored_setups = []
        atr = market_structure['atr']
    
        for i, setup in enumerate(setups):
            score = setup['base_score']
            confluence_notes = []

            # === CONFLUENCE CHECK 1: ALIGNMENT OF DIFFERENT SETUP TYPES ===
            # Does this setup's zone overlap with another setup's zone?
            for j, other_setup in enumerate(setups):
                if i == j:  # Don't compare a setup to itself
                    continue
                
                # Check for same direction and that zones are defined
                if setup['direction'] == other_setup['direction'] and 'zone' in setup and 'zone' in other_setup:
                    price_distance = abs(setup['zone'].center - other_setup['zone'].center)
                    
                    # If zones are within 25% of ATR, it's a "Zone of Confluence"
                    if price_distance < (atr * 0.25):
                        score += 0.20  # Significant bonus for direct confluence
                        confluence_notes.append(f"aligns with {other_setup['type']}")

            # === CONFLUENCE CHECK 2: ALIGNMENT WITH KEY STRUCTURE ===
            for level in market_structure['structure_levels']:
                if 'zone' in setup and abs(setup['zone'].center - level['price']) < (atr * 0.2):
                    score += level['strength'] * 0.15 # Bonus weighted by level strength
                    confluence_notes.append(f"level ({level['type']})")
                    
            # === CONFLUENCE CHECK 3: TREND ALIGNMENT ===
            if setup['direction'] == market_structure['trend']['direction']:
                score += 0.15 * market_structure['trend']['strength'] # Bonus weighted by trend strength
                confluence_notes.append("H1_trend")
                
            # === CONFLUENCE CHECK 4: MOMENTUM CONFIRMATION ===
            if market_structure['trend'].get('momentum_aligned'):
                score += 0.1
                confluence_notes.append("momentum")
                
            # === CONFLUENCE CHECK 5: PRICE ACTION CONFIRMATION ===
            # Add a bonus for strong candle rejection within the zone
            if 'zone' in setup:
                rejection_strength = setup['zone'].get_rejection_strength(df, lookback=5)
                if rejection_strength > 1.5:
                    score += 0.1
                    confluence_notes.append(f"rejection ({rejection_strength:.1f})")
                if rejection_strength > 2.5: # Extra bonus for very strong rejection
                    score += 0.1

            # Session quality bonus
            session_info = self.market_context.get_trading_session()
            session = session_info['name']
            volatility_mult = session_info['volatility_multiplier']
            if session in ['LONDON_TOKYO_OVERLAP', 'LONDON_NY_OVERLAP']:
                score += 0.05
                
            setup['score'] = min(score, 1.0)  # Cap score at 1.0
            setup['confluence_notes'] = list(set(confluence_notes)) # Store reasons for the high score
            scored_setups.append(setup)
        
        if 'amd' in market_structure and market_structure['amd']['phase']:
            amd = market_structure['amd']
            
            for setup in scored_setups:
                amd_adjustment = 0
                
                if amd['phase'] == 'ACCUMULATION':
                    if setup['type'] in ['range_extreme', 'fibonacci']:
                        amd_adjustment += 0.15
                    elif setup['type'] == 'momentum':
                        amd_adjustment -= 0.25
                        
                elif amd['phase'] == 'MANIPULATION_CONFIRMED':
                    if amd['setup_bias'] and setup['direction'] == amd['setup_bias']:
                        amd_adjustment += 0.35
                        # Bonus for targeting key AMD level
                        if amd['key_levels'] and 'zone' in setup:
                            for level in amd['key_levels']:
                                if abs(setup['zone'].center - level) < market_structure['atr']:
                                    amd_adjustment += 0.15
                                    break
                    else:
                        amd_adjustment -= 0.40
                        
                elif amd['phase'] == 'DISTRIBUTION':
                    if setup['type'] in ['momentum', 'ma_bounce']:
                        amd_adjustment += 0.20
                    if amd['setup_bias'] and setup['direction'] == amd['setup_bias']:
                        amd_adjustment += 0.25
                    elif amd['setup_bias'] and setup['direction'] != amd['setup_bias']:
                        amd_adjustment -= 0.35
                        
                setup['amd_phase'] = amd['phase']
                setup['amd_adjustment'] = amd_adjustment
                # Apply adjustment to the base score
                setup['score'] = max(0, min(1.0, setup['score'] + amd_adjustment))

        scored_setups.sort(key=lambda x: x['score'], reverse=True)
        return scored_setups
        
    def execute_setup(self, setup, market_structure):
        """Execute the trading setup"""
        try:
            symbol = self.market_data.symbol
            if symbol in self.last_trade_time:
                time_since_last = (datetime.now(timezone.utc) - self.last_trade_time[symbol]).total_seconds()
                if time_since_last < 3600:  # 1 hour cooldown
                    return False, "Too soon after last trade"    
            
            direction = setup['direction']
            
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False, "Failed to get tick"
                
            entry_price = tick.ask if direction == 'buy' else tick.bid
            
            # Calculate stops based on setup type and market regime
            stop_distance, take_profit_distance = self.calculate_dynamic_stops(
                setup, market_structure
            )
            
            if direction == 'buy':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + take_profit_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - take_profit_distance
                
            # Position sizing with regime adjustment
            risk_mult = market_structure['regime_params']['risk_multiplier']
            lot_size = self.calculate_adaptive_position_size(
                stop_distance, 
                market_structure,
                risk_mult
            )
            
            # Place order
            result = self.order_manager.place_order(
                symbol, lot_size, direction, stop_loss, take_profit
            )
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.last_trade_time[symbol] = datetime.now(timezone.utc)
                self.log_trade_details(setup, market_structure, result)
                return True, "Trade executed"
            else:
                return False, f"Order failed: {result.comment if result else 'Unknown'}"
                
        except Exception as e:
            logging.error(f"Error executing setup: {e}")
            return False, str(e)
            
    def calculate_dynamic_stops(self, setup, market_structure):
        """Calculate stops based on setup type and market conditions"""
        atr = market_structure['atr']
        pip_size = self.market_data.get_pip_size()
        tick = mt5.symbol_info_tick(self.market_data.symbol)
        if not tick:
            # Fallback if tick is not available
            return atr * 1.5, atr * 3.0 
        
        entry_price = tick.ask if setup['direction'] == 'buy' else tick.bid
        stop_distance = 0
        
        # Base stop distance
        if setup['type'] == 'fibonacci':
            # Stop beyond the Fibonacci zone
            if setup['direction'] == 'buy':
                stop_distance = setup['zone'].center - setup['zone'].lower + atr * 0.5
            else:  # sell
                stop_distance = setup['zone'].upper - setup['zone'].center + atr * 0.5
        elif setup['type'] == 'momentum':
            # Tighter stop for momentum trades
            stop_distance = atr * 0.8
        elif setup['type'] == 'ma_bounce' and 'stop_loss_level' in setup and setup['stop_loss_level'] is not None:
            if setup['direction'] == 'buy':
                stop_distance = (entry_price - setup['stop_loss_level']) + (atr * 0.20)
            else:  # sell
                stop_distance = (setup['stop_loss_level'] - entry_price) + (atr * 0.20)
        elif setup['type'] == 'range_extreme':
            # Stop outside the range
            stop_distance = atr * 1.2
        elif setup['type'] == 'shallow_pullback':
            # Stop below the trend low for buys, above trend high for sells
            if setup['direction'] == 'buy':
                stop_distance = entry_price - setup['trend_low'] + (atr * 0.2)
            else:
                stop_distance = setup['trend_high'] - entry_price + (atr * 0.2)
                
        elif setup['type'] == 'structure_break':
            # Stop beyond the broken level
            if setup['direction'] == 'buy':
                stop_distance = entry_price - setup['zone'].lower + (atr * 0.3)
            else:
                stop_distance = setup['zone'].upper - entry_price + (atr * 0.3)
        else:
            stop_distance = atr * 1.2
            
        # Adjust for market regime
        if market_structure['regime'] == 'STRONG_TREND':
            stop_distance *= 0.8  # Tighter stops in strong trends
        elif market_structure['regime'] == 'VOLATILE':
            stop_distance *= 1.3  # Wider stops in volatile markets
            
        # Calculate take profit based on R:R and market conditions
        if market_structure['regime'] == 'STRONG_TREND':
            rr_ratio = 2.5  # Higher R:R in strong trends
        elif market_structure['regime'] == 'RANGING':
            rr_ratio = 1.5  # Lower R:R in ranges
        else:
            rr_ratio = 2.0
            
        take_profit_distance = stop_distance * rr_ratio
        
        # Ensure minimum distances
        min_stop = 15 * pip_size
        stop_distance = max(stop_distance, min_stop)
        
        return stop_distance, take_profit_distance
        
    def calculate_adaptive_position_size(self, stop_distance, market_structure, risk_mult):
        """Calculate position size with adaptive risk.
        **Corrected for JPY pair volume errors.**
        """
        account_info = mt5.account_info()
        if not account_info:
            logging.error("Failed to get account info for position sizing.")
            return 0.01 # Return minimum as a fallback

        leverage = account_info.leverage
        free_margin = account_info.margin_free
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        pip_size = self.market_data.get_pip_size()
        
        if not all([leverage, free_margin, symbol_info, pip_size]):
             logging.error("Missing critical info for position sizing.")
             return 0.01

        # Base risk
        base_risk = 0.01  # 1%
        
        # Adjust for setup quality
        setup_quality_mult = 1.0
        if market_structure['trend']['strength'] > 0.8:
            setup_quality_mult = 1.1
            
        # Final risk calculation
        risk_percent = base_risk * risk_mult * setup_quality_mult
        risk_percent = min(risk_percent, 0.015)  # Cap at 1.5%
        
        # Calculate lot size
        risk_amount = account_info.equity * risk_percent
        
        # Prevent division by zero if stop_distance is too small
        min_stop_distance = pip_size * 10  # At least 10 pips
        if stop_distance < min_stop_distance:
            logging.warning(f"Stop distance too small ({stop_distance/pip_size:.1f} pips), using minimum")
            stop_distance = min_stop_distance

        stop_distance_pips = stop_distance / pip_size
        
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        
        # Prevent division by zero for tick_size
        if tick_size <= 1e-10:
            logging.error(f"Invalid tick_size for {self.market_data.symbol}: {tick_size}")
            return symbol_info.volume_min

        pip_value_per_lot = tick_value * (pip_size / tick_size)
        
        if stop_distance_pips * pip_value_per_lot <= 1e-10:
            logging.error("Risk calculation denominator is zero. Cannot size position.")
            return symbol_info.volume_min

        lot_size = risk_amount / (stop_distance_pips * pip_value_per_lot)
        
        # --- START OF CORRECTED ROUNDING LOGIC ---

        lot_step = symbol_info.volume_step
        
        # Round the calculated lot size to the nearest valid step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Apply broker volume limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # Determine required precision from lot_step (e.g., 0.01 -> 2 decimals)
        # This is the crucial step to prevent "Invalid Volume" errors from float precision
        precision = 2 # Default
        if lot_step > 0:
            try:
                # Use log10 to find number of decimal places
                precision = int(round(-math.log10(lot_step)))
            except (ValueError, TypeError):
                logging.warning(f"Could not determine precision from lot_step {lot_step}. Defaulting to 2.")
                precision = 2
        
        # Final rounding to the determined precision
        lot_size = round(lot_size, precision)
        
        # --- END OF CORRECTED ROUNDING LOGIC ---

        # Check margin requirements
        try:
            margin_required = (lot_size * symbol_info.trade_contract_size * symbol_info.ask) / leverage
            if margin_required > free_margin * 0.5:  # Use max 50% of free margin
                new_lot_size = (free_margin * 0.5 * leverage) / (symbol_info.trade_contract_size * symbol_info.ask)
                # Round down the new lot size to ensure it's within margin
                lot_size = math.floor(new_lot_size / lot_step) * lot_step
                lot_size = round(lot_size, precision) # Re-apply precision rounding
        except ZeroDivisionError:
            logging.error("ZeroDivisionError during margin calculation. Leverage or contract size might be zero.")
            return symbol_info.volume_min # Fallback to minimum volume

        # Final check to ensure volume is not zero after all calculations
        if lot_size < symbol_info.volume_min:
            return symbol_info.volume_min
            
        return lot_size
        
    def log_trade_details(self, setup, market_structure, result):
        """Log comprehensive trade details"""
        logging.info(f"{'='*60}")
        logging.info(f"TRADE EXECUTED: {self.market_data.symbol} ({setup['direction'].upper()})")
        logging.info(f"  - Setup Type: {setup['type']}")
        logging.info(f"  - Market Regime: {market_structure['regime']}")
        logging.info(f"  - Entry: {result.price:.5f} | Score: {setup['score']:.2f}")
        
        # Display the confluence factors that led to the high score
        if setup.get('confluence_notes'):
            notes = ', '.join(setup['confluence_notes'])
            logging.info(f"  - Confluence Factors: [{notes}]")
        
        logging.info(f"  - HTF Trend: {market_structure['trend']['direction']} (Strength: {market_structure['trend']['strength']:.2f})")
        logging.info(f"{'='*60}")
        
        if 'amd' in market_structure:
            amd = market_structure['amd']
            logging.info(f"  - AMD Phase: {amd['phase']}")
            if amd['setup_bias']:
                logging.info(f"  - AMD Bias: {amd['setup_bias']}")
            if amd['asian_range']:
                logging.info(f"  - Asian Range: {amd['asian_range']['low']:.5f} - {amd['asian_range']['high']:.5f}")
        
    def find_swing_points(self, data, window=10):
        """Find swing points - keep existing logic"""
        df = data.copy()
        
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
            
            for j in range(1, window + 1):
                if df.iloc[i - j]['high'] >= current_high:
                    is_swing_high = False
                if df.iloc[i - j]['low'] <= current_low:
                    is_swing_low = False
                    
                if df.iloc[i + j]['high'] >= current_high:
                    is_swing_high = False
                if df.iloc[i + j]['low'] <= current_low:
                    is_swing_low = False
            
            if is_swing_high:
                swing_highs.append({'time': df.index[i], 'bar_index': i, 'price': current_high})
            if is_swing_low:
                swing_lows.append({'time': df.index[i], 'bar_index': i, 'price': current_low})
                
        swing_highs_df = pd.DataFrame(swing_highs)
        swing_lows_df = pd.DataFrame(swing_lows)
        if not swing_highs_df.empty:
            swing_highs_df.set_index('time', inplace=True)
        if not swing_lows_df.empty:
            swing_lows_df.set_index('time', inplace=True)
        return swing_highs_df, swing_lows_df

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
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
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

# Use existing IndicatorCalculator class from original code
class IndicatorCalculator:
    def __init__(self):
        # Define default periods
        self.default_periods = {
            'atr': 14,
            'adx': 14,
            'rsi': 14,
            'ema_fast': 8,
            'ema_slow': 21,
            'ema_trend': 50,
            'bb_period': 20,
            'bb_std': 2
        }
        
    def calculate_volatility_factor(self, data, atr_period=14, ma_period=30):
        """Calculate volatility factor using standard deviations and normalized ATR"""
        try:
            # ATR from talib
            atr = ATR(data['high'], data['low'], data['close'], timeperiod=atr_period)
            if atr is None:
                return 1.0
            
            atr_percentage = (atr / data['close']) * 100                
            atrp_ma = atr_percentage.rolling(window=ma_period).mean()
            atrp_std = atr_percentage.rolling(window=ma_period).std()
            if pd.isna(atrp_std.iloc[-1]) or atrp_std.iloc[-1] < 0.0001:
                return 1.0
            current_zscore = (atr_percentage.iloc[-1] - atrp_ma.iloc[-1]) / atrp_std.iloc[-1]
            
            volatility_factor = 1 + np.tanh(current_zscore / 2) * 0.3            
            return round(volatility_factor, 2)
            
        except Exception as e:
            logging.error(f"Error calculating volatility factor: {e}")
            return 1.0
    
    def calculate_indicators(self, data, periods=None):
        """Calculate all technical indicators using TALib"""
        try:
            periods = periods or self.default_periods
            df = data.copy()
            
            # Calculate EMAs
            df['ema_fast'] = EMA(df['close'], timeperiod=periods['ema_fast']) 
            df['ema_slow'] = EMA(df['close'], timeperiod=periods['ema_slow'])   
            
            # Momentum Indicators
            df['rsi'] = RSI(df['close'], timeperiod=periods['rsi'])    
            
            # Trend Indicators
            df['adx'] = ADX(df['high'], df['low'], df['close'], timeperiod=periods['adx'])
            df['plus_di'] = PLUS_DI(df['high'], df['low'], df['close'], timeperiod=periods['adx'])
            df['minus_di'] = MINUS_DI(df['high'], df['low'], df['close'], timeperiod=periods['adx'])      
                        
            # Volatility Indicators
            df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=periods['atr'])
            df['atr_dynamic'] = df['atr'].rolling(window=5).mean()
            
            # Calculate volatility factor
            df['volatility_factor'] = self.calculate_volatility_factor(df)   
                                    
            return df
            
        except Exception as e:
            logging.error(f"Error in calculate_indicators: {str(e)}")
            logging.error(traceback.format_exc())
            return None

# Keep existing OrderManager class
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
            "magic": 234000,
            "comment": "Adaptive EA",
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

def check_symbol_adaptive(symbol, timeframes, market_context):
    """
    Adaptive signal checking for a single symbol
    """
    try:
        # Pre-checks
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info or not symbol_info.visible:
            return False, f"[{symbol}] Not available", None
        
        # Initialize components
        market_data = MarketData(symbol, timeframes)
        trade_manager = AdaptiveTradeManager(None, market_data , market_context)
        
        # Fetch data
        data = {}
        for tf in timeframes:
            data[tf] = market_data.fetch_data(tf)
            if data[tf] is None:
                return False, f"[{symbol}] Failed to fetch {tf} data", None    
        
        latest_m5_time = data[mt5.TIMEFRAME_M5].index[-1]
        analysis_time = latest_m5_time
        
        # Check for news
        news_check = market_context.is_news_time(
            symbol, 
            minutes_before=30, 
            minutes_after=15,
            min_impact='High',
            current_time=analysis_time
        )
        if news_check['is_news']:
            return False, f"[{symbol}] {news_check['impact']} impact {news_check['currency']} news in {news_check['minutes_to_event']}min: {news_check['name']}", None
            
        # Check spread
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, f"[{symbol}] Failed to get tick", None
        spread_price = tick.ask - tick.bid 
        max_spread_pips = 4.0
        pip_size = market_data.get_pip_size()      # e.g. 0.0001 for EURUSD
        max_spread_price = max_spread_pips * pip_size
        if spread_price > max_spread_price:
            return False, f"[{symbol}] Spread too high ({spread_price:.5f})", None
        
                        
        # Analyze market structure on H1
        h1_structure = trade_manager.analyze_market_structure(data[mt5.TIMEFRAME_H1])
        if not h1_structure or 'atr' not in h1_structure:
            return False, f"[{symbol}] Failed to analyze H1 structure", None
            
        # Log market analysis
        logging.info(f"[{symbol}] Market: {h1_structure['regime']}, "
                    f"Trend: {h1_structure['trend']['direction']} "
                    f"({h1_structure['trend']['strength']:.2f})")
        
        # Find setups on M15
        m15_setups = trade_manager.find_entry_setups(h1_structure, data[mt5.TIMEFRAME_M15])
        
        if not m15_setups:
            # Log why no setups
            if h1_structure['trend']['pullback_depth'] < 0.2:
                reason = "No pullback yet"
            elif h1_structure['regime'] == 'STRONG_TREND':
                reason = "Trend too strong for Fib entries"
            else:
                reason = "No confluence zones"
            return False, f"[{symbol}] No setups: {reason}", None
            
        # Log found setups
        logging.info(f"[{symbol}] Found {len(m15_setups)} setups:")
        for setup in m15_setups[:3]:  # Log top 3
            logging.info(f"  - {setup['type']} ({setup['direction']}) score: {setup['score']:.2f}")
            
        # Take the best setup
        best_setup = m15_setups[0]
        
        # Confirm timing on M5
        m5_df = data[mt5.TIMEFRAME_M5]
        current_price = m5_df['close'].iloc[-1]
        
        # Quick M5 confirmation based on setup type
        timing_confirmed = False
        
        if best_setup['type'] == 'momentum':
            # Check last 3 closed M5 bars for momentum
            last_three = m5_df.iloc[-4:-1]  # Last 3 closed bars
            if best_setup['direction'] == 'buy':
                bullish_bars = sum(1 for _, bar in last_three.iterrows() if bar['close'] > bar['open'])
                timing_confirmed = bullish_bars >= 2  # At least 2 of 3 bullish
            else:
                bearish_bars = sum(1 for _, bar in last_three.iterrows() if bar['close'] < bar['open'])
                timing_confirmed = bearish_bars >= 2  # At least 2 of 3 bearish
                
        elif best_setup['type'] in ['fibonacci', 'ma_bounce']:
            # For reversal setups, need rejection confirmation
            if best_setup.get('zone'):
                rejection = best_setup['zone'].get_rejection_strength(m5_df, lookback=3)
                timing_confirmed = rejection > 0.5
            else:
                timing_confirmed = True  # Fallback
                
        else:
            timing_confirmed = True  # Other setup types
            
        if not timing_confirmed:
            return False, f"[{symbol}] Waiting for M5 confirmation", None
        
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            return False, f"[{symbol}] Already have position", None
            
        # Execute the setup
        success, message = trade_manager.execute_setup(best_setup, h1_structure)
        
        if success:
            trade_info = {
                'symbol': symbol,
                'setup_type': best_setup['type'],
                'direction': best_setup['direction'],
                'score': best_setup['score'],
                'regime': h1_structure['regime']
            }
            return True, f"[{symbol}] {message}", trade_info
        else:
            return False, f"[{symbol}] {message}", None
            
    except Exception as e:
        logging.error(f"[{symbol}] Error in adaptive check: {str(e)}")
        logging.error(traceback.format_exc())
        return False, f"[{symbol}] Error: {str(e)}", None

def main():
    """Main execution loop with adaptive strategy"""
    # Configuration
    SYMBOLS = ['AUDUSD', 'CHFJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 
               'EURCAD', 'GBPJPY', 'AUDCHF', 'AUDCAD', 'AUDJPY', 'EURAUD', 'EURJPY', 
               'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 
               'GBPNZD', 'NZDUSD']
    TIMEFRAMES = (mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1)
    CHECK_INTERVAL = 300  # 5 minutes
    
    # Initialize MT5
    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("Failed to connect to MetaTrader 5")
        return
        
    # Display account info
    account_info = client.get_account_info()
    if not account_info:
        logging.error("Failed to get account info")
        mt5.shutdown()
        return
        
    logging.info(f"Adaptive Trading System Started")
    logging.info(f"Account: {account_info.login}, Balance: ${account_info.balance:.2f}, "
                f"Equity: ${account_info.equity:.2f}")
    
    # Initialize market context for news checking
    market_context = MarketContext(auto_update=True)
    
    # Performance tracking
    performance = {
        'trades_by_type': {},
        'trades_by_regime': {},
        'total_trades': 0,
        'last_update': time.time()
    }
    
    try:
        while True:
            start_time = time.time()
            
            # Get session info
            session_info = market_context.get_trading_session()
            session = session_info['name']
            volatility_mult = session_info['volatility_multiplier']
            logging.info(f"\n[Session: {session}] Starting market scan...")
            
            # Check for upcoming news in next 2 hours
            upcoming_news = market_context.get_news_summary(hours_ahead=2)
            if upcoming_news:
                logging.info("Upcoming high impact news:")
                for news in upcoming_news:
                    logging.info(f"  {news['time']} UTC - {news['currency']}: {news['event']} (in {news['hours_until']:.1f}h)")
            
            # Scan all symbols
            signals_found = 0
            regime_summary = {}
            news_blocked = 0
            
            for symbol in SYMBOLS:
                try:
                    success, message, trade_info = check_symbol_adaptive(symbol, TIMEFRAMES, market_context)
                    
                    logging.info(message)
                    
                    # Track news blocks
                    if "news" in message.lower():
                        news_blocked += 1
                    
                    if success and trade_info:
                        signals_found += 1
                        performance['total_trades'] += 1
                        
                        # Track by type
                        setup_type = trade_info['setup_type']
                        performance['trades_by_type'][setup_type] = \
                            performance['trades_by_type'].get(setup_type, 0) + 1
                            
                        # Track by regime
                        regime = trade_info['regime']
                        performance['trades_by_regime'][regime] = \
                            performance['trades_by_regime'].get(regime, 0) + 1
                            
                        # Log trade
                        logging.info(f"\n{'*'*60}")
                        logging.info(f"NEW TRADE: {symbol}")
                        logging.info(f"Setup: {setup_type} in {regime} market")
                        logging.info(f"Direction: {trade_info['direction'].upper()}")
                        logging.info(f"Confidence: {trade_info['score']:.2f}")
                        logging.info(f"{'*'*60}\n")
                        
                    # Track market regimes
                    if "Market:" in message:
                        regime = message.split("Market: ")[1].split(",")[0]
                        regime_summary[regime] = regime_summary.get(regime, 0) + 1
                        
                except Exception as e:
                    logging.error(f"Error checking {symbol}: {str(e)}")
                    continue
            
            # Summary logging
            if regime_summary:
                logging.info("\nMarket Regime Summary:")
                for regime, count in regime_summary.items():
                    logging.info(f"  {regime}: {count} pairs")
            
            if news_blocked > 0:
                logging.info(f"\nBlocked {news_blocked} trades due to upcoming news events")
            
            # Monitor positions
            positions = mt5.positions_get()
            if positions:
                logging.info(f"\nOpen Positions: {len(positions)}")
                total_pl = sum(pos.profit for pos in positions)
                logging.info(f"Total P/L: ${total_pl:.2f}")
            
            # Performance update every 30 minutes
            if time.time() - performance['last_update'] > 1800:
                # Reload calendar to get latest news
                market_context.reload_calendar()
                
                if performance['total_trades'] > 0:
                    logging.info("\n" + "="*50)
                    logging.info("PERFORMANCE UPDATE")
                    logging.info(f"Total Trades: {performance['total_trades']}")
                    logging.info("Trades by Type:")
                    for setup_type, count in performance['trades_by_type'].items():
                        pct = (count / performance['total_trades']) * 100
                        logging.info(f"  {setup_type}: {count} ({pct:.1f}%)")
                    logging.info("Trades by Market:")
                    for regime, count in performance['trades_by_regime'].items():
                        pct = (count / performance['total_trades']) * 100
                        logging.info(f"  {regime}: {count} ({pct:.1f}%)")
                    logging.info("="*50 + "\n")
                performance['last_update'] = time.time()
            
            # Calculate sleep time
            cycle_duration = time.time() - start_time
            sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
            
            if sleep_time > 0:
                if signals_found > 0:
                    logging.info(f"\nExecuted {signals_found} trades. Next scan in {sleep_time:.0f} seconds...")
                else:
                    logging.info(f"\nNo new setups. Next scan in {sleep_time:.0f} seconds...")
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logging.info("\nShutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
    finally:
        # Cleanup
        mt5.shutdown()
        logging.info("\nTrading system terminated")
        
        # Final summary
        if performance['total_trades'] > 0:
            logging.info("\nFINAL SUMMARY")
            logging.info(f"Total Trades Executed: {performance['total_trades']}")
            logging.info("\nMost Active Setups:")
            sorted_types = sorted(performance['trades_by_type'].items(), 
                                key=lambda x: x[1], reverse=True)
            for setup_type, count in sorted_types[:3]:
                logging.info(f"  {setup_type}: {count} trades")

if __name__ == "__main__":
    main()