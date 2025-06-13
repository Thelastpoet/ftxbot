import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timezone
from talib import ATR, EMA, RSI, ADX, PLUS_DI, MINUS_DI, MACD, BBANDS, STOCH
from collections import deque

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketRegimeDetector:
    """Detects market regime to adapt strategy accordingly"""
    
    @staticmethod
    def classify_market(adx, atr_ratio, price_position):
        """
        Classify market into: STRONG_TREND, NORMAL_TREND, RANGING, VOLATILE
        """
        if adx > 40 and (price_position > 0.8 or price_position < 0.2):
            return 'STRONG_TREND', {
                'fib_levels': [0.236, 0.382],  # Shallow only
                'confluence_required': 0.6,
                'entry_methods': ['momentum', 'shallow_pullback', 'ma_bounce'],
                'risk_multiplier': 1.2
            }
        elif adx > 25 and (price_position > 0.6 or price_position < 0.4):
            return 'NORMAL_TREND', {
                'fib_levels': [0.382, 0.5, 0.618],
                'confluence_required': 0.8,
                'entry_methods': ['fibonacci', 'structure_break', 'ma_bounce'],
                'risk_multiplier': 1.0
            }
        elif adx < 20:
            return 'RANGING', {
                'fib_levels': [0.5, 0.618, 0.786],  # Deeper levels
                'confluence_required': 0.9,
                'entry_methods': ['fibonacci', 'range_extreme'],
                'risk_multiplier': 0.8
            }
        else:
            return 'VOLATILE', {
                'fib_levels': [0.382, 0.5, 0.618],
                'confluence_required': 1.0,
                'entry_methods': ['fibonacci'],
                'risk_multiplier': 0.7
            }

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
        
        for _, bar in recent_bars.iterrows():
            # Check for bullish rejection (lower wick)
            if bar['low'] <= self.center <= bar['close']:
                wick_size = min(bar['open'], bar['close']) - bar['low']
                body_size = abs(bar['close'] - bar['open'])
                if body_size > 0:
                    rejection_score += min(wick_size / body_size, 3.0)
                    
            # Check for bearish rejection (upper wick)
            elif bar['high'] >= self.center >= bar['close']:
                wick_size = bar['high'] - max(bar['open'], bar['close'])
                body_size = abs(bar['close'] - bar['open'])
                if body_size > 0:
                    rejection_score += min(wick_size / body_size, 3.0)
                    
        return rejection_score

class AdaptiveTradeManager:
    """Enhanced trade manager that adapts to market conditions"""
    
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.order_manager = OrderManager(client, market_data)
        self.indicator_calc = IndicatorCalculator()
        self.market_context = MarketContext()
        self.regime_detector = MarketRegimeDetector()
        
        # Setup tracking
        self.setup_history = deque(maxlen=100)  # Track recent setups
        
        # Initialize timeframe hierarchy
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] 
        self.tf_lower = min(self.timeframes)
        
    def analyze_market_structure(self, data):
        """
        Comprehensive market structure analysis
        Returns: structure dict with all relevant info
        """
        try:
            df = data.copy()
            
            # Calculate indicators
            df = self.indicator_calc.calculate_indicators(df)
            if df is None:
                return None
                
            # Get basic metrics
            current_price = df['close'].iloc[-1]
            adx = df['adx'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Price position in recent range
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            range_20 = high_20 - low_20
            price_position = (current_price - low_20) / range_20 if range_20 > 0 else 0.5
            
            # ATR ratio (current vs average)
            atr_ma = df['atr'].rolling(20).mean().iloc[-1]
            atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0
            
            # Classify market regime
            regime, params = self.regime_detector.classify_market(adx, atr_ratio, price_position)
            
            # Find all relevant structure levels
            structure_levels = self.find_key_levels(df)
            
            # Trend analysis
            trend_info = self.analyze_trend_comprehensive(df)
            
            return {
                'regime': regime,
                'regime_params': params,
                'trend': trend_info,
                'structure_levels': structure_levels,
                'current_price': current_price,
                'atr': atr,
                'adx': adx,
                'price_position': price_position
            }
            
        except Exception as e:
            logging.error(f"Error in analyze_market_structure: {e}")
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
                direction = 'uptrend'
                strength_base = 1.0
            elif price < ema_fast < ema_slow:
                direction = 'downtrend'
                strength_base = 1.0
            else:
                direction = 'unclear'
                strength_base = 0.3
                
            # Calculate pullback depth if in trend
            if direction != 'unclear':
                # Find recent extreme
                if direction == 'uptrend':
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
            momentum_aligned = (direction == 'uptrend' and rsi > 50) or (direction == 'downtrend' and rsi < 50)
            
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
        round_interval = 50 * pip_size if 'JPY' in self.market_data.symbol else 50 * pip_size
        
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
        
        # Get appropriate entry methods for current regime
        entry_methods = params['entry_methods']
        
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
            
        # Score and filter setups
        scored_setups = self.score_setups(setups, market_structure)
        
        # Return only high-quality setups
        min_score = params['confluence_required']
        return [s for s in scored_setups if s['score'] >= min_score]
        
    def find_fibonacci_setups(self, df, market_structure, fib_levels):
        """Enhanced Fibonacci setup detection using zones"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['uptrend', 'downtrend']:
            return setups
            
        # Get recent swing points
        swing_highs, swing_lows = self.find_swing_points(df, window=15)
        if len(swing_highs) < 1 or len(swing_lows) < 1:
            return setups
            
        # Determine impulse move
        if trend == 'uptrend':
            last_high = swing_highs.iloc[-1]
            # Find swing lows that occurred BEFORE this high
            valid_lows = swing_lows[swing_lows.index < last_high.name] if hasattr(last_high, 'name') else swing_lows[:-1]
            if len(valid_lows) == 0:
                return setups
            impulse_start = valid_lows.iloc[-1]['price']
            impulse_end = last_high['price']
        else:
            last_low = swing_lows.iloc[-1]
            # Find swing highs that occurred BEFORE this low
            valid_highs = swing_highs[swing_highs.index < last_low.name] if hasattr(last_low, 'name') else swing_highs[:-1]
            if len(valid_highs) == 0:
                return setups
            impulse_start = valid_highs.iloc[-1]['price']
            impulse_end = last_low['price']
            
        swing_range = abs(impulse_end - impulse_start)
        atr = market_structure['atr']
        current_price = df['close'].iloc[-1]
        
        # Check each Fib level
        for ratio in fib_levels:
            if trend == 'uptrend':
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
                        'direction': 'buy' if trend == 'uptrend' else 'sell',
                        'entry_price': current_price,
                        'rejection_strength': rejection_strength,
                        'base_score': 0.7
                    }
                    setups.append(setup)
                    
        return setups
        
    def find_momentum_setups(self, df, market_structure):
        """Find momentum continuation setups for strong trends"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['uptrend', 'downtrend'] or market_structure['regime'] != 'STRONG_TREND':
            return setups
            
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        
        # Look for shallow pullback to EMA
        ema_fast = market_structure['trend']['ema_fast']
        zone = PriceActionZone(ema_fast, atr, 'structure')
        
        if zone.contains_price(current_price):
            # Check momentum
            last_3_bars = df.iloc[-3:]
            momentum_confirmed = False
            
            if trend == 'uptrend':
                # Bullish momentum: higher lows, bullish bars
                if (last_3_bars['low'].is_monotonic_increasing and 
                    last_3_bars.iloc[-1]['close'] > last_3_bars.iloc[-1]['open']):
                    momentum_confirmed = True
            else:
                # Bearish momentum: lower highs, bearish bars
                if (last_3_bars['high'].is_monotonic_decreasing and
                    last_3_bars.iloc[-1]['close'] < last_3_bars.iloc[-1]['open']):
                    momentum_confirmed = True
                    
            if momentum_confirmed:
                setup = {
                    'type': 'momentum',
                    'zone': zone,
                    'direction': 'buy' if trend == 'uptrend' else 'sell',
                    'entry_price': current_price,
                    'base_score': 0.8
                }
                setups.append(setup)
                
        return setups
        
    def find_ma_bounce_setups(self, df, market_structure):
        """Find moving average bounce setups"""
        setups = []
        trend = market_structure['trend']['direction']
        
        if trend not in ['uptrend', 'downtrend']:
            return setups
            
        current_price = df['close'].iloc[-1]
        atr = market_structure['atr']
        pip_size = self.market_data.get_pip_size()
        
        # Check different MAs
        ma_periods = [20, 50] if len(df) >= 50 else [20]
        
        for period in ma_periods:
            ma_value = df['close'].rolling(period).mean().iloc[-1]
            ma_slope = (ma_value - df['close'].rolling(period).mean().iloc[-5]) / 5  # MA direction
            
            # MA must be sloping in trend direction
            if trend == 'uptrend' and ma_slope <= 0:
                continue  # Skip falling MA in uptrend
            if trend == 'downtrend' and ma_slope >= 0:
                continue  # Skip rising MA in downtrend
                
            # Check for ACTUAL bounce pattern
            bounce_found = False
            bounce_bar_index = None
            
            # Look for bounce in last 3 bars (not 5!)
            for i in range(-3, 0):
                bar = df.iloc[i]
                prev_bar = df.iloc[i-1] if i > -len(df) else None
                
                if trend == 'uptrend':
                    # Bullish bounce: Low touches/pierces MA, then closes above
                    ma_at_bar = df['close'].rolling(period).mean().iloc[i]
                    
                    if (bar['low'] <= ma_at_bar * 1.002 and  # Within 0.2% of MA
                        bar['close'] > ma_at_bar and
                        bar['close'] > bar['open']):  # Bullish close
                        
                        # Verify price is moving away from MA
                        if i < -1:  # Not the last bar
                            next_bar = df.iloc[i+1]
                            if next_bar['low'] > ma_at_bar:  # Held above MA
                                bounce_found = True
                                bounce_bar_index = i
                                break
                                
                else:  # downtrend
                    # Bearish bounce: High touches/pierces MA, then closes below
                    ma_at_bar = df['close'].rolling(period).mean().iloc[i]
                    
                    if (bar['high'] >= ma_at_bar * 0.998 and  # Within 0.2% of MA
                        bar['close'] < ma_at_bar and
                        bar['close'] < bar['open']):  # Bearish close
                        
                        # Verify price is moving away from MA
                        if i < -1:  # Not the last bar
                            next_bar = df.iloc[i+1]
                            if next_bar['high'] < ma_at_bar:  # Held below MA
                                bounce_found = True
                                bounce_bar_index = i
                                break
            
            if not bounce_found:
                continue
                
            # Additional quality checks
            distance_from_ma = abs(current_price - ma_value)
            
            # Don't chase - price should still be near MA
            if distance_from_ma > atr * 0.5:
                continue  # Too far from MA now
                
            # Create setup
            setup = {
                'type': 'ma_bounce',
                'ma_period': period,
                'ma_value': ma_value,
                'direction': 'buy' if trend == 'uptrend' else 'sell',
                'entry_price': current_price,
                'bounce_bar_index': bounce_bar_index,
                'distance_from_ma': distance_from_ma / pip_size,
                'base_score': 0.6  # Lower base score
            }
            setups.append(setup)
            
            # Only take the first valid MA bounce
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
        
    def score_setups(self, setups, market_structure):
        """Score setups based on multiple confluence factors"""
        scored_setups = []
        
        for setup in setups:
            score = setup['base_score']
            
            # 1. Add confluence from structure levels
            for level in market_structure['structure_levels']:
                if setup.get('zone') and abs(setup['zone'].center - level['price']) < market_structure['atr'] * 0.2:
                    score += level['strength'] * 0.2
                    
            # 2. Trend alignment bonus
            if setup['direction'] == 'buy' and market_structure['trend']['direction'] == 'uptrend':
                score += 0.1
            elif setup['direction'] == 'sell' and market_structure['trend']['direction'] == 'downtrend':
                score += 0.1
                
            # 3. Momentum confirmation
            if market_structure['trend'].get('momentum_aligned'):
                score += 0.1
                
            # 4. Session quality
            session, session_mult = self.market_context.get_trading_session()
            if session in ['LONDON_OPEN', 'LONDON_NY_OVERLAP']:
                score += 0.1
                
            # 5. Setup-specific bonuses
            if setup['type'] == 'fibonacci' and setup.get('rejection_strength', 0) > 2.0:
                score += 0.2
            elif setup['type'] == 'momentum' and market_structure['adx'] > 40:
                score += 0.2
                
            setup['score'] = min(score, 1.0)  # Cap at 1.0
            scored_setups.append(setup)
            
        # Sort by score
        scored_setups.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_setups
        
    def execute_setup(self, setup, market_structure):
        """Execute the trading setup"""
        try:
            symbol = self.market_data.symbol
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
        elif setup['type'] == 'ma_bounce':
            # Stop beyond the MA
            stop_distance = atr * 1.0
        elif setup['type'] == 'range_extreme':
            # Stop outside the range
            stop_distance = atr * 1.2
        else:
            stop_distance = atr * 1.0
            
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
        """Calculate position size with adaptive risk"""
        account_info = mt5.account_info()
        symbol_info = mt5.symbol_info(self.market_data.symbol)
        pip_size = self.market_data.get_pip_size()
        
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
        stop_distance_pips = stop_distance / pip_size
        
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        pip_value_per_lot = tick_value * (pip_size / tick_size)
        
        lot_size = risk_amount / (stop_distance_pips * pip_value_per_lot)
        
        # Round to lot step
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        
        # Apply limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        return lot_size
        
    def log_trade_details(self, setup, market_structure, result):
        """Log comprehensive trade details"""
        logging.info(f"{'='*60}")
        logging.info(f"TRADE EXECUTED: {self.market_data.symbol}")
        logging.info(f"Setup Type: {setup['type']}")
        logging.info(f"Market Regime: {market_structure['regime']}")
        logging.info(f"Direction: {setup['direction'].upper()}")
        logging.info(f"Entry: {result.price:.5f}")
        logging.info(f"Setup Score: {setup['score']:.2f}")
        logging.info(f"Trend Strength: {market_structure['trend']['strength']:.2f}")
        logging.info(f"{'='*60}")
        
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
                swing_highs.append({'index': i, 'price': current_high, 'time': df.index[i]})
            if is_swing_low:
                swing_lows.append({'index': i, 'price': current_low, 'time': df.index[i]})
        
        return pd.DataFrame(swing_highs), pd.DataFrame(swing_lows)

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

class MarketContext:
    """Professional market context analyzer"""
    
    @staticmethod
    def get_trading_session():
        """Get current trading session with proper timezone handling"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        # Define sessions in UTC
        if 22 <= hour or hour < 7:  # Sydney/Tokyo
            return 'ASIAN', 0.8  # Lower volatility multiplier
        elif 7 <= hour < 12:  # London morning
            return 'LONDON_OPEN', 1.2  # High volatility
        elif 12 <= hour < 15:  # London/NY overlap
            return 'LONDON_NY_OVERLAP', 1.3  # Highest volatility
        elif 15 <= hour < 20:  # NY main
            return 'NY_MAIN', 1.1
        else:  # NY close
            return 'NY_CLOSE', 0.9
            
    @staticmethod
    def is_news_time():
        """Check if we're near major news (simplified - integrate with calendar API)"""
        now = datetime.now(timezone.utc)
        hour, minute = now.hour, now.minute
        
        # Major news times (add your specific times)
        news_times = [
            (8, 30),   # UK news
            (13, 30),  # US news
            (15, 0),   # US news
        ]
        
        for news_hour, news_min in news_times:
            time_to_news = (news_hour - hour) * 60 + (news_min - minute)
            if -5 <= time_to_news <= 30:  # 5 min after to 30 min before
                return True
        return False

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

def check_symbol_adaptive(symbol, timeframes):
    """
    Adaptive signal checking for a single symbol
    """
    try:
        # Pre-checks
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info or not symbol_info.visible:
            return False, f"[{symbol}] Not available", None
            
        # Check spread
        max_spread = 40 if 'JPY' in symbol else 40  # Slightly wider for adaptability
        if symbol_info.spread > max_spread:
            return False, f"[{symbol}] Spread too high ({symbol_info.spread})", None
            
        # Check existing positions
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            return False, f"[{symbol}] Already have position", None
            
        # Initialize components
        market_data = MarketData(symbol, timeframes)
        trade_manager = AdaptiveTradeManager(None, market_data)
        
        # Fetch data
        data = {}
        for tf in timeframes:
            data[tf] = market_data.fetch_data(tf)
            if data[tf] is None:
                return False, f"[{symbol}] Failed to fetch {tf} data", None
                
        # Analyze market structure on H1
        h1_structure = trade_manager.analyze_market_structure(data[mt5.TIMEFRAME_H1])
        if not h1_structure:
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
            # For momentum, just need price action confirmation
            last_bar = m5_df.iloc[-2]
            if best_setup['direction'] == 'buy':
                timing_confirmed = last_bar['close'] > last_bar['open']
            else:
                timing_confirmed = last_bar['close'] < last_bar['open']
                
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
    """
    Main execution loop with adaptive strategy
    """
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
            session, volatility_mult = MarketContext.get_trading_session()
            logging.info(f"\n[Session: {session}] Starting market scan...")
            
            # Check for news
            if MarketContext.is_news_time():
                logging.warning("Near news time - reduced activity")
            
            # Scan all symbols
            signals_found = 0
            regime_summary = {}
            
            for symbol in SYMBOLS:
                try:
                    success, message, trade_info = check_symbol_adaptive(symbol, TIMEFRAMES)
                    
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
            
            # Monitor positions
            positions = mt5.positions_get()
            if positions:
                logging.info(f"\nOpen Positions: {len(positions)}")
                total_pl = sum(pos.profit for pos in positions)
                logging.info(f"Total P/L: ${total_pl:.2f}")
            
            # Performance update every 30 minutes
            if time.time() - performance['last_update'] > 1800:
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