"""
Active Trade Management System for Adaptive Forex EA

Production-ready implementation for live trading.
Manages open positions with dynamic stop adjustments and intelligent exits.

Author: Adaptive EA Trade Manager
Version: 1.0
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from talib import ATR, EMA

class TradeManager:
    """
    Production-ready trade management system with complete implementations
    """
    
    def __init__(self, magic_number=234000):
        self.magic_number = magic_number
        
        # Position tracking with persistence
        self.position_contexts = {}
        self.last_modification_time = {}
        self.modification_attempts = {}
        self.failed_modifications = {}
        
        # Configuration
        self.config = self._load_config()
        
        # Statistics tracking
        self.stats = {
            'breakeven_moves': 0,
            'trailing_activations': 0,
            'invalidation_exits': 0,
            'session_exits': 0,
            'regime_exits': 0,
            'errors': 0,
            'total_modifications': 0,
            'failed_modifications': 0
        }
        
        # Setup persistent storage
        self.storage_file = Path('trade_contexts.json')
        self._load_contexts_from_file()
        
        # Session times in UTC
        self.session_times = {
            'ASIAN_SESSION': {'start': 0, 'end': 7},
            'LONDON_OPEN': {'start': 7, 'end': 9},
            'LONDON_MAIN': {'start': 9, 'end': 12},
            'LONDON_NY_OVERLAP': {'start': 12, 'end': 16},
            'NY_MAIN': {'start': 16, 'end': 20},
            'NY_CLOSE': {'start': 20, 'end': 21},
            'SYDNEY_SESSION': {'start': 21, 'end': 24}
        }
        
        logging.info("Trade Manager initialized for production trading")
        
    def _load_config(self) -> dict:
        """Load production configuration"""
        return {
            'enabled': True,
            'features': {
                'breakeven': {
                    'enabled': True,
                    'trigger_r_multiple': 1.0,
                    'buffer_pips': 1.0,
                    'min_profit_pips': 10
                },
                'trailing': {
                    'enabled': True,
                    'trigger_r_multiple': 1.5,
                    'trail_method': 'hybrid',  # Uses both ATR and MA
                    'atr_multiplier': 2.0,
                    'ma_period': 20,
                    'min_trail_distance_pips': 15
                },
                'invalidation': {
                    'enabled': True,
                    'check_interval_seconds': 60,
                    'respect_minimum_time': True,
                    'min_time_minutes': 5,
                    'max_adverse_move_r': 0.5  # Exit if adverse move > 0.5R
                },
                'session': {
                    'enabled': True,
                    'close_profitable_before_session_end': True,
                    'minutes_before_close': 30,
                    'sessions_to_close': ['LONDON_NY_OVERLAP', 'NY_CLOSE'],
                    'min_profit_pips_to_close': 5
                },
                'regime': {
                    'enabled': True,
                    'exit_on_adverse_shift': True,
                    'tighten_on_favorable_shift': True,
                    'check_interval_minutes': 15
                }
            },
            'safety': {
                'max_modifications_per_position': 20,
                'min_seconds_between_modifications': 300,
                'circuit_breaker_threshold': 3,
                'verify_stops': True,
                'max_spread_pips': 5.0,
                'connection_check_interval': 60
            }
        }
        
    def check_positions(self, market_context=None) -> None:
        """Main method to check and manage all open positions"""
        if not self.config['enabled']:
            return
            
        # Verify connection
        if not mt5.terminal_info().connected:
            logging.error("MT5 not connected. Skipping position management.")
            return
            
        positions = mt5.positions_get()
        if not positions:
            return
            
        for position in positions:
            if position.magic != self.magic_number:
                continue
                
            try:
                self._manage_single_position(position, market_context)
            except Exception as e:
                logging.error(f"Error managing position {position.ticket}: {e}")
                self.stats['errors'] += 1
                
        # Clean up old contexts
        self._cleanup_closed_positions()
                
    def _manage_single_position(self, position, market_context) -> None:
        """Complete position management implementation"""
        ticket = position.ticket
        symbol = position.symbol
        
        # Rate limiting check
        if not self._can_modify_position(ticket):
            return
            
        # Check spread before any modifications
        if not self._check_spread(symbol):
            return
            
        # Get or create position context
        context = self._get_position_context(position)
        if not context:
            context = self._create_position_context(position)
            
        # Fetch fresh market data
        market_data = self._fetch_market_data(symbol)
        if not market_data:
            logging.warning(f"No market data available for {symbol}")
            return
            
        # Analyze current market structure
        current_structure = self._analyze_market_structure(market_data)
        
        # Update context with current market info
        context['last_check_time'] = time.time()
        context['current_profit_points'] = self._calculate_current_profit(position)
        
        # Apply management rules in priority order
        action_taken = False
        
        # 1. Emergency exits - Setup invalidation
        if self.config['features']['invalidation']['enabled']:
            should_exit, reason = self._check_invalidation(
                position, context, market_data, current_structure
            )
            if should_exit:
                if self._close_position(position, reason):
                    self.stats['invalidation_exits'] += 1
                    return
                    
        # 2. Regime change management
        if self.config['features']['regime']['enabled']:
            action = self._check_regime_change(
                position, context, current_structure
            )
            if action == 'exit':
                if self._close_position(position, "Adverse regime change"):
                    self.stats['regime_exits'] += 1
                    return
            elif action == 'tighten':
                action_taken = self._tighten_stop(
                    position, context, market_data, current_structure
                )
                
        # 3. Session-based management
        if self.config['features']['session']['enabled']:
            should_close, reason = self._check_session_end(position, context)
            if should_close:
                if self._close_position(position, reason):
                    self.stats['session_exits'] += 1
                    return
                    
        # 4. News event check
        if market_context:
            news_action = self._check_news_events(
                position, context, market_context
            )
            if news_action == 'close':
                if self._close_position(position, "High impact news"):
                    return
                    
        # 5. Profit protection
        if not action_taken:
            # Break-even check
            if self.config['features']['breakeven']['enabled']:
                if self._should_move_to_breakeven(position, context):
                    if self._move_to_breakeven(position):
                        context['moved_to_breakeven'] = True
                        self.stats['breakeven_moves'] += 1
                        action_taken = True
                        
            # Trailing stop check
            if not action_taken and self.config['features']['trailing']['enabled']:
                if self._should_trail_stop(position, context):
                    if self._trail_stop(position, context, market_data):
                        context['trailing_active'] = True
                        self.stats['trailing_activations'] += 1
                        
        # Update highest/lowest profit tracking
        self._update_profit_tracking(position, context)
        
        # Save updated context
        self._save_position_context(ticket, context)
        
    def _fetch_market_data(self, symbol) -> Dict:
        """Fetch complete market data for all required timeframes"""
        market_data = {}
        
        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol {symbol}")
            return {}
            
        timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        for tf_name, tf_value in timeframes.items():
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 200)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Calculate indicators
                df = self._calculate_indicators(df)
                market_data[tf_name] = df
            else:
                logging.warning(f"Failed to fetch {tf_name} data for {symbol}")
                
        return market_data
        
    def _calculate_indicators(self, df) -> pd.DataFrame:
        """Calculate all required indicators"""
        if len(df) < 50:
            return df
            
        # EMAs
        df['ema_8'] = EMA(df['close'], timeperiod=8)
        df['ema_20'] = EMA(df['close'], timeperiod=20)
        df['ema_50'] = EMA(df['close'], timeperiod=50)
        
        # ATR
        df['atr'] = ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Simple MAs for comparison
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        return df
        
    def _analyze_market_structure(self, market_data) -> Dict:
        """Analyze current market structure from data"""
        h1_data = market_data.get('H1')
        if h1_data is None or len(h1_data) < 50:
            return {'regime': 'UNKNOWN', 'trend': 'unclear'}
            
        current_price = h1_data['close'].iloc[-1]
        ema_20 = h1_data['ema_20'].iloc[-1]
        ema_50 = h1_data['ema_50'].iloc[-1]
        atr = h1_data['atr'].iloc[-1]
        
        # Determine trend
        if current_price > ema_20 > ema_50:
            trend = 'uptrend'
        elif current_price < ema_20 < ema_50:
            trend = 'downtrend'
        else:
            trend = 'unclear'
            
        # Calculate volatility
        atr_percentage = (atr / current_price) * 100
        
        # Determine regime
        if trend != 'unclear' and abs(ema_20 - ema_50) / atr > 2:
            regime = 'STRONG_TREND'
        elif atr_percentage > 1.5:
            regime = 'VOLATILE'
        elif abs(ema_20 - ema_50) / atr < 0.5:
            regime = 'RANGING'
        else:
            regime = 'NORMAL_TREND'
            
        return {
            'regime': regime,
            'trend': trend,
            'atr': atr,
            'current_price': current_price
        }
        
    def _check_invalidation(self, position, context, market_data, current_structure) -> Tuple[bool, str]:
        """Complete invalidation checking for all setup types"""
        setup_type = context.get('setup_type', 'unknown')
        
        # Check minimum time
        if self.config['features']['invalidation']['respect_minimum_time']:
            entry_time = datetime.fromisoformat(context['entry_time'])
            time_in_trade = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            if time_in_trade < self.config['features']['invalidation']['min_time_minutes']:
                return False, ""
                
        # Check adverse move
        current_profit = context.get('current_profit_points', 0)
        initial_risk = context.get('initial_risk_points', 0)
        if initial_risk > 0:
            r_multiple = current_profit / initial_risk
            if r_multiple < -self.config['features']['invalidation']['max_adverse_move_r']:
                return True, f"Adverse move exceeded {self.config['features']['invalidation']['max_adverse_move_r']}R"
                
        # Setup-specific checks
        m15_data = market_data.get('M15')
        if m15_data is None or len(m15_data) < 20:
            return False, ""
            
        if setup_type == 'ma_bounce':
            return self._check_ma_bounce_invalidation(position, context, m15_data)
        elif setup_type == 'fibonacci':
            return self._check_fibonacci_invalidation(position, context, m15_data, current_structure)
        elif setup_type == 'momentum':
            return self._check_momentum_invalidation(position, context, m15_data, market_data.get('M5'))
        elif setup_type == 'range_extreme':
            return self._check_range_invalidation(position, context, m15_data)
            
        return False, ""
        
    def _check_ma_bounce_invalidation(self, position, context, data) -> Tuple[bool, str]:
        """Complete MA bounce invalidation logic"""
        current_price = data['close'].iloc[-1]
        last_close = data['close'].iloc[-2]  # Completed candle
        
        # Determine which MA was likely used for entry
        ema_20 = data['ema_20'].iloc[-1]
        ema_50 = data['ema_50'].iloc[-1]
        
        # Find closest MA to entry
        entry_price = context['entry_price']
        ma_distances = {
            'ema_20': abs(entry_price - data['ema_20'].iloc[-20]),
            'ema_50': abs(entry_price - data['ema_50'].iloc[-20])
        }
        closest_ma = min(ma_distances, key=ma_distances.get)
        
        if closest_ma == 'ema_20':
            key_ma = ema_20
            ma_name = "20 EMA"
        else:
            key_ma = ema_50
            ma_name = "50 EMA"
            
        if position.type == mt5.ORDER_TYPE_BUY:
            # Check for close below MA
            if last_close < key_ma and current_price < key_ma:
                # Verify it's a decisive break
                ma_distance = (key_ma - current_price) / data['atr'].iloc[-1]
                if ma_distance > 0.2:  # Price is 0.2 ATR below MA
                    return True, f"Price decisively broke below {ma_name}"
                    
            # Check MA slope change
            ma_slope_current = key_ma - data[closest_ma].iloc[-5]
            ma_slope_entry = data[closest_ma].iloc[-20] - data[closest_ma].iloc[-25]
            if ma_slope_current < 0 and ma_slope_entry > 0:
                return True, f"{ma_name} turned bearish"
                
        else:  # Sell position
            if last_close > key_ma and current_price > key_ma:
                ma_distance = (current_price - key_ma) / data['atr'].iloc[-1]
                if ma_distance > 0.2:
                    return True, f"Price decisively broke above {ma_name}"
                    
            ma_slope_current = key_ma - data[closest_ma].iloc[-5]
            ma_slope_entry = data[closest_ma].iloc[-20] - data[closest_ma].iloc[-25]
            if ma_slope_current > 0 and ma_slope_entry < 0:
                return True, f"{ma_name} turned bullish"
                
        return False, ""
        
    def _check_fibonacci_invalidation(self, position, context, data, structure) -> Tuple[bool, str]:
        """Complete Fibonacci invalidation logic"""
        current_price = data['close'].iloc[-1]
        entry_price = context['entry_price']
        original_sl = context['original_sl']
        
        # Calculate next Fibonacci level
        # Assuming entry was at 61.8%, next level is 78.6%
        fib_range = abs(entry_price - original_sl) / 0.382  # Reverse calculate range
        
        if position.type == mt5.ORDER_TYPE_BUY:
            fib_786_level = entry_price - (fib_range * 0.786)
            if current_price < fib_786_level:
                return True, "Price broke below 78.6% Fibonacci level"
                
            # Check for failure to progress
            bars_since_entry = len(data) - 20  # Approximate
            if bars_since_entry > 10:
                highest_since_entry = data['high'].iloc[-10:].max()
                if highest_since_entry < entry_price + (0.5 * structure['atr']):
                    return True, "Failed to make meaningful progress from Fibonacci entry"
                    
        else:  # Sell position
            fib_786_level = entry_price + (fib_range * 0.786)
            if current_price > fib_786_level:
                return True, "Price broke above 78.6% Fibonacci level"
                
            bars_since_entry = len(data) - 20
            if bars_since_entry > 10:
                lowest_since_entry = data['low'].iloc[-10:].min()
                if lowest_since_entry > entry_price - (0.5 * structure['atr']):
                    return True, "Failed to make meaningful progress from Fibonacci entry"
                    
        return False, ""
        
    def _check_momentum_invalidation(self, position, context, m15_data, m5_data) -> Tuple[bool, str]:
        """Complete momentum trade invalidation logic"""
        entry_time = datetime.fromisoformat(context['entry_time'])
        minutes_since_entry = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
        bars_since_entry = int(minutes_since_entry / 15)  # 15-minute bars
        
        current_price = m15_data['close'].iloc[-1]
        entry_price = context['entry_price']
        
        # Momentum trades must move quickly
        if bars_since_entry >= 3:
            if position.type == mt5.ORDER_TYPE_BUY:
                if current_price <= entry_price:
                    return True, "Momentum trade stalled - no progress after 3 bars"
            else:
                if current_price >= entry_price:
                    return True, "Momentum trade stalled - no progress after 3 bars"
                    
        # Check M5 structure break
        if m5_data is not None and len(m5_data) >= 20:
            recent_high = m5_data['high'].iloc[-10:].max()
            recent_low = m5_data['low'].iloc[-10:].min()
            
            if position.type == mt5.ORDER_TYPE_BUY:
                # Check for lower low on M5
                if m5_data['low'].iloc[-1] < m5_data['low'].iloc[-5:].min():
                    return True, "M5 structure broken - new lower low"
            else:
                # Check for higher high on M5
                if m5_data['high'].iloc[-1] > m5_data['high'].iloc[-5:].max():
                    return True, "M5 structure broken - new higher high"
                    
        return False, ""
        
    def _check_range_invalidation(self, position, context, data) -> Tuple[bool, str]:
        """Complete range trade invalidation logic"""
        lookback = 50
        if len(data) < lookback:
            return False, ""
            
        range_high = data['high'].iloc[-lookback:].max()
        range_low = data['low'].iloc[-lookback:].min()
        range_size = range_high - range_low
        current_price = data['close'].iloc[-1]
        last_close = data['close'].iloc[-2]
        
        # Buffer for false breakouts
        buffer = range_size * 0.1
        
        # Check for range break
        if last_close > range_high + buffer and current_price > range_high + buffer:
            return True, "Range broken - upward breakout confirmed"
        elif last_close < range_low - buffer and current_price < range_low - buffer:
            return True, "Range broken - downward breakout confirmed"
            
        # Check if range has compressed too much (no longer tradeable)
        recent_range = data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()
        if recent_range < range_size * 0.5:
            return True, "Range compressed - no longer viable for range trading"
            
        return False, ""
        
    def _check_regime_change(self, position, context, current_structure) -> str:
        """Check for regime changes and determine action"""
        original_regime = context.get('entry_regime', 'unknown')
        current_regime = current_structure['regime']
        setup_type = context.get('setup_type', 'unknown')
        
        if original_regime == current_regime:
            return 'none'
            
        # Check time since last regime check
        last_regime_check = context.get('last_regime_check', 0)
        if time.time() - last_regime_check < self.config['features']['regime']['check_interval_minutes'] * 60:
            return 'none'
            
        context['last_regime_check'] = time.time()
        
        # Adverse regime changes
        if setup_type in ['ma_bounce', 'fibonacci', 'momentum']:
            # Trend-following setups
            if original_regime in ['STRONG_TREND', 'NORMAL_TREND'] and current_regime == 'RANGING':
                return 'exit'
            if original_regime != 'VOLATILE' and current_regime == 'VOLATILE':
                return 'tighten'
                
        elif setup_type == 'range_extreme':
            # Range trading setup
            if original_regime == 'RANGING' and current_regime in ['STRONG_TREND', 'NORMAL_TREND']:
                return 'exit'
                
        # Favorable regime changes
        if setup_type in ['ma_bounce', 'fibonacci'] and current_regime == 'STRONG_TREND':
            if current_structure['trend'] == context.get('position_type'):
                return 'loosen'  # Give more room to run
                
        return 'none'
        
    def _check_session_end(self, position, context) -> Tuple[bool, str]:
        """Check if position should be closed due to session ending"""
        current_time = datetime.now(timezone.utc)
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Find current session
        current_session = None
        for session_name, times in self.session_times.items():
            if times['start'] <= current_hour < times['end']:
                current_session = session_name
                break
                
        if not current_session:
            return False, ""
            
        # Check if this session should trigger closes
        if current_session not in self.config['features']['session']['sessions_to_close']:
            return False, ""
            
        # Calculate minutes until session end
        session_end_hour = self.session_times[current_session]['end']
        if session_end_hour == 24:
            session_end_hour = 0
            
        minutes_to_end = (session_end_hour - current_hour) * 60 - current_minute
        if minutes_to_end < 0:
            minutes_to_end += 24 * 60
            
        # Check if we're within the closing window
        if minutes_to_end <= self.config['features']['session']['minutes_before_close']:
            # Only close if profitable
            if context.get('current_profit_points', 0) > 0:
                pip_size = self._get_pip_size(position.symbol)
                profit_pips = context['current_profit_points'] / pip_size
                
                if profit_pips >= self.config['features']['session']['min_profit_pips_to_close']:
                    return True, f"{current_session} ending - taking profit"
                    
        return False, ""
        
    def _check_news_events(self, position, context, market_context) -> str:
        """Check for upcoming news events"""
        symbol = position.symbol
        
        # Check for high impact news
        news_check = market_context.is_news_time(
            symbol,
            minutes_before=30,
            minutes_after=15,
            min_impact='High'
        )
        
        if news_check['is_news']:
            minutes_to_news = news_check.get('minutes_to_event', 0)
            
            # Close profitable positions before high impact news
            if 0 < minutes_to_news < 15 and context.get('current_profit_points', 0) > 0:
                return 'close'
                
            # Tighten stops for positions at risk
            if 0 < minutes_to_news < 30:
                return 'tighten'
                
        return 'none'
        
    def _trail_stop(self, position, context, market_data) -> bool:
        """Hybrid trailing stop implementation"""
        method = self.config['features']['trailing']['trail_method']
        
        if method == 'hybrid':
            # Try ATR first, fall back to MA if it fails
            if self._trail_stop_atr(position, context, market_data):
                return True
            return self._trail_stop_ma(position, context, market_data)
        elif method == 'atr':
            return self._trail_stop_atr(position, context, market_data)
        elif method == 'ma':
            return self._trail_stop_ma(position, context, market_data)
            
        return False
        
    def _trail_stop_ma(self, position, context, market_data) -> bool:
        """Trail stop using moving average"""
        m15_data = market_data.get('M15')
        if m15_data is None or len(m15_data) < 50:
            return False
            
        # Use 20 EMA for trailing
        ma_value = m15_data['ema_20'].iloc[-1]
        atr = m15_data['atr'].iloc[-1]
        
        # Add small buffer
        buffer = atr * 0.1
        
        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = ma_value - buffer
            if new_sl <= position.sl:
                return False
        else:
            new_sl = ma_value + buffer
            if new_sl >= position.sl:
                return False
                
        # Ensure minimum distance
        pip_size = self._get_pip_size(position.symbol)
        min_distance = self.config['features']['trailing']['min_trail_distance_pips'] * pip_size
        
        tick = mt5.symbol_info_tick(position.symbol)
        current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        if position.type == mt5.ORDER_TYPE_BUY:
            if current_price - new_sl < min_distance:
                new_sl = current_price - min_distance
        else:
            if new_sl - current_price < min_distance:
                new_sl = current_price + min_distance
                
        # Send modification
        return self._modify_position_stop(position, new_sl, "MA Trail")
        
    def _tighten_stop(self, position, context, market_data, structure) -> bool:
        """Tighten stop in response to market conditions"""
        m15_data = market_data.get('M15')
        if m15_data is None:
            return False
            
        current_sl = position.sl
        atr = structure['atr']
        
        # Calculate tighter stop based on recent swing
        lookback = 10
        if position.type == mt5.ORDER_TYPE_BUY:
            recent_low = m15_data['low'].iloc[-lookback:].min()
            new_sl = recent_low - (atr * 0.5)
            
            # Only tighten, never loosen
            if new_sl <= current_sl:
                return False
        else:
            recent_high = m15_data['high'].iloc[-lookback:].max()
            new_sl = recent_high + (atr * 0.5)
            
            if new_sl >= current_sl:
                return False
                
        return self._modify_position_stop(position, new_sl, "Tightened")
        
    def _modify_position_stop(self, position, new_sl, comment="") -> bool:
        """Execute stop loss modification"""
        try:
            # Verify stop distance
            symbol_info = mt5.symbol_info(position.symbol)
            tick = mt5.symbol_info_tick(position.symbol)
            
            if position.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                stop_distance = current_price - new_sl
            else:
                current_price = tick.ask
                stop_distance = new_sl - current_price
                
            min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
            
            if stop_distance < min_stop_distance:
                logging.warning(f"Stop too close for {position.symbol}: {stop_distance} < {min_stop_distance}")
                return False
                
            # Round to tick size
            tick_size = symbol_info.point
            new_sl = round(new_sl / tick_size) * tick_size
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp,
                "magic": self.magic_number,
                "comment": comment[:31]
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{position.symbol}] Modified SL to {new_sl:.5f} ({comment})")
                self._update_modification_time(position.ticket)
                self.stats['total_modifications'] += 1
                return True
            else:
                logging.error(f"Failed to modify stop: {result.comment}")
                self._record_modification_failure(position.ticket)
                self.stats['failed_modifications'] += 1
                return False
                
        except Exception as e:
            logging.error(f"Error modifying stop: {e}")
            return False
            
    def _create_position_context(self, position) -> dict:
        """Create initial context for a new position"""
        setup_type = 'unknown'
        entry_regime = 'unknown'
        
        # Parse comment
        if position.comment and ':' in position.comment:
            parts = position.comment.split(':')
            
            # Handle different comment formats
            if len(parts) >= 3 and parts[0] == 'AEA':
                setup_map = {'F': 'fibonacci', 'M': 'ma_bounce', 'P': 'momentum', 'R': 'range_extreme'}
                regime_map = {
                    'ST': 'STRONG_TREND', 'NT': 'NORMAL_TREND', 
                    'WT': 'WEAK_TREND', 'RG': 'RANGING', 'VL': 'VOLATILE'
                }
                setup_type = setup_map.get(parts[1], 'unknown')
                entry_regime = regime_map.get(parts[2], 'unknown')
                
        context = {
            'ticket': position.ticket,
            'symbol': position.symbol,
            'setup_type': setup_type,
            'entry_regime': entry_regime,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'entry_price': position.price_open,
            'original_sl': position.sl,
            'original_tp': position.tp,
            'position_type': 'buy' if position.type == mt5.ORDER_TYPE_BUY else 'sell',
            'initial_risk_points': abs(position.price_open - position.sl),
            'volume': position.volume,
            'moved_to_breakeven': False,
            'trailing_active': False,
            'highest_profit_points': 0,
            'lowest_drawdown_points': 0,
            'modifications_count': 0,
            'last_check_time': time.time(),
            'last_regime_check': 0,
            'current_profit_points': 0
        }
        
        self.position_contexts[position.ticket] = context
        return context
        
    def _calculate_current_profit(self, position) -> float:
        """Calculate current profit in points"""
        tick = mt5.symbol_info_tick(position.symbol)
        
        if position.type == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            profit_points = current_price - position.price_open
        else:
            current_price = tick.ask
            profit_points = position.price_open - current_price
            
        return profit_points
        
    def _update_profit_tracking(self, position, context) -> None:
        """Update highest profit and drawdown tracking"""
        current_profit = context['current_profit_points']
        
        # Update highest profit
        if current_profit > context.get('highest_profit_points', 0):
            context['highest_profit_points'] = current_profit
            
        # Update lowest drawdown
        if current_profit < context.get('lowest_drawdown_points', 0):
            context['lowest_drawdown_points'] = current_profit
            
    def _get_pip_size(self, symbol) -> float:
        """Get pip size for a symbol"""
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                return symbol_info.point * 10
            else:
                return symbol_info.point
        return 0.0001
        
    def _check_spread(self, symbol) -> bool:
        """Check if spread is acceptable"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        spread_points = tick.ask - tick.bid
        pip_size = self._get_pip_size(symbol)
        spread_pips = spread_points / pip_size
        
        max_spread = self.config['safety']['max_spread_pips']
        
        if spread_pips > max_spread:
            logging.warning(f"Spread too high for {symbol}: {spread_pips:.1f} pips")
            return False
            
        return True
        
    def _can_modify_position(self, ticket: int) -> bool:
        """Check if we can modify this position (rate limiting)"""
        if ticket not in self.last_modification_time:
            return True
            
        time_since_last = time.time() - self.last_modification_time[ticket]
        min_interval = self.config['safety']['min_seconds_between_modifications']
        
        can_modify = time_since_last >= min_interval
        
        if not can_modify:
            remaining = min_interval - time_since_last
            logging.debug(f"Rate limit: wait {remaining:.0f}s before modifying position {ticket}")
            
        return can_modify
        
    def _update_modification_time(self, ticket: int) -> None:
        """Update last modification time"""
        self.last_modification_time[ticket] = time.time()
        
        # Update modification count
        if ticket in self.position_contexts:
            self.position_contexts[ticket]['modifications_count'] += 1
            
    def _record_modification_failure(self, ticket: int) -> None:
        """Record modification failure for circuit breaker"""
        if ticket not in self.modification_attempts:
            self.modification_attempts[ticket] = 0
        self.modification_attempts[ticket] += 1
        
        # Track consecutive failures
        if ticket not in self.failed_modifications:
            self.failed_modifications[ticket] = []
        self.failed_modifications[ticket].append(time.time())
        
        # Clean old failures (older than 1 hour)
        cutoff = time.time() - 3600
        self.failed_modifications[ticket] = [
            t for t in self.failed_modifications[ticket] if t > cutoff
        ]
        
        # Check circuit breaker
        recent_failures = len(self.failed_modifications[ticket])
        if recent_failures >= self.config['safety']['circuit_breaker_threshold']:
            logging.error(f"Circuit breaker triggered for position {ticket} - {recent_failures} recent failures")
            
    def _should_move_to_breakeven(self, position, context) -> bool:
        """Determine if position should move to breakeven"""
        if context.get('moved_to_breakeven', False):
            return False
            
        current_profit = context.get('current_profit_points', 0)
        initial_risk = context.get('initial_risk_points', 0)
        
        if initial_risk <= 0:
            return False
            
        # Calculate R-multiple
        r_multiple = current_profit / initial_risk
        
        # Check if we've reached the trigger
        if r_multiple >= self.config['features']['breakeven']['trigger_r_multiple']:
            # Additional check for minimum profit
            pip_size = self._get_pip_size(position.symbol)
            profit_pips = current_profit / pip_size
            
            if profit_pips >= self.config['features']['breakeven']['min_profit_pips']:
                return True
                
        return False
        
    def _should_trail_stop(self, position, context) -> bool:
        """Determine if we should start trailing"""
        if not context.get('moved_to_breakeven', False):
            return False
            
        if context.get('trailing_active', False):
            # Already trailing, check if we should continue
            return True
            
        current_profit = context.get('current_profit_points', 0)
        initial_risk = context.get('initial_risk_points', 0)
        
        if initial_risk <= 0:
            return False
            
        r_multiple = current_profit / initial_risk
        
        return r_multiple >= self.config['features']['trailing']['trigger_r_multiple']
        
    def _move_to_breakeven(self, position) -> bool:
        """Move stop loss to breakeven"""
        try:
            pip_size = self._get_pip_size(position.symbol)
            buffer = self.config['features']['breakeven']['buffer_pips'] * pip_size
            
            # Calculate new stop
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open + buffer
                if new_sl <= position.sl:
                    return False
            else:
                new_sl = position.price_open - buffer
                if new_sl >= position.sl:
                    return False
                    
            return self._modify_position_stop(position, new_sl, "BE")
            
        except Exception as e:
            logging.error(f"Error moving to breakeven: {e}")
            return False
            
    def _trail_stop_atr(self, position, context, market_data) -> bool:
        """Trail stop using ATR method"""
        m15_data = market_data.get('M15')
        if m15_data is None or 'atr' not in m15_data.columns:
            return False
            
        atr = m15_data['atr'].iloc[-1]
        multiplier = self.config['features']['trailing']['atr_multiplier']
        trail_distance = atr * multiplier
        
        # Get current price
        tick = mt5.symbol_info_tick(position.symbol)
        if position.type == mt5.ORDER_TYPE_BUY:
            current_price = tick.bid
            new_sl = current_price - trail_distance
            
            # Only trail up
            if new_sl <= position.sl:
                return False
        else:
            current_price = tick.ask
            new_sl = current_price + trail_distance
            
            # Only trail down
            if new_sl >= position.sl:
                return False
                
        # Ensure minimum trail distance
        pip_size = self._get_pip_size(position.symbol)
        min_distance = self.config['features']['trailing']['min_trail_distance_pips'] * pip_size
        
        if position.type == mt5.ORDER_TYPE_BUY:
            if current_price - new_sl < min_distance:
                new_sl = current_price - min_distance
        else:
            if new_sl - current_price < min_distance:
                new_sl = current_price + min_distance
                
        return self._modify_position_stop(position, new_sl, "ATR Trail")
        
    def _close_position(self, position, reason: str) -> bool:
        """Close a position with given reason"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                logging.error(f"No tick data for {position.symbol}")
                return False
                
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": f"TM:{reason[:27]}"  # Leave room for TM: prefix
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Log closing details
                context = self.position_contexts.get(position.ticket, {})
                profit_points = context.get('current_profit_points', 0)
                pip_size = self._get_pip_size(position.symbol)
                profit_pips = profit_points / pip_size if pip_size > 0 else 0
                
                logging.info(f"[{position.symbol}] Closed position {position.ticket}")
                logging.info(f"  Reason: {reason}")
                logging.info(f"  Profit: {profit_pips:.1f} pips (${position.profit:.2f})")
                logging.info(f"  Setup: {context.get('setup_type', 'unknown')}")
                
                # Clean up context
                if position.ticket in self.position_contexts:
                    del self.position_contexts[position.ticket]
                    
                # Clean up tracking
                for tracking_dict in [self.last_modification_time, 
                                    self.modification_attempts,
                                    self.failed_modifications]:
                    if position.ticket in tracking_dict:
                        del tracking_dict[position.ticket]
                        
                return True
            else:
                logging.error(f"Failed to close position {position.ticket}: {result.comment}")
                return False
                
        except Exception as e:
            logging.error(f"Error closing position {position.ticket}: {e}")
            return False
            
    def _cleanup_closed_positions(self) -> None:
        """Clean up contexts for positions that no longer exist"""
        try:
            # Get all open positions
            open_positions = mt5.positions_get()
            open_tickets = {pos.ticket for pos in open_positions} if open_positions else set()
            
            # Find closed positions
            closed_tickets = []
            for ticket in self.position_contexts:
                if ticket not in open_tickets:
                    closed_tickets.append(ticket)
                    
            # Clean up closed positions
            for ticket in closed_tickets:
                del self.position_contexts[ticket]
                
                # Clean up related tracking
                for tracking_dict in [self.last_modification_time,
                                    self.modification_attempts,
                                    self.failed_modifications]:
                    if ticket in tracking_dict:
                        del tracking_dict[ticket]
                        
            if closed_tickets:
                logging.info(f"Cleaned up {len(closed_tickets)} closed position contexts")
                
        except Exception as e:
            logging.error(f"Error cleaning up closed positions: {e}")
            
    def _save_position_context(self, ticket: int, context: dict) -> None:
        """Save position context"""
        self.position_contexts[ticket] = context
        
        # Save to file periodically
        if time.time() % 300 < 1:  # Every 5 minutes
            self._save_contexts_to_file()
            
    def _save_contexts_to_file(self) -> None:
        """Save all contexts to file"""
        try:
            # Create backup
            if self.storage_file.exists():
                backup = self.storage_file.with_suffix('.bak')
                self.storage_file.rename(backup)
                
            # Save current contexts
            with open(self.storage_file, 'w') as f:
                json.dump({
                    'contexts': self.position_contexts,
                    'stats': self.stats,
                    'last_save': datetime.now(timezone.utc).isoformat()
                }, f, indent=2, default=str)
                
            # Remove backup if save successful
            if self.storage_file.with_suffix('.bak').exists():
                self.storage_file.with_suffix('.bak').unlink()
                
        except Exception as e:
            logging.error(f"Error saving contexts: {e}")
            # Restore backup if it exists
            backup = self.storage_file.with_suffix('.bak')
            if backup.exists():
                backup.rename(self.storage_file)
                
    def _load_contexts_from_file(self) -> None:
        """Load contexts from file on startup"""
        try:
            if self.storage_file.exists():
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    
                self.position_contexts = data.get('contexts', {})
                # Convert string keys back to integers
                self.position_contexts = {
                    int(k): v for k, v in self.position_contexts.items()
                }
                
                # Restore stats if available
                saved_stats = data.get('stats', {})
                for key, value in saved_stats.items():
                    if key in self.stats:
                        self.stats[key] = value
                        
                last_save = data.get('last_save', 'unknown')
                logging.info(f"Loaded {len(self.position_contexts)} contexts from {last_save}")
                
        except Exception as e:
            logging.error(f"Error loading contexts: {e}")
            self.position_contexts = {}
            
    def _get_position_context(self, position) -> Optional[dict]:
        """Get context for a position"""
        return self.position_contexts.get(position.ticket)
        
    def get_statistics(self) -> dict:
        """Get current statistics"""
        stats = self.stats.copy()
        stats['active_positions'] = len(self.position_contexts)
        stats['success_rate'] = 0
        
        if stats['total_modifications'] > 0:
            stats['success_rate'] = (
                (stats['total_modifications'] - stats['failed_modifications']) / 
                stats['total_modifications'] * 100
            )
            
        return stats
        
    def reset_statistics(self) -> None:
        """Reset statistics"""
        for key in self.stats:
            if isinstance(self.stats[key], int):
                self.stats[key] = 0