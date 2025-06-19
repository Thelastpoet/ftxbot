"""
Trade Exit Manager Module
Handles dynamic exit strategies, trailing stops, and partial profit taking
Integrates with existing market analysis components
"""
import logging
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import traceback
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from technical_indicators import IndicatorCalculator

@dataclass
class ExitSignal:
    """Represents an exit signal with reasoning"""
    action: str  # 'full_exit', 'partial_exit', 'trail_stop', 'move_to_breakeven'
    reason: str
    urgency: str  # 'immediate', 'normal', 'monitor'
    comment_tag: str
    new_stop_loss: Optional[float] = None
    exit_percent: Optional[float] = None  # For partial exits
    
    
@dataclass
class PositionContext:
    """Enhanced position information with entry context"""
    position: object  # MT5 position object
    entry_setup_type: str
    entry_market_regime: str
    entry_amd_phase: str
    entry_time: datetime
    highest_profit: float
    lowest_profit: float
    last_trail_update: datetime
    partial_exits_done: int
    exit_strategy: str = 'full_partial'  # Default strategy
    

class TradeExitManager:
    """
    Manages all aspects of trade exits including:
    - Dynamic exit conditions based on market structure changes
    - Intelligent trailing stops
    - Partial profit taking
    - Time-based exits
    - Session-aware exit strategies
    """
    
    def __init__(self, market_context, correlation_manager=None):
        self.market_context = market_context
        self.correlation_manager = correlation_manager
        
        # Position tracking
        self.position_contexts = {}  # ticket -> PositionContext
        
        # Configuration for different strategies
        self.strategies = {
            'full_partial': {
                'min_profit_for_breakeven': 1.0,
                'trail_start_profit': 1.5,
                'partial_exit_levels': [1.0, 2.0, 3.0],
                'partial_exit_percents': [0.3, 0.3, 0.4],
            },
            'reduced_partial': {
                'min_profit_for_breakeven': 0.8,
                'trail_start_profit': 1.2,
                'partial_exit_levels': [1.0, 2.5],
                'partial_exit_percents': [0.5, 0.5],
            },
            'simple_partial': {
                'min_profit_for_breakeven': 0.7,
                'trail_start_profit': 1.0,
                'partial_exit_levels': [1.0],
                'partial_exit_percents': [0.5],
            },
            'aggressive_trail': {
                'min_profit_for_breakeven': 0.5,  # Move to BE early
                'trail_start_profit': 0.8,  # Start trailing early
                'partial_exit_levels': [],  # No partials
                'partial_exit_percents': [],
            }
        }
        
        # Default config
        self.config = {
            'max_trade_duration': 24,  # Hours
            'session_exit_buffer': 30,  # Minutes before session end
        }
        
        # Performance tracking
        self.exit_stats = defaultdict(lambda: {'count': 0, 'total_r': 0})
        
        logging.info("TradeExitManager initialized")
        
    def _initialize_position_with_context(self, position, entry_context):
        """Initialize position with full entry context"""
        
        # Determine exit strategy based on position size and market conditions
        strategy, strategy_name = self._get_position_strategy(position)
        
        context = PositionContext(
            position=position,
            entry_setup_type=entry_context['setup']['type'],
            entry_market_regime=entry_context['regime'],
            entry_amd_phase=entry_context['amd_phase'],
            entry_time=entry_context['entry_time'],
            highest_profit=0,
            lowest_profit=0,
            last_trail_update=datetime.now(timezone.utc),
            partial_exits_done=0,
            exit_strategy=strategy_name
        )
        
        # Store additional context that might be useful
        context.entry_score = entry_context['setup']['score']
        context.entry_session = entry_context['session']
        context.entry_confluence = entry_context['setup'].get('confluence_notes', [])
        
        self.position_contexts[position.ticket] = context
        
        logging.info(f"Exit manager initialized for position {position.ticket}: "
                    f"{context.entry_setup_type} in {context.entry_market_regime} "
                    f"using {strategy_name} exit strategy")
        
    def analyze_position(self, position, market_data, timeframes) -> List[ExitSignal]:
        """
        Comprehensive position analysis for exit signals
        
        Args:
            position: MT5 position object
            market_data: Current market data dict {timeframe: DataFrame}
            timeframes: Tuple of timeframes to analyze
            
        Returns:
            List of ExitSignal objects with recommended actions
        """
        signals = []
        
        try:
            # Create IndicatorCalculator instance if not exists
            if not hasattr(self, 'indicator_calc'):
                self.indicator_calc = IndicatorCalculator()
            
            # Calculate indicators for each timeframe
            processed_data = {}
            for tf, df in market_data.items():
                if df is not None:
                    processed_data[tf] = self.indicator_calc.calculate_indicators(df)
                else:
                    logging.warning(f"No data for timeframe {tf}")
                    processed_data[tf] = None
            
        
            # Get or create position context
            if position.ticket not in self.position_contexts:
                self._initialize_position_context(position)
                
            context = self.position_contexts[position.ticket]
            
            # Update profit tracking
            current_profit_r = self._calculate_r_multiple(position)
            context.highest_profit = max(context.highest_profit, current_profit_r)
            context.lowest_profit = min(context.lowest_profit, current_profit_r)
            
            # Calculate time since entry for logging
            time_since_entry = (datetime.now(timezone.utc) - context.entry_time).total_seconds() / 60
            
            # Log position status for debugging
            logging.debug(f"[Exit Analysis] {position.symbol} ticket:{position.ticket} "
                        f"R:{current_profit_r:.2f} Age:{time_since_entry:.0f}min "
                        f"Setup:{context.entry_setup_type}")
            
            # 1. Check critical exit conditions (FIXED - much more conservative)
            critical_signals = self._check_critical_exits(position, processed_data, context)
            signals.extend(critical_signals)
            
            # Log if critical signals found
            for sig in critical_signals:
                logging.info(f"[Exit] CRITICAL signal for {position.symbol}: {sig.reason}")
            
            # 2. Check for profit taking opportunities (UNCHANGED - this is good)
            profit_signals = self._check_profit_targets(position, context, current_profit_r)
            signals.extend(profit_signals)
            
            # 3. Check for trailing stop updates (UNCHANGED - this is good)
            trail_signal = self._check_trailing_stop(position, processed_data, context, current_profit_r)
            if trail_signal:
                signals.append(trail_signal)
                
            # 4. Check time-based exits (REFINED)
            time_signals = self._check_time_exits(position, context)
            signals.extend(time_signals)
            
            # 5. Check correlation-based exits (UNCHANGED)
            if self.correlation_manager:
                corr_signals = self._check_correlation_exits(position)
                signals.extend(corr_signals)
                
            return signals
        
        except Exception as e:
            logging.error(f"Error in analyze_position: {e}")
            logging.error(traceback.format_exc())
            return signals
        
    def _check_critical_exits(self, position, market_data, context) -> List[ExitSignal]:
        """
        Check for immediate exit conditions - FIXED VERSION
        
        Philosophy: Only intervene in truly EXCEPTIONAL circumstances.
        Let the stop loss handle normal adverse price movement.
        """
        signals = []
        
        current_price = mt5.symbol_info_tick(position.symbol).bid if position.type == 0 else mt5.symbol_info_tick(position.symbol).ask
        current_profit_r = self._calculate_r_multiple(position)
        
        # Calculate time since position opened
        time_since_entry = (datetime.now(timezone.utc) - context.entry_time).total_seconds() / 60  # minutes
        
        # === TRULY CRITICAL CONDITIONS (EXTRAORDINARY CIRCUMSTANCES) ===
        
        # 1. High impact news approaching (genuinely critical - can cause gaps)
        news_warning = self._check_upcoming_news(position)
        if news_warning and news_warning <= 5:  # Only within 5 minutes
            signals.append(ExitSignal(
                action='full_exit',
                reason=f"High impact news in {news_warning} minutes",
                urgency='immediate',
                comment_tag='NewsExit'
            ))
            
        # 2. Major AMD phase conflict (only for AMD-specific strategies)
        if context.entry_setup_type in ['manipulation', 'amd_reversal']:  # Only if it was an AMD-based trade
            amd_conflict = self._check_amd_conflict(position, context)
            if amd_conflict:
                signals.append(ExitSignal(
                    action='full_exit',
                    reason=f"AMD strategy invalidated: {amd_conflict}",
                    urgency='immediate',
                    comment_tag='AMD_Invalid'
                ))
        
        # === PROFIT PROTECTION CONDITIONS (ONLY WHEN IN MEANINGFUL PROFIT) ===
        
        # 3. Structure break - BUT ONLY if we're protecting profits or position is mature
        if current_profit_r > 0.5 or time_since_entry > 120:  # Only if 0.5R+ profit OR 2+ hours old
            structure_break = self._detect_structure_break(
                position, market_data[mt5.TIMEFRAME_M15], current_price, current_profit_r
            )
            if structure_break:
                # If in profit, this is profit protection
                # If old and losing, the original thesis may be wrong
                urgency = 'immediate' if current_profit_r < -0.5 else 'normal'
                signals.append(ExitSignal(
                    action='full_exit',
                    reason=f"Structure break: {structure_break}",
                    urgency=urgency,
                    comment_tag='StructBreak'
                ))
                
        # 4. Momentum exhaustion - ONLY when protecting profits
        if current_profit_r > 0.3:  # Only if we have profits to protect
            if self._detect_momentum_exhaustion(position, market_data[mt5.TIMEFRAME_M15]):
                signals.append(ExitSignal(
                    action='partial_exit',
                    reason="Momentum exhaustion - protecting profits",
                    urgency='normal',
                    comment_tag='MomentumExhaust',
                    exit_percent=0.5
                ))
        
        # 5. Session end for session-specific trades
        if self._is_session_specific_trade(context, self.market_context.get_trading_session()):
            session_info = self.market_context.get_trading_session()
            minutes_to_end = self._get_minutes_to_session_end(datetime.now(timezone.utc), session_info)
            if minutes_to_end and minutes_to_end < 30:
                signals.append(ExitSignal(
                    action='full_exit',
                    reason=f"Session-specific trade: {session_info['name']} ending in {minutes_to_end}min",
                    urgency='normal',
                    comment_tag='SessionEnd'
                ))
                
        return signals
    
    def _detect_structure_break(self, position, df, current_price, current_profit_r) -> Optional[str]:
        """
        Conservative structure break detection - only trigger when it makes sense
        """
        if df is None or len(df) < 30:
            return None

        # Get the last CLOSED candle for confirmation
        last_closed_candle = df.iloc[-2]
        
        # Make buffer more intelligent based on current profit
        atr = df['atr'].iloc[-1]
        if pd.isna(atr):
            return None
            
        # If we're in profit, be more sensitive to structure breaks (protecting profits)
        # If we're at a loss, be less sensitive (let stop loss handle it)
        if current_profit_r > 0:
            buffer = atr * 0.1  # Tighter for profit protection
        else:
            buffer = atr * 0.3  # Looser for losing positions
        
        # Find significant swing points with more conservative parameters
        recent_highs, recent_lows = self._find_swing_points(df, lookback=50, window=8)  # Longer lookback, wider window

        if position.type == 0:  # Buy position
            if not recent_lows:
                return None
            
            # Only use the most significant support (highest recent low)
            significant_low = max(recent_lows)
            
            # Only trigger if it's a CLEAR break, not just a minor breach
            if last_closed_candle['close'] < (significant_low - buffer):
                # Additional check: ensure this isn't just noise
                breach_size = (significant_low - last_closed_candle['close']) / atr
                if breach_size > 0.2:  # At least 20% of ATR breach
                    return f"Clear break below support {significant_low:.5f}"
                    
        else:  # Sell position
            if not recent_highs:
                return None
                
            # Only use the most significant resistance (lowest recent high)
            significant_high = min(recent_highs)
            
            if last_closed_candle['close'] > (significant_high + buffer):
                breach_size = (last_closed_candle['close'] - significant_high) / atr
                if breach_size > 0.2:
                    return f"Clear break above resistance {significant_high:.5f}"
                    
        return None
        
    def _get_position_strategy(self, position) -> dict:
        """Determine which exit strategy to use based on position size"""
        symbol_info = mt5.symbol_info(position.symbol)
        min_volume = symbol_info.volume_min
        
        # Determine strategy based on position size
        if position.volume >= min_volume * 10:
            strategy_name = 'full_partial'
        elif position.volume >= min_volume * 5:
            strategy_name = 'reduced_partial'
        elif position.volume >= min_volume * 2:
            strategy_name = 'simple_partial'
        else:
            strategy_name = 'aggressive_trail'
            
        return self.strategies[strategy_name], strategy_name
    
    def _check_profit_targets(self, position, context, current_profit_r) -> List[ExitSignal]:
        """Check for partial profit taking opportunities"""
        signals = []
        
        # Get appropriate strategy for this position
        strategy, strategy_name = self._get_position_strategy(position)
        
        # If no partial exits in strategy, handle differently
        if not strategy['partial_exit_levels']:
            # For small positions, just ensure we move to breakeven
            if current_profit_r >= strategy['min_profit_for_breakeven'] and position.sl < position.price_open:
                signals.append(ExitSignal(
                    action='move_to_breakeven',
                    reason=f"Small position ({strategy_name}): Moving to breakeven",
                    urgency='normal',
                    comment_tag='MoveToBE',
                    new_stop_loss=position.price_open
                ))
            return signals
        
        # Check if position is large enough for the planned partials
        symbol_info = mt5.symbol_info(position.symbol)
        min_volume = symbol_info.volume_min
        
        # Determine which partial exit level we're at
        for i, (level, percent) in enumerate(zip(
            strategy['partial_exit_levels'],
            strategy['partial_exit_percents']
        )):
            if current_profit_r >= level and context.partial_exits_done <= i:
                # Verify this partial is executable
                volume_to_close = position.volume * percent
                if volume_to_close >= min_volume:
                    signals.append(ExitSignal(
                        action='partial_exit',
                        reason=f"Reached {level}R profit target ({strategy_name})",
                        urgency='normal',
                        comment_tag=f'TP_{level}R',
                        exit_percent=percent
                    ))
                    context.partial_exits_done = i + 1
                    break
                
        # Move to breakeven after first partial
        if (context.partial_exits_done >= 1 and 
            position.sl != position.price_open and
            current_profit_r > 0):
            signals.append(ExitSignal(
                action='move_to_breakeven',
                reason="Moving stop to breakeven after partial exit",
                urgency='normal',
                comment_tag='MoveToBE',
                new_stop_loss=position.price_open
            ))
            
        return signals
        
    def _check_trailing_stop(self, position, market_data, context, current_profit_r) -> Optional[ExitSignal]:
        """Dynamic trailing stop management"""
        
        # Get appropriate strategy
        strategy, strategy_name = self._get_position_strategy(position)
        
        # Only trail if in sufficient profit
        if current_profit_r < strategy['trail_start_profit']:
            return None
            
        # Don't update too frequently
        if (datetime.now(timezone.utc) - context.last_trail_update).seconds < 300:  # 5 min
            return None
            
        # Calculate new stop based on method
        new_stop = self._calculate_trail_stop(position, market_data, context, strategy_name)
        
        if new_stop:
            # For buy positions, new stop must be higher than current
            # For sell positions, new stop must be lower than current
            if position.type == 0:  # Buy
                if new_stop > position.sl and new_stop < position.price_current:
                    context.last_trail_update = datetime.now(timezone.utc)
                    return ExitSignal(
                        action='trail_stop',
                        reason=f"Trailing stop to {new_stop:.5f} ({strategy_name})",
                        urgency='normal',
                        comment_tag='TrailStop',
                        new_stop_loss=new_stop
                    )
            else:  # Sell
                if new_stop < position.sl and new_stop > position.price_current:
                    context.last_trail_update = datetime.now(timezone.utc)
                    return ExitSignal(
                        action='trail_stop',
                        reason=f"Trailing stop to {new_stop:.5f} ({strategy_name})",
                        urgency='normal',
                        comment_tag='TrailStop',
                        new_stop_loss=new_stop
                    )
                    
        return None
        
    def _calculate_trail_stop(self, position, market_data, context, strategy_name) -> Optional[float]:
        """Calculate trailing stop level using multiple methods"""
        
        # Get relevant data
        df_m15 = market_data[mt5.TIMEFRAME_M15]
        df_h1 = market_data[mt5.TIMEFRAME_H1]
        
        # Method 1: ATR-based trailing
        atr = df_m15['atr'].iloc[-1]
        atr_stop = self._calculate_atr_trail(position, atr, context, strategy_name)
        
        # Method 2: Structure-based trailing
        structure_stop = self._calculate_structure_trail(position, df_m15)
        
        # Method 3: Moving average trailing
        ma_stop = self._calculate_ma_trail(position, df_h1)
        
        # Choose the most conservative (closest to price)
        if position.type == 0:  # Buy - highest stop
            stops = [s for s in [atr_stop, structure_stop, ma_stop] if s is not None]
            return max(stops) if stops else None
        else:  # Sell - lowest stop
            stops = [s for s in [atr_stop, structure_stop, ma_stop] if s is not None]
            return min(stops) if stops else None
            
    def _calculate_atr_trail(self, position, atr, context, strategy_name) -> Optional[float]:
        """ATR-based trailing stop"""
        # Base multiplier depends on strategy
        strategy_multipliers = {
            'aggressive_trail': 1.0,  # Tight trailing for small positions
            'simple_partial': 1.5,
            'reduced_partial': 1.8,
            'full_partial': 2.0
        }
        
        multiplier = strategy_multipliers.get(strategy_name, 2.0)
        
        # Tighten as profit increases
        profit_r = self._calculate_r_multiple(position)
        if profit_r > 3:
            multiplier *= 0.75  # Tighten by 25%
        elif profit_r > 5:
            multiplier *= 0.5   # Tighten by 50%
            
        if position.type == 0:  # Buy
            return position.price_current - (atr * multiplier)
        else:  # Sell
            return position.price_current + (atr * multiplier)
            
    def _calculate_structure_trail(self, position, df) -> Optional[float]:
        """Trail stop based on market structure"""
        lookback = 20
        
        if position.type == 0:  # Buy - trail below recent lows
            recent_lows = df['low'].iloc[-lookback:].rolling(5).min()
            structure_low = recent_lows.iloc[-1]
            if not pd.isna(structure_low):
                return structure_low - (df['atr'].iloc[-1] * 0.1)
        else:  # Sell - trail above recent highs
            recent_highs = df['high'].iloc[-lookback:].rolling(5).max()
            structure_high = recent_highs.iloc[-1]
            if not pd.isna(structure_high):
                return structure_high + (df['atr'].iloc[-1] * 0.1)
                
        return None
        
    def _calculate_ma_trail(self, position, df) -> Optional[float]:
        """Trail stop based on moving average"""
        if len(df) < 20:
            return None
            
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        
        if pd.isna(ma20):
            return None
            
        # Add buffer
        buffer = df['atr'].iloc[-1] * 0.2
        
        if position.type == 0:  # Buy
            return ma20 - buffer
        else:  # Sell
            return ma20 + buffer
            
    def _check_time_exits(self, position, context) -> List[ExitSignal]:
        """
        More conservative time-based exits
        """
        signals = []
        current_time = datetime.now(timezone.utc)
        
        # 1. Maximum trade duration - but only if position is stale and not working
        trade_duration = (current_time - context.entry_time).total_seconds() / 3600
        current_profit_r = self._calculate_r_multiple(position)
        
        # Only close on time if it's been a long time AND not working
        if trade_duration > 12 and current_profit_r < 0.2:  # 12 hours and barely profitable
            signals.append(ExitSignal(
                action='full_exit',
                reason=f"Stale position: {trade_duration:.1f}h old, not working",
                urgency='normal',
                comment_tag='StaleExit'
            ))
        elif trade_duration > 24:  # Hard limit at 24 hours regardless
            signals.append(ExitSignal(
                action='full_exit',
                reason=f"Maximum duration exceeded: {trade_duration:.1f}h",
                urgency='normal',
                comment_tag='TimeLimit'
            ))
            
        # 2. Weekend approaching - unchanged, this is genuinely important
        if current_time.weekday() == 4 and current_time.hour >= 20:  # Friday 8PM UTC
            signals.append(ExitSignal(
                action='full_exit',
                reason="Weekend approaching - closing position",
                urgency='normal',
                comment_tag='WeekendClose'
            ))
            
        return signals
    
    def _find_swing_points(self, df: pd.DataFrame, lookback=30, window=5) -> Tuple[list, list]:
        """Finds more robust swing points using a wider window for confirmation."""
        highs = []
        lows = []
        # Ensure we don't look past the end of the dataframe
        end_range = len(df) - window
        
        # Start looking from 'lookback' bars ago up to the confirmation window
        for i in range(len(df) - lookback, end_range):
            # To be a swing high, it must be the highest high in the surrounding 'window*2+1' bars
            is_swing_high = df['high'].iloc[i] == df['high'].iloc[i-window : i+window+1].max()
            if is_swing_high:
                highs.append(df['high'].iloc[i])
                
            # To be a swing low, it must be the lowest low in the surrounding 'window*2+1' bars
            is_swing_low = df['low'].iloc[i] == df['low'].iloc[i-window : i+window+1].min()
            if is_swing_low:
                lows.append(df['low'].iloc[i])
        
        return highs, lows
                
    def _detect_momentum_exhaustion(self, position, df) -> bool:
        """Detect if momentum is exhausting"""
        
        # Check RSI divergence
        rsi = df['rsi'].iloc[-10:]
        price = df['close'].iloc[-10:]
        
        if len(rsi) < 10:
            return False
            
        if position.type == 0:  # Buy
            # Bearish divergence: price making new highs but RSI isn't
            if (price.iloc[-1] > price.iloc[-5] and 
                rsi.iloc[-1] < rsi.iloc[-5] and 
                rsi.iloc[-1] < 70):
                return True
        else:  # Sell
            # Bullish divergence: price making new lows but RSI isn't
            if (price.iloc[-1] < price.iloc[-5] and 
                rsi.iloc[-1] > rsi.iloc[-5] and 
                rsi.iloc[-1] > 30):
                return True
                
        # Check for momentum candles drying up
        recent_candles = df.iloc[-5:]
        if position.type == 0:  # Buy
            bullish_candles = sum(1 for _, candle in recent_candles.iterrows() 
                                if candle['close'] > candle['open'])
            if bullish_candles < 2:  # Less than 40% bullish
                return True
        else:  # Sell
            bearish_candles = sum(1 for _, candle in recent_candles.iterrows() 
                                if candle['close'] < candle['open'])
            if bearish_candles < 2:  # Less than 40% bearish
                return True
                
        return False
        
    def _check_amd_conflict(self, position, context) -> Optional[str]:
        """Check if current AMD phase conflicts with position"""
        
        current_time = datetime.now(timezone.utc)
        amd_session = self.market_context.get_amd_session(current_time)
        
        # Get current bias based on AMD
        if amd_session['name'] == 'LONDON_SESSION':
            # During manipulation phase, positions against manipulation should exit
            if context.entry_amd_phase != 'MANIPULATION_CONFIRMED':
                return None  # Not a manipulation trade
                
        elif amd_session['name'] == 'NY_SESSION':
            # Distribution phase - trend following preferred
            if position.type == 0 and amd_session.get('bias') == 'sell':
                return "NY distribution phase favors sells"
            elif position.type == 1 and amd_session.get('bias') == 'buy':
                return "NY distribution phase favors buys"
                
        return None
        
    def _check_upcoming_news(self, position) -> Optional[int]:
        """
        Check for upcoming high-impact news - REFINED to only truly critical timeframes
        """
        news_check = self.market_context.is_news_time(
            position.symbol,
            minutes_before=10,  # Reduced from 30 to 10 - only immediate threats
            minutes_after=0,
            current_time=datetime.now(timezone.utc)
        )
        
        if news_check['is_news'] and news_check['impact'] == 'High':
            return news_check['minutes_to_event']
            
        return None
        
    def _check_correlation_exits(self, position) -> List[ExitSignal]:
        """Check for correlation-based exit signals"""
        signals = []
        
        if not self.correlation_manager:
            return signals
            
        # Get all open positions
        all_positions = mt5.positions_get()
        if not all_positions:
            return signals
            
        # Check if correlation regime has changed dramatically
        regime = self.correlation_manager.get_correlation_regime()
        
        if regime['regime'] == 'high_correlation':
            # In high correlation, consider reducing exposure
            correlated_positions = []
            
            for other_pos in all_positions:
                if other_pos.ticket == position.ticket:
                    continue
                    
                correlation = self.correlation_manager.get_correlation(
                    position.symbol, 
                    other_pos.symbol
                )
                
                if correlation and abs(correlation) > 0.8:
                    correlated_positions.append(other_pos.symbol)
                    
            if len(correlated_positions) >= 2:
                signals.append(ExitSignal(
                    action='partial_exit',
                    reason=f"High correlation with {len(correlated_positions)} other positions",
                    urgency='normal',
                    exit_percent=0.5
                ))
                
        return signals
        
    def execute_exit_signal(self, signal: ExitSignal, position) -> bool:
        """Execute the exit signal"""
        try:
            if signal.action == 'full_exit':
                return self._close_position(position, signal)
                
            elif signal.action == 'partial_exit':
                return self._partial_close_position(position, signal)
                
            elif signal.action == 'trail_stop':
                return self._modify_stop_loss(position, signal)
                
            elif signal.action == 'move_to_breakeven':
                return self._modify_stop_loss(position, signal)
            
            logging.warning(f"Unknown exit signal action: {signal.action}") 
            return False
            
        except Exception as e:
            logging.error(f"Error executing exit signal: {e}")
            return False
            
    def _close_position(self, position, signal: ExitSignal) -> bool:
        """Close entire position"""
        
        # Calculate final metrics
        profit_r = self._calculate_r_multiple(position)
                
        # Prepare close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "magic": position.magic,
            "comment": signal.comment_tag,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        
        # Check if the order_send call failed and returned None
        if result is None:
            logging.error(f"Failed to send close order for position {position.ticket}. order_send() returned None.")
            last_error = mt5.last_error()
            if last_error:
                logging.error(f"MT5 Last Error: {last_error}")
            return False
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Closed position {position.ticket}: {signal.reason} (Final R: {profit_r:.2f})")
            
            # Update statistics
            context = self.position_contexts.get(position.ticket)
            if context:
                self.exit_stats[signal.reason]['count'] += 1
                self.exit_stats[signal.reason]['total_r'] += profit_r
                
                # Clean up context
                del self.position_contexts[position.ticket]
                
            return True
        else:
            logging.error(f"Failed to close position {position.ticket}: {result.comment}")
            return False
            
    def _partial_close_position(self, position, signal: ExitSignal) -> bool:
        """Partially close position"""
        
        # Calculate volume to close
        volume_to_close = round(position.volume * signal.exit_percent, 2)
        
        # Ensure minimum volume
        symbol_info = mt5.symbol_info(position.symbol)
        if volume_to_close < symbol_info.volume_min:
            return False
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume_to_close,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "magic": position.magic,
            "comment": signal.comment_tag,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            logging.error(f"Failed to send partial close order for {position.ticket}. order_send() returned None.")
            last_error = mt5.last_error()
            if last_error:
                logging.error(f"MT5 Last Error: {last_error}")
            return False
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Partial close {signal.exit_percent*100:.0f}% of position {position.ticket}: {signal.reason}")
            return True
        else:
            logging.error(f"Failed to partial close position {position.ticket}: {result.comment}")
            return False
            
    def _modify_stop_loss(self, position, signal: ExitSignal) -> bool:
        """Modify position stop loss"""
                
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": signal.new_stop_loss,
            "tp": position.tp,
            "magic": position.magic,
            "comment": signal.comment_tag,
        }
        
        result = mt5.order_send(request)
        
        if result is None:
            logging.error(f"Failed to send SL modification for {position.ticket}. order_send() returned None.")
            last_error = mt5.last_error()
            if last_error:
                logging.error(f"MT5 Last Error: {last_error}")
            return False
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Modified SL for position {position.ticket} to {signal.new_stop_loss:.5f}: {signal.reason}")
            return True
        else:
            logging.error(f"Failed to modify SL for position {position.ticket}: {result.comment}")
            return False
            
    def _calculate_r_multiple(self, position) -> float:
        """Calculate current R-multiple of position"""
        
        # Get initial risk (SL distance)
        if position.type == 0:  # Buy
            initial_risk = position.price_open - position.sl
        else:  # Sell
            initial_risk = position.sl - position.price_open
            
        if initial_risk <= 0:
            return 0
            
        # Current profit in price terms
        if position.type == 0:  # Buy
            current_profit = position.price_current - position.price_open
        else:  # Sell
            current_profit = position.price_open - position.price_current
            
        return current_profit / initial_risk
        
    def _get_minutes_to_session_end(self, current_time: datetime, session_info: dict) -> Optional[int]:
        """Calculate minutes until current session ends"""
        
        session_name = session_info['name']
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Define session end times (UTC)
        session_ends = {
            'TOKYO_MAIN': 7,
            'LONDON_TOKYO_OVERLAP': 9,
            'LONDON_MAIN': 13,
            'LONDON_NY_OVERLAP': 16,
            'NY_MAIN': 22,
            'SYDNEY_OPEN': 24
        }
        
        end_hour = session_ends.get(session_name)
        if not end_hour:
            return None
            
        # Calculate minutes remaining
        if end_hour == 24:
            end_hour = 0
            
        if current_hour < end_hour:
            hours_remaining = end_hour - current_hour - 1
            minutes_remaining = 60 - current_minute + (hours_remaining * 60)
        else:
            return None  # Session already ended or about to end
            
        return minutes_remaining
        
    def _is_session_specific_trade(self, context: PositionContext, current_session: dict) -> bool:
        """
        Determine if trade is specific to current session - REFINED
        """
        # Only very specific trade types are session-dependent
        session_specific_setups = ['amd_manipulation', 'session_open', 'overlap_momentum']
        
        if context.entry_setup_type in session_specific_setups:
            return True
            
        # AMD trades during their specific phases
        if context.entry_amd_phase in ['MANIPULATION_CONFIRMED'] and current_session['name'] != 'LONDON_SESSION':
            return True
            
        return False
    
    def _initialize_position_context(self, position):
        """
        Initialize context for a new or recovered position.
        This now correctly parses the compact "BOT:XXX-YY-ZZZ" comment format
        and dynamically assigns the best exit strategy.
        """
        # Default values in case parsing fails
        entry_setup_type = 'unknown'
        entry_market_regime = 'unknown'
        entry_amd_phase = 'unknown'
        
        # Reverse mapping from short codes back to full names for context recovery
        setup_names = {
            'FIB': 'fibonacci', 'MOM': 'momentum', 'MAB': 'ma_bounce',
            'RNG': 'range_extreme', 'SHP': 'shallow_pullback', 'BRK': 'structure_break'
        }
        regime_names = {
            'ST': 'STRONG_TREND', 'NT': 'NORMAL_TREND', 'RG': 'RANGING',
            'TE': 'TREND_EXHAUSTION', 'VE': 'VOLATILE_EXPANSION'
        }
        amd_names = {
            'ACC': 'ACCUMULATION', 'AWM': 'AWAITING_MANIPULATION',
            'DST': 'DISTRIBUTION', 'NCS': 'NO_CLEAR_SETUP', 'CLS': 'SESSION_CLOSE'
        }

        # Try to parse the new, compact comment format for context recovery
        if position.comment and position.comment.startswith("BOT:"):
            try:
                # Remove the "BOT:" prefix and split by hyphen
                tags = position.comment[4:]
                parts = tags.split('-')
                
                if len(parts) == 3:
                    setup_tag, regime_tag, amd_tag = parts
                    # Look up the full name from the short code, providing a default if not found
                    entry_setup_type = setup_names.get(setup_tag, 'unknown_setup')
                    entry_market_regime = regime_names.get(regime_tag, 'unknown_regime')
                    entry_amd_phase = amd_names.get(amd_tag, 'unknown_amd')
                    logging.info(f"Recovered context for position {position.ticket} from comment: {entry_setup_type}, {entry_market_regime}, {entry_amd_phase}")
                else:
                    logging.warning(f"Could not parse comment for position {position.ticket}: '{position.comment}'")
            except Exception as e:
                logging.error(f"Error parsing comment for position {position.ticket}: {e}")

        # Determine the exit strategy to use for this position
        # This now uses the dedicated _get_position_strategy method for consistency
        strategy, strategy_name = self._get_position_strategy(position)
                
        # Create the final context object
        context = PositionContext(
            position=position,
            entry_setup_type=entry_setup_type,
            entry_market_regime=entry_market_regime,
            entry_amd_phase=entry_amd_phase,
            entry_time=datetime.fromtimestamp(position.time, tz=timezone.utc),
            highest_profit=0,
            lowest_profit=0,
            last_trail_update=datetime.now(timezone.utc),
            partial_exits_done=0,
            exit_strategy=strategy_name  # Use the determined strategy name
        )
        
        # Store the context object in our tracking dictionary
        self.position_contexts[position.ticket] = context
        
        logging.info(f"Exit manager context initialized for position {position.ticket} using '{strategy_name}' strategy.")
        
    def _initialize_position_context(self, position):
        """
        Initialize context for a new or recovered position.
        This now correctly parses the compact "BOT:XXX-YY-ZZZ" comment format
        and dynamically assigns the best exit strategy.
        """
        # Default values in case parsing fails
        entry_setup_type = 'unknown'
        entry_market_regime = 'unknown'
        entry_amd_phase = 'unknown'
        
        # Reverse mapping from short codes back to full names for context recovery
        setup_names = {
            'FIB': 'fibonacci', 'MOM': 'momentum', 'MAB': 'ma_bounce',
            'RNG': 'range_extreme', 'SHP': 'shallow_pullback', 'BRK': 'structure_break'
        }
        regime_names = {
            'ST': 'STRONG_TREND', 'NT': 'NORMAL_TREND', 'RG': 'RANGING',
            'TE': 'TREND_EXHAUSTION', 'VE': 'VOLATILE_EXPANSION'
        }
        amd_names = {
            'ACC': 'ACCUMULATION', 'AWM': 'AWAITING_MANIPULATION',
            'DST': 'DISTRIBUTION', 'NCS': 'NO_CLEAR_SETUP', 'CLS': 'SESSION_CLOSE'
        }

        # Try to parse the new, compact comment format for context recovery
        if position.comment and position.comment.startswith("BOT:"):
            try:
                # Remove the "BOT:" prefix and split by hyphen
                tags = position.comment[4:]
                parts = tags.split('-')
                
                if len(parts) == 3:
                    setup_tag, regime_tag, amd_tag = parts
                    # Look up the full name from the short code, providing a default if not found
                    entry_setup_type = setup_names.get(setup_tag, 'unknown_setup')
                    entry_market_regime = regime_names.get(regime_tag, 'unknown_regime')
                    entry_amd_phase = amd_names.get(amd_tag, 'unknown_amd')
                    logging.info(f"Recovered context for position {position.ticket} from comment: {entry_setup_type}, {entry_market_regime}, {entry_amd_phase}")
                else:
                    logging.warning(f"Could not parse comment for position {position.ticket}: '{position.comment}'")
            except Exception as e:
                logging.error(f"Error parsing comment for position {position.ticket}: {e}")

        # Determine the exit strategy to use for this position
        # This now uses the dedicated _get_position_strategy method for consistency
        strategy, strategy_name = self._get_position_strategy(position)
                
        # Create the final context object
        context = PositionContext(
            position=position,
            entry_setup_type=entry_setup_type,
            entry_market_regime=entry_market_regime,
            entry_amd_phase=entry_amd_phase,
            entry_time=datetime.fromtimestamp(position.time, tz=timezone.utc),
            highest_profit=0,
            lowest_profit=0,
            last_trail_update=datetime.now(timezone.utc),
            partial_exits_done=0,
            exit_strategy=strategy_name  # Use the determined strategy name
        )
        
        # Store the context object in our tracking dictionary
        self.position_contexts[position.ticket] = context
        
        logging.info(f"Exit manager context initialized for position {position.ticket} using '{strategy_name}' strategy.")
        
    def get_exit_statistics(self) -> dict:
        """Return exit performance statistics"""
        
        stats = {}
        for reason, data in self.exit_stats.items():
            if data['count'] > 0:
                avg_r = data['total_r'] / data['count']
                stats[reason] = {
                    'count': data['count'],
                    'avg_r_multiple': round(avg_r, 2),
                    'total_r': round(data['total_r'], 2)
                }
                
        return stats