"""
ICT Liquidity Detection Module
Implements proper buy-side/sell-side liquidity concepts based on ICT methodology
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LiquidityDetector:
    """
    Detects liquidity levels following ICT methodology.
    Focuses on institutional liquidity targeting concepts.
    """
    
    def __init__(self):
        pass
        
    def get_liquidity_levels(self, ohlc_df: pd.DataFrame, session_context: dict, 
                           daily_df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        """
        Get all liquidity levels following ICT methodology.
        Returns both buy-side and sell-side liquidity separately.
        """
        current_price = ohlc_df['close'].iloc[-1]
        
        liquidity_levels = {
            'buy_side': [],  # Above current price - targets for bearish moves
            'sell_side': []  # Below current price - targets for bullish moves
        }
        
        # 1. Session-based liquidity (highest priority)
        session_liquidity = self._get_session_liquidity(session_context, current_price)
        liquidity_levels['buy_side'].extend(session_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(session_liquidity['sell_side'])
        
        # 2. Equal highs/lows (very high priority)
        equal_levels = self._get_equal_highs_lows(ohlc_df, current_price)
        liquidity_levels['buy_side'].extend(equal_levels['buy_side'])
        liquidity_levels['sell_side'].extend(equal_levels['sell_side'])
        
        # 3. Recent swing highs/lows
        swing_liquidity = self._get_swing_liquidity(ohlc_df, current_price)
        liquidity_levels['buy_side'].extend(swing_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(swing_liquidity['sell_side'])
        
        # 4. Daily/Weekly levels (if available)
        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df, current_price)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])
        
        # 5. Psychological levels (round numbers)
        round_levels = self._get_round_number_liquidity(current_price)
        liquidity_levels['buy_side'].extend(round_levels['buy_side'])
        liquidity_levels['sell_side'].extend(round_levels['sell_side'])
        
        # Sort by distance from current price
        liquidity_levels['buy_side'] = sorted(liquidity_levels['buy_side'], 
                                            key=lambda x: x['level'])
        liquidity_levels['sell_side'] = sorted(liquidity_levels['sell_side'], 
                                             key=lambda x: x['level'], reverse=True)
        
        return liquidity_levels
    
    def _get_session_liquidity(self, session_context: dict, current_price: float) -> Dict[str, List[Dict]]:
        """Get liquidity from session ranges (Asian, London)."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Asian Range liquidity (highest priority)
        if session_context.get('last_asian_range'):
            asian_range = session_context['last_asian_range']
            asian_high = asian_range['high']
            asian_low = asian_range['low']
            
            if asian_high > current_price:
                liquidity['buy_side'].append({
                    'level': asian_high,
                    'type': 'asian_high',
                    'priority': 'very_high',
                    'description': 'Asian Session High'
                })
            
            if asian_low < current_price:
                liquidity['sell_side'].append({
                    'level': asian_low,
                    'type': 'asian_low', 
                    'priority': 'very_high',
                    'description': 'Asian Session Low'
                })
        
        # London Range liquidity
        if session_context.get('last_london_range'):
            london_range = session_context['last_london_range']
            london_high = london_range['high']
            london_low = london_range['low']
            
            if london_high > current_price:
                liquidity['buy_side'].append({
                    'level': london_high,
                    'type': 'london_high',
                    'priority': 'high',
                    'description': 'London Session High'
                })
            
            if london_low < current_price:
                liquidity['sell_side'].append({
                    'level': london_low,
                    'type': 'london_low',
                    'priority': 'high', 
                    'description': 'London Session Low'
                })
        
        return liquidity
    
    def _get_equal_highs_lows(self, ohlc_df: pd.DataFrame, current_price: float) -> Dict[str, List[Dict]]:
        """Detect equal highs and lows - major liquidity magnets."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Look at last 50 candles for equal levels
        recent_data = ohlc_df.tail(50)
        tolerance = 0.0002  # 2 pips tolerance for "equal" levels
        
        # Find equal highs
        highs = recent_data['high'].values
        unique_highs = []
        
        for i, high in enumerate(highs[:-5]):  # Don't include last 5 candles
            # Count how many times this high appears (within tolerance)
            equal_count = 0
            
            for other_high in highs:
                if abs(high - other_high) <= tolerance:
                    equal_count += 1
            
            # If we have 2+ equal highs, it's liquidity
            if equal_count >= 2 and high > current_price:
                level_exists = any(abs(high - existing['level']) <= tolerance 
                                 for existing in unique_highs)
                
                if not level_exists:
                    unique_highs.append({
                        'level': high,
                        'type': 'equal_highs',
                        'priority': 'very_high',
                        'description': f'Equal Highs ({equal_count} times)',
                        'count': equal_count
                    })
        
        # Find equal lows
        lows = recent_data['low'].values
        unique_lows = []
        
        for i, low in enumerate(lows[:-5]):
            equal_count = 0
            
            for other_low in lows:
                if abs(low - other_low) <= tolerance:
                    equal_count += 1
            
            if equal_count >= 2 and low < current_price:
                level_exists = any(abs(low - existing['level']) <= tolerance 
                                 for existing in unique_lows)
                
                if not level_exists:
                    unique_lows.append({
                        'level': low,
                        'type': 'equal_lows',
                        'priority': 'very_high',
                        'description': f'Equal Lows ({equal_count} times)',
                        'count': equal_count
                    })
        
        liquidity['buy_side'].extend(unique_highs)
        liquidity['sell_side'].extend(unique_lows)
        
        return liquidity
    
    def _get_swing_liquidity(self, ohlc_df: pd.DataFrame, current_price: float) -> Dict[str, List[Dict]]:
        """Get liquidity from recent swing highs and lows."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Use simple swing detection for recent pivots
        recent_data = ohlc_df.tail(100)
        window = 5
        
        # Find swing highs
        for i in range(window, len(recent_data) - window):
            current_high = recent_data['high'].iloc[i]
            
            # Check if it's a swing high
            left_highs = recent_data['high'].iloc[i-window:i]
            right_highs = recent_data['high'].iloc[i+1:i+window+1]
            
            if (current_high > left_highs.max() and 
                current_high > right_highs.max() and 
                current_high > current_price):
                
                liquidity['buy_side'].append({
                    'level': current_high,
                    'type': 'swing_high',
                    'priority': 'medium',
                    'description': 'Recent Swing High'
                })
        
        # Find swing lows
        for i in range(window, len(recent_data) - window):
            current_low = recent_data['low'].iloc[i]
            
            # Check if it's a swing low
            left_lows = recent_data['low'].iloc[i-window:i]
            right_lows = recent_data['low'].iloc[i+1:i+window+1]
            
            if (current_low < left_lows.min() and 
                current_low < right_lows.min() and 
                current_low < current_price):
                
                liquidity['sell_side'].append({
                    'level': current_low,
                    'type': 'swing_low',
                    'priority': 'medium',
                    'description': 'Recent Swing Low'
                })
        
        return liquidity
    
    def _get_htf_liquidity(self, daily_df: pd.DataFrame, current_price: float) -> Dict[str, List[Dict]]:
        """Get higher timeframe liquidity levels."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        if daily_df is None or len(daily_df) < 5:
            return liquidity
        
        # Previous day high/low
        prev_day = daily_df.iloc[-2]  # Yesterday
        
        if prev_day['high'] > current_price:
            liquidity['buy_side'].append({
                'level': prev_day['high'],
                'type': 'prev_day_high',
                'priority': 'high',
                'description': 'Previous Day High'
            })
        
        if prev_day['low'] < current_price:
            liquidity['sell_side'].append({
                'level': prev_day['low'],
                'type': 'prev_day_low',
                'priority': 'high',
                'description': 'Previous Day Low'
            })
        
        # Weekly high/low (last 5 days)
        week_high = daily_df.tail(5)['high'].max()
        week_low = daily_df.tail(5)['low'].min()
        
        if week_high > current_price:
            liquidity['buy_side'].append({
                'level': week_high,
                'type': 'week_high',
                'priority': 'medium',
                'description': 'Weekly High'
            })
        
        if week_low < current_price:
            liquidity['sell_side'].append({
                'level': week_low,
                'type': 'week_low',
                'priority': 'medium',
                'description': 'Weekly Low'
            })
        
        return liquidity
    
    def _get_round_number_liquidity(self, current_price: float) -> Dict[str, List[Dict]]:
        """Get psychological round number levels."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Convert price to pips for round number calculation
        price_in_pips = int(current_price * 10000)
        
        # Find nearby round numbers (00, 50 levels)
        for offset in range(-200, 201, 50):  # Check ±200 pips in 50 pip increments
            round_pip = ((price_in_pips // 100) * 100) + offset
            round_price = round_pip / 10000
            
            if abs(round_price - current_price) > 0.0020:  # At least 20 pips away
                if round_price > current_price:
                    liquidity['buy_side'].append({
                        'level': round_price,
                        'type': 'round_number',
                        'priority': 'low',
                        'description': f'Round Number {round_price:.4f}'
                    })
                elif round_price < current_price:
                    liquidity['sell_side'].append({
                        'level': round_price,
                        'type': 'round_number',
                        'priority': 'low',
                        'description': f'Round Number {round_price:.4f}'
                    })
        
        return liquidity
    
    def get_target_for_bias(self, bias: str, liquidity_levels: Dict[str, List[Dict]], 
                           entry_price: float, min_reward_risk: float = 1.0, sl_price: float = None) -> Optional[Dict]:
        """
        Get the best liquidity target based on ICT methodology.
        
        ICT Rules:
        - Bullish bias (BUY): Target buy-side liquidity (above entry price for profit)
        - Bearish bias (SELL): Target sell-side liquidity (below entry price for profit)
        """
        if bias == 'bullish':
            # Target buy-side liquidity (above price for profit)
            candidates = [liq for liq in liquidity_levels['buy_side'] 
                         if liq['level'] > entry_price]
            target_direction = 'buy_side'
        elif bias == 'bearish':
            # Target sell-side liquidity (below price for profit)
            candidates = [liq for liq in liquidity_levels['sell_side'] 
                         if liq['level'] < entry_price]
            target_direction = 'sell_side'
        else:
            return None
        
        if not candidates:
            logger.warning(f"No {target_direction} liquidity targets found for {bias} bias")
            return None
        
        # Filter candidates that meet minimum R:R requirement
        profitable_candidates = []
        for candidate in candidates:
            potential_reward = abs(candidate['level'] - entry_price)
            
            # If we have SL price, use it for proper R:R calculation
            if sl_price is not None:
                risk = abs(entry_price - sl_price)
                if risk > 0 and potential_reward >= min_reward_risk * risk:
                    profitable_candidates.append(candidate)
            else:
                # Fallback: use a minimum distance (e.g., 10 pips for major pairs)
                min_distance = 0.0010  # 10 pips for major pairs
                if potential_reward >= min_distance:
                    profitable_candidates.append(candidate)
        
        # If no profitable candidates, return None rather than unprofitable target
        if not profitable_candidates:
            logger.warning(f"No {target_direction} liquidity targets meet minimum R:R requirement of {min_reward_risk:.1f}")
            return None
        
        # Sort by priority and distance
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        profitable_candidates.sort(key=lambda x: (
            priority_order.get(x['priority'], 999),
            abs(x['level'] - entry_price)
        ))
        
        # Return the best profitable candidate
        best_target = profitable_candidates[0]
                
        return best_target