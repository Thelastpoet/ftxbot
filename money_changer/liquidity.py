"""
ICT Liquidity Detection Module
Implements proper buy-side/sell-side liquidity concepts based on ICT methodology
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class LiquidityDetector:
    """
    Detects liquidity levels following a more sophisticated ICT methodology.
    Focuses on concepts like Dealing Range, Inducement, and a clear liquidity hierarchy.
    """
    
    def __init__(self, swing_lookback=10):
        self.swing_lookback = swing_lookback

    def _get_swings(self, ohlc_df: pd.DataFrame) -> pd.DataFrame:
        """Identifies swing highs and lows using a rolling window."""
        highs = ohlc_df['high'].rolling(self.swing_lookback, center=True).max()
        lows = ohlc_df['low'].rolling(self.swing_lookback, center=True).min()
        
        swing_highs = ohlc_df[ohlc_df['high'] == highs]
        swing_lows = ohlc_df[ohlc_df['low'] == lows]
        
        swings = pd.concat([swing_highs.assign(type='high'), swing_lows.assign(type='low')])
        swings = swings.sort_index().drop_duplicates(subset=['type'], keep='last')
        return swings

    def _identify_dealing_range(self, ohlc_df: pd.DataFrame) -> Optional[Dict]:
        """
        Identifies the current dealing range based on the most recent
        significant swing high and low that have taken liquidity.
        """
        swings = self._get_swings(ohlc_df.tail(200)) # Look at last 200 candles
        if len(swings) < 2:
            return None

        # A dealing range is confirmed when a swing takes out a prior swing
        last_swing = swings.iloc[-1]
        prior_swing = swings.iloc[-2]

        # Bullish scenario: new swing low takes out prior swing low
        if last_swing['type'] == 'low' and last_swing['low'] < prior_swing['low']:
            # Find the highest high since that new low was formed
            highest_high_since_low = ohlc_df.loc[last_swing.name:]['high'].max()
            highest_high_candle = ohlc_df[ohlc_df['high'] == highest_high_since_low].iloc[-1]
            
            return {
                'start_time': last_swing.name,
                'end_time': highest_high_candle.name,
                'external_high': highest_high_since_low,
                'external_low': last_swing['low'],
                'type': 'bullish'
            }

        # Bearish scenario: new swing high takes out prior swing high
        if last_swing['type'] == 'high' and last_swing['high'] > prior_swing['high']:
            # Find the lowest low since that new high was formed
            lowest_low_since_high = ohlc_df.loc[last_swing.name:]['low'].min()
            lowest_low_candle = ohlc_df[ohlc_df['low'] == lowest_low_since_high].iloc[-1]

            return {
                'start_time': last_swing.name,
                'end_time': lowest_low_candle.name,
                'external_high': last_swing['high'],
                'external_low': lowest_low_since_high,
                'type': 'bearish'
            }
            
        return None

    def get_liquidity_levels(self, ohlc_df: pd.DataFrame, session_context: dict, 
                           daily_df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        """
        Get all liquidity levels following the refined ICT methodology.
        """
        current_price = ohlc_df['close'].iloc[-1]
        dealing_range = self._identify_dealing_range(ohlc_df)

        liquidity_levels = {
            'buy_side': [],
            'sell_side': [],
            'dealing_range': dealing_range
        }

        # Add external liquidity from the dealing range
        if dealing_range:
            if dealing_range['external_high'] > current_price:
                liquidity_levels['buy_side'].append({
                    'level': dealing_range['external_high'], 'type': 'external_range_high',
                    'priority': 'primary', 'description': 'Dealing Range High'
                })
            if dealing_range['external_low'] < current_price:
                 liquidity_levels['sell_side'].append({
                    'level': dealing_range['external_low'], 'type': 'external_range_low',
                    'priority': 'primary', 'description': 'Dealing Range Low'
                })

        # Add session liquidity (always important)
        session_liquidity = self._get_session_liquidity(session_context, current_price)
        liquidity_levels['buy_side'].extend(session_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(session_liquidity['sell_side'])

        # Add HTF liquidity
        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df, current_price)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])

        # Identify and add internal liquidity (inducement, EQL)
        internal_liquidity = self._get_internal_range_liquidity(ohlc_df, current_price, dealing_range)
        liquidity_levels['buy_side'].extend(internal_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(internal_liquidity['sell_side'])

        # Sort by distance from current price
        liquidity_levels['buy_side'] = sorted(liquidity_levels['buy_side'], key=lambda x: x['level'])
        liquidity_levels['sell_side'] = sorted(liquidity_levels['sell_side'], key=lambda x: x['level'], reverse=True)
        
        return liquidity_levels

    def _get_internal_range_liquidity(self, ohlc_df: pd.DataFrame, current_price: float, dealing_range: Optional[Dict]) -> Dict:
        """
        Identifies internal liquidity such as inducement and equal highs/lows
        WITHIN the context of the current dealing range.
        """
        liquidity = {'buy_side': [], 'sell_side': []}
        if not dealing_range:
            return liquidity

        # Look for liquidity only within the identified dealing range
        range_df = ohlc_df.loc[dealing_range['start_time']:dealing_range['end_time']]
        if len(range_df) < 5:
            return liquidity

        # 1. Inducement Points (minor swings before the external range)
        swings = self._get_swings(range_df)
        if len(swings) > 2:
            # Inducement high in a bearish range
            if dealing_range['type'] == 'bearish':
                inducement_high = swings[swings['type'] == 'high'].iloc[-1]
                if inducement_high['high'] < dealing_range['external_high'] and inducement_high['high'] > current_price:
                    liquidity['buy_side'].append({
                        'level': inducement_high['high'], 'type': 'inducement_high',
                        'priority': 'secondary', 'description': 'Inducement High'
                    })
            # Inducement low in a bullish range
            if dealing_range['type'] == 'bullish':
                inducement_low = swings[swings['type'] == 'low'].iloc[-1]
                if inducement_low['low'] > dealing_range['external_low'] and inducement_low['low'] < current_price:
                    liquidity['sell_side'].append({
                        'level': inducement_low['low'], 'type': 'inducement_low',
                        'priority': 'secondary', 'description': 'Inducement Low'
                    })

        # 2. Refined Equal Highs/Lows (structurally significant)
        equal_levels = self._get_refined_equal_highs_lows(range_df, current_price)
        liquidity['buy_side'].extend(equal_levels['buy_side'])
        liquidity['sell_side'].extend(equal_levels['sell_side'])

        return liquidity

    def _get_refined_equal_highs_lows(self, ohlc_df: pd.DataFrame, current_price: float) -> Dict:
        """
        More robust EQL detection that looks for clean, consecutive swings
        at a similar price level.
        """
        liquidity = {'buy_side': [], 'sell_side': []}
        swings = self._get_swings(ohlc_df)
        tolerance = ohlc_df['high'].mean() * 0.0005 # 0.05% tolerance

        # Find equal highs
        high_swings = swings[swings['type'] == 'high']
        for i in range(len(high_swings) - 1):
            h1 = high_swings.iloc[i]
            h2 = high_swings.iloc[i+1]
            if abs(h1['high'] - h2['high']) <= tolerance and h1['high'] > current_price:
                level = max(h1['high'], h2['high'])
                liquidity['buy_side'].append({
                    'level': level, 'type': 'equal_highs',
                    'priority': 'secondary', 'description': 'Structurally Equal Highs'
                })

        # Find equal lows
        low_swings = swings[swings['type'] == 'low']
        for i in range(len(low_swings) - 1):
            l1 = low_swings.iloc[i]
            l2 = low_swings.iloc[i+1]
            if abs(l1['low'] - l2['low']) <= tolerance and l1['low'] < current_price:
                level = min(l1['low'], l2['low'])
                liquidity['sell_side'].append({
                    'level': level, 'type': 'equal_lows',
                    'priority': 'secondary', 'description': 'Structurally Equal Lows'
                })
        return liquidity
    
    def _get_session_liquidity(self, session_context: dict, current_price: float) -> Dict[str, List[Dict]]:
        """Get liquidity from session ranges (Asian, London)."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Asian Range liquidity
        if session_context.get('last_asian_range'):
            asian_range = session_context['last_asian_range']
            if asian_range['high'] > current_price:
                liquidity['buy_side'].append({
                    'level': asian_range['high'], 'type': 'asian_high',
                    'priority': 'tertiary', 'description': 'Asian Session High'
                })
            if asian_range['low'] < current_price:
                liquidity['sell_side'].append({
                    'level': asian_range['low'], 'type': 'asian_low', 
                    'priority': 'tertiary', 'description': 'Asian Session Low'
                })
        
        # London Range liquidity
        if session_context.get('last_london_range'):
            london_range = session_context['last_london_range']
            if london_range['high'] > current_price:
                liquidity['buy_side'].append({
                    'level': london_range['high'], 'type': 'london_high',
                    'priority': 'tertiary', 'description': 'London Session High'
                })
            if london_range['low'] < current_price:
                liquidity['sell_side'].append({
                    'level': london_range['low'], 'type': 'london_low',
                    'priority': 'tertiary', 'description': 'London Session Low'
                })
        
        return liquidity
    
    def _get_htf_liquidity(self, daily_df: pd.DataFrame, current_price: float) -> Dict[str, List[Dict]]:
        """Get higher timeframe liquidity levels."""
        liquidity = {'buy_side': [], 'sell_side': []}
        if daily_df is None or len(daily_df) < 5:
            return liquidity
        
        # Previous day high/low
        prev_day = daily_df.iloc[-2]
        if prev_day['high'] > current_price:
            liquidity['buy_side'].append({
                'level': prev_day['high'], 'type': 'prev_day_high',
                'priority': 'secondary', 'description': 'Previous Day High'
            })
        if prev_day['low'] < current_price:
            liquidity['sell_side'].append({
                'level': prev_day['low'], 'type': 'prev_day_low',
                'priority': 'secondary', 'description': 'Previous Day Low'
            })
        
        # Weekly high/low (last 5 days)
        week_high = daily_df.tail(5)['high'].max()
        week_low = daily_df.tail(5)['low'].min()
        if week_high > current_price:
            liquidity['buy_side'].append({
                'level': week_high, 'type': 'week_high',
                'priority': 'primary', 'description': 'Weekly High'
            })
        if week_low < current_price:
            liquidity['sell_side'].append({
                'level': week_low, 'type': 'week_low',
                'priority': 'primary', 'description': 'Weekly Low'
            })
        
        return liquidity
    
    def get_target_for_bias(self, bias: str, liquidity_levels: Dict[str, List[Dict]], 
                           entry_price: float, min_reward_risk: float = 1.0, sl_price: float = None) -> Optional[Dict]:
        """
        Get the best liquidity target based on the new, structured ICT methodology.
        - Primary target is always the opposing external range liquidity.
        - Secondary targets can be internal liquidity if the R:R is favorable.
        """
        if bias == 'bullish':
            target_side = 'buy_side'
        elif bias == 'bearish':
            target_side = 'sell_side'
        else:
            return None

        candidates = [liq for liq in liquidity_levels[target_side] if (bias == 'bullish' and liq['level'] > entry_price) or (bias == 'bearish' and liq['level'] < entry_price)]
        if not candidates:
            logger.warning(f"No valid {target_side} liquidity targets found for {bias} bias.")
            return None

        # Prioritize by a clear hierarchy
        priority_order = {'primary': 0, 'secondary': 1, 'tertiary': 2}
        candidates.sort(key=lambda x: (
            priority_order.get(x['priority'], 99),
            abs(x['level'] - entry_price) # For same-priority, nearest is not always best, but it's a simple heuristic
        ))

        # Filter for profitability
        risk = abs(entry_price - sl_price) if sl_price is not None else 0
        if risk == 0:
            logger.warning("Cannot calculate R:R, risk is zero.")
            return None

        for candidate in candidates:
            potential_reward = abs(candidate['level'] - entry_price)
            if potential_reward >= min_reward_risk * risk:
                logger.info(f"Selected profitable target: {candidate['description']} at {candidate['level']:.5f} ({potential_reward/risk:.1f}R)")
                return candidate

        logger.warning(f"No {target_side} liquidity targets for {bias} bias met the minimum R:R of {min_reward_risk:.1f}")
        return None
