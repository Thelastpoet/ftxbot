"""
ICT Liquidity Detection Module
Implements proper buy-side/sell-side liquidity concepts based on ICT methodology
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from smc import smc
import pytz

logger = logging.getLogger(__name__)

class LiquidityDetector:
    """Detects liquidity levels following ICT methodology."""
    
    def __init__(self, swing_lookback=15):
        self.swing_lookback = swing_lookback
        
    def get_liquidity_levels(self, ohlc_df: pd.DataFrame, session_context: dict, 
                           daily_df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        """Get all liquidity levels following ICT methodology."""
        current_price = ohlc_df['close'].iloc[-1]
        
        swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
        
        liquidity_levels = {'buy_side': [], 'sell_side': []}
        
        # Session-based liquidity (highest priority)
        session_liquidity = self._get_session_liquidity(session_context, ohlc_df)
        liquidity_levels['buy_side'].extend(session_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(session_liquidity['sell_side'])
        
        # Equal highs/lows (very high priority)
        equal_levels = self._get_equal_highs_lows(ohlc_df, swings)
        liquidity_levels['buy_side'].extend(equal_levels['buy_side'])
        liquidity_levels['sell_side'].extend(equal_levels['sell_side'])
        
        # Recent swing highs/lows
        swing_liquidity = self._get_swing_liquidity(swings)
        liquidity_levels['buy_side'].extend(swing_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(swing_liquidity['sell_side'])
        
        # Daily/Weekly levels
        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])
        
        # Remove duplicates and filter by current price
        liquidity_levels['buy_side'] = self._finalize_levels(liquidity_levels['buy_side'], current_price, 'above')
        liquidity_levels['sell_side'] = self._finalize_levels(liquidity_levels['sell_side'], current_price, 'below')

        return liquidity_levels

    def _get_session_liquidity(self, session_context: dict, ohlc_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Get liquidity from session ranges (Asian, London, and NY)."""
        liquidity = {'buy_side': [], 'sell_side': []}
        
        # Asian Range liquidity
        if session_context.get('last_asian_range'):
            asian_range = session_context['last_asian_range']
            liquidity['buy_side'].append({'level': asian_range['high'], 'type': 'asian_high', 'priority': 'very_high', 'description': 'Asian Session High'})
            liquidity['sell_side'].append({'level': asian_range['low'], 'type': 'asian_low', 'priority': 'very_high', 'description': 'Asian Session Low'})
        
        # New York Session liquidity (8:30-11:00 AM NY)
        current_ny_time = ohlc_df.index[-1].astimezone(pytz.timezone("America/New_York"))
        ny_start = current_ny_time.replace(hour=8, minute=30, second=0, microsecond=0)
        ny_end = current_ny_time.replace(hour=11, minute=0, second=0, microsecond=0)
        ny_data = ohlc_df[(ohlc_df.index >= ny_start.astimezone(pytz.UTC)) & (ohlc_df.index < ny_end.astimezone(pytz.UTC))]
        if not ny_data.empty:
            liquidity['buy_side'].append({'level': ny_data['high'].max(), 'type': 'ny_high', 'priority': 'high', 'description': 'New York Session High'})
            liquidity['sell_side'].append({'level': ny_data['low'].min(), 'type': 'ny_low', 'priority': 'high', 'description': 'New York Session Low'})

        return liquidity

    def _get_equal_highs_lows(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detects equal highs/lows using smc.liquidity for consistency."""
        liquidity = {'buy_side': [], 'sell_side': []}
        eq_levels = smc.liquidity(ohlc_df, swings, range_percent=0.01)
        
        if not eq_levels.empty:
            unswept = eq_levels[eq_levels['Swept'] == 0]
            for _, row in unswept.iterrows():
                if row['Liquidity'] == 1:
                    liquidity['sell_side'].append({'level': row['Level'], 'type': 'equal_lows', 'priority': 'very_high', 'description': 'Equal Lows'})
                elif row['Liquidity'] == -1:
                    liquidity['buy_side'].append({'level': row['Level'], 'type': 'equal_highs', 'priority': 'very_high', 'description': 'Equal Highs'})
        return liquidity

    def _get_swing_liquidity(self, swings: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Gets liquidity from swing points provided by the smc library."""
        liquidity = {'buy_side': [], 'sell_side': []}
        recent_swings = swings.dropna().tail(10)

        for _, swing in recent_swings.iterrows():
            if swing['HighLow'] == 1:
                liquidity['buy_side'].append({'level': swing['Level'], 'type': 'swing_high', 'priority': 'medium', 'description': 'Recent Swing High'})
            elif swing['HighLow'] == -1:
                liquidity['sell_side'].append({'level': swing['Level'], 'type': 'swing_low', 'priority': 'medium', 'description': 'Recent Swing Low'})
        return liquidity

    def _get_htf_liquidity(self, daily_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Gets higher timeframe liquidity levels (Previous Day, Previous Week)."""
        liquidity = {'buy_side': [], 'sell_side': []}
        if daily_df is None or len(daily_df) < 5: return liquidity
        
        prev_day = daily_df.iloc[-2]
        liquidity['buy_side'].append({'level': prev_day['high'], 'type': 'prev_day_high', 'priority': 'high', 'description': 'Previous Day High'})
        liquidity['sell_side'].append({'level': prev_day['low'], 'type': 'prev_day_low', 'priority': 'high', 'description': 'Previous Day Low'})
        
        prev_week_df = daily_df.tail(6).iloc[:-1]
        liquidity['buy_side'].append({'level': prev_week_df['high'].max(), 'type': 'prev_week_high', 'priority': 'high', 'description': 'Previous Week High'})
        liquidity['sell_side'].append({'level': prev_week_df['low'].min(), 'type': 'prev_week_low', 'priority': 'high', 'description': 'Previous Week Low'})
        return liquidity

    def _finalize_levels(self, levels: List[Dict], current_price: float, side: str) -> List[Dict]:
        """Filters levels by price, removes duplicates, and sorts."""
        if not levels:
            return []
            
        # Filter by price
        if side == 'above':
            filtered = [l for l in levels if l['level'] > current_price]
        else:
            filtered = [l for l in levels if l['level'] < current_price]

        # Remove duplicates
        unique_levels = {l['level']: l for l in filtered}.values()
        
        # Sort by distance
        return sorted(list(unique_levels), key=lambda x: abs(x['level'] - current_price))

    def get_target_for_bias(self, bias: str, liquidity_levels: Dict[str, List[Dict]], 
                           entry_price: float, min_rr: float = 1.0, sl_price: Optional[float] = None) -> Optional[Dict]:
        """Gets the best liquidity target for the given bias."""
        target_side = 'buy_side' if bias == 'bullish' else 'sell_side'
        candidates = liquidity_levels.get(target_side, [])
        if not candidates:
            return None

        profitable_candidates = []
        # If sl_price is provided, filter by R:R ratio
        if sl_price is not None:
            risk = abs(entry_price - sl_price)
            if risk > 0:
                for cand in candidates:
                    if abs(cand['level'] - entry_price) >= min_rr * risk:
                        profitable_candidates.append(cand)
            if not profitable_candidates:
                return None
        else:
            # If no sl_price, all candidates are valid
            profitable_candidates = candidates

        # Sort by priority, then by distance
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        profitable_candidates.sort(key=lambda x: (priority_order.get(x['priority'], 99), abs(x['level'] - entry_price)))
        
        return profitable_candidates[0] if profitable_candidates else None