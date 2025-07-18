"""
ICT Liquidity Detection Module
Implements proper buy-side/sell-side liquidity concepts based on ICT methodology
"""

import pandas as pd
import numpy as np
import logging
from logger import operations_logger as logger
from typing import List, Dict, Optional
from smc import smc
import pytz

logger = logging.getLogger(__name__)

class LiquidityDetector:
    """Detects liquidity levels following ICT methodology."""
    
    def __init__(self, swing_lookback=15):
        self.swing_lookback = swing_lookback
        
    def get_liquidity_levels(self, ohlc_df: pd.DataFrame, session_context: dict, 
                       daily_df: pd.DataFrame = None, structure: pd.DataFrame = None) -> Dict[str, List[Dict]]:
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
        
        # Recent swing highs/lows (REFACTORED IMPLEMENTATION)
        swing_liquidity = self._get_swing_liquidity(ohlc_df, swings, structure)
        liquidity_levels['buy_side'].extend(swing_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(swing_liquidity['sell_side'])
        
        # Daily/Weekly levels
        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])
        
        # Finalize and sort levels
        liquidity_levels['buy_side'] = self._finalize_levels(liquidity_levels['buy_side'], current_price, 'above')
        liquidity_levels['sell_side'] = self._finalize_levels(liquidity_levels['sell_side'], current_price, 'below')

        return liquidity_levels

    def _get_session_liquidity(self, session_context: dict, ohlc_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Get liquidity from session ranges (Asian, London, and NY)."""
        liquidity = {'buy_side': [], 'sell_side': []}
        ny_timezone = pytz.timezone("America/New_York")
        
        current_ny_time = ohlc_df.index[-1].astimezone(ny_timezone)

        if session_context.get('last_asian_range'):
            asian_range = session_context['last_asian_range']
            liquidity['buy_side'].append({'level': asian_range['high'], 'type': 'asian_high', 'priority': 'very_high', 'description': 'Asian Session High'})
            liquidity['sell_side'].append({'level': asian_range['low'], 'type': 'asian_low', 'priority': 'very_high', 'description': 'Asian Session Low'})
        
        london_start = current_ny_time.replace(hour=2, minute=0, second=0, microsecond=0)
        london_end = current_ny_time.replace(hour=5, minute=0, second=0, microsecond=0)
        
        if current_ny_time >= london_end:
            london_data = ohlc_df[(ohlc_df.index >= london_start.astimezone(pytz.UTC)) & (ohlc_df.index < london_end.astimezone(pytz.UTC))]
            if not london_data.empty:
                liquidity['buy_side'].append({'level': london_data['high'].max(), 'type': 'london_high', 'priority': 'very_high', 'description': 'London Session High'})
                liquidity['sell_side'].append({'level': london_data['low'].min(), 'type': 'london_low', 'priority': 'very_high', 'description': 'London Session Low'})
        
        ny_open_start = current_ny_time.replace(hour=8, minute=30, second=0, microsecond=0)
        ny_open_end = current_ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
        
        ny_open_data = ohlc_df[(ohlc_df.index >= ny_open_start.astimezone(pytz.UTC)) & (ohlc_df.index < ny_open_end.astimezone(pytz.UTC))]
        if not ny_open_data.empty:
            liquidity['buy_side'].append({'level': ny_open_data['high'].max(), 'type': 'ny_open_high', 'priority': 'high', 'description': 'New York Opening Range High'})
            liquidity['sell_side'].append({'level': ny_open_data['low'].min(), 'type': 'ny_open_low', 'priority': 'high', 'description': 'New York Opening Range Low'})

        return liquidity

    def _get_equal_highs_lows(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detects equal highs/lows using smc.liquidity for consistency."""
        liquidity = {'buy_side': [], 'sell_side': []}
        eq_levels = smc.liquidity(ohlc_df, swings, range_percent=0.01)
        
        if not eq_levels.empty:
            unswept = eq_levels[eq_levels['Swept'] == 0]
            for _, row in unswept.iterrows():
                if row['Liquidity'] == -1:
                    liquidity['sell_side'].append({'level': row['Level'], 'type': 'equal_lows', 'priority': 'very_high', 'description': 'Equal Lows'})
                elif row['Liquidity'] == 1:
                    liquidity['buy_side'].append({'level': row['Level'], 'type': 'equal_highs', 'priority': 'very_high', 'description': 'Equal Highs'})
        return liquidity
    
    def get_preliminary_liquidity(self, ohlc_df: pd.DataFrame, session_context: dict,
                                daily_df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        """
        Gets only the high-level, obvious liquidity pools that do NOT depend on
        market structure analysis. Used for the initial daily bias check.
        """
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
        
        # Daily/Weekly levels
        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])
        
        # Finalize and sort levels
        liquidity_levels['buy_side'] = self._finalize_levels(liquidity_levels['buy_side'], current_price, 'above')
        liquidity_levels['sell_side'] = self._finalize_levels(liquidity_levels['sell_side'], current_price, 'below')

        return liquidity_levels

    def _get_swing_liquidity(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, structure: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Identifies liquidity pools around swing highs and lows, classifying them
        into Internal and External range liquidity, and identifying inducement.
        """
        if swings.empty:
            return {'buy_side': [], 'sell_side': []}
        
        liquidity = {'buy_side': [], 'sell_side': []}
        valid_swings = swings.dropna().copy()
        if len(valid_swings) < 3:
            return liquidity

        # 1. Identify the current major dealing range (External Range Liquidity)
        last_high = valid_swings[valid_swings['HighLow'] == 1].iloc[-1]
        last_low = valid_swings[valid_swings['HighLow'] == -1].iloc[-1]
        
        # Determine the most recent major swing to define the range boundary
        if last_high.name > last_low.name:
            # We are likely in a bearish swing, look for the preceding low
            preceding_lows = valid_swings[(valid_swings['HighLow'] == -1) & (valid_swings.index < last_high.name)]
            if not preceding_lows.empty:
                external_low = preceding_lows.iloc[-1]
                external_high = last_high
            else: # Fallback if structure is unclear
                external_low, external_high = last_low, last_high
        else:
            # We are likely in a bullish swing, look for the preceding high
            preceding_highs = valid_swings[(valid_swings['HighLow'] == 1) & (valid_swings.index < last_low.name)]
            if not preceding_highs.empty:
                external_high = preceding_highs.iloc[-1]
                external_low = last_low
            else: # Fallback
                external_high, external_low = last_high, last_low

        liquidity['buy_side'].append({'level': external_high['Level'], 'type': 'external_high', 'priority': 'high', 'description': 'External Range High (Major Swing)'})
        liquidity['sell_side'].append({'level': external_low['Level'], 'type': 'external_low', 'priority': 'high', 'description': 'External Range Low (Major Swing)'})

        # 2. Identify Internal Range Liquidity and Inducement
        internal_swings = valid_swings[
            (valid_swings.index > external_low.name) & (valid_swings.index < external_high.name)
        ]

        if not internal_swings.empty:
            internal_highs = internal_swings[internal_swings['HighLow'] == 1]
            internal_lows = internal_swings[internal_swings['HighLow'] == -1]

            # Inducement is often the most recent, prominent internal swing
            # that price might sweep before reaching for external liquidity.
            if not internal_highs.empty:
                inducement_high = internal_highs.iloc[-1]
                liquidity['buy_side'].append({'level': inducement_high['Level'], 'type': 'inducement_high', 'priority': 'medium', 'description': 'Inducement High (Internal)'})
            
            if not internal_lows.empty:
                inducement_low = internal_lows.iloc[-1]
                liquidity['sell_side'].append({'level': inducement_low['Level'], 'type': 'inducement_low', 'priority': 'medium', 'description': 'Inducement Low (Internal)'})

            # Add other internal swings as minor liquidity points
            for _, swing in internal_highs.iloc[:-1].iterrows():
                 liquidity['buy_side'].append({'level': swing['Level'], 'type': 'internal_swing_high', 'priority': 'low', 'description': 'Internal Swing High'})
            for _, swing in internal_lows.iloc[:-1].iterrows():
                 liquidity['sell_side'].append({'level': swing['Level'], 'type': 'internal_swing_low', 'priority': 'low', 'description': 'Internal Swing Low'})

        return liquidity
    
    def _get_htf_liquidity(self, daily_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Gets higher timeframe liquidity levels (Previous Day, Previous Week, Previous Month).
        """
        liquidity = {'buy_side': [], 'sell_side': []}
        if daily_df is None or len(daily_df) < 25: 
            return liquidity
        
        # Previous Day High/Low (PDH/PDL)
        prev_day = daily_df.iloc[-2]
        liquidity['buy_side'].append({'level': prev_day['high'], 'type': 'prev_day_high', 'priority': 'high', 'description': 'Previous Day High'})
        liquidity['sell_side'].append({'level': prev_day['low'], 'type': 'prev_day_low', 'priority': 'high', 'description': 'Previous Day Low'})
        
        # Previous Week High/Low (PWH/PWL)
        if len(daily_df) > 6:
            prev_week_df = daily_df.tail(6).iloc[:-1]
            if not prev_week_df.empty:
                liquidity['buy_side'].append({'level': prev_week_df['high'].max(), 'type': 'prev_week_high', 'priority': 'high', 'description': 'Previous Week High'})
                liquidity['sell_side'].append({'level': prev_week_df['low'].min(), 'type': 'prev_week_low', 'priority': 'high', 'description': 'Previous Week Low'})

        # Previous Month High/Low (PMH/PML)
        last_date = daily_df.index[-1]
        prev_month_date = last_date - pd.DateOffset(months=1)
        target_year = prev_month_date.year
        target_month = prev_month_date.month

        # Filter the DataFrame to get all data for the previous month
        prev_month_df = daily_df[(daily_df.index.year == target_year) & (daily_df.index.month == target_month)]

        if not prev_month_df.empty:
            liquidity['buy_side'].append({'level': prev_month_df['high'].max(), 'type': 'prev_month_high', 'priority': 'high', 'description': 'Previous Month High'})
            liquidity['sell_side'].append({'level': prev_month_df['low'].min(), 'type': 'prev_month_low', 'priority': 'high', 'description': 'Previous Month Low'})
            
        return liquidity

    def _finalize_levels(self, levels: List[Dict], current_price: float, side: str) -> List[Dict]:
        """Filters levels by price, removes duplicates, and sorts."""
        if not levels:
            return []
            
        if side == 'above':
            filtered = [l for l in levels if l['level'] > current_price]
        else:
            filtered = [l for l in levels if l['level'] < current_price]

        unique_levels = {}
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        for level in filtered:
            key = level['level']
            if key not in unique_levels or priority_order.get(level['priority'], 9) < priority_order.get(unique_levels[key]['priority'], 9):
                 unique_levels[key] = level
        
        return sorted(list(unique_levels.values()), key=lambda x: abs(x['level'] - current_price))

    def get_target_for_bias(self, bias: str, liquidity_levels: Dict[str, List[Dict]], 
                           entry_price: float, min_rr: float = 1.0, sl_price: Optional[float] = None) -> Optional[Dict]:
        """Gets the best liquidity target for the given bias."""
        target_side = 'buy_side' if bias == 'bullish' else 'sell_side'
        candidates = liquidity_levels.get(target_side, [])
        if not candidates:
            return None

        profitable_candidates = []
        if sl_price is not None:
            risk = abs(entry_price - sl_price)
            if risk > 0:
                for cand in candidates:
                    if abs(cand['level'] - entry_price) >= min_rr * risk:
                        profitable_candidates.append(cand)
            if not profitable_candidates:
                return None
        else:
            profitable_candidates = candidates

        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        profitable_candidates.sort(key=lambda x: (priority_order.get(x['priority'], 99), abs(x['level'] - entry_price)))
        
        return profitable_candidates[0] if profitable_candidates else None