import pandas as pd
import logging
from typing import List, Dict, Optional
from smc import smc
import pytz

logger = logging.getLogger(__name__)

class LiquidityDetector:
    
    def __init__(self, settings: Dict = None):
        self.settings = settings or {}
        self.swing_lookback = self.settings['swing_lookback']
        self.eq_range_percent = self.settings['eq_range_percent']
        
    def get_liquidity_levels(self, ohlc_df: pd.DataFrame, session_context: dict, 
                           daily_df: pd.DataFrame = None) -> Dict[str, List[Dict]]:
        current_price = ohlc_df['close'].iloc[-1]
        
        swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
        
        liquidity_levels = {'buy_side': [], 'sell_side': []}
        

        session_liquidity = self._get_session_liquidity(session_context, ohlc_df)
        liquidity_levels['buy_side'].extend(session_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(session_liquidity['sell_side'])
        

        equal_levels = self._get_equal_highs_lows(ohlc_df, swings)
        liquidity_levels['buy_side'].extend(equal_levels['buy_side'])
        liquidity_levels['sell_side'].extend(equal_levels['sell_side'])
        

        swing_liquidity = self._get_swing_liquidity(swings)
        liquidity_levels['buy_side'].extend(swing_liquidity['buy_side'])
        liquidity_levels['sell_side'].extend(swing_liquidity['sell_side'])
        

        if daily_df is not None:
            htf_liquidity = self._get_htf_liquidity(daily_df)
            liquidity_levels['buy_side'].extend(htf_liquidity['buy_side'])
            liquidity_levels['sell_side'].extend(htf_liquidity['sell_side'])
        

        liquidity_levels['buy_side'] = self._finalize_levels(liquidity_levels['buy_side'], current_price, 'above')
        liquidity_levels['sell_side'] = self._finalize_levels(liquidity_levels['sell_side'], current_price, 'below')

        return liquidity_levels

    def _get_session_liquidity(self, session_context: dict, ohlc_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        liquidity = {'buy_side': [], 'sell_side': []}
        
        try:

            if ohlc_df.index.tzinfo is None:
                ohlc_df.index = ohlc_df.index.tz_localize('UTC')
            

            if session_context.get('last_asian_range'):
                asian_range = session_context['last_asian_range']
                liquidity['buy_side'].append({'level': asian_range['high'], 'type': 'asian_high', 'priority': 'very_high', 'description': 'Asian Session High'})
                liquidity['sell_side'].append({'level': asian_range['low'], 'type': 'asian_low', 'priority': 'very_high', 'description': 'Asian Session Low'})
            

            current_ny_time = ohlc_df.index[-1].tz_convert(pytz.timezone("America/New_York"))
            ny_start = current_ny_time.replace(hour=self.settings['ny_session_start_hour'], 
                                               minute=self.settings['ny_session_start_minute'], 
                                               second=0, microsecond=0)
            ny_end = current_ny_time.replace(hour=self.settings['ny_session_end_hour'], 
                                             minute=self.settings['ny_session_end_minute'], 
                                             second=0, microsecond=0)
            

            ny_start_utc = ny_start.tz_convert('UTC')
            ny_end_utc = ny_end.tz_convert('UTC')
            
            ny_data = ohlc_df[(ohlc_df.index >= ny_start_utc) & (ohlc_df.index < ny_end_utc)]
            if not ny_data.empty:
                liquidity['buy_side'].append({'level': ny_data['high'].max(), 'type': 'ny_high', 'priority': 'high', 'description': 'New York Session High'})
                liquidity['sell_side'].append({'level': ny_data['low'].min(), 'type': 'ny_low', 'priority': 'high', 'description': 'New York Session Low'})

        except Exception as e:
            logger.error(f"Error processing session liquidity: {str(e)}")
            
        return liquidity

    def _get_equal_highs_lows(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame) -> Dict[str, List[Dict]]:
        liquidity = {'buy_side': [], 'sell_side': []}
        eq_levels = smc.liquidity(ohlc_df, swings, range_percent=self.eq_range_percent)
        
        if not eq_levels.empty:
            unswept = eq_levels[eq_levels['Swept'] == 0]
            for _, row in unswept.iterrows():
                if row['Liquidity'] == 1:
                    liquidity['sell_side'].append({'level': row['Level'], 'type': 'equal_lows', 'priority': 'very_high', 'description': 'Equal Lows'})
                elif row['Liquidity'] == -1:
                    liquidity['buy_side'].append({'level': row['Level'], 'type': 'equal_highs', 'priority': 'very_high', 'description': 'Equal Highs'})
        return liquidity

    def _get_swing_liquidity(self, swings: pd.DataFrame) -> Dict[str, List[Dict]]:
        liquidity = {'buy_side': [], 'sell_side': []}
        recent_swings = swings.dropna().tail(10)

        for _, swing in recent_swings.iterrows():
            if swing['HighLow'] == 1:
                liquidity['buy_side'].append({'level': swing['Level'], 'type': 'swing_high', 'priority': 'medium', 'description': 'Recent Swing High'})
            elif swing['HighLow'] == -1:
                liquidity['sell_side'].append({'level': swing['Level'], 'type': 'swing_low', 'priority': 'medium', 'description': 'Recent Swing Low'})
        return liquidity

    def _get_htf_liquidity(self, daily_df: pd.DataFrame) -> Dict[str, List[Dict]]:
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
        if not levels:
            return []
        

        if side == 'above':
            filtered_by_side = [l for l in levels if l['level'] > current_price]
        else: # 'below'
            filtered_by_side = [l for l in levels if l['level'] < current_price]
        
        if not filtered_by_side:
            return []


        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        filtered_by_side.sort(key=lambda x: priority_order.get(x['priority'], 99))
        

        unique_levels = {}
        for level in filtered_by_side:
            price = level['level']
            if price not in unique_levels: # Only add if we haven't seen this price before
                unique_levels[price] = level
        
        final_list = list(unique_levels.values())
        

        final_list.sort(key=lambda x: abs(x['level'] - current_price))
        
        return final_list

    def get_target_for_bias(self, bias: str, liquidity_levels: Dict[str, List[Dict]], 
                           entry_price: float, min_rr: float = 1.0, sl_price: Optional[float] = None) -> Optional[Dict]:
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