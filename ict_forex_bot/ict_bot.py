"""
ICT Trading Engine 
"""

import pytz
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from smc import smc
from talib import ATR

from liquidity import LiquidityDetector

logger = logging.getLogger(__name__)

@dataclass
class ICTNarrative:
    """Represents the complete ICT narrative for a trading opportunity."""
    daily_bias: str
    po3_phase: str
    manipulation_confirmed: bool
    manipulation_level: float
    structure_broken: bool
    structure_level: float
    in_killzone: bool
    killzone_name: str
    ote_zone: Dict[str, float]
    order_blocks: List[Dict]
    current_price: float
    entry_model: str

class ICTAnalyzer:
    """Analyzes price action through the ICT lens."""
    
    def __init__(self, config):
        self.config = config
        self.swing_lookback = config.SWING_LOOKBACK
        self.structure_lookback = config.STRUCTURE_LOOKBACK
        self.liquidity_detector = LiquidityDetector()
        
    def analyze(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Dict:
        """Perform comprehensive ICT analysis following the narrative sequence."""
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback:
            logger.warning(f"{symbol}: Insufficient data for ICT analysis")
            return {}
        
        swings = self._get_swings(ohlc_df)
        session_context = self._get_session_context(ohlc_df)
        if session_context.get('error'):
            logger.warning(f"{symbol}: Could not determine session context. Reason: {session_context['error']}")
            return {}

        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_context, symbol, daily_df, swings)
        manipulation = po3_analysis.get('manipulation', {'detected': False})
        
        structure = self._get_structure(ohlc_df, swings, daily_bias)
        order_blocks = self._get_order_blocks(ohlc_df, swings)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, daily_bias, manipulation)
        htf_levels = self._get_htf_levels(h4_df)
        
        analysis_result = {
            'symbol': symbol, 'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1], 'swings': swings, 'structure': structure,
            'daily_bias': daily_bias, 'po3_analysis': po3_analysis, 'manipulation': manipulation,
            'order_blocks': order_blocks, 'fair_value_gaps': fair_value_gaps,
            'premium_discount': pd_analysis,
            'ote_zones': ote_zones, 'session': session_context, 'htf_levels': htf_levels,
            'ohcl_df': ohlc_df,
            'daily_df': daily_df,
        }
        
        return analysis_result

    def _get_htf_levels(self, h4_df):
        """Get higher timeframe levels using H4 data from broker."""
        if h4_df is None or h4_df.empty or len(h4_df) < 2:
            logger.warning("Insufficient H4 data for HTF levels")
            return {}
        
        try:
            previous_h4 = h4_df.iloc[-2]
            current_h4 = h4_df.iloc[-1]
            
            current_high = current_h4['high']
            current_low = current_h4['low']
            prev_high = previous_h4['high']
            prev_low = previous_h4['low']
            
            return {
                'h4_high': prev_high,
                'h4_low': prev_low,
                'h4_broken_high': current_high > prev_high,
                'h4_broken_low': current_low < prev_low
            }
            
        except Exception as e:
            logger.warning(f"Error getting HTF levels from H4 data: {e}")
        
        return {}
    
    def _get_swings(self, ohlc_df):
        """Identify swing highs and lows."""
        try:
            swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
            return swings
        except Exception as e:
            logger.error(f"Error getting swings: {e}")
            return pd.DataFrame()
    
    def _get_structure(self, ohlc_df, swings, daily_bias):
        """Analyze ICT market structure using framework context."""
        if swings.empty:
            return pd.DataFrame()
            
        try:
            structure = self._get_ict_structure(ohlc_df, swings, daily_bias)
            return structure
        except Exception as e:
            logger.error(f"Error getting ICT structure: {e}")
            return pd.DataFrame()
        
    def _get_session_context(self, ohlc_df: pd.DataFrame) -> dict:
        """Get proper ICT session context with corrected Asian range timing."""
        context = {
            'last_asian_range': None,
            'in_killzone': False,
            'killzone_name': None,
            'error': None
        }
        
        if ohlc_df.index.tz is None:
            context['error'] = "DataFrame index is not timezone-aware."
            logger.error(context['error'])
            return context

        try:
            latest_utc_time = ohlc_df.index[-1]
            latest_ny_time = latest_utc_time.astimezone(self.config.NY_TIMEZONE)
            
            current_ny_date = latest_ny_time.date()
            current_ny_hour = latest_ny_time.hour
            current_ny_minute = latest_ny_time.minute
            
            # Determine which Asian range to use
            if current_ny_hour < 19:
                asian_date = current_ny_date - pd.Timedelta(days=1)
            else:
                asian_date = current_ny_date
            
            # Create Asian range times
            asian_start_ny = latest_ny_time.replace(
                year=asian_date.year,
                month=asian_date.month,
                day=asian_date.day,
                hour=self.config.ICT_ASIAN_RANGE['start'].hour,
                minute=self.config.ICT_ASIAN_RANGE['start'].minute,
                second=0,
                microsecond=0
            )

            london_kz_start_time = self.config.ICT_LONDON_KILLZONE['start']
            london_kz_date = asian_date + pd.Timedelta(days=1)

            asian_end_ny = latest_ny_time.replace(
                year=london_kz_date.year,
                month=london_kz_date.month,
                day=london_kz_date.day,
                hour=london_kz_start_time.hour,
                minute=london_kz_start_time.minute,
                second=0,
                microsecond=0
            )
            
            asian_start_utc = asian_start_ny.astimezone(pytz.UTC)
            asian_end_utc = asian_end_ny.astimezone(pytz.UTC)
            
            asian_data = ohlc_df[(ohlc_df.index >= asian_start_utc) & (ohlc_df.index < asian_end_utc)]
            
            if not asian_data.empty:
                context['last_asian_range'] = {
                    'start_time_utc': asian_data.index[0],
                    'end_time_utc': asian_end_utc,
                    'high': asian_data['high'].max(),
                    'low': asian_data['low'].min(),
                    'start_idx': ohlc_df.index.get_loc(asian_data.index[0])
                }
                logger.debug(f"Asian Range (7PM-10PM NY): High={context['last_asian_range']['high']:.5f}, Low={context['last_asian_range']['low']:.5f}")
            else:
                logger.debug(f"No Asian range data found for {asian_date}")
            
            # Check Kill Zones using NY time
            if 2 <= current_ny_hour < 5:
                context['in_killzone'] = True
                context['killzone_name'] = 'London'
                
            elif (current_ny_hour == 8 and current_ny_minute >= 30) or (9 <= current_ny_hour < 11):
                context['in_killzone'] = True
                context['killzone_name'] = 'NewYork'
                
            elif 10 <= current_ny_hour < 12:
                context['in_killzone'] = True
                context['killzone_name'] = 'LondonClose'
                
        except Exception as e:
            context['error'] = f"Error calculating session context: {e}"
            logger.error(context['error'], exc_info=True)
                        
        return context
    
    def _determine_daily_bias(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None) -> Tuple[str, Dict]:
        """Determines bias using HTF Order Flow and Draw on Liquidity analysis."""
        logger.debug(f"\n--- {symbol} ICT BIAS CHECKLIST ---")
        if daily_df is None or daily_df.empty:
            return 'neutral', {'error': 'No daily data available'}
        
        htf_order_flow = self._analyze_daily_order_flow(daily_df)
        
        session_context = self._get_session_context(ohlc_df)
        liquidity_levels = self.liquidity_detector.get_liquidity_levels(ohlc_df, session_context, daily_df)
        
        best_buyside_target = self.liquidity_detector.get_target_for_bias('bullish', liquidity_levels, ohlc_df['close'].iloc[-1])
        best_sellside_target = self.liquidity_detector.get_target_for_bias('bearish', liquidity_levels, ohlc_df['close'].iloc[-1])

        draw_on_liquidity = "None"
        draw_reason = "No clear high-priority liquidity target."
        
        if best_buyside_target and best_sellside_target:
            if best_buyside_target['priority'] > best_sellside_target['priority']:
                draw_on_liquidity = "Buyside"
                draw_reason = f"Draw is on {best_buyside_target['description']} at {best_buyside_target['level']:.5f}"
            else:
                draw_on_liquidity = "Sellside"
                draw_reason = f"Draw is on {best_sellside_target['description']} at {best_sellside_target['level']:.5f}"
        elif best_buyside_target:
            draw_on_liquidity = "Buyside"
            draw_reason = f"Draw is on {best_buyside_target['description']} at {best_buyside_target['level']:.5f}"
        elif best_sellside_target:
            draw_on_liquidity = "Sellside"
            draw_reason = f"Draw is on {best_sellside_target['description']} at {best_sellside_target['level']:.5f}"

        final_bias = 'neutral'
        reasons = []
        
        if htf_order_flow == 'bullish':
            reasons.append("Primary Factor: Daily Order Flow is Bullish.")
            if draw_on_liquidity == "Buyside":
                reasons.append(f"Confluence: Draw on Buyside Liquidity Confirms Bias. ({draw_reason})")
                final_bias = 'bullish'
            else:
                reasons.append("Conflict: Order flow is bullish but draw is on sellside. Bias is neutral.")
                final_bias = 'neutral'
        
        elif htf_order_flow == 'bearish':
            reasons.append("Primary Factor: Daily Order Flow is Bearish.")
            if draw_on_liquidity == "Sellside":
                reasons.append(f"Confluence: Draw on Sellside Liquidity Confirms Bias. ({draw_reason})")
                final_bias = 'bearish'
            else:
                reasons.append("Conflict: Order flow is bearish but draw is on buyside. Bias is neutral.")
                final_bias = 'neutral'
        else:
            reasons.append("Primary Factor: Daily Order Flow is Neutral.")

        bias_details = {
            'htf_order_flow': htf_order_flow,
            'liquidity_draw_direction': draw_on_liquidity,
            'liquidity_draw_reason': draw_reason,
            'final_bias_decision': final_bias,
            'reasoning': ' | '.join(reasons)
        }
        
        return final_bias, bias_details

    def _analyze_session_po3(self, ohlc_df: pd.DataFrame, session_context: dict, symbol: str, daily_df: pd.DataFrame = None, swings: pd.DataFrame = None) -> Tuple[str, Dict]:
        """Orchestrates bias analysis and identifies PO3 phase based on session context and manipulation patterns."""
        daily_bias, bias_details = self._determine_daily_bias(ohlc_df, symbol, daily_df)
        manipulation_details = self._check_manipulation_patterns(ohlc_df, daily_bias, session_context, swings)

        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        
        po3_analysis = {
            'current_phase': 'unknown',
            'type': 'session_based',
            'manipulation': {'detected': False},
            'bias_details': bias_details
        }
        
        if 19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 1:
            po3_analysis['current_phase'] = 'accumulation'
            po3_analysis['session'] = 'Asian'
            logger.debug(f"{symbol}: Asian Session - Accumulation Phase")
            
        elif 2 <= current_ny_hour < 5:
            if manipulation_details:
                po3_analysis['current_phase'] = 'manipulation'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
                po3_analysis['type'] = manipulation_details.get('type', 'unknown')
                logger.debug(f"{symbol}: London Session - Manipulation Active: {manipulation_details.get('type')}")
            else:
                po3_analysis['current_phase'] = 'manipulation_pending'
                logger.debug(f"{symbol}: London Session - Waiting for Manipulation")
            po3_analysis['session'] = 'London'
            
        elif 7 <= current_ny_hour < 11:
            if manipulation_details:
                po3_analysis['current_phase'] = 'distribution'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
                po3_analysis['type'] = manipulation_details.get('type', 'unknown')
                logger.debug(f"{symbol}: NY Session - Distribution Phase after {manipulation_details.get('type')}")
            else:
                if session_context.get('in_killzone') and daily_bias != 'neutral':
                    po3_analysis['current_phase'] = 'distribution_continuation'
                    logger.debug(f"{symbol}: NY Session - Distribution Continuation (no new manipulation)")
                else:
                    po3_analysis['current_phase'] = 'distribution_pending'
                    logger.debug(f"{symbol}: NY Session - Waiting for Setup")
            po3_analysis['session'] = 'NewYork'
            
        elif 10 <= current_ny_hour < 12:
            if manipulation_details:
                po3_analysis['current_phase'] = 'london_close_distribution'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
                po3_analysis['type'] = manipulation_details.get('type', 'unknown')
                logger.debug(f"{symbol}: London Close Session - Distribution Phase")
            else:
                po3_analysis['current_phase'] = 'london_close_pending'
                logger.debug(f"{symbol}: London Close Session - Waiting for Setup")
            po3_analysis['session'] = 'LondonClose'
            
        else:
            if manipulation_details:
                po3_analysis['current_phase'] = 'post_distribution'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
            else:
                po3_analysis['current_phase'] = 'consolidation'
            po3_analysis['session'] = 'Other'
            logger.debug(f"{symbol}: Outside main sessions - {po3_analysis['current_phase']}")
        
        if manipulation_details:
            po3_analysis['manipulation'] = manipulation_details
            po3_analysis['manipulation']['detected'] = True
            po3_analysis['type'] = manipulation_details.get('type')
        
        logger.info(f"{symbol}: PO3 Phase: {po3_analysis['current_phase']} | Session: {po3_analysis.get('session', 'Unknown')} | Bias: {daily_bias}")
        
        return daily_bias, po3_analysis
            
    def _analyze_daily_order_flow(self, daily_df):
        """Analyze daily timeframe order flow."""
        if daily_df is None or daily_df.empty or len(daily_df) < 20:
            logger.warning("Insufficient daily data for order flow analysis")
            return 'neutral'
        
        swings = smc.swing_highs_lows(daily_df, swing_length=5)
        
        if swings.empty:
            return 'neutral'
        
        swing_highs = swings[swings['HighLow'] == 1]['Level'].dropna()
        swing_lows = swings[swings['HighLow'] == -1]['Level'].dropna()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral'
        
        last_two_highs = swing_highs.tail(2).values
        last_two_lows = swing_lows.tail(2).values
        
        if last_two_highs[1] > last_two_highs[0] and last_two_lows[1] > last_two_lows[0]:
            logger.info(f"Daily Order Flow: BULLISH (HH: {last_two_highs[0]:.5f} -> {last_two_highs[1]:.5f}, HL: {last_two_lows[0]:.5f} -> {last_two_lows[1]:.5f})")
            return 'bullish'
        elif last_two_highs[1] < last_two_highs[0] and last_two_lows[1] < last_two_lows[0]:
            logger.info(f"Daily Order Flow: BEARISH (LH: {last_two_highs[0]:.5f} -> {last_two_highs[1]:.5f}, LL: {last_two_lows[0]:.5f} -> {last_two_lows[1]:.5f})")
            return 'bearish'
        
        return 'neutral'
   
    def _analyze_premium_discount(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame = None) -> Dict:
        """Use SMC retracements for premium/discount analysis."""
        try:
            if swings is None or swings.empty:
                swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
                
            retracements = smc.retracements(ohlc_df, swings)
            
            if retracements.empty:
                return {}
                
            current_retracement = retracements['CurrentRetracement%'].iloc[-1]
            direction = retracements['Direction'].iloc[-1]
            
            if pd.isna(current_retracement):
                zone = 'unknown'
            elif current_retracement < 38.2:
                zone = 'deep_discount' if direction == 1 else 'deep_premium'
            elif current_retracement < 50:
                zone = 'discount' if direction == 1 else 'premium'  
            elif current_retracement < 61.8:
                zone = 'premium' if direction == 1 else 'discount'
            else:
                zone = 'deep_premium' if direction == 1 else 'deep_discount'
                
            return {
                'current_zone': zone,
                'retracement_percent': current_retracement,
                'direction': direction,
                'deepest_retracement': retracements['DeepestRetracement%'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error in premium/discount analysis: {e}")
            return {}
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context, swings):
        """Checks for manipulation patterns with corrected Judas Swing timing."""
        if daily_bias == 'neutral' or swings.empty:
            return None

        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour

        if 0 <= current_ny_hour < 5 and session_context.get('last_asian_range'):
            asian_high = session_context['last_asian_range']['high']
            asian_low = session_context['last_asian_range']['low']
            
            swings_before_now = swings[swings.index < len(ohlc_df) - 1].dropna()

            if daily_bias == 'bearish':
                search_start_time = session_context['last_asian_range']['end_time_utc']
                sweep_search_df = ohlc_df[ohlc_df.index >= search_start_time]
                sweep_up_candles = sweep_search_df[sweep_search_df['high'] > asian_high]
                
                if not sweep_up_candles.empty:
                    sweep_candle_time = sweep_up_candles.index[0]
                    sweep_index = ohlc_df.index.get_loc(sweep_candle_time)
                    
                    opposing_swings = swings_before_now[swings_before_now['HighLow'] == -1]
                    if not opposing_swings.empty:
                        mss_level = opposing_swings.iloc[-1]['Level']
                        
                        if self._confirm_displacement_and_mss(ohlc_df, sweep_index, 'bearish', swings):
                            logger.debug(f"Bearish Judas Swing detected at {current_ny_hour}:00 NY time")
                            return {'type': 'bearish_judas', 'level': sweep_up_candles['high'].iloc[0], 
                                'index': sweep_index, 'mss_level': mss_level, 'detected': True}

            elif daily_bias == 'bullish':
                search_start_time = session_context['last_asian_range']['end_time_utc']
                sweep_search_df = ohlc_df[ohlc_df.index >= search_start_time]
                sweep_down_candles = sweep_search_df[sweep_search_df['low'] < asian_low]

                if not sweep_down_candles.empty:
                    sweep_candle_time = sweep_down_candles.index[0]
                    sweep_index = ohlc_df.index.get_loc(sweep_candle_time)

                    opposing_swings = swings_before_now[swings_before_now['HighLow'] == 1]
                    if not opposing_swings.empty:
                        mss_level = opposing_swings.iloc[-1]['Level']

                        if self._confirm_displacement_and_mss(ohlc_df, sweep_index, 'bullish', swings):
                            logger.debug(f"Bullish Judas Swing detected at {current_ny_hour}:00 NY time")
                            return {'type': 'bullish_judas', 'level': sweep_down_candles['low'].iloc[0], 
                                'index': sweep_index, 'mss_level': mss_level, 'detected': True}

        turtle_soup = self._check_turtle_soup_pattern(ohlc_df, daily_bias)
        if turtle_soup:
            return turtle_soup

        liquidity_sweep = self._check_liquidity_sweep(ohlc_df, daily_bias, swings)
        if liquidity_sweep:
            return liquidity_sweep

        return None

    def _check_turtle_soup_pattern(self, ohlc_df, daily_bias):
        """Identifies a Turtle Soup pattern by scanning recent closed candles."""
        scan_window = 5 
        
        if len(ohlc_df) < 20 + scan_window:
            return None
        
        for i in range(-2, -(scan_window + 2), -1):
            bar = ohlc_df.iloc[i]
            lookback_df = ohlc_df.iloc[i-20 : i]
            
            recent_high = lookback_df['high'].max()
            recent_low = lookback_df['low'].min()
            
            if daily_bias == 'bearish' and bar['high'] > recent_high:
                if bar['close'] < bar['open'] and bar['close'] < recent_high:
                    return {'type': 'bearish_turtle_soup', 'level': bar['high'], 'swept_level': recent_high, 'index': len(ohlc_df) + i, 'detected': True}
            
            elif daily_bias == 'bullish' and bar['low'] < recent_low:
                if bar['close'] > bar['open'] and bar['close'] > recent_low:
                    return {'type': 'bullish_turtle_soup', 'level': bar['low'], 'swept_level': recent_low, 'index': len(ohlc_df) + i, 'detected': True}
        
        return None
    
    def _check_liquidity_sweep(self, ohlc_df, daily_bias, swings):
        """Identifies a sweep of a MAJOR swing point followed by MSS confirmation."""
        if len(ohlc_df) < 10 or swings.empty:
            return None
        
        recent_swings = swings.dropna().tail(10)
        
        for i in range(-5, -1):
            current_candle = ohlc_df.iloc[i]
            current_index = len(ohlc_df) + i
            swings_before_candle = recent_swings[recent_swings.index < current_index]
            if swings_before_candle.empty:
                continue

            if daily_bias == 'bullish':
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['low'] < swept_level:
                    if self._confirm_displacement_and_mss(ohlc_df, current_index, 'bullish', swings):
                        logger.info(f"Bullish liquidity sweep confirmed with MSS and Displacement at index {current_index}")
                        opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                        mss_level = opposing_swings.iloc[-1]['Level'] if not opposing_swings.empty else 0
                        return {'type': 'bullish_liquidity_sweep_mss', 'level': current_candle['low'], 'swept_level': swept_level, 'index': current_index, 'mss_level': mss_level, 'detected': True}

            elif daily_bias == 'bearish':
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['high'] > swept_level:
                    if self._confirm_displacement_and_mss(ohlc_df, current_index, 'bearish', swings):
                        logger.info(f"Bearish liquidity sweep confirmed with MSS and Displacement at index {current_index}")
                        opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                        mss_level = opposing_swings.iloc[-1]['Level'] if not opposing_swings.empty else 0
                        return {'type': 'bearish_liquidity_sweep_mss', 'level': current_candle['high'], 'swept_level': swept_level, 'index': current_index, 'mss_level': mss_level, 'detected': True}
        
        return None
        
    def _get_ict_structure(self, ohlc_df, swings: pd.DataFrame, daily_bias: str):
        """Detects BOS and CHoCH based on swing breaks validated by displacement."""
        n = len(ohlc_df)
        bos, choch, level, broken_index = [np.full(n, np.nan) for _ in range(4)]
        
        relevant_swings = swings.dropna().tail(50)
        all_fvgs = self._get_fvgs(ohlc_df)

        for swing_idx, swing in relevant_swings.iterrows():
            swing_level, swing_type = swing['Level'], swing['HighLow']
            try:
                swing_position = ohlc_df.index.get_loc(swing_idx)
            except KeyError:
                continue
            
            post_swing_data = ohlc_df.iloc[swing_position + 1:]
            if post_swing_data.empty:
                continue

            breaking_candles = post_swing_data[post_swing_data['close'] > swing_level] if swing_type == 1 else post_swing_data[post_swing_data['close'] < swing_level]
            if breaking_candles.empty:
                continue
            
            break_candle_idx = ohlc_df.index.get_loc(breaking_candles.index[0])

            fvg_confirmed = False
            break_direction = 'bullish' if swing_type == 1 else 'bearish'
            for fvg in all_fvgs:
                if (fvg['type'] == break_direction and
                        swing_position < fvg['index'] <= break_candle_idx):
                    fvg_confirmed = True
                    break
            
            if fvg_confirmed:
                is_bos = (daily_bias == break_direction)
                target_array = bos if is_bos else choch
                signal_value = 1 if break_direction == 'bullish' else -1
                
                target_array[swing_position] = signal_value
                level[swing_position] = swing_level
                broken_index[swing_position] = break_candle_idx

        return pd.DataFrame({'BOS': bos, 'CHOCH': choch, 'Level': level, 'BrokenIndex': broken_index})
    
    def _create_empty_structure(self, length):
        """Create empty structure DataFrame."""
        return pd.DataFrame({
            'BOS': np.full(length, np.nan),
            'CHOCH': np.full(length, np.nan),
            'Level': np.full(length, np.nan),
            'BrokenIndex': np.full(length, np.nan)
        })
                
    def _get_order_blocks(self, ohlc_df, swings):
        """Get refined order blocks."""
        try:
            obs = smc.ob(ohlc_df, swings, close_mitigation=False)
            if obs.empty:
                return []
            
            recent_obs = obs[obs.index >= len(ohlc_df) - 50]
            unmitigated = recent_obs[
                recent_obs['OB'].notna() & 
                recent_obs['MitigatedIndex'].isna()
            ]
            
            ob_list = []
            for idx, ob in unmitigated.iterrows():
                ob_list.append({
                    'index': idx,
                    'type': 'bullish' if ob['OB'] == 1 else 'bearish',
                    'top': ob['Top'],
                    'bottom': ob['Bottom'],
                    'volume': ob.get('OBVolume', 0)
                })
            
            return ob_list
            
        except Exception as e:
            logger.error(f"Error getting order blocks: {e}")
            return []
    
    def _get_fvgs(self, ohlc_df):
        """Get fair value gaps."""
        try:
            fvgs = smc.fvg(ohlc_df, join_consecutive=True)
            if fvgs.empty:
                return []
            
            recent_fvgs = fvgs[fvgs.index >= len(ohlc_df) - 50]
            unmitigated = recent_fvgs[
                recent_fvgs['FVG'].notna() & 
                recent_fvgs['MitigatedIndex'].isna()
            ]
            
            fvg_list = []
            for idx, fvg in unmitigated.iterrows():
                fvg_list.append({
                    'index': idx,
                    'type': 'bullish' if fvg['FVG'] == 1 else 'bearish',
                    'top': fvg['Top'],
                    'bottom': fvg['Bottom']
                })
            
            return fvg_list
            
        except Exception as e:
            logger.error(f"Error getting FVGs: {e}")
            return []
        
    def _calculate_ote_zones(self, ohlc_df, daily_bias, manipulation):
        """Calculate OTE zones following ICT's methodology."""
        if not manipulation.get('detected'):
            return []

        ote_zones = []
        manipulation_level = manipulation.get('level')
        manipulation_index = manipulation.get('index', -1)

        if manipulation_index < 0 or manipulation_index >= len(ohlc_df) - 1:
            logger.debug("OTE Calc: Manipulation is too recent or index is invalid; no data to form range.")
            return []

        post_manipulation_data = ohlc_df.iloc[manipulation_index + 1:]
        if post_manipulation_data.empty:
            logger.debug("OTE Calc: No candles found after manipulation event to form a range.")
            return []

        if daily_bias == 'bullish':
            swing_low = manipulation_level
            swing_high = post_manipulation_data['high'].max()
            
            if swing_high > swing_low:
                range_size = swing_high - swing_low
                                
                ote_zones.append({
                    'direction': 'bullish',
                    'high': swing_low + (range_size * 0.38),
                    'sweet': swing_low + (range_size * 0.295),
                    'low': swing_low + (range_size * 0.21),
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'equilibrium': swing_low + (range_size * 0.5)
                })
                logger.debug(f"Bullish OTE: Low={swing_low:.5f}, High={swing_high:.5f}, "
                            f"OTE Zone={ote_zones[-1]['low']:.5f} to {ote_zones[-1]['high']:.5f}")
            else:
                logger.debug(f"OTE Calc (Bullish): No upward move after manipulation. High {swing_high:.5f} <= Low {swing_low:.5f}")

        elif daily_bias == 'bearish':
            swing_high = manipulation_level
            swing_low = post_manipulation_data['low'].min()
            
            if swing_high > swing_low:
                range_size = swing_high - swing_low
                
                ote_zones.append({
                    'direction': 'bearish',
                    'high': swing_high - (range_size * 0.21),
                    'sweet': swing_high - (range_size * 0.295),
                    'low': swing_high - (range_size * 0.38),
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'equilibrium': swing_high - (range_size * 0.5)
                })
                logger.debug(f"Bearish OTE: High={swing_high:.5f}, Low={swing_low:.5f}, "
                            f"OTE Zone={ote_zones[-1]['low']:.5f} to {ote_zones[-1]['high']:.5f}")
            else:
                logger.debug(f"OTE Calc (Bearish): No downward move after manipulation. Low {swing_low:.5f} >= High {swing_high:.5f}")
            
        return ote_zones

    def _analyze_market_phase(self, ohlc_df, manipulation, daily_bias):
        """Determine current ICT PO3 phase and whether entry is viable."""
        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        current_ny_minute = current_ny.minute
        current_price = ohlc_df['close'].iloc[-1]

        logger.debug(f"Analyzing market phase for NY time: {current_ny.strftime('%H:%M')}, Hour={current_ny_hour}")

        manipulation_detected = manipulation.get('detected', False)
        manipulation_index = manipulation.get('index', -1)
        manipulation_price = manipulation.get('level', current_price)

        phase = 'unknown'
        can_enter = False
        session_name = 'Other'
        
        if 2 <= current_ny_hour < 5:
            session_name = 'London'
            if manipulation_detected:
                phase = 'manipulation_complete'
                can_enter = True
            else:
                phase = 'manipulation_pending'
                can_enter = False

        elif (current_ny_hour == 8 and current_ny_minute >= 30) or (9 <= current_ny_hour < 11):
            session_name = 'NewYork'
            if manipulation_detected:
                phase = 'distribution'
                can_enter = True
            else:
                phase = 'distribution_pending'
                can_enter = False

        elif 19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 2:
            session_name = 'Asian'
            phase = 'accumulation'
            can_enter = False
        
        else:
            session_name = 'Other'
            if manipulation_detected:
                phase = 'post_distribution'
            else:
                phase = 'consolidation'
            can_enter = False

        retracement_percent = 0
        extreme_price = current_price
        if manipulation_detected:
            post_manipulation = ohlc_df.iloc[manipulation_index:]
            if daily_bias == 'bullish' and current_price > manipulation_price:
                extreme_price = post_manipulation['high'].max()
                retracement_range = extreme_price - manipulation_price
                current_pullback = extreme_price - current_price
                if retracement_range > 0:
                    retracement_percent = (current_pullback / retracement_range) * 100
            elif daily_bias == 'bearish' and current_price < manipulation_price:
                extreme_price = post_manipulation['low'].min()
                retracement_range = manipulation_price - extreme_price
                current_pullback = current_price - extreme_price
                if retracement_range > 0:
                    retracement_percent = (current_pullback / retracement_range) * 100

        if (daily_bias == 'bullish' and manipulation_detected and current_price < manipulation_price) or \
           (daily_bias == 'bearish' and manipulation_detected and current_price > manipulation_price):
            phase = 'invalidated'
            can_enter = False
            
        return {
            'phase': phase,
            'can_enter': can_enter,
            'retracement_percent': round(retracement_percent, 1),
            'manipulation_price': manipulation_price,
            'extreme_price': extreme_price,
            'bars_since_manipulation': len(ohlc_df) - 1 - manipulation_index if manipulation_index >= 0 else 0,
            'current_session': session_name
        }
        
    def _confirm_displacement_and_mss(self, ohlc_df: pd.DataFrame, reaction_index: int, bias: str, swings: pd.DataFrame) -> bool:
        """Confirms displacement and MSS by leveraging existing FVG detection."""
        if not self.config.REQUIRE_ENTRY_CONFIRMATION:
            return True

        confirmation_window_size = 5
        start_idx = reaction_index + 1
        end_idx = start_idx + confirmation_window_size
        
        if end_idx > len(ohlc_df):
            logger.debug("MSS/Displacement: Not enough candles after POI reaction to confirm signal.")
            return False
            
        confirmation_df = ohlc_df.iloc[start_idx:end_idx]
        if confirmation_df.empty:
            return False

        swings_before_reaction = swings[swings.index < reaction_index].dropna()
        if swings_before_reaction.empty:
            return False
            
        mss_level = None
        if bias == 'bullish':
            recent_swing_highs = swings_before_reaction[swings_before_reaction['HighLow'] == 1]
            if not recent_swing_highs.empty: mss_level = recent_swing_highs['Level'].iloc[-1]
        else:
            recent_swing_lows = swings_before_reaction[swings_before_reaction['HighLow'] == -1]
            if not recent_swing_lows.empty: mss_level = recent_swing_lows['Level'].iloc[-1]
        
        if mss_level is None:
            logger.debug("MSS/Displacement: Could not determine a valid MSS level to break.")
            return False

        mss_confirmed = False
        mss_break_index = -1
        for i, candle in confirmation_df.iterrows():
            candle_pos = ohlc_df.index.get_loc(i)
            if bias == 'bullish' and candle['close'] > mss_level and candle['close'] > candle['open']:
                mss_confirmed = True
                mss_break_index = candle_pos
                logger.debug(f"MSS Confirmed: Bullish body close at index {mss_break_index} above level {mss_level:.5f}")
                break
            elif bias == 'bearish' and candle['close'] < mss_level and candle['close'] < candle['open']:
                mss_confirmed = True
                mss_break_index = candle_pos
                logger.debug(f"MSS Confirmed: Bearish body close at index {mss_break_index} below level {mss_level:.5f}")
                break
        
        if not mss_confirmed:
            logger.debug(f"MSS/Displacement: FAILED. No candle body closed past the MSS level of {mss_level:.5f}.")
            return False
            
        all_fvgs = self._get_fvgs(ohlc_df)
        fvg_confirmed = False

        for fvg in all_fvgs:
            fvg_index = fvg['index']
            fvg_type_matches_bias = (fvg['type'] == bias)

            if fvg_type_matches_bias and (reaction_index < fvg_index <= mss_break_index):
                fvg_confirmed = True
                logger.debug(f"Displacement confirmed: Found a matching {fvg['type']} FVG at index {fvg_index} within the displacement move.")
                break

        if not fvg_confirmed:
            logger.debug(f"MSS/Displacement: FAILED. The move breaking structure did not leave a Fair Value Gap (FVG) within the confirmation window.")
            return False

        logger.info(f"ICT CONFIRMATION SUCCESS: MSS confirmed at {mss_level:.5f} and Displacement confirmed by FVG creation.")
        return True

class ICTSignalGenerator:
    """Generates trading signals following the ICT narrative sequence."""
    
    def __init__(self, config):
        self.config = config
        self.analyzer = ICTAnalyzer(config)
        self.liquidity_detector = LiquidityDetector()
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str, spread: float, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """Generate trading signal following the ICT narrative."""
        analysis = self.analyzer.analyze(ohlc_df, symbol, daily_df, h4_df) 
        if not analysis:
            return None, None, None, None
        
        daily_bias = analysis['daily_bias']
        london_manipulation = analysis['manipulation']
        current_price = analysis['current_price']
        session = analysis['session']
        swings = analysis.get('swings', pd.DataFrame())
        require_killzone = getattr(self.config, 'REQUIRE_KILLZONE', True)
               
        narrative = self._build_narrative(analysis)
        
        if daily_bias == 'neutral':
            logger.debug(f"{symbol}: No clear daily bias from daily chart.")
            return None, None, None, None
        
        market_phase = self.analyzer._analyze_market_phase(ohlc_df, london_manipulation, daily_bias)
        logger.info(f"{symbol}: Phase: {market_phase['phase']} | Retracement: {market_phase.get('retracement_percent', 0):.1f}%")
        
        if market_phase['phase'] == 'invalidated':
            return None, None, None, None
        
        entry_signal, sl_price, tp_price = None, None, None
        
        if require_killzone and not session.get('in_killzone'):
            return None, None, None, None

        killzone_name = session.get('killzone_name')
        
        if killzone_name == 'London':
            if london_manipulation.get('detected'):
                manipulation_type = london_manipulation.get('type', 'manipulation')
                logger.info(f"{symbol}: LondonKZ - Looking for entry after '{manipulation_type}'.")
                entry_signal, sl_price, tp_price = self._setup_judas_entry(
                    daily_bias, current_price, london_manipulation, ohlc_df, analysis
                )
                narrative.entry_model = f"LONDON_{manipulation_type.upper()}"
                    
        elif killzone_name == 'NewYork':
            
            if london_manipulation.get('detected'):
                logger.info(f"{symbol}: NYKZ - Activating CONTINUATION model based on prior London manipulation.")
                entry_signal, sl_price, tp_price = self._find_continuation_entry(analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation)
                if entry_signal:
                    narrative.entry_model = "NY_CONTINUATION"

            else:
                logger.info(f"{symbol}: NYKZ - No prior manipulation. Activating REVERSAL model search.")
                new_manipulation = self.analyzer._check_manipulation_patterns(ohlc_df, daily_bias, session, swings)
                
                if new_manipulation and new_manipulation.get('detected'):
                    logger.info(f"{symbol}: NEW manipulation '{new_manipulation.get('type')}' detected in NYKZ!")
                    entry_signal, sl_price, tp_price = self._setup_judas_entry(
                        daily_bias, current_price, new_manipulation, ohlc_df, analysis
                    )
                    narrative.entry_model = f"NY_{new_manipulation.get('type', 'REVERSAL').upper()}"
                    narrative.manipulation_confirmed = True
                    narrative.manipulation_level = new_manipulation.get('level', 0)

        elif killzone_name == 'LondonClose':
            logger.info(f"{symbol}: London Close KZ - Activating ICT-aligned logic...")

            logger.debug(f"{symbol}: LCKZ - Checking for PRIMARY profile (Continuation).")
            entry_signal, sl_price, tp_price = self._find_continuation_entry(
                analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation
            )
            if entry_signal:
                narrative.entry_model = "LONDON_CLOSE_CONTINUATION"
                logger.info(f"{symbol}: LCKZ - Found valid CONTINUATION setup.")

            if not entry_signal:
                logger.debug(f"{symbol}: LCKZ - No continuation setup. Checking for SECONDARY profile (Reversal).")
                entry_signal, sl_price, tp_price = self._check_london_close_reversal(analysis, ohlc_df, swings)
                if entry_signal:
                    narrative.entry_model = "LONDON_CLOSE_REVERSAL"
                    logger.info(f"{symbol}: LCKZ - Found valid REVERSAL setup.")
            
            if not entry_signal:
                logger.debug(f"{symbol}: LCKZ - No valid Continuation or Reversal setup found. Waiting for next session.")
            
        if entry_signal:
            if not self._is_safe_entry_location(ohlc_df, current_price, sl_price, daily_bias) or \
               not self._validate_levels(entry_signal, current_price, sl_price, tp_price):
                logger.warning(f"{symbol}: Entry location or SL/TP levels are invalid - skipping.")
                return None, None, None, None
            
            logger.info(f"ICT Signal Generated: {entry_signal} {symbol} @ {current_price:.5f}")
            logger.info(f"  Narrative: {daily_bias.upper()} | {narrative.entry_model}")
            logger.info(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            return entry_signal, sl_price, tp_price, narrative        
        
        return None, None, None, None
    
    def _check_london_close_reversal(self, analysis, ohlc_df, swings) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Checks for the London Close Reversal Profile."""
        logger.debug("LCKZ: Checking for Reversal Profile (Sweep of NY High/Low -> MSS).")

        ny_session_start_hour = 8
        ny_session_start_minute = 30
        ny_session_end_hour = 10 

        current_ny_time = ohlc_df.index[-1].astimezone(self.config.NY_TIMEZONE)
        today_ny_date = current_ny_time.date()

        ny_session_start_ny = current_ny_time.replace(
            year=today_ny_date.year, month=today_ny_date.month, day=today_ny_date.day,
            hour=ny_session_start_hour, minute=ny_session_start_minute, second=0, microsecond=0
        )
        ny_session_end_ny = ny_session_start_ny.replace(hour=ny_session_end_hour, minute=0)

        ny_session_start_utc = ny_session_start_ny.astimezone(pytz.UTC)
        ny_session_end_utc = ny_session_end_ny.astimezone(pytz.UTC)

        ny_session_data = ohlc_df[(ohlc_df.index >= ny_session_start_utc) & (ohlc_df.index < ny_session_end_utc)]

        if ny_session_data.empty:
            logger.debug("LCKZ Reversal: No data found for NY session to determine high/low.")
            return None, None, None

        ny_high = ny_session_data['high'].max()
        ny_low = ny_session_data['low'].min()
        logger.debug(f"LCKZ Reversal: NY Session Range to Sweep -> High={ny_high:.5f}, Low={ny_low:.5f}")

        lckz_data = ohlc_df[ohlc_df.index >= ny_session_end_utc]
        
        sweep_high = lckz_data['high'].max() > ny_high
        sweep_low = lckz_data['low'].min() < ny_low
        
        daily_bias = analysis['daily_bias']

        if daily_bias == 'bullish' and sweep_low:
            sweep_candle_idx = lckz_data[lckz_data['low'] < ny_low].index[0]
            sweep_candle_pos = ohlc_df.index.get_loc(sweep_candle_idx)
            
            if self.analyzer._confirm_displacement_and_mss(ohlc_df, sweep_candle_pos, 'bullish', swings):
                logger.info("LCKZ Reversal: CONFIRMED. Sweep of NY Low followed by Bullish Displacement & MSS.")
                fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], 'bullish', analysis['current_price'], analysis['manipulation'], ohlc_df, swings)
                if fvg_entry:
                    return self._setup_fvg_entry('bullish', analysis['current_price'], fvg_entry, {'level': lckz_data['low'].min()}, ohlc_df, analysis)

        elif daily_bias == 'bearish' and sweep_high:
            sweep_candle_idx = lckz_data[lckz_data['high'] > ny_high].index[0]
            sweep_candle_pos = ohlc_df.index.get_loc(sweep_candle_idx)

            if self.analyzer._confirm_displacement_and_mss(ohlc_df, sweep_candle_pos, 'bearish', swings):
                logger.info("LCKZ Reversal: CONFIRMED. Sweep of NY High followed by Bearish Displacement & MSS.")
                fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], 'bearish', analysis['current_price'], analysis['manipulation'], ohlc_df, swings)
                if fvg_entry:
                    return self._setup_fvg_entry('bearish', analysis['current_price'], fvg_entry, {'level': lckz_data['high'].max()}, ohlc_df, analysis)

        return None, None, None
    
    def _find_poi_reaction_index(self, ohlc_df: pd.DataFrame, poi_range: Dict[str, float], lookback: int = 10) -> Optional[int]:
        """Scans backwards to find the integer index of the candle that first touched a POI."""
        if 'top' not in poi_range or 'bottom' not in poi_range:
            return None
            
        for i in range(-1, -(lookback + 1), -1):
            if abs(i) >= len(ohlc_df):
                break
                
            candle = ohlc_df.iloc[i]
            
            if candle['high'] >= poi_range['bottom'] and candle['low'] <= poi_range['top']:
                return len(ohlc_df) + i
                
        return None

    def _find_continuation_entry(self, analysis, daily_bias, current_price, ohlc_df, spread, manipulation):
        """Helper to find OTE or FVG entry confirmed by Displacement and MSS."""
        swings = analysis.get('swings', pd.DataFrame())

        ote_zones = analysis['ote_zones']
        ote_zone = next((z for z in ote_zones if z['direction'].lower() == daily_bias), None)
        
        if ote_zone and self._is_price_in_ote(current_price, [ote_zone], daily_bias):
            reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': ote_zone['high'], 'bottom': ote_zone['low']})
            
            if reaction_index is not None and self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                logger.info(f"{analysis['symbol']}: OTE continuation entry confirmed with Displacement & MSS.")
                return self._setup_ote_entry(daily_bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis)

        fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], daily_bias, current_price, manipulation, ohlc_df, swings)
        
        if fvg_entry:
            reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': fvg_entry['top'], 'bottom': fvg_entry['bottom']})

            if reaction_index is not None and self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                logger.info(f"{analysis['symbol']}: FVG continuation entry confirmed with Displacement & MSS.")
                return self._setup_fvg_entry(daily_bias, current_price, fvg_entry, manipulation, ohlc_df, analysis)

        return None, None, None
    
    def _setup_fvg_entry(self, bias, current_price, fvg_entry, manipulation, ohlc_df, analysis):
        """Setup entry from a Fair Value Gap."""
        if bias == 'bullish':
            signal = "BUY"
            sl_price = min(fvg_entry['bottom'], manipulation['level']) - self._calculate_atr_buffer(ohlc_df)
        else:
            signal = "SELL"
            sl_price = max(fvg_entry['top'], manipulation['level']) + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        return signal, sl_price, tp_price
        
    def _find_retracement_fvg(self, fair_value_gaps, daily_bias, current_price, manipulation, ohlc_df, swings):
        """Find FVGs suitable for continuation entries following ICT methodology."""
        if not fair_value_gaps or ohlc_df.empty:
            return None
        
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14).iloc[-1]
        manipulation_index = manipulation.get('index', -1)
        
        premium_discount = self.analyzer._analyze_premium_discount(ohlc_df, swings=swings)
        equilibrium = premium_discount.get('equilibrium', 0)
        
        if equilibrium == 0:
            logger.warning("Could not determine equilibrium for FVG selection")
            return None
        
        logger.debug(f"FVG Search: Bias={daily_bias}, Current Price={current_price:.5f}, "
                    f"Equilibrium={equilibrium:.5f}, Current Zone={premium_discount.get('current_zone', 'unknown')}")
        
        valid_fvgs = []
        
        for fvg in fair_value_gaps:
            if (daily_bias == 'bullish' and fvg['type'] != 'bullish') or \
            (daily_bias == 'bearish' and fvg['type'] != 'bearish'):
                continue
            
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            fvg_midpoint = (fvg_top + fvg_bottom) / 2
            fvg_size = fvg_top - fvg_bottom
            
            if daily_bias == 'bullish':
                if fvg_midpoint > equilibrium:
                    logger.debug(f"Skipping FVG at {fvg_midpoint:.5f} - in premium zone (above {equilibrium:.5f})")
                    continue
                    
                if current_price < fvg_top:
                    logger.debug(f"Skipping FVG - price {current_price:.5f} already below FVG top {fvg_top:.5f}")
                    continue
                    
            else:
                if fvg_midpoint < equilibrium:
                    logger.debug(f"Skipping FVG at {fvg_midpoint:.5f} - in discount zone (below {equilibrium:.5f})")
                    continue
                    
                if current_price > fvg_bottom:
                    logger.debug(f"Skipping FVG - price {current_price:.5f} already above FVG bottom {fvg_bottom:.5f}")
                    continue
            
            if fvg_size > atr * 2:
                logger.debug(f"Skipping large FVG - size {fvg_size:.5f} > 2x ATR {atr*2:.5f}")
                continue
            
            if daily_bias == 'bullish':
                distance_to_fvg = current_price - fvg_top
            else:
                distance_to_fvg = fvg_bottom - current_price
            
            if distance_to_fvg > atr * 3:
                logger.debug(f"Skipping distant FVG - distance {distance_to_fvg:.5f} > 3x ATR")
                continue
            
            priority_score = 0
            
            if fvg.get('index', 0) > manipulation_index:
                priority_score += 3
                logger.debug(f"FVG at index {fvg.get('index')} is post-manipulation (+3 priority)")
            
            size_score = 1 - (fvg_size / (atr * 2))
            priority_score += size_score * 2
            
            distance_score = 1 - (distance_to_fvg / (atr * 3))
            priority_score += distance_score
            
            if daily_bias == 'bullish':
                zone_score = (equilibrium - fvg_midpoint) / equilibrium
            else:
                zone_score = (fvg_midpoint - equilibrium) / equilibrium
            priority_score += max(0, zone_score) * 2
            
            valid_fvgs.append({
                'fvg': fvg,
                'priority_score': priority_score,
                'distance': distance_to_fvg,
                'size': fvg_size,
                'midpoint': fvg_midpoint,
                'zone_position': 'discount' if fvg_midpoint < equilibrium else 'premium'
            })
        
        if valid_fvgs:
            valid_fvgs.sort(key=lambda x: x['priority_score'], reverse=True)
            best_fvg = valid_fvgs[0]
            
            logger.info(f"Selected FVG: {best_fvg['fvg']['type']} in {best_fvg['zone_position']} zone, "
                    f"score={best_fvg['priority_score']:.2f}, distance={best_fvg['distance']:.5f}, "
                    f"size={best_fvg['size']:.5f}")
            
            result = best_fvg['fvg'].copy()
            result['consequent_encroachment'] = best_fvg['midpoint']
            return result
        
        logger.debug("No valid FVGs found matching ICT criteria")
        return None
    
    def _is_safe_entry_location(self, ohlc_df, entry_price, sl_price, daily_bias):
        """Intelligently handles trend-following entries during shallow pullbacks."""
        atr = self._calculate_atr_buffer(ohlc_df)
        risk = abs(entry_price - sl_price)

        if risk == 0:
            logger.warning("Risk is zero, trade is invalid.")
            return False

        recent_high = ohlc_df['high'].tail(20).max()
        recent_low = ohlc_df['low'].tail(20).min()
        broader_high = ohlc_df['high'].tail(50).max()
        broader_low = ohlc_df['low'].tail(50).min()

        is_strong_trend_continuation = False
        
        if daily_bias == 'bullish':
            distance_from_high = broader_high - entry_price
            if distance_from_high < atr:
                logger.info(f"Entry is near 50-period highs; considering this a strong trend continuation setup.")
                is_strong_trend_continuation = True
        else:
            distance_from_low = entry_price - broader_low
            if distance_from_low < atr:
                logger.info(f"Entry is near 50-period lows; considering this a strong trend continuation setup.")
                is_strong_trend_continuation = True
        
        if not is_strong_trend_continuation:
            potential_reward = 0
            if daily_bias == 'bullish':
                potential_reward = recent_high - entry_price
            else:
                potential_reward = entry_price - recent_low

            if potential_reward < risk * self.config.MIN_TARGET_RR:
                logger.warning(
                    f"Entry has limited reward potential before hitting recent structure. "
                    f"Reward: {potential_reward:.5f}, Risk: {risk:.5f}. Rejecting trade."
                )
                return False
        
        logger.info(f"Entry location at {entry_price:.5f} with SL at {sl_price:.5f} is validated as safe.")
        return True
    
    def _build_narrative(self, analysis) -> ICTNarrative:
        """Build the complete ICT narrative from analysis."""
        manipulation = analysis['manipulation']
        structure = analysis['structure']
        
        recent_bos = None
        recent_choch = None
        
        if not structure.empty and 'BOS' in structure.columns:
            bos_signals = structure[structure['BOS'].notna()].tail(1)
            if not bos_signals.empty:
                recent_bos = bos_signals.iloc[-1]
                
        if not structure.empty and 'CHOCH' in structure.columns:
            choch_signals = structure[structure['CHOCH'].notna()].tail(1)
            if not choch_signals.empty:
                recent_choch = choch_signals.iloc[-1]
        
        structure_broken = recent_bos is not None or recent_choch is not None
        structure_level = None
        if recent_bos is not None:
            structure_level = recent_bos.get('Level', 0)
        elif recent_choch is not None:
            structure_level = recent_choch.get('Level', 0)
        
        return ICTNarrative(
            daily_bias=analysis['daily_bias'],
            po3_phase=analysis['po3_analysis'].get('current_phase', 'unknown'),
            manipulation_confirmed=manipulation['detected'],
            manipulation_level=manipulation.get('level', 0),
            structure_broken=structure_broken,
            structure_level=structure_level or 0,
            in_killzone=analysis['session']['in_killzone'],
            killzone_name=analysis['session']['killzone_name'] or '',
            ote_zone=analysis['ote_zones'][0] if analysis['ote_zones'] else {},
            order_blocks=analysis['order_blocks'],
            current_price=analysis['current_price'],
            entry_model=''
        )
        
    def _is_price_in_ote(self, current_price, ote_zones, daily_bias):
        """Check if price is within OTE zone."""
        for ote in ote_zones:
            if ote['direction'].lower() == daily_bias:
                if daily_bias == 'bullish':
                    if ote['low'] <= current_price <= ote['high']:
                        return True
                else:
                    if ote['low'] <= current_price <= ote['high']:
                        return True
        return False
    
    def _setup_ote_entry(self, bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis):
        """Setup entry from OTE zone."""
        ote = None
        for zone in ote_zones:
            if zone['direction'].lower() == bias:
                ote = zone
                break
        
        if not ote:
            return None, None, None
        
        if bias == 'bullish':
            signal = "BUY"
            sl_price = min(ote['swing_low'], manipulation.get('level', ote['swing_low']))
            sl_price -= self._calculate_atr_buffer(ohlc_df)
            
        else:
            signal = "SELL"
            sl_price = max(ote['swing_high'], manipulation.get('level', ote['swing_high']))
            sl_price += self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)        
        return signal, sl_price, tp_price
        
    def _setup_ob_entry(self, bias, current_price, order_block, manipulation, ohlc_df, analysis):
        """Setup entry from order block."""
        if bias == 'bullish':
            signal = "BUY"
            sl_price = order_block['bottom'] - self._calculate_atr_buffer(ohlc_df)
        else:
            signal = "SELL"
            sl_price = order_block['top'] + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        
        return signal, sl_price, tp_price
    
    def _setup_judas_entry(self, bias, current_price, manipulation, ohlc_df, analysis):
        """Setup entry from Judas Swing manipulation."""
        if bias == 'bullish':
            signal = "BUY"
            sl_price = manipulation.get('level', current_price) - self._calculate_atr_buffer(ohlc_df)
        else:
            signal = "SELL"
            sl_price = manipulation.get('level', current_price) + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        
        return signal, sl_price, tp_price
    
    def _calculate_atr_buffer(self, ohlc_df):
        """Calculate ATR-based buffer for stop loss."""
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        return atr.iloc[-1] * self.config.SL_ATR_MULTIPLIER
    
    def _calculate_target(self, entry_price, sl_price, bias, analysis):
        """Calculate take profit by targeting liquidity using the LiquidityDetector."""
        risk = abs(entry_price - sl_price)
        
        ohlc_df = analysis.get('ohlc_df')
        session_context = analysis.get('session', {})
        daily_df = analysis.get('daily_df')
        
        if ohlc_df is None:
            logger.warning("No OHLC data available for liquidity analysis, using fallback")
            return self._calculate_target_fallback(entry_price, sl_price, bias, risk)
        
        liquidity_levels = self.liquidity_detector.get_liquidity_levels(
            ohlc_df, session_context, daily_df
        )
        
        min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)
        best_target = self.liquidity_detector.get_target_for_bias(
            bias, liquidity_levels, entry_price, min_rr, sl_price
        )
        
        if best_target:
            target_level = best_target['level']
            logger.info(f"Using {best_target['description']} at {target_level:.5f} for TP (Priority: {best_target['priority']})")
            return target_level
        
        logger.debug("No suitable liquidity targets found, using fixed R:R fallback")
        return self._calculate_target_fallback(entry_price, sl_price, bias, risk)

    def _calculate_target_fallback(self, entry_price, sl_price, bias, risk):
        """Fallback target calculation using fixed R:R."""
        tp_rr_ratio = getattr(self.config, 'TP_RR_RATIO', 1.5)
        logger.debug(f"Using fixed R:R fallback ({tp_rr_ratio}:1)")
        
        if bias == 'bullish':
            return entry_price + (risk * tp_rr_ratio)
        else:
            return entry_price - (risk * tp_rr_ratio)
    
    def _validate_levels(self, signal, entry, sl, tp):
        """Validate entry, SL, and TP levels."""
        if signal == "BUY":
            return sl < entry < tp
        else:
            return tp < entry < sl