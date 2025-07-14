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
from ipda_analyzer import IPDAAnalyzer
from macro_analyzer import MacroAnalyzer
from entry_models import EntryModels

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
        self.ipda_analyzer = IPDAAnalyzer(config, self.liquidity_detector)
        self.macro_analyzer = MacroAnalyzer(config)
        self.entry_models = EntryModels(config, self.liquidity_detector)
        
    def analyze(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Dict:
        """Perform ICT analysis."""
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback:
            logger.warning(f"{symbol}: Insufficient data for ICT analysis")
            return {}
        
        # Foundational Elements
        swings = self._get_swings(ohlc_df)
        session_context = self._get_session_context(ohlc_df)
        if session_context.get('error'):
            logger.warning(f"{symbol}: Could not determine session context. Reason: {session_context['error']}")
            return {}

        # Narrative Core: Bias and Structure
        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_context, symbol, daily_df, swings)
        manipulation = po3_analysis.get('manipulation', {'detected': False})
        structure = self._get_structure(ohlc_df, swings, daily_bias)
        
        # Liquidity Analysis (using reliable structure data)
        liquidity_levels = self.liquidity_detector.get_liquidity_levels(
            ohlc_df, session_context, daily_df, structure=structure
        )

        # Identify Potential Entry Points (PD Arrays) and Context
        order_blocks = self._get_order_blocks(ohlc_df, swings, structure)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        premium_discount_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, daily_bias, manipulation)
        htf_levels = self._get_htf_levels(h4_df)
        
        # Advanced Entry Model Identification
        breaker_blocks = self.entry_models.identify_breaker_blocks(ohlc_df, swings, order_blocks, liquidity_levels)
        mitigation_blocks = self.entry_models.identify_mitigation_blocks(ohlc_df, swings, order_blocks, daily_bias)
        unicorn_setups = self.entry_models.identify_unicorn_setups(fair_value_gaps, breaker_blocks, premium_discount_analysis)
        
        # External and Macro Context
        ipda_analysis = {}
        if daily_df is not None and len(daily_df) >= 60:
            ipda_analysis = self.ipda_analyzer.analyze_ipda_data_ranges(daily_df, symbol, ohlc_df, session_context)
            if ipda_analysis.get('comprehensive_liquidity'):
                liquidity_levels = ipda_analysis['comprehensive_liquidity']
        
        macro_analysis = self.macro_analyzer.get_current_macro_status()
        if macro_analysis['in_macro']:
            macro_setup = self.macro_analyzer.analyze_macro_setup(ohlc_df, symbol)
            if macro_setup:
                macro_analysis.update(macro_setup)
                
        # Assemble the Final Analysis Dictionary
        analysis_result = {
            'symbol': symbol, 'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1], 
            'ohlc_df': ohlc_df, 'daily_df': daily_df,
            'swings': swings, 
            'structure': structure,
            'daily_bias': daily_bias,
            'po3_analysis': po3_analysis, 
            'manipulation': manipulation,
            'order_blocks': order_blocks, 
            'fair_value_gaps': fair_value_gaps,
            'breaker_blocks': breaker_blocks, 
            'mitigation_blocks': mitigation_blocks,
            'unicorn_setups': unicorn_setups, 
            'premium_discount': premium_discount_analysis,
            'liquidity_levels': liquidity_levels, 
            'ote_zones': ote_zones,
            'session': session_context, 
            'htf_levels': htf_levels,
            'ipda_analysis': ipda_analysis, 
            'macro_analysis': macro_analysis,
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
        """Get ICT session context"""
        context = {
            'last_asian_range': None,
            'in_killzone': False,
            'killzone_name': None,
            'error': None
        }
        
        if ohlc_df is not None and ohlc_df.index.tz is None:
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
        """
        Determines bias using HTF Order Flow as the primary driver, and Draw on Liquidity as the contextual target.
        """
        logger.debug(f"\n--- {symbol} ICT BIAS CHECKLIST ---")
        if daily_df is None or daily_df.empty:
            return 'neutral', {'error': 'No daily data available'}
        
        # 1. Primary Factor: Determine the Higher Timeframe Order Flow
        htf_order_flow = self._analyze_daily_order_flow(daily_df)
        
        # If HTF order flow is neutral, there is no institutional direction. Bias is neutral.
        if htf_order_flow == 'neutral':
            bias_details = {
                'htf_order_flow': 'neutral',
                'liquidity_draw_direction': 'None',
                'liquidity_draw_reason': 'HTF order flow is neutral.',
                'final_bias_decision': 'neutral',
                'reasoning': 'Primary Factor: Daily Order Flow is Neutral/Indecisive.'
            }
            return 'neutral', bias_details

        # 2. Secondary Factor: Determine the most likely Draw on Liquidity
        session_context = self._get_session_context(ohlc_df)
        liquidity_levels = self.liquidity_detector.get_preliminary_liquidity(ohlc_df, session_context, daily_df)
        
        best_buyside_target = self.liquidity_detector.get_target_for_bias('bullish', liquidity_levels, ohlc_df['close'].iloc[-1])
        best_sellside_target = self.liquidity_detector.get_target_for_bias('bearish', liquidity_levels, ohlc_df['close'].iloc[-1])

        draw_on_liquidity = "None"
        draw_reason = "No clear high-priority liquidity target."

        # Logic to determine the most probable draw
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3, None: 99}
        buyside_priority = priority_order.get(best_buyside_target.get('priority') if best_buyside_target else None)
        sellside_priority = priority_order.get(best_sellside_target.get('priority') if best_sellside_target else None)

        if best_buyside_target and buyside_priority <= sellside_priority:
            draw_on_liquidity = "Buyside"
            draw_reason = f"Draw is on {best_buyside_target['description']} at {best_buyside_target['level']:.5f}"
        elif best_sellside_target:
            draw_on_liquidity = "Sellside"
            draw_reason = f"Draw is on {best_sellside_target['description']} at {best_sellside_target['level']:.5f}"

        # 3. Final Decision: The HTF Order Flow IS the bias. The draw provides context.
        # This is the key correction. We no longer set bias to neutral if there's a conflict.
        final_bias = htf_order_flow
        reasons = []

        if final_bias == 'bullish':
            reasons.append("Primary Factor: Daily Order Flow is Bullish.")
            if draw_on_liquidity == "Buyside":
                reasons.append("Confluence: Draw on Buyside Liquidity suggests trend continuation.")
            else: # Draw is on Sellside or None
                reasons.append("Context: Draw on Sellside Liquidity suggests a retracement before continuation.")
        
        elif final_bias == 'bearish':
            reasons.append("Primary Factor: Daily Order Flow is Bearish.")
            if draw_on_liquidity == "Sellside":
                reasons.append("Confluence: Draw on Sellside Liquidity suggests trend continuation.")
            else: # Draw is on Buyside or None
                reasons.append("Context: Draw on Buyside Liquidity suggests a retracement before continuation.")

        bias_details = {
            'htf_order_flow': htf_order_flow,
            'liquidity_draw_direction': draw_on_liquidity,
            'liquidity_draw_reason': draw_reason,
            'final_bias_decision': final_bias,
            'reasoning': ' | '.join(reasons)
        }
        
        return final_bias, bias_details

    def _analyze_session_po3(self, ohlc_df, session_context, symbol, daily_df, swings, ipda_analysis=None):
        """Orchestrates bias analysis and identifies PO3 phase based on session context and manipulation patterns."""
        if ipda_analysis and ipda_analysis.get('ipda_bias'):
            ipda_bias = ipda_analysis['ipda_bias']
            if ipda_bias in ['bullish', 'bullish_leaning']:
                daily_bias = 'bullish'
                bias_details = ipda_analysis.get('integration_summary', {})
            elif ipda_bias in ['bearish', 'bearish_leaning']:
                daily_bias = 'bearish'
                bias_details = ipda_analysis.get('integration_summary', {})
            else:
                daily_bias, bias_details = self._determine_daily_bias(ohlc_df, symbol, daily_df)
        else:
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
        """
        Calculates the current trading range's premium, discount, and equilibrium (50%) level.
        """
        try:
            if swings is None or swings.empty:
                swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
            
            valid_swings = swings.dropna()
            if len(valid_swings) < 2:
                return {}

            # 1. Determine the current dealing range from the last two opposing swings
            last_swing = valid_swings.iloc[-1]
            prev_swing = valid_swings.iloc[-2]

            if last_swing['HighLow'] == prev_swing['HighLow']:
                # If last two swings are the same, grab the one before that
                if len(valid_swings) < 3: return {}
                prev_swing = valid_swings.iloc[-3]

            swing_high = max(last_swing['Level'], prev_swing['Level'])
            swing_low = min(last_swing['Level'], prev_swing['Level'])
            
            # 2. Calculate Equilibrium
            equilibrium = (swing_high + swing_low) / 2
            
            # 3. Determine if current price is in premium or discount
            current_price = ohlc_df['close'].iloc[-1]
            current_zone = 'unknown'
            direction = 0 # 1 for bullish range, -1 for bearish range
            
            if last_swing['HighLow'] == 1 and prev_swing['HighLow'] == -1: # Bullish range (low to high)
                direction = 1
                if current_price > equilibrium:
                    current_zone = 'premium'
                else:
                    current_zone = 'discount'
            elif last_swing['HighLow'] == -1 and prev_swing['HighLow'] == 1: # Bearish range (high to low)
                direction = -1
                if current_price < equilibrium:
                    current_zone = 'discount'
                else:
                    current_zone = 'premium'

            # 4. Calculate retracement percentage
            range_size = swing_high - swing_low
            retracement_percent = 0
            if range_size > 0:
                if direction == 1: # Bullish range
                    retracement_percent = ((swing_high - current_price) / range_size) * 100
                elif direction == -1: # Bearish range
                    retracement_percent = ((current_price - swing_low) / range_size) * 100

            return {
                'current_zone': current_zone,
                'retracement_percent': round(retracement_percent, 2),
                'direction': direction,
                'equilibrium': equilibrium, # <-- The FIX: Now returning the equilibrium price
                'swing_high': swing_high,
                'swing_low': swing_low
            }
            
        except Exception as e:
            logger.error(f"Error in premium/discount analysis: {e}", exc_info=True)
            return {}
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context, swings):
        """Checks for manipulation patterns."""
        if daily_bias == 'neutral' or swings.empty:
            return None

        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour

        if 0 <= current_ny_hour < 5 and session_context.get('last_asian_range'):
            asian_high = session_context['last_asian_range']['high']
            asian_low = session_context['last_asian_range']['low']
            
            if len(ohlc_df) > 1:
                cutoff_time = ohlc_df.index[-2]  # Second to last timestamp
                swings_before_now = swings[swings.index < cutoff_time].dropna()
            else:
                swings_before_now = swings.dropna()

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
            current_timestamp = current_candle.name
            swings_before_candle = recent_swings[recent_swings.index < current_timestamp]
            if swings_before_candle.empty:
                continue

            if daily_bias == 'bullish':
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['low'] < swept_level:
                    current_index_pos = len(ohlc_df) + i
                    if self._confirm_displacement_and_mss(ohlc_df, current_index_pos, 'bullish', swings):
                        logger.info(f"Bullish liquidity sweep confirmed with MSS and Displacement at timestamp {current_timestamp}")
                        opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                        mss_level = opposing_swings.iloc[-1]['Level'] if not opposing_swings.empty else 0
                        return {'type': 'bullish_liquidity_sweep_mss', 'level': current_candle['low'], 'swept_level': swept_level, 'index': current_index_pos, 'mss_level': mss_level, 'detected': True}

            elif daily_bias == 'bearish':
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['high'] > swept_level:
                    # Convert negative index to positive for the confirmation function
                    current_index_pos = len(ohlc_df) + i
                    if self._confirm_displacement_and_mss(ohlc_df, current_index_pos, 'bearish', swings):
                        logger.info(f"Bearish liquidity sweep confirmed with MSS and Displacement at timestamp {current_timestamp}")
                        opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                        mss_level = opposing_swings.iloc[-1]['Level'] if not opposing_swings.empty else 0
                        return {'type': 'bearish_liquidity_sweep_mss', 'level': current_candle['high'], 'swept_level': swept_level, 'index': current_index_pos, 'mss_level': mss_level, 'detected': True}
        
        return None
        
    def _get_ict_structure(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, daily_bias: str) -> pd.DataFrame:
        """
        Detects BOS and CHoCH based on swing breaks that are VALIDATED by displacement (an FVG).
        """
        n = len(ohlc_df)
        bos = pd.Series(np.nan, index=ohlc_df.index, dtype=float)
        choch = pd.Series(np.nan, index=ohlc_df.index, dtype=float)
        level = pd.Series(np.nan, index=ohlc_df.index, dtype=float)
        broken_index = pd.Series(np.nan, index=ohlc_df.index, dtype=float)

        relevant_swings = swings.dropna().tail(50)
        all_fvgs = self._get_fvgs(ohlc_df)
        
        # log fvgs for debugging
        if not all_fvgs:
            logger.debug("No FVGs found for structure validation.")

        if relevant_swings.empty:
            return self._create_empty_structure(n)

        for swing_timestamp, swing in relevant_swings.iterrows():
            swing_level, swing_type = swing['Level'], swing['HighLow']
            
            try:
                swing_pos = ohlc_df.index.get_loc(swing_timestamp)
            except KeyError:
                continue

            # Define the search space for the break
            post_swing_data = ohlc_df.iloc[swing_pos + 1:]
            if post_swing_data.empty:
                continue

            # 1. Find the candle that breaks the structure
            if swing_type == 1: # Bullish swing (high) looking for a bearish break
                breaking_candles = post_swing_data[post_swing_data['close'] < swing_level]
            else: # Bearish swing (low) looking for a bullish break
                breaking_candles = post_swing_data[post_swing_data['close'] > swing_level]

            if breaking_candles.empty:
                continue
            
            break_candle_timestamp = breaking_candles.index[0]
            break_candle_pos = ohlc_df.index.get_loc(break_candle_timestamp)

            # 2. VALIDATE with Displacement: Check for an FVG in the structural-breaking leg
            # The "leg" is the series of candles from the swing to the break.
            fvg_confirmed = False
            break_direction = 'bullish' if swing_type == -1 else 'bearish'

            for fvg in all_fvgs:
                # Ensure the FVG type matches the direction of the break
                if fvg.get('type') != break_direction:
                    continue
                
                try:
                    # Get the integer position of the FVG
                    fvg_pos = ohlc_df.index.get_loc(fvg['index'])
                except (KeyError, TypeError):
                    continue
                
                # The FVG must have formed *after* the swing but *at or before* the break
                if swing_pos < fvg_pos <= break_candle_pos:
                    fvg_confirmed = True
                    logger.debug(f"Structure break at {break_candle_timestamp} validated by FVG at {fvg['index']}")
                    break 
            
            # 3. If validated, classify as BOS or CHoCH
            if fvg_confirmed:
                # A break is a BOS if it's in the direction of HTF bias. Otherwise, it's a CHoCH.
                is_bos = (daily_bias == 'bullish' and swing_type == -1) or \
                         (daily_bias == 'bearish' and swing_type == 1)
                
                target_series = bos if is_bos else choch
                signal_value = 1 if break_direction == 'bullish' else -1

                # Use .loc with the original swing_timestamp to assign the value
                target_series.loc[swing_timestamp] = signal_value
                level.loc[swing_timestamp] = swing_level
                broken_index.loc[swing_timestamp] = break_candle_pos

        return pd.DataFrame({
            'BOS': bos, 'CHOCH': choch, 'Level': level, 'BrokenIndex': broken_index.astype('Int64') # Use nullable integer
        })
    
    def _create_empty_structure(self, length):
        """Create empty structure DataFrame."""
        return pd.DataFrame({
            'BOS': np.full(length, np.nan),
            'CHOCH': np.full(length, np.nan),
            'Level': np.full(length, np.nan),
            'BrokenIndex': np.full(length, np.nan)
        })
                
    def _get_order_blocks(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, structure: pd.DataFrame) -> List[Dict]:
        """
        Identifies high-probability ICT Order Blocks that have caused a valid Market Structure Shift.
        """
        if structure.empty or swings.empty:
            return []

        # We only care about breaks that actually happened and have a "BrokenIndex"
        valid_breaks = structure[structure['BrokenIndex'].notna()].dropna(subset=['BOS', 'CHOCH'], how='all')
        
        if valid_breaks.empty:
            return []

        order_blocks = []
        # Keep track of OBs to avoid duplicates from the same move
        processed_indices = set()

        # Iterate backwards to find the most recent OBs first
        for break_timestamp, break_info in valid_breaks.iloc[::-1].iterrows():
            
            # 1. Determine the context from the break
            break_level = break_info['Level']
            break_type_is_bos = pd.notna(break_info['BOS'])
            break_direction = 'bullish' if (break_type_is_bos and break_info['BOS'] == 1) or \
                                           (not break_type_is_bos and break_info['CHOCH'] == 1) else 'bearish'
            
            # Get the integer position of the swing that was broken
            try:
                swing_pos = ohlc_df.index.get_loc(break_timestamp)
            except KeyError:
                continue

            # 2. Find the candle that initiated the move that caused the break
            # We must find the opposing swing high/low that occurred *before* the structural break
            if break_direction == 'bullish':
                # For a bullish break, we need the swing low that started the up-move
                candidate_swings = swings[(swings.index < break_timestamp) & (swings['HighLow'] == -1)]
                if candidate_swings.empty: continue
                origin_swing_timestamp = candidate_swings.index[-1]
            else: # Bearish break
                # For a bearish break, we need the swing high that started the down-move
                candidate_swings = swings[(swings.index < break_timestamp) & (swings['HighLow'] == 1)]
                if candidate_swings.empty: continue
                origin_swing_timestamp = candidate_swings.index[-1]
            
            try:
                origin_swing_pos = ohlc_df.index.get_loc(origin_swing_timestamp)
            except KeyError:
                continue

            # 3. Identify the Order Block Candle itself
            # The OB is the last opposing candle before the displacement move began.
            # We search between the origin swing and the swing that was broken.
            search_df = ohlc_df.iloc[origin_swing_pos:swing_pos]
            
            if break_direction == 'bullish': # Bullish OB is the last DOWN candle
                ob_candle_series = search_df[search_df['close'] < search_df['open']]
                if ob_candle_series.empty: continue
                ob_candle = ob_candle_series.iloc[-1]
            else: # Bearish OB is the last UP candle
                ob_candle_series = search_df[search_df['close'] > search_df['open']]
                if ob_candle_series.empty: continue
                ob_candle = ob_candle_series.iloc[-1]

            ob_index = ob_candle.name
            if ob_index in processed_indices:
                continue

            # 4. Check if the OB has been mitigated
            # Mitigation: Price trades back through the OB after the structural break
            post_break_data = ohlc_df.iloc[int(break_info['BrokenIndex']) + 1:]
            mitigated = False
            if not post_break_data.empty:
                if break_direction == 'bullish' and post_break_data['low'].min() < ob_candle['low']:
                    mitigated = True
                elif break_direction == 'bearish' and post_break_data['high'].max() > ob_candle['high']:
                    mitigated = True
            
            if not mitigated:
                order_blocks.append({
                    'index': ob_index,
                    'type': break_direction,
                    'top': ob_candle['high'],
                    'bottom': ob_candle['low'],
                    'volume': ob_candle.get('volume', 0),
                    'source_break_level': break_level
                })
                processed_indices.add(ob_index)

        return sorted(order_blocks, key=lambda x: x['index'], reverse=True)
    
    def _get_fvgs(self, ohlc_df: pd.DataFrame) -> List[Dict]:
        """
        Identifies unmitigated Fair Value Gaps (FVGs) from OHLC data.
        """
        fvg_list = []
        high = ohlc_df['high']
        low = ohlc_df['low']

        # Vectorized conditions for identifying potential FVGs
        # Bullish FVG: The high of the previous candle is lower than the low of the next candle.
        bullish_fvg_mask = high.shift(1) < low.shift(-1)
        
        # Bearish FVG: The low of the previous candle is higher than the high of the next candle.
        bearish_fvg_mask = low.shift(1) > high.shift(-1)

        # Get the indices where potential FVGs occur
        potential_fvg_indices = ohlc_df.index[bullish_fvg_mask | bearish_fvg_mask]

        for idx in potential_fvg_indices:
            try:
                i = ohlc_df.index.get_loc(idx)
                # Ensure we have room for previous and next candles
                if i == 0 or i == len(ohlc_df) - 1:
                    continue
            except KeyError:
                continue

            fvg_type = 'bullish' if bullish_fvg_mask.loc[idx] else 'bearish'
            
            if fvg_type == 'bullish':
                fvg_top = low.iloc[i+1]
                fvg_bottom = high.iloc[i-1]
            else: # bearish
                fvg_top = low.iloc[i-1]
                fvg_bottom = high.iloc[i+1]

            # Check for mitigation
            mitigated = False
            # Look for mitigation in candles after the FVG formed (i+2 onwards)
            for j in range(i + 2, len(ohlc_df)):
                if fvg_type == 'bullish' and low.iloc[j] <= fvg_top:
                    mitigated = True
                    break
                elif fvg_type == 'bearish' and high.iloc[j] >= fvg_bottom:
                    mitigated = True
                    break
            
            if not mitigated:
                fvg_list.append({
                    'index': idx,
                    'type': fvg_type,
                    'top': fvg_top,
                    'bottom': fvg_bottom
                })
        
        if fvg_list:
            logger.debug(f"Found {len(fvg_list)} unmitigated FVGs using the new detector.")
        else:
            logger.debug("No unmitigated FVGs were found in the provided data.")
            
        return fvg_list
        
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
        """
        Confirms displacement and MSS by leveraging existing FVG detection.
        """
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

        # Get the timestamp of the reaction candle to filter swings correctly
        reaction_timestamp = ohlc_df.index[reaction_index]
        swings_before_reaction = swings[swings.index < reaction_timestamp].dropna()
        if swings_before_reaction.empty:
            logger.debug("MSS/Displacement: Could not find any swings before the reaction point.")
            return False
            
        mss_level = None
        if bias == 'bullish':
            recent_swing_highs = swings_before_reaction[swings_before_reaction['HighLow'] == 1]
            if not recent_swing_highs.empty: mss_level = recent_swing_highs['Level'].iloc[-1]
        else: # bias == 'bearish'
            recent_swing_lows = swings_before_reaction[swings_before_reaction['HighLow'] == -1]
            if not recent_swing_lows.empty: mss_level = recent_swing_lows['Level'].iloc[-1]
        
        if mss_level is None:
            logger.debug("MSS/Displacement: Could not determine a valid MSS level to break.")
            return False

        # --- 1. Confirm Market Structure Shift (MSS) ---
        mss_confirmed = False
        mss_break_index = -1
        for timestamp, candle in confirmation_df.iterrows():
            candle_pos = ohlc_df.index.get_loc(timestamp)
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
            logger.debug(f"MSS/Displacement: FAILED. No candle body closed past the MSS level of {mss_level:.5f} within the confirmation window.")
            return False
            
        # --- 2. Confirm Displacement via Fair Value Gap (FVG) ---
        all_fvgs = self._get_fvgs(ohlc_df)
        if not all_fvgs:
            logger.debug("MSS/Displacement: FAILED. No FVGs found in the dataset to confirm displacement.")
            return False

        fvg_confirmed = False
        for fvg in all_fvgs:
            fvg_type_matches_bias = (fvg.get('type') == bias)
            
            # --- START: CRITICAL BUG FIX ---
            # We must convert the FVG's timestamp to an integer position for a valid comparison.
            try:
                fvg_timestamp = fvg['index']
                fvg_position = ohlc_df.index.get_loc(fvg_timestamp)
            except (KeyError, IndexError):
                continue # Skip if the FVG's timestamp isn't in the main dataframe for some reason
            
            # The FVG must be created within the impulsive leg (between the reaction and the break).
            if fvg_type_matches_bias and (reaction_index < fvg_position <= mss_break_index):
                fvg_confirmed = True
                logger.debug(f"Displacement confirmed: Found a matching {fvg['type']} FVG at index {fvg_position} within the displacement move.")
                break
            # --- END: CRITICAL BUG FIX ---

        if not fvg_confirmed:
            logger.debug(f"MSS/Displacement: FAILED. The move breaking structure did not leave a Fair Value Gap (FVG).")
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
        london_manipulation = analysis.get('manipulation', {'detected': False})
        current_price = analysis['current_price']
        session = analysis['session']
        swings = analysis.get('swings', pd.DataFrame())
        macro_analysis = analysis.get('macro_analysis', {})
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
        
        macro_active = macro_analysis.get('in_macro', False)
        macro_significance = macro_analysis.get('significance', 'medium')
        
        if require_killzone and not session.get('in_killzone') and not (macro_active and macro_significance in ['very_high', 'high']):
            return None, None, None, None

        killzone_name = session.get('killzone_name')
        
        if killzone_name == 'London':
            if london_manipulation.get('detected'):
                manipulation_type = london_manipulation.get('type', 'manipulation')
                logger.info(f"{symbol}: LondonKZ - Looking for entry after '{manipulation_type}'.")
                entry_signal, sl_price, tp_price = self._setup_manipulation_entry(
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
                    entry_signal, sl_price, tp_price = self._setup_manipulation_entry(
                        daily_bias, current_price, new_manipulation, ohlc_df, analysis
                    )
                    narrative.entry_model = f"NY_{new_manipulation.get('type', 'REVERSAL').upper()}"
                    narrative.manipulation_confirmed = True
                    narrative.manipulation_level = new_manipulation.get('level', 0)

        elif killzone_name == 'LondonClose':
            logger.info(f"{symbol}: London Close KZ - Activating ICT-aligned logic...")
            entry_signal, sl_price, tp_price = self._find_continuation_entry(
                analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation
            )
            if entry_signal:
                narrative.entry_model = "LONDON_CLOSE_CONTINUATION"
                logger.info(f"{symbol}: LCKZ - Found valid CONTINUATION setup.")
            if not entry_signal:
                entry_signal, sl_price, tp_price = self._check_london_close_reversal(analysis, ohlc_df, swings)
                if entry_signal:
                    narrative.entry_model = "LONDON_CLOSE_REVERSAL"
                    logger.info(f"{symbol}: LCKZ - Found valid REVERSAL setup.")
        
        elif macro_active and macro_significance in ['very_high', 'high']:
            logger.info(f"{symbol}: High significance macro {macro_analysis.get('macro_name')} - Checking for setup")
            entry_signal, sl_price, tp_price = self._find_continuation_entry(
                analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation
            )
            if entry_signal:
                narrative.entry_model = f"MACRO_{macro_analysis.get('macro_name', 'ACTIVE').upper()}"
            
        if entry_signal:
            if not self._is_safe_entry_location(current_price, sl_price, tp_price, daily_bias) or \
               not self._validate_levels(entry_signal, current_price, sl_price, tp_price):
                return None, None, None, None
            
            logger.info(f"ICT Signal Generated: {entry_signal} {symbol} @ {current_price:.5f}")
            logger.info(f"  Narrative: {daily_bias.upper()} | {narrative.entry_model}")
            logger.info(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            return entry_signal, sl_price, tp_price, narrative        
        
        return None, None, None, None

    def _get_structural_sl_price(self, bias: str, primary_poi: Dict, ohlc_df: pd.DataFrame, analysis: Dict) -> float:
        """
        Calculates a robust, structurally-sound Stop Loss price.
        Anchors the SL not just to the POI, but to the lowest/highest point of the POI
        and any nearby supporting/resisting PD arrays (FVGs, OBs).
        """
        atr_buffer = self._calculate_atr_buffer(ohlc_df)
        
        anchor_level = primary_poi['bottom'] if bias == 'bullish' else primary_poi['top']

        if bias == 'bullish':
            nearby_fvgs = [f for f in analysis['fair_value_gaps'] if f['type'] == 'bullish' and f['bottom'] < anchor_level]
            nearby_obs = [ob for ob in analysis['order_blocks'] if ob['type'] == 'bullish' and ob['bottom'] < anchor_level]
            
            support_levels = [anchor_level]
            if nearby_fvgs:
                support_levels.append(min(f['bottom'] for f in nearby_fvgs))
            if nearby_obs:
                support_levels.append(min(ob['bottom'] for ob in nearby_obs))
                
            final_sl_anchor = min(support_levels)
            sl_price = final_sl_anchor - atr_buffer
            logger.info(f"Structural SL (Bullish): Initial anchor={anchor_level:.5f}, Final anchor={final_sl_anchor:.5f}, SL={sl_price:.5f}")

        else: # bias == 'bearish'
            nearby_fvgs = [f for f in analysis['fair_value_gaps'] if f['type'] == 'bearish' and f['top'] > anchor_level]
            nearby_obs = [ob for ob in analysis['order_blocks'] if ob['type'] == 'bearish' and ob['top'] > anchor_level]

            resistance_levels = [anchor_level]
            if nearby_fvgs:
                resistance_levels.append(max(f['top'] for f in nearby_fvgs))
            if nearby_obs:
                resistance_levels.append(max(ob['top'] for ob in nearby_obs))

            final_sl_anchor = max(resistance_levels)
            sl_price = final_sl_anchor + atr_buffer
            logger.info(f"Structural SL (Bearish): Initial anchor={anchor_level:.5f}, Final anchor={final_sl_anchor:.5f}, SL={sl_price:.5f}")
        
        return sl_price

    def _setup_manipulation_entry(self, bias, current_price, manipulation, ohlc_df, analysis):
        """Setup entry from Judas Swing, Turtle Soup, or other manipulation patterns."""
        primary_poi = {'top': manipulation.get('level'), 'bottom': manipulation.get('level')}
        sl_price = self._get_structural_sl_price(bias, primary_poi, ohlc_df, analysis)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_fvg_entry(self, bias, current_price, fvg_entry, manipulation, ohlc_df, analysis):
        """Setup entry from a Fair Value Gap, now using structural SL."""
        sl_price = self._get_structural_sl_price(bias, fvg_entry, ohlc_df, analysis)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_breaker_entry(self, bias, current_price, breaker_block, ohlc_df, analysis):
        """Setup entry from a Breaker Block, now using structural SL."""
        sl_price = self._get_structural_sl_price(bias, breaker_block, ohlc_df, analysis)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_unicorn_entry(self, bias, current_price, unicorn_setup, ohlc_df, analysis):
        """Setup entry from a Unicorn setup, using the breaker component for structural SL."""
        sl_price = self._get_structural_sl_price(bias, unicorn_setup['breaker_block'], ohlc_df, analysis)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_ote_entry(self, bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis):
        """Setup entry from OTE zone. SL is already structural."""
        ote = next((zone for zone in ote_zones if zone['direction'].lower() == bias), None)
        if not ote:
            return None, None, None
        
        if bias == 'bullish':
            sl_price = min(ote['swing_low'], manipulation.get('level', ote['swing_low'])) - self._calculate_atr_buffer(ohlc_df)
        else:
            sl_price = max(ote['swing_high'], manipulation.get('level', ote['swing_high'])) + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)        
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

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
                fvg_entry = self._find_retracement_fvg(analysis)
                if fvg_entry:
                    return self._setup_fvg_entry('bullish', analysis['current_price'], fvg_entry, {'level': lckz_data['low'].min()}, ohlc_df, analysis)

        elif daily_bias == 'bearish' and sweep_high:
            sweep_candle_idx = lckz_data[lckz_data['high'] > ny_high].index[0]
            sweep_candle_pos = ohlc_df.index.get_loc(sweep_candle_idx)

            if self.analyzer._confirm_displacement_and_mss(ohlc_df, sweep_candle_pos, 'bearish', swings):
                logger.info("LCKZ Reversal: CONFIRMED. Sweep of NY High followed by Bearish Displacement & MSS.")
                fvg_entry = self._find_retracement_fvg(analysis)
                if fvg_entry:
                    return self._setup_fvg_entry('bearish', analysis['current_price'], fvg_entry, {'level': lckz_data['high'].max()}, ohlc_df, analysis)

        return None, None, None
    
    def _check_zone_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Helper function to check if two price zones (like an FVG and OB) overlap."""
        try:
            # True if the top of one zone is below the bottom of the other
            no_overlap = zone1['top'] < zone2['bottom'] or zone2['top'] < zone1['bottom']
            return not no_overlap
        except (KeyError, TypeError):
            return False
    
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
        """Finds continuation entries with a priority: Unicorn > Breaker > FVG > OTE."""
        swings = analysis.get('swings', pd.DataFrame())
        data_len = len(ohlc_df)

        # --- UNICORN ENTRY ---
        unicorn_setups = analysis.get('unicorn_setups', [])
        for unicorn in unicorn_setups:
            if unicorn['breaker_block']['original_ob_type'] != daily_bias:
                poi_range = {'top': unicorn['top'], 'bottom': unicorn['bottom']}
                reaction_index = self._find_poi_reaction_index(ohlc_df, poi_range)
                if reaction_index is not None and reaction_index < data_len - 1:
                    if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                        logger.info(f"{analysis['symbol']}: UNICORN entry confirmed with Displacement & MSS.")
                        return self._setup_unicorn_entry(daily_bias, current_price, unicorn, ohlc_df, analysis)

        # --- BREAKER BLOCK ENTRY ---
        breaker_blocks = analysis.get('breaker_blocks', [])
        for breaker in breaker_blocks:
            if (daily_bias == 'bullish' and breaker['original_ob_type'] == 'bearish') or \
                (daily_bias == 'bearish' and breaker['original_ob_type'] == 'bullish'):
                poi_range = {'top': breaker['top'], 'bottom': breaker['bottom']}
                reaction_index = self._find_poi_reaction_index(ohlc_df, poi_range)
                if reaction_index is not None and reaction_index < data_len - 1:
                    if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                        logger.info(f"{analysis['symbol']}: BREAKER BLOCK entry confirmed with Displacement & MSS.")
                        return self._setup_breaker_entry(daily_bias, current_price, breaker, ohlc_df, analysis)

        # --- FVG ENTRY ---
        fvg_entry = self._find_retracement_fvg(analysis)
        if fvg_entry:
            reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': fvg_entry['top'], 'bottom': fvg_entry['bottom']})
            if reaction_index is not None and reaction_index < data_len - 1:
                if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                    logger.info(f"{analysis['symbol']}: FVG continuation entry confirmed with Displacement & MSS.")
                    return self._setup_fvg_entry(daily_bias, current_price, fvg_entry, manipulation, ohlc_df, analysis)

        # --- OTE ENTRY ---
        ote_zones = analysis['ote_zones']
        ote_zone = next((z for z in ote_zones if z['direction'].lower() == daily_bias), None)
        if ote_zone and self._is_price_in_ote(current_price, [ote_zone], daily_bias):
            reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': ote_zone['high'], 'bottom': ote_zone['low']})
            if reaction_index is not None and reaction_index < data_len - 1:
                if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                    logger.info(f"{analysis['symbol']}: OTE continuation entry confirmed with Displacement & MSS.")
                    return self._setup_ote_entry(daily_bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis)

        return None, None, None
        
    def _find_retracement_fvg(self, analysis: Dict) -> Optional[Dict]:
        """
        Finds the highest-probability FVG for a continuation entry by calculating a confluence score.
        --- ADVANCED IMPLEMENTATION ---
        """
        # Extract all necessary data from the main analysis dictionary
        fair_value_gaps = analysis.get('fair_value_gaps', [])
        daily_bias = analysis.get('daily_bias')
        current_price = analysis.get('current_price')
        manipulation = analysis.get('manipulation', {})
        ohlc_df = analysis.get('ohlc_df')
        order_blocks = analysis.get('order_blocks', [])
        structure = analysis.get('structure', pd.DataFrame())
        premium_discount = analysis.get('premium_discount', {})

        if not fair_value_gaps or ohlc_df.empty or not premium_discount:
            return None
        
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14).iloc[-1]
        manipulation_index = manipulation.get('index', -1)
        equilibrium = premium_discount.get('equilibrium')
        
        logger.debug(f"FVG Search: Bias={daily_bias}, Current Price={current_price:.5f}, "
                    f"Equilibrium={equilibrium:.5f}, Current Zone={premium_discount.get('current_zone', 'unknown')}")
        
        valid_fvgs = []
        
        for fvg in fair_value_gaps:
            # --- Basic Filtering ---
            if (daily_bias == 'bullish' and fvg.get('type') != 'bullish') or \
               (daily_bias == 'bearish' and fvg.get('type') != 'bearish'):
                continue
            
            fvg_top, fvg_bottom = fvg['top'], fvg['bottom']
            
            # Filter out FVGs that are in the wrong P/D array
            if daily_bias == 'bullish' and fvg_bottom > equilibrium:
                continue
            elif daily_bias == 'bearish' and fvg_top < equilibrium:
                continue
            
            # Filter out FVGs that price has already traded completely through
            if (daily_bias == 'bullish' and current_price < fvg_bottom) or \
               (daily_bias == 'bearish' and current_price > fvg_top):
                 continue
            
            # --- Confluence Scoring ---
            priority_score = 0
            score_reasons = []

            # 1. Post-Manipulation Factor (+2)
            if manipulation_index != -1:
                try:
                    fvg_position = ohlc_df.index.get_loc(fvg['index'])
                    if fvg_position > manipulation_index:
                        priority_score += 2
                        score_reasons.append("Post-Manipulation")
                except (KeyError, TypeError):
                    pass

            # 2. Optimal Trade Entry (OTE) Factor (+2)
            pd_range = premium_discount.get('swing_high', 0) - premium_discount.get('swing_low', 0)
            if pd_range > 0:
                ote_low = premium_discount['swing_low'] + (pd_range * 0.618)
                ote_high = premium_discount['swing_high'] - (pd_range * 0.618)
                if (daily_bias == 'bullish' and fvg_bottom >= ote_low) or \
                   (daily_bias == 'bearish' and fvg_top <= ote_high):
                    priority_score += 2
                    score_reasons.append("In OTE Zone")

            # 3. Confluence with Order Block Factor (+4)
            for ob in order_blocks:
                if ob.get('type') == daily_bias and self._check_zone_overlap(fvg, ob):
                    priority_score += 4
                    score_reasons.append("OB Confluence")
                    break # Only score once for OB confluence

            # 4. Caused Break of Structure Factor (+3)
            if not structure.empty:
                valid_breaks = structure[structure['BrokenIndex'].notna()].dropna(subset=['BOS'], how='all')
                for _, brk in valid_breaks.iterrows():
                    if (daily_bias == 'bullish' and brk['BOS'] == 1) or (daily_bias == 'bearish' and brk['BOS'] == -1):
                        try:
                             fvg_pos = ohlc_df.index.get_loc(fvg['index'])
                             break_pos = int(brk['BrokenIndex'])
                             if fvg_pos <= break_pos: # Simplified check: FVG must exist at or before the break
                                 priority_score += 3
                                 score_reasons.append("Caused BOS")
                                 break
                        except (KeyError, TypeError):
                             continue
            
            fvg['priority_score'] = priority_score
            fvg['score_reasons'] = " & ".join(score_reasons) if score_reasons else "N/A"
            valid_fvgs.append(fvg)
        
        if valid_fvgs:
            # Sort by score (descending), then by proximity to current price (ascending)
            valid_fvgs.sort(key=lambda x: (-x['priority_score'], abs(current_price - x['top'])))
            
            best_fvg = valid_fvgs[0]
            logger.info(f"Selected FVG: {best_fvg['type']}, Score: {best_fvg['priority_score']:.2f}, Reasons: {best_fvg['score_reasons']}")
            
            result = best_fvg.copy()
            result['consequent_encroachment'] = (result['top'] + result['bottom']) / 2
            return result
        
        logger.debug("No valid FVGs found matching ICT criteria")
        return None

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
        
    def _is_safe_entry_location(self, entry_price: float, sl_price: float, tp_price: float, daily_bias: str) -> bool:
        """
        Validates the proposed trade's Risk:Reward ratio against the configured minimum.
        """
        # Ensure prices are valid to prevent division by zero or nonsensical calculations
        if tp_price is None or sl_price is None or entry_price is None:
            logger.warning("Cannot validate trade safety: Missing entry, SL, or TP price.")
            return False

        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)

        if risk == 0:
            logger.warning("Risk is zero, trade is invalid.")
            return False

        min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)

        # The core validation: Does the actual reward meet the minimum required R:R?
        if reward < (risk * min_rr):
            actual_rr = round(reward / risk, 2) if risk > 0 else float('inf')
            logger.warning(
                f"Trade Rejected: The proposed trade does not meet the minimum R:R of {min_rr}:1. "
                f"Actual R:R is {actual_rr}:1 (Risk: {risk:.5f}, Reward: {reward:.5f})."
            )
            return False
        
        logger.info(f"Trade geometry validated. R:R is sufficient.")
        return True
        
    def _is_price_in_ote(self, current_price, ote_zones, daily_bias):
        """Check if price is within OTE zone."""
        for ote in ote_zones:
            if ote['direction'].lower() == daily_bias:
                if daily_bias == 'bullish':
                    if ote['low'] <= current_price <= ote['high']:
                        return True
                else: # bearish
                    if ote['low'] <= current_price <= ote['high']:
                        return True
        return False
    
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
        if tp is None or sl is None or entry is None:
            return False
        if signal == "BUY":
            return sl < entry < tp
        else:
            return tp < entry < sl