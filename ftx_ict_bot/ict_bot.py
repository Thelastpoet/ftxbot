"""
ICT Trading Engine 
"""

import pytz
import pandas as pd
import numpy as np
import logging
from logger import operations_logger as logger
from dataclasses import dataclass
from datetime import timedelta, datetime
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
            logger.debug(f"{symbol}: Insufficient data for ICT analysis")
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
        ote_zones = self._calculate_ote_zones(ohlc_df, daily_bias, premium_discount_analysis)
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
            logger.debug("Insufficient H4 data for HTF levels")
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
        
        if ohlc_df is None or ohlc_df.index.tz is None:
            context['error'] = "DataFrame is None or index is not timezone-aware."
            logger.error(context['error'])
            return context

        try:
            latest_utc_time = ohlc_df.index[-1]
            latest_ny_time = latest_utc_time.astimezone(self.config.NY_TIMEZONE)
            
            # Try to find the Asian range for the last 3 days, starting with the most recent valid day.
            # This handles Monday mornings by automatically looking back to Friday.
            for i in range(4):
                current_ny_date_to_check = (latest_ny_time - timedelta(days=i)).date()
                
                # Skip weekends
                if current_ny_date_to_check.weekday() in [5, 6]: # Saturday or Sunday
                    continue

                # Determine the start of the Asian session for the given date.
                # The Asian session for a trading day (e.g., Monday) starts on the evening of the previous calendar day (e.g., Sunday).
                # So, for Monday's session, we look at Sunday evening.
                asian_session_date = current_ny_date_to_check - timedelta(days=1) if current_ny_date_to_check.weekday() != 0 else current_ny_date_to_check

                # On a Monday, the Asian session starts on Sunday evening.
                if latest_ny_time.weekday() == 0 and i == 0: # It's Monday
                     asian_session_date = latest_ny_time.date() - timedelta(days=1)
                else: # For other days, it's the previous day
                     asian_session_date = (latest_ny_time - timedelta(days=i)).date() - timedelta(days=1)


                # Define the start and end times for the Asian range in NY time
                asian_start_ny = datetime.combine(
                    asian_session_date, 
                    self.config.ICT_ASIAN_RANGE['start'], 
                    self.config.NY_TIMEZONE
                )
                asian_end_ny = (asian_start_ny + timedelta(hours=7)).replace(hour=2, minute=0) # 7 PM NY to 2 AM NY

                # Convert to UTC to filter the dataframe
                asian_start_utc = asian_start_ny.astimezone(pytz.UTC)
                asian_end_utc = asian_end_ny.astimezone(pytz.UTC)
                
                asian_data = ohlc_df[(ohlc_df.index >= asian_start_utc) & (ohlc_df.index < asian_end_utc)]
                
                if not asian_data.empty:
                    context['last_asian_range'] = {
                        'start_time_utc': asian_data.index[0],
                        'end_time_utc': asian_end_utc,
                        'high': asian_data['high'].max(),
                        'low': asian_data['low'].min(),
                        'start_idx': ohlc_df.index.get_loc(asian_data.index[0]),
                        'date_found': current_ny_date_to_check.strftime('%Y-%m-%d')
                    }
                    logger.info(f"Found valid Asian Range for trading day {current_ny_date_to_check.strftime('%Y-%m-%d')} "
                                f"(High={context['last_asian_range']['high']:.5f}, Low={context['last_asian_range']['low']:.5f})")
                    break # Exit loop once the most recent range is found
            
            if context['last_asian_range'] is None:
                 logger.warning(f"No valid Asian range found within the last 3 trading days for {latest_ny_time.date()}")

            # Check Kill Zones using NY time
            current_ny_hour = latest_ny_time.hour
            current_ny_minute = latest_ny_time.minute
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
    
    def _extract_bias_from_manipulation(self, manipulation: Dict) -> str:
        """
        Extract bias indication from manipulation context per ICT methodology.
        Judas swings and liquidity sweeps indicate where smart money is NOT going.
        """
        if not manipulation.get('detected'):
            return 'neutral'
        
        manipulation_type = manipulation.get('type', '')
        
        # ICT Logic: Manipulation is typically opposite to intended direction
        if 'bullish_judas' in manipulation_type or 'bullish_liquidity_sweep' in manipulation_type:
            # Bullish manipulation usually precedes bearish move
            return 'bearish'
        elif 'bearish_judas' in manipulation_type or 'bearish_liquidity_sweep' in manipulation_type:
            # Bearish manipulation usually precedes bullish move  
            return 'bullish'
        elif 'turtle_soup' in manipulation_type:
            # Turtle soup indicates reversal from the swept direction
            if 'bullish' in manipulation_type:
                return 'bullish'
            elif 'bearish' in manipulation_type:
                return 'bearish'
        
        return 'neutral'
    
    def _enhance_bias_with_manipulation_context(self, htf_bias: str, manipulation: Dict) -> Tuple[str, str]:
        """
        Enhance bias determination with manipulation context per ICT teachings.
        Returns (final_bias, bias_strength)
        """
        if not manipulation.get('detected'):
            return htf_bias, 'standard'
        
        manipulation_bias = self._extract_bias_from_manipulation(manipulation)
        
        if manipulation_bias == 'neutral':
            return htf_bias, 'standard'
        
        # ICT Confluence: When manipulation aligns with HTF order flow
        if manipulation_bias == htf_bias:
            logger.info(f"STRONG bias confluence: HTF={htf_bias}, Manipulation supports {manipulation_bias}")
            return htf_bias, 'strong'
        elif manipulation_bias != htf_bias and htf_bias != 'neutral':
            # Conflicting signals - reduce confidence but keep HTF as primary
            logger.warning(f"WEAK bias: HTF={htf_bias} conflicts with Manipulation={manipulation_bias}")
            return htf_bias, 'weak'
        
        return htf_bias, 'standard'
    
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
        
        # Enhanced: Integrate manipulation context into bias strength
        if manipulation_details:
            enhanced_bias, bias_strength = self._enhance_bias_with_manipulation_context(daily_bias, manipulation_details)
            bias_details['manipulation_confluence'] = bias_strength
            bias_details['enhanced_reasoning'] = f"Manipulation context: {bias_strength} {enhanced_bias} bias"
            logger.info(f"{symbol}: Enhanced bias: {enhanced_bias} ({bias_strength}) with {manipulation_details.get('type', 'unknown')} manipulation")

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
            logger.debug("Insufficient daily data for order flow analysis")
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
        
    def _analyze_market_phase(self, ohlc_df: pd.DataFrame, daily_bias: str, premium_discount: Dict, manipulation: Dict) -> Dict:
        """
        Analyzes the current market phase (impulse, retracement, invalidated) based on the daily bias
        and the current price's position within the premium/discount range.
        """
        current_price = ohlc_df['close'].iloc[-1]
        
        # 1. Check for invalidation first - this is the most critical check
        if manipulation.get('detected'):
            manipulation_level = manipulation.get('level')
            if daily_bias == 'bullish' and current_price < manipulation_level:
                return {'phase': 'invalidated', 'reason': 'Price traded below bullish manipulation level.'}
            if daily_bias == 'bearish' and current_price > manipulation_level:
                return {'phase': 'invalidated', 'reason': 'Price traded above bearish manipulation level.'}

        # 2. Determine phase (impulse vs retracement) using the primary P/D range
        phase = 'consolidation'
        retracement_percent = 0

        if premium_discount:
            retracement_percent = premium_discount.get('retracement_percent', 0)
            current_zone = premium_discount.get('current_zone')
            range_direction = premium_discount.get('direction') # 1 for bullish, -1 for bearish

            if daily_bias == 'bullish':
                # In a bullish trend, a move into discount is a retracement.
                # A move in premium is an impulse (or expansion).
                if range_direction == 1: # Bullish range
                    phase = 'retracement' if current_zone == 'discount' else 'impulse'
                else: # We are in a bearish range, but have bullish bias -> likely consolidation/complex pullback
                    phase = 'consolidation'

            elif daily_bias == 'bearish':
                # In a bearish trend, a move into premium is a retracement.
                # A move in discount is an impulse (or expansion).
                if range_direction == -1: # Bearish range
                    phase = 'retracement' if current_zone == 'premium' else 'impulse'
                else: # We are in a bullish range, but have bearish bias -> likely consolidation/complex pullback
                    phase = 'consolidation'
        
        return {
            'phase': phase,
            'retracement_percent': round(retracement_percent, 2)
        }
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context, swings):
        """Checks for manipulation patterns."""
        if daily_bias == 'neutral' or swings.empty:
            return None

        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        manipulation_hours = [2, 3, 4, 5, 8, 9, 10, 11, 12]
        if current_ny_hour in manipulation_hours:
            # Check if Asian range data is available
            asian_range = session_context.get('last_asian_range')
            if not asian_range:
                logger.debug("No Asian range data available for manipulation pattern detection")
                return None
                
            asian_high = asian_range['high']
            asian_low = asian_range['low']
            
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
        Detects BOS and CHoCH based on swing breaks.
        """
        if swings.empty:
            return self._create_empty_structure(len(ohlc_df))

        structure = smc.bos_choch(ohlc_df, swings, close_break=True)
        
        # Ensure the structure dataframe has the same index as the ohlc_df
        structure.index = ohlc_df.index
        
        return structure
    
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
        
    def _calculate_ote_zones(self, ohlc_df, daily_bias, premium_discount):
        """
        Calculate OTE zones based on the current premium/discount dealing range.
        """
        if not premium_discount or 'swing_high' not in premium_discount:
            return []

        swing_high = premium_discount['swing_high']
        swing_low = premium_discount['swing_low']
        
        if swing_high <= swing_low:
            return []

        range_size = swing_high - swing_low
        ote_zones = []

        # The OTE levels are the same regardless of bias, but the direction helps in interpretation.
        # ICT OTE Fib levels: 0.62, 0.705 (sweet spot), 0.79
        
        if daily_bias == 'bullish':
            # For bullish bias, we are looking to BUY in the DISCOUNT area (below 50%)
            ote_level_high = swing_high - (range_size * 0.62) # 62% retracement
            ote_level_sweet = swing_high - (range_size * 0.705) # 70.5% retracement
            ote_level_low = swing_high - (range_size * 0.79) # 79% retracement
            
            ote_zones.append({
                'direction': 'bullish',
                'high': ote_level_high,
                'sweet': ote_level_sweet,
                'low': ote_level_low,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'equilibrium': premium_discount.get('equilibrium')
            })
            logger.debug(f"Bullish OTE Zone (Discount): Low={ote_level_low:.5f}, High={ote_level_high:.5f}")

        elif daily_bias == 'bearish':
            # For bearish bias, we are looking to SELL in the PREMIUM area (above 50%)
            ote_level_low = swing_low + (range_size * 0.62) # 62% retracement
            ote_level_sweet = swing_low + (range_size * 0.705) # 70.5% retracement
            ote_level_high = swing_low + (range_size * 0.79) # 79% retracement
            
            ote_zones.append({
                'direction': 'bearish',
                'high': ote_level_high,
                'sweet': ote_level_sweet,
                'low': ote_level_low,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'equilibrium': premium_discount.get('equilibrium')
            })
            logger.debug(f"Bearish OTE Zone (Premium): Low={ote_level_low:.5f}, High={ote_level_high:.5f}")
            
        return ote_zones

    
        
    def _confirm_displacement_and_mss(self, ohlc_df: pd.DataFrame, reaction_index: int, bias: str, swings: pd.DataFrame) -> bool:
        """
        Confirms displacement and MSS using RECENT candles only.
        Looks for immediate market structure shift and displacement (FVG)
        """
        if not self.config.REQUIRE_ENTRY_CONFIRMATION:
            return True

        # 2-3 candles of confirmation
        confirmation_window_size = 3
        start_idx = reaction_index + 1
        end_idx = min(start_idx + confirmation_window_size, len(ohlc_df))
        
        if end_idx <= start_idx:
            logger.debug("MSS/Displacement: No candles available for confirmation")
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
            
        # Determine MSS level based on bias
        mss_level = None
        if bias == 'bullish':
            recent_swing_highs = swings_before_reaction[swings_before_reaction['HighLow'] == 1]
            if not recent_swing_highs.empty: 
                mss_level = recent_swing_highs['Level'].iloc[-1]
        else:  # bias == 'bearish'
            recent_swing_lows = swings_before_reaction[swings_before_reaction['HighLow'] == -1]
            if not recent_swing_lows.empty: 
                mss_level = recent_swing_lows['Level'].iloc[-1]
        
        if mss_level is None:
            logger.debug("MSS/Displacement: Could not determine a valid MSS level to break.")
            return False

        # Check for immediate momentum and structure shift ---
        
        # 1. Check for immediate displacement/momentum (not full MSS yet)
        displacement_confirmed = False
        mss_confirmed = False
        
        # For bullish: Look for strong bullish momentum
        if bias == 'bullish':
            for i, (timestamp, candle) in enumerate(confirmation_df.iterrows()):
                # Check for strong bullish candle (displacement)
                candle_body = candle['close'] - candle['open']
                candle_range = candle['high'] - candle['low']
                
                if candle_body > 0 and candle_body > (candle_range * 0.6):
                    displacement_confirmed = True
                    logger.debug(f"Bullish displacement detected at candle {i+1} after reaction")
                    
                # Check if this candle breaks the MSS level
                if candle['close'] > mss_level:
                    mss_confirmed = True
                    logger.debug(f"MSS Confirmed: Bullish close at {timestamp} above level {mss_level:.5f}")
                    break
                    
        else:  # bearish
            for i, (timestamp, candle) in enumerate(confirmation_df.iterrows()):
                # Check for strong bearish candle (displacement)
                candle_body = candle['open'] - candle['close']
                candle_range = candle['high'] - candle['low']
                
                if candle_body > 0 and candle_body > (candle_range * 0.6):
                    displacement_confirmed = True
                    logger.debug(f"Bearish displacement detected at candle {i+1} after reaction")
                    
                # Check if this candle breaks the MSS level
                if candle['close'] < mss_level:
                    mss_confirmed = True
                    logger.debug(f"MSS Confirmed: Bearish close at {timestamp} below level {mss_level:.5f}")
                    break
        
        # 2. For LIVE trading, we accept either:
        # - Immediate displacement (strong momentum candle) OR
        # - Actual MSS break OR
        # - FVG creation in the right direction
        
        # Check for FVG creation as additional confirmation
        fvg_confirmed = False
        if displacement_confirmed or mss_confirmed:
            # Quick FVG check in the confirmation window
            for i in range(len(confirmation_df) - 2):
                if bias == 'bullish':
                    # Bullish FVG: Gap between candle[i] high and candle[i+2] low
                    if confirmation_df.iloc[i]['high'] < confirmation_df.iloc[i+2]['low']:
                        fvg_confirmed = True
                        logger.debug(f"Bullish FVG detected in confirmation window")
                        break
                else:  # bearish
                    # Bearish FVG: Gap between candle[i] low and candle[i+2] high
                    if confirmation_df.iloc[i]['low'] > confirmation_df.iloc[i+2]['high']:
                        fvg_confirmed = True
                        logger.debug(f"Bearish FVG detected in confirmation window")
                        break
        
        # LIVE TRADING DECISION: More flexible criteria
        if mss_confirmed and (displacement_confirmed or fvg_confirmed):
            logger.info(f"LIVE CONFIRMATION SUCCESS: MSS + Displacement/FVG confirmed")
            return True
        elif displacement_confirmed and fvg_confirmed:
            logger.info(f"LIVE CONFIRMATION SUCCESS: Strong displacement + FVG (MSS pending)")
            return True
        elif mss_confirmed:
            logger.info(f"LIVE CONFIRMATION SUCCESS: MSS break confirmed (momentum building)")
            return True
        else:
            reasons = []
            if not displacement_confirmed:
                reasons.append("No displacement")
            if not mss_confirmed:
                reasons.append(f"No MSS break of {mss_level:.5f}")
            if not fvg_confirmed:
                reasons.append("No FVG")
            logger.debug(f"MSS/Displacement: FAILED. {', '.join(reasons)}")
            return False

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
                       
        narrative = self._build_narrative(analysis)
        
        if narrative.po3_phase == 'manipulation_pending':
            logger.info(f"{symbol}: Holding off on entry, manipulation is pending.")
            return None, None, None, None
        
        if daily_bias == 'neutral':
            logger.debug(f"{symbol}: No clear daily bias from daily chart.")
            return None, None, None, None
        
        market_phase = self.analyzer._analyze_market_phase(ohlc_df, daily_bias, analysis['premium_discount'], london_manipulation)
        logger.info(f"{symbol}: Phase: {market_phase['phase']} | Retracement: {market_phase.get('retracement_percent', 0):.1f}%")
        
        if market_phase['phase'] == 'invalidated':
            return None, None, None, None
        
        # --- CONTEXT-AWARE ENTRY LOGIC ---
        entry_signal, sl_price, tp_price = None, None, None
        
        # Determine the market context: Is it a reversal or continuation?
        # A reversal is typically signified by a recent manipulation followed by a market structure shift.
        manipulation_detected = london_manipulation.get('detected', False)
        structure_changed = narrative.structure_broken

        # REVERSAL SCENARIO
        if manipulation_detected and structure_changed:
            logger.info(f"{symbol}: REVERSAL context detected. Searching for Unicorn, Breaker, or Mitigation entries.")
            entry_signal, sl_price, tp_price = self._find_reversal_entry(analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation)
            if entry_signal:
                narrative.entry_model = f"REVERSAL_{narrative.entry_model}"

        # CONTINUATION SCENARIO
        else:
            logger.info(f"{symbol}: CONTINUATION context detected. Searching for FVG or Order Block entries.")
            entry_signal, sl_price, tp_price = self._find_continuation_entry(analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation)
            if entry_signal:
                narrative.entry_model = f"CONTINUATION_{narrative.entry_model}"
            
        if entry_signal:
            if not self._is_safe_entry_location(current_price, sl_price, tp_price, daily_bias) or \
               not self._validate_levels(entry_signal, current_price, sl_price, tp_price):
                return None, None, None, None
            
            logger.info(f"ICT Signal Generated: {entry_signal} {symbol} @ {current_price:.5f}")
            logger.info(f"  Narrative: {daily_bias.upper()} | {narrative.entry_model}")
            logger.info(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            return entry_signal, sl_price, tp_price, narrative        
        
        return None, None, None, None

    def _get_structural_sl_price(self, bias: str, primary_poi: Dict, ohlc_df: pd.DataFrame, analysis: Dict, spread: float, manipulation_level: Optional[float] = None) -> float:
        """
        Calculates a robust, structurally-sound Stop Loss price.
        PRIORITY 1: The manipulation level, as it's the ultimate invalidation point.
        PRIORITY 2: The lowest/highest point of the POI combined with other nearby PD arrays.
        """
        atr_buffer = self._calculate_atr_buffer(ohlc_df)

        # --- PRIORITY 1: Invalidation based on Manipulation Level ---
        if manipulation_level:
            if bias == 'bullish':
                sl_price = manipulation_level - atr_buffer
                logger.info(f"Structural SL (Bullish) anchored to MANIPULATION LEVEL at {manipulation_level:.5f}. SL: {sl_price:.5f}")
                return sl_price
            else: # bias == 'bearish'
                sl_price = manipulation_level + atr_buffer
                logger.info(f"Structural SL (Bearish) anchored to MANIPULATION LEVEL at {manipulation_level:.5f}. SL: {sl_price:.5f}")
                return sl_price

        # --- PRIORITY 2: Invalidation based on POI and surrounding structure (if no manipulation) ---
        logger.debug("No manipulation level found, using POI and structural analysis for SL.")
        anchor_level = primary_poi['bottom'] if bias == 'bullish' else primary_poi['top']
        
        if bias == 'bullish':
            support_levels = [anchor_level]
            nearby_fvgs = [f for f in analysis['fair_value_gaps'] if f['type'] == 'bullish' and f['bottom'] < anchor_level]
            nearby_obs = [ob for ob in analysis['order_blocks'] if ob['type'] == 'bullish' and ob['bottom'] < anchor_level]
            
            if nearby_fvgs:
                support_levels.append(min(f['bottom'] for f in nearby_fvgs))
            if nearby_obs:
                support_levels.append(min(ob['bottom'] for ob in nearby_obs))
                
            final_sl_anchor = min(support_levels)
            sl_price = final_sl_anchor - atr_buffer
            logger.info(f"Structural SL (Bullish): Initial anchor={anchor_level:.5f}, Final anchor={final_sl_anchor:.5f}, SL={sl_price:.5f}")

        else: # bias == 'bearish'
            resistance_levels = [anchor_level]
            nearby_fvgs = [f for f in analysis['fair_value_gaps'] if f['type'] == 'bearish' and f['top'] > anchor_level]
            nearby_obs = [ob for ob in analysis['order_blocks'] if ob['type'] == 'bearish' and ob['top'] > anchor_level]

            if nearby_fvgs:
                resistance_levels.append(max(f['top'] for f in nearby_fvgs))
            if nearby_obs:
                resistance_levels.append(max(ob['top'] for ob in nearby_obs))

            final_sl_anchor = max(resistance_levels)
            sl_price = final_sl_anchor + atr_buffer
            logger.info(f"Structural SL (Bearish): Initial anchor={anchor_level:.5f}, Final anchor={final_sl_anchor:.5f}, SL={sl_price:.5f}")
        
        return sl_price    

    def _setup_fvg_entry(self, bias, current_price, fvg_entry, manipulation, ohlc_df, analysis, spread):
        """Setup entry from a Fair Value Gap, now using structural SL."""
        manipulation_level = manipulation.get('level') if manipulation else None
        sl_price = self._get_structural_sl_price(bias, fvg_entry, ohlc_df, analysis, manipulation_level=manipulation_level, spread=spread)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_breaker_entry(self, bias, current_price, breaker_block, ohlc_df, analysis, manipulation, spread):
        """Setup entry from a Breaker Block, now using structural SL."""
        manipulation_level = manipulation.get('level') if manipulation else None
        sl_price = self._get_structural_sl_price(bias, breaker_block, ohlc_df, analysis, manipulation_level=manipulation_level, spread=spread)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_unicorn_entry(self, bias, current_price, unicorn_setup, ohlc_df, analysis, manipulation, spread):
        """Setup entry from a Unicorn setup, using the breaker component for structural SL."""
        manipulation_level = manipulation.get('level') if manipulation else None
        sl_price = self._get_structural_sl_price(bias, unicorn_setup['breaker_block'], ohlc_df, analysis, manipulation_level=manipulation_level, spread=spread  )
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _setup_ote_entry(self, bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis):
        """Setup entry from OTE zone, now using the standard structural SL function."""
        ote = next((zone for zone in ote_zones if zone['direction'].lower() == bias), None)
        if not ote:
            return None, None, None

        # For OTE, the POI for SL purposes is the swing that defines the range.
        ote_poi = {'top': ote['swing_high'], 'bottom': ote['swing_low']}
        manipulation_level = manipulation.get('level') if manipulation else None

        sl_price = self._get_structural_sl_price(bias, ote_poi, ohlc_df, analysis, spread, manipulation_level=manipulation_level)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)        
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price    
    
    def _check_zone_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Helper function to check if two price zones (like an FVG and OB) overlap."""
        try:
            # True if the top of one zone is below the bottom of the other
            no_overlap = zone1['top'] < zone2['bottom'] or zone2['top'] < zone1['bottom']
            return not no_overlap
        except (KeyError, TypeError):
            return False

    def _setup_mitigation_entry(self, bias, current_price, mitigation_block, ohlc_df, analysis, manipulation, spread):
        """Setup entry from a Mitigation Block, using structural SL."""
        manipulation_level = manipulation.get('level') if manipulation else None
        sl_price = self._get_structural_sl_price(bias, mitigation_block, ohlc_df, analysis, manipulation_level=manipulation_level, spread=spread)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        return signal, sl_price, tp_price

    def _find_reversal_entry(self, analysis, daily_bias, current_price, ohlc_df, spread, manipulation):
        """Finds reversal entries with a priority: Unicorn > Breaker > Mitigation > FVG."""
        current_high = ohlc_df['high'].iloc[-1]
        current_low = ohlc_df['low'].iloc[-1]
        last_3_candles = ohlc_df.tail(3)
        
        logger.debug(f"{analysis['symbol']}: Checking REVERSAL entries. Current: {current_price:.5f}")

        # --- UNICORN ENTRY (HIGHEST PRIORITY) ---
        unicorn_setups = analysis.get('unicorn_setups', [])
        for unicorn in unicorn_setups:
            if unicorn.get('entry_direction') == daily_bias:
                if self._is_price_at_poi_now(current_price, current_high, current_low, unicorn):
                    logger.info(f"{analysis['symbol']}: Price at UNICORN zone NOW [{unicorn['bottom']:.5f}-{unicorn['top']:.5f}]")
                    if self._detect_rejection_pattern(last_3_candles, daily_bias, unicorn):
                        logger.info(f"{analysis['symbol']}: UNICORN entry confirmed with rejection pattern")
                        return self._setup_unicorn_entry(daily_bias, current_price, unicorn, ohlc_df, analysis, manipulation, spread)

        # --- BREAKER BLOCK ENTRY ---
        breaker_blocks = analysis.get('breaker_blocks', [])
        for breaker in breaker_blocks:
            if (daily_bias == 'bullish' and breaker['original_ob_type'] == 'bearish') or \
               (daily_bias == 'bearish' and breaker['original_ob_type'] == 'bullish'):
                if self._is_price_at_poi_now(current_price, current_high, current_low, breaker):
                    logger.info(f"{analysis['symbol']}: Price at BREAKER BLOCK NOW [{breaker['bottom']:.5f}-{breaker['top']:.5f}]")
                    if self._detect_rejection_pattern(last_3_candles, daily_bias, breaker):
                        logger.info(f"{analysis['symbol']}: BREAKER BLOCK entry confirmed with rejection")
                        return self._setup_breaker_entry(daily_bias, current_price, breaker, ohlc_df, analysis, manipulation, spread)

        # --- MITIGATION BLOCK ENTRY ---
        mitigation_blocks = analysis.get('mitigation_blocks', [])
        for mitigation in mitigation_blocks:
            if self._is_price_at_poi_now(current_price, current_high, current_low, mitigation):
                logger.info(f"{analysis['symbol']}: Price at MITIGATION BLOCK NOW [{mitigation['bottom']:.5f}-{mitigation['top']:.5f}]")
                if self._detect_rejection_pattern(last_3_candles, daily_bias, mitigation):
                    logger.info(f"{analysis['symbol']}: MITIGATION BLOCK entry confirmed with rejection")
                    return self._setup_mitigation_entry(daily_bias, current_price, mitigation, ohlc_df, analysis, manipulation, spread)

        # --- FVG ENTRY (FALLBACK) ---
        fair_value_gaps = analysis.get('fair_value_gaps', [])
        for fvg in fair_value_gaps:
            if fvg['type'] == daily_bias:
                if self._is_price_at_poi_now(current_price, current_high, current_low, fvg):
                    logger.info(f"{analysis['symbol']}: Price at {fvg['type']} FVG NOW [{fvg['bottom']:.5f}-{fvg['top']:.5f}]")
                    if self._detect_momentum_entry(last_3_candles, daily_bias):
                        logger.info(f"{analysis['symbol']}: FVG entry on directional candle")
                        return self._setup_fvg_entry(daily_bias, current_price, fvg, manipulation, ohlc_df, analysis, spread)

        logger.debug(f"{analysis['symbol']}: No reversal entry found at current price")
        return None, None, None

    def _find_continuation_entry(self, analysis, daily_bias, current_price, ohlc_df, spread, manipulation):
        """Finds continuation entries, explicitly ignoring manipulation for SL calculation."""
        current_high = ohlc_df['high'].iloc[-1]
        current_low = ohlc_df['low'].iloc[-1]
        last_3_candles = ohlc_df.tail(3)
        
        logger.debug(f"{analysis['symbol']}: Checking CONTINUATION entries. Current: {current_price:.5f}")

        # --- ADVANCED FVG ENTRY (HIGHEST PRIORITY) ---
        fvg_entry = self._find_retracement_fvg(analysis)
        if fvg_entry:
            if self._is_price_at_poi_now(current_price, current_high, current_low, fvg_entry):
                logger.info(f"{analysis['symbol']}: Price at high-probability {fvg_entry['type']} FVG NOW [{fvg_entry['bottom']:.5f}-{fvg_entry['top']:.5f}]")
                if self._detect_momentum_entry(last_3_candles, daily_bias):
                    logger.info(f"{analysis['symbol']}: High-probability FVG entry on directional candle")
                    # Pass manipulation=None to use POI-based SL for continuation
                    return self._setup_fvg_entry(daily_bias, current_price, fvg_entry, None, ohlc_df, analysis, spread)

        # --- ORDER BLOCK ENTRY ---
        order_blocks = analysis.get('order_blocks', [])
        for ob in order_blocks:
            if ob['type'] == daily_bias:
                if self._is_price_at_poi_now(current_price, current_high, current_low, ob):
                    logger.info(f"{analysis['symbol']}: Price at {ob['type']} ORDER BLOCK NOW [{ob['bottom']:.5f}-{ob['top']:.5f}]")
                    if self._detect_rejection_pattern(last_3_candles, daily_bias, ob):
                        logger.info(f"{analysis['symbol']}: ORDER BLOCK entry confirmed with rejection")
                        # Pass manipulation=None to use POI-based SL for continuation
                        return self._setup_order_block_entry(daily_bias, current_price, ob, None, ohlc_df, analysis, spread)

        # --- OTE ENTRY ---
        ote_zones = analysis.get('ote_zones', [])
        for ote in ote_zones:
            if ote['direction'].lower() == daily_bias:
                if ote['low'] <= current_price <= ote['high']:
                    logger.info(f"{analysis['symbol']}: Price in OTE zone [{ote['low']:.5f}-{ote['high']:.5f}]")
                    if self._detect_momentum_entry(last_3_candles, daily_bias):
                        logger.info(f"{analysis['symbol']}: OTE entry on momentum")
                        # Pass manipulation=None to use POI-based SL for continuation
                        return self._setup_ote_entry(daily_bias, current_price, ote_zones, None, ohlc_df, spread, analysis)

        logger.debug(f"{analysis['symbol']}: No continuation entry found at current price")
        return None, None, None
    
    def _is_price_at_poi_now(self, current_price: float, current_high: float, current_low: float, 
                         poi: Dict, tolerance_percent: float = 0.0005) -> bool:
        """
        Check if current price is at or very near a POI.
        For LIVE trading - checks if price is touching the POI NOW.
        """
        poi_top = poi.get('top', 0)
        poi_bottom = poi.get('bottom', 0)
        
        # Check if current candle is touching/penetrating the POI
        if current_low <= poi_top and current_high >= poi_bottom:
            # Price is definitely in the POI zone
            return True
        
        # Check if we're very close (within tolerance) - for spread considerations
        tolerance = current_price * tolerance_percent
        
        # Check proximity to POI boundaries
        distance_to_top = abs(current_price - poi_top)
        distance_to_bottom = abs(current_price - poi_bottom)
        distance_to_center = abs(current_price - (poi_top + poi_bottom) / 2)
        
        min_distance = min(distance_to_top, distance_to_bottom, distance_to_center)
        
        return min_distance <= tolerance
    
    def _detect_rejection_pattern(self, last_candles: pd.DataFrame, bias: str, poi: Dict) -> bool:
        """
        Detect rejection pattern in last 2-3 candles.
        More flexible for LIVE trading - accepts various rejection patterns.
        """
        if len(last_candles) < 2:
            return False
        
        poi_top = poi.get('top', 0)
        poi_bottom = poi.get('bottom', 0)
        
        if bias == 'bullish':
            # Look for bullish rejection patterns
            for i in range(len(last_candles)):
                candle = last_candles.iloc[i]
                
                # Pattern 1: Hammer/Pin bar with wick below POI
                if candle['low'] <= poi_bottom:
                    body = abs(candle['close'] - candle['open'])
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    
                    # Strong rejection wick
                    if lower_wick > body * 1.5 and candle['close'] > poi_bottom:
                        logger.debug(f"Bullish rejection: Pin bar at POI")
                        return True
                
                # Pattern 2: Bullish engulfing after touching POI
                if i > 0:
                    prev_candle = last_candles.iloc[i-1]
                    if (prev_candle['low'] <= poi_bottom and 
                        candle['close'] > candle['open'] and
                        candle['close'] > prev_candle['high']):
                        logger.debug(f"Bullish rejection: Engulfing pattern")
                        return True
                        
        else:  # bearish
            # Look for bearish rejection patterns
            for i in range(len(last_candles)):
                candle = last_candles.iloc[i]
                
                # Pattern 1: Shooting star with wick above POI
                if candle['high'] >= poi_top:
                    body = abs(candle['close'] - candle['open'])
                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    
                    # Strong rejection wick
                    if upper_wick > body * 1.5 and candle['close'] < poi_top:
                        logger.debug(f"Bearish rejection: Shooting star at POI")
                        return True
                
                # Pattern 2: Bearish engulfing after touching POI
                if i > 0:
                    prev_candle = last_candles.iloc[i-1]
                    if (prev_candle['high'] >= poi_top and 
                        candle['close'] < candle['open'] and
                        candle['close'] < prev_candle['low']):
                        logger.debug(f"Bearish rejection: Engulfing pattern")
                        return True
        
        return False
    
    def _detect_momentum_entry(self, last_candles: pd.DataFrame, bias: str) -> bool:
        """
        Detect momentum for entry - less strict than full rejection pattern.
        For LIVE trading where we need quick decisions.
        """
        if len(last_candles) < 1:
            return False
        
        # Check last candle for momentum
        last_candle = last_candles.iloc[-1]
        
        if bias == 'bullish':
            # Bullish momentum: close > open and decent body
            if last_candle['close'] > last_candle['open']:
                body = last_candle['close'] - last_candle['open']
                range_size = last_candle['high'] - last_candle['low']
                # Body should be at least 40% of range
                return body >= (range_size * 0.4)
        else:  # bearish
            # Bearish momentum: close < open and decent body
            if last_candle['close'] < last_candle['open']:
                body = last_candle['open'] - last_candle['close']
                range_size = last_candle['high'] - last_candle['low']
                # Body should be at least 40% of range
                return body >= (range_size * 0.4)
        
        return False
    
    def _setup_order_block_entry(self, bias, current_price, ob, ohlc_df, analysis, manipulation, spread):
        """Setup entry from Order Block with proper SL/TP."""
        manipulation_level = manipulation.get('level')
        sl_price = self._get_structural_sl_price(bias, ob, ohlc_df, analysis, manipulation_level=manipulation_level, spread=spread)
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        signal = "BUY" if bias == 'bullish' else "SELL"
        logger.info(f"Order Block Entry Setup: {signal} @ {current_price:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
        return signal, sl_price, tp_price
        
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
                             if fvg_pos <= break_pos:
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
    
    def _calculate_atr_buffer(self, ohlc_df):
        """Calculate ATR-based buffer for stop loss."""
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        return atr.iloc[-1] * self.config.SL_ATR_MULTIPLIER
    
    def _calculate_target(self, entry_price, sl_price, bias, analysis):
        """
        Calculate take profit with a hierarchical approach:
        1. Liquidity Targeting
        2. Equilibrium Fallback
        3. Fibonacci Extension Fallback
        """
        risk = abs(entry_price - sl_price)
        ohlc_df = analysis.get('ohlc_df')
        
        if ohlc_df is None:
            logger.warning("No OHLC data available for target analysis.")
            return None

        # 1. Primary: Target Liquidity
        liquidity_levels = self.liquidity_detector.get_liquidity_levels(
            ohlc_df, analysis.get('session', {}), analysis.get('daily_df')
        )
        min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)
        best_target = self.liquidity_detector.get_target_for_bias(
            bias, liquidity_levels, entry_price, min_rr, sl_price
        )
        
        if best_target:
            target_level = best_target['level']
            logger.info(f"Using {best_target['description']} at {target_level:.5f} for TP (Priority: {best_target['priority']})")
            return target_level

        # 2. Fallback: Target Equilibrium
        logger.debug("No suitable liquidity targets found, trying equilibrium fallback.")
        premium_discount = analysis.get('premium_discount')
        equilibrium_target = self._calculate_target_equilibrium(entry_price, bias, premium_discount)
        if equilibrium_target:
            logger.info(f"Using Equilibrium fallback target at {equilibrium_target:.5f}")
            return equilibrium_target

        # 3. Fallback: Target Fibonacci Extension
        logger.debug("Equilibrium target not suitable, trying Fibonacci extension fallback.")
        fib_target = self._calculate_target_fibonacci(entry_price, bias, premium_discount)
        if fib_target:
            logger.info(f"Using Fibonacci extension fallback target at {fib_target:.5f}")
            return fib_target
            
        logger.warning("Could not determine any valid TP target (Liquidity, Equilibrium, or Fib Extension).")
        return None

    def _calculate_target_equilibrium(self, entry_price, bias, premium_discount):
        """Calculates TP target using the premium/discount equilibrium as a fallback."""
        if premium_discount and 'equilibrium' in premium_discount:
            equilibrium = premium_discount['equilibrium']
            if (bias == 'bullish' and equilibrium > entry_price) or \
               (bias == 'bearish' and equilibrium < entry_price):
                return equilibrium
        return None

    def _calculate_target_fibonacci(self, entry_price, bias, premium_discount):
        """Calculates TP target using Fibonacci extensions as a second fallback."""
        if not premium_discount or 'swing_high' not in premium_discount or 'swing_low' not in premium_discount:
            return None

        swing_high = premium_discount['swing_high']
        swing_low = premium_discount['swing_low']
        range_size = swing_high - swing_low

        if range_size <= 0:
            return None

        # Using the -0.27 extension level as the first target
        fib_level = -0.27 
        
        if bias == 'bullish':
            target = swing_high - (range_size * fib_level)
            if target > entry_price:
                return target
        elif bias == 'bearish':
            target = swing_low + (range_size * fib_level)
            if target < entry_price:
                return target
            
        return None
    
    def _validate_levels(self, signal, entry, sl, tp):
        """Validate entry, SL, and TP levels."""
        if tp is None or sl is None or entry is None:
            return False
        if signal == "BUY":
            return sl < entry < tp
        else:
            return tp < entry < sl