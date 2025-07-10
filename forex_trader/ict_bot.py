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

logger = logging.getLogger(__name__)

@dataclass
class ICTNarrative:
    """Represents the complete ICT narrative for a trading opportunity."""
    daily_bias: str  # 'bullish' or 'bearish'
    po3_phase: str  # 'accumulation', 'manipulation', 'distribution'
    manipulation_confirmed: bool  # Judas swing or liquidity sweep
    manipulation_level: float  # Price level of manipulation
    structure_broken: bool  # BOS/CHoCH confirmed
    structure_level: float  # Level that was broken
    in_killzone: bool
    killzone_name: str
    ote_zone: Dict[str, float]  # {'high': x, 'sweet': x, 'low': x}
    order_blocks: List[Dict]  # Refined OBs in play
    current_price: float
    entry_model: str  # Which ICT model is being used

class ICTAnalyzer:
    """Analyzes price action through the ICT lens."""
    
    def __init__(self, config):
        self.config = config
        self.swing_lookback = config.SWING_LOOKBACK
        self.structure_lookback = config.STRUCTURE_LOOKBACK
        
    def analyze(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Dict:
        """
        Perform comprehensive ICT analysis following the narrative sequence.
        Now properly uses daily data for bias determination and real H4 data for HTF levels.
        """
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback:
            logger.warning(f"{symbol}: Insufficient data for ICT analysis")
            return {}
        
        # Step 1: Get foundational data
        swings = self._get_swings(ohlc_df)
        session_context = self._get_session_context(ohlc_df)
        if session_context.get('error'):
            logger.warning(f"{symbol}: Could not determine session context. Reason: {session_context['error']}")
            return {}

        # Step 2: Determine Daily Bias and PO3 using daily data
        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_context, symbol, daily_df, swings)

        manipulation = po3_analysis.get('manipulation', {'detected': False})
        
        # Step 3: Analyze the rest of the market concepts using the robust bias
        structure = self._get_structure(ohlc_df, swings, daily_bias)
        order_blocks = self._get_order_blocks(ohlc_df, swings)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        liquidity_zones = self._get_liquidity(ohlc_df, swings)
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, daily_bias, manipulation)
        htf_levels = self._get_htf_levels(h4_df)
        
        analysis_result = {
            'symbol': symbol, 'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1], 'swings': swings, 'structure': structure,
            'daily_bias': daily_bias, 'po3_analysis': po3_analysis, 'manipulation': manipulation,
            'order_blocks': order_blocks, 'fair_value_gaps': fair_value_gaps,
            'liquidity_zones': liquidity_zones, 'premium_discount': pd_analysis,
            'ote_zones': ote_zones, 'session': session_context, 'htf_levels': htf_levels
        }
        
        return analysis_result

    def _get_htf_levels(self, h4_df):
        """Get higher timeframe levels using REAL H4 data from broker."""
        if h4_df is None or h4_df.empty or len(h4_df) < 2:
            logger.warning("Insufficient H4 data for HTF levels")
            return {}
        
        try:
            # Use the last two complete H4 candles                           
            previous_h4 = h4_df.iloc[-2]  # Previous complete H4 candle
            current_h4 = h4_df.iloc[-1]   # Current H4 candle
            
            # Check if current price has broken previous levels
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
        """
        Get proper ICT session context with corrected Asian range timing
        """
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
            
            # Calculate Asian Range (7 PM - 10 PM NY from previous day)
            current_ny_date = latest_ny_time.date()
            current_ny_hour = latest_ny_time.hour
            current_ny_minute = latest_ny_time.minute
            
            # Determine which Asian range to use
            # We always want yesterday's Asian range for today's London/NY sessions
            if current_ny_hour < 19:  # Before 7 PM today
                asian_date = current_ny_date - pd.Timedelta(days=1)
            else:  # After 7 PM, we're in today's Asian session
                # But for trading tomorrow's London, we still want today's Asian range
                asian_date = current_ny_date
            
            # Create Asian range times. Start is fixed, end is dynamic.
            asian_start_ny = latest_ny_time.replace(
                year=asian_date.year,
                month=asian_date.month,
                day=asian_date.day,
                hour=self.config.ICT_ASIAN_RANGE['start'].hour, # Use config for start time (19:00 NY)
                minute=self.config.ICT_ASIAN_RANGE['start'].minute,
                second=0,
                microsecond=0
            )

            # The Asian range for London analysis should extend to the start of the London Kill Zone
            # to capture all relevant liquidity, not end at a fixed time like 10 PM.
            london_kz_start_time = self.config.ICT_LONDON_KILLZONE['start']
            london_kz_date = asian_date + pd.Timedelta(days=1) # London KZ is on the next calendar day

            asian_end_ny = latest_ny_time.replace(
                year=london_kz_date.year,
                month=london_kz_date.month,
                day=london_kz_date.day,
                hour=london_kz_start_time.hour, # Use London KZ start hour (e.g., 2 AM)
                minute=london_kz_start_time.minute,
                second=0,
                microsecond=0
            )
            
            # Convert to UTC for data filtering
            asian_start_utc = asian_start_ny.astimezone(pytz.UTC)
            asian_end_utc = asian_end_ny.astimezone(pytz.UTC)
            
            # Get the Asian range data
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
            # London Kill Zone: 2-5 AM NY time
            if 2 <= current_ny_hour < 5:
                context['in_killzone'] = True
                context['killzone_name'] = 'London'
                
            # New York Kill Zone: 8:30-11 AM NY time
            elif (current_ny_hour == 8 and current_ny_minute >= 30) or (9 <= current_ny_hour < 11):
                context['in_killzone'] = True
                context['killzone_name'] = 'NewYork'
                
            # London Close Kill Zone: 3-5 PM NY time
            elif 15 <= current_ny_hour < 17:
                context['in_killzone'] = True
                context['killzone_name'] = 'LondonClose'
                
        except Exception as e:
            context['error'] = f"Error calculating session context: {e}"
            logger.error(context['error'], exc_info=True)
                        
        return context
    
    def _determine_daily_bias(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None) -> Tuple[str, Dict]:
        """
        Determines bias using actual daily timeframe data
        """
        logger.debug(f"\n--- {symbol} ICT BIAS CHECKLIST ---")

        # Daily bias MUST use daily data - no fallback to M15
        if daily_df is None or daily_df.empty:
            logger.error(f"{symbol}: No daily data provided for bias determination. Cannot determine bias without D1 data.")
            return 'neutral', {'error': 'No daily data available'}
        
        analysis_df = daily_df

        # 1. Gather all evidence from source functions
        htf_order_flow = self._analyze_daily_order_flow(analysis_df)
        liquidity_draw = self._analyze_liquidity_draw(ohlc_df)  # Still use M15 for immediate liquidity
        swings = self._get_swings(ohlc_df)  # M15 swings for PD zones
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        is_in_discount = 'discount' in pd_analysis.get('current_zone', '')
        is_in_premium = 'premium' in pd_analysis.get('current_zone', '')
        
        final_bias = 'neutral'
        reasons = []

        # 2. Apply ICT rules with hierarchy (most important checks first)
        # The primary driver is HTF Order Flow from daily chart
        if htf_order_flow == 'bullish':
            reasons.append("Primary Factor: Daily Order Flow is Bullish (HH+HL)")
            final_bias = 'bullish'
            
            # Additional factors strengthen the bias but aren't required
            if is_in_discount:
                reasons.append("Supporting Factor: Price is in a Discount zone")
            elif liquidity_draw == 'bullish':
                reasons.append("Supporting Factor: Draw on Liquidity is Buyside")
        elif htf_order_flow == 'bearish':
            reasons.append("Primary Factor: Daily Order Flow is Bearish (LH+LL)")
            final_bias = 'bearish'
            
            if is_in_premium:
                reasons.append("Supporting Factor: Price is in a Premium zone")
            elif liquidity_draw == 'bearish':
                reasons.append("Supporting Factor: Draw on Liquidity is Sellside")
        else:
            reasons.append("Primary Factor: Daily Order Flow is Neutral. No clear bias.")
            final_bias = 'neutral'
        
        bias_details = {
            'htf_order_flow': htf_order_flow,
            'liquidity_draw': liquidity_draw,
            'premium_discount_zone': pd_analysis.get('current_zone', 'unknown'),
            'final_bias_decision': final_bias,
            'reasoning': ' | '.join(reasons)
        }
        
        return final_bias, bias_details

    def _analyze_session_po3(self, ohlc_df: pd.DataFrame, session_context: dict, symbol: str, daily_df: pd.DataFrame = None, swings: pd.DataFrame = None) -> Tuple[str, Dict]:
        """
        Orchestrates bias analysis and identifies PO3 phase based on session context and manipulation patterns.
        Now properly uses daily data for bias determination and session-based PO3 phase detection.
        """
        daily_bias, bias_details = self._determine_daily_bias(ohlc_df, symbol, daily_df)
        manipulation_details = self._check_manipulation_patterns(ohlc_df, daily_bias, session_context, swings)

        # Get current session info for PO3 phase determination
        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        
        po3_analysis = {
            'current_phase': 'unknown',
            'type': 'session_based',
            'manipulation': {'detected': False},
            'bias_details': bias_details
        }
        
        # Determine PO3 phase based on ICT session structure
        if 19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 1:
            # Asian Session = Accumulation Phase
            po3_analysis['current_phase'] = 'accumulation'
            po3_analysis['session'] = 'Asian'
            logger.debug(f"{symbol}: Asian Session - Accumulation Phase")
            
        elif 2 <= current_ny_hour < 5:
            # London Session = Manipulation Phase
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
            # New York Session = Distribution Phase
            if manipulation_details:
                po3_analysis['current_phase'] = 'distribution'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
                po3_analysis['type'] = manipulation_details.get('type', 'unknown')
                logger.debug(f"{symbol}: NY Session - Distribution Phase after {manipulation_details.get('type')}")
            else:
                # Can still be distribution if we had earlier manipulation
                # Check if we're in a kill zone and have a clear bias
                if session_context.get('in_killzone') and daily_bias != 'neutral':
                    po3_analysis['current_phase'] = 'distribution_continuation'
                    logger.debug(f"{symbol}: NY Session - Distribution Continuation (no new manipulation)")
                else:
                    po3_analysis['current_phase'] = 'distribution_pending'
                    logger.debug(f"{symbol}: NY Session - Waiting for Setup")
            po3_analysis['session'] = 'NewYork'
            
        # FIX 2: Add London Close Session
        elif 15 <= current_ny_hour < 17:
            # London Close Session = Can be Reversal or Continuation
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
            # Outside main sessions
            if manipulation_details:
                po3_analysis['current_phase'] = 'post_distribution'
                po3_analysis['manipulation'] = manipulation_details
                po3_analysis['manipulation']['detected'] = True
            else:
                po3_analysis['current_phase'] = 'consolidation'
            po3_analysis['session'] = 'Other'
            logger.debug(f"{symbol}: Outside main sessions - {po3_analysis['current_phase']}")
        
        # Set manipulation details if found
        if manipulation_details:
            po3_analysis['manipulation'] = manipulation_details
            po3_analysis['manipulation']['detected'] = True
            po3_analysis['type'] = manipulation_details.get('type')
        
        logger.info(f"{symbol}: PO3 Phase: {po3_analysis['current_phase']} | Session: {po3_analysis.get('session', 'Unknown')} | Bias: {daily_bias}")
        
        return daily_bias, po3_analysis
            
    def _analyze_daily_order_flow(self, daily_df):
        """
        Analyze ACTUAL daily timeframe order flow.
        ICT: "Banks and institutional traders mostly utilize the daily chart"
        """
        if daily_df is None or daily_df.empty or len(daily_df) < 20:
            logger.warning("Insufficient daily data for order flow analysis")
            return 'neutral'
        
        # Use appropriate swing detection for daily timeframe
        swings = smc.swing_highs_lows(daily_df, swing_length=5)  # 5 days lookback
        
        if swings.empty:
            return 'neutral'
        
        # Get swing highs and lows
        swing_highs = swings[swings['HighLow'] == 1]['Level'].dropna()
        swing_lows = swings[swings['HighLow'] == -1]['Level'].dropna()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral'
        
        # Check last two swing highs and lows
        last_two_highs = swing_highs.tail(2).values
        last_two_lows = swing_lows.tail(2).values
        
        # Higher high and higher low = bullish
        if last_two_highs[1] > last_two_highs[0] and last_two_lows[1] > last_two_lows[0]:
            logger.info(f"Daily Order Flow: BULLISH (HH: {last_two_highs[0]:.5f} -> {last_two_highs[1]:.5f}, HL: {last_two_lows[0]:.5f} -> {last_two_lows[1]:.5f})")
            return 'bullish'
        # Lower high and lower low = bearish
        elif last_two_highs[1] < last_two_highs[0] and last_two_lows[1] < last_two_lows[0]:
            logger.info(f"Daily Order Flow: BEARISH (LH: {last_two_highs[0]:.5f} -> {last_two_highs[1]:.5f}, LL: {last_two_lows[0]:.5f} -> {last_two_lows[1]:.5f})")
            return 'bearish'
        
        return 'neutral'

    def _analyze_liquidity_draw(self, analysis_df, is_daily_context=False):
        """
        analysis_df: DataFrame to analyze (daily_df for bias, ohlc_df for entry)
        is_daily_context: True when called for bias determination
        """
        if len(analysis_df) < 5:
            return 'neutral'
        
        current_price = analysis_df['close'].iloc[-1]
        
        if is_daily_context:
            # For BIAS: Look at weekly/monthly levels using daily data
            if len(analysis_df) > 7:  # Need at least a week of daily data
                prev_week_high = analysis_df['high'].iloc[-7:-1].max()  # Last week's high
                prev_week_low = analysis_df['low'].iloc[-7:-1].min()    # Last week's low
                
                # Distance to weekly liquidity
                dist_to_weekly_high = prev_week_high - current_price
                dist_to_weekly_low = current_price - prev_week_low
                
                if dist_to_weekly_low < dist_to_weekly_high * 0.3:
                    return 'bullish'  # Closer to weekly low
                elif dist_to_weekly_high < dist_to_weekly_low * 0.3:
                    return 'bearish'  # Closer to weekly high
        else:
            # For ENTRY: Keep existing M15 logic
            bars_per_day = 96  # M15 bars
            if len(analysis_df) > bars_per_day:
                prev_day_high = analysis_df['high'].iloc[-bars_per_day:-bars_per_day//2].max()
                prev_day_low = analysis_df['low'].iloc[-bars_per_day:-bars_per_day//2].min()
                
                dist_to_high = prev_day_high - current_price
                dist_to_low = current_price - prev_day_low
                
                if dist_to_low < dist_to_high * 0.3:
                    return 'bullish'
                elif dist_to_high < dist_to_low * 0.3:
                    return 'bearish'
        
        return 'neutral'

    def _analyze_premium_discount(self, analysis_df: pd.DataFrame, swings: pd.DataFrame = None, is_daily_context=False) -> Dict:
        """
        analysis_df: DataFrame to analyze (daily_df for bias, ohlc_df for entry)
        swings: Swing data (daily swings for bias, M15 swings for entry)
        is_daily_context: True when called for bias determination
        """
        if swings is None or swings.empty:
            # Generate swings from the provided data
            if is_daily_context:
                swings = smc.swing_highs_lows(analysis_df, swing_length=5)  # 5 days for daily
            else:
                swings = smc.swing_highs_lows(analysis_df, swing_length=self.swing_lookback)  # M15 swings
        
        if swings.empty:
            return {}

        # Use last 20 swings for daily context, or existing logic for M15
        lookback_count = 10 if is_daily_context else 20
        recent_swings = swings.dropna().tail(lookback_count)
        
        if len(recent_swings) < 4:
            return {}

        # Rest of the function remains the same...
        high = recent_swings['Level'].max()
        low = recent_swings['Level'].min()
        
        if high == low:
            return {}

        range_size = high - low
        levels = {
            'high': high, 'low': low, 'range': range_size,
            'equilibrium': low + (range_size * 0.5),
            'premium_75': low + (range_size * 0.75),
            'premium_80': low + (range_size * 0.80),
            'discount_25': low + (range_size * 0.25),
            'discount_20': low + (range_size * 0.20),
        }
        
        current_price = analysis_df['close'].iloc[-1]
        
        if current_price > levels['equilibrium']:
            zone = 'premium'
        else:
            zone = 'discount'

        if zone == 'premium' and current_price > levels['premium_75']:
            zone = 'deep_premium'
        elif zone == 'discount' and current_price < levels['discount_25']:
            zone = 'deep_discount'
        
        levels['current_zone'] = zone
        return levels
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context, swings):
        """
        [FIXED] Checks for manipulation patterns with corrected Judas Swing timing
        """
        if daily_bias == 'neutral' or swings.empty:
            return None

        # Get current NY time for Judas Swing check
        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour

        # PATTERN 1: Judas Swing (Midnight to 5AM NY time - ICT specific timing)
        if 0 <= current_ny_hour < 5 and session_context.get('last_asian_range'):
            asian_high = session_context['last_asian_range']['high']
            asian_low = session_context['last_asian_range']['low']
            
            # Find the most recent opposing swing to serve as our MSS level
            swings_before_now = swings[swings.index < len(ohlc_df) - 1].dropna()

            if daily_bias == 'bearish':
                # Find the sweep of the Asian High
                search_start_time = session_context['last_asian_range']['end_time_utc']
                sweep_search_df = ohlc_df[ohlc_df.index >= search_start_time]
                sweep_up_candles = sweep_search_df[sweep_search_df['high'] > asian_high]
                
                if not sweep_up_candles.empty:
                    sweep_candle_time = sweep_up_candles.index[0]
                    sweep_index = ohlc_df.index.get_loc(sweep_candle_time)
                    
                    # Find the most recent swing low before the sweep to be the MSS level
                    opposing_swings = swings_before_now[swings_before_now['HighLow'] == -1]
                    if not opposing_swings.empty:
                        mss_level = opposing_swings.iloc[-1]['Level']
                        
                        # Check for a close below the MSS level after the sweep
                        post_sweep_data = ohlc_df.iloc[sweep_index + 1:]
                        if any(post_sweep_data['close'] < mss_level):
                            logger.debug(f"Bearish Judas Swing detected at {current_ny_hour}:00 NY time")
                            return {'type': 'bearish_judas', 'level': sweep_up_candles['high'].iloc[0], 
                                'index': sweep_index, 'mss_level': mss_level, 'detected': True}

            elif daily_bias == 'bullish':
                # Find the sweep of the Asian Low
                search_start_time = session_context['last_asian_range']['end_time_utc']
                sweep_search_df = ohlc_df[ohlc_df.index >= search_start_time]
                sweep_down_candles = sweep_search_df[sweep_search_df['low'] < asian_low]

                if not sweep_down_candles.empty:
                    sweep_candle_time = sweep_down_candles.index[0]
                    sweep_index = ohlc_df.index.get_loc(sweep_candle_time)

                    # Find the most recent swing high before the sweep to be the MSS level
                    opposing_swings = swings_before_now[swings_before_now['HighLow'] == 1]
                    if not opposing_swings.empty:
                        mss_level = opposing_swings.iloc[-1]['Level']

                        # Check for a close above the MSS level after the sweep
                        post_sweep_data = ohlc_df.iloc[sweep_index + 1:]
                        if any(post_sweep_data['close'] > mss_level):
                            logger.debug(f"Bullish Judas Swing detected at {current_ny_hour}:00 NY time")
                            return {'type': 'bullish_judas', 'level': sweep_down_candles['low'].iloc[0], 
                                'index': sweep_index, 'mss_level': mss_level, 'detected': True}

        # PATTERN 2: Turtle Soup (General price action pattern)
        turtle_soup = self._check_turtle_soup_pattern(ohlc_df, daily_bias)
        if turtle_soup:
            return turtle_soup

        # PATTERN 3: General Liquidity Sweep (Lowest priority)
        liquidity_sweep = self._check_liquidity_sweep(ohlc_df, daily_bias, swings)
        if liquidity_sweep:
            return liquidity_sweep

        return None

    def _check_turtle_soup_pattern(self, ohlc_df, daily_bias):
        """
        [CORRECTED & DYNAMIC] Identifies a Turtle Soup pattern by scanning the
        last few closed candles to find a confirmed rejection.
        """
        # Define a window to scan, e.g., the last 5 closed candles.
        scan_window = 5 
        
        # Ensure there's enough data for a full scan.
        # We need 20 for the lookback + the scan_window size.
        if len(ohlc_df) < 20 + scan_window:
            return None
        
        # Loop backwards through the recent closed candles.
        for i in range(-2, -(scan_window + 2), -1):
            
            # Define the candle we are currently checking for a sweep.
            bar = ohlc_df.iloc[i]
            
            # <<< CHANGE 3: Define the 20-period lookback *relative to the current bar*.
            # This is the most critical part of the fix. The lookback history is now
            # dynamic and always precedes the candle being checked.
            lookback_df = ohlc_df.iloc[i-20 : i]
            
            # This logic remains the same, but is now applied to each candle in the loop.
            recent_high = lookback_df['high'].max()
            recent_low = lookback_df['low'].min()
            
            if daily_bias == 'bearish' and bar['high'] > recent_high:
                # Confirmation: The close must be strong and bearish.
                if bar['close'] < bar['open'] and bar['close'] < recent_high:
                    # If a pattern is found, return it immediately. We've found the most recent one.
                    return {'type': 'bearish_turtle_soup', 'level': bar['high'], 'swept_level': recent_high, 'index': len(ohlc_df) + i, 'detected': True}
            
            elif daily_bias == 'bullish' and bar['low'] < recent_low:
                # Confirmation: The close must be strong and bullish.
                if bar['close'] > bar['open'] and bar['close'] > recent_low:
                    # If a pattern is found, return it immediately.
                    return {'type': 'bullish_turtle_soup', 'level': bar['low'], 'swept_level': recent_low, 'index': len(ohlc_df) + i, 'detected': True}
        
        # If the loop completes without finding any pattern in the scan_window, return None.
        return None
    
    def _check_liquidity_sweep(self, ohlc_df, daily_bias, swings):
        """
        [CORRECTED] Identifies a sweep of a MAJOR swing point followed by a
        robust Market Structure Shift (MSS). Fix ensures there is a candle
        available for MSS confirmation.
        """
        if len(ohlc_df) < 10 or swings.empty:
            return None
        
        recent_swings = swings.dropna().tail(10)
        
        # <<< FIX: Change loop to range(-5, -1) to exclude the last candle.
        # This ensures a "post_sweep_data" candle always exists for the MSS check.
        for i in range(-5, -1): # Iterates from -5, -4, -3, -2.
            current_candle = ohlc_df.iloc[i]
            current_index = len(ohlc_df) + i

            # Filter for swings that occurred before the current candle
            swings_before_candle = recent_swings[recent_swings.index < current_index]
            if swings_before_candle.empty:
                continue

            if daily_bias == 'bullish':
                # ... (rest of the bullish logic is unchanged)
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['low'] < swept_level:
                    opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                    if opposing_swings.empty: continue
                    mss_level = opposing_swings.iloc[-1]['Level']
                    
                    # This check now works reliably as post_sweep_data is never empty.
                    post_sweep_data = ohlc_df.iloc[current_index + 1:]
                    if any(post_sweep_data['close'] > mss_level):
                        return {'type': 'bullish_liquidity_sweep_mss', 'level': current_candle['low'], 'swept_level': swept_level, 'index': current_index, 'mss_level': mss_level, 'detected': True}

            elif daily_bias == 'bearish':
                # ... (rest of the bearish logic is unchanged and now works reliably)
                target_swings = swings_before_candle[swings_before_candle['HighLow'] == 1]
                if target_swings.empty: continue
                swept_level = target_swings.iloc[-1]['Level']

                if current_candle['high'] > swept_level:
                    opposing_swings = swings_before_candle[swings_before_candle['HighLow'] == -1]
                    if opposing_swings.empty: continue
                    mss_level = opposing_swings.iloc[-1]['Level']
                    
                    post_sweep_data = ohlc_df.iloc[current_index + 1:]
                    if any(post_sweep_data['close'] < mss_level):
                        return {'type': 'bearish_liquidity_sweep_mss', 'level': current_candle['high'], 'swept_level': swept_level, 'index': current_index, 'mss_level': mss_level, 'detected': True}
        
        return None
    
    def _check_london_close_reversal(self, ohlc_df: pd.DataFrame, session_context: dict) -> Optional[Dict]:
        """
        [NEW & SPECIALIZED] Identifies a London Close Reversal pattern.
        This pattern looks for a sweep of the New York session high/low,
        followed by an aggressive close back inside the range, confirming a reversal.
        This function does NOT use a bias; it finds the pattern objectively.
        """
        # Ensure we have enough data and are in the correct session
        if len(ohlc_df) < 50 or not (session_context.get('in_killzone') and session_context['killzone_name'] == 'LondonClose'):
            return None

        # --- 1. Define the New York Session Range to be targeted ---
        try:
            latest_ny_time = ohlc_df.index[-1].astimezone(self.config.NY_TIMEZONE)
            current_day = latest_ny_time.date()

            # Define NY Killzone times (e.g., 8:30 AM to 11:00 AM NY)
            ny_kz_start_config = self.config.ICT_NEW_YORK_KILLZONE['start']
            ny_kz_end_config = self.config.ICT_NEW_YORK_KILLZONE['end']
            
            ny_start = latest_ny_time.replace(
                hour=ny_kz_start_config.hour, minute=ny_kz_start_config.minute, second=0, microsecond=0
            )
            ny_end = latest_ny_time.replace(
                hour=ny_kz_end_config.hour, minute=ny_kz_end_config.minute, second=0, microsecond=0
            )
            
            # Filter the DataFrame to get the price data from the NY Killzone
            ny_session_data = ohlc_df[(ohlc_df.index >= ny_start) & (ohlc_df.index <= ny_end)]

            if ny_session_data.empty:
                logger.debug("No New York session data found to define a range for London Close reversal.")
                return None
            
            ny_session_high = ny_session_data['high'].max()
            ny_session_low = ny_session_data['low'].min()
            
        except Exception as e:
            logger.error(f"Error defining NY session range for LC reversal: {e}")
            return None

        # --- 2. Scan the last few candles for a sweep and rejection ---
        scan_window = 5
        for i in range(-2, -(scan_window + 2), -1):
            if abs(i) > len(ohlc_df): continue # Prevent index error on small dataframes

            sweep_candle = ohlc_df.iloc[i]
            
            # --- Check for BEARISH Reversal (Sweep of NY High) ---
            if sweep_candle['high'] > ny_session_high:
                # CONFIRMATION: Did price close aggressively back BELOW the NY high?
                if sweep_candle['close'] < ny_session_high and sweep_candle['close'] < sweep_candle['open']:
                    logger.info(f"Confirmed London Close BEARISH Reversal: Sweep of NY high {ny_session_high:.5f}")
                    return {
                        'type': 'london_close_bearish_reversal',
                        'level': sweep_candle['high'], # The extreme of the sweep
                        'swept_level': ny_session_high,
                        'detected': True,
                        'index': len(ohlc_df) + i
                    }

            # --- Check for BULLISH Reversal (Sweep of NY Low) ---
            if sweep_candle['low'] < ny_session_low:
                # CONFIRMATION: Did price close aggressively back ABOVE the NY low?
                if sweep_candle['close'] > ny_session_low and sweep_candle['close'] > sweep_candle['open']:
                    logger.info(f"Confirmed London Close BULLISH Reversal: Sweep of NY low {ny_session_low:.5f}")
                    return {
                        'type': 'london_close_bullish_reversal',
                        'level': sweep_candle['low'], # The extreme of the sweep
                        'swept_level': ny_session_low,
                        'detected': True,
                        'index': len(ohlc_df) + i
                    }

        return None
    
    def _get_ict_structure(self, ohlc_df, swings: pd.DataFrame, daily_bias: str):
        """
        [METHODOLOGICALLY PURE] Detects BOS and CHoCH based purely on swing breaks.
        """
        n = len(ohlc_df)
        bos = np.full(n, np.nan)
        choch = np.full(n, np.nan)
        level = np.full(n, np.nan)
        broken_index = np.full(n, np.nan)
        
        relevant_swings = swings.dropna().tail(50)
        
        for swing_idx, swing in relevant_swings.iterrows():
            swing_level = swing['Level']
            swing_type = swing['HighLow']
            
            # Convert swing index to position in ohlc_df
            try:
                swing_position = ohlc_df.index.get_loc(swing_idx)
            except KeyError:
                continue  # Skip if swing index not found in ohlc_df
            
            # Data after the swing point was formed
            post_swing_data = ohlc_df.iloc[swing_position + 1:]
            
            if post_swing_data.empty:
                continue
                
            # Check for breaks
            if swing_type == 1:  # Swing HIGH
                breaking_candles = post_swing_data[post_swing_data['close'] > swing_level]
                if not breaking_candles.empty:
                    break_candle_idx = ohlc_df.index.get_loc(breaking_candles.index[0])
                    
                    if self._validate_displacement_break(ohlc_df, break_candle_idx, swing_level, 'bullish'):
                        if daily_bias == 'bullish':
                            bos[swing_position] = 1  # Use swing_position for array indexing
                            level[swing_position] = swing_level
                            broken_index[swing_position] = break_candle_idx
                        else:
                            choch[swing_position] = 1
                            level[swing_position] = swing_level
                            broken_index[swing_position] = break_candle_idx
            
            elif swing_type == -1:  # Swing LOW
                breaking_candles = post_swing_data[post_swing_data['close'] < swing_level]
                if not breaking_candles.empty:
                    break_candle_idx = ohlc_df.index.get_loc(breaking_candles.index[0])
                    
                    if self._validate_displacement_break(ohlc_df, break_candle_idx, swing_level, 'bearish'):
                        if daily_bias == 'bearish':
                            bos[swing_position] = -1
                            level[swing_position] = swing_level
                            broken_index[swing_position] = break_candle_idx
                        else:
                            choch[swing_position] = -1
                            level[swing_position] = swing_level
                            broken_index[swing_position] = break_candle_idx
        
        return pd.DataFrame({
            'BOS': bos,
            'CHOCH': choch, 
            'Level': level,
            'BrokenIndex': broken_index
        })
    
    def _validate_displacement_break(self, ohlc_df, break_index, level, direction):
        """
        Use existing displacement validation for structure breaks.
        Reuse the displacement logic from Judas swing detection.
        """
        # Get the breaking candle and previous few candles
        start_idx = max(0, break_index - 3)
        displacement_window = ohlc_df.iloc[start_idx:break_index + 1]
        
        if len(displacement_window) < 2:
            return False
        
        # Check for range expansion (institutional involvement)
        avg_range = (displacement_window['high'] - displacement_window['low']).mean()
        breaking_candle = displacement_window.iloc[-1]
        breaking_range = breaking_candle['high'] - breaking_candle['low']
        
        # Must be at least 1.5x average range (shows energy)
        if breaking_range < avg_range * 1.5:
            return False
        
        # Check for strong directional candle (body close)
        body_size = abs(breaking_candle['close'] - breaking_candle['open'])
        candle_range = breaking_range
        
        if direction == 'bullish':
            # Strong bullish candle: close in top 25% and body break
            close_position = (breaking_candle['close'] - breaking_candle['low']) / candle_range
            strong_candle = (close_position >= 0.75 and 
                            breaking_candle['close'] > breaking_candle['open'] and
                            breaking_candle['close'] > level)
        else:
            # Strong bearish candle: close in bottom 25% and body break
            close_position = (breaking_candle['close'] - breaking_candle['low']) / candle_range
            strong_candle = (close_position <= 0.25 and 
                            breaking_candle['close'] < breaking_candle['open'] and
                            breaking_candle['close'] < level)
        
        return strong_candle

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
            
            # Get recent unmitigated OBs
            recent_obs = obs[obs.index >= len(ohlc_df) - 50]
            unmitigated = recent_obs[
                recent_obs['OB'].notna() & 
                recent_obs['MitigatedIndex'].isna()
            ]
            
            # Convert to list of dicts for easier handling
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
            
            # Get recent unmitigated FVGs
            recent_fvgs = fvgs[fvgs.index >= len(ohlc_df) - 50]
            unmitigated = recent_fvgs[
                recent_fvgs['FVG'].notna() & 
                recent_fvgs['MitigatedIndex'].isna()
            ]
            
            # Convert to list
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
    
    def _get_liquidity(self, ohlc_df, swings):
        """Get liquidity zones."""
        try:
            range_percent = 0.01
            
            liquidity = smc.liquidity(ohlc_df, swings, range_percent=range_percent)
            if liquidity.empty:
                return []
            
            # Get recent liquidity zones (last 50 candles)
            recent_liq = liquidity.iloc[-50:] if len(liquidity) > 50 else liquidity
            
            # Filter for valid, unswept liquidity
            # Swept == 0 means unswept, Swept > 0 means swept at that index
            unswept = recent_liq[
                recent_liq['Liquidity'].notna() & 
                (recent_liq['Swept'] == 0)
            ]
            
            # Convert to list
            liq_list = []
            for idx, liq in unswept.iterrows():
                liq_list.append({
                    'index': idx,
                    'type': 'bullish' if liq['Liquidity'] == 1 else 'bearish',
                    'level': liq['Level']
                })
            
            logger.debug(f"Found {len(liq_list)} unswept liquidity zones")
            return liq_list
            
        except Exception as e:
            logger.error(f"Error getting liquidity: {e}")
            return []
    
    def _calculate_ote_zones(self, ohlc_df, daily_bias, manipulation):
        """
        Calculate OTE zones following ICT's actual methodology.
        Using standard Fibonacci retracement percentages.
        OTE Zone = 62% to 79% retracement of the impulse move.
        """
        if not manipulation.get('detected'):
            return []

        ote_zones = []
        manipulation_level = manipulation.get('level')
        manipulation_index = manipulation.get('index', -1)

        if manipulation_index < 0 or manipulation_index >= len(ohlc_df) - 1:
            logger.debug("OTE Calc: Manipulation is too recent or index is invalid.")
            return []

        post_manipulation_data = ohlc_df.iloc[manipulation_index + 1:]
        if post_manipulation_data.empty:
            logger.debug("OTE Calc: No candles after manipulation event.")
            return []

        if daily_bias == 'bullish':
            # Bullish: Price moves from manipulation LOW up to a HIGH, then retraces
            swing_low = manipulation_level
            swing_high = post_manipulation_data['high'].max()
            
            if swing_high > swing_low:
                range_size = swing_high - swing_low
                
                # Price retraces DOWN from the high
                # 62% retracement = high - (range * 0.62)
                # 79% retracement = high - (range * 0.79)
                ote_zones.append({
                    'direction': 'bullish',
                    'high': swing_high - (range_size * 0.62),   # 62% retracement
                    'sweet': swing_high - (range_size * 0.705), # 70.5% retracement
                    'low': swing_high - (range_size * 0.79),    # 79% retracement
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'equilibrium': swing_low + (range_size * 0.5)
                })
                logger.debug(f"Bullish OTE: Impulse {swing_low:.5f} to {swing_high:.5f}, "
                            f"OTE Zone {ote_zones[-1]['low']:.5f} to {ote_zones[-1]['high']:.5f}")

        elif daily_bias == 'bearish':
            # Bearish: Price moves from manipulation HIGH down to a LOW, then retraces
            swing_high = manipulation_level
            swing_low = post_manipulation_data['low'].min()
            
            if swing_high > swing_low:
                range_size = swing_high - swing_low
                
                # Price retraces UP from the low
                # 62% retracement = low + (range * 0.62)
                # 79% retracement = low + (range * 0.79)
                ote_zones.append({
                    'direction': 'bearish',
                    'low': swing_low + (range_size * 0.62),    # 62% retracement
                    'sweet': swing_low + (range_size * 0.705),  # 70.5% retracement
                    'high': swing_low + (range_size * 0.79),   # 79% retracement
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'equilibrium': swing_high - (range_size * 0.5)
                })
                logger.debug(f"Bearish OTE: Impulse {swing_high:.5f} to {swing_low:.5f}, "
                            f"OTE Zone {ote_zones[-1]['low']:.5f} to {ote_zones[-1]['high']:.5f}")
            
        return ote_zones

    def _analyze_market_phase(self, ohlc_df, manipulation, daily_bias):
        """
        [CORRECTED 2.0] Determine current ICT PO3 phase and whether entry is viable.
        Fixes session time-checking logic and simplifies phase determination.
        """
        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        current_ny_minute = current_ny.minute
        current_price = ohlc_df['close'].iloc[-1]

        # Diagnostic Log: This will show us exactly what hour is being checked.
        logger.debug(f"Analyzing market phase for NY time: {current_ny.strftime('%H:%M')}, Hour={current_ny_hour}")

        manipulation_detected = manipulation.get('detected', False)
        manipulation_index = manipulation.get('index', -1)
        manipulation_price = manipulation.get('level', current_price)

        phase = 'unknown'
        can_enter = False
        session_name = 'Other'
        
        # 1. Determine Correct Session and Phase
        # London Session Kill Zone (2 AM - 5 AM NY)
        if 2 <= current_ny_hour < 5:
            session_name = 'London'
            if manipulation_detected:
                # Manipulation happened. We are in the distribution/entry phase.
                phase = 'manipulation_complete'
                can_enter = True # It's now valid to look for an entry
            else:
                # Still waiting for manipulation.
                phase = 'manipulation_pending'
                can_enter = False

        # New York Session Kill Zone (8:30 AM - 11 AM NY)
        elif (current_ny_hour == 8 and current_ny_minute >= 30) or (9 <= current_ny_hour < 11):
            session_name = 'NewYork'
            if manipulation_detected:
                # Manipulation from London (or a new one) exists. We are in the distribution phase.
                phase = 'distribution'
                # For NY, we often wait for a retracement to an OTE or FVG.
                # The main signal logic will handle this; here we just permit entry checks.
                can_enter = True
            else:
                # NY open but no prior manipulation to trade off of.
                phase = 'distribution_pending'
                can_enter = False

        # Asian Session (7 PM - 2 AM NY)
        elif 19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 2:
            session_name = 'Asian'
            phase = 'accumulation'
            can_enter = False
        
        # Outside of main sessions
        else:
            session_name = 'Other'
            if manipulation_detected:
                phase = 'post_distribution'
            else:
                phase = 'consolidation'
            can_enter = False

        # 2. Calculate Retracement (for context, not direct entry logic here)
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

        # Invalidate if price has aggressively reversed against the manipulation
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
        ICT-compliant displacement and MSS confirmation following actual ICT methodology.
        
        Displacement Requirements:
        - At least 3 consecutive candles in same direction (or 1-2 strong ones on HTF)
        - Large bodies (60%+ of candle range)
        - Minimal wicks
        - Range expansion (1.5x+ average)
        - Creates Fair Value Gaps (preferred but not required)
        
        MSS Requirements:
        - BODY must close beyond swing level (not just wick)
        - The break must be accompanied by displacement
        - Bullish MSS: Body closes above previous swing high
        - Bearish MSS: Body closes below previous swing low
        """
        if not self.config.REQUIRE_ENTRY_CONFIRMATION:
            return True

        confirmation_window_size = 5
        start_idx = reaction_index + 1
        end_idx = start_idx + confirmation_window_size
        
        if end_idx > len(ohlc_df):
            logger.debug("Not enough candles after POI reaction to confirm signal.")
            return False
            
        confirmation_df = ohlc_df.iloc[start_idx:end_idx]
        if confirmation_df.empty:
            return False

        # Find MSS level (recent swing to break)
        swings_before_reaction = swings[swings.index < reaction_index].dropna()
        if swings_before_reaction.empty:
            return False
            
        mss_level = None
        if bias == 'bullish':
            # For bullish, we need to break above a swing HIGH
            recent_swing_highs = swings_before_reaction[swings_before_reaction['HighLow'] == 1]
            if not recent_swing_highs.empty:
                mss_level = recent_swing_highs['Level'].iloc[-1]
        else:
            # For bearish, we need to break below a swing LOW
            recent_swing_lows = swings_before_reaction[swings_before_reaction['HighLow'] == -1]
            if not recent_swing_lows.empty:
                mss_level = recent_swing_lows['Level'].iloc[-1]
        
        if mss_level is None:
            logger.debug("Could not determine valid MSS level for confirmation.")
            return False

        # Calculate average range for displacement comparison
        avg_range = (ohlc_df['high'] - ohlc_df['low']).iloc[reaction_index-20:reaction_index].mean()
        
        # Track displacement candles
        displacement_candles = []
        mss_confirmed = False
        
        for i, (_, candle) in enumerate(confirmation_df.iterrows()):
            candle_range = candle['high'] - candle['low']
            body_size = abs(candle['close'] - candle['open'])
            
            # ICT Displacement criteria
            body_dominance = (body_size / candle_range) if candle_range > 0 else 0
            range_expansion = candle_range > avg_range * 1.5
            
            # Direction check
            is_bullish_candle = candle['close'] > candle['open']
            is_bearish_candle = candle['close'] < candle['open']
            correct_direction = (bias == 'bullish' and is_bullish_candle) or (bias == 'bearish' and is_bearish_candle)
            
            # Check if this is a displacement candle
            if body_dominance >= 0.6 and range_expansion and correct_direction:
                displacement_candles.append({
                    'index': i,
                    'body_dominance': body_dominance,
                    'range_expansion_ratio': candle_range / avg_range
                })
                
                # ICT MSS REQUIREMENT: BODY must close beyond MSS level
                if bias == 'bullish':
                    # For bullish MSS, the BODY must close above the level
                    body_close = candle['close']
                    body_open = candle['open']
                    if body_close > mss_level and body_open < body_close:  # Bullish candle body above level
                        mss_confirmed = True
                        logger.debug(f"Bullish MSS confirmed: Body close {body_close:.5f} > MSS {mss_level:.5f}")
                else:  # bearish
                    # For bearish MSS, the BODY must close below the level
                    body_close = candle['close']
                    body_open = candle['open']
                    if body_close < mss_level and body_open > body_close:  # Bearish candle body below level
                        mss_confirmed = True
                        logger.debug(f"Bearish MSS confirmed: Body close {body_close:.5f} < MSS {mss_level:.5f}")
        
        # ICT requires consecutive displacement candles (at least 2-3)
        has_consecutive_displacement = False
        if len(displacement_candles) >= 2:
            # Check if we have at least 2 consecutive displacement candles
            for i in range(len(displacement_candles) - 1):
                if displacement_candles[i+1]['index'] - displacement_candles[i]['index'] == 1:
                    has_consecutive_displacement = True
                    logger.debug(f"Found consecutive displacement candles with avg body dominance "
                            f"{np.mean([d['body_dominance'] for d in displacement_candles]):.1%}")
                    break
        
        # Single very strong displacement candle can also be valid
        if not has_consecutive_displacement and displacement_candles:
            strongest = max(displacement_candles, key=lambda x: x['body_dominance'] * x['range_expansion_ratio'])
            if strongest['body_dominance'] >= 0.8 and strongest['range_expansion_ratio'] >= 2.0:
                has_consecutive_displacement = True
                logger.debug(f"Found single strong displacement: {strongest['body_dominance']:.1%} body, "
                        f"{strongest['range_expansion_ratio']:.1f}x range")
        
        # Final confirmation requires BOTH displacement AND MSS
        confirmation_result = has_consecutive_displacement and mss_confirmed
        
        if not has_consecutive_displacement:
            logger.warning("No valid displacement found (need consecutive strong candles or single very strong candle)")
        if not mss_confirmed:
            logger.warning(f"MSS not confirmed (need body close beyond {mss_level:.5f})")
        
        logger.info(f"ICT Confirmation: Displacement={has_consecutive_displacement}, MSS={mss_confirmed}, "
                f"Final={confirmation_result}")
        
        return confirmation_result

class ICTSignalGenerator:
    """
    Generates trading signals following the ICT narrative sequence.
    """
    
    def __init__(self, config):
        self.config = config
        self.analyzer = ICTAnalyzer(config)        
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str, spread: float, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """
        [VERSION 3.0 - CORRECTED SESSION LOGIC]
        Generate trading signal following the ICT narrative.
        This version has corrected logic for NY and London Close killzones
        to properly handle both continuation and reversal profiles.
        """
        analysis = self.analyzer.analyze(ohlc_df, symbol, daily_df, h4_df) 
        if not analysis:
            return None, None, None, None
        
        daily_bias = analysis['daily_bias']
        london_manipulation = analysis['manipulation'] # The potential manipulation from London
        current_price = analysis['current_price']
        session = analysis['session']
        swings = analysis.get('swings', pd.DataFrame())
        require_killzone = getattr(self.config, 'REQUIRE_KILLZONE', True)
        
        narrative = self._build_narrative(analysis)
        
        # === NARRATIVE STEP 1: Daily Bias ===
        if daily_bias == 'neutral':
            logger.debug(f"{symbol}: No clear daily bias from daily chart.")
            return None, None, None, None
        
        # === NARRATIVE STEP 2: Killzone Entry Logic ===
        market_phase = self.analyzer._analyze_market_phase(ohlc_df, london_manipulation, daily_bias)
        logger.info(f"{symbol}: Phase: {market_phase['phase']} | Retracement: {market_phase.get('retracement_percent', 0):.1f}%")
        
        if market_phase['phase'] == 'invalidated':
            return None, None, None, None
        
        entry_signal, sl_price, tp_price = None, None, None
        
        if require_killzone and not session.get('in_killzone'):
            return None, None, None, None

        killzone_name = session.get('killzone_name')
        
        # --- LONDON KILLZONE LOGIC ---
        if killzone_name == 'London':
            if london_manipulation.get('detected'):
                manipulation_type = london_manipulation.get('type', 'manipulation')
                logger.info(f"{symbol}: LondonKZ - Looking for entry after '{manipulation_type}'.")
                entry_signal, sl_price, tp_price = self._setup_judas_entry(
                    daily_bias, current_price, london_manipulation, ohlc_df, analysis
                )
                narrative.entry_model = f"LONDON_{manipulation_type.upper()}"
                    
        # --- NEW YORK KILLZONE LOGIC (Dual Path: Continuation vs. Reversal) ---
        elif killzone_name == 'NewYork':
            
            # PATH A: CONTINUATION MODEL (A manipulation from London already exists)
            if london_manipulation.get('detected'):
                logger.info(f"{symbol}: NYKZ - Activating CONTINUATION model based on prior London manipulation.")
                # Check for OTE, FVG, or OB retracement
                entry_signal, sl_price, tp_price = self._find_continuation_entry(analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation)
                if entry_signal:
                    narrative.entry_model = "NY_CONTINUATION"

            # PATH B: REVERSAL MODEL (No prior manipulation, so we search for one now)
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

        # --- LONDON CLOSE KILLZONE LOGIC (Dual Path: Reversal vs. Continuation) ---
        elif killzone_name == 'LondonClose':
    
            # Check for London Close Reversal pattern
            reversal_signal = self.analyzer._check_london_close_reversal(ohlc_df, session)

            # PATH A: REVERSAL MODEL
            if reversal_signal and reversal_signal.get('detected'):
                logger.info(f"{symbol}: LondonClose REVERSAL signal found ('{reversal_signal.get('type')}')")
                
                # Determine trade direction from signal type
                reversal_bias = 'bearish' if 'bearish' in reversal_signal['type'] else 'bullish'
                
                # We trade with the REVERSAL bias for this setup
                entry_signal, sl_price, tp_price = self._setup_judas_entry(
                    reversal_bias, current_price, reversal_signal, ohlc_df, analysis
                )
                narrative.entry_model = "LONDON_CLOSE_REVERSAL"
                narrative.daily_bias = f"{daily_bias} (Reversal to {reversal_bias})"

            # PATH B: CONTINUATION MODEL
            else:
                logger.info(f"{symbol}: LondonCloseKZ - No reversal detected. Checking for CONTINUATION.")
                
                # If we have a prior manipulation from London, look for continuation
                if london_manipulation.get('detected'):
                    entry_signal, sl_price, tp_price = self._find_continuation_entry(
                        analysis, daily_bias, current_price, ohlc_df, spread, london_manipulation
                    )
                    if entry_signal:
                        narrative.entry_model = "LONDON_CLOSE_CONTINUATION"
                else:
                    # No prior manipulation, check for a new one
                    new_manipulation = self.analyzer._check_manipulation_patterns(ohlc_df, daily_bias, session, swings)
                    
                    if new_manipulation and new_manipulation.get('detected'):
                        logger.info(f"{symbol}: New manipulation detected in London Close")
                        entry_signal, sl_price, tp_price = self._setup_judas_entry(
                            daily_bias, current_price, new_manipulation, ohlc_df, analysis
                        )
                        narrative.entry_model = f"LONDON_CLOSE_{new_manipulation.get('type', 'MANIPULATION').upper()}"

        # === FINAL VALIDATION ===
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
    
    def _find_poi_reaction_index(self, ohlc_df: pd.DataFrame, poi_range: Dict[str, float], lookback: int = 10) -> Optional[int]:
        """Scans backwards to find the integer index of the candle that first touched a POI."""
        if 'top' not in poi_range or 'bottom' not in poi_range:
            return None
            
        # Scan the last `lookback` candles
        for i in range(-1, -(lookback + 1), -1):
            # Ensure index is valid for the dataframe
            if abs(i) >= len(ohlc_df):
                break
                
            candle = ohlc_df.iloc[i]
            
            # Check if the candle's high/low wicks touched or entered the POI range
            if candle['high'] >= poi_range['bottom'] and candle['low'] <= poi_range['top']:
                # Return the integer position for iloc
                return len(ohlc_df) + i
                
        return None

    def _find_continuation_entry(self, analysis, daily_bias, current_price, ohlc_df, spread, manipulation):
        """
        [ENHANCED] Enhanced continuation entry with comprehensive logging.
        """
        swings = analysis.get('swings', pd.DataFrame())
        symbol = analysis.get('symbol', 'UNKNOWN')
        
        logger.info(f"=== {symbol} CONTINUATION ENTRY ANALYSIS ===")
        logger.info(f"Daily Bias: {daily_bias}")
        logger.info(f"Current Price: {current_price:.5f}")
        logger.info(f"Manipulation: {manipulation.get('type', 'None')} at {manipulation.get('level', 0):.5f}")

        # 1. Check for OTE entry
        ote_zones = analysis['ote_zones']
        logger.info(f"OTE Zones Found: {len(ote_zones)}")
        
        ote_zone = next((z for z in ote_zones if z['direction'].lower() == daily_bias), None)
        
        if ote_zone:
            logger.info(f"Matching OTE Zone: {ote_zone}")
            logger.info(f"Price {current_price:.5f} vs OTE High={ote_zone['high']:.5f}, Low={ote_zone['low']:.5f}")
            
            if self._is_price_in_ote(current_price, [ote_zone], daily_bias):
                logger.info("Price is within OTE zone - checking for reaction")
                reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': ote_zone['high'], 'bottom': ote_zone['low']})
                
                if reaction_index is not None:
                    logger.info(f"POI reaction found at index {reaction_index}")
                    if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                        logger.info("ICT confirmation SUCCESS - OTE entry valid")
                        return self._setup_ote_entry(daily_bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis)
                    else:
                        logger.warning("ICT confirmation FAILED - no displacement/MSS")
                else:
                    logger.warning("No POI reaction found in OTE zone")
            else:
                logger.warning(f"Price {current_price:.5f} NOT in OTE zone")
        else:
            logger.warning("No OTE zone found matching daily bias")

        # 2. Check for FVG entry
        logger.info("--- Checking FVG Entry ---")
        fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], daily_bias, current_price, manipulation, ohlc_df, swings)
        
        if fvg_entry:
            logger.info(f"FVG found: {fvg_entry}")
            reaction_index = self._find_poi_reaction_index(ohlc_df, {'top': fvg_entry['top'], 'bottom': fvg_entry['bottom']})

            if reaction_index is not None:
                logger.info(f"FVG reaction found at index {reaction_index}")
                if self.analyzer._confirm_displacement_and_mss(ohlc_df, reaction_index, daily_bias, swings):
                    logger.info("ICT confirmation SUCCESS - FVG entry valid")
                    return self._setup_fvg_entry(daily_bias, current_price, fvg_entry, manipulation, ohlc_df, analysis)
                else:
                    logger.warning("ICT confirmation FAILED for FVG")
            else:
                logger.warning("No POI reaction found for FVG")
        else:
            logger.warning("No suitable FVG found")

        logger.warning(f"{symbol}: No valid continuation entry found")
        return None, None, None
    
    def _setup_fvg_entry(self, bias, current_price, fvg_entry, manipulation, ohlc_df, analysis):
        """Setup entry from a Fair Value Gap."""
        if bias == 'bullish':
            signal = "BUY"
            # SL below the FVG or manipulation low, whichever is lower
            sl_price = min(fvg_entry['bottom'], manipulation['level']) - self._calculate_atr_buffer(ohlc_df)
        else: # bearish
            signal = "SELL"
            # SL above the FVG or manipulation high, whichever is higher
            sl_price = max(fvg_entry['top'], manipulation['level']) + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        return signal, sl_price, tp_price
        
    def _find_retracement_fvg(self, fair_value_gaps, daily_bias, current_price, manipulation, ohlc_df, swings):
        """
        Find FVGs suitable for continuation entries following ICT methodology.
        
        ICT Requirements:
        - Bullish bias: FVG must be in DISCOUNT zone (below equilibrium)
        - Bearish bias: FVG must be in PREMIUM zone (above equilibrium)
        - FVGs formed AFTER manipulation are preferred
        - Price should be APPROACHING the FVG, not already inside it
        - Avoid large FVGs
        - FVGs after liquidity sweeps are highest priority
        """
        if not fair_value_gaps or ohlc_df.empty:
            return None
        
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14).iloc[-1]
        manipulation_index = manipulation.get('index', -1)
        
        # Get premium/discount analysis
        premium_discount = self.analyzer._analyze_premium_discount(ohlc_df, swings=swings)
        equilibrium = premium_discount.get('equilibrium', 0)
        
        if equilibrium == 0:
            logger.warning("Could not determine equilibrium for FVG selection")
            return None
        
        logger.debug(f"FVG Search: Bias={daily_bias}, Current Price={current_price:.5f}, "
                    f"Equilibrium={equilibrium:.5f}, Current Zone={premium_discount.get('current_zone', 'unknown')}")
        
        valid_fvgs = []
        
        for fvg in fair_value_gaps:
            # Skip FVGs that don't match our bias direction
            if (daily_bias == 'bullish' and fvg['type'] != 'bullish') or \
            (daily_bias == 'bearish' and fvg['type'] != 'bearish'):
                continue
            
            fvg_top = fvg['top']
            fvg_bottom = fvg['bottom']
            fvg_midpoint = (fvg_top + fvg_bottom) / 2  # Consequent encroachment
            fvg_size = fvg_top - fvg_bottom
            
            # ICT CRITICAL: Check if FVG is in the correct zone
            if daily_bias == 'bullish':
                # For bullish, FVG must be in DISCOUNT (below equilibrium)
                if fvg_midpoint > equilibrium:
                    logger.debug(f"Skipping FVG at {fvg_midpoint:.5f} - in premium zone (above {equilibrium:.5f})")
                    continue
                    
                # Price should be ABOVE the FVG (approaching from above)
                if current_price < fvg_top:
                    logger.debug(f"Skipping FVG - price {current_price:.5f} already below FVG top {fvg_top:.5f}")
                    continue
                    
            else:  # bearish
                # For bearish, FVG must be in PREMIUM (above equilibrium)
                if fvg_midpoint < equilibrium:
                    logger.debug(f"Skipping FVG at {fvg_midpoint:.5f} - in discount zone (below {equilibrium:.5f})")
                    continue
                    
                # Price should be BELOW the FVG (approaching from below)
                if current_price > fvg_bottom:
                    logger.debug(f"Skipping FVG - price {current_price:.5f} already above FVG bottom {fvg_bottom:.5f}")
                    continue
            
            # Skip very large FVGs (more than 2x ATR)
            if fvg_size > atr * 2:
                logger.debug(f"Skipping large FVG - size {fvg_size:.5f} > 2x ATR {atr*2:.5f}")
                continue
            
            # Calculate distance to FVG
            if daily_bias == 'bullish':
                distance_to_fvg = current_price - fvg_top
            else:
                distance_to_fvg = fvg_bottom - current_price
            
            # FVG should be reasonably close (within 3x ATR)
            if distance_to_fvg > atr * 3:
                logger.debug(f"Skipping distant FVG - distance {distance_to_fvg:.5f} > 3x ATR")
                continue
            
            # Calculate priority score
            priority_score = 0
            
            # Priority 1: FVG formed after manipulation
            if fvg.get('index', 0) > manipulation_index:
                priority_score += 3
                logger.debug(f"FVG at index {fvg.get('index')} is post-manipulation (+3 priority)")
            
            # Priority 2: Smaller FVGs get higher priority
            size_score = 1 - (fvg_size / (atr * 2))  # Normalized 0-1
            priority_score += size_score * 2
            
            # Priority 3: Closer FVGs get higher priority
            distance_score = 1 - (distance_to_fvg / (atr * 3))  # Normalized 0-1
            priority_score += distance_score
            
            # Priority 4: Deep discount/premium positioning
            if daily_bias == 'bullish':
                # Further below equilibrium is better
                zone_score = (equilibrium - fvg_midpoint) / equilibrium
            else:
                # Further above equilibrium is better
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
            # Sort by priority score (highest first)
            valid_fvgs.sort(key=lambda x: x['priority_score'], reverse=True)
            best_fvg = valid_fvgs[0]
            
            logger.info(f"Selected FVG: {best_fvg['fvg']['type']} in {best_fvg['zone_position']} zone, "
                    f"score={best_fvg['priority_score']:.2f}, distance={best_fvg['distance']:.5f}, "
                    f"size={best_fvg['size']:.5f}")
            
            # Return the FVG with consequent encroachment info
            result = best_fvg['fvg'].copy()
            result['consequent_encroachment'] = best_fvg['midpoint']
            return result
        
        logger.debug("No valid FVGs found matching ICT criteria")
        return None
    
    def _is_safe_entry_location(self, ohlc_df, entry_price, sl_price, daily_bias):
        """
        Intelligently handles trend-following entries
        during shallow pullbacks.
        """
        # Calculate ATR for volatility context
        atr = self._calculate_atr_buffer(ohlc_df)
        risk = abs(entry_price - sl_price)

        # Basic check: Stop loss should never be excessively large
        if risk == 0:
            logger.warning("Risk is zero, trade is invalid.")
            return False

        # Get key short-term and long-term price levels
        recent_high = ohlc_df['high'].tail(20).max()
        recent_low = ohlc_df['low'].tail(20).min()
        broader_high = ohlc_df['high'].tail(50).max()
        broader_low = ohlc_df['low'].tail(50).min()

        # Define a flag to check if we are in a strong trend continuation scenario
        is_strong_trend_continuation = False
        
        if daily_bias == 'bullish':
            # A shallow pullback near the highs is a sign of a strong trend
            distance_from_high = broader_high - entry_price
            if distance_from_high < atr: # Entering within one ATR of the 50-period high
                logger.info(f"Entry is near 50-period highs; considering this a strong trend continuation setup.")
                is_strong_trend_continuation = True
        else:  # bearish
            # A shallow pullback near the lows is a sign of a strong trend
            distance_from_low = entry_price - broader_low
            if distance_from_low < atr: # Entering within one ATR of the 50-period low
                logger.info(f"Entry is near 50-period lows; considering this a strong trend continuation setup.")
                is_strong_trend_continuation = True
        
        # If it's NOT a strong trend continuation, we apply the stricter reward check.
        if not is_strong_trend_continuation:
            # For standard entries, ensure there's enough room to a potential target.
            potential_reward = 0
            if daily_bias == 'bullish':
                # Target is at least the recent high
                potential_reward = recent_high - entry_price
            else: # bearish
                # Target is at least the recent low
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
        
        # Get most recent structure break
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
                    # For bullish, OTE is a buy zone
                    if ote['low'] <= current_price <= ote['high']:
                        return True
                else:
                    # For bearish, OTE is a sell zone
                    if ote['low'] <= current_price <= ote['high']:
                        return True
        return False
    
    def _setup_ote_entry(self, bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis):
        """Setup entry from OTE zone."""
        # Find the relevant OTE zone
        ote = None
        for zone in ote_zones:
            if zone['direction'].lower() == bias:
                ote = zone
                break
        
        if not ote:
            return None, None, None
        
        if bias == 'bullish':
            signal = "BUY"
            # SL below the swing low or manipulation low
            sl_price = min(ote['swing_low'], manipulation.get('level', ote['swing_low']))
            sl_price -= self._calculate_atr_buffer(ohlc_df)
            
        else:  # bearish
            signal = "SELL"
            # SL above the swing high or manipulation high
            sl_price = max(ote['swing_high'], manipulation.get('level', ote['swing_high']))
            sl_price += self._calculate_atr_buffer(ohlc_df)
        
        # Calculate target
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
            # SL below the Judas sweep low
            sl_price = manipulation.get('level', current_price) - self._calculate_atr_buffer(ohlc_df)
        else:  # bearish
            signal = "SELL"
            # SL above the Judas sweep high
            sl_price = manipulation.get('level', current_price) + self._calculate_atr_buffer(ohlc_df)
        
        # Target the opposite side of Asian range or liquidity
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        
        return signal, sl_price, tp_price
    
    def _calculate_atr_buffer(self, ohlc_df):
        """Calculate ATR-based buffer for stop loss."""
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        return atr.iloc[-1] * self.config.SL_ATR_MULTIPLIER
    
    def _calculate_target(self, entry_price, sl_price, bias, analysis):
        """
        [VERSION 2.0 - HIERARCHICAL TARGETING]
        Calculate take profit by targeting liquidity in a hierarchical manner.
        1. Prioritize "engineered liquidity" (equal highs/lows).
        2. If none, fall back to the nearest single swing high/low.
        3. If no clear swings, fall back to a fixed R:R.
        """
        risk = abs(entry_price - sl_price)
        liquidity_zones = analysis.get('liquidity_zones', []) # From smc.liquidity (equal H/L)
        swings = analysis.get('swings', pd.DataFrame()).dropna()

        target_level = None

        # --- Tier 1: Look for Engineered Liquidity (Equal Highs/Lows) ---
        if liquidity_zones:
            if bias == 'bullish':
                potential_targets = [
                    liq['level'] for liq in liquidity_zones 
                    if liq['type'] == 'bearish' and liq['level'] > entry_price
                ]
                if potential_targets:
                    target_level = min(potential_targets)
                    logger.debug(f"Tier 1 Target: Found engineered liquidity (equal highs) at {target_level}")
            else: # bearish
                potential_targets = [
                    liq['level'] for liq in liquidity_zones 
                    if liq['type'] == 'bullish' and liq['level'] < entry_price
                ]
                if potential_targets:
                    target_level = max(potential_targets)
                    logger.debug(f"Tier 1 Target: Found engineered liquidity (equal lows) at {target_level}")

        # --- Tier 2: If no engineered liquidity, find nearest single Swing Point ---
        if target_level is None and not swings.empty:
            if bias == 'bullish':
                # Target the nearest swing HIGH above the entry
                swing_highs = swings[swings['HighLow'] == 1]
                potential_targets = swing_highs[swing_highs['Level'] > entry_price]
                if not potential_targets.empty:
                    target_level = potential_targets['Level'].min()
                    logger.debug(f"Tier 2 Target: No engineered liquidity found. Targeting nearest swing high at {target_level}")
            else: # bearish
                # Target the nearest swing LOW below the entry
                swing_lows = swings[swings['HighLow'] == -1]
                potential_targets = swing_lows[swing_lows['Level'] < entry_price]
                if not potential_targets.empty:
                    target_level = potential_targets['Level'].max()
                    logger.debug(f"Tier 2 Target: No engineered liquidity found. Targeting nearest swing low at {target_level}")

        # --- Final Validation and Fallback ---
        if target_level:
            potential_reward = abs(target_level - entry_price)
            min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)
            
            if potential_reward >= risk * min_rr:
                logger.info(f"Using liquidity target at {target_level:.5f} for TP.")
                return target_level
            else:
                logger.debug(f"Liquidity target at {target_level:.5f} is too close for minimum R:R ({potential_reward/risk:.1f}R). Using fallback.")

        # --- Tier 3: Fallback to fixed R:R if no suitable liquidity target is found ---
        tp_rr_ratio = getattr(self.config, 'TP_RR_RATIO', 1.5)
        logger.debug(f"Tier 3 Target: Using fixed R:R fallback ({tp_rr_ratio}:1)")
        if bias == 'bullish':
            return entry_price + (risk * tp_rr_ratio)
        else:
            return entry_price - (risk * tp_rr_ratio)
    
    def _validate_levels(self, signal, entry, sl, tp):
        """Validate entry, SL, and TP levels."""
        if signal == "BUY":
            return sl < entry < tp
        else:  # SELL
            return tp < entry < sl