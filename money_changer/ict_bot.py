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
    daily_bias: str  # 'bullish' or 'bearish'
    po3_phase: str  # 'accumulation', 'manipulation', 'distribution'
    manipulation_confirmed: bool  # Judas swing or liquidity sweep
    manipulation_level: float  # Price level of manipulation
    structure_broken: bool  # BOS/CHoCH confirmed
    structure_level: float  # Level that was broken
    in_killzone: bool
    killzone_name: str
    ote_zone: Dict[str, float]  # {'high': x, 'sweet': x, 'low': x}
    fvg_zone: Dict[str, float] # The FVG that confirms the MSS
    order_blocks: List[Dict]  # Refined OBs in play
    current_price: float
    entry_model: str  # Which ICT model is being used

class ICTAnalyzer:
    """Analyzes price action through the ICT lens."""
    
    def __init__(self, config):
        self.config = config
        self.swing_lookback = config.SWING_LOOKBACK
        self.structure_lookback = config.STRUCTURE_LOOKBACK
        self._last_analysis_cache = {}
        self._cache_max_size = 50
        self.liquidity_detector = LiquidityDetector()
        
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
        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_context, symbol, daily_df)

        manipulation = po3_analysis.get('manipulation', {'detected': False})
        
        # Step 3: Analyze the rest of the market concepts using the robust bias
        structure, fvg_zone = self._get_structure(ohlc_df, swings, session_context, daily_bias, manipulation)
        order_blocks = self._get_order_blocks(ohlc_df, swings)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        liquidity_levels = self.liquidity_detector.get_liquidity_levels(ohlc_df, session_context, daily_df)
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, swings, daily_bias, manipulation, fvg_zone)
        htf_levels = self._get_htf_levels(h4_df) 
        
        analysis_result = {
            'symbol': symbol, 'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1], 'swings': swings, 'structure': structure,
            'daily_bias': daily_bias, 'po3_analysis': po3_analysis, 'manipulation': manipulation,
            'order_blocks': order_blocks, 'fair_value_gaps': fair_value_gaps,
            'liquidity_levels': liquidity_levels, 'premium_discount': pd_analysis,
            'ote_zones': ote_zones, 'fvg_zone': fvg_zone, 'session': session_context, 'htf_levels': htf_levels
        }
        
        # Cache the analysis
        self._last_analysis_cache[symbol] = analysis_result
        if len(self._last_analysis_cache) > self._cache_max_size:
            oldest_symbol = min(self._last_analysis_cache.keys(), 
                            key=lambda k: self._last_analysis_cache[k].get('timestamp', 0))
            del self._last_analysis_cache[oldest_symbol]
        
        return analysis_result

    def _get_htf_levels(self, h4_df):
        """Get higher timeframe levels using REAL H4 data from broker."""
        if h4_df is None or h4_df.empty or len(h4_df) < 2:
            logger.warning("Insufficient H4 data for HTF levels")
            return {}
        
        try:
            # Use the REAL H4 data instead of resampled M15
            htf_levels = smc.previous_high_low(h4_df, time_frame="4h")
            
            if htf_levels is not None and not htf_levels.empty:
                return {
                    'h4_high': htf_levels['PreviousHigh'].iloc[-1],
                    'h4_low': htf_levels['PreviousLow'].iloc[-1],
                    'h4_broken_high': htf_levels['BrokenHigh'].iloc[-1] == 1,
                    'h4_broken_low': htf_levels['BrokenLow'].iloc[-1] == 1
                }
        except Exception as e:
            logger.warning(f"Error getting HTF levels from real H4 data: {e}")
        
        return {}
    
    def _get_swings(self, ohlc_df):
        """Identify swing highs and lows."""
        try:
            swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
            return swings
        except Exception as e:
            logger.error(f"Error getting swings: {e}")
            return pd.DataFrame()
    
    def _get_structure(self, ohlc_df, swings, session_context, daily_bias, manipulation):
        """Analyze ICT market structure using framework context."""
        if swings.empty:
            return pd.DataFrame(), None
            
        try:
            structure, fvg_zone = self._get_ict_structure(ohlc_df, swings, session_context, daily_bias, manipulation)
            return structure, fvg_zone
        except Exception as e:
            logger.error(f"Error getting ICT structure: {e}")
            return pd.DataFrame(), None
        
    def _get_session_context(self, ohlc_df: pd.DataFrame) -> dict:
        """
        Get proper ICT session context, now including London session range for NY analysis.
        """
        context = {
            'last_asian_range': None,
            'last_london_range': None,
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

            # --- Asian Range (always based on previous trading day for London/NY) ---
            if current_ny_hour < 19:
                asian_date = current_ny_date - pd.Timedelta(days=1)
            else:
                asian_date = current_ny_date
            
            asian_start_ny = latest_ny_time.replace(year=asian_date.year, month=asian_date.month, day=asian_date.day, hour=19, minute=0, second=0, microsecond=0)
            asian_end_ny = asian_start_ny.replace(hour=22)
            asian_start_utc = asian_start_ny.astimezone(pytz.UTC)
            asian_end_utc = asian_end_ny.astimezone(pytz.UTC)
            asian_data = ohlc_df[(ohlc_df.index >= asian_start_utc) & (ohlc_df.index < asian_end_utc)]
            
            if not asian_data.empty:
                context['last_asian_range'] = {
                    'start_time_utc': asian_data.index[0], 'end_time_utc': asian_end_utc,
                    'high': asian_data['high'].max(), 'low': asian_data['low'].min()
                }

            # --- London Range (for NY session analysis) ---
            # London session is 2 AM to 5 AM NY time on the CURRENT day
            london_start_ny = latest_ny_time.replace(hour=2, minute=0, second=0, microsecond=0)
            london_end_ny = latest_ny_time.replace(hour=5, minute=0, second=0, microsecond=0)
            london_start_utc = london_start_ny.astimezone(pytz.UTC)
            london_end_utc = london_end_ny.astimezone(pytz.UTC)
            london_data = ohlc_df[(ohlc_df.index >= london_start_utc) & (ohlc_df.index < london_end_utc)]

            if not london_data.empty:
                context['last_london_range'] = {
                    'start_time_utc': london_data.index[0], 'end_time_utc': london_end_utc,
                    'high': london_data['high'].max(), 'low': london_data['low'].min()
                }

            # --- Kill Zone Detection ---
            if 2 <= current_ny_hour < 5:
                context['in_killzone'] = True
                context['killzone_name'] = 'London'
            elif (current_ny_hour == 8 and current_ny_minute >= 30) or (9 <= current_ny_hour < 11):
                context['in_killzone'] = True
                context['killzone_name'] = 'NewYork'
            elif 15 <= current_ny_hour < 17:
                context['in_killzone'] = True
                context['killzone_name'] = 'LondonClose'
                
        except Exception as e:
            context['error'] = f"Error calculating session context: {e}"
            logger.error(context['error'], exc_info=True)
                        
        return context
    
    def _determine_daily_bias(self, ohlc_df: pd.DataFrame, symbol: str, daily_df: pd.DataFrame = None) -> Tuple[str, Dict]:
        """
        [REFINED] Determines bias using actual daily timeframe data for all components.
        """
        logger.debug(f"\n--- {symbol} ICT BIAS CHECKLIST ---")

        if daily_df is None or daily_df.empty or len(daily_df) < 20:
            logger.warning(f"{symbol}: Insufficient daily data for bias determination. Bias will be neutral.")
            return 'neutral', {}

        # 1. Gather all evidence from the DAILY chart
        htf_order_flow = self._analyze_daily_order_flow(daily_df)
        # Use daily data to find the draw on weekly liquidity
        liquidity_draw = self._analyze_liquidity_draw(daily_df, is_daily_context=True)
        # Use daily swings to find the daily premium/discount zone
        pd_analysis = self._analyze_premium_discount(daily_df, is_daily_context=True)
        
        is_in_discount = 'discount' in pd_analysis.get('current_zone', '')
        is_in_premium = 'premium' in pd_analysis.get('current_zone', '')
        
        final_bias = 'neutral'
        reasons = []

        # 2. Apply ICT rules with hierarchy (most important checks first)
        if htf_order_flow == 'bullish':
            reasons.append("Primary Factor: Daily Order Flow is Bullish (HH+HL)")
            final_bias = 'bullish'
            if is_in_discount:
                reasons.append("Supporting Factor: Price is in a Daily Discount zone")
            if liquidity_draw == 'bullish':
                reasons.append("Supporting Factor: Draw on Liquidity is towards Weekly Buyside")

        elif htf_order_flow == 'bearish':
            reasons.append("Primary Factor: Daily Order Flow is Bearish (LH+LL)")
            final_bias = 'bearish'
            if is_in_premium:
                reasons.append("Supporting Factor: Price is in a Daily Premium zone")
            if liquidity_draw == 'bearish':
                reasons.append("Supporting Factor: Draw on Liquidity is towards Weekly Sellside")
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

    def _analyze_session_po3(self, ohlc_df: pd.DataFrame, session_context: dict, symbol: str, daily_df: pd.DataFrame = None) -> Tuple[str, Dict]:
        """
        Orchestrates bias analysis and identifies PO3 phase based on session context and manipulation patterns.
        Now properly uses daily data for bias determination and session-based PO3 phase detection.
        """
        daily_bias, bias_details = self._determine_daily_bias(ohlc_df, symbol, daily_df)
        manipulation_details = self._check_manipulation_patterns(ohlc_df, daily_bias, session_context)

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
        
        logger.debug(f"{symbol}: PO3 Phase: {po3_analysis['current_phase']} | Session: {po3_analysis.get('session', 'Unknown')} | Bias: {daily_bias}")
        
        return daily_bias, po3_analysis
            
    def _check_market_structure_shift(self, ohlc_df, sweep_index, direction, swept_level):
        """
        [REFINED & FIXED] Confirms a Market Structure Shift (MSS) with DISPLACEMENT.
        An MSS is valid if the break of structure is energetic, confirmed by either:
        1. Leaving behind a Fair Value Gap (FVG).
        2. A strong, high-range candle breaking the level.
        """
        try:
            # 1. Find the leg that broke the structure point
            post_sweep_df = ohlc_df.iloc[sweep_index:]
            breaking_candle_idx = -1
            breaking_candle = None

            for i in range(len(post_sweep_df)):
                candle = post_sweep_df.iloc[i]
                if (direction == 'bearish' and candle['close'] < swept_level) or \
                   (direction == 'bullish' and candle['close'] > swept_level):
                    breaking_candle_idx = sweep_index + i
                    breaking_candle = candle
                    break
            
            if breaking_candle_idx == -1:
                return False, None, None # Structure not broken

            # 2. Check for DISPLACEMENT via FVG (Primary confirmation)
            fvg_check_start = max(0, breaking_candle_idx - 2)
            fvg_check_end = min(len(ohlc_df), breaking_candle_idx + 2)
            fvg_df = ohlc_df.iloc[fvg_check_start:fvg_check_end]
            
            fvgs = self._get_fvgs(fvg_df)
            fvg_found = None
            for fvg in fvgs:
                if (direction == 'bullish' and fvg['type'] == 'bullish') or \
                   (direction == 'bearish' and fvg['type'] == 'bearish'):
                    fvg_found = fvg
                    break

            if fvg_found:
                return True, swept_level, fvg_found

            # 3. If no FVG, check for DISPLACEMENT via candle force (Secondary confirmation)
            displacement_window = ohlc_df.iloc[max(0, breaking_candle_idx - 2) : breaking_candle_idx + 1]
            if len(displacement_window) > 1:
                avg_range = (displacement_window['high'][:-1] - displacement_window['low'][:-1]).mean()
                breaking_range = breaking_candle['high'] - breaking_candle['low']

                if breaking_range > avg_range * 1.5: # Energetic break
                    candle_range = breaking_range
                    if candle_range > 0:
                        if direction == 'bullish':
                            close_position = (breaking_candle['close'] - breaking_candle['low']) / candle_range
                            if close_position >= 0.75 and breaking_candle['close'] > breaking_candle['open']:
                                return True, swept_level, None # No FVG, but displacement confirmed
                        else: # bearish
                            close_position = (breaking_candle['close'] - breaking_candle['low']) / candle_range
                            if close_position <= 0.25 and breaking_candle['close'] < breaking_candle['open']:
                                return True, swept_level, None # No FVG, but displacement confirmed

            return False, None, None

        except Exception as e:
            logger.error(f"Error checking MSS: {e}", exc_info=True)
            return False, None, None
        
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
            return 'bullish'
        # Lower high and lower low = bearish
        elif last_two_highs[1] < last_two_highs[0] and last_two_lows[1] < last_two_lows[0]:
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
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context):
        """
        [CORRECTED & COMPLETE V3] Checks for a sequence of manipulation patterns (Judas, Turtle Soup, etc.)
        and returns the first one found that aligns with the established daily bias.
        Fixes the MSS check to look for a break of the correct internal structure that *precedes* the sweep.
        """
        if daily_bias == 'neutral':
            return None

        # PATTERN 1: Judas Swing (Time-specific, highest priority during London KZ)
        if session_context.get('in_killzone') and session_context['killzone_name'] == 'London':
            if session_context.get('last_asian_range'):
                asian_high = session_context['last_asian_range']['high']
                asian_low = session_context['last_asian_range']['low']
                search_window = ohlc_df[ohlc_df.index >= session_context['last_asian_range']['end_time_utc']]

                if daily_bias == 'bearish':
                    sweep_up = search_window[search_window['high'] > asian_high]
                    if not sweep_up.empty:
                        sweep_time = sweep_up.index[0]
                        sweep_index = ohlc_df.index.get_loc(sweep_time)
                        
                        # Find the last swing LOW *before* the sweep to be broken for a bearish MSS
                        swings = self._get_swings(ohlc_df)
                        candidate_lows = swings[(swings['HighLow'] == -1) & (swings.index < sweep_time)].dropna()
                        if not candidate_lows.empty:
                            mss_target_level = candidate_lows['Level'].iloc[-1]
                            mss_confirmed, mss_level, _ = self._check_market_structure_shift(ohlc_df, sweep_index, 'bearish', mss_target_level)
                            if mss_confirmed:
                                return {'type': 'bearish_judas', 'level': sweep_up['high'].iloc[0], 'index': sweep_index, 'mss_level': mss_level}
                
                elif daily_bias == 'bullish':
                    sweep_down = search_window[search_window['low'] < asian_low]
                    if not sweep_down.empty:
                        sweep_time = sweep_down.index[0]
                        sweep_index = ohlc_df.index.get_loc(sweep_time)

                        # Find the last swing HIGH *before* the sweep to be broken for a bullish MSS
                        swings = self._get_swings(ohlc_df)
                        candidate_highs = swings[(swings['HighLow'] == 1) & (swings.index < sweep_time)].dropna()
                        if not candidate_highs.empty:
                            mss_target_level = candidate_highs['Level'].iloc[-1]
                            mss_confirmed, mss_level, _ = self._check_market_structure_shift(ohlc_df, sweep_index, 'bullish', mss_target_level)
                            if mss_confirmed:
                                return {'type': 'bullish_judas', 'level': sweep_down['low'].iloc[0], 'index': sweep_index, 'mss_level': mss_level}

        # PATTERN 2: Turtle Soup (General price action pattern)
        turtle_soup = self._check_turtle_soup_pattern(ohlc_df, daily_bias)
        if turtle_soup:
            return turtle_soup

        # PATTERN 3: General Liquidity Sweep (Lowest priority)
        liquidity_sweep = self._check_liquidity_sweep(ohlc_df, daily_bias)
        if liquidity_sweep:
            return liquidity_sweep

        return None

    def _check_turtle_soup_pattern(self, ohlc_df, daily_bias):
        """
        [CORRECTED] Identifies a Turtle Soup pattern, which is a false breakout of a
        clear 20-period high/low followed by a strong reversal close.
        """
        if len(ohlc_df) < 21:  # Need 20 for the lookback + 1 for the current
            return None
        
        # 1. Identify a clear, recent high/low (the liquidity pool)
        lookback_df = ohlc_df.iloc[-21:-1]
        recent_high = lookback_df['high'].max()
        recent_low = lookback_df['low'].min()
        
        bar = ohlc_df.iloc[-1]
        prev_bar = ohlc_df.iloc[-2]

        # 2. Check for the sweep and a strong, confirming close
        if daily_bias == 'bearish' and bar['high'] > recent_high:
            # Confirmation: The close must be strong and bearish, ideally below the open.
            if bar['close'] < bar['open'] and bar['close'] < recent_high:
                return {'type': 'bearish_turtle_soup', 'level': bar['high'], 'swept_level': recent_high, 'index': len(ohlc_df) - 1}
        
        elif daily_bias == 'bullish' and bar['low'] < recent_low:
            # Confirmation: The close must be strong and bullish, ideally above the open.
            if bar['close'] > bar['open'] and bar['close'] > recent_low:
                return {'type': 'bullish_turtle_soup', 'level': bar['low'], 'swept_level': recent_low, 'index': len(ohlc_df) - 1}
        
        return None

    def _check_liquidity_sweep(self, ohlc_df, daily_bias):
        """
        [CORRECTED] Identifies the core ICT manipulation model: a sweep of a recent
        swing point that is IMMEDIATELY followed by a Market Structure Shift (MSS).
        """
        if len(ohlc_df) < 10:
            return None
        
        swings = smc.swing_highs_lows(ohlc_df, swing_length=5) # Use shorter length for recent swings
        if swings.empty:
            return None
        
        recent_swings = swings.dropna().tail(5)
        
        for i in range(-5, 0): # Check the last 5 candles for a potential sweep
            sweep_candle = ohlc_df.iloc[i]
            sweep_index = len(ohlc_df) + i

            for _, swing in recent_swings.iterrows():
                # --- Check for a Bearish Sweep + MSS ---
                if daily_bias == 'bearish' and swing['HighLow'] == 1: # Target swing highs
                    swept_level = swing['Level']
                    if sweep_candle['high'] > swept_level:
                        # After sweeping a high, the MSS is a break of a swing LOW
                        pre_sweep_df = ohlc_df.iloc[:sweep_index]
                        pre_sweep_swings = smc.swing_highs_lows(pre_sweep_df, swing_length=5)
                        swing_lows = pre_sweep_swings[pre_sweep_swings['HighLow'] == -1].dropna()
                        
                        if not swing_lows.empty:
                            mss_target_level = swing_lows['Level'].iloc[-1]
                            mss_confirmed, mss_level, _ = self._check_market_structure_shift(
                                ohlc_df, sweep_index, 'bearish', mss_target_level
                            )
                            if mss_confirmed:
                                return {'type': 'bearish_liquidity_sweep_mss', 'level': sweep_candle['high'], 'swept_level': swept_level, 'index': sweep_index, 'mss_level': mss_level}

                # --- Check for a Bullish Sweep + MSS ---
                elif daily_bias == 'bullish' and swing['HighLow'] == -1: # Target swing lows
                    swept_level = swing['Level']
                    if sweep_candle['low'] < swept_level:
                        # After sweeping a low, the MSS is a break of a swing HIGH
                        pre_sweep_df = ohlc_df.iloc[:sweep_index]
                        pre_sweep_swings = smc.swing_highs_lows(pre_sweep_df, swing_length=5)
                        swing_highs = pre_sweep_swings[pre_sweep_swings['HighLow'] == 1].dropna()

                        if not swing_highs.empty:
                            mss_target_level = swing_highs['Level'].iloc[-1]
                            mss_confirmed, mss_level, _ = self._check_market_structure_shift(
                                ohlc_df, sweep_index, 'bullish', mss_target_level
                            )
                            if mss_confirmed:
                                return {'type': 'bullish_liquidity_sweep_mss', 'level': sweep_candle['low'], 'swept_level': swept_level, 'index': sweep_index, 'mss_level': mss_level}
        return None
    
    def _get_ict_structure(self, ohlc_df, swings, session_context, daily_bias, manipulation):
        """
        ICT-aligned BOS/CHoCH detection using existing framework context.
        Uses Asian range as external liquidity and displacement validation.
        """
        
        n = len(ohlc_df)
        bos = np.full(n, np.nan)
        choch = np.full(n, np.nan)
        level = np.full(n, np.nan)
        broken_index = np.full(n, np.nan)
        
        # Get external range context (Asian session = major liquidity)
        if not session_context['last_asian_range']:
            return self._create_empty_structure(n), None
        
        asian_range = session_context['last_asian_range']
        external_high = asian_range['high']
        external_low = asian_range['low']
        
        # Get internal swings (within Asian range)
        internal_swings = swings.dropna()
        if internal_swings.empty:
            return self._create_empty_structure(n), None
        
        # Track key internal levels based on bias direction
        key_internal_high = None
        key_internal_low = None
        internal_high_idx = None
        internal_low_idx = None
        
        # Identify the key internal levels to watch
        manipulation_index = manipulation.get('index', -1)
        manipulation_time = ohlc_df.index[manipulation_index] if manipulation_index != -1 else pd.Timestamp(0, tz='UTC')

        if daily_bias == 'bullish':
            # In bullish bias, watch for break of internal low (CHoCH signal)
            post_manipulation_swings = internal_swings[internal_swings.index > manipulation_time]
            internal_lows = post_manipulation_swings[post_manipulation_swings['HighLow'] == -1]
            
            if not internal_lows.empty:
                # Get the most recent internal low
                key_internal_low = internal_lows['Level'].iloc[-1]
                internal_low_idx = internal_lows.index[-1]
        
        elif daily_bias == 'bearish':
            # In bearish bias, watch for break of internal high (CHoCH signal)
            post_manipulation_swings = internal_swings[internal_swings.index > manipulation_time]
            internal_highs = post_manipulation_swings[post_manipulation_swings['HighLow'] == 1]
            
            if not internal_highs.empty:
                # Get the most recent internal high
                key_internal_high = internal_highs['Level'].iloc[-1]
                internal_high_idx = internal_highs.index[-1]
        
        fvg_zone = None
        # Process each candle for structure breaks
        for i in range(manipulation_index + 1, n):  # Only check after manipulation
            current_close = ohlc_df['close'].iloc[i]
            
            # Check for EXTERNAL range breaks (major BOS)
            if daily_bias == 'bullish' and current_close > external_high:
                # Bullish BOS - break of external high WITH trend
                mss_confirmed, mss_level, fvg = self._check_market_structure_shift(ohlc_df, i, 'bullish', external_high)
                if mss_confirmed:
                    # Find index where external high was set (start of Asian session)
                    asian_start_idx = ohlc_df.index.get_loc(asian_range['start_time_utc'])
                    bos[asian_start_idx] = 1
                    level[asian_start_idx] = external_high
                    broken_index[asian_start_idx] = i
                    fvg_zone = fvg
            
            elif daily_bias == 'bearish' and current_close < external_low:
                # Bearish BOS - break of external low WITH trend  
                mss_confirmed, mss_level, fvg = self._check_market_structure_shift(ohlc_df, i, 'bearish', external_low)
                if mss_confirmed:  
                    asian_start_idx = ohlc_df.index.get_loc(asian_range['start_time_utc'])
                    bos[asian_start_idx] = -1
                    level[asian_start_idx] = external_low
                    broken_index[asian_start_idx] = i
                    fvg_zone = fvg
            
            # Check for INTERNAL structure breaks (CHoCH signals)
            if daily_bias == 'bullish' and key_internal_low is not None:
                mss_confirmed, mss_level, fvg = self._check_market_structure_shift(ohlc_df, i, 'bearish', key_internal_low)
                if mss_confirmed:
                    choch[internal_low_idx] = -1
                    level[internal_low_idx] = key_internal_low
                    broken_index[internal_low_idx] = i
                    fvg_zone = fvg
                    # This internal level is now broken
                    key_internal_low = None
            
            elif daily_bias == 'bearish' and key_internal_high is not None:
                mss_confirmed, mss_level, fvg = self._check_market_structure_shift(ohlc_df, i, 'bullish', key_internal_high)
                if mss_confirmed:
                    choch[internal_high_idx] = 1
                    level[internal_high_idx] = key_internal_high
                    broken_index[internal_high_idx] = i
                    fvg_zone = fvg
                    # This internal level is now broken
                    key_internal_high = None
        
        return pd.DataFrame({
            'BOS': bos,
            'CHOCH': choch,
            'Level': level,
            'BrokenIndex': broken_index
        }), fvg_zone

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
    
    
    
    def _calculate_ote_zones(self, ohlc_df, swings, daily_bias, manipulation, fvg_zone):
        """
        [REFINED] Calculate OTE zones from the true displacement leg.
        The leg is measured from the manipulation point to the extreme of the
        move that created the FVG after the Market Structure Shift.
        """
        if not manipulation.get('detected') or not fvg_zone:
            return []

        manipulation_level = manipulation['level']
        
        # The displacement leg starts at the manipulation level
        start_of_leg = manipulation_level

        # The end of the leg is the extreme of the candle that created the FVG
        fvg_candle_index = fvg_zone['index']
        fvg_candle = ohlc_df.iloc[fvg_candle_index]

        if daily_bias == 'bullish':
            # After a bullish MSS, the displacement leg is from the low of manipulation
            # up to the high of the candle that formed the bullish FVG.
            end_of_leg = fvg_candle['high']
            if end_of_leg < start_of_leg: return [] # Invalid range
            range_size = end_of_leg - start_of_leg
            return [{
                'direction': 'bullish',
                'high': end_of_leg - (range_size * 0.62),
                'sweet': end_of_leg - (range_size * 0.705),
                'low': end_of_leg - (range_size * 0.79),
                'swing_high': end_of_leg,
                'swing_low': start_of_leg
            }]

        elif daily_bias == 'bearish':
            # After a bearish MSS, the displacement leg is from the high of manipulation
            # down to the low of the candle that formed the bearish FVG.
            end_of_leg = fvg_candle['low']
            if end_of_leg > start_of_leg: return [] # Invalid range
            range_size = start_of_leg - end_of_leg
            return [{
                'direction': 'bearish',
                'high': end_of_leg + (range_size * 0.79),
                'sweet': end_of_leg + (range_size * 0.705),
                'low': end_of_leg + (range_size * 0.62),
                'swing_high': start_of_leg,
                'swing_low': end_of_leg
            }]
        
        return []
        
    

    
        
    def _check_rejection_confirmation(self, ohlc_df, level, bias, lookback=3):
        """
        Enhanced confirmation check with more realistic criteria
        """
        if not self.config.REQUIRE_ENTRY_CONFIRMATION:
            return True
            
        recent_candles = ohlc_df.tail(lookback)
                
        confirmations = []
        
        for idx, candle in recent_candles.iterrows():
            candle_range = candle['high'] - candle['low']
            body_size = abs(candle['close'] - candle['open'])
            
            if candle_range == 0:
                continue
                
            wick_ratio = body_size / candle_range
            
            if bias == 'bullish':
                # Check for bullish rejection
                if candle['close'] > candle['open']:  # Bullish candle
                    lower_wick = candle['open'] - candle['low']
                else:  # Bearish candle  
                    lower_wick = candle['close'] - candle['low']
                
                # RELAXED: Pin bar (from 1.5x to 1.2x)
                if lower_wick > body_size * 1.2 and wick_ratio < 0.4:
                    confirmations.append('pin_bar')
                
                # Strong bullish close near highs
                if candle['close'] > candle['open']:
                    close_position = (candle['close'] - candle['low']) / candle_range if candle_range > 0 else 0
                    if close_position > 0.8:  # Closed in top 20% of range
                        confirmations.append('strong_close')
                
                # EXISTING: Bullish engulfing
                if idx > recent_candles.index[0]:
                    prev_candle = ohlc_df.loc[ohlc_df.index[ohlc_df.index.get_loc(idx) - 1]]
                    if (candle['close'] > candle['open'] and 
                        prev_candle['close'] < prev_candle['open'] and
                        candle['close'] > prev_candle['open'] and 
                        candle['open'] < prev_candle['close']):
                        confirmations.append('engulfing')
                        
            else:  # bearish
                # Similar relaxation for bearish signals
                if candle['close'] < candle['open']:  # Bearish candle
                    upper_wick = candle['high'] - candle['open']
                else:  # Bullish candle
                    upper_wick = candle['high'] - candle['close']
                
                # RELAXED: Pin bar
                if upper_wick > body_size * 1.2 and wick_ratio < 0.4:
                    confirmations.append('pin_bar')
                    
                # Strong bearish close near lows
                if candle['close'] < candle['open']:
                    close_position = (candle['close'] - candle['low']) / candle_range if candle_range > 0 else 0
                    if close_position < 0.2:  # Closed in bottom 20% of range
                        confirmations.append('strong_close')
                
                # Bearish engulfing
                if idx > recent_candles.index[0]:
                    prev_candle = ohlc_df.loc[ohlc_df.index[ohlc_df.index.get_loc(idx) - 1]]
                    if (candle['close'] < candle['open'] and 
                        prev_candle['close'] > prev_candle['open'] and
                        candle['close'] < prev_candle['open'] and 
                        candle['open'] > prev_candle['close']):
                        confirmations.append('engulfing')
        
        # Check for momentum
        if len(recent_candles) >= 2:
            # Look for consistent directional movement
            if bias == 'bullish':
                consecutive_higher_lows = sum(1 for i in range(1, len(recent_candles)) 
                                            if recent_candles['low'].iloc[i] > recent_candles['low'].iloc[i-1])
                if consecutive_higher_lows >= 2:
                    confirmations.append('momentum_shift')
            else:
                consecutive_lower_highs = sum(1 for i in range(1, len(recent_candles)) 
                                            if recent_candles['high'].iloc[i] < recent_candles['high'].iloc[i-1])
                if consecutive_lower_highs >= 2:
                    confirmations.append('momentum_shift')
        
        # Check for increased volatility/momentum using price action
        # Instead of volume, use range expansion in forex
        if len(recent_candles) >= 3:
            recent_ranges = recent_candles['high'] - recent_candles['low']
            avg_range = recent_ranges[:-1].mean()  # Average of previous candles
            current_range = recent_ranges.iloc[-1]
            
            # Range expansion can indicate momentum
            if current_range > avg_range * 1.5:
                confirmations.append('range_expansion')
        
        min_confirmations = getattr(self.config, 'MIN_CONFIRMATIONS_REQUIRED', 1)
        return len(confirmations) >= min_confirmations

class ICTSignalGenerator:
    """
    Generates trading signals following the ICT narrative sequence.
    """
    
    def __init__(self, config):
        self.config = config
        self.analyzer = ICTAnalyzer(config)  
        self.liquidity_detector = LiquidityDetector()      
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str, spread: float, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """
        [REFACTORED] Generate trading signal with decoupled, session-specific logic.
        """
        analysis = self.analyzer.analyze(ohlc_df, symbol, daily_df, h4_df)
        if not analysis or analysis.get('daily_bias', 'neutral') == 'neutral':
            return None, None, None, None

        session = analysis['session']
        if not session.get('in_killzone'):
            return None, None, None, None

        entry_signal, sl_price, tp_price, narrative = None, None, None, None
        killzone = session['killzone_name']
        
        # --- LONDON KILL ZONE ---
        if killzone == 'London':
            entry_signal, sl_price, tp_price, narrative = self._handle_london_session(analysis, ohlc_df, spread)

        # --- NEW YORK KILL ZONE ---
        elif killzone == 'NewYork':
            entry_signal, sl_price, tp_price, narrative = self._handle_new_york_session(analysis, ohlc_df, spread)

        # --- LONDON CLOSE KILL ZONE ---
        elif killzone == 'LondonClose':
            entry_signal, sl_price, tp_price, narrative = self._handle_london_close_session(analysis, ohlc_df, spread)

        # --- Final Validation ---
        if entry_signal:
            # Check for signal invalidation before final trade validation
            if not self._is_signal_still_valid(analysis, narrative, ohlc_df):
                logger.info(f"{narrative.entry_model}: Signal invalidated by subsequent price action.")
                return None, None, None, None

            if not self._validate_trade(entry_signal, analysis['current_price'], sl_price, tp_price, narrative):
                return None, None, None, None
            
            logger.info(f"ICT Signal Generated: {entry_signal} {symbol} @ {analysis['current_price']:.5f}")
            logger.info(f"  Narrative: {narrative.daily_bias.upper()} | {narrative.po3_phase} | {narrative.entry_model}")
            logger.info(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            return entry_signal, sl_price, tp_price, narrative

        return None, None, None, None

    def _handle_london_session(self, analysis, ohlc_df, spread):
        """Logic for London Kill Zone: Focus on Judas Swing of Asian Range."""
        narrative = self._build_narrative(analysis)
        narrative.entry_model = "LONDON_JUDAS_SWING"
        
        # Check for Judas Swing of Asian Range
        asian_range = analysis['session'].get('last_asian_range')
        if not asian_range:
            return None, None, None, None

        manipulation = self._check_session_sweep(ohlc_df, analysis['daily_bias'], asian_range, 'asian')
        if not manipulation:
            return None, None, None, None
        
        narrative.manipulation_confirmed = True
        narrative.manipulation_level = manipulation['level']
        
        # Confirm with MSS
        mss_confirmed, mss_level, fvg_found = self.analyzer._check_market_structure_shift(ohlc_df, manipulation['index'], analysis['daily_bias'], manipulation['swept_level'])
        if not mss_confirmed:
            return None, None, None, None
        
        narrative.structure_broken = True
        narrative.structure_level = mss_level

        # Set the FVG as the new POI for entry
        narrative.fvg_zone = fvg_found

        # Setup trade
        signal, sl, tp = self._setup_entry_from_manipulation(analysis, manipulation, ohlc_df, spread)
        return signal, sl, tp, narrative

    def _handle_new_york_session(self, analysis, ohlc_df, spread):
        """Logic for New York Kill Zone: Focus on sweep of London Range or OTE continuation."""
        narrative = self._build_narrative(analysis)
        
        # Model 1: Sweep of London Session High/Low
        london_range = analysis['session'].get('last_london_range')
        if london_range:
            manipulation = self._check_session_sweep(ohlc_df, analysis['daily_bias'], london_range, 'london')
            if manipulation:
                mss_confirmed, mss_level, fvg_found = self.analyzer._check_market_structure_shift(ohlc_df, manipulation['index'], analysis['daily_bias'], manipulation['swept_level'])
                if mss_confirmed:
                    narrative.entry_model = "NY_LONDON_SWEEP"
                    narrative.manipulation_confirmed = True
                    narrative.manipulation_level = manipulation['level']
                    narrative.structure_broken = True
                    narrative.structure_level = mss_level
                    narrative.fvg_zone = fvg_found
                    signal, sl, tp = self._setup_entry_from_manipulation(analysis, manipulation, ohlc_df, spread)
                    return signal, sl, tp, narrative

        # Model 2: OTE entry if no new sweep is found
        if analysis['ote_zones']:
            ote_zone = self._get_valid_ote_zone(analysis['current_price'], analysis['ote_zones'], analysis['daily_bias'])
            if ote_zone and self.analyzer._check_rejection_confirmation(ohlc_df, ote_zone['sweet'], analysis['daily_bias']):
                narrative.entry_model = "NY_OTE_CONTINUATION"
                signal, sl, tp = self._setup_ote_entry(analysis, ote_zone, ohlc_df, spread)
                return signal, sl, tp, narrative

        return None, None, None, None

    def _handle_london_close_session(self, analysis, ohlc_df, spread):
        """Logic for London Close: Focus on FVG retracements for reversals or continuations."""
        narrative = self._build_narrative(analysis)
        
        fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], analysis['daily_bias'], analysis['current_price'], analysis['manipulation'])
        if fvg_entry:
            fvg_mid = (fvg_entry['top'] + fvg_entry['bottom']) / 2
            if self.analyzer._check_rejection_confirmation(ohlc_df, fvg_mid, analysis['daily_bias']):
                narrative.entry_model = "LC_FVG_RETRACEMENT"
                signal, sl, tp = self._setup_fvg_entry(analysis, fvg_entry, ohlc_df, spread)
                return signal, sl, tp, narrative
                
        return None, None, None, None

    def _check_session_sweep(self, ohlc_df, bias, session_range, session_name):
        """Checks for a sweep of a given session's high or low."""
        if bias == 'neutral': return None
        
        search_window = ohlc_df[ohlc_df.index >= session_range['end_time_utc']]
        
        if bias == 'bearish':
            sweep_up = search_window[search_window['high'] > session_range['high']]
            if not sweep_up.empty:
                sweep_time = sweep_up.index[0]
                sweep_index = ohlc_df.index.get_loc(sweep_time)
                return {'type': f'bearish_{session_name}_sweep', 'level': sweep_up['high'].iloc[0], 'swept_level': session_range['high'], 'index': sweep_index}
        
        elif bias == 'bullish':
            sweep_down = search_window[search_window['low'] < session_range['low']]
            if not sweep_down.empty:
                sweep_time = sweep_down.index[0]
                sweep_index = ohlc_df.index.get_loc(sweep_time)
                return {'type': f'bullish_{session_name}_sweep', 'level': sweep_down['low'].iloc[0], 'swept_level': session_range['low'], 'index': sweep_index}
        
        return None

    def _is_signal_still_valid(self, analysis, narrative, ohlc_df):
        """
        Checks if the generated signal is still valid based on current price action.
        A signal is invalidated if:
        1. The manipulation level has been violated.
        2. The FVG that confirmed the MSS has been violated.
        """
        current_price = ohlc_df['close'].iloc[-1]
        
        # Check 1: Manipulation level violation
        if narrative.manipulation_confirmed:
            manipulation_level = narrative.manipulation_level
            if narrative.daily_bias == 'bullish': # Swept low, price should stay above
                if current_price < manipulation_level:
                    logger.debug(f"Signal invalidated: Price ({current_price:.5f}) below manipulation low ({manipulation_level:.5f})")
                    return False
            elif narrative.daily_bias == 'bearish': # Swept high, price should stay below
                if current_price > manipulation_level:
                    logger.debug(f"Signal invalidated: Price ({current_price:.5f}) above manipulation high ({manipulation_level:.5f})")
                    return False

        # Check 2: FVG violation (if an FVG was part of the narrative)
        if narrative.fvg_zone:
            fvg_top = narrative.fvg_zone['top']
            fvg_bottom = narrative.fvg_zone['bottom']
            
            if narrative.daily_bias == 'bullish': # Bullish FVG, price should not go below bottom
                if current_price < fvg_bottom:
                    logger.debug(f"Signal invalidated: Price ({current_price:.5f}) below bullish FVG bottom ({fvg_bottom:.5f})")
                    return False
            elif narrative.daily_bias == 'bearish': # Bearish FVG, price should not go above top
                if current_price > fvg_top:
                    logger.debug(f"Signal invalidated: Price ({current_price:.5f}) above bearish FVG top ({fvg_top:.5f})")
                    return False

        return True

    def _setup_entry_from_manipulation(self, analysis, manipulation, ohlc_df, spread):
        """Generic trade setup based on a manipulation event."""
        bias = analysis['daily_bias']
        current_price = analysis['current_price']
        
        if bias == 'bullish':
            signal = "BUY"
            sl_price = manipulation['level'] - (self._calculate_atr_buffer(ohlc_df) * 0.5) # 0.5 ATR buffer below manipulation
        else: # bearish
            signal = "SELL"
            sl_price = manipulation['level'] + (self._calculate_atr_buffer(ohlc_df) * 0.5) # 0.5 ATR buffer above manipulation
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        return signal, sl_price, tp_price

    def _get_valid_ote_zone(self, current_price, ote_zones, daily_bias):
        """Finds the first OTE zone that matches the bias and contains the current price."""
        for ote in ote_zones:
            if ote['direction'].lower() == daily_bias:
                if ote['low'] <= current_price <= ote['high']:
                    return ote
        return None

    def _validate_trade(self, signal, entry, sl, tp, narrative):
        """Centralized validation for any trade signal."""
        if not all([signal, entry, sl, tp, narrative]):
            return False
        
        if tp is None or sl is None:
            logger.warning(f"{narrative.entry_model}: Trade rejected because SL or TP is None.")
            return False

        # Basic price validation
        if signal == "BUY" and not (sl < entry < tp):
            logger.warning(f"{narrative.entry_model}: BUY signal invalid levels. SL={sl}, Entry={entry}, TP={tp}")
            return False
        if signal == "SELL" and not (tp < entry < sl):
            logger.warning(f"{narrative.entry_model}: SELL signal invalid levels. SL={sl}, Entry={entry}, TP={tp}")
            return False
            
        # Risk validation
        risk = abs(entry - sl)
        if risk == 0:
            logger.warning(f"{narrative.entry_model}: Risk is zero, invalid trade.")
            return False
            
        # Add any other global validation rules here
        
        return True

    def _setup_fvg_entry(self, analysis, fvg_entry, ohlc_df, spread):
        """Setup entry from a Fair Value Gap."""
        bias = analysis['daily_bias']
        current_price = analysis['current_price']
        manipulation = analysis['manipulation']

        if bias == 'bullish':
            signal = "BUY"
            sl_price = fvg_entry['bottom'] - (self._calculate_atr_buffer(ohlc_df) * 0.5)
        else: # bearish
            signal = "SELL"
            sl_price = fvg_entry['top'] + (self._calculate_atr_buffer(ohlc_df) * 0.5)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        return signal, sl_price, tp_price
        
    def _find_retracement_fvg(self, fair_value_gaps, daily_bias, current_price, manipulation, total_candles=None):
        """
        Enhanced FVG finder that considers:
        1. Post-manipulation FVGs (priority)
        2. Pre-manipulation FVGs that align with bias
        3. Partially filled FVGs
        """
        if not fair_value_gaps:
            return None
        
        manipulation_index = manipulation.get('index', -1)
        
        # First, try to find post-manipulation FVGs (highest priority)
        for fvg in fair_value_gaps:
            if (daily_bias == 'bullish' and fvg['type'] == 'bullish') or \
            (daily_bias == 'bearish' and fvg['type'] == 'bearish'):
                
                if fvg.get('index', 0) > manipulation_index:
                    if fvg['bottom'] <= current_price <= fvg['top']:
                        logger.debug(f"Found post-manipulation FVG: {fvg}")
                        return fvg
        
        # If no post-manipulation FVG, consider pre-manipulation ones
        # that are still valid (unmitigated) and align with our bias
        for fvg in fair_value_gaps:
            if (daily_bias == 'bullish' and fvg['type'] == 'bullish') or \
            (daily_bias == 'bearish' and fvg['type'] == 'bearish'):
                
                # Even if created before manipulation, if price is retracing into it now
                # and it aligns with our bias, it's still a valid level
                if fvg['bottom'] <= current_price <= fvg['top']:
                    # Additional check: ensure FVG is relatively recent
                    # If total_candles provided, check if FVG is within last 50 candles
                    if total_candles is None or fvg.get('index', 0) > total_candles - 50:
                        logger.debug(f"Found pre-manipulation aligned FVG: {fvg}")
                        return fvg
        
        # Check for FVGs we're approaching (within 5 pips)
        pip_tolerance = 0.0005  # 5 pips for forex
        for fvg in fair_value_gaps:
            if (daily_bias == 'bullish' and fvg['type'] == 'bullish'):
                distance_to_fvg = fvg['bottom'] - current_price
                if 0 < distance_to_fvg < pip_tolerance:
                    logger.debug(f"Approaching bullish FVG: {fvg}")
                    return fvg
            elif (daily_bias == 'bearish' and fvg['type'] == 'bearish'):
                distance_to_fvg = current_price - fvg['top']
                if 0 < distance_to_fvg < pip_tolerance:
                    logger.debug(f"Approaching bearish FVG: {fvg}")
                    return fvg
        
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
        fvg_zone = analysis['fvg_zone']
        
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
            fvg_zone=fvg_zone if fvg_zone else {},
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
    
    def _setup_ote_entry(self, analysis, ote_zone, ohlc_df, spread):
        """Setup entry from OTE zone."""
        bias = analysis['daily_bias']
        current_price = analysis['current_price']
        
        if bias == 'bullish':
            signal = "BUY"
            sl_price = ote_zone['swing_low'] - (self._calculate_atr_buffer(ohlc_df) * 0.5)
        else:  # bearish
            signal = "SELL"
            sl_price = ote_zone['swing_high'] + (self._calculate_atr_buffer(ohlc_df) * 0.5)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)        
        return signal, sl_price, tp_price
        
    def _setup_ob_entry(self, bias, current_price, order_block, manipulation, ohlc_df, analysis):
        """Setup entry from order block."""
        if bias == 'bullish':
            signal = "BUY"
            sl_price = order_block['bottom'] - (self._calculate_atr_buffer(ohlc_df) * 0.5)
        else:
            signal = "SELL"
            sl_price = order_block['top'] + (self._calculate_atr_buffer(ohlc_df) * 0.5)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        
        return signal, sl_price, tp_price
    
    def _setup_judas_entry(self, bias, current_price, manipulation, ohlc_df, analysis):
        """Setup entry from Judas Swing manipulation."""
        if bias == 'bullish':
            signal = "BUY"
            # SL below the Judas sweep low
            sl_price = manipulation.get('level', current_price) - (self._calculate_atr_buffer(ohlc_df) * 0.5)
        else:  # bearish
            signal = "SELL"
            sl_price = manipulation.get('level', current_price) + (self._calculate_atr_buffer(ohlc_df) * 0.5)
        
        # Target the opposite side of Asian range or liquidity
        tp_price = self._calculate_target(current_price, sl_price, bias, analysis)
        
        return signal, sl_price, tp_price
    
    def _calculate_atr_buffer(self, ohlc_df):
        """Calculate ATR-based buffer for stop loss."""
        atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        return atr.iloc[-1] * self.config.SL_ATR_MULTIPLIER
    
    def _calculate_target(self, entry_price, sl_price, bias, analysis):
        """
        Calculate take profit by targeting proper ICT liquidity levels.
        If no valid target is found, it falls back to a fixed R:R ratio.
        """
        risk = abs(entry_price - sl_price)
        if risk == 0:
            return None

        liquidity_levels = analysis.get('liquidity_levels', {'buy_side': [], 'sell_side': []})
        
        # Use proper ICT liquidity targeting
        target = self.liquidity_detector.get_target_for_bias(
            bias, liquidity_levels, entry_price, 
            min_reward_risk=getattr(self.config, 'MIN_TARGET_RR', 1.0),
            sl_price=sl_price
        )
        
        if target:
            target_level = target['level']
            potential_reward = abs(target_level - entry_price)
            min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)
            
            if potential_reward >= risk * min_rr:
                logger.info(f"Valid liquidity target: {target['description']} at {target_level:.5f}, offering {potential_reward/risk:.1f}R.")
                return target_level
            else:
                logger.warning(
                    f"Liquidity target at {target_level:.5f} is too close for minimum R:R. "
                    f"Required reward: {risk * min_rr:.5f}, Potential reward: {potential_reward:.5f}. "
                    f"Falling back to fixed R:R."
                )
        else:
            target_side = 'sell-side' if bias == 'bullish' else 'buy-side'
            logger.warning(f"No valid {target_side} liquidity target found for {bias} trade. Falling back to fixed R:R.")

        # Fallback to fixed R:R
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
        else:  # SELL
            return tp < entry < sl