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
        self._last_analysis_cache = {}
        self._cache_max_size = 50
        
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
        structure = self._get_structure(ohlc_df, swings, session_context, daily_bias, manipulation)
        order_blocks = self._get_order_blocks(ohlc_df, swings)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        liquidity_zones = self._get_liquidity(ohlc_df, swings)
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, swings, daily_bias, manipulation)
        htf_levels = self._get_htf_levels(h4_df)  # ✅ Now uses real H4 data
        
        analysis_result = {
            'symbol': symbol, 'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1], 'swings': swings, 'structure': structure,
            'daily_bias': daily_bias, 'po3_analysis': po3_analysis, 'manipulation': manipulation,
            'order_blocks': order_blocks, 'fair_value_gaps': fair_value_gaps,
            'liquidity_zones': liquidity_zones, 'premium_discount': pd_analysis,
            'ote_zones': ote_zones, 'session': session_context, 'htf_levels': htf_levels
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
            logger.debug(f"HTF levels - h4_df columns: {list(h4_df.columns)}")
            if 'volume' not in h4_df.columns:
                logger.error(f"MISSING VOLUME COLUMN in HTF levels! Columns: {list(h4_df.columns)}")
                return {}
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
            return pd.DataFrame()
            
        try:
            structure = self._get_ict_structure(ohlc_df, swings, session_context, daily_bias, manipulation)
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
            
            # Create Asian range times (7 PM - 10 PM NY)
            asian_start_ny = latest_ny_time.replace(
                year=asian_date.year,
                month=asian_date.month,
                day=asian_date.day,
                hour=19,  # 7 PM NY time
                minute=0,
                second=0,
                microsecond=0
            )
            asian_end_ny = asian_start_ny.replace(hour=22)  # 10 PM NY time
            
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

        # Use daily data if provided, otherwise warn and use M15
        if daily_df is None or daily_df.empty:
            logger.warning(f"{symbol}: No daily data provided for bias determination. Results may be inaccurate.")
            analysis_df = ohlc_df
        else:
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
        
        logger.info(f"{symbol}: PO3 Phase: {po3_analysis['current_phase']} | Session: {po3_analysis.get('session', 'Unknown')} | Bias: {daily_bias}")
        
        return daily_bias, po3_analysis
            
    def _check_market_structure_shift(self, ohlc_df, sweep_index, direction, swept_level):
        """
        Corrected version using a more robust definition of a Market Structure Shift (MSS)
        based on ICT principles.
        """
        try:
            # 1. Isolate the data of the last price leg leading to the sweep.
            # We look back from the point of the sweep to find the move that took liquidity.
            lookback = 50 
            start_index = max(0, sweep_index - lookback)
            last_leg_df = ohlc_df.iloc[start_index:sweep_index]

            if len(last_leg_df) < 5:
                return False, None

            # 2. Find the internal swings within this last leg ONLY.
            # A shorter swing length is used to find the more sensitive internal structure.
            internal_swings = smc.swing_highs_lows(last_leg_df, swing_length=5)
            
            if internal_swings.empty:
                return False, None

            # 3. Identify the specific internal swing high/low to be broken.
            mss_level = None
            if direction == 'bearish':  # Swept high, looking for break of internal low
                internal_lows = internal_swings[internal_swings['HighLow'] == -1].dropna()
                if internal_lows.empty:
                    return False, None
                mss_level = internal_lows['Level'].iloc[-1]
            
            else:  # 'bullish'. Swept low, looking for break of internal high
                internal_highs = internal_swings[internal_swings['HighLow'] == 1].dropna()
                if internal_highs.empty:
                    return False, None
                mss_level = internal_highs['Level'].iloc[-1]

            # 4. Check for a break of the identified internal MSS level.
            post_sweep_data = ohlc_df.iloc[sweep_index:]
            for idx, candle in post_sweep_data.head(10).iterrows():
                if direction == 'bearish' and candle['close'] < mss_level:
                    return True, mss_level
                elif direction == 'bullish' and candle['close'] > mss_level:
                    return True, mss_level
            
            return False, None

        except Exception as e:
            logger.error(f"Error checking MSS: {e}", exc_info=True)
            return False, None
        
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
        
    def _check_manipulation_patterns(self, ohlc_df, daily_bias, session_context):
        """
        [CORRECTED & COMPLETE] Checks for a sequence of manipulation patterns (Judas, Turtle Soup, etc.)
        and returns the first one found that aligns with the established daily bias.
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
                        mss_confirmed, mss_level = self._check_market_structure_shift(ohlc_df, sweep_index, 'bearish', asian_high)
                        if mss_confirmed:
                            return {'type': 'bearish_judas', 'level': sweep_up['high'].iloc[0], 'index': sweep_index, 'mss_level': mss_level}
                
                elif daily_bias == 'bullish':
                    sweep_down = search_window[search_window['low'] < asian_low]
                    if not sweep_down.empty:
                        sweep_time = sweep_down.index[0]
                        sweep_index = ohlc_df.index.get_loc(sweep_time)
                        mss_confirmed, mss_level = self._check_market_structure_shift(ohlc_df, sweep_index, 'bullish', asian_low)
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
                    # Did this candle sweep the high?
                    if sweep_candle['high'] > swept_level:
                        # Proof: Did this sweep lead to a bearish MSS?
                        mss_confirmed, mss_level = self._check_market_structure_shift(
                            ohlc_df, sweep_index, 'bearish', swept_level
                        )
                        if mss_confirmed:
                            return {'type': 'bearish_liquidity_sweep_mss', 'level': sweep_candle['high'], 'swept_level': swept_level, 'index': sweep_index, 'mss_level': mss_level}

                # --- Check for a Bullish Sweep + MSS ---
                elif daily_bias == 'bullish' and swing['HighLow'] == -1: # Target swing lows
                    swept_level = swing['Level']
                    # Did this candle sweep the low?
                    if sweep_candle['low'] < swept_level:
                        # Proof: Did this sweep lead to a bullish MSS?
                        mss_confirmed, mss_level = self._check_market_structure_shift(
                            ohlc_df, sweep_index, 'bullish', swept_level
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
            return self._create_empty_structure(n)
        
        asian_range = session_context['last_asian_range']
        external_high = asian_range['high']
        external_low = asian_range['low']
        
        # Get internal swings (within Asian range)
        internal_swings = swings.dropna()
        if internal_swings.empty:
            return self._create_empty_structure(n)
        
        # Track key internal levels based on bias direction
        key_internal_high = None
        key_internal_low = None
        internal_high_idx = None
        internal_low_idx = None
        
        # Identify the key internal levels to watch
        manipulation_index = manipulation.get('index', -1)
        
        if daily_bias == 'bullish':
            # In bullish bias, watch for break of internal low (CHoCH signal)
            post_manipulation_swings = internal_swings[internal_swings.index > manipulation_index]
            internal_lows = post_manipulation_swings[post_manipulation_swings['HighLow'] == -1]
            
            if not internal_lows.empty:
                # Get the most recent internal low
                key_internal_low = internal_lows['Level'].iloc[-1]
                internal_low_idx = internal_lows.index[-1]
        
        elif daily_bias == 'bearish':
            # In bearish bias, watch for break of internal high (CHoCH signal)
            post_manipulation_swings = internal_swings[internal_swings.index > manipulation_index]
            internal_highs = post_manipulation_swings[post_manipulation_swings['HighLow'] == 1]
            
            if not internal_highs.empty:
                # Get the most recent internal high
                key_internal_high = internal_highs['Level'].iloc[-1]
                internal_high_idx = internal_highs.index[-1]
        
        # Process each candle for structure breaks
        for i in range(manipulation_index + 1, n):  # Only check after manipulation
            current_close = ohlc_df['close'].iloc[i]
            
            # Check for EXTERNAL range breaks (major BOS)
            if daily_bias == 'bullish' and current_close > external_high:
                # Bullish BOS - break of external high WITH trend
                if self._validate_displacement_break(ohlc_df, i, external_high, 'bullish'):
                    # Find index where external high was set (start of Asian session)
                    asian_start_idx = ohlc_df.index.get_loc(asian_range['start_time_utc'])
                    bos[asian_start_idx] = 1
                    level[asian_start_idx] = external_high
                    broken_index[asian_start_idx] = i
            
            elif daily_bias == 'bearish' and current_close < external_low:
                # Bearish BOS - break of external low WITH trend  
                if self._validate_displacement_break(ohlc_df, i, external_low, 'bearish'):
                    asian_start_idx = ohlc_df.index.get_loc(asian_range['start_time_utc'])
                    bos[asian_start_idx] = -1
                    level[asian_start_idx] = external_low
                    broken_index[asian_start_idx] = i
            
            # Check for INTERNAL structure breaks (CHoCH signals)
            if daily_bias == 'bullish' and key_internal_low is not None:
                if current_close < key_internal_low:
                    # Break of internal low AGAINST bullish bias = CHoCH
                    if self._validate_displacement_break(ohlc_df, i, key_internal_low, 'bearish'):
                        choch[internal_low_idx] = -1
                        level[internal_low_idx] = key_internal_low
                        broken_index[internal_low_idx] = i
                        # This internal level is now broken
                        key_internal_low = None
            
            elif daily_bias == 'bearish' and key_internal_high is not None:
                if current_close > key_internal_high:
                    # Break of internal high AGAINST bearish bias = CHoCH
                    if self._validate_displacement_break(ohlc_df, i, key_internal_high, 'bullish'):
                        choch[internal_high_idx] = 1
                        level[internal_high_idx] = key_internal_high
                        broken_index[internal_high_idx] = i
                        # This internal level is now broken
                        key_internal_high = None
        
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
            logger.debug(f"Order blocks - ohlc_df columns: {list(ohlc_df.columns)}")
            if 'volume' not in ohlc_df.columns:
                logger.error(f"MISSING VOLUME COLUMN in order blocks! Columns: {list(ohlc_df.columns)}")
                return []
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
    
    def _analyze_premium_discount(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame) -> Dict:
        """
        Analyze premium/discount zones based on the most recent dealing range
        defined by swing highs and lows, aligning with ICT principles.
        """
        if swings.empty:
            return {}

        # Find the last two swing points to define the current dealing range
        recent_swings = swings.dropna().tail(20)
        if len(recent_swings) < 4:
            return {} # Not enough swings to create a range

        # The dealing range is the high and low of the last two swing points
        high = recent_swings['Level'].max()
        low = recent_swings['Level'].min()
        
        if high == low:
            return {} # Invalid range

        range_size = high - low
        
        # Key Fibonacci levels
        levels = {
            'high': high,
            'low': low,
            'range': range_size,
            'equilibrium': low + (range_size * 0.5),
            'premium_75': low + (range_size * 0.75),
            'premium_80': low + (range_size * 0.80),
            'discount_25': low + (range_size * 0.25),
            'discount_20': low + (range_size * 0.20),
        }
        
        current_price = ohlc_df['close'].iloc[-1]
        
        # Determine zone
        if current_price > levels['equilibrium']:
            zone = 'premium'
        else:
            zone = 'discount'

        # Add deep zones for more granularity if needed
        if zone == 'premium' and current_price > levels['premium_75']:
            zone = 'deep_premium'
        elif zone == 'discount' and current_price < levels['discount_25']:
            zone = 'deep_discount'
        
        levels['current_zone'] = zone
        
        return levels
    
    def _calculate_ote_zones(self, ohlc_df, swings, daily_bias, manipulation):
        """
        Calculate Optimal Trade Entry (OTE) zones based on the specific
        price leg created by the manipulation, aligning with ICT principles.
        """
        if not manipulation.get('detected') or swings.empty:
            return []

        ote_zones = []
        manipulation_level = manipulation['level']
        manipulation_index = manipulation.get('index', -1)

        if daily_bias == 'bullish':
            # After a bullish manipulation (sweep down), we look for a swing high that forms AFTER it.
            # The range is from the manipulation low up to that swing high.
            swing_highs_after = swings[(swings['HighLow'] == 1) & (swings.index > manipulation_index)]
            if not swing_highs_after.empty:
                subsequent_high = swing_highs_after['Level'].iloc[0]
                # Ensure the high is actually above the manipulation level to form a valid range
                if subsequent_high > manipulation_level:
                    range_size = subsequent_high - manipulation_level
                    ote_zones.append({
                        'direction': 'bullish',
                        'high': subsequent_high - (range_size * 0.62),
                        'sweet': subsequent_high - (range_size * 0.705),
                        'low': subsequent_high - (range_size * 0.79),
                        'swing_high': subsequent_high,
                        'swing_low': manipulation_level
                    })

        elif daily_bias == 'bearish':
            # After a bearish manipulation (sweep up), we look for a swing low that forms AFTER it.
            # The range is from the manipulation high down to that swing low.
            swing_lows_after = swings[(swings['HighLow'] == -1) & (swings.index > manipulation_index)]
            if not swing_lows_after.empty:
                subsequent_low = swing_lows_after['Level'].iloc[0]
                # Ensure the low is actually below the manipulation level
                if subsequent_low < manipulation_level:
                    range_size = manipulation_level - subsequent_low
                    ote_zones.append({
                        'direction': 'bearish',
                        'high': subsequent_low + (range_size * 0.79),
                        'sweet': subsequent_low + (range_size * 0.705),
                        'low': subsequent_low + (range_size * 0.62),
                        'swing_high': manipulation_level,
                        'swing_low': subsequent_low
                    })
        
        return ote_zones
        
    def _get_htf_levels(self, ohlc_df):
        """Get higher timeframe levels."""
        try:
            # Get H4 levels
            htf_levels = smc.previous_high_low(ohlc_df, time_frame="4h")
            
            if htf_levels is not None and not htf_levels.empty:
                return {
                    'h4_high': htf_levels['PreviousHigh'].iloc[-1],
                    'h4_low': htf_levels['PreviousLow'].iloc[-1],
                    'h4_broken_high': htf_levels['BrokenHigh'].iloc[-1] == 1,
                    'h4_broken_low': htf_levels['BrokenLow'].iloc[-1] == 1
                }
        except Exception as e:
            logger.warning(f"Error getting HTF levels: {e}")
        
        return {}

    def _analyze_market_phase(self, ohlc_df, manipulation, daily_bias):
        """
        Determine current ICT PO3 phase (Accumulation, Manipulation, Distribution) based on 
        session context, manipulation detection, and market behavior.
        Returns: Dict with phase info and key levels
        """
        current_utc = ohlc_df.index[-1]
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_ny_hour = current_ny.hour
        current_price = ohlc_df['close'].iloc[-1]
        
        manipulation_detected = manipulation.get('detected', False)
        manipulation_index = manipulation.get('index', -1)
        manipulation_price = manipulation.get('level', current_price)
        
        # Default values
        phase = 'unknown'
        can_enter = False
        retracement_percent = 0
        extreme_price = current_price
        
        # Determine phase based on ICT PO3 session structure and manipulation state
        
        # === ACCUMULATION PHASE ===
        # Asian Session (7 PM - 12 AM NY) = Accumulation Phase
        if 19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 1:
            phase = 'accumulation'
            can_enter = False  # Don't trade during accumulation
            logger.debug(f"Asian Session - Accumulation Phase detected")
            
        # === MANIPULATION PHASE ===  
        # London Session (2 AM - 5 AM NY) = Manipulation Phase
        elif 2 <= current_ny_hour < 5:
            if manipulation_detected:
                phase = 'manipulation_active'
                # Check if manipulation just happened (within last 5 candles)
                bars_since_manipulation = len(ohlc_df) - 1 - manipulation_index if manipulation_index >= 0 else 999
                
                if bars_since_manipulation <= 5:
                    can_enter = getattr(self.config, 'ALLOW_MANIPULATION_PHASE_ENTRY', False)
                    logger.debug(f"Active manipulation detected {bars_since_manipulation} bars ago")
                else:
                    phase = 'manipulation_complete'
                    can_enter = True
                    logger.debug(f"Manipulation completed, ready for distribution")
            else:
                phase = 'manipulation_pending'
                can_enter = False
                logger.debug(f"London Session - Waiting for manipulation")
        
        # === DISTRIBUTION PHASE ===
        # New York Session (7 AM - 11 AM NY) = Distribution Phase  
        elif 7 <= current_ny_hour < 11:
            if manipulation_detected:
                phase = 'distribution'
                
                # Calculate retracement from manipulation
                post_manipulation = ohlc_df.iloc[manipulation_index:] if manipulation_index >= 0 else ohlc_df.tail(10)
                
                if daily_bias == 'bullish':
                    # After bullish manipulation (sweep down), look for move up
                    post_high = post_manipulation['high'].max()
                    extreme_price = post_high
                    
                    if current_price < manipulation_price:
                        phase = 'invalidated'
                        can_enter = False
                    else:
                        # Calculate retracement from the high
                        retracement_range = post_high - manipulation_price
                        current_retracement = post_high - current_price
                        retracement_percent = (current_retracement / retracement_range * 100) if retracement_range > 0 else 0
                        
                        retracement_threshold = getattr(self.config, 'RETRACEMENT_THRESHOLD_PERCENT', 25.0)
                        
                        if retracement_percent >= retracement_threshold:
                            phase = 'distribution_retracement'
                            can_enter = True
                        elif 0 < retracement_percent < retracement_threshold:
                            phase = 'distribution_shallow_pullback'
                            can_enter = False
                        else:
                            phase = 'distribution_expansion'
                            can_enter = False
                            
                else:  # bearish bias
                    # After bearish manipulation (sweep up), look for move down
                    post_low = post_manipulation['low'].min()
                    extreme_price = post_low
                    
                    if current_price > manipulation_price:
                        phase = 'invalidated'
                        can_enter = False
                    else:
                        # Calculate retracement from the low
                        retracement_range = manipulation_price - post_low
                        current_retracement = current_price - post_low
                        retracement_percent = (current_retracement / retracement_range * 100) if retracement_range > 0 else 0
                        
                        retracement_threshold = getattr(self.config, 'RETRACEMENT_THRESHOLD_PERCENT', 25.0)
                        
                        if retracement_percent >= retracement_threshold:
                            phase = 'distribution_retracement'
                            can_enter = True
                        elif 0 < retracement_percent < retracement_threshold:
                            phase = 'distribution_shallow_pullback'
                            can_enter = False
                        else:
                            phase = 'distribution_expansion'
                            can_enter = False
            else:
                phase = 'distribution_no_manipulation'
                can_enter = False
                logger.debug(f"NY Session but no manipulation detected - waiting")
        
        # === POST-DISTRIBUTION / CONSOLIDATION ===
        # After NY session or outside main sessions
        else:
            if manipulation_detected:
                phase = 'post_distribution'
                can_enter = False  # Avoid trading outside main sessions
            else:
                phase = 'consolidation'
                can_enter = False
        
        # Calculate bars since manipulation for additional context
        bars_since_manipulation = 0
        if manipulation_index >= 0:
            bars_since_manipulation = len(ohlc_df) - 1 - manipulation_index
        
        return {
            'phase': phase,
            'can_enter': can_enter,
            'retracement_percent': round(retracement_percent, 1),
            'manipulation_price': manipulation_price,
            'extreme_price': extreme_price,
            'bars_since_manipulation': bars_since_manipulation,
            'current_session': 'Asian' if (19 <= current_ny_hour < 24 or 0 <= current_ny_hour < 1) else
                            'London' if 2 <= current_ny_hour < 5 else
                            'NewYork' if 7 <= current_ny_hour < 11 else 'Other'
        }
        
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
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str, spread: float, daily_df: pd.DataFrame = None, h4_df: pd.DataFrame = None) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """
        Generate trading signal following the ICT narrative with proper daily bias and real H4 data.
        Now accepts daily_df for accurate bias determination and h4_df for real HTF levels.
        """
        # Pass H4 data to analyzer
        analysis = self.analyzer.analyze(ohlc_df, symbol, daily_df, h4_df) 
        if not analysis:
            return None, None, None, None
        
        # Extract key components
        daily_bias = analysis['daily_bias']
        manipulation = analysis['manipulation']
        current_price = analysis['current_price']
        session = analysis['session']
        
        # Get relaxation flags from config
        require_killzone = getattr(self.config, 'REQUIRE_KILLZONE', True)
        
        # Build the narrative
        narrative = self._build_narrative(analysis)
        
        # === NARRATIVE STEP 1: Daily Bias from DAILY chart
        if daily_bias == 'neutral':
            logger.debug(f"{symbol}: No clear daily bias from daily chart.")
            return None, None, None, None
        
        # === NARRATIVE STEP 2: Check Market Phase & Invalidation ===
        market_phase = self.analyzer._analyze_market_phase(ohlc_df, manipulation, daily_bias)
        logger.info(f"{symbol}: Phase: {market_phase['phase']} | Retracement: {market_phase.get('retracement_percent', 0):.1f}%")
        
        if market_phase['phase'] == 'invalidated':
            return None, None, None, None

        # === NARRATIVE STEP 3: Find Entry Model Based on Kill Zone ===
        entry_signal, sl_price, tp_price = None, None, None
        
        # Kill Zone specific models
        if session['in_killzone']:
            if session['killzone_name'] == 'London':
                # London: Focus on Judas Swing
                if manipulation.get('detected') and 'judas' in manipulation.get('type', ''):
                    logger.info(f"{symbol}: Looking for Judas Swing entry in London Kill Zone")
                    entry_signal, sl_price, tp_price = self._setup_judas_entry(
                        daily_bias, current_price, manipulation, ohlc_df, analysis
                    )
                    narrative.entry_model = "JUDAS_SWING"
                    
            elif session['killzone_name'] == 'NewYork':
                # New York: Focus on OTE/Continuation
                if market_phase['can_enter']:
                    ote_zones = analysis['ote_zones']
                    if ote_zones and self._is_price_in_ote(current_price, ote_zones, daily_bias):
                        ote_zone = ote_zones[0]
                        if self.analyzer._check_rejection_confirmation(ohlc_df, ote_zone['sweet'], daily_bias):
                            logger.info(f"{symbol}: OTE entry in New York Kill Zone")
                            entry_signal, sl_price, tp_price = self._setup_ote_entry(
                                daily_bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis
                            )
                            narrative.entry_model = "OTE_ENTRY"
                            
            elif session['killzone_name'] == 'LondonClose':
                # London Close: Look for reversals/retracements
                if market_phase['can_enter']:
                    # Check for FVG entries during London Close
                    fvg_entry = self._find_retracement_fvg(
                        analysis['fair_value_gaps'], daily_bias, current_price, manipulation
                    )
                    if fvg_entry:
                        fvg_mid = (fvg_entry['top'] + fvg_entry['bottom']) / 2
                        if self.analyzer._check_rejection_confirmation(ohlc_df, fvg_mid, daily_bias):
                            logger.info(f"{symbol}: FVG retracement entry in London Close")
                            entry_signal, sl_price, tp_price = self._setup_fvg_entry(
                                daily_bias, current_price, fvg_entry, manipulation, ohlc_df, analysis
                            )
                            narrative.entry_model = "FVG_RETRACEMENT"

        # === FINAL VALIDATION ===
        if entry_signal:
            if require_killzone and not session['in_killzone']:
                logger.info(f"{symbol}: Signal found, but outside kill zone - skipping.")
                return None, None, None, None
            
            if not self._is_safe_entry_location(ohlc_df, current_price, sl_price, daily_bias) or \
            not self._validate_levels(entry_signal, current_price, sl_price, tp_price):
                logger.warning(f"{symbol}: Entry location or SL/TP levels are invalid - skipping.")
                return None, None, None, None
            
            logger.info(f"ICT Signal Generated: {entry_signal} {symbol} @ {current_price:.5f}")
            logger.info(f"  Narrative: {daily_bias.upper()} | {manipulation.get('type', 'None')} | {narrative.entry_model}")
            logger.info(f"  SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            narrative.entry_model = narrative.entry_model or "ICT_NARRATIVE"
            return entry_signal, sl_price, tp_price, narrative        
        return None, None, None, None
    
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
        Calculate take profit by targeting the nearest major liquidity pool.
        Falls back to a fixed R:R if no clear liquidity target is found.
        """
        risk = abs(entry_price - sl_price)
        
        # Ensure we have liquidity zones
        liquidity_zones = []
        if isinstance(analysis, dict) and 'liquidity_zones' in analysis:
            liquidity_zones = analysis.get('liquidity_zones', [])
        
        # If no liquidity zones, try to get them from the latest analysis
        if not liquidity_zones and hasattr(self, '_last_analysis_cache'):
            cached_analysis = self.analyzer._last_analysis_cache.get(analysis.get('symbol', ''))
            if cached_analysis and 'liquidity_zones' in cached_analysis:
                liquidity_zones = cached_analysis['liquidity_zones']
        
        target_liquidity_level = None

        if bias == 'bullish' and liquidity_zones:
            # For a buy, target the nearest bearish liquidity (a swing high) above the entry.
            potential_targets = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'bearish' and liq['level'] > entry_price
            ]
            if potential_targets:
                # Find the closest high to target
                target_liquidity_level = min(potential_targets)
                logger.debug(f"Bullish liquidity target found: {target_liquidity_level}")

        elif bias == 'bearish' and liquidity_zones:
            # For a sell, target the nearest bullish liquidity (a swing low) below the entry.
            potential_targets = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'bullish' and liq['level'] < entry_price
            ]
            if potential_targets:
                # Find the closest low to target
                target_liquidity_level = max(potential_targets)
                logger.debug(f"Bearish liquidity target found: {target_liquidity_level}")

        # If a liquidity target is found, use it. Otherwise, use the fixed R:R as a fallback.
        if target_liquidity_level:
            # Basic check to ensure the liquidity target offers a reasonable R:R
            potential_reward = abs(target_liquidity_level - entry_price)
            min_rr = getattr(self.config, 'MIN_TARGET_RR', 1.0)
            if potential_reward >= risk * min_rr:
                logger.info(f"Using liquidity target at {target_liquidity_level:.5f} for TP")
                return target_liquidity_level
            else:
                logger.debug(f"Liquidity target too close ({potential_reward/risk:.1f}R), using fallback R:R")

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