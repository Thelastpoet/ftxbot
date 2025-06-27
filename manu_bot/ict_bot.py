"""
ICT Trading Engine 
"""

import pytz
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from smartmoneyconcepts import smc
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
        
    def analyze(self, ohlc_df: pd.DataFrame, symbol: str) -> Dict:
        """
        Perform comprehensive ICT analysis following the narrative sequence.
        Returns a complete analysis dictionary with all ICT concepts.
        """
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback:
            logger.warning(f"{symbol}: Insufficient data for ICT analysis")
            return {}
        
        # Step 1: Get swings
        swings = self._get_swings(ohlc_df)
        
        # Step 2: Analyze Sessions and determine session-based PO3 and Daily Bias
        session_context = self._get_session_context(ohlc_df)

        # Check if context was successfully created
        if session_context.get('error'):
            logger.warning(f"{symbol}: Could not determine session context. Reason: {session_context['error']}")
            return {} # Cannot proceed without time context

        # Pass the context object to the analysis function
        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_context, symbol)

        # Manipulation detection 
        manipulation = po3_analysis.get('manipulation', {
            'detected': False, 'type': None, 'level': None, 'judas_swing': None, 'liquidity_sweep': None
        })
        
        # Step 3: Analyze ICT market structure
        structure = self._get_structure(ohlc_df, swings, session_context, daily_bias, manipulation)
        
        # Step 4: Identify key levels (order blocks, FVGs, liquidity)
        order_blocks = self._get_order_blocks(ohlc_df, swings)
        fair_value_gaps = self._get_fvgs(ohlc_df)
        liquidity_zones = self._get_liquidity(ohlc_df, swings)
        
        # Step 5: Calculate premium/discount and OTE zones
        pd_analysis = self._analyze_premium_discount(ohlc_df, swings)
        ote_zones = self._calculate_ote_zones(ohlc_df, swings, daily_bias, manipulation)
                
        # Step 7: Get higher timeframe context
        htf_levels = self._get_htf_levels(ohlc_df)
        
        analysis_result = {
            'symbol': symbol,
            'current_price': ohlc_df['close'].iloc[-1],
            'timestamp': ohlc_df.index[-1],
            'swings': swings,
            'structure': structure,
            'daily_bias': daily_bias,
            'po3_analysis': po3_analysis,
            'manipulation': manipulation,
            'order_blocks': order_blocks,
            'fair_value_gaps': fair_value_gaps,
            'liquidity_zones': liquidity_zones,
            'premium_discount': pd_analysis,
            'ote_zones': ote_zones,
            'session': session_context,
            'htf_levels': htf_levels
        }
        
        # Cache this analysis
        self._last_analysis_cache[symbol] = analysis_result
        
        if len(self._last_analysis_cache) > self._cache_max_size:
            oldest_symbol = min(self._last_analysis_cache.keys(), 
                            key=lambda k: self._last_analysis_cache[k].get('timestamp', 0))
            del self._last_analysis_cache[oldest_symbol]
        
        return analysis_result
    
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
        Fixed version: Uses correct ICT timing for Asian range (NY midnight to 5 AM)
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
            
            # ICT Asian Range: NY midnight to 5 AM (not 8 PM to midnight!)
            # Find the most recent NY midnight
            current_ny_date = latest_ny_time.date()
            if latest_ny_time.hour < 5:  # Still in today's Asian range
                midnight_ny = latest_ny_time.replace(hour=0, minute=0, second=0, microsecond=0)
            else:  # Past 5 AM, look at last night's range
                midnight_ny = (latest_ny_time - pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Asian range ends at 5 AM NY time
            range_end_ny = midnight_ny + pd.Timedelta(hours=5)
            
            # Convert to UTC for filtering
            range_start_utc = midnight_ny.astimezone(pytz.UTC)
            range_end_utc = range_end_ny.astimezone(pytz.UTC)
            
            # Get the Asian range data
            session_df = ohlc_df[(ohlc_df.index >= range_start_utc) & (ohlc_df.index < range_end_utc)]
            
            if not session_df.empty:
                context['last_asian_range'] = {
                    'start_time_utc': session_df.index[0],
                    'end_time_utc': session_df.index[-1],
                    'high': session_df['high'].max(),
                    'low': session_df['low'].min(),
                    'start_idx': ohlc_df.index.get_loc(session_df.index[0])
                }
                
            # Check Kill Zones using NY time
            current_ny_hour = latest_ny_time.hour
            
            # London Kill Zone: 3-5 AM NY time (when Judas Swing occurs)
            if 3 <= current_ny_hour < 5:
                context['in_killzone'] = True
                context['killzone_name'] = 'London'
                
            # New York Kill Zone: 7-10 AM NY time 
            elif 7 <= current_ny_hour < 10:
                context['in_killzone'] = True
                context['killzone_name'] = 'NewYork'
                
        except Exception as e:
            context['error'] = f"Error calculating session context: {e}"
            logger.error(context['error'])
            
        return context
    
    def _confirm_displacement(self, ohlc_df, sweep_time, swept_level, direction, asian_range):
        """
        Corrected and simplified displacement check based on the core ICT principle:
        A true displacement move will leave behind a Fair Value Gap (FVG).
        """
        try:
            # 1. Isolate the candles immediately following the sweep to check for an FVG.
            # We look at the 5 candles after the sweep.
            sweep_index = ohlc_df.index.get_loc(sweep_time)
            post_sweep_df = ohlc_df.iloc[sweep_index : sweep_index + 5]

            if len(post_sweep_df) < 3:
                return False

            # 2. Use the smc.fvg function to detect any Fair Value Gaps in this small window.
            # We only care if an FVG was created, not if it was mitigated yet.
            fvgs = smc.fvg(post_sweep_df)
            
            if fvgs.empty:
                logger.debug("  Displacement Check: No FVGs found in post-sweep window.")
                return False

            # 3. Check if any of the detected FVGs align with the intended displacement direction.
            if direction == 'bearish':
                # For a bearish displacement, we need a bearish FVG (-1).
                bearish_fvg_found = -1 in fvgs['FVG'].values
                if bearish_fvg_found:
                    logger.info("  Displacement Check: CONFIRMED. Bearish FVG found after MSS.")
                return bearish_fvg_found
                
            elif direction == 'bullish':
                # For a bullish displacement, we need a bullish FVG (1).
                bullish_fvg_found = 1 in fvgs['FVG'].values
                if bullish_fvg_found:
                    logger.info("  Displacement Check: CONFIRMED. Bullish FVG found after MSS.")
                return bullish_fvg_found

            return False

        except Exception as e:
            logger.error(f"Error in displacement confirmation: {e}", exc_info=True)
            return False
        
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
    
    def _analyze_session_po3(self, ohlc_df: pd.DataFrame, session_context: dict, symbol: str) -> Tuple[str, Dict]:
        """
        ICT Daily Bias Analysis using official 2022 methodology.
        Based on research of Michael Huddleston's actual teachings.
        """
        daily_bias = 'neutral'
        po3_analysis = {
            'current_phase': 'unknown',
            'type': 'unknown',
            'manipulation': {'detected': False, 'type': None, 'level': None}
        }

        if not session_context['last_asian_range']:
            po3_analysis['reason'] = "No recent Asian session range found."
            logger.debug(f"{symbol} PO3: No Asian range found. Cannot determine bias.")
            return daily_bias, po3_analysis
        
        # === ICT METHOD 1: JUDAS SWING DETECTION (Original) ===
        latest_utc_time = ohlc_df.index[-1]
        latest_ny_time = latest_utc_time.astimezone(self.config.NY_TIMEZONE)
        london_kz_start_ny = latest_ny_time.replace(
            hour=self.config.ICT_LONDON_KILLZONE['start'].hour,
            minute=self.config.ICT_LONDON_KILLZONE['start'].minute,
            second=0, microsecond=0
        )
        london_kz_start_utc = london_kz_start_ny.astimezone(pytz.UTC)

        asian_range_start_utc = session_context['last_asian_range']['start_time_utc']
        pre_london_asian_df = ohlc_df[(ohlc_df.index >= asian_range_start_utc) & (ohlc_df.index < london_kz_start_utc)]
        
        if not pre_london_asian_df.empty:
            asian_high = pre_london_asian_df['high'].max()
            asian_low = pre_london_asian_df['low'].min()
            
            search_window = ohlc_df[ohlc_df.index >= london_kz_start_utc]
            
            if not search_window.empty:
                po3_analysis['accumulation_range'] = {'high': asian_high, 'low': asian_low}
                
                # Check for Bearish Judas Swing
                sweep_up = search_window[search_window['high'] > asian_high]
                if not sweep_up.empty:
                    sweep_candle_time = sweep_up.index[0]
                    sweep_index = ohlc_df.index.get_loc(sweep_candle_time)
                    
                    mss_confirmed, mss_level = self._check_market_structure_shift(
                        ohlc_df, sweep_index, 'bearish', asian_high
                    )
                    
                    if mss_confirmed:
                        displacement_confirmed = self._confirm_displacement(ohlc_df, sweep_candle_time, asian_high, 'bearish', None)
                        if displacement_confirmed:
                            logger.info(f"{symbol} PO3: BEARISH BIAS CONFIRMED via Judas Swing.")
                            daily_bias = 'bearish'
                            po3_analysis['type'] = 'bearish_po3'
                            po3_analysis['current_phase'] = 'distribution'
                            po3_analysis['manipulation'] = {
                                'detected': True,
                                'type': 'bearish_judas',
                                'level': sweep_up['high'].iloc[0],
                                'swept_level': asian_high,
                                'mss_level': mss_level,
                                'index': sweep_index
                            }

                # Check for Bullish Judas Swing
                if daily_bias == 'neutral':
                    sweep_down = search_window[search_window['low'] < asian_low]
                    if not sweep_down.empty:
                        sweep_candle_time = sweep_down.index[0]
                        sweep_index = ohlc_df.index.get_loc(sweep_candle_time)
                        
                        mss_confirmed, mss_level = self._check_market_structure_shift(
                            ohlc_df, sweep_index, 'bullish', asian_low
                        )
                        
                        if mss_confirmed:
                            displacement_confirmed = self._confirm_displacement(ohlc_df, sweep_candle_time, asian_low, 'bullish', None)
                            if displacement_confirmed:
                                logger.info(f"{symbol} PO3: BULLISH BIAS CONFIRMED via Judas Swing.")
                                daily_bias = 'bullish'
                                po3_analysis['type'] = 'bullish_po3'
                                po3_analysis['current_phase'] = 'distribution'
                                po3_analysis['manipulation'] = {
                                    'detected': True,
                                    'type': 'bullish_judas',
                                    'level': sweep_down['low'].iloc[0],
                                    'swept_level': asian_low,
                                    'mss_level': mss_level,
                                    'index': sweep_index
                                }

        # === ICT METHOD 2: DAILY TIMEFRAME ORDER FLOW (Research: Source 61-1) ===
        # "Banks and Institutional players rely primarily on the daily chart to execute large traders effectively"
        if daily_bias == 'neutral':            
            # Get recent daily structure (last 5-10 daily candles)
            daily_timeframe_df = ohlc_df.resample('1D').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().tail(10)
            
            if len(daily_timeframe_df) >= 3:
                # ICT: "Bullish order flow is characterized by higher highs and higher lows"
                recent_highs = daily_timeframe_df['high'].tail(3).values
                recent_lows = daily_timeframe_df['low'].tail(3).values
                
                # Check for bullish order flow (higher highs and higher lows)
                higher_highs = recent_highs[1] > recent_highs[0] and recent_highs[2] > recent_highs[1]
                higher_lows = recent_lows[1] > recent_lows[0] and recent_lows[2] > recent_lows[1]
                
                # Check for bearish order flow (lower highs and lower lows)  
                lower_highs = recent_highs[1] < recent_highs[0] and recent_highs[2] < recent_highs[1]
                lower_lows = recent_lows[1] < recent_lows[0] and recent_lows[2] < recent_lows[1]
                
                if higher_highs and higher_lows:
                    daily_bias = 'bullish'
                    po3_analysis['type'] = 'bullish_structure'
                    po3_analysis['manipulation'] = {
                        'detected': True,
                        'type': 'daily_structure_bullish',
                        'level': recent_highs[-1]
                    }
                elif lower_highs and lower_lows:
                    daily_bias = 'bearish'
                    po3_analysis['type'] = 'bearish_structure'
                    po3_analysis['manipulation'] = {
                        'detected': True,
                        'type': 'daily_structure_bearish',
                        'level': recent_lows[-1]
                    }

        # === ICT METHOD 3: SESSION LIQUIDITY SWEEPS (Research: Source 71-1) ===
        # "When the New York session open you have to look for the liquidity sweep on either side (high/low) affirming your daily bias"
        if daily_bias == 'neutral' and session_context['killzone_name'] == 'NewYork':
            logger.info(f"{symbol} PO3: Checking for NY session liquidity sweeps...")
            
            # Get NY session start
            ny_session_start_ny = latest_ny_time.replace(hour=7, minute=0, second=0, microsecond=0)
            ny_session_start_utc = ny_session_start_ny.astimezone(pytz.UTC)
            
            # Get current NY session data
            ny_session_df = ohlc_df[ohlc_df.index >= ny_session_start_utc]
            
            if not ny_session_df.empty and len(ny_session_df) >= 5:
                # Check if NY session has swept above Asian high or below Asian low
                asian_high = session_context['last_asian_range']['high']
                asian_low = session_context['last_asian_range']['low']
                
                ny_session_high = ny_session_df['high'].max()
                ny_session_low = ny_session_df['low'].min()
                
                # ICT: Look for liquidity sweep in NY session
                if ny_session_high > asian_high:
                    sweep_candle = ny_session_df[ny_session_df['high'] == ny_session_high]
                    if not sweep_candle.empty:
                        sweep_index = ohlc_df.index.get_loc(sweep_candle.index[0])
                
                    daily_bias = 'bearish'
                    po3_analysis['type'] = 'ny_liquidity_sweep'
                    po3_analysis['manipulation'] = {
                        'detected': True,
                        'type': 'ny_sweep_high',
                        'level': ny_session_high,
                        'swept_level': asian_high,
                        'index': sweep_index
                    }
                elif ny_session_low < asian_low:
                    sweep_candle = ny_session_df[ny_session_df['low'] == ny_session_low]
                    if not sweep_candle.empty:
                        sweep_index = ohlc_df.index.get_loc(sweep_candle.index[0])
                
                    daily_bias = 'bullish'
                    po3_analysis['type'] = 'ny_liquidity_sweep'
                    po3_analysis['manipulation'] = {
                        'detected': True,
                        'type': 'ny_sweep_low',
                        'level': ny_session_low,
                        'swept_level': asian_low,
                        'index': sweep_index
                    }

        # Update phase based on current context
        if po3_analysis['manipulation']['detected']:
            if session_context['killzone_name'] == 'London':
                po3_analysis['current_phase'] = 'manipulation'
            elif session_context['killzone_name'] == 'NewYork':
                po3_analysis['current_phase'] = 'distribution'
        else:
            po3_analysis['current_phase'] = 'accumulation'
        
        return daily_bias, po3_analysis
    
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
        Determine if price is in expansion, retracement, or invalidated phase post-manipulation.
        Returns: Dict with phase info and key levels
        """
        if not manipulation['detected']:
            return {'phase': 'unknown', 'can_enter': False}
        
        manipulation_index = manipulation.get('index', -1)
        if manipulation_index < 0:
            return {'phase': 'unknown', 'can_enter': False}
        
        post_manipulation = ohlc_df.iloc[manipulation_index:]
        if len(post_manipulation) < 2:  # Reduced from 3
            return {'phase': 'early', 'can_enter': False}
        
        current_price = ohlc_df['close'].iloc[-1]
        manipulation_price = manipulation['level']
        
        phase = 'unknown'
        can_enter = False
        retracement_percent = 0
        extreme_price = 0

        retracement_threshold = getattr(self.config, 'RETRACEMENT_THRESHOLD_PERCENT', 25.0)
        allow_manipulation_entry = getattr(self.config, 'ALLOW_MANIPULATION_PHASE_ENTRY', False)

        if daily_bias == 'bullish':
            post_high = post_manipulation['high'].max()
            extreme_price = post_high
            
            if current_price < manipulation_price:
                phase = 'invalidated'
                can_enter = False
            else:
                retracement_range = post_high - manipulation_price
                current_retracement = post_high - current_price
                retracement_percent = (current_retracement / retracement_range * 100) if retracement_range > 0 else 0
                
                # Allow entry during manipulation phase if configured
                if allow_manipulation_entry and len(post_manipulation) <= 5:
                    phase = 'manipulation_ongoing'
                    can_enter = True
                elif retracement_percent >= retracement_threshold:
                    phase = 'retracement'
                    can_enter = True
                elif 0 < retracement_percent < retracement_threshold:
                    phase = 'shallow_pullback'
                    can_enter = False
                else:
                    phase = 'expansion'
                    can_enter = False
                    
        else:  # bearish
            post_low = post_manipulation['low'].min()
            extreme_price = post_low
            
            if current_price > manipulation_price:
                phase = 'invalidated'
                can_enter = False
            else:
                retracement_range = manipulation_price - post_low
                current_retracement = current_price - post_low
                retracement_percent = (current_retracement / retracement_range * 100) if retracement_range > 0 else 0

                # Allow entry during manipulation phase if configured
                if allow_manipulation_entry and len(post_manipulation) <= 5:
                    phase = 'manipulation_ongoing'
                    can_enter = True
                elif retracement_percent >= retracement_threshold:
                    phase = 'retracement'
                    can_enter = True
                elif 0 < retracement_percent < retracement_threshold:
                    phase = 'shallow_pullback'
                    can_enter = False
                else:
                    phase = 'expansion'
                    can_enter = False
        
        return {
            'phase': phase,
            'can_enter': can_enter,
            'retracement_percent': retracement_percent,
            'manipulation_price': manipulation_price,
            'extreme_price': extreme_price,
            'bars_since_manipulation': len(post_manipulation)
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
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str, spread: float) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """
        Generate trading signal following the ICT narrative:
        1. Confirm daily bias & manipulation.
        2. Check if setup has been invalidated.
        3. Find an optimal entry context (FVG, OTE, OB).
        
        Returns: (signal, sl_price, tp_price, narrative)
        """
        analysis = self.analyzer.analyze(ohlc_df, symbol)
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
        
        # === NARRATIVE STEP 1: Daily Bias & Manipulation ===
        if daily_bias == 'neutral' or not manipulation['detected']:
            logger.debug(f"{symbol}: No clear bias or manipulation yet.")
            return None, None, None, None
        
        # === NARRATIVE STEP 2: Check Market Phase & Invalidation ===
        market_phase = self.analyzer._analyze_market_phase(ohlc_df, manipulation, daily_bias)
        logger.info(f"{symbol}: Phase: {market_phase['phase']} | Retracement: {market_phase.get('retracement_percent', 0):.1f}%")
        
        if market_phase['phase'] == 'invalidated':
            return None, None, None, None

        # === NARRATIVE STEP 3: Find Entry Model ===
        entry_signal, sl_price, tp_price = None, None, None
        
        # MODEL 1: FVG Entry (Highest Priority - works even in shallow pullbacks)
        fvg_entry = self._find_retracement_fvg(analysis['fair_value_gaps'], daily_bias, current_price, manipulation)
        if fvg_entry:
            fvg_mid = (fvg_entry['top'] + fvg_entry['bottom']) / 2
            if self.analyzer._check_rejection_confirmation(ohlc_df, fvg_mid, daily_bias):
                logger.info(f"{symbol}: FVG entry confirmed during pullback.")
                entry_signal, sl_price, tp_price = self._setup_fvg_entry(daily_bias, current_price, fvg_entry, manipulation, ohlc_df, analysis)
                narrative.entry_model = "FVG_RETRACEMENT_ENTRY"
                
        if not entry_signal and market_phase['phase'] == 'shallow_pullback':
            # Try structure continuation entry for shallow pullbacks
            structure_entry = self._check_structure_continuation_entry(ohlc_df, daily_bias, current_price, analysis)
            if structure_entry and self.analyzer._check_rejection_confirmation(ohlc_df, structure_entry['level'], daily_bias):
                logger.info(f"{symbol}: Structure continuation entry in shallow pullback.")
                # Setup entry similar to other models
                signal = structure_entry['signal']
                if signal == "BUY":
                    sl_price = structure_entry['level'] - self._calculate_atr_buffer(ohlc_df)
                else:
                    sl_price = structure_entry['level'] + self._calculate_atr_buffer(ohlc_df)
                tp_price = self._calculate_target(current_price, sl_price, daily_bias, analysis)
                entry_signal = signal
                narrative.entry_model = "STRUCTURE_CONTINUATION"

        # If no FVG entry, check for deeper retracements for OTE/OB models
        if not entry_signal:
            if not market_phase['can_enter']:
                logger.debug(f"{symbol}: In '{market_phase['phase']}' phase. Waiting for a valid entry context (FVG or deeper retracement).")
                return None, None, None, None

            # MODEL 2: OTE Entry (Requires deep retracement)
            ote_zones = analysis['ote_zones']
            if ote_zones and self._is_price_in_ote(current_price, ote_zones, daily_bias):
                ote_zone = ote_zones[0]
                if self.analyzer._check_rejection_confirmation(ohlc_df, ote_zone['sweet'], daily_bias):
                    logger.info(f"{symbol}: OTE entry confirmed.")
                    entry_signal, sl_price, tp_price = self._setup_ote_entry(daily_bias, current_price, ote_zones, manipulation, ohlc_df, spread, analysis)
                    narrative.entry_model = "OTE_ENTRY"

            # MODEL 3: Order Block Entry (Requires deep retracement)
            elif analysis['order_blocks']:
                ob_entry = self._check_order_block_entry(current_price, analysis['order_blocks'], daily_bias, manipulation)
                if ob_entry:
                    ob_mid = (ob_entry['top'] + ob_entry['bottom']) / 2
                    if self.analyzer._check_rejection_confirmation(ohlc_df, ob_mid, daily_bias):
                        logger.info(f"{symbol}: Order Block entry confirmed.")
                        entry_signal, sl_price, tp_price = self._setup_ob_entry(daily_bias, current_price, ob_entry, manipulation, ohlc_df, analysis)
                        narrative.entry_model = "ORDER_BLOCK"

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
            logger.info(f"  Narrative: {daily_bias.upper()} | {manipulation['type']} | {narrative.entry_model}")
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
    
    def _check_structure_continuation_entry(self, ohlc_df, daily_bias, current_price, analysis):
        """
        New entry model: Structure continuation in shallow pullbacks
        Looks for micro-structure breaks in the direction of bias
        """
        if daily_bias == 'neutral':
            return None
            
        # Look at recent micro-structure (last 10-20 candles)
        recent_bars = 10
        if len(ohlc_df) < recent_bars:
            return None
            
        recent_data = ohlc_df.tail(recent_bars)
        
        if daily_bias == 'bullish':
            # Look for a micro higher low
            lows = recent_data['low'].values
            recent_low_idx = lows.argmin()
            
            # Check if we've made a higher low after the lowest point
            if recent_low_idx < len(lows) - 3:  # Ensure we have some bars after the low
                subsequent_lows = lows[recent_low_idx + 1:]
                if any(low > lows[recent_low_idx] + (lows[recent_low_idx] * 0.0005) for low in subsequent_lows):
                    # We have a higher low structure
                    return {
                        'type': 'structure_continuation',
                        'level': lows[recent_low_idx],
                        'signal': 'BUY'
                    }
        else:  # bearish
            # Look for a micro lower high
            highs = recent_data['high'].values
            recent_high_idx = highs.argmax()
            
            if recent_high_idx < len(highs) - 3:
                subsequent_highs = highs[recent_high_idx + 1:]
                if any(high < highs[recent_high_idx] - (highs[recent_high_idx] * 0.0005) for high in subsequent_highs):
                    return {
                        'type': 'structure_continuation',
                        'level': highs[recent_high_idx],
                        'signal': 'SELL'
                    }
        
        return None
    
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
    
    def _check_structure_alignment(self, structure, daily_bias):
        """Check if recent structure breaks align with daily bias."""
        if structure.empty:
            return False
        
        # Look for recent BOS/CHoCH 
        recent_structure = structure.tail(self.config.STRUCTURE_LOOKBACK)
        
        if daily_bias == 'bullish':
            # Need bullish BOS or CHoCH
            bullish_bos = recent_structure[
                (recent_structure['BOS'] == 1) | 
                (recent_structure['CHOCH'] == 1)
            ]
            return not bullish_bos.empty
            
        elif daily_bias == 'bearish':
            # Need bearish BOS or CHoCH
            bearish_bos = recent_structure[
                (recent_structure['BOS'] == -1) | 
                (recent_structure['CHOCH'] == -1)
            ]
            return not bearish_bos.empty
        
        return False
    
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
    
    def _check_order_block_entry(self, current_price, order_blocks, bias, manipulation):
        """Check if price is at a valid order block."""
        for ob in order_blocks:
            # OB type must match bias
            if (bias == 'bullish' and ob['type'] == 'bullish') or \
               (bias == 'bearish' and ob['type'] == 'bearish'):
                # Price must be within OB range
                if ob['bottom'] <= current_price <= ob['top']:
                    # OB must be after manipulation for validity
                    if ob['index'] > manipulation.get('index', -1):
                        return ob
        return None
    
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