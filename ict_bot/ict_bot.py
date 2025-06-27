"""
ICT Trading Engine 
"""

import pandas as pd
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
        
    def analyze(self, ohlc_df: pd.DataFrame, symbol: str) -> Dict:
        """
        Perform comprehensive ICT analysis following the narrative sequence.
        Returns a complete analysis dictionary with all ICT concepts.
        """
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback:
            logger.warning(f"{symbol}: Insufficient data for ICT analysis")
            return {}
        
        # Step 1: Get market structure (swings, BOS, CHoCH)
        swings = self._get_swings(ohlc_df)
        structure = self._get_structure(ohlc_df, swings)
        
        # Step 2: Analyze Sessions and determine session-based PO3 and Daily Bias
        session_analysis = self._analyze_session(ohlc_df)
        session_ranges = self._get_session_ranges(ohlc_df)
        daily_bias, po3_analysis = self._analyze_session_po3(ohlc_df, session_ranges, session_analysis)

        # Manipulation detection is now integrated within the PO3 analysis
        manipulation = po3_analysis.get('manipulation', {
            'detected': False, 'type': None, 'level': None, 'judas_swing': None, 'liquidity_sweep': None
        })
        
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
            'session': session_analysis,
            'htf_levels': htf_levels
        }
        
        # Cache this analysis
        self._last_analysis_cache[symbol] = analysis_result
        
        return analysis_result
    
    def _get_swings(self, ohlc_df):
        """Identify swing highs and lows."""
        try:
            swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
            return swings
        except Exception as e:
            logger.error(f"Error getting swings: {e}")
            return pd.DataFrame()
    
    def _get_structure(self, ohlc_df, swings):
        """Analyze market structure (BOS/CHoCH)."""
        if swings.empty:
            return pd.DataFrame()
            
        try:
            structure = smc.bos_choch(ohlc_df, swings, close_break=True)
            return structure
        except Exception as e:
            logger.error(f"Error getting structure: {e}")
            return pd.DataFrame()
        
    def _get_session_ranges(self, ohlc_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Identify the last completed Asian session range.
        Updated to use the FULL Tokyo session (00:00-09:00 UTC) for accumulation range
        """
        ranges: Dict[str, dict] = {}
        
        try:
            # Use the FULL Tokyo session for Asian accumulation (not just the kill zone)
            # This gives us the complete overnight range
            sess = smc.sessions(ohlc_df, session="Tokyo", time_zone="UTC")
            if sess is None:
                return ranges

            active = sess['Active'] == 1
            flips = active.astype(int).diff().fillna(0)
            
            starts = flips[flips == 1].index
            ends = flips[flips == -1].index

            if starts.empty:
                return ranges

            # Handle ongoing session
            if not ends.empty:
                if starts[-1] > ends[-1]:
                    # Session is ongoing, use current time as temporary end
                    last_start_time = starts[-1]
                    last_end_time = ohlc_df.index[-1]
                    
                    # Only use if session has been active for at least 2 hours
                    session_duration = (last_end_time - last_start_time).total_seconds() / 3600
                    if session_duration >= 2:
                        session_df = ohlc_df.loc[last_start_time:last_end_time]
                        ranges["Asia_ongoing"] = {
                            'start': last_start_time,
                            'end': last_end_time,
                            'high': session_df['high'].max(),
                            'low': session_df['low'].min(),
                            'is_ongoing': True
                        }
                    
                    # Also check for last completed session
                    if len(starts) > 1:
                        starts = starts[:-1]
                    else:
                        return ranges

            # Get last completed session
            if not ends.empty:
                last_start_time = starts[-1]
                corresponding_ends = ends[ends > last_start_time]
                if corresponding_ends.empty:
                    return ranges
                last_end_time = corresponding_ends[0]

                session_df = ohlc_df.loc[last_start_time:last_end_time]

                ranges["Asia"] = {
                    'start': last_start_time,
                    'end': last_end_time,
                    'high': session_df['high'].max(),
                    'low': session_df['low'].min(),
                    'is_ongoing': False
                }
                
        except Exception as e:
            logger.error(f"Error getting session range: {e}", exc_info=True)

        return ranges
    
    def _analyze_session_po3(self, ohlc_df: pd.DataFrame, session_ranges: Dict, session_analysis: Dict) -> Tuple[str, Dict]:
        """
        Determine daily bias and Power of Three phase based on session dynamics.
        """
        daily_bias = 'neutral'
        po3_analysis = {
            'current_phase': 'unknown',
            'type': 'unknown',
            'manipulation': {'detected': False, 'type': None, 'level': None}
        }

        if 'Asia' not in session_ranges:
            po3_analysis['reason'] = "No recent Asian session range found."
            return daily_bias, po3_analysis
        
        asian_range = session_ranges['Asia']
        po3_analysis['accumulation_range'] = {
            'high': asian_range['high'], 'low': asian_range['low']
        }
        po3_analysis['current_phase'] = 'accumulation' # Default state

        # Define the search window for manipulation (from end of Asia to now)
        search_window = ohlc_df.loc[asian_range['end']:]
        if len(search_window) < 2:
            po3_analysis['reason'] = "Not enough data after Asian session to check for manipulation."
            return daily_bias, po3_analysis

        # Check for Manipulation (Judas Swing) of Asian Range
        asian_high = asian_range['high']
        asian_low = asian_range['low']
        
        # Check for Bearish Judas Swing (sweep of Asian high)
        sweep_up = search_window[search_window['high'] > asian_high]
        if not sweep_up.empty:
            sweep_candle_time = sweep_up.index[0]
            price_after_sweep = search_window.loc[sweep_candle_time:]
            # Confirm reversal back below the Asian high
            if not price_after_sweep[price_after_sweep['close'] < asian_high].empty:
                daily_bias = 'bearish'
                po3_analysis['type'] = 'bearish_po3'
                po3_analysis['current_phase'] = 'distribution'
                po3_analysis['manipulation'] = {
                    'detected': True,
                    'type': 'bearish_judas',
                    'level': sweep_up['high'].iloc[0],
                    'swept_level': asian_high,
                    'index': ohlc_df.index.get_loc(sweep_candle_time)
                }

        # Check for Bullish Judas Swing (sweep of Asian low)
        sweep_down = search_window[search_window['low'] < asian_low]
        if not sweep_down.empty:
            sweep_candle_time = sweep_down.index[0]
            price_after_sweep = search_window.loc[sweep_candle_time:]
            # Confirm reversal back above the Asian low
            if not price_after_sweep[price_after_sweep['close'] > asian_low].empty:
                # If a bearish signal was also found, we have conflicting signals
                if daily_bias == 'bearish':
                     po3_analysis['reason'] = "Conflicting signals: both Asian high and low swept."
                     return 'neutral', po3_analysis

                daily_bias = 'bullish'
                po3_analysis['type'] = 'bullish_po3'
                po3_analysis['current_phase'] = 'distribution'
                po3_analysis['manipulation'] = {
                    'detected': True,
                    'type': 'bullish_judas',
                    'level': sweep_down['low'].iloc[0],
                    'swept_level': asian_low,
                    'index': ohlc_df.index.get_loc(sweep_candle_time)
                }
        
        # Refine current phase if manipulation has not yet led to distribution
        if po3_analysis['manipulation']['detected']:
             # If price is still hovering around the manipulation area, we're in manipulation phase
             if session_analysis['killzone_name'] == 'london':
                 po3_analysis['current_phase'] = 'manipulation'
        elif session_analysis['session_name'] in ['London', 'New York']:
             po3_analysis['current_phase'] = 'manipulation' # Awaiting manipulation
        
        return daily_bias, po3_analysis
            
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
            # Dynamic range calculation
            atr = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14).iloc[-1]
            range_percent = min(0.002, atr / ohlc_df['close'].iloc[-1])
            
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
    
    def _analyze_session(self, ohlc_df):
        """Analyze current trading session and kill zones."""
        current_time = ohlc_df.index[-1]
        current_hour = current_time.hour
        
        session_info = {
            'current_time': current_time,
            'in_session': False,
            'session_name': None,
            'in_killzone': False,
            'killzone_name': None
        }
        
        # Check ICT kill zones (PRIMARY - this is what matters for ICT)
        for zone_name, zone_times in self.config.ICT_SESSIONS.items():
            if zone_times['start'] <= current_hour < zone_times['end']:
                session_info['in_killzone'] = True
                session_info['killzone_name'] = zone_name.lower()  # Make lowercase for consistency
                break
        
        # Check which full session we're in
        for session_name, session_times in self.config.FULL_SESSIONS.items():
            start = session_times['start']
            end = session_times['end']
            
            # Handle sessions that cross midnight
            if start > end:
                if current_hour >= start or current_hour < end:
                    session_info['in_session'] = True
                    session_info['session_name'] = session_name
                    break
            else:
                if start <= current_hour < end:
                    session_info['in_session'] = True
                    session_info['session_name'] = session_name
                    break
        
        return session_info
    
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
        Enhanced entry location validation that considers:
        1. Longer-term structure (not just 20 candles)
        2. The fact that price often breaks recent highs/lows
        3. Actual risk:reward based on realistic targets
        """
        # Calculate ATR for context
        atr = self._calculate_atr_buffer(ohlc_df)
        
        # Get both short-term and longer-term levels
        recent_high = ohlc_df['high'].tail(20).max()
        recent_low = ohlc_df['low'].tail(20).min()
        
        # Also check 50-period for broader context
        broader_high = ohlc_df['high'].tail(50).max()
        broader_low = ohlc_df['low'].tail(50).min()
        
        if daily_bias == 'bullish':
            # Check if we're buying at extreme highs (use broader context)
            distance_from_high = broader_high - entry_price
            if distance_from_high < atr * 0.25:  # Very close to 50-period high
                # This is OK in strong trends - just log it
                logger.info(f"Entry near 50-period highs - strong trend scenario")
            
            # Calculate realistic reward potential
            risk = entry_price - sl_price
            
            # Method 1: Check if there's expansion potential beyond recent structure
            # In ICT, we expect price to seek liquidity beyond obvious levels
            expected_expansion = recent_high + (atr * 1.5)  # Expect at least 1.5 ATR beyond highs
            potential_reward = expected_expansion - entry_price
            
            # Method 2: Use the actual range for projection
            recent_range = recent_high - recent_low
            projected_target = recent_high + (recent_range * 0.5)  # 50% expansion
            alternative_reward = projected_target - entry_price
            
            # Use the more conservative of the two
            best_reward = min(potential_reward, alternative_reward)
            
            if best_reward < risk * 1.0:  # Reduced from 1.5 to 1.0
                # Only reject if we're also at extreme highs
                if distance_from_high < atr * 0.5:
                    logger.warning(f"Entry at extreme with limited reward potential")
                    return False
                # Otherwise, it's acceptable - maybe catching a pullback in trend
                
        else:  # bearish
            # Check if we're selling at extreme lows
            distance_from_low = entry_price - broader_low
            if distance_from_low < atr * 0.25:
                logger.info(f"Entry near 50-period lows - strong trend scenario")
            
            # Calculate realistic reward potential
            risk = sl_price - entry_price
            
            # Method 1: Expected expansion beyond structure
            expected_expansion = recent_low - (atr * 1.5)
            potential_reward = entry_price - expected_expansion
            
            # Method 2: Range projection
            recent_range = recent_high - recent_low
            projected_target = recent_low - (recent_range * 0.5)
            alternative_reward = entry_price - projected_target
            
            # Use the more conservative
            best_reward = min(potential_reward, alternative_reward)
            
            if best_reward < risk * 1.0:
                if distance_from_low < atr * 0.5:
                    logger.warning(f"Entry at extreme with limited reward potential")
                    return False
        
        # Additional safety check: Ensure stop loss isn't too far
        if risk > atr * 3:
            logger.warning(f"Stop loss too far ({risk:.5f} vs {atr*3:.5f} max)")
            return False
        
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