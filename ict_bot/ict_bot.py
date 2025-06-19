"""
ICT Trading Engine - The specialized trading brain that follows the ICT narrative.
This module contains the SMC analysis and signal generation following the proper
ICT sequence: Daily Bias → Manipulation → Structure Break → Optimal Entry
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
        
        return {
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
        This version assumes the smc.py library has been fixed to return the correct index.
        """
        ranges: Dict[str, dict] = {}
        sessions_to_check = { "Asia": "Tokyo" }

        for name, smc_name in sessions_to_check.items():
            try:
                # Use smartmoneyconcepts to mark session activity
                sess = smc.sessions(ohlc_df, session=smc_name, time_zone="UTC")
                if sess is None:
                    continue

                active = sess['Active'] == 1
                flips = active.astype(int).diff().fillna(0)
                
                starts = flips[flips == 1].index
                ends = flips[flips == -1].index

                if starts.empty or ends.empty:
                    continue

                if starts[-1] > ends[-1]:
                    starts = starts[:-1] 
                    if starts.empty:
                        continue

                last_start_time = starts[-1]
                corresponding_ends = ends[ends > last_start_time]
                if corresponding_ends.empty:
                    continue
                last_end_time = corresponding_ends[0]

                session_df = ohlc_df.loc[last_start_time:last_end_time]

                ranges[name] = {
                    'start': last_start_time,
                    'end': last_end_time,
                    'high': session_df['high'].max(),
                    'low': session_df['low'].min()
                }
            except Exception as e:
                logger.error(f"Error getting session range for {name}: {e}", exc_info=True)
                continue

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
             if session_analysis['killzone_name'] == 'London Kill Zone':
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
            
            # Get unswept liquidity
            recent_liq = liquidity[liquidity.index >= len(ohlc_df) - 50]
            unswept = recent_liq[
                recent_liq['Liquidity'].notna() & 
                recent_liq['Swept'].isna()
            ]
            
            # Convert to list
            liq_list = []
            for idx, liq in unswept.iterrows():
                liq_list.append({
                    'index': idx,
                    'type': 'bullish' if liq['Liquidity'] == 1 else 'bearish',
                    'level': liq['Level']
                })
            
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
        last_swings = swings.dropna().tail(2)
        if len(last_swings) < 2:
            return {} # Not enough swings to create a range

        # The dealing range is the high and low of the last two swing points
        high = last_swings['Level'].max()
        low = last_swings['Level'].min()
        
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
        
        # Check ICT kill zones
        for zone_name, zone_times in self.config.ICT_SESSIONS.items():
            if zone_times['start'] <= current_hour < zone_times['end']:
                session_info['in_killzone'] = True
                session_info['killzone_name'] = zone_name
                break
        
        # Check general sessions using SMC
        try:
            sessions_to_check = ['London', 'New York', 'Tokyo']
            for session in sessions_to_check:
                result = smc.sessions(ohlc_df, session=session, time_zone="UTC")
                if result is not None and 'Active' in result.columns:
                    if result['Active'].iloc[-1] == 1:
                        session_info['in_session'] = True
                        session_info['session_name'] = session
                        break
        except Exception as e:
            logger.warning(f"Error checking sessions: {e}")
        
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


class ICTSignalGenerator:
    """
    Generates trading signals following the ICT narrative sequence.
    This is the key difference - we follow a story, not a checklist.
    """
    
    def __init__(self, config):
        self.config = config
        self.analyzer = ICTAnalyzer(config)
        
    def generate_signal(self, ohlc_df: pd.DataFrame, symbol: str) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[ICTNarrative]]:
        """
        Generate trading signal following the ICT narrative:
        1. Confirm daily bias (PO3/Market structure)
        2. Confirm manipulation has occurred
        3. Confirm structure break in bias direction
        4. Find optimal entry (OTE/Order Block)
        
        Returns: (signal, sl_price, tp_price, narrative)
        """
        # Get comprehensive analysis
        analysis = self.analyzer.analyze(ohlc_df, symbol)
        if not analysis:
            return None, None, None, None
        
        # Extract key components
        daily_bias = analysis['daily_bias']
        manipulation = analysis['manipulation']
        structure = analysis['structure']
        current_price = analysis['current_price']
        pd_levels = analysis['premium_discount']
        ote_zones = analysis['ote_zones']
        order_blocks = analysis['order_blocks']
        session = analysis['session']
        
        # Get relaxation flags from config. Default to True (strict) if not specified.
        require_structure_alignment = getattr(self.config, 'REQUIRE_STRUCTURE_ALIGNMENT', True)
        require_killzone = getattr(self.config, 'REQUIRE_KILLZONE', True)
        
        # Build the narrative
        narrative = self._build_narrative(analysis)
        
        # === STEP 1: Check Daily Bias ===
        if daily_bias == 'neutral':
            logger.debug(f"{symbol}: No clear daily bias")
            return None, None, None, None
        
        # === STEP 2: Confirm Manipulation ===
        if not manipulation['detected']:
            logger.debug(f"{symbol}: No manipulation detected yet")
            return None, None, None, None
        
        # === STEP 3: Check Market Structure ===
        if require_structure_alignment:
            structure_aligned = self._check_structure_alignment(structure, daily_bias)
            if not structure_aligned:
                logger.debug(f"{symbol}: Structure not aligned with bias.")
                return None, None, None, None
        
        # === STEP 4: Price Location Check ===
        # For longs: Need to be in discount
        # For shorts: Need to be in premium
        if daily_bias == 'bullish' and pd_levels['current_zone'] not in ['discount', 'deep_discount']:
            logger.debug(f"{symbol}: Bullish bias but price not in discount")
            return None, None, None, None
            
        if daily_bias == 'bearish' and pd_levels['current_zone'] not in ['premium', 'deep_premium']:
            logger.debug(f"{symbol}: Bearish bias but price not in premium")
            return None, None, None, None
        
        # === STEP 5: Find Entry Model ===
        entry_signal, sl_price, tp_price = None, None, None
        
        # Model 1: OTE Entry (Primary model after manipulation and structure break)
        if ote_zones and self._is_price_in_ote(current_price, ote_zones, daily_bias):
            logger.info(f"{symbol}: Price in OTE zone after manipulation")
            entry_signal, sl_price, tp_price = self._setup_ote_entry(
                daily_bias, current_price, ote_zones, manipulation, ohlc_df
            )
            narrative.entry_model = "OTE_ENTRY"
        
        # Model 2: Order Block Entry (If we have refined OB at current price)
        elif order_blocks and not entry_signal:
            ob_entry = self._check_order_block_entry(
                current_price, order_blocks, daily_bias, manipulation
            )
            if ob_entry:
                entry_signal, sl_price, tp_price = self._setup_ob_entry(
                    daily_bias, current_price, ob_entry, manipulation, ohlc_df
                )
                narrative.entry_model = "ORDER_BLOCK"
        
        # Model 3: Direct reversal from manipulation (Judas/Sweep reversal)
        elif not entry_signal:
            # Look for the first FVG created after the manipulation
            fvg_after_manipulation = None
            for fvg in analysis.get('fair_value_gaps', []):
                # FVG must be in the direction of the bias and occur after the manipulation event
                if (daily_bias == 'bullish' and fvg['type'] == 'bullish' and fvg['index'] > manipulation.get('index', -1)) or \
                   (daily_bias == 'bearish' and fvg['type'] == 'bearish' and fvg['index'] > manipulation.get('index', -1)):
                    fvg_after_manipulation = fvg
                    break # Use the very first one
            
            if fvg_after_manipulation:
                # Check if the current price has retraced into this FVG
                fvg_top = fvg_after_manipulation['top']
                fvg_bottom = fvg_after_manipulation['bottom']
                
                if fvg_bottom <= current_price <= fvg_top:
                    logger.info(f"{symbol}: Price entered post-manipulation FVG.")
                    if daily_bias == 'bullish':
                        entry_signal = "BUY"
                        # SL goes below the manipulation low, which is the true invalidation level
                        sl_price = manipulation['level'] - self._calculate_atr_buffer(ohlc_df)
                        narrative.entry_model = "FVG_REVERSAL_ENTRY"
                    
                    elif daily_bias == 'bearish':
                        entry_signal = "SELL"
                        # SL goes above the manipulation high
                        sl_price = manipulation['level'] + self._calculate_atr_buffer(ohlc_df)
                        narrative.entry_model = "FVG_REVERSAL_ENTRY"
                    
                    if entry_signal:
                        tp_price = self._calculate_target(current_price, sl_price, daily_bias, analysis)
        
        # Final validation
        if entry_signal:
            # Must be in kill zone for higher probability
            if require_killzone and not session['in_killzone']:
                logger.info(f"{symbol}: Signal found, but outside kill zone - skipping (Requirement is ON).")
                return None, None, None, None
            
            # Validate SL/TP levels
            if not self._validate_levels(entry_signal, current_price, sl_price, tp_price):
                logger.error(f"{symbol}: Invalid SL/TP levels")
                return None, None, None, None
            
            # Complete the narrative
            narrative.entry_model = narrative.entry_model or "ICT_NARRATIVE"
            
            logger.info(f"{symbol}: ICT Signal Generated - {entry_signal}")
            logger.info(f"  Narrative: {daily_bias} bias, {manipulation['type']} manipulation, "
                       f"entry via {narrative.entry_model}")
            logger.info(f"  Levels: Entry={current_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")
            
            return entry_signal, sl_price, tp_price, narrative
        
        return None, None, None, None
    
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
        
        # Look for recent BOS/CHoCH in last 20 candles
        recent_structure = structure.tail(20)
        
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
    
    def _setup_ote_entry(self, bias, current_price, ote_zones, manipulation, ohlc_df):
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
        tp_price = self._calculate_target(current_price, sl_price, bias, {'ote': ote})
        
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
    
    def _setup_ob_entry(self, bias, current_price, order_block, manipulation, ohlc_df):
        """Setup entry from order block."""
        if bias == 'bullish':
            signal = "BUY"
            sl_price = order_block['bottom'] - self._calculate_atr_buffer(ohlc_df)
        else:
            signal = "SELL"
            sl_price = order_block['top'] + self._calculate_atr_buffer(ohlc_df)
        
        tp_price = self._calculate_target(current_price, sl_price, bias, {'ob': order_block})
        
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
        liquidity_zones = analysis.get('liquidity_zones', [])
        
        target_liquidity_level = None

        if bias == 'bullish':
            # For a buy, target the nearest bearish liquidity (a swing high) above the entry.
            potential_targets = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'bearish' and liq['level'] > entry_price
            ]
            if potential_targets:
                # Find the closest high to target
                target_liquidity_level = min(potential_targets)
                logger.debug(f"Liquidity target found for bullish trade: {target_liquidity_level}")

        elif bias == 'bearish':
            # For a sell, target the nearest bullish liquidity (a swing low) below the entry.
            potential_targets = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'bullish' and liq['level'] < entry_price
            ]
            if potential_targets:
                # Find the closest low to target
                target_liquidity_level = max(potential_targets)
                logger.debug(f"Liquidity target found for bearish trade: {target_liquidity_level}")

        # If a liquidity target is found, use it. Otherwise, use the fixed R:R as a fallback.
        if target_liquidity_level:
            # Basic check to ensure the liquidity target offers a reasonable R:R
            potential_reward = abs(target_liquidity_level - entry_price)
            if potential_reward > risk: # Ensure at least 1R
                return target_liquidity_level
            else:
                logger.debug("Liquidity target too close for valid R:R, using fallback.")

        # Fallback to fixed R:R
        logger.info("No suitable liquidity target found, using fixed R:R fallback.")
        tp_rr_ratio = getattr(self.config, 'TP_RR_RATIO', 2.0)
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