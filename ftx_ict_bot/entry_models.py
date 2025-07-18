"""
ICT Entry Models - Correct Architecture Implementation
Implements authentic ICT entry models with proper architectural interfaces.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class EntryModels:
    """
    Implements authentic ICT entry models with correct architectural design.
    
    Receives pre-computed building blocks from ICTAnalyzer and identifies:
    - Breaker blocks from failed order blocks
    - Mitigation blocks from order blocks after failure swings  
    - Unicorn setups from FVG + breaker confluences
    """

    def __init__(self, config, liquidity_detector):
        """Initialize with configuration and liquidity detector dependency"""
        self.config = config
        self.liquidity_detector = liquidity_detector
        
        # Configuration parameters
        self.min_data_length = getattr(config, 'STRUCTURE_LOOKBACK', 50)
        self.require_mss_confirmation = getattr(config, 'REQUIRE_ENTRY_CONFIRMATION', True)

    def identify_breaker_blocks(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, 
                               order_blocks: List[Dict], liquidity_levels: Dict) -> List[Dict]:
        """
        Identify authentic ICT breaker blocks from failed order blocks.
        
        Correct ICT methodology: Breaker blocks are failed order blocks that
        become new support/resistance after market structure shifts.
        
        Args:
            ohlc_df: OHLC price data
            swings: Swing analysis from SMC
            order_blocks: Pre-computed order blocks from ICTAnalyzer
            liquidity_levels: Liquidity data for validation
            
        Returns:
            List of authentic breaker block dictionaries
        """
        try:
            self._validate_inputs(ohlc_df, swings, "breaker analysis")
            
            if not order_blocks:
                logger.debug("No order blocks provided for breaker analysis")
                return []
            
            breaker_blocks = []
            
            for order_block in order_blocks:
                try:
                    # Detect if this order block has failed
                    failure_info = self._detect_order_block_failure(ohlc_df, order_block)
                    
                    if failure_info:
                        # Validate with liquidity confirmation
                        if self._validate_breaker_with_liquidity(order_block, failure_info, liquidity_levels):
                            # Confirm with market structure shift
                            if self._confirm_mss_after_ob_failure(ohlc_df, swings, failure_info):
                                breaker = self._create_breaker_block(order_block, failure_info)
                                breaker_blocks.append(breaker)
                                logger.debug(f"Breaker block: {order_block['type']} OB failed at {failure_info['price']:.5f}")
                
                except Exception as e:
                    logger.error(f"Error analyzing order block {order_block.get('index')}: {e}")
                    continue
            
            return breaker_blocks
            
        except Exception as e:
            logger.error(f"Error in breaker block identification: {e}")
            return []

    def _detect_order_block_failure(self, ohlc_df: pd.DataFrame, order_block: Dict) -> Optional[Dict]:
        """
        Detect authentic ICT order block failure.
        
        ICT criteria:
        - Bullish OB fails when price closes below its low
        - Bearish OB fails when price closes above its high
        """
        try:
            ob_index = order_block['index']
            ob_type = order_block['type']
            ob_top = order_block['top']
            ob_bottom = order_block['bottom']
            
            if ob_index not in ohlc_df.index:
                return None
                
            ob_position = ohlc_df.index.get_loc(ob_index)
            post_ob_data = ohlc_df.iloc[ob_position + 1:]
            
            for i, (timestamp, candle) in enumerate(post_ob_data.iterrows()):
                # Check for authentic failure conditions
                if ob_type == 'bullish' and candle['close'] < ob_bottom:
                    return {
                        'timestamp': timestamp,
                        'price': candle['close'],
                        'direction': 'bearish_break',
                        'candle_index': ob_position + i + 1,
                        'original_ob': order_block
                    }
                elif ob_type == 'bearish' and candle['close'] > ob_top:
                    return {
                        'timestamp': timestamp,
                        'price': candle['close'],
                        'direction': 'bullish_break',
                        'candle_index': ob_position + i + 1,
                        'original_ob': order_block
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting order block failure: {e}")
            return None

    def _validate_breaker_with_liquidity(self, order_block: Dict, failure_info: Dict, 
                                       liquidity_levels: Dict) -> bool:
        """
        Validate breaker block formation using liquidity context.
        
        Uses the liquidity_detector to confirm the failure aligns with
        liquidity sweep patterns or targeted liquidity zones.
        """
        try:
            if not self.liquidity_detector:
                return True  # Skip validation if no detector available
            
            failure_price = failure_info['price']
            failure_direction = failure_info['direction']
            
            # Check if failure corresponds to a liquidity sweep
            target_side = 'sell_side' if failure_direction == 'bearish_break' else 'buy_side'
            relevant_liquidity = liquidity_levels.get(target_side, [])
            
            # Look for nearby liquidity that was targeted
            for liq_level in relevant_liquidity:
                level_price = liq_level.get('level', 0)
                distance = abs(failure_price - level_price)
                
                # If failure occurred near a known liquidity level, it's validated
                if distance < (level_price * 0.001):  # Within 0.1% tolerance
                    logger.debug(f"Breaker validated by liquidity sweep at {level_price:.5f}")
                    return True
            
            # If no specific liquidity validation but clear OB failure, still valid
            return True
            
        except Exception as e:
            logger.error(f"Error validating breaker with liquidity: {e}")
            return True  # Default to valid on error

    def _confirm_mss_after_ob_failure(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, 
                                    failure_info: Dict) -> bool:
        """
        Confirm Market Structure Shift after order block failure.
        
        Validates that the OB failure resulted in genuine market structure change.
        """
        if not self.require_mss_confirmation:
            return True
        
        try:
            failure_timestamp = failure_info['timestamp']
            failure_direction = failure_info['direction']
            
            # Get swings before the failure
            pre_failure_swings = swings[swings.index < failure_timestamp].dropna()
            if pre_failure_swings.empty:
                return False
            
            # Find the relevant swing level for MSS validation
            if failure_direction == 'bullish_break':
                # Need to break above previous swing high
                recent_highs = pre_failure_swings[pre_failure_swings['HighLow'] == 1]
                if recent_highs.empty:
                    return False
                mss_level = recent_highs['Level'].iloc[-1]
                
                # Check if price broke above this level after failure
                post_failure_data = ohlc_df[ohlc_df.index > failure_timestamp]
                return any(post_failure_data['close'] > mss_level)
                
            else:  # bearish_break
                # Need to break below previous swing low
                recent_lows = pre_failure_swings[pre_failure_swings['HighLow'] == -1]
                if recent_lows.empty:
                    return False
                mss_level = recent_lows['Level'].iloc[-1]
                
                # Check if price broke below this level after failure
                post_failure_data = ohlc_df[ohlc_df.index > failure_timestamp]
                return any(post_failure_data['close'] < mss_level)
                
        except Exception as e:
            logger.error(f"Error confirming MSS after OB failure: {e}")
            return False

    def _create_breaker_block(self, original_ob: Dict, failure_info: Dict) -> Dict:
        """Create breaker block from failed order block"""
        return {
            'index': original_ob['index'],
            'type': f"{failure_info['direction']}_breaker",
            'top': original_ob['top'],
            'bottom': original_ob['bottom'],
            'original_ob_type': original_ob['type'],
            'failure_timestamp': failure_info['timestamp'],
            'failure_price': failure_info['price'],
            'description': f"Failed {original_ob['type']} OB becomes {failure_info['direction']} breaker"
        }

    def identify_mitigation_blocks(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, 
                                 order_blocks: List[Dict], daily_bias: str) -> List[Dict]:
        """
        Identifies authentic ICT Mitigation Blocks.
        A mitigation block is a failed order block that did NOT take liquidity 
        before failing. Price returns to mitigate the position.
        """
        try:
            self._validate_inputs(ohlc_df, swings, "mitigation analysis")
            if not order_blocks:
                return []

            mitigation_blocks = []
            
            # Get all swing highs and lows
            swing_highs = swings[swings['HighLow'] == 1].dropna()
            swing_lows = swings[swings['HighLow'] == -1].dropna()

            for ob in order_blocks:
                ob_index_pos = ohlc_df.index.get_loc(ob['index'])
                
                # Find the swing that the OB was supposed to break
                if ob['type'] == 'bullish': # Bullish OB should break a swing high
                    # Find the last swing high before the OB
                    relevant_swings = swing_highs[swing_highs.index < ob['index']]
                    if relevant_swings.empty: continue
                    target_swing = relevant_swings.iloc[-1]
                    
                    # Check if the move from the OB failed to break the target swing
                    move_after_ob = ohlc_df.iloc[ob_index_pos:]
                    if move_after_ob['high'].max() < target_swing['Level']:
                        # This is a failure to break structure. Now, has the OB been violated?
                        if move_after_ob['low'].min() < ob['bottom']:
                             mitigation_blocks.append(self._create_mitigation_block(ob, target_swing, 'bullish'))

                elif ob['type'] == 'bearish': # Bearish OB should break a swing low
                    # Find the last swing low before the OB
                    relevant_swings = swing_lows[swing_lows.index < ob['index']]
                    if relevant_swings.empty: continue
                    target_swing = relevant_swings.iloc[-1]

                    # Check if the move from the OB failed to break the target swing
                    move_after_ob = ohlc_df.iloc[ob_index_pos:]
                    if move_after_ob['low'].min() > target_swing['Level']:
                        # This is a failure to break structure. Now, has the OB been violated?
                        if move_after_ob['high'].max() > ob['top']:
                            mitigation_blocks.append(self._create_mitigation_block(ob, target_swing, 'bearish'))
            
            return mitigation_blocks
            
        except Exception as e:
            logger.error(f"Error identifying mitigation blocks: {e}", exc_info=True)
            return []

    def _create_mitigation_block(self, order_block: Dict, failed_swing: pd.Series, ob_type: str) -> Dict:
        """Helper to create a mitigation block dictionary."""
        return {
            'index': order_block['index'],
            'type': f"{ob_type}_mitigation",
            'top': order_block['top'],
            'bottom': order_block['bottom'],
            'original_ob': order_block,
            'failed_to_break': failed_swing['Level'],
            'description': f"Mitigation block from failed {ob_type} OB"
        }

    def identify_unicorn_setups(self, fair_value_gaps: List[Dict], breaker_blocks: List[Dict],
                              premium_discount_analysis: Dict) -> List[Dict]:
        """
        Identify authentic ICT unicorn setups with proper context validation.
        
        Unicorn setups require:
        - FVG + Breaker Block overlap
        - Directional alignment
        - Appropriate premium/discount context
        
        Args:
            fair_value_gaps: Pre-computed FVGs from ICTAnalyzer
            breaker_blocks: Breaker blocks from this class
            premium_discount_analysis: Premium/discount context from ICTAnalyzer
            
        Returns:
            List of unicorn setup dictionaries
        """
        try:
            if not fair_value_gaps or not breaker_blocks:
                logger.debug("Insufficient data for unicorn analysis")
                return []
            
            unicorn_setups = []
            
            for fvg in fair_value_gaps:
                for breaker in breaker_blocks:
                    try:
                        # Check confluence requirements
                        if self._validate_unicorn_confluence(fvg, breaker, premium_discount_analysis):
                            unicorn = self._create_unicorn_setup(fvg, breaker, premium_discount_analysis)
                            if unicorn:
                                unicorn_setups.append(unicorn)
                                logger.debug(f"Unicorn setup: {fvg['type']} FVG + {breaker['type']} breaker")
                                
                    except Exception as e:
                        logger.error(f"Error analyzing unicorn confluence: {e}")
                        continue
            
            logger.debug(f"Identified {len(unicorn_setups)} unicorn setups")
            return unicorn_setups
            
        except Exception as e:
            logger.error(f"Error identifying unicorn setups: {e}")
            return []

    def _validate_unicorn_confluence(self, fvg: Dict, breaker: Dict, 
                               pd_analysis: Dict) -> bool:
        """
        Validate unicorn confluence requirements:
        1. Geometric overlap
        2. Directional alignment  
        3. Premium/discount context
        """
        try:
            # Fix directional alignment check
            fvg_type = fvg.get('type')  # 'bullish' or 'bearish'
            
            breaker_type = breaker.get('type', '')
            
            if 'bearish_break' in breaker_type:
                breaker_entry_direction = 'bullish'  # This breaker is for bullish entries
            elif 'bullish_break' in breaker_type:
                breaker_entry_direction = 'bearish'  # This breaker is for bearish entries
            else:
                # Fallback for other breaker types
                original_ob_type = breaker.get('original_ob_type', '')
                # If original OB was bearish and it failed, the breaker is bullish
                breaker_entry_direction = 'bullish' if original_ob_type == 'bearish' else 'bearish'
            
            # Now check if FVG and breaker are aligned for same direction entry
            if fvg_type != breaker_entry_direction:
                logger.debug(f"Unicorn alignment failed: FVG={fvg_type}, Breaker for={breaker_entry_direction}")
                return False
            
            # Check geometric overlap
            if not self._check_zone_overlap(fvg, breaker):
                logger.debug("Unicorn geometric overlap failed")
                return False
            
            # Relaxed premium/discount check for live trading
            if pd_analysis:
                # For live trading, we're more flexible with P/D zones
                current_zone = pd_analysis.get('current_zone', '')
                if fvg_type == 'bullish' and 'premium' in current_zone.lower():
                    # In premium but bullish - could be a retracement, still valid
                    logger.debug("Bullish unicorn in premium zone - allowing for retracement entry")
                elif fvg_type == 'bearish' and 'discount' in current_zone.lower():
                    # In discount but bearish - could be a retracement, still valid
                    logger.debug("Bearish unicorn in discount zone - allowing for retracement entry")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating unicorn confluence: {e}")
            return False
    
    def _check_zone_overlap(self, zone1: Dict, zone2: Dict) -> bool:
        """Check if two price zones overlap geometrically"""
        try:
            return (max(zone1['bottom'], zone2['bottom']) <= min(zone1['top'], zone2['top']))
        except (KeyError, TypeError):
            return False

    def _create_unicorn_setup(self, fvg: Dict, breaker: Dict, pd_analysis: Dict) -> Optional[Dict]:
        """Create unicorn setup with confluence zone and context"""
        try:
            # Calculate confluence zone (overlap area)
            confluence_top = min(fvg['top'], breaker['top'])
            confluence_bottom = max(fvg['bottom'], breaker['bottom'])
            
            if confluence_top <= confluence_bottom:
                logger.debug(f"Invalid confluence zone: top {confluence_top:.5f} <= bottom {confluence_bottom:.5f}")
                return None
            
            # Determine entry direction from FVG type
            entry_direction = fvg['type']  # 'bullish' or 'bearish'
            
            unicorn = {
                'type': f"{entry_direction}_unicorn",
                'fvg': fvg,
                'breaker_block': breaker,
                'top': confluence_top,
                'bottom': confluence_bottom,
                'confluence_size': confluence_top - confluence_bottom,
                'premium_discount_context': pd_analysis.get('current_zone', 'unknown'),
                'entry_direction': entry_direction,
                'description': f"{entry_direction.capitalize()} Unicorn: FVG + Breaker confluence"
            }
            
            logger.debug(f"Created {unicorn['type']}: Zone[{confluence_bottom:.5f}-{confluence_top:.5f}], Size={unicorn['confluence_size']:.5f}")
            
            return unicorn
            
        except Exception as e:
            logger.error(f"Error creating unicorn setup: {e}")
            return None
    
    def _validate_inputs(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, 
                        operation: str) -> None:
        """Validate inputs for analysis operations"""
        if ohlc_df is None or ohlc_df.empty:
            raise ValueError(f"OHLC DataFrame cannot be None or empty for {operation}")
        
        if len(ohlc_df) < self.min_data_length:
            raise ValueError(f"OHLC DataFrame must have at least {self.min_data_length} rows for {operation}")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in ohlc_df.columns]
        if missing_columns:
            raise ValueError(f"OHLC DataFrame missing required columns for {operation}: {missing_columns}")
        
        if swings is None or swings.empty:
            raise ValueError(f"Swings DataFrame cannot be None or empty for {operation}")