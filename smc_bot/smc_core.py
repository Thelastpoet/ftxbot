"""
Smart Money Concepts Analyzer 
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from talib import ATR
from smartmoneyconcepts import smc
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

@dataclass
class SMCSetup:
    """Represents a clean SMC trading setup"""
    signal: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_reason: str  # Simple description of setup
    confidence: float  # 0-1 score based on confluences
    risk_reward: float

class SMCAnalyzer:
    """
    Simplified SMC Analyzer focusing on core concepts that work in Forex.
    Removes ICT complexity and manipulation theories.
    """
    
    def __init__(self, config):
        self.config = config
        self.swing_lookback = config.SWING_LOOKBACK
        self.atr_period = config.ATR_PERIOD
        
    def analyze_market_structure(self, ohlc_df: pd.DataFrame) -> Dict:
        """
        Analyze market structure using BOS and CHoCH.
        Returns current trend and strength.
        """
        # Get swing points
        swings = smc.swing_highs_lows(ohlc_df, swing_length=self.swing_lookback)
        if swings.empty:
            return {'trend': 'neutral', 'strength': 0}
        
        # Get structure breaks
        structure = smc.bos_choch(ohlc_df, swings, close_break=True)
        if structure.empty:
            return {'trend': 'neutral', 'strength': 0}
        
        # Analyze recent structure
        recent_bos = structure[structure['BOS'].notna()].tail(5)
        recent_choch = structure[structure['CHOCH'].notna()].tail(3)
        
        # Count bullish vs bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        if not recent_bos.empty:
            bullish_signals += len(recent_bos[recent_bos['BOS'] == 1])
            bearish_signals += len(recent_bos[recent_bos['BOS'] == -1])
        
        if not recent_choch.empty:
            # CHoCH signals are weighted more as they indicate trend change
            bullish_signals += len(recent_choch[recent_choch['CHOCH'] == 1]) * 2
            bearish_signals += len(recent_choch[recent_choch['CHOCH'] == -1]) * 2
        
        # Determine trend
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            return {'trend': 'neutral', 'strength': 0}
        
        if bullish_signals > bearish_signals:
            trend = 'bullish'
            strength = bullish_signals / total_signals
        elif bearish_signals > bullish_signals:
            trend = 'bearish'
            strength = bearish_signals / total_signals
        else:
            trend = 'neutral'
            strength = 0
        
        return {
            'trend': trend,
            'strength': strength,
            'last_bos': recent_bos.iloc[-1] if not recent_bos.empty else None,
            'last_choch': recent_choch.iloc[-1] if not recent_choch.empty else None,
            'swings': swings
        }
    
    def find_order_blocks(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame, trend: str) -> List[Dict]:
        """
        Find valid order blocks in the direction of the trend.
        """
        try:
            obs = smc.ob(ohlc_df, swings, close_mitigation=False)
            if obs.empty:
                return []
            
            # Get unmitigated OBs in trend direction
            recent_obs = obs[obs.index >= len(ohlc_df) - 100]  # Last 100 candles
            
            if trend == 'bullish':
                valid_obs = recent_obs[
                    (recent_obs['OB'] == 1) & 
                    (recent_obs['MitigatedIndex'].isna())
                ]
            elif trend == 'bearish':
                valid_obs = recent_obs[
                    (recent_obs['OB'] == -1) & 
                    (recent_obs['MitigatedIndex'].isna())
                ]
            else:
                return []
            
            # Convert to list and sort by recency
            ob_list = []
            current_price = ohlc_df['close'].iloc[-1]
            
            for idx, ob in valid_obs.iterrows():
                # Calculate distance from current price
                ob_mid = (ob['Top'] + ob['Bottom']) / 2
                distance_pct = abs(current_price - ob_mid) / current_price * 100
                
                # Only include OBs within reasonable distance (e.g., 2%)
                if distance_pct <= 2.0:
                    ob_list.append({
                        'index': idx,
                        'type': 'bullish' if ob['OB'] == 1 else 'bearish',
                        'top': ob['Top'],
                        'bottom': ob['Bottom'],
                        'mid': ob_mid,
                        'distance_pct': distance_pct,
                        'strength': ob.get('Percentage', 50) / 100  # Normalize to 0-1
                    })
            
            # Sort by distance (closest first)
            ob_list.sort(key=lambda x: x['distance_pct'])
            return ob_list[:3]  # Return top 3 closest OBs
            
        except Exception as e:
            logger.error(f"Error finding order blocks: {e}")
            return []
    
    def find_liquidity_zones(self, ohlc_df: pd.DataFrame, swings: pd.DataFrame) -> List[Dict]:
        """
        Find unswept liquidity zones (equal highs/lows).
        These are key targets for smart money.
        """
        try:
            # Dynamic range based on recent volatility
            recent_range = ohlc_df['high'].tail(50).max() - ohlc_df['low'].tail(50).min()
            range_percent = min(0.002, recent_range * 0.01 / ohlc_df['close'].iloc[-1])
            
            liquidity = smc.liquidity(ohlc_df, swings, range_percent=range_percent)
            if liquidity.empty:
                return []
            
            # Get unswept liquidity
            recent_liq = liquidity[liquidity.index >= len(ohlc_df) - 100]
            unswept = recent_liq[
                recent_liq['Liquidity'].notna() & 
                (recent_liq['Swept'].isna() | (recent_liq['Swept'] == 0))
            ]
            
            # Convert to list
            liq_list = []
            current_price = ohlc_df['close'].iloc[-1]
            
            for idx, liq in unswept.iterrows():
                level = liq['Level']
                distance_pct = abs(current_price - level) / current_price * 100
                
                # Only include liquidity within 3%
                if distance_pct <= 3.0:
                    liq_list.append({
                        'index': idx,
                        'type': 'buy_side' if liq['Liquidity'] == 1 else 'sell_side',
                        'level': level,
                        'distance_pct': distance_pct
                    })
            
            return sorted(liq_list, key=lambda x: x['distance_pct'])[:5]
            
        except Exception as e:
            logger.error(f"Error finding liquidity: {e}")
            return []
        
    def _get_rejection_candle(self, last_candle: pd.Series, poi: Dict, direction: str) -> bool:
        """
        Checks if the last candle is a rejection candle from a Point of Interest (POI).
        A POI is a dictionary with 'top' and 'bottom' keys, like an order block.
        """
        if direction == 'BUY':
            # For a bullish OB, price should dip into it and get rejected (close higher).
            touched_poi = last_candle['low'] <= poi['top']
            is_rejection = last_candle['close'] > last_candle['open'] # Bullish candle
            # Optional: ensure close is in upper half of candle for stronger signal
            strong_rejection = last_candle['close'] > (last_candle['high'] + last_candle['low']) / 2
            return touched_poi and is_rejection and strong_rejection
            
        elif direction == 'SELL':
            # For a bearish OB, price should spike into it and get rejected (close lower).
            touched_poi = last_candle['high'] >= poi['bottom']
            is_rejection = last_candle['close'] < last_candle['open'] # Bearish candle
            # Optional: ensure close is in lower half of candle for stronger signal
            strong_rejection = last_candle['close'] < (last_candle['high'] + last_candle['low']) / 2
            return touched_poi and is_rejection and strong_rejection
            
        return False
    
    def find_fair_value_gaps(self, ohlc_df: pd.DataFrame, trend: str) -> List[Dict]:
        """Find unmitigated FVGs in the direction of the trend."""
        try:
            fvgs = smc.fvg(ohlc_df, join_consecutive=True)
            if fvgs.empty:
                return []
            
            # Get recent unmitigated FVGs
            recent_fvgs = fvgs[fvgs.index >= len(ohlc_df) - 50]
            
            if trend == 'bullish':
                valid_fvgs = recent_fvgs[
                    (recent_fvgs['FVG'] == 1) & 
                    (recent_fvgs['MitigatedIndex'].isna())
                ]
            elif trend == 'bearish':
                valid_fvgs = recent_fvgs[
                    (recent_fvgs['FVG'] == -1) & 
                    (recent_fvgs['MitigatedIndex'].isna())
                ]
            else:
                return []
            
            # Convert to list
            fvg_list = []
            current_price = ohlc_df['close'].iloc[-1]
            
            for idx, fvg in valid_fvgs.iterrows():
                fvg_mid = (fvg['Top'] + fvg['Bottom']) / 2
                gap_size = abs(fvg['Top'] - fvg['Bottom'])
                
                # Check if price can reach FVG
                if trend == 'bullish' and fvg['Bottom'] <= current_price <= fvg['Top']:
                    fvg_list.append({
                        'index': idx,
                        'type': 'bullish',
                        'top': fvg['Top'],
                        'bottom': fvg['Bottom'],
                        'mid': fvg_mid,
                        'size': gap_size,
                        'touched': True
                    })
                elif trend == 'bearish' and fvg['Bottom'] <= current_price <= fvg['Top']:
                    fvg_list.append({
                        'index': idx,
                        'type': 'bearish',
                        'top': fvg['Top'],
                        'bottom': fvg['Bottom'],
                        'mid': fvg_mid,
                        'size': gap_size,
                        'touched': True
                    })
            
            return fvg_list[:3]  # Return top 3 FVGs
            
        except Exception as e:
            logger.error(f"Error finding FVGs: {e}")
            return []
    
    def calculate_trade_levels(self, entry_price: float, direction: str, 
                            liquidity_zones: List[Dict], 
                            atr: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit based on market structure.
        Uses liquidity zones as targets and structure for stops.
        """
        # Stop loss: Beyond recent structure or 1.5 ATR
        sl_buffer = atr * 1.5
        
        if direction == 'BUY':
            sl_price = entry_price - sl_buffer
            
            # Find sell-side liquidity above for TP
            tp_candidates = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'sell_side' and liq['level'] > entry_price
            ]
            
            if tp_candidates:
                tp_price = min(tp_candidates)  # Nearest liquidity
            else:
                tp_price = entry_price + (sl_buffer * 2.0)  # 2:1 default
                
        else:  # SELL
            sl_price = entry_price + sl_buffer
            
            # Find buy-side liquidity below for TP
            tp_candidates = [
                liq['level'] for liq in liquidity_zones 
                if liq['type'] == 'buy_side' and liq['level'] < entry_price
            ]
            
            if tp_candidates:
                tp_price = max(tp_candidates)  # Nearest liquidity
            else:
                tp_price = entry_price - (sl_buffer * 2.0)  # 2:1 default
        
        return sl_price, tp_price
    
    def generate_smc_signal(self, ohlc_df: pd.DataFrame, symbol: str) -> Optional[SMCSetup]:
        """
        Generate trading signal based on a REACTION from a core SMC level.
        """
        if len(ohlc_df) < 100:
            return None
        
        # 1. Analyze market structure for overall trend
        structure = self.analyze_market_structure(ohlc_df)
        if structure['trend'] == 'neutral' or structure['strength'] < 0.6:
            logger.debug(f"{symbol}: No clear trend (strength: {structure.get('strength', 0):.2f})")
            return None
        
        trend = structure['trend']
        swings = structure['swings']
        
        # 2. Find key SMC levels
        order_blocks = self.find_order_blocks(ohlc_df, swings, trend)
        liquidity_zones = self.find_liquidity_zones(ohlc_df, swings)
        
        # 3. Calculate ATR for stop loss buffer
        try:
            atr_series = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=self.atr_period)
            atr = atr_series.iloc[-1]
            if np.isnan(atr) or atr == 0:
                logger.warning(f"{symbol}: Invalid ATR value ({atr}). Skipping analysis.")
                return None
        except Exception as e:
            logger.error(f"{symbol}: Failed to calculate ATR: {e}")
            return None
        
        last_candle = ohlc_df.iloc[-1]
        setup = None
        
        # 4. NEW LOGIC: Look for a REJECTION from a Point of Interest (POI)
        if order_blocks:
            # Check the most recent, relevant order block
            ob = order_blocks[0] 
            
            # Check for a bullish rejection from a bullish OB
            if trend == 'bullish' and self._get_rejection_candle(last_candle, ob, 'BUY'):
                confidence = 0.8 + (0.1 * structure['strength'])
                entry_price = last_candle['close']
                
                sl_price, tp_price = self.calculate_trade_levels(
                    entry_price, 'BUY', liquidity_zones, atr, last_candle
                )
                
                # Ensure SL is below the OB bottom for safety
                sl_price = min(sl_price, ob['bottom'] - (atr * 0.1))
                
                setup = SMCSetup(
                    signal='BUY',
                    entry_price=entry_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    entry_reason=f"Bullish OB Rejection (conf: {confidence:.2f})",
                    confidence=confidence,
                    risk_reward=abs(tp_price - entry_price) / abs(entry_price - sl_price) if abs(entry_price-sl_price) > 0 else 0
                )
            
            # Check for a bearish rejection from a bearish OB
            elif trend == 'bearish' and self._get_rejection_candle(last_candle, ob, 'SELL'):
                confidence = 0.8 + (0.1 * structure['strength'])
                entry_price = last_candle['close']
                
                sl_price, tp_price = self.calculate_trade_levels(
                    entry_price, 'SELL', liquidity_zones, atr, last_candle
                )

                # Ensure SL is above the OB top for safety
                sl_price = max(sl_price, ob['top'] + (atr * 0.1))
                
                setup = SMCSetup(
                    signal='SELL',
                    entry_price=entry_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    entry_reason=f"Bearish OB Rejection (conf: {confidence:.2f})",
                    confidence=confidence,
                    risk_reward=abs(entry_price - tp_price) / abs(sl_price - entry_price) if abs(sl_price-entry_price) > 0 else 0
                )

        # 5. Validate setup against config rules
        if setup:
            # Minimum confidence threshold
            if setup.confidence < self.config.MIN_CONFIDENCE:
                logger.debug(f"{symbol}: Setup confidence too low ({setup.confidence:.2f})")
                return None
            
            # Minimum RR ratio
            if setup.risk_reward < self.config.MIN_RR_RATIO:
                logger.debug(f"{symbol}: Risk-reward too low ({setup.risk_reward:.2f})")
                return None
            
            logger.info(f"{symbol}: SMC Setup found - {setup.entry_reason}")
            return setup
        
        return None

class MultiTimeframeSMC:
    """
    Handles multi-timeframe SMC analysis for better trade confirmation.
    """
    
    def __init__(self, mt5_client, analyzer: SMCAnalyzer):
        self.mt5_client = mt5_client
        self.analyzer = analyzer
        self.timeframes = {
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
    
    def get_mtf_bias(self, symbol: str) -> Dict[str, str]:
        """
        Get market bias across multiple timeframes.
        Higher timeframes have more weight.
        """
        biases = {}
        
        for tf_name, tf_const in self.timeframes.items():
            # Fetch data for each timeframe
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 150)
            
            if rates is None or len(rates) < 100:
                logger.warning(f"{symbol}: Insufficient data for {tf_name}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Analyze structure
            structure = self.analyzer.analyze_market_structure(df)
            biases[tf_name] = structure['trend']
        
        return biases
    
    def confirm_trade_with_mtf(self, symbol: str, signal: str) -> bool:
        """
        Confirm trade signal with higher timeframe alignment.
        Returns True if at least H1 agrees with the signal.
        """
        mtf_biases = self.get_mtf_bias(symbol)
        
        # Convert signal to trend
        signal_trend = 'bullish' if signal == 'BUY' else 'bearish'
        
        # Check H1 and H4 alignment
        h1_aligned = mtf_biases.get('H1') == signal_trend
        h4_aligned = mtf_biases.get('H4') == signal_trend
        
        # Need at least H1 alignment
        if not h1_aligned:
            logger.info(f"{symbol}: MTF not aligned - H1: {mtf_biases.get('H1')}, "
                       f"H4: {mtf_biases.get('H4')}, Signal: {signal_trend}")
            return False
        
        # Bonus if H4 also aligned
        if h4_aligned:
            logger.info(f"{symbol}: Strong MTF alignment across all timeframes")
        
        return True


class SMCTradingSystem:
    """
    Complete SMC trading system integrating analyzer with MT5.
    """
    
    def __init__(self, mt5_client, config):
        self.mt5_client = mt5_client
        self.config = config
        self.analyzer = SMCAnalyzer(config)
        self.mtf_analyzer = MultiTimeframeSMC(mt5_client, self.analyzer)
        
    def analyze_and_trade(self, symbol: str, ohlc_df: pd.DataFrame, 
                         position_sizer, trade_executor) -> Optional[Dict]:
        """
        Complete trading workflow: analyze, confirm, and execute.
        """
        # 1. Generate SMC signal
        setup = self.analyzer.generate_smc_signal(ohlc_df, symbol)
        if not setup:
            return None
        
        # 2. Confirm with multi-timeframe analysis
        if not self.mtf_analyzer.confirm_trade_with_mtf(symbol, setup.signal):
            logger.info(f"{symbol}: Signal rejected by MTF analysis")
            return None
        
        # 3. Calculate position size
        volume = position_sizer.calculate_volume(symbol, setup.stop_loss, setup.signal)
        if not volume:
            logger.error(f"{symbol}: Failed to calculate position size")
            return None
        
        # 4. Execute trade
        comment = f"SMC_{setup.entry_reason[:20]}"  # Truncate for MT5 limit
        result = trade_executor.place_market_order(
            symbol, setup.signal, volume, 
            setup.stop_loss, setup.take_profit, comment
        )
        
        if result:
            logger.info(f"{symbol}: Trade executed - {setup.signal} @ {setup.entry_price}")
            logger.info(f"  SL: {setup.stop_loss}, TP: {setup.take_profit}, RR: {setup.risk_reward:.2f}")
            return {
                'symbol': symbol,
                'setup': setup,
                'volume': volume,
                'result': result
            }
        
        return None