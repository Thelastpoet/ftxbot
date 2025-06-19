import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from talib import ATR
from smartmoneyconcepts import smc as smc_package

# Import config directly for components that need it at init or as defaults
import config as global_config

logger = logging.getLogger(__name__)

class MetaTrader5Client:
    """Handles connection and basic interactions with MetaTrader 5."""
    def __init__(self):
        self.initialized = False
        self.connection_start_time = None

    def initialize(self):
        try:
            self.initialized = mt5.initialize()
        except Exception as e:
            logger.error(f"Exception during mt5.initialize(): {e}")
            self.initialized = False
            return False

        if not self.initialized:
            logger.error(f"MT5 initialize() failed. Code: {mt5.last_error()}")
            return False

        self.connection_start_time = datetime.now()

        terminal_info = self.get_terminal_info()
        if not terminal_info:
            logger.error("Failed to get terminal_info. Shutting down MT5 connection.")
            self.shutdown()
            return False

        if not terminal_info.trade_allowed:
            logger.error("Algorithmic trading NOT allowed in MT5. Shutting down.")
            self.shutdown()
            return False

        account_info = self.get_account_info()
        if not account_info:
            logger.error("Failed to get account_info. Shutting down MT5 connection.")
            self.shutdown()
            return False

        logger.info(f"Connected to account: {account_info.login}, Server: {account_info.server}, Bal: {account_info.balance:.2f} {account_info.currency}")
        return True

    def shutdown(self):
        if self.initialized:
            mt5.shutdown()
            self.initialized = False

    def is_connected(self):
        return self.initialized

    def get_account_info(self):
        if not self.initialized: return None
        return mt5.account_info()

    def get_terminal_info(self):
        if not self.initialized: return None
        return mt5.terminal_info()

    def get_symbol_info(self, symbol):
        if not self.initialized: return None
        return mt5.symbol_info(symbol)

    def get_symbol_ticker(self, symbol):
        if not self.initialized: return None
        return mt5.symbol_info_tick(symbol)

    def get_current_positions(self, symbol=None):
        if not self.initialized: return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return positions if positions is not None else []

    def timeframe_to_mql(self, tf_string):
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
        }
        mql_tf = tf_map.get(tf_string.upper())
        if mql_tf is None:
            raise ValueError(f"Unsupported timeframe string: {tf_string}")
        return mql_tf

class MarketDataProvider:
    """Fetches and prepares market data."""
    def __init__(self, mt5_client: MetaTrader5Client):
        self.client = mt5_client

    def get_ohlc(self, symbol, timeframe_str, count):
        if not self.client.is_connected():
            logger.error(f"MarketDataProvider: MT5 not connected for {symbol}.")
            return None
        try:
            timeframe_mql = self.client.timeframe_to_mql(timeframe_str)
        except ValueError as e:
            logger.error(f"MarketDataProvider ({symbol}): {e}")
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe_mql, 0, count) 
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates returned for {symbol} on {timeframe_str}. Error: {mt5.last_error()}")
            return pd.DataFrame()

        ohlc_df = pd.DataFrame(rates)
        ohlc_df['time'] = pd.to_datetime(ohlc_df['time'], unit='s')
        ohlc_df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)
        
        logger.debug(f"MarketDataProvider ({symbol}): Fetched {len(ohlc_df)} candles, "
                    f"Time range: {ohlc_df['time'].min()} to {ohlc_df['time'].max()}")
        
        return ohlc_df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
class SMCAnalyzer:
    """Enhanced SMC Analysis with ICT concepts: PO3, Market Maker Models, OTE, and Judas Swings."""
    
    def __init__(self, swing_lookback, structure_lookback, pd_fib_level):
        self.swing_lookback = swing_lookback
        self.structure_lookback = structure_lookback
        self.pd_fib_level = pd_fib_level
        
        # ICT-specific Fibonacci levels for OTE (Optimal Trade Entry)
        self.ote_levels = [0.62, 0.705, 0.79]  # 62%, 70.5%, 79% retracement levels
        
        # Session timing windows (in hours UTC) - ICT high-probability windows
        self.ict_windows = {
            "asian_open": (0, 3),      # 00:00-03:00 UTC
            "london_open": (7, 10),    # 07:00-10:00 UTC (08:00-11:00 BST)
            "ny_open": (12, 15),       # 12:00-15:00 UTC (08:00-11:00 EST)
            "london_close": (15, 17)   # 15:00-17:00 UTC
        }
        
    def get_swing_points(self, ohlc_df: pd.DataFrame):
        """Get swing highs and lows."""
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_swing_points: Input ohlc_df is None or empty.")
            return None
        
        logger.debug(f"get_swing_points: Processing {len(ohlc_df)} candles with lookback={self.swing_lookback}")
        
        try:
            swings = smc_package.swing_highs_lows(ohlc_df.copy(), swing_length=self.swing_lookback)
            return swings
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_swing_points: Exception: {e}", exc_info=True)
            return None

    def get_market_structure(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """Get BOS/CHoCH market structure."""
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_market_structure: Input ohlc_df is None or empty.")
            return ohlc_df 
        if swings_df is None or swings_df.empty:
            logger.warning("SMCAnalyzer.get_market_structure: Input swings_df is None or empty.")
            return ohlc_df
            
        logger.debug(f"get_market_structure: Analyzing structure with {len(ohlc_df)} candles")
        
        try:
            analysis_results_df = smc_package.bos_choch(
                ohlc_df.copy(), 
                swing_highs_lows=swings_df,
                close_break=True
            )
            if analysis_results_df is None or not isinstance(analysis_results_df, pd.DataFrame):
                logger.error(f"SMCAnalyzer.get_market_structure: smc_package.bos_choch returned invalid data type or None.")
                return ohlc_df
            if ohlc_df.shape[0] != analysis_results_df.shape[0]:
                logger.error(f"SMCAnalyzer.get_market_structure: Row count mismatch between ohlc ({ohlc_df.shape[0]}) and analysis ({analysis_results_df.shape[0]}).")
                return ohlc_df 
                
            enriched_ohlc_df = ohlc_df.copy()
            for col_name in analysis_results_df.columns:
                enriched_ohlc_df[col_name] = analysis_results_df[col_name].values
            return enriched_ohlc_df
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_market_structure: Exception: {e}", exc_info=True)
            return ohlc_df

    def detect_power_of_three(self, ohlc_df: pd.DataFrame, lookback: int = 20):
        """
        Detect ICT Power of Three pattern: Accumulation → Manipulation → Distribution
        This is a key ICT concept for understanding daily/session price delivery
        """
        if ohlc_df is None or len(ohlc_df) < lookback:
            return None
            
        recent_data = ohlc_df.tail(lookback)
        
        # Calculate range and identify phases
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        range_size = high - low
        
        # Divide into three time segments
        segment_size = len(recent_data) // 3
        accumulation = recent_data.iloc[:segment_size]
        manipulation = recent_data.iloc[segment_size:segment_size*2]
        distribution = recent_data.iloc[segment_size*2:]
        
        # Analyze each phase
        acc_range = accumulation['high'].max() - accumulation['low'].min()
        manip_range = manipulation['high'].max() - manipulation['low'].min()
        dist_range = distribution['high'].max() - distribution['low'].min()
        
        # Classic PO3 patterns
        # Bullish: Tight accumulation → manipulation down → distribution up
        if acc_range < range_size * 0.3 and manipulation['close'].iloc[-1] < accumulation['close'].mean():
            if distribution['close'].iloc[-1] > manipulation['close'].iloc[-1]:
                return {
                    'type': 'bullish_po3',
                    'accumulation_level': accumulation['close'].mean(),
                    'manipulation_low': manipulation['low'].min(),
                    'current_phase': 'distribution'
                }
        
        # Bearish: Tight accumulation → manipulation up → distribution down
        if acc_range < range_size * 0.3 and manipulation['close'].iloc[-1] > accumulation['close'].mean():
            if distribution['close'].iloc[-1] < manipulation['close'].iloc[-1]:
                return {
                    'type': 'bearish_po3',
                    'accumulation_level': accumulation['close'].mean(),
                    'manipulation_high': manipulation['high'].max(),
                    'current_phase': 'distribution'
                }
                
        return None

    def detect_judas_swing(self, ohlc_df: pd.DataFrame, session_data: dict, lookback: int = 10):
        """
        Detect Judas Swing - false breakout in early session followed by reversal
        Key ICT concept for identifying stop hunts before true directional moves
        """
        if ohlc_df is None or len(ohlc_df) < lookback or not session_data:
            return None
            
        # Get session high/low from first 30 minutes
        session_start_idx = -lookback
        early_session = ohlc_df.iloc[session_start_idx:session_start_idx + 2]  # First 2 candles of session
        
        if len(early_session) < 2:
            return None
            
        initial_high = early_session['high'].max()
        initial_low = early_session['low'].min()
        
        # Look for breakout and reversal in subsequent candles
        remaining_session = ohlc_df.iloc[session_start_idx + 2:]
        
        for i in range(len(remaining_session) - 1):
            current = remaining_session.iloc[i]
            next_candle = remaining_session.iloc[i + 1]
            
            # Bearish Judas: Break above initial high then reverse
            if current['high'] > initial_high and next_candle['close'] < initial_high:
                # Confirm with strong bearish candle
                if next_candle['close'] < next_candle['open'] and \
                   (next_candle['open'] - next_candle['close']) > (current['high'] - initial_high):
                    return {
                        'type': 'bearish_judas',
                        'judas_high': current['high'],
                        'initial_range_high': initial_high,
                        'reversal_candle_idx': remaining_session.index[i + 1]
                    }
            
            # Bullish Judas: Break below initial low then reverse  
            if current['low'] < initial_low and next_candle['close'] > initial_low:
                # Confirm with strong bullish candle
                if next_candle['close'] > next_candle['open'] and \
                   (next_candle['close'] - next_candle['open']) > (initial_low - current['low']):
                    return {
                        'type': 'bullish_judas',
                        'judas_low': current['low'],
                        'initial_range_low': initial_low,
                        'reversal_candle_idx': remaining_session.index[i + 1]
                    }
                    
        return None

    def calculate_ote_zone(self, swing_high: float, swing_low: float, direction: str = 'bullish'):
        """
        Calculate Optimal Trade Entry (OTE) zone using ICT Fibonacci levels
        Returns the 62%-79% retracement zone which is the optimal entry area
        """
        range_size = swing_high - swing_low
        
        if direction == 'bullish':
            # For bullish OTE, we measure retracement from high to low
            ote_62 = swing_high - (range_size * 0.62)
            ote_705 = swing_high - (range_size * 0.705)  # Sweet spot
            ote_79 = swing_high - (range_size * 0.79)
            
            return {
                'ote_high': ote_62,     # 62% level (shallowest)
                'ote_sweet': ote_705,   # 70.5% level (sweet spot)
                'ote_low': ote_79,      # 79% level (deepest)
                'direction': 'bullish'
            }
        else:
            # For bearish OTE, we measure retracement from low to high
            ote_62 = swing_low + (range_size * 0.62)
            ote_705 = swing_low + (range_size * 0.705)
            ote_79 = swing_low + (range_size * 0.79)
            
            return {
                'ote_low': ote_62,      # 62% level (shallowest)
                'ote_sweet': ote_705,   # 70.5% level (sweet spot)  
                'ote_high': ote_79,     # 79% level (deepest)
                'direction': 'bearish'
            }

    def detect_market_maker_model(self, ohlc_df: pd.DataFrame, current_time: datetime):
        """
        Detect Market Maker Buy/Sell Models - ICT's daily manipulation patterns
        These models describe how smart money accumulates/distributes throughout the day
        """
        if ohlc_df is None or len(ohlc_df) < 50:
            return None
            
        # Get current hour in UTC
        current_hour = current_time.hour
        
        # Market Maker Buy Model (Bullish day):
        # 1. Asian session: Consolidation/slight decline
        # 2. London Open: Manipulation move down (grab sell stops)
        # 3. NY Session: Distribution rally up
        
        # Market Maker Sell Model (Bearish day):
        # 1. Asian session: Consolidation/slight rally  
        # 2. London Open: Manipulation move up (grab buy stops)
        # 3. NY Session: Distribution sell-off down
        
        # Analyze daily structure
        today_start = pd.Timestamp(current_time.date())
        today_data = ohlc_df[ohlc_df.index >= today_start] if hasattr(ohlc_df, 'index') else ohlc_df[ohlc_df['time'] >= today_start]
        
        if len(today_data) < 10:  # Not enough data for today
            return None
            
        # Define session times (rough approximations)
        asian_data = today_data[today_data.index.hour < 7] if hasattr(today_data, 'index') else today_data[pd.to_datetime(today_data['time']).dt.hour < 7]
        london_data = today_data[(today_data.index.hour >= 7) & (today_data.index.hour < 12)] if hasattr(today_data, 'index') else today_data[(pd.to_datetime(today_data['time']).dt.hour >= 7) & (pd.to_datetime(today_data['time']).dt.hour < 12)]
        
        if len(asian_data) > 0 and len(london_data) > 0:
            asian_range = asian_data['high'].max() - asian_data['low'].min()
            asian_close = asian_data['close'].iloc[-1]
            
            # Check for manipulation in London
            london_low = london_data['low'].min()
            london_high = london_data['high'].max()
            
            # Buy Model: London sweeps below Asian range
            if london_low < asian_data['low'].min() - (asian_range * 0.5):
                if current_hour >= 12:  # In NY session
                    return {
                        'type': 'mm_buy_model',
                        'asian_low': asian_data['low'].min(),
                        'manipulation_low': london_low,
                        'target': asian_data['high'].max() + asian_range  # Project above Asian high
                    }
            
            # Sell Model: London sweeps above Asian range
            if london_high > asian_data['high'].max() + (asian_range * 0.5):
                if current_hour >= 12:  # In NY session
                    return {
                        'type': 'mm_sell_model',
                        'asian_high': asian_data['high'].max(),
                        'manipulation_high': london_high,
                        'target': asian_data['low'].min() - asian_range  # Project below Asian low
                    }
                    
        return None

    def is_in_ict_killzone(self, current_time: datetime):
        """
        Check if current time is within ICT high-probability kill zones
        These are specific time windows where institutional activity is highest
        """
        current_hour = current_time.hour
        
        for window_name, (start, end) in self.ict_windows.items():
            if start <= current_hour < end:
                return True, window_name
                
        return False, None

    def get_refined_order_blocks(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """
        Enhanced Order Block detection focusing on ICT criteria:
        - Last up/down candle before structure break
        - Volume characteristics
        - Inefficiency (FVG) preceding the OB
        """
        # First get standard OBs
        standard_obs = self.get_order_blocks(ohlc_df, swings_df)
        if standard_obs is None or standard_obs.empty:
            return None
            
        # Refine based on ICT criteria
        refined_obs = []
        
        for idx, ob in standard_obs.iterrows():
            if pd.isna(ob['OB']):
                continue
                
            # Check for preceding FVG (inefficiency before OB is significant)
            fvg_check_range = range(max(0, idx - 5), idx)
            has_preceding_fvg = False
            
            for i in fvg_check_range:
                if i >= 2:  # Need at least 3 candles for FVG
                    # Check for bullish FVG
                    if ob['OB'] == 1:
                        if ohlc_df.iloc[i-2]['high'] < ohlc_df.iloc[i]['low']:
                            has_preceding_fvg = True
                            break
                    # Check for bearish FVG
                    elif ob['OB'] == -1:
                        if ohlc_df.iloc[i-2]['low'] > ohlc_df.iloc[i]['high']:
                            has_preceding_fvg = True
                            break
            
            # Volume analysis - OB should have above-average volume
            if 'OBVolume' in ob:
                avg_volume = ohlc_df['volume'].rolling(20).mean().iloc[idx]
                if pd.notna(avg_volume) and ob['OBVolume'] > avg_volume * 1.5:
                    ob['volume_significance'] = 'high'
                else:
                    ob['volume_significance'] = 'normal'
            
            # Check if it's the last candle before structure break
            ob['has_preceding_fvg'] = has_preceding_fvg
            ob['is_refined'] = has_preceding_fvg or ob.get('volume_significance') == 'high'
            
            refined_obs.append(ob)
            
        return pd.DataFrame(refined_obs) if refined_obs else None

    def detect_enhanced_liquidity_sweep(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame, 
                                      liquidity_zones: pd.DataFrame, candle_index: int = -1):
        """
        Enhanced liquidity sweep detection with ICT characteristics:
        - Stop hunt with immediate reversal
        - Volume spike on sweep
        - Respect of higher timeframe levels
        """
        if ohlc_df is None or len(ohlc_df) < abs(candle_index) + 3:
            return None
            
        # Standard sweep detection first
        basic_sweep = self.detect_liquidity_sweep(ohlc_df, swings_df, candle_index)
        
        if not basic_sweep:
            return None
            
        # Enhance with ICT criteria
        current_idx = len(ohlc_df) + candle_index if candle_index < 0 else candle_index
        current = ohlc_df.iloc[candle_index]
        
        # Check for volume spike
        avg_volume = ohlc_df['volume'].rolling(20).mean().iloc[current_idx]
        volume_spike = current['volume'] > avg_volume * 2 if pd.notna(avg_volume) else False
        
        # Check for immediate reversal (within 2 candles)
        reversal_confirmed = False
        if current_idx < len(ohlc_df) - 2:
            next_candles = ohlc_df.iloc[current_idx + 1:current_idx + 3]
            
            if basic_sweep == "bullish_sweep":
                # Should see bullish momentum after sweep
                reversal_confirmed = any(c['close'] > c['open'] and 
                                       (c['close'] - c['open']) > (c['high'] - c['low']) * 0.6 
                                       for _, c in next_candles.iterrows())
            else:
                # Should see bearish momentum after sweep
                reversal_confirmed = any(c['close'] < c['open'] and 
                                       (c['open'] - c['close']) > (c['high'] - c['low']) * 0.6 
                                       for _, c in next_candles.iterrows())
        
        # Check if sweep occurred at a liquidity zone
        at_liquidity_zone = False
        if liquidity_zones is not None and not liquidity_zones.empty:
            for _, liq in liquidity_zones.iterrows():
                if pd.notna(liq.get('Level')):
                    if abs(current['high'] - liq['Level']) < (current['high'] * 0.001) or \
                       abs(current['low'] - liq['Level']) < (current['low'] * 0.001):
                        at_liquidity_zone = True
                        break
        
        if volume_spike or reversal_confirmed or at_liquidity_zone:
            return {
                'type': basic_sweep,
                'enhanced': True,
                'volume_spike': volume_spike,
                'reversal_confirmed': reversal_confirmed,
                'at_liquidity_zone': at_liquidity_zone,
                'strength': sum([volume_spike, reversal_confirmed, at_liquidity_zone])
            }
            
        return {'type': basic_sweep, 'enhanced': False, 'strength': 0}

    def get_order_blocks(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """Standard order block detection from the original implementation."""
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_order_blocks: Invalid input data.")
            return None
        
        try:
            ob_results = smc_package.ob(
                ohlc_df.copy(), 
                swing_highs_lows=swings_df,
                close_mitigation=False
            )
            
            if ob_results is None:
                logger.error("get_order_blocks: smc_package.ob returned None")
                return None
                
            # Get recent OBs from last 50 candles
            recent_obs = ob_results[ob_results.index >= len(ohlc_df) - 50]
            
            # Filter for unmitigated OBs
            if 'MitigatedIndex' in recent_obs.columns:
                fresh_obs = recent_obs[(recent_obs['OB'].notna()) & (recent_obs['MitigatedIndex'].isna())]
                return fresh_obs if len(fresh_obs) > 0 else recent_obs
            
            return recent_obs
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_order_blocks: Exception: {e}", exc_info=True)
            return None

    def get_fair_value_gaps(self, ohlc_df: pd.DataFrame):
        """Standard FVG detection."""
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_fair_value_gaps: Input ohlc_df is None or empty.")
            return None
        
        try:
            fvg_results = smc_package.fvg(
                ohlc_df.copy(),
                join_consecutive=True
            )
            
            if fvg_results is None:
                logger.error("get_fair_value_gaps: smc_package.fvg returned None")
                return None
            
            # Get recent FVGs from last 50 candles
            recent_fvgs = fvg_results[fvg_results.index >= len(ohlc_df) - 50]
            
            # Filter for unmitigated FVGs
            if 'MitigatedIndex' in fvg_results.columns:
                open_fvgs = recent_fvgs[(recent_fvgs['FVG'].notna()) & (recent_fvgs['MitigatedIndex'].isna())]
                return open_fvgs if len(open_fvgs) > 0 else recent_fvgs
            
            return recent_fvgs
                
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_fair_value_gaps: Exception: {e}", exc_info=True)
            return None

    def get_liquidity_zones(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """Standard liquidity detection."""
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_liquidity_zones: Invalid input data.")
            return None
        
        try:
            # Calculate dynamic range
            total_swings = len(swings_df[swings_df['HighLow'].notna()])
            if total_swings < 4:
                logger.warning(f"Insufficient swing points for liquidity detection: {total_swings}")
                return pd.DataFrame()
            
            # Dynamic range calculation
            recent_candles = min(20, len(ohlc_df))
            recent_data = ohlc_df.tail(recent_candles)
            
            # ATR-based range
            high_low = recent_data['high'] - recent_data['low']
            high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
            low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.mean()
            
            avg_candle_range = (recent_data['high'] - recent_data['low']).mean()
            current_price = ohlc_df['close'].iloc[-1]
            is_jpy_pair = current_price > 50
            
            total_data_range = ohlc_df['high'].max() - ohlc_df['low'].min()
            typical_movement = max(atr, avg_candle_range)
            
            if is_jpy_pair:
                liquidity_threshold = typical_movement * 1.5
            else:
                liquidity_threshold = typical_movement * 2.0
            
            range_percent = liquidity_threshold / total_data_range if total_data_range > 0 else 0.001
            range_percent = max(0.0005, min(range_percent, 0.02))
            
            liquidity_results = smc_package.liquidity(
                ohlc_df.copy(),
                swing_highs_lows=swings_df,
                range_percent=range_percent
            )
            
            return liquidity_results if liquidity_results is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_liquidity_zones: Exception: {e}", exc_info=True)
            return pd.DataFrame()

    def detect_liquidity_sweep(self, ohlc_df, swings_df, candle_index=-1):
        """Basic liquidity sweep detection."""
        if ohlc_df is None or len(ohlc_df) < abs(candle_index) + 1 or swings_df is None: 
            return None
            
        try:
            if candle_index < 0:
                candle_position = len(ohlc_df) + candle_index
            else:
                candle_position = candle_index
                
            current_candle = ohlc_df.iloc[candle_index]
            
            # Check for bullish sweep
            relevant_prev_lows = swings_df['Level'][
                (swings_df['HighLow'] == -1) & 
                (swings_df.index < candle_position)
            ].dropna()
            
            if not relevant_prev_lows.empty:
                recent_low = relevant_prev_lows.iloc[-1]
                if (current_candle['low'] < recent_low and 
                    current_candle['close'] > recent_low):
                    logger.debug(f"Detected bullish liquidity sweep at position {candle_position}")
                    return "bullish_sweep"
            
            # Check for bearish sweep
            relevant_prev_highs = swings_df['Level'][
                (swings_df['HighLow'] == 1) & 
                (swings_df.index < candle_position)
            ].dropna()
            
            if not relevant_prev_highs.empty:
                recent_high = relevant_prev_highs.iloc[-1]
                if (current_candle['high'] > recent_high and 
                    current_candle['close'] < recent_high):
                    logger.debug(f"Detected bearish liquidity sweep at position {candle_position}")
                    return "bearish_sweep"
                    
        except Exception as e: 
            logger.error(f"Error in detect_liquidity_sweep: {e}")
            
        return None

    def detect_active_session(self, ohlc_df):
        """Session detection with focus on ICT kill zones."""
        try:
            # Get current time from last candle
            if hasattr(ohlc_df, 'index'):
                current_time = ohlc_df.index[-1]
            else:
                current_time = pd.to_datetime(ohlc_df['time'].iloc[-1])
            
            # Check if in ICT kill zone first
            in_killzone, killzone_name = self.is_in_ict_killzone(current_time)
            
            # Standard session check
            available_sessions = ["Sydney", "Tokyo", "London", "New York", 
                                "Asian kill zone", "London open kill zone", 
                                "New York kill zone", "london close kill zone"]
            
            active_sessions = []
            
            for session in available_sessions:
                try:
                    session_result = smc_package.sessions(
                        ohlc_df.copy(),
                        session=session,
                        time_zone="EET"  
                    )
                    
                    if session_result is not None and 'Active' in session_result.columns:
                        latest_active = session_result['Active'].iloc[-1]
                        
                        if latest_active == 1:
                            active_sessions.append(session)
                            
                except Exception as e:
                    logger.warning(f"Error checking session '{session}': {e}")
                    continue
            
            # Add ICT kill zone info
            if in_killzone:
                active_sessions.append(f"ICT_{killzone_name}")
            
            is_any_active = len(active_sessions) > 0
            
            return is_any_active, active_sessions, in_killzone, killzone_name
            
        except Exception as e:
            logger.error(f"Error in detect_active_session: {e}", exc_info=True)
            return False, [], False, None

    def get_premium_discount(self, ohlc_df):
        """Enhanced Premium/Discount with multiple Fibonacci levels."""
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback: 
            logger.warning("get_premium_discount: Insufficient data for P/D analysis")
            return None, None, None 
            
        relevant_data = ohlc_df.tail(self.structure_lookback)
        period_high, period_low = relevant_data['high'].max(), relevant_data['low'].min()
        
        # Multiple Fibonacci levels for more nuanced analysis
        range_size = period_high - period_low
        
        levels = {
            'equilibrium': period_low + (range_size * 0.5),
            'premium_25': period_low + (range_size * 0.75),
            'premium_20': period_low + (range_size * 0.80),
            'discount_25': period_low + (range_size * 0.25),
            'discount_20': period_low + (range_size * 0.20),
            'ote_62': period_low + (range_size * 0.62),
            'ote_70': period_low + (range_size * 0.705),
            'ote_79': period_low + (range_size * 0.79)
        }
                
        return levels['equilibrium'], period_high, period_low, levels

    def get_retracement_analysis(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """Get retracement analysis using SMC library."""
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_retracement_analysis: Invalid input data.")
            return None
        
        try:
            retracement_results = smc_package.retracements(
                ohlc_df.copy(),
                swing_highs_lows=swings_df
            )
            
            return retracement_results
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_retracement_analysis: Exception: {e}", exc_info=True)
            return None

    def get_comprehensive_analysis(self, ohlc_df: pd.DataFrame, higher_timeframe: str = "4h"):
        """Enhanced comprehensive analysis with all ICT concepts."""
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_comprehensive_analysis: Empty input data")
            return {}
        
        try:
            # Basic SMC analysis
            swings = self.get_swing_points(ohlc_df)
            if swings is None:
                logger.error("Failed to get swing points - aborting analysis")
                return {}
                
            structure = self.get_market_structure(ohlc_df, swings)
            
            # Standard SMC components
            order_blocks = self.get_order_blocks(ohlc_df, swings)
            fvgs = self.get_fair_value_gaps(ohlc_df)
            liquidity = self.get_liquidity_zones(ohlc_df, swings)
            retracements = self.get_retracement_analysis(ohlc_df, swings)
            
            # Enhanced ICT components
            refined_obs = self.get_refined_order_blocks(ohlc_df, swings)
            
            # Previous timeframe levels
            prev_levels = None
            try:
                prev_levels = smc_package.previous_high_low(ohlc_df.copy(), time_frame=higher_timeframe)
            except Exception as e:
                logger.warning(f"Could not get previous {higher_timeframe} levels: {e}")
            
            # Current time for session analysis
            latest_time = ohlc_df.index[-1] if hasattr(ohlc_df, 'index') else ohlc_df['time'].iloc[-1]
            current_time = pd.to_datetime(latest_time) if not isinstance(latest_time, pd.Timestamp) else latest_time
            
            # Session and ICT timing
            in_session, active_sessions, in_killzone, killzone_name = self.detect_active_session(ohlc_df)
            
            # ICT-specific patterns
            po3_pattern = self.detect_power_of_three(ohlc_df)
            mm_model = self.detect_market_maker_model(ohlc_df, current_time)
            
            # Judas Swing detection (if in session)
            judas_swing = None
            if in_session and len(active_sessions) > 0:
                session_data = {'active_sessions': active_sessions}
                judas_swing = self.detect_judas_swing(ohlc_df, session_data)
            
            # Premium/Discount with multiple levels
            eq, prem_high, disc_low, fib_levels = self.get_premium_discount(ohlc_df)
            
            # OTE zones for recent swings
            ote_zones = []
            recent_swing_highs = swings[swings['HighLow'] == 1].tail(3)
            recent_swing_lows = swings[swings['HighLow'] == -1].tail(3)
            
            if len(recent_swing_highs) > 0 and len(recent_swing_lows) > 0:
                last_high = recent_swing_highs['Level'].iloc[-1]
                last_low = recent_swing_lows['Level'].iloc[-1]
                
                # Determine trend direction
                if recent_swing_highs.index[-1] > recent_swing_lows.index[-1]:
                    # Last swing was a high, so we're potentially retracing down
                    ote_zones.append(self.calculate_ote_zone(last_high, last_low, 'bearish'))
                else:
                    # Last swing was a low, so we're potentially retracing up
                    ote_zones.append(self.calculate_ote_zone(last_high, last_low, 'bullish'))
            
            analysis = {
                'ohlc_df': ohlc_df,
                'swings': swings,
                'structure': structure,
                'order_blocks': order_blocks,
                'refined_order_blocks': refined_obs,
                'fair_value_gaps': fvgs,
                'liquidity_zones': liquidity,
                'retracements': retracements,
                'previous_levels': prev_levels,
                'in_session': in_session,
                'active_sessions': active_sessions,
                'in_killzone': in_killzone,
                'killzone_name': killzone_name,
                'power_of_three': po3_pattern,
                'market_maker_model': mm_model,
                'judas_swing': judas_swing,
                'ote_zones': ote_zones,
                'equilibrium': eq,
                'premium_high': prem_high,
                'discount_low': disc_low,
                'fibonacci_levels': fib_levels,
                'timestamp': current_time
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_comprehensive_analysis: Exception: {e}", exc_info=True)
            return {}


class SignalGenerator:
    """
    Enhanced ICT Signal Generator with proper entry models and timing.
    Implements OTE entries, Judas Swing trades, and Market Maker Models.
    """
    
    def __init__(self, smc_analyzer, sl_buffer_points, tp_rr_ratio, higher_timeframe="4h", sl_atr_multiplier=1.0):
        self.smc_analyzer = smc_analyzer
        self.sl_buffer_points = sl_buffer_points
        self.tp_rr_ratio = tp_rr_ratio
        self.higher_timeframe = higher_timeframe
        self.sl_atr_multiplier = sl_atr_multiplier

    def generate(self, ohlc_df_full, symbol_point_size, current_trading_symbol):
        """
        Generate ICT-based trading signals with proper entry models.
        
        Key ICT Entry Models:
        1. OTE (Optimal Trade Entry) - Enter at 62-79% retracement
        2. Judas Swing Reversal - Trade the reversal after stop hunt
        3. Market Maker Model - Trade in direction of daily bias
        4. Refined Order Block with FVG - High probability institutional levels
        """
        
        if ohlc_df_full is None or len(ohlc_df_full) < self.smc_analyzer.structure_lookback:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): Insufficient data")
            return None, None, None

        # Get comprehensive SMC analysis
        smc_analysis = self.smc_analyzer.get_comprehensive_analysis(ohlc_df_full, higher_timeframe=self.higher_timeframe)
        if not smc_analysis:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): SMC analysis failed.")
            return None, None, None

        # Extract all analysis components
        structure_df = smc_analysis.get('structure', ohlc_df_full)
        refined_obs = smc_analysis.get('refined_order_blocks')
        fvgs = smc_analysis.get('fair_value_gaps')
        liquidity_zones = smc_analysis.get('liquidity_zones')
        in_killzone = smc_analysis.get('in_killzone', False)
        killzone_name = smc_analysis.get('killzone_name')
        po3_pattern = smc_analysis.get('power_of_three')
        mm_model = smc_analysis.get('market_maker_model')
        judas_swing = smc_analysis.get('judas_swing')
        ote_zones = smc_analysis.get('ote_zones', [])
        fib_levels = smc_analysis.get('fibonacci_levels', {})
        
        # Get current price
        latest_data = structure_df.iloc[-1].copy()
        current_price = latest_data['close']
        
        # Premium/Discount
        eq = smc_analysis.get('equilibrium')
        if eq is None:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): No equilibrium level")
            return None, None, None
        
        # Check for recent structure
        recent_structure_indices = structure_df.index[-20:]
        recent_bullish_structure = any(
            (structure_df.loc[idx, 'BOS'] == 1 if 'BOS' in structure_df.columns and not pd.isna(structure_df.loc[idx, 'BOS']) else False) or
            (structure_df.loc[idx, 'CHOCH'] == 1 if 'CHOCH' in structure_df.columns and not pd.isna(structure_df.loc[idx, 'CHOCH']) else False)
            for idx in recent_structure_indices if idx in structure_df.index
        )
        recent_bearish_structure = any(
            (structure_df.loc[idx, 'BOS'] == -1 if 'BOS' in structure_df.columns and not pd.isna(structure_df.loc[idx, 'BOS']) else False) or
            (structure_df.loc[idx, 'CHOCH'] == -1 if 'CHOCH' in structure_df.columns and not pd.isna(structure_df.loc[idx, 'CHOCH']) else False)
            for idx in recent_structure_indices if idx in structure_df.index
        )

        # === ICT ENTRY MODEL 1: JUDAS SWING REVERSAL ===
        if judas_swing:
            if judas_swing['type'] == 'bullish_judas':
                # Bullish Judas: Price swept lows and reversed up
                logger.info(f"ICT JUDAS SWING BULLISH ({current_trading_symbol}): "
                           f"Swept low {judas_swing['judas_low']:.5f}, reversed above {judas_swing['initial_range_low']:.5f}")
                
                sl_price = judas_swing['judas_low'] - (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                tp_price = current_price + ((current_price - sl_price) * self.tp_rr_ratio)
                
                return "BUY", sl_price, tp_price
                
            elif judas_swing['type'] == 'bearish_judas':
                # Bearish Judas: Price swept highs and reversed down
                logger.info(f"ICT JUDAS SWING BEARISH ({current_trading_symbol}): "
                           f"Swept high {judas_swing['judas_high']:.5f}, reversed below {judas_swing['initial_range_high']:.5f}")
                
                sl_price = judas_swing['judas_high'] + (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                tp_price = current_price - ((sl_price - current_price) * self.tp_rr_ratio)
                
                return "SELL", sl_price, tp_price

        # === ICT ENTRY MODEL 2: MARKET MAKER MODEL ===
        if mm_model and in_killzone:
            if mm_model['type'] == 'mm_buy_model':
                # In NY session after London manipulation down
                if current_price < eq:  # Still in discount after manipulation
                    logger.info(f"ICT MARKET MAKER BUY MODEL ({current_trading_symbol}): "
                               f"London swept {mm_model['manipulation_low']:.5f}, targeting {mm_model['target']:.5f}")
                    
                    sl_price = mm_model['manipulation_low'] - (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                    tp_price = mm_model['target']
                    
                    return "BUY", sl_price, tp_price
                    
            elif mm_model['type'] == 'mm_sell_model':
                # In NY session after London manipulation up
                if current_price > eq:  # Still in premium after manipulation
                    logger.info(f"ICT MARKET MAKER SELL MODEL ({current_trading_symbol}): "
                               f"London swept {mm_model['manipulation_high']:.5f}, targeting {mm_model['target']:.5f}")
                    
                    sl_price = mm_model['manipulation_high'] + (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                    tp_price = mm_model['target']
                    
                    return "SELL", sl_price, tp_price

        # === ICT ENTRY MODEL 3: OTE WITH REFINED ORDER BLOCKS ===
        
        # BULLISH SETUP: Discount + Structure + OTE/OB confluence
        is_in_discount = current_price < eq
        if is_in_discount and recent_bullish_structure:
            # Check if price is in OTE zone
            in_ote = False
            for ote in ote_zones:
                if ote['direction'] == 'bullish':
                    if ote['ote_low'] <= current_price <= ote['ote_high']:
                        in_ote = True
                        logger.debug(f"Price in bullish OTE zone: {ote['ote_low']:.5f} - {ote['ote_high']:.5f}")
                        break
            
            # Look for refined order blocks
            has_refined_ob = False
            ob_level = None
            if refined_obs is not None and not refined_obs.empty:
                bull_obs = refined_obs[(refined_obs['OB'] == 1) & (refined_obs.get('is_refined', False) == True)]
                for _, ob in bull_obs.iterrows():
                    if ob['Bottom'] <= current_price <= ob['Top']:
                        has_refined_ob = True
                        ob_level = ob['Bottom']
                        break
            
            # Power of Three confluence
            po3_bullish = po3_pattern and po3_pattern['type'] == 'bullish_po3'
            
            # Entry decision with ICT criteria
            if (in_ote and (has_refined_ob or po3_bullish)) or (in_killzone and has_refined_ob):
                logger.info(f"ICT OTE/OB BULLISH SIGNAL ({current_trading_symbol}): "
                           f"In OTE={in_ote}, Refined OB={has_refined_ob}, PO3={po3_bullish}, "
                           f"Killzone={killzone_name if in_killzone else 'No'}")
                
                # Calculate levels
                sl_price, tp_price = self._calculate_ict_bullish_levels(
                    latest_data, current_price, smc_analysis, symbol_point_size, ob_level
                )
                
                if sl_price and tp_price and sl_price < current_price < tp_price:
                    return "BUY", sl_price, tp_price

        # BEARISH SETUP: Premium + Structure + OTE/OB confluence
        is_in_premium = current_price > eq
        if is_in_premium and recent_bearish_structure:
            # Check if price is in OTE zone
            in_ote = False
            for ote in ote_zones:
                if ote['direction'] == 'bearish':
                    if ote['ote_low'] <= current_price <= ote['ote_high']:
                        in_ote = True
                        logger.debug(f"Price in bearish OTE zone: {ote['ote_low']:.5f} - {ote['ote_high']:.5f}")
                        break
            
            # Look for refined order blocks
            has_refined_ob = False
            ob_level = None
            if refined_obs is not None and not refined_obs.empty:
                bear_obs = refined_obs[(refined_obs['OB'] == -1) & (refined_obs.get('is_refined', False) == True)]
                for _, ob in bear_obs.iterrows():
                    if ob['Bottom'] <= current_price <= ob['Top']:
                        has_refined_ob = True
                        ob_level = ob['Top']
                        break
            
            # Power of Three confluence
            po3_bearish = po3_pattern and po3_pattern['type'] == 'bearish_po3'
            
            # Entry decision with ICT criteria
            if (in_ote and (has_refined_ob or po3_bearish)) or (in_killzone and has_refined_ob):
                logger.info(f"ICT OTE/OB BEARISH SIGNAL ({current_trading_symbol}): "
                           f"In OTE={in_ote}, Refined OB={has_refined_ob}, PO3={po3_bearish}, "
                           f"Killzone={killzone_name if in_killzone else 'No'}")
                
                # Calculate levels
                sl_price, tp_price = self._calculate_ict_bearish_levels(
                    latest_data, current_price, smc_analysis, symbol_point_size, ob_level
                )
                
                if sl_price and tp_price and tp_price < current_price < sl_price:
                    return "SELL", sl_price, tp_price

        # === ICT ENTRY MODEL 4: ENHANCED LIQUIDITY SWEEP ===
        enhanced_sweep = self.smc_analyzer.detect_enhanced_liquidity_sweep(
            ohlc_df_full, 
            smc_analysis.get('swings'),
            liquidity_zones
        )
        
        if enhanced_sweep and enhanced_sweep.get('enhanced') and enhanced_sweep.get('strength', 0) >= 2:
            if enhanced_sweep['type'] == 'bullish_sweep' and is_in_discount:
                logger.info(f"ICT ENHANCED LIQUIDITY SWEEP BULLISH ({current_trading_symbol}): "
                           f"Strong sweep with score {enhanced_sweep['strength']}")
                
                # Use recent low as stop
                recent_low = ohlc_df_full['low'].iloc[-10:].min()
                sl_price = recent_low - (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                tp_price = current_price + ((current_price - sl_price) * self.tp_rr_ratio)
                
                return "BUY", sl_price, tp_price
                
            elif enhanced_sweep['type'] == 'bearish_sweep' and is_in_premium:
                logger.info(f"ICT ENHANCED LIQUIDITY SWEEP BEARISH ({current_trading_symbol}): "
                           f"Strong sweep with score {enhanced_sweep['strength']}")
                
                # Use recent high as stop
                recent_high = ohlc_df_full['high'].iloc[-10:].max()
                sl_price = recent_high + (self.sl_atr_multiplier * self._calculate_atr(ohlc_df_full))
                tp_price = current_price - ((sl_price - current_price) * self.tp_rr_ratio)
                
                return "SELL", sl_price, tp_price

        return None, None, None

    def _calculate_atr(self, ohlc_df):
        """Calculate ATR for dynamic stop loss."""
        atr_series = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        return atr_series.iloc[-1]

    def _calculate_ict_bullish_levels(self, latest_data, current_price, smc_analysis, symbol_point_size, ob_level=None):
        """Calculate bullish levels using ICT methodology."""
        ohlc_df = smc_analysis.get('ohlc_df')
        
        # Use order block low or recent structure low
        if ob_level:
            structural_low = ob_level
        else:
            recent_lows = ohlc_df['low'].iloc[-20:-1]
            structural_low = recent_lows.min()
        
        # ATR-based buffer
        atr = self._calculate_atr(ohlc_df)
        dynamic_buffer = atr * self.sl_atr_multiplier
        
        sl_price = structural_low - dynamic_buffer
        
        # For TP, use next liquidity or structure level
        liquidity_zones = smc_analysis.get('liquidity_zones')
        if liquidity_zones is not None and not liquidity_zones.empty:
            # Find next liquidity above
            above_liquidity = liquidity_zones[liquidity_zones['Level'] > current_price]['Level']
            if not above_liquidity.empty:
                tp_price = above_liquidity.iloc[0]
            else:
                tp_price = current_price + ((current_price - sl_price) * self.tp_rr_ratio)
        else:
            tp_price = current_price + ((current_price - sl_price) * self.tp_rr_ratio)
        
        return sl_price, tp_price

    def _calculate_ict_bearish_levels(self, latest_data, current_price, smc_analysis, symbol_point_size, ob_level=None):
        """Calculate bearish levels using ICT methodology."""
        ohlc_df = smc_analysis.get('ohlc_df')
        
        # Use order block high or recent structure high
        if ob_level:
            structural_high = ob_level
        else:
            recent_highs = ohlc_df['high'].iloc[-20:-1]
            structural_high = recent_highs.max()
        
        # ATR-based buffer
        atr = self._calculate_atr(ohlc_df)
        dynamic_buffer = atr * self.sl_atr_multiplier
        
        sl_price = structural_high + dynamic_buffer
        
        # For TP, use next liquidity or structure level
        liquidity_zones = smc_analysis.get('liquidity_zones')
        if liquidity_zones is not None and not liquidity_zones.empty:
            # Find next liquidity below
            below_liquidity = liquidity_zones[liquidity_zones['Level'] < current_price]['Level']
            if not below_liquidity.empty:
                tp_price = below_liquidity.iloc[-1]
            else:
                tp_price = current_price - ((sl_price - current_price) * self.tp_rr_ratio)
        else:
            tp_price = current_price - ((sl_price - current_price) * self.tp_rr_ratio)
        
        return sl_price, tp_price


class PositionSizer:
    """Calculates trade volume based on risk parameters."""
    def __init__(self, mt5_client: MetaTrader5Client, risk_percent_per_trade):
        self.client = mt5_client
        self.risk_percent = risk_percent_per_trade

    def calculate_volume(self, symbol, sl_price, order_type):
        account_info = self.client.get_account_info()
        symbol_info = self.client.get_symbol_info(symbol)
        ticker = self.client.get_symbol_ticker(symbol)

        if not all([account_info, symbol_info, ticker]):
            logger.error(f"PositionSizer ({symbol}): Missing account, symbol or ticker info.")
            return None
        balance = account_info.balance
        risk_amount_account_currency = (self.risk_percent / 100.0) * balance
        current_price = ticker.ask if order_type == "BUY" else ticker.bid
        if (order_type == "BUY" and sl_price >= current_price) or (order_type == "SELL" and sl_price <= current_price):
            logger.error(f"PositionSizer ({symbol}): Invalid SL {sl_price} vs current {current_price} for {order_type}.")
            return None
        sl_distance_price_units = abs(current_price - sl_price)
        if sl_distance_price_units == 0:
            logger.error(f"PositionSizer ({symbol}): SL distance is zero.")
            return None
        
        value_per_point_per_lot = symbol_info.trade_tick_value 
        points_in_pip = 10 if "JPY" not in symbol_info.name else 100 
        value_per_pip_per_lot = value_per_point_per_lot * points_in_pip
        sl_pips_distance = sl_distance_price_units / (symbol_info.point * points_in_pip)

        if value_per_pip_per_lot == 0 or sl_pips_distance == 0:
             logger.error(f"PositionSizer ({symbol}): Invalid pip value ({value_per_pip_per_lot}) or SL pips ({sl_pips_distance}).")
             return None
        volume = risk_amount_account_currency / (sl_pips_distance * value_per_pip_per_lot)
        lot_step = symbol_info.volume_step
        volume = (volume // lot_step) * lot_step 
        volume = round(max(symbol_info.volume_min, volume), len(str(lot_step).split('.')[-1]) if '.' in str(lot_step) else 0)
        volume = min(symbol_info.volume_max, volume)

        if volume < symbol_info.volume_min or volume == 0: 
            logger.warning(f"PositionSizer ({symbol}): Calculated volume {volume} invalid. Min lot: {symbol_info.volume_min}.")
            return None
        logger.info(f"PositionSizer ({symbol}): Vol={volume:.2f}, SL pips={sl_pips_distance:.1f}")
        return volume

class TradeExecutor:
    """Places and manages trades."""
    def __init__(self, mt5_client: MetaTrader5Client, magic_prefix):
        self.client = mt5_client
        self.magic_prefix = magic_prefix

    def place_market_order(self, symbol, order_type, volume, sl_price, tp_price):
        if not self.client.is_connected(): return None
        ticker = self.client.get_symbol_ticker(symbol)
        if not ticker: return None
        mt5_order_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = ticker.ask if order_type == "BUY" else ticker.bid
        if (order_type == "BUY" and (sl_price >= price or tp_price <= price)) or \
           (order_type == "SELL" and (sl_price <= price or tp_price >= price)):
            logger.error(f"TradeExecutor ({symbol}): Invalid SL/TP. SL {sl_price:.5f}, TP {tp_price:.5f}, Price {price:.5f}")
            return None
        magic_number = int(f"{self.magic_prefix}{int(datetime.now().timestamp()) % 100000 + sum(ord(c) for c in symbol[:3]) % 1000}")
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume), "type": mt5_order_type,
                   "price": price, "sl": float(sl_price), "tp": float(tp_price), "deviation": 20, "magic": magic_number,
                   "comment": f"ICT_SMC_{symbol}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"TradeExecutor ({symbol}): Order send failed. Code {result.retcode if result else 'None'} - {result.comment if result else mt5.last_error()}. Req: {request}")
            return None
        logger.info(f"TradeExecutor ({symbol}): Order sent. Ticket: {result.order}, Deal: {result.deal}")
        return result