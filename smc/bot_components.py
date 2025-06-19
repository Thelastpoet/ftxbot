import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from talib import ATR
from smartmoneyconcepts import smc as smc_package # Alias

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
        
        # Debug logging
        logger.debug(f"MarketDataProvider ({symbol}): Fetched {len(ohlc_df)} candles, "
                    f"Time range: {ohlc_df['time'].min()} to {ohlc_df['time'].max()}")
        
        return ohlc_df[['time', 'open', 'high', 'low', 'close', 'volume']]
    
class SMCAnalyzer:
    """Professional SMC Analysis with Order Blocks, FVG, Liquidity, and Session filtering."""
    
    def __init__(self, swing_lookback, structure_lookback, pd_fib_level):
        self.swing_lookback = swing_lookback
        self.structure_lookback = structure_lookback
        self.pd_fib_level = pd_fib_level
        
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
                
            # Debug: Check for BOS/CHoCH
            bos_count = len(analysis_results_df[analysis_results_df['BOS'].notna()]) if 'BOS' in analysis_results_df.columns else 0
            choch_count = len(analysis_results_df[analysis_results_df['CHOCH'].notna()]) if 'CHOCH' in analysis_results_df.columns else 0
                        
            # Log recent structure changes
            if bos_count > 0 or choch_count > 0:
                recent_bos = analysis_results_df[analysis_results_df['BOS'].notna()].tail(3)
                recent_choch = analysis_results_df[analysis_results_df['CHOCH'].notna()].tail(3)
                if not recent_bos.empty:
                    logger.debug(f"Recent BOS (last 3): {recent_bos[['BOS', 'Level']].to_dict('records')}")
                if not recent_choch.empty:
                    logger.debug(f"Recent CHoCH (last 3): {recent_choch[['CHOCH', 'Level']].to_dict('records')}")
            
            enriched_ohlc_df = ohlc_df.copy()
            for col_name in analysis_results_df.columns:
                enriched_ohlc_df[col_name] = analysis_results_df[col_name].values
            return enriched_ohlc_df
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_market_structure: Exception: {e}", exc_info=True)
            return ohlc_df

    def get_order_blocks(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """
        Detect Order Blocks - institutional footprints where smart money placed orders.
        Returns recent order blocks, prioritizing fresh unmitigated ones.
        """
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_order_blocks: Invalid input data.")
            return None
        
        logger.debug(f"get_order_blocks: Analyzing {len(ohlc_df)} candles for OBs")
        
        try:
            ob_results = smc_package.ob(
                ohlc_df.copy(), 
                swing_highs_lows=swings_df,
                close_mitigation=False  # Use wick mitigation for more sensitive detection
            )
            
            # Debug logging
            if ob_results is None:
                logger.error("get_order_blocks: smc_package.ob returned None")
                return None
                
            total_obs = len(ob_results[ob_results['OB'].notna()]) if 'OB' in ob_results.columns else 0
            
            if total_obs == 0:
                logger.warning("No Order Blocks found at all - check if swings are properly detected")
                # Log some swing info for debugging
                if swings_df is not None and 'HighLow' in swings_df.columns:
                    swing_count = len(swings_df[swings_df['HighLow'].notna()])
                    logger.debug(f"Total swings available for OB detection: {swing_count}")
                return ob_results
            
            # Get recent OBs from last 50 candles
            recent_obs = ob_results[ob_results.index >= len(ohlc_df) - 50]
            recent_ob_count = len(recent_obs[recent_obs['OB'].notna()])
            
            if 'MitigatedIndex' in recent_obs.columns:
                fresh_obs = recent_obs[(recent_obs['OB'].notna()) & (recent_obs['MitigatedIndex'].isna())]
                mitigated_obs = recent_obs[(recent_obs['OB'].notna()) & (recent_obs['MitigatedIndex'].notna())]
                fresh_count = len(fresh_obs)
                
                # Log details of fresh OBs
                if fresh_count > 0:
                    for idx, ob in fresh_obs.iterrows():
                        ob_type = "Bullish" if ob['OB'] == 1 else "Bearish"
                        logger.debug(f"Fresh {ob_type} OB at index {idx}: Top={ob['Top']:.5f}, Bottom={ob['Bottom']:.5f}, Volume={ob['OBVolume']:.0f}")
                
                # Return fresh OBs if any, otherwise return all recent OBs
                if fresh_count > 0:
                    return fresh_obs
                else:
                    return recent_obs
            else:
                return recent_obs
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_order_blocks: Exception: {e}", exc_info=True)
            return None

    def get_fair_value_gaps(self, ohlc_df: pd.DataFrame):
        """
        Detect Fair Value Gaps - areas where price moved too quickly, 
        leaving inefficiencies that smart money will fill.
        """
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_fair_value_gaps: Input ohlc_df is None or empty.")
            return None
        
        logger.debug(f"get_fair_value_gaps: Analyzing {len(ohlc_df)} candles for FVGs")
            
        try:
            fvg_results = smc_package.fvg(
                ohlc_df.copy(),
                join_consecutive=True  # Merge consecutive FVGs for cleaner analysis
            )
            
            if fvg_results is None:
                logger.error("get_fair_value_gaps: smc_package.fvg returned None")
                return None
            
            # Debug logging
            total_fvgs = len(fvg_results[fvg_results['FVG'].notna()]) if 'FVG' in fvg_results.columns else 0
            
            if total_fvgs == 0:
                logger.warning("No Fair Value Gaps found at all - market may be in equilibrium")
                return fvg_results
            
            # Get recent FVGs from last 50 candles
            recent_fvgs = fvg_results[fvg_results.index >= len(ohlc_df) - 50]
            recent_fvg_count = len(recent_fvgs[recent_fvgs['FVG'].notna()])
            
            # Filter for unmitigated FVGs (price hasn't returned to fill the gap)
            if 'MitigatedIndex' in fvg_results.columns:
                open_fvgs = recent_fvgs[(recent_fvgs['FVG'].notna()) & (recent_fvgs['MitigatedIndex'].isna())]
                open_count = len(open_fvgs)            
                
                # Log details of open FVGs
                if open_count > 0:
                    for idx, fvg in open_fvgs.iterrows():
                        fvg_type = "Bullish" if fvg['FVG'] == 1 else "Bearish"
                        logger.debug(f"Open {fvg_type} FVG at index {idx}: Top={fvg['Top']:.5f}, Bottom={fvg['Bottom']:.5f}")
                
                # Return open FVGs if any, otherwise return all recent FVGs
                if open_count > 0:
                    return open_fvgs
                else:
                    return recent_fvgs
            else:
                return recent_fvgs
                
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_fair_value_gaps: Exception: {e}", exc_info=True)
            return None

    def get_liquidity_zones(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """
        Detect Liquidity zones - areas where stop losses cluster,
        prime targets for smart money manipulation before real moves.
        
        Enhanced with dynamic range calculation based on:
        1. Instrument type (JPY vs non-JPY pairs)
        2. Recent volatility (ATR-based)
        3. Swing point density
        """
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_liquidity_zones: Invalid input data.")
            return None
        
        logger.debug(f"get_liquidity_zones: Analyzing {len(ohlc_df)} candles for liquidity")
            
        try:
            # First, check if we have enough swing points for liquidity
            total_swings = len(swings_df[swings_df['HighLow'].notna()])
            if total_swings < 4:  # Need at least 4 swings for meaningful liquidity
                logger.warning(f"Insufficient swing points for liquidity detection: {total_swings}")
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            # Calculate dynamic range based on recent price action
            recent_candles = min(20, len(ohlc_df))  # Use last 20 candles or available
            recent_data = ohlc_df.tail(recent_candles)
            
            # Method 1: ATR-based (Average True Range)
            high_low = recent_data['high'] - recent_data['low']
            high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
            low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.mean()
            
            # Method 2: Average candle range
            avg_candle_range = (recent_data['high'] - recent_data['low']).mean()
            
            # Method 3: Check instrument type
            current_price = ohlc_df['close'].iloc[-1]
            is_jpy_pair = current_price > 50  # Simple heuristic for JPY pairs
            
            # Calculate appropriate range_percent
            total_data_range = ohlc_df['high'].max() - ohlc_df['low'].min()
            
            # Use the larger of ATR or average candle range for robustness
            typical_movement = max(atr, avg_candle_range)
            
            # Liquidity should form within 1-2 typical movements
            # This makes it adaptive to current volatility
            if is_jpy_pair:
                liquidity_threshold = typical_movement * 1.5  # 1.5x typical movement for JPY
            else:
                liquidity_threshold = typical_movement * 2.0  # 2x typical movement for others
            
            # Convert to percentage of total range
            range_percent = liquidity_threshold / total_data_range if total_data_range > 0 else 0.001
            
            # Apply bounds to prevent extreme values
            # Min: 0.0005 (0.05%), Max: 0.02 (2%)
            range_percent = max(0.0005, min(range_percent, 0.02))
            
            # Additional check: Analyze swing point clustering
            swing_highs = swings_df[swings_df['HighLow'] == 1]['Level'].dropna()
            swing_lows = swings_df[swings_df['HighLow'] == -1]['Level'].dropna()
            
            # Calculate minimum distances between swings
            min_high_dist = float('inf')
            min_low_dist = float('inf')
            
            if len(swing_highs) > 1:
                high_sorted = np.sort(swing_highs.values)
                high_diffs = np.diff(high_sorted)
                if len(high_diffs) > 0:
                    min_high_dist = np.min(high_diffs)
            
            if len(swing_lows) > 1:
                low_sorted = np.sort(swing_lows.values)
                low_diffs = np.diff(low_sorted)
                if len(low_diffs) > 0:
                    min_low_dist = np.min(low_diffs)
            
            # If swings are very close, reduce range_percent to capture them
            min_swing_dist = min(min_high_dist, min_low_dist)
            if min_swing_dist < float('inf'):
                suggested_range = (min_swing_dist * 1.2) / total_data_range  # 20% buffer
                if suggested_range < range_percent:
                    logger.debug(f"Adjusting range_percent from {range_percent:.5f} to {suggested_range:.5f} based on swing clustering")
                    range_percent = max(0.0005, suggested_range)
                        
            # Call the SMC library with calculated range
            liquidity_results = smc_package.liquidity(
                ohlc_df.copy(),
                swing_highs_lows=swings_df,
                range_percent=range_percent
            )
            
            if liquidity_results is None:
                logger.error("get_liquidity_zones: smc_package.liquidity returned None")
                return pd.DataFrame()
                            
            return liquidity_results
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_liquidity_zones: Exception: {e}", exc_info=True)
            return pd.DataFrame()

    def detect_active_session(self, ohlc_df):
        """
        Use SMC library's built-in sessions without bias.
        Let the market data determine which sessions are active.
        Returns (is_any_active: bool, active_sessions: list)
        """
        try:
            # Check ALL SMC sessions without bias or priority
            available_sessions = ["Sydney", "Tokyo", "London", "New York", 
                                "Asian kill zone", "London open kill zone", 
                                "New York kill zone", "london close kill zone"]
            
            active_sessions = []
            
            logger.debug(f"Checking {len(available_sessions)} SMC sessions...")
            
            for session in available_sessions:
                try:
                    # Use SMC library's session detection
                    session_result = smc_package.sessions(
                        ohlc_df.copy(),
                        session=session,
                        time_zone="EET"  
                    )
                    
                    if session_result is not None and 'Active' in session_result.columns:
                        latest_active = session_result['Active'].iloc[-1]
                        
                        if latest_active == 1:
                            active_sessions.append(session)
                            logger.debug(f"Active: {session}")
                            
                            # Optional: Log session levels for context
                            if 'High' in session_result.columns and 'Low' in session_result.columns:
                                session_high = session_result['High'].iloc[-1]
                                session_low = session_result['Low'].iloc[-1]
                                if session_high > 0 and session_low > 0:
                                    logger.debug(f"   Session levels - High: {session_high:.5f}, Low: {session_low:.5f}")
                        else:
                            logger.debug(f"Inactive: {session}")
                    else:
                        logger.warning(f"Invalid session result for: {session}")
                        
                except Exception as e:
                    logger.warning(f"Error checking session '{session}': {e}")
                    continue
            
            is_any_active = len(active_sessions) > 0
                            
            return is_any_active, active_sessions
            
        except Exception as e:
            logger.error(f"Error in detect_active_session: {e}", exc_info=True)
            return False, []

    def get_premium_discount(self, ohlc_df):
        """Enhanced Premium/Discount with session context."""
        if ohlc_df is None or len(ohlc_df) < self.structure_lookback: 
            logger.warning("get_premium_discount: Insufficient data for P/D analysis")
            return None, None, None 
            
        relevant_data = ohlc_df.tail(self.structure_lookback)
        period_high, period_low = relevant_data['high'].max(), relevant_data['low'].min()
        equilibrium = period_low + (period_high - period_low) * self.pd_fib_level
                
        return equilibrium, period_high, period_low

    def detect_liquidity_sweep(self, ohlc_df, swings_df, candle_index=-1):
        """Enhanced liquidity sweep detection."""
        if ohlc_df is None or len(ohlc_df) < abs(candle_index) + 1 or swings_df is None: 
            return None
            
        try:
            # Convert negative index to positive integer position
            if candle_index < 0:
                candle_position = len(ohlc_df) + candle_index
            else:
                candle_position = candle_index
                
            current_candle = ohlc_df.iloc[candle_index]
            
            # Check for bullish sweep (take sell stops, then reverse up)
            # Use integer positions instead of index comparison
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
            
            # Check for bearish sweep (take buy stops, then reverse down)  
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
    
    def get_retracement_analysis(self, ohlc_df: pd.DataFrame, swings_df: pd.DataFrame):
        """
        Get retracement analysis using SMC library.
        Returns retracement data for confluence scoring.
        """
        if ohlc_df is None or swings_df is None or ohlc_df.empty or swings_df.empty:
            logger.warning("SMCAnalyzer.get_retracement_analysis: Invalid input data.")
            return None
        
        logger.debug(f"get_retracement_analysis: Analyzing {len(ohlc_df)} candles for retracements")
            
        try:
            retracement_results = smc_package.retracements(
                ohlc_df.copy(),
                swing_highs_lows=swings_df
            )
            
            if retracement_results is None:
                logger.error("get_retracement_analysis: smc_package.retracements returned None")
                return None                
            
            return retracement_results
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_retracement_analysis: Exception: {e}", exc_info=True)
            return None

    def get_comprehensive_analysis(self, ohlc_df: pd.DataFrame, higher_timeframe: str = "4h"):
        """
        Get all SMC analysis in one call for efficiency.
        Returns dictionary with all indicators for professional trading decisions.
        """
        if ohlc_df is None or ohlc_df.empty:
            logger.warning("SMCAnalyzer.get_comprehensive_analysis: Empty input data")
            return {}
                    
        try:            
            swings = self.get_swing_points(ohlc_df)
            if swings is None:
                logger.error("Failed to get swing points - aborting analysis")
                return {}
                
            structure = self.get_market_structure(ohlc_df, swings)
            
            # Advanced analysis
            order_blocks = self.get_order_blocks(ohlc_df, swings)
            fvgs = self.get_fair_value_gaps(ohlc_df)
            liquidity = self.get_liquidity_zones(ohlc_df, swings)
            
            # Retracement analysis
            retracements = self.get_retracement_analysis(ohlc_df, swings)
            
            # Previous timeframe levels using SMC library
            prev_levels = None
            try:
                prev_levels = smc_package.previous_high_low(ohlc_df.copy(), time_frame=higher_timeframe)
                logger.debug("Previous H4 levels calculated successfully")
            except Exception as e:
                logger.warning(f"Could not get previous H4 levels: {e}")
            
            # Session context
            in_killzone, active_sessions = self.detect_active_session(ohlc_df)
            latest_time = ohlc_df.index[-1] if hasattr(ohlc_df, 'index') else ohlc_df['time'].iloc[-1]
            
            # Premium/Discount zones
            eq, prem_high, disc_low = self.get_premium_discount(ohlc_df)
            
            analysis = {
                'ohlc_df': ohlc_df,
                'swings': swings,
                'structure': structure,
                'order_blocks': order_blocks,
                'fair_value_gaps': fvgs,
                'liquidity_zones': liquidity,
                'retracements': retracements,
                'previous_levels': prev_levels,
                'in_killzone': in_killzone,
                'active_sessions': active_sessions,
                'equilibrium': eq,
                'premium_high': prem_high,
                'discount_low': disc_low,
                'timestamp': latest_time
            }
            return analysis
            
        except Exception as e:
            logger.error(f"SMCAnalyzer.get_comprehensive_analysis: Exception: {e}", exc_info=True)
            return {}

class SignalGenerator:
    """
    Professional SMC Signal Generator with Order Block, FVG, and Liquidity confluence.
    Focuses on high-probability setups that institutional traders use.
    """
    
    def __init__(self, smc_analyzer, sl_buffer_points, tp_rr_ratio, higher_timeframe="4h", sl_atr_multiplier=1.0):
        self.smc_analyzer = smc_analyzer
        self.sl_buffer_points = sl_buffer_points
        self.tp_rr_ratio = tp_rr_ratio
        self.higher_timeframe = higher_timeframe
        self.sl_atr_multiplier = sl_atr_multiplier

    def generate(self, ohlc_df_full, symbol_point_size, current_trading_symbol):
        """
        Generate professional SMC trading signals with multiple confluence factors.
        Enhanced for 24/5 trading with dynamic confluence requirements.
        
        Entry Requirements (Professional SMC approach):
        1. Market Structure: BOS/CHoCH in trade direction
        2. Premium/Discount: Price in correct zone for bias
        3. Confluence factors (dynamic based on session):
        - In killzone: Require 2+ factors
        - Outside killzone: Require 1+ factors (more flexible)
        """
        confluence_score = 0
        
        if ohlc_df_full is None or len(ohlc_df_full) < self.smc_analyzer.structure_lookback:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): Insufficient data")
            return None, None, None

        # Get comprehensive SMC analysis
        smc_analysis = self.smc_analyzer.get_comprehensive_analysis(ohlc_df_full, higher_timeframe=self.higher_timeframe)
        if not smc_analysis:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): SMC analysis failed.")
            return None, None, None

        # Extract analysis components
        structure_df = smc_analysis.get('structure', ohlc_df_full)
        order_blocks = smc_analysis.get('order_blocks')
        fvgs = smc_analysis.get('fair_value_gaps')
        liquidity_zones = smc_analysis.get('liquidity_zones')
        in_killzone = smc_analysis.get('in_killzone', False)
        active_sessions = smc_analysis.get('active_sessions', [])
        retracements = smc_analysis.get('retracements')
        previous_levels = smc_analysis.get('previous_levels')
        
        # Dynamic confluence requirement based on session
        min_confluence_required = 2 if in_killzone else 0.5
        
        # Market structure validation
        if 'BOS' not in structure_df.columns or 'CHOCH' not in structure_df.columns:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): Missing structure columns.")
            return None, None, None

        latest_data = structure_df.iloc[-1].copy()
        # Ensure we have a consistent integer index for position-based operations
        if hasattr(latest_data, 'name') and not isinstance(latest_data.name, int):
            # If the name is a timestamp, replace it with the integer position
            latest_data.name = len(structure_df) - 1
        current_price = latest_data['close']
        
        # Premium/Discount analysis
        eq = smc_analysis.get('equilibrium')
        if eq is None:
            logger.warning(f"SignalGenerator ({current_trading_symbol}): No equilibrium level")
            return None, None, None

        # Debug market structure
        recent_bos = structure_df['BOS'].dropna().tail(3).tolist() if 'BOS' in structure_df.columns else []
        recent_choch = structure_df['CHOCH'].dropna().tail(3).tolist() if 'CHOCH' in structure_df.columns else []
        
        # Check for recent structure (not just current candle)
        recent_structure_indices = structure_df.index[-20:]  # Last 10 candles
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
        
        logger.debug(f"Market Structure ({current_trading_symbol}): "
                    f"Recent BOS={recent_bos}, Recent CHOCH={recent_choch}, "
                    f"Current Price={current_price:.5f}, Equilibrium={eq:.5f}, "
                    f"Recent Bull Structure={recent_bullish_structure}, Recent Bear Structure={recent_bearish_structure}")

        # === BULLISH SIGNAL LOGIC ===
        is_in_discount = current_price < eq
        
        if is_in_discount and recent_bullish_structure:
            # Check for bullish confluence factors
            confluence_score, confluence_details = self._calculate_bullish_confluence(
                current_price, order_blocks, fvgs, liquidity_zones, 
                in_killzone, ohlc_df_full, active_sessions, retracements, previous_levels
            )
            
            logger.info(f"BULLISH CHECK ({current_trading_symbol}): "
                    f"Confluence={confluence_score}/{min_confluence_required}, In Discount={is_in_discount}, "
                    f"Recent Structure OK={recent_bullish_structure}, Session={active_sessions}")
            
            if confluence_score >= min_confluence_required:
                # Major factors are weighted >= 1.5 in the confluence logic
                major_factor_present = any(
                    "(+2)" in detail or "(+1.5)" in detail for detail in confluence_details
                )

                if not major_factor_present:
                    logger.info(f"BULLISH SKIP ({current_trading_symbol}): Score ({confluence_score}) is sufficient, "
                                f"but no major confluence factor found. Details: {confluence_details}")
                    return None, None, None # Veto the signal
                
                sl_price, tp_price = self._calculate_bullish_levels(
                    latest_data, current_price, smc_analysis, symbol_point_size
                )
                
                if sl_price and tp_price and sl_price < current_price < tp_price:
                    logger.info(f"BULLISH SIGNAL ({current_trading_symbol}): "
                            f"Confluence={confluence_score}, Session={active_sessions}, "
                            f"Entry={current_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")
                    return "BUY", sl_price, tp_price
                else:
                    logger.warning(f"BULLISH LEVELS INVALID ({current_trading_symbol}): "
                                f"SL={sl_price}, Current={current_price}, TP={tp_price}")

        # === BEARISH SIGNAL LOGIC ===
        is_in_premium = current_price > eq
        
        if is_in_premium and recent_bearish_structure:
            # Check for bearish confluence factors
            confluence_score, confluence_details = self._calculate_bearish_confluence(
                current_price, order_blocks, fvgs, liquidity_zones,
                in_killzone, ohlc_df_full, active_sessions,retracements, previous_levels
            )
            
            logger.info(f"BEARISH CHECK ({current_trading_symbol}): "
                    f"Confluence={confluence_score}/{min_confluence_required}, In Premium={is_in_premium}, "
                    f"Recent Structure OK={recent_bearish_structure}, Session={active_sessions}")
            
            if confluence_score >= min_confluence_required:
                # Major factors are weighted >= 1.5 in the confluence logic
                major_factor_present = any(
                    "(+2)" in detail or "(+1.5)" in detail for detail in confluence_details
                )

                if not major_factor_present:
                    logger.info(f"BEARISH SKIP ({current_trading_symbol}): Score ({confluence_score}) is sufficient, "
                                f"but no major confluence factor found. Details: {confluence_details}")
                    return None, None, None # Veto the signal
                
                sl_price, tp_price = self._calculate_bearish_levels(
                    latest_data, current_price, smc_analysis, symbol_point_size
                )
                
                if sl_price and tp_price and tp_price < current_price < sl_price:
                    logger.info(f"BEARISH SIGNAL ({current_trading_symbol}): "
                            f"Confluence={confluence_score}, Session={active_sessions}, "
                            f"Entry={current_price:.5f}, SL={sl_price:.5f}, TP={tp_price:.5f}")
                    return "SELL", sl_price, tp_price
                else:
                    logger.warning(f"BEARISH LEVELS INVALID ({current_trading_symbol}): "
                                f"TP={tp_price}, Current={current_price}, SL={sl_price}")

        return None, None, None

    # COMPLETE UPDATED BULLISH CONFLUENCE METHOD:
    def _calculate_bullish_confluence(self, current_price, order_blocks, fvgs, 
                                    liquidity_zones, in_killzone, ohlc_df, active_sessions, 
                                    retracements=None, previous_levels=None):
        """
        Calculate confluence factors for bullish signals using SMC library methods.
        Enhanced with retracement analysis and previous timeframe levels.
        """
        confluence = 0
        confluence_details = []
        
        # 1. Session timing - Any active session adds value
        if in_killzone and active_sessions:
            session_boost = min(len(active_sessions), 2)  # Cap at 2
            confluence += session_boost
            confluence_details.append(f"Active killzone sessions: {active_sessions} (+{session_boost})")
        elif active_sessions:
            confluence += 1
            confluence_details.append(f"Sessions active: {active_sessions} (+1)")
            
        # 2. SMC Retracement Analysis (NEW - using SMC library)
        if retracements is not None and 'CurrentRetracement%' in retracements.columns:
            current_direction = retracements['Direction'].iloc[-1] if 'Direction' in retracements.columns else None
            current_retracement_pct = retracements['CurrentRetracement%'].iloc[-1]
            
            # Only add retracement confluence if we're in a bullish trend
            if current_direction == 1 and pd.notna(current_retracement_pct):
                # Perfect retracement zones for bullish entries
                if 38.2 <= current_retracement_pct <= 50:  # Golden zone
                    confluence += 2
                    confluence_details.append(f"Golden retracement zone {current_retracement_pct:.1f}% (+2)")
                elif 50 <= current_retracement_pct <= 61.8:  # Deep retracement
                    confluence += 1.5
                    confluence_details.append(f"Deep retracement {current_retracement_pct:.1f}% (+1.5)")
                elif 23.6 <= current_retracement_pct <= 38.2:  # Shallow retracement
                    confluence += 1
                    confluence_details.append(f"Shallow retracement {current_retracement_pct:.1f}% (+1)")
                elif current_retracement_pct < 23.6:  # Very shallow
                    confluence += 0.5
                    confluence_details.append(f"Minimal retracement {current_retracement_pct:.1f}% (+0.5)")
            elif current_direction == -1:
                # Penalize trading against the trend
                confluence -= 1
                confluence_details.append(f"Against bearish trend direction (-1)")
            else:
                logger.debug(f"Neutral retracement: Direction={current_direction}, Pct={current_retracement_pct}")
        
        # 3. Previous Timeframe Levels (NEW - using SMC library)
        if previous_levels is not None and 'PreviousLow' in previous_levels.columns:
            prev_low = previous_levels['PreviousLow'].iloc[-1]
            if pd.notna(prev_low):
                # Check if current price is near H4 previous low (support)
                distance_pct = abs(current_price - prev_low) / prev_low * 100
                if distance_pct <= 0.1:  # Within 0.1% of H4 previous low
                    confluence += 1.5
                    confluence_details.append(f"At H4 previous low {prev_low:.5f} (+1.5)")
                elif distance_pct <= 0.2:  # Within 0.2%
                    confluence += 1
                    confluence_details.append(f"Near H4 previous low {prev_low:.5f} (+1)")
        
        # 4. Order Block analysis
        if order_blocks is not None and not order_blocks.empty:
            # Consider OBs from the last 30 candles, most recent first
            recent_bull_obs = order_blocks[order_blocks['OB'] == 1].tail(30).iloc[::-1]
            
            for idx, ob in recent_bull_obs.iterrows():
                # Check if current price is reacting to this OB
                if ob['Bottom'] <= current_price <= ob['Top']:
                    is_mitigated = 'MitigatedIndex' in order_blocks.columns and pd.notna(ob.get('MitigatedIndex'))
                    
                    if not is_mitigated:
                        confluence += 2
                        confluence_details.append(f"Fresh bullish OB at {ob['Bottom']:.5f}-{ob['Top']:.5f} (+2)")
                    else:
                        confluence += 1
                        confluence_details.append(f"Mitigated bullish OB at {ob['Bottom']:.5f}-{ob['Top']:.5f} (+1)")
                    
                    # Found the relevant OB, no need to check older ones
                    break
                            
        # 5. Fair Value Gap analysis
        if fvgs is not None and not fvgs.empty:
            open_bull_fvgs = fvgs[
                (fvgs['FVG'] == 1) & 
                (fvgs['MitigatedIndex'].isna() if 'MitigatedIndex' in fvgs.columns else True)
            ]
            if not open_bull_fvgs.empty:
                for idx, fvg in open_bull_fvgs.iterrows():
                    if fvg['Bottom'] <= current_price <= fvg['Top']:
                        confluence += 1.5
                        confluence_details.append(f"Open bullish FVG at {fvg['Bottom']:.5f}-{fvg['Top']:.5f} (+1.5)")
                        break
            
            # Check if price is returning to recent FVG area
            if confluence < 2:
                recent_fvgs = fvgs[
                    (fvgs['FVG'] == 1) & 
                    (fvgs.index >= len(ohlc_df) - 20)
                ]
                for idx, fvg in recent_fvgs.iterrows():
                    if fvg['Bottom'] * 0.999 <= current_price <= fvg['Top'] * 1.001:
                        confluence += 0.5
                        confluence_details.append(f"Price at bullish FVG area (+0.5)")
                        break
                            
        # 6. Liquidity sweep detection
        recent_sweep = self.smc_analyzer.detect_liquidity_sweep(ohlc_df, 
                                                            self.smc_analyzer.get_swing_points(ohlc_df))
        if recent_sweep == "bullish_sweep":
            confluence += 1.5
            confluence_details.append("Recent bullish liquidity sweep (+1.5)")
        
        # 7. Price action confluence
        if len(ohlc_df) >= 3:
            last_3_candles = ohlc_df.tail(3)
            
            # Bullish engulfing
            if (last_3_candles.iloc[-2]['close'] < last_3_candles.iloc[-2]['open'] and
                last_3_candles.iloc[-1]['close'] > last_3_candles.iloc[-1]['open'] and
                last_3_candles.iloc[-1]['close'] > last_3_candles.iloc[-2]['open']):
                confluence += 1
                confluence_details.append("Bullish engulfing pattern (+1)")
            
            # Support rejection (long lower wick)
            current_candle = last_3_candles.iloc[-1]
            body_size = abs(current_candle['close'] - current_candle['open'])
            lower_wick = (current_candle['open'] - current_candle['low'] 
                        if current_candle['close'] > current_candle['open'] 
                        else current_candle['close'] - current_candle['low'])
            if lower_wick > body_size * 2:
                confluence += 1
                confluence_details.append("Support rejection (long lower wick) (+1)")
                            
        logger.debug(f"Bullish confluence factors: {', '.join(confluence_details)} = Total: {confluence}")
        return confluence, confluence_details

    # COMPLETE UPDATED BEARISH CONFLUENCE METHOD:
    def _calculate_bearish_confluence(self, current_price, order_blocks, fvgs, 
                                    liquidity_zones, in_killzone, ohlc_df, active_sessions,
                                    retracements=None, previous_levels=None):
        """
        Calculate confluence factors for bearish signals using SMC library methods.
        Enhanced with retracement analysis and previous timeframe levels.
        """
        confluence = 0
        confluence_details = []
        
        # 1. Session timing
        if in_killzone and active_sessions:
            session_boost = min(len(active_sessions), 2)
            confluence += session_boost
            confluence_details.append(f"Active killzone sessions: {active_sessions} (+{session_boost})")
        elif active_sessions:
            confluence += 1
            confluence_details.append(f"Sessions active: {active_sessions} (+1)")
            
        # 2. SMC Retracement Analysis (NEW - using SMC library)
        if retracements is not None and 'CurrentRetracement%' in retracements.columns:
            current_direction = retracements['Direction'].iloc[-1] if 'Direction' in retracements.columns else None
            current_retracement_pct = retracements['CurrentRetracement%'].iloc[-1]
            if pd.notna(current_retracement_pct):
                current_retracement_pct = max(0, min(100, current_retracement_pct))
            
            # Only add retracement confluence if we're in a bearish trend
            if current_direction == -1 and pd.notna(current_retracement_pct):
                # Perfect retracement zones for bearish entries
                if 38.2 <= current_retracement_pct <= 50:  # Golden zone
                    confluence += 2
                    confluence_details.append(f"Golden retracement zone {current_retracement_pct:.1f}% (+2)")
                elif 50 <= current_retracement_pct <= 61.8:  # Deep retracement
                    confluence += 1.5
                    confluence_details.append(f"Deep retracement {current_retracement_pct:.1f}% (+1.5)")
                elif 23.6 <= current_retracement_pct <= 38.2:  # Shallow retracement
                    confluence += 1
                    confluence_details.append(f"Shallow retracement {current_retracement_pct:.1f}% (+1)")
                elif current_retracement_pct < 23.6:  # Very shallow
                    confluence += 0.5
                    confluence_details.append(f"Minimal retracement {current_retracement_pct:.1f}% (+0.5)")
            elif current_direction == 1:
                # Penalize trading against the trend
                confluence -= 1
                confluence_details.append(f"Against bullish trend direction (-1)")
            else:
                logger.debug(f"Neutral retracement: Direction={current_direction}, Pct={current_retracement_pct}")
        
        # 3. Previous Timeframe Levels (NEW - using SMC library)
        if previous_levels is not None and 'PreviousHigh' in previous_levels.columns:
            prev_high = previous_levels['PreviousHigh'].iloc[-1]
            if pd.notna(prev_high):
                # Check if current price is near H4 previous high (resistance)
                distance_pct = abs(current_price - prev_high) / prev_high * 100
                if distance_pct <= 0.1:  # Within 0.1% of H4 previous high
                    confluence += 1.5
                    confluence_details.append(f"At H4 previous high {prev_high:.5f} (+1.5)")
                elif distance_pct <= 0.2:  # Within 0.2%
                    confluence += 1
                    confluence_details.append(f"Near H4 previous high {prev_high:.5f} (+1)")
        
        # 4. Order Block analysis
        if order_blocks is not None and not order_blocks.empty:
            # Consider OBs from the last 30 candles, most recent first
            recent_bear_obs = order_blocks[order_blocks['OB'] == -1].tail(30).iloc[::-1]

            for idx, ob in recent_bear_obs.iterrows():
                # Check if current price is reacting to this OB
                if ob['Bottom'] <= current_price <= ob['Top']:
                    is_mitigated = 'MitigatedIndex' in order_blocks.columns and pd.notna(ob.get('MitigatedIndex'))
                    
                    if not is_mitigated:
                        confluence += 2
                        confluence_details.append(f"Fresh bearish OB at {ob['Bottom']:.5f}-{ob['Top']:.5f} (+2)")
                    else:
                        confluence += 1
                        confluence_details.append(f"Mitigated bearish OB at {ob['Bottom']:.5f}-{ob['Top']:.5f} (+1)")

                    # Found the relevant OB, no need to check older ones
                    break
                            
        # 5. Fair Value Gap analysis
        if fvgs is not None and not fvgs.empty:
            open_bear_fvgs = fvgs[
                (fvgs['FVG'] == -1) & 
                (fvgs['MitigatedIndex'].isna() if 'MitigatedIndex' in fvgs.columns else True)
            ]
            if not open_bear_fvgs.empty:
                for idx, fvg in open_bear_fvgs.iterrows():
                    if fvg['Bottom'] <= current_price <= fvg['Top']:
                        confluence += 1.5
                        confluence_details.append(f"Open bearish FVG at {fvg['Bottom']:.5f}-{fvg['Top']:.5f} (+1.5)")
                        break
            
            # Check recent FVG areas
            if confluence < 2:
                recent_fvgs = fvgs[
                    (fvgs['FVG'] == -1) & 
                    (fvgs.index >= len(ohlc_df) - 20)
                ]
                for idx, fvg in recent_fvgs.iterrows():
                    if fvg['Bottom'] * 0.999 <= current_price <= fvg['Top'] * 1.001:
                        confluence += 0.5
                        confluence_details.append(f"Price at bearish FVG area (+0.5)")
                        break
                            
        # 6. Liquidity sweep
        recent_sweep = self.smc_analyzer.detect_liquidity_sweep(ohlc_df, 
                                                            self.smc_analyzer.get_swing_points(ohlc_df))
        if recent_sweep == "bearish_sweep":
            confluence += 1.5
            confluence_details.append("Recent bearish liquidity sweep (+1.5)")
        
        # 7. Price action confluence
        if len(ohlc_df) >= 3:
            last_3_candles = ohlc_df.tail(3)
            
            # Bearish engulfing
            if (last_3_candles.iloc[-2]['close'] > last_3_candles.iloc[-2]['open'] and
                last_3_candles.iloc[-1]['close'] < last_3_candles.iloc[-1]['open'] and
                last_3_candles.iloc[-1]['close'] < last_3_candles.iloc[-2]['open']):
                confluence += 1
                confluence_details.append("Bearish engulfing pattern (+1)")
            
            # Resistance rejection (long upper wick)
            current_candle = last_3_candles.iloc[-1]
            body_size = abs(current_candle['close'] - current_candle['open'])
            upper_wick = (current_candle['high'] - current_candle['close'] 
                        if current_candle['close'] < current_candle['open'] 
                        else current_candle['high'] - current_candle['open'])
            if upper_wick > body_size * 2:
                confluence += 1
                confluence_details.append("Resistance rejection (long upper wick) (+1)")
                            
        logger.debug(f"Bearish confluence factors: {', '.join(confluence_details)} = Total: {confluence}")
        return confluence, confluence_details

    def _calculate_bullish_levels(self, latest_data, current_price, smc_analysis, symbol_point_size):
        ohlc_df = smc_analysis.get('ohlc_df')
        
        # Get H4 previous levels
        prev_levels = smc_package.previous_high_low(ohlc_df.copy(), time_frame="4h")
        latest_prev_low = prev_levels['PreviousLow'].iloc[-1]
        
        # Get recent swing low (not signal candle)
        recent_lows = ohlc_df['low'].iloc[-20:-1]  # Exclude current candle
        recent_structural_low = recent_lows.min()
        
        # Use the HIGHER low (more protective for longs)
        if pd.notna(latest_prev_low):
            # If H4 low is above recent lows, use recent low
            # If H4 low is below recent lows, use H4 low
            structural_low = latest_prev_low if latest_prev_low < recent_structural_low else recent_structural_low
        else:
            structural_low = recent_structural_low
        
        # Calculate ATR-based buffer
        atr_series = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        atr = atr_series.iloc[-1] # Get the latest ATR value
        
        dynamic_buffer = atr * self.sl_atr_multiplier
        
        sl_price = structural_low - dynamic_buffer
        
        # Validate minimum distance
        min_risk = max(atr * 0.3, 100 * symbol_point_size)  # At least 10 pips
        risk_amount = current_price - sl_price
        
        if risk_amount < min_risk:
            sl_price = current_price - min_risk
            
        # Calculate TP
        tp_price = current_price + (risk_amount * self.tp_rr_ratio)
        
        return sl_price, tp_price

    def _calculate_bearish_levels(self, latest_data, current_price, smc_analysis, symbol_point_size):
        ohlc_df = smc_analysis.get('ohlc_df')
        
        # Get H4 previous levels
        prev_levels = smc_package.previous_high_low(ohlc_df.copy(), time_frame="4h")
        latest_prev_high = prev_levels['PreviousHigh'].iloc[-1]
        
        # Get recent swing high (not signal candle)
        recent_highs = ohlc_df['high'].iloc[-20:-1]  # Exclude current candle
        recent_structural_high = recent_highs.max()
        
        # Use the LOWER high (more protective for shorts)
        if pd.notna(latest_prev_high):
            # If H4 high is below recent highs, use recent high
            # If H4 high is above recent highs, use H4 high
            structural_high = latest_prev_high if latest_prev_high > recent_structural_high else recent_structural_high
        else:
            structural_high = recent_structural_high
        
        # Calculate ATR-based buffer
        atr_series = ATR(ohlc_df['high'], ohlc_df['low'], ohlc_df['close'], timeperiod=14)
        atr = atr_series.iloc[-1] # Get the latest ATR value
        
        dynamic_buffer = atr * self.sl_atr_multiplier
        
        sl_price = structural_high + dynamic_buffer
        
        # Validate minimum distance
        min_risk = max(atr * 0.3, 100 * symbol_point_size)  # At least 10 pips
        risk_amount = sl_price - current_price
        
        if risk_amount < min_risk:
            sl_price = current_price + min_risk
            
        # Calculate TP
        tp_price = current_price - (risk_amount * self.tp_rr_ratio)
        
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
        magic_number = int(f"{self.magic_prefix}{int(datetime.now().timestamp()) % 100000 + sum(ord(c) for c in symbol[:3]) % 1000}") # More unique
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume), "type": mt5_order_type,
                   "price": price, "sl": float(sl_price), "tp": float(tp_price), "deviation": 20, "magic": magic_number,
                   "comment": f"SMC_BOT_{symbol}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"TradeExecutor ({symbol}): Order send failed. Code {result.retcode if result else 'None'} - {result.comment if result else mt5.last_error()}. Req: {request}")
            return None
        logger.info(f"TradeExecutor ({symbol}): Order sent. Ticket: {result.order}, Deal: {result.deal}")
        return result