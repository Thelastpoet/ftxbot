import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import traceback
import asyncio
import numpy as np
from scipy.signal import argrelextrema
import os
from typing import Tuple
from enum import Enum
import json

from technical_analysis import IndicatorCalculator
from liquidity import LiquidityDetector

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PERIOD = 1000
CHECK_INTERVAL = 300  # Main loop

class DivergenceType(Enum):
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"

class MetaTrader5Client:
    def __init__(self):
        # Initialize the MetaTrader5 connection
        self.initialized = mt5.initialize()
        if self.initialized:
            logging.info("MetaTrader5 initialized successfully.")
        else:
            logging.error("Failed to initialize MetaTrader5.")

    def __del__(self):
        # Shutdown the MetaTrader5 connection when the object is destroyed
        mt5.shutdown()
        logging.info("MetaTrader5 connection shut down.")

    def is_initialized(self):
        # Check if the MetaTrader5 connection is initialized
        return self.initialized

    def get_account_info(self):
        # Retrieve account information from MetaTrader5
        logging.info("Getting account info.")
        return mt5.account_info()

class MarketData:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes
        self.num_candles = {tf: None for tf in timeframes}
        
    def calculate_num_candles(self, timeframe):
        timeframe_in_minutes = timeframe / 60 if timeframe < mt5.TIMEFRAME_H1 else timeframe / 60 / 60
        num_candles = MAX_PERIOD * 2
        self.num_candles[timeframe] = int(num_candles / timeframe_in_minutes)         

    def fetch_data(self, timeframe):
        if self.num_candles[timeframe] is None:
            self.calculate_num_candles(timeframe)

        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.num_candles[timeframe])
        if rates is None:
            logging.error(f"No rates available for {self.symbol} on timeframe {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df = df[df['tick_volume'] != 0]
        df.reset_index(drop=True, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
                        
        return df
           
class SwingPointDetector:
    def __init__(self, timeframe, lookback_period=20, prominence_factor=0.001):
        self.timeframe = timeframe
        self.lookback_period = lookback_period 
        
        self.prominence_factor = prominence_factor

    def find_swing_points(self, data):
        """
        Finds swing high and swing low points in price data.
        """
        if not isinstance(data, pd.DataFrame) or not {'high', 'low'}.issubset(data.columns) or data.empty:
            logging.error("Invalid input data. Must be a non-empty DataFrame with 'high' and 'low' columns.")
            return None

        if len(data) < self.lookback_period:
            logging.warning("Not enough data to calculate swing points.")
            return None
        
        price_range = data['high'].max() - data['low'].min()
        self.prominence = price_range * self.prominence_factor        

        try:
            high_indices = argrelextrema(data['high'].values, np.greater_equal, order=self.lookback_period)[0]
            swing_highs = data.iloc[high_indices]

            low_indices = argrelextrema(data['low'].values, np.less_equal, order=self.lookback_period)[0]
            swing_lows = data.iloc[low_indices]

            swing_highs = self.filter_by_prominence(data, swing_highs, 'high')
            swing_lows = self.filter_by_prominence(data, swing_lows, 'low')

            return {'highs': swing_highs, 'lows': swing_lows}

        except Exception as e:
            logging.error(f"Error finding swing points: {e}")
            return None

    def filter_by_prominence(self, data, swing_points, price_type):
        if swing_points.empty:
            return swing_points

        indices = swing_points.index
        prices = swing_points[price_type].values

        left_indices = np.searchsorted(data.index, indices, side='left')
        left_prices = []
        for i, left_index in enumerate(left_indices):
            if left_index > 0:
                left_prices.append(data.iloc[left_index - 1][price_type])
            else:
                left_prices.append(prices[i])
        left_prices = np.array(left_prices)

        right_indices = np.searchsorted(data.index, indices, side='right')
        right_prices = []
        for i, right_index in enumerate(right_indices):
            if right_index < len(data):
                right_prices.append(data.iloc[right_index][price_type])
            else:
                right_prices.append(prices[i])
        right_prices = np.array(right_prices)

        prominence = np.minimum(np.abs(prices - left_prices), np.abs(prices - right_prices))
        filtered_indices = indices[prominence >= self.prominence]
        filtered_swing_points = swing_points.loc[filtered_indices]

        return filtered_swing_points

    def analyze_price_action(self, data, swing_points, trend_lookback=3):
        """Analyzes price action for trend and consolidation using multiple swing points."""
        if swing_points is None or data is None or data.empty:
            return None

        highs = swing_points.get('highs', pd.DataFrame())
        lows = swing_points.get('lows', pd.DataFrame())

        if highs.empty or lows.empty or len(highs) < trend_lookback or len(lows) < trend_lookback:
            return {'trend': 'Insufficient Data', 'consolidation': False}

        price_action_info = {'trend': None, 'consolidation': False}

        highs_subset = highs.tail(trend_lookback)
        lows_subset = lows.tail(trend_lookback)
        high_trend = np.polyfit(range(len(highs_subset)), highs_subset['high'], 1)[0]
        low_trend = np.polyfit(range(len(lows_subset)), lows_subset['low'], 1)[0]

        if high_trend > 0 and low_trend > 0:
            price_action_info['trend'] = 'bullish'
        elif high_trend < 0 and low_trend < 0:
            price_action_info['trend'] = 'bearish'
        else:
            price_action_info['trend'] = 'sideways'

        recent_highs = highs.tail(3)
        recent_lows = lows.tail(3)

        if not recent_highs.empty and not recent_lows.empty:
            max_high = recent_highs['high'].max()
            min_low = recent_lows['low'].min()
            range_percentage = (max_high - min_low) / min_low * 100
            consolidation_threshold = 1.0
            if range_percentage <= consolidation_threshold:
                price_action_info['consolidation'] = True

        return price_action_info
    
class DivergenceDetector:
    def __init__(self, order=5, k=2):
        self.order = order

    def to_numpy(self, data):
        return data.values if hasattr(data, 'values') else np.array(data)

    def find_higher_highs(self, data):
        data = self.to_numpy(data)
        extrema_indices = argrelextrema(data, np.greater, order=self.order)[0]
        result = []
        for i in range(len(extrema_indices)):
            for j in range(i + 1, len(extrema_indices)):
                idx1, idx2 = extrema_indices[i], extrema_indices[j]
                if data[idx2] > data[idx1]:
                    result.append([idx1, idx2])
        return result

    def find_lower_lows(self, data):
        data = self.to_numpy(data)
        extrema_indices = argrelextrema(data, np.less, order=self.order)[0]
        result = []
        for i in range(len(extrema_indices)):
            for j in range(i + 1, len(extrema_indices)):
                idx1, idx2 = extrema_indices[i], extrema_indices[j]
                if data[idx2] < data[idx1]:
                    result.append([idx1, idx2])
        return result

    def find_lower_highs(self, data):
        data = self.to_numpy(data)
        extrema_indices = argrelextrema(data, np.greater, order=self.order)[0]
        result = []
        for i in range(len(extrema_indices)):
            for j in range(i + 1, len(extrema_indices)):
                idx1, idx2 = extrema_indices[i], extrema_indices[j]
                if data[idx2] < data[idx1]:
                    result.append([idx1, idx2])
        return result

    def find_higher_lows(self, data):
        data = self.to_numpy(data)
        extrema_indices = argrelextrema(data, np.less, order=self.order)[0]
        result = []
        for i in range(len(extrema_indices)):
            for j in range(i + 1, len(extrema_indices)):
                idx1, idx2 = extrema_indices[i], extrema_indices[j]
                if data[idx2] > data[idx1]:
                    result.append([idx1, idx2])
        return result

    def check_divergence(self, price_data, rsi_data, proximity=5):
        price_higher_highs = self.find_higher_highs(price_data)
        price_lower_lows = self.find_lower_lows(price_data)
        price_lower_highs = self.find_lower_highs(price_data)
        price_higher_lows = self.find_higher_lows(price_data)

        rsi_higher_highs = self.find_higher_highs(rsi_data)
        rsi_lower_lows = self.find_lower_lows(rsi_data)
        rsi_lower_highs = self.find_lower_highs(rsi_data)
        rsi_higher_lows = self.find_higher_lows(rsi_data)

        regular_bullish = any(
            abs(p_lows[0] - r_lows[0]) < proximity and abs(p_lows[1] - r_lows[1]) < proximity
            for p_lows in price_lower_lows
            for r_lows in rsi_higher_lows
        )
        regular_bearish = any(
            abs(p_highs[0] - r_highs[0]) < proximity and abs(p_highs[1] - r_highs[1]) < proximity
            for p_highs in price_higher_highs
            for r_highs in rsi_lower_highs
        )
        hidden_bullish = any(
            abs(p_lows[0] - r_lows[0]) < proximity and abs(p_lows[1] - r_lows[1]) < proximity
            for p_lows in price_higher_lows
            for r_lows in rsi_lower_lows
        )
        hidden_bearish = any(
            abs(p_highs[0] - r_highs[0]) < proximity and abs(p_highs[1] - r_highs[1]) < proximity
            for p_highs in price_lower_highs
            for r_highs in rsi_higher_highs
        )

        regular_divergence = DivergenceType.REGULAR_BULLISH if regular_bullish else \
                            DivergenceType.REGULAR_BEARISH if regular_bearish else None
        hidden_divergence = DivergenceType.HIDDEN_BULLISH if hidden_bullish else \
                            DivergenceType.HIDDEN_BEARISH if hidden_bearish else None

        return regular_divergence, hidden_divergence

class TradeManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.trade_logger = TradeLogger()
        self.indicator_calc = IndicatorCalculator()
        self.liquidity_detector = LiquidityDetector()
        self.order_manager = OrderManager(client, market_data, self.indicator_calc, self.trade_logger, self.liquidity_detector)
        self.divergence_detector = DivergenceDetector(order=5, k=2)        
        
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1]
        self.tf_lower = min(self.timeframes)
        
        self.max_positions = 20
        
    def analyze_market_structure(self, symbol, data, timeframe):
        swing_detector = SwingPointDetector(timeframe=timeframe, lookback_period=20)
        try:
            swing_points = swing_detector.find_swing_points(data)
            if not swing_points:
                return None
                
            price_action = swing_detector.analyze_price_action(data, swing_points)
            
            price_levels = self.indicator_calc.identify_support_resistance_levels(data)
            
            key_levels = []
            if price_levels:
                current_price = data['close'].iloc[-1]
                for level in price_levels:
                    if level > current_price:
                        key_levels.append({'price': level, 'type': 'resistance'})
                    else:
                        key_levels.append({'price': level, 'type': 'support'})
            
            structure = {
                'trend': price_action['trend'],
                'consolidation': price_action['consolidation'],
                'swing_points': swing_points,
                'key_levels': key_levels,
                'timeframe': timeframe
            }
            return structure
        except Exception as e:
            logging.error(f"Error in market structure analysis: {str(e)}")
            return None
        
    def _analyze_divergence(self, data_dict, indicators_dict, direction):
        try:
            higher_tf_data = data_dict[self.tf_higher]
            higher_tf_indicators = indicators_dict[self.tf_higher]
            
            price_data = higher_tf_data['close'].tail(100)
            rsi_data = higher_tf_indicators['rsi'].tail(100)
            
            regular_div, hidden_div = self.divergence_detector.check_divergence(price_data, rsi_data, proximity=5)
            
            if direction == 'buy':
                if hidden_div == DivergenceType.HIDDEN_BULLISH:
                    return True
                if regular_div == DivergenceType.REGULAR_BEARISH:
                    return False
            elif direction == 'sell':
                if hidden_div == DivergenceType.HIDDEN_BEARISH:
                    return True
                if regular_div == DivergenceType.REGULAR_BULLISH:
                    return False
            return None
        except Exception as e:
            logging.error(f"Error in divergence analysis: {str(e)}")
            return None
                        
    def analyze_timeframe_alignment(self, symbol, data_dict, indicators_dict):
        try:
            alignment = {}
            for tf in [self.tf_higher, self.tf_medium, self.tf_lower]:
                data = data_dict[tf]
                indicators = indicators_dict[tf]
                
                # --- Calculate market structure for BOTH Higher and Medium TFs ---
                if tf == self.tf_higher or tf == self.tf_medium:
                    structure_analysis = self.analyze_market_structure(symbol, data, tf)
                    if tf == self.tf_higher:
                        alignment[tf] = {'structure': structure_analysis}
                    else: # Medium Timeframe
                        alignment[tf] = {
                            'structure': structure_analysis,
                            'trend': self.determine_overall_trend(symbol, data, indicators)
                        }
                elif tf == self.tf_lower:
                    alignment[tf] = {'momentum': self._analyze_momentum(symbol, data, indicators)}
            return alignment
        except Exception as e:
            logging.error(f"Error in timeframe alignment analysis: {str(e)}")
            return None
            
    def _analyze_momentum(self, symbol, data, indicators):
        try:
            rsi = indicators['rsi'].iloc[-1]
            rsi_trend = 'bullish' if rsi > 50 else 'bearish'
            adx = indicators['adx'].iloc[-1]
            trend_strength = 'strong' if adx > 25 else 'weak'
            return {'rsi_trend': rsi_trend, 'trend_strength': trend_strength}
        except Exception as e:
            logging.error(f"Error analyzing momentum: {str(e)}")
            return None
        
    def determine_overall_trend(self, symbol, data, indicators):        
        bullish_conditions = (
            data['close'].iloc[-1] > indicators['ichimoku_senkou_span_a'].iloc[-1] and
            data['close'].iloc[-1] > indicators['ichimoku_senkou_span_b'].iloc[-1] and
            indicators['ichimoku_tenkan_sen'].iloc[-1] > indicators['ichimoku_kijun_sen'].iloc[-1]
        )
        bearish_conditions = (
            data['close'].iloc[-1] < indicators['ichimoku_senkou_span_a'].iloc[-1] and
            data['close'].iloc[-1] < indicators['ichimoku_senkou_span_b'].iloc[-1] and
            indicators['ichimoku_tenkan_sen'].iloc[-1] < indicators['ichimoku_kijun_sen'].iloc[-1]
        )                             
        if bullish_conditions:
            return "bullish"
        elif bearish_conditions:
            return "bearish"
        else:
            return "neutral"

    async def check_for_signals(self, symbol):
        try:
            data, indicators = self._prepare_data(symbol)
            if not data or not indicators: return

            alignment = self.analyze_timeframe_alignment(symbol, data, indicators)
            if not alignment: return

            # Get liquidity levels
            liquidity_levels = self.liquidity_detector.get_liquidity_levels(data[self.tf_medium], {}, daily_df=data[self.tf_higher])

            trade_setup = self._evaluate_trade_setup(symbol, data, indicators, alignment)
            if not trade_setup or not trade_setup.get('valid'): return

            await self._execute_trade(symbol, trade_setup, indicators, alignment, liquidity_levels)
        except Exception as e:
            logging.error(f"Error in check_for_signals for {symbol}: {str(e)}\n{traceback.format_exc()}")

    def _prepare_data(self, symbol):
        try:
            data = {tf: self.market_data.fetch_data(tf) for tf in self.timeframes}
            if any(d is None for d in data.values()): return None, None
            indicators = {tf: self.indicator_calc.calculate_indicators(d) for tf, d in data.items()}
            if any(i is None for i in indicators.values()): return None, None
            return data, indicators
        except Exception as e:
            logging.error(f"Error preparing data for {symbol}: {str(e)}")
            return None, None

    async def _execute_trade(self, symbol, trade_setup, indicators, alignment, liquidity_levels):
        try:
            can_trade, reason = self.check_position_limit(symbol)
            if not can_trade:
                logging.info(f"[{symbol}] Trade execution halted: {reason}")
                return
            if self._has_opposing_position(symbol, trade_setup['direction']):
                logging.info(f"[{symbol}] Trade execution halted: Opposing position exists.")
                return

            order_params = self._calculate_order_parameters(symbol, trade_setup, indicators, alignment, liquidity_levels)
            if not order_params:
                logging.error(f"[{symbol}] Failed to calculate order parameters.")
                return

            trade_result = self.order_manager.place_order(
                symbol, order_params['lot_size'], trade_setup['direction'],
                order_params['stop_loss'], order_params['take_profit']
            )
            
            if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                market_context = {
                    'alignment': {k: v for k, v in alignment.items() if k in self.timeframes},
                    'indicators': {tf: ind.iloc[-1].to_dict() for tf, ind in indicators.items()}
                }
                
                # Convert non-serializable DataFrames in swing_points to a list of dicts
                try:
                    # Iterate over both higher and medium timeframes as both can have structure analysis
                    for tf_to_process in [self.tf_higher, self.tf_medium]:
                        if tf_to_process in market_context['alignment'] and 'structure' in market_context['alignment'][tf_to_process]:
                            structure = market_context['alignment'][tf_to_process].get('structure')
                            if structure and 'swing_points' in structure:
                                swing_points = structure['swing_points']
                                if swing_points and isinstance(swing_points, dict):
                                    if 'highs' in swing_points and isinstance(swing_points['highs'], pd.DataFrame):
                                        swing_points['highs'] = swing_points['highs'].reset_index().to_dict('records')
                                    if 'lows' in swing_points and isinstance(swing_points['lows'], pd.DataFrame):
                                        swing_points['lows'] = swing_points['lows'].reset_index().to_dict('records')
                except Exception as e:
                    logging.error(f"Could not process market_context for JSON serialization: {e}")
                    market_context = {"error": "context_serialization_failed"}

                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return obj

                self.trade_logger.log_open_trade(
                    ticket_id=trade_result.order,
                    symbol=symbol,
                    direction=trade_setup['direction'],
                    open_price=trade_result.price,
                    stop_loss=order_params['stop_loss'],
                    take_profit=order_params['take_profit'],
                    lot_size=order_params['lot_size'],
                    reason=trade_setup['setup_type'],
                    market_context=json.dumps(market_context, default=convert_numpy, indent=2)
                )
                logging.info(f"[{symbol}] Successfully opened trade {trade_result.order}.")

        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}\n{traceback.format_exc()}")
            
    async def manage_open_positions(self, symbol: str):
        """Checks for closed positions and updates the log."""
        try:
            open_logged_trades = self.trade_logger.get_open_trades(symbol)
            if open_logged_trades.empty:
                return

            # Get currently open positions from MT5
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                # This could mean no positions or an error; assume no positions for this symbol
                open_mt5_tickets = set()
            else:
                open_mt5_tickets = {pos.ticket for pos in positions}

            # Find trades that are in our log as 'open' but not in MT5 anymore
            for index, trade in open_logged_trades.iterrows():
                ticket_id = trade['ticket_id']
                if ticket_id not in open_mt5_tickets:
                    # This trade has been closed, get history
                    deals = mt5.history_deals_get(position=ticket_id)
                    if deals:
                        # The last deal on a position is usually the closing one
                        closing_deal = deals[-1]
                        close_price = closing_deal.price
                        close_time = pd.to_datetime(closing_deal.time, unit='s')
                        pnl = closing_deal.profit
                        
                        # Determine reason for closing (very basic)
                        status = "closed_manual"
                        if abs(close_price - trade['take_profit']) < 0.0001: # Check for TP hit
                           status = "closed_tp"
                        elif abs(close_price - trade['stop_loss']) < 0.0001: # Check for SL hit
                           status = "closed_sl"

                        self.trade_logger.log_close_trade(
                            ticket_id=ticket_id,
                            close_price=close_price,
                            close_time=close_time,
                            pnl=pnl,
                            status=status
                        )
                        logging.info(f"[{symbol}] Logged closure for trade {ticket_id}. PnL: {pnl:.2f}")

        except Exception as e:
            logging.error(f"Error managing open positions for {symbol}: {str(e)}\n{traceback.format_exc()}")


    def _has_opposing_position(self, symbol, direction):
        symbol_positions = mt5.positions_get(symbol=symbol)
        if not symbol_positions: return False
        for position in symbol_positions:
            existing_type = 'buy' if position.type == mt5.POSITION_TYPE_BUY else 'sell'
            if existing_type != direction: return True
        return False

    def _calculate_order_parameters(self, symbol, trade_setup, indicators, alignment, liquidity_levels):
        try:
            account_info = self.client.get_account_info()
            if not account_info:
                logging.error(f"[{symbol}] Could not retrieve account info.")
                return None
            
            atr_value = indicators[self.tf_medium]['atr'].iloc[-1]
            key_levels = alignment[self.tf_higher]['structure']['key_levels']
            setup_swing_points = alignment[self.tf_medium]['structure']['swing_points']
            
            is_valid, stop_loss, take_profit, stop_loss_pips = self.order_manager.calculate_sl_tp(
                symbol, trade_setup['direction'], atr_value, key_levels, setup_swing_points, liquidity_levels, rr_ratio=1.2
            )
            if not is_valid:
                logging.info(f"[{symbol}] Trade setup invalidated due to poor risk/reward or obstructed path.")
                return None

            lot_size = self.order_manager.calculate_lot_size(
                symbol, account_info.balance, 1.0, stop_loss_pips
            )
            if lot_size is None or lot_size <= 0:
                logging.error(f"[{symbol}] Calculated lot size is invalid: {lot_size}")
                return None

            return {'lot_size': lot_size, 'stop_loss': stop_loss, 'take_profit': take_profit}
        except Exception as e:
            logging.error(f"Error calculating order parameters for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return None
            
    def _get_market_regime(self, symbol, indicators):
        try:
            data = indicators[self.tf_higher]
            if data is None: return 'unknown'
            adx = data['adx'].iloc[-1]
            if adx > 25: return 'trending'
            atr_short = data['atr'].rolling(window=10).mean().iloc[-1]
            atr_long = data['atr'].rolling(window=50).mean().iloc[-1]
            if atr_short > atr_long * 1.5: return 'volatile'
            return 'ranging'
        except Exception as e:
            logging.error(f"Error getting market regime for {symbol}: {str(e)}")
            return 'unknown'

    def _evaluate_trade_setup(self, symbol, data, indicators, alignment):
        try:
            market_regime = self._get_market_regime(symbol, indicators)
            if market_regime == 'volatile':
                logging.info(f"[{symbol}] Trade setup invalid: Market is too volatile.")
                return None
            elif market_regime == 'trending':
                return self._evaluate_trending_setup(symbol, data, indicators, alignment)
            elif market_regime == 'ranging':
                return self._evaluate_ranging_setup(symbol, data, indicators, alignment)
            else:
                return None
        except Exception as e:
            logging.error(f"Error evaluating trade setup for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return None

    def _evaluate_trending_setup(self, symbol, data, indicators, alignment):
        try:
            higher_trend = alignment[self.tf_higher]['structure']['trend']
            medium_trend = alignment[self.tf_medium]['trend']

            if higher_trend not in ['bullish', 'bearish']:
                return None
            if (higher_trend == 'bullish' and medium_trend == 'bearish') or \
               (higher_trend == 'bearish' and medium_trend == 'bullish'):
                return None
            if alignment[self.tf_higher]['structure']['consolidation']:
                return None

            trade_direction = 'buy' if higher_trend == 'bullish' else 'sell'

            med_indicators = indicators[self.tf_medium]
            current_price = data[self.tf_medium]['close'].iloc[-1]
            long_ema = med_indicators['ema_long'].iloc[-1]
            atr = med_indicators['atr'].iloc[-1]
            if not abs(current_price - long_ema) < (atr * 1.5):
                return None
            
            lower_momentum = alignment[self.tf_lower]['momentum']
            if lower_momentum['rsi_trend'] != trade_direction:
                return None            

            divergence_signal = self._analyze_divergence(data, indicators, trade_direction)
            if divergence_signal is False:
                return None
                
            confluence_note = "A+ (Hidden Divergence Confirmed)" if divergence_signal is True else "Standard"
            
            return {
                'valid': True, 'symbol': symbol, 'direction': trade_direction,
                'higher_tf_trend': higher_trend, 'setup_type': f'quality_trend_pullback_{confluence_note}'
            }
        except Exception as e:
            logging.error(f"An exception occurred during trade evaluation for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return None

    def _evaluate_ranging_setup(self, symbol, data, indicators, alignment):
        try:
            higher_tf_structure = alignment[self.tf_higher]['structure']
            key_levels = higher_tf_structure['key_levels']
            if not key_levels: return None

            support_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'support']
            resistance_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'resistance']
            if not support_levels or not resistance_levels: return None

            current_price = data[self.tf_lower]['close'].iloc[-1]
            price_range = resistance_levels[-1] - support_levels[0]

            if abs(current_price - resistance_levels[-1]) < price_range * 0.1:
                if self._validate_sell(symbol, indicators[self.tf_lower]):
                    return {'valid': True, 'symbol': symbol, 'direction': 'sell', 'higher_tf_trend': 'ranging', 'setup_type': 'ranging'}
            elif abs(current_price - support_levels[0]) < price_range * 0.1:
                if self._validate_buy(symbol, indicators[self.tf_lower]):
                    return {'valid': True, 'symbol': symbol, 'direction': 'buy', 'higher_tf_trend': 'ranging', 'setup_type': 'ranging'}
            return None
        except Exception as e:
            logging.error(f"Error evaluating ranging setup for {symbol}: {str(e)}")
            return None

    def _validate_buy(self, symbol: str, data: pd.DataFrame) -> bool:
        try:
            rsi = data['rsi'].iloc[-1]
            if rsi > 50:
                logging.info(f"[{symbol}] Ranging Buy Validation Passed.")
                return True
            logging.info(f"[{symbol}] Ranging Buy Validation Failed: RSI is {rsi:.2f}.")
            return False
        except Exception as e:
            logging.error(f"Error validating buy entry for {symbol}: {str(e)}")
            return False

    def _validate_sell(self, symbol: str, data: pd.DataFrame) -> bool:
        try:
            rsi = data['rsi'].iloc[-1]
            if rsi < 50:
                logging.info(f"[{symbol}] Ranging Sell Validation Passed.")
                return True
            logging.info(f"[{symbol}] Ranging Sell Validation Failed: RSI is {rsi:.2f}.")
            return False
        except Exception as e:
            logging.error(f"Error validating sell entry for {symbol}: {str(e)}")
            return False
    
    def check_position_limit(self, symbol: str) -> Tuple[bool, str]:
        """Check if we can open new positions based on maximum allowed positions per symbol
        
        Args:
            symbol: The trading symbol to check positions for
            
        Returns:
            Tuple[bool, str]: (can_trade, reason)
                - can_trade: True if we can open new position, False otherwise
                - reason: Explanation string
        """
        try:
            # Get all current positions
            positions = mt5.positions_get()
            
            if positions is None:
                return True, "No positions found"
                
            total_positions = len(positions)
            
            # Check overall position limit first
            if total_positions >= self.max_positions:
                return False, f"Max overall positions limit reached ({self.max_positions})"
                
            # Count positions for the specific symbol
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            symbol_position_count = len(symbol_positions)
            
            # Maximum 2 positions per symbol
            if symbol_position_count >= 2:
                return False, f"Max positions limit reached for {symbol} ({symbol_position_count}/2)"
                
            return True, f"Position limit ok (Total: {total_positions}/{self.max_positions}, {symbol}: {symbol_position_count}/2)"
                
        except Exception as e:
            logging.error(f"Error checking position limit: {str(e)}")
            return False, f"Error checking position limit: {str(e)}"

class TradeLogger:
    def __init__(self, filename='trade_log.csv'):
        self.filename = filename
        self.columns = [
            'ticket_id', 'symbol', 'direction', 'open_time', 'open_price', 
            'stop_loss', 'take_profit', 'lot_size', 'reason', 'market_context',
            'close_time', 'close_price', 'pnl', 'status'
        ]
        self._initialize_file()

    def _initialize_file(self):
        """Create the log file with a header if it doesn't exist."""
        try:
            if not os.path.exists(self.filename):
                df = pd.DataFrame(columns=self.columns)
                df.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error initializing trade log: {str(e)}")

    def _load_log(self) -> pd.DataFrame:
        """Load the CSV log file into a pandas DataFrame."""
        try:
            return pd.read_csv(self.filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=self.columns)
        except Exception as e:
            logging.error(f"Error loading trade log: {e}")
            return pd.DataFrame(columns=self.columns)

    def _save_log(self, df: pd.DataFrame):
        """Save the DataFrame back to the CSV log file."""
        try:
            df.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error saving trade log: {e}")
            
    def get_open_trades(self, symbol: str) -> pd.DataFrame:
        """Gets all trades logged as 'open' for a given symbol."""
        log_df = self._load_log()
        if log_df.empty:
            return pd.DataFrame()
        return log_df[(log_df['symbol'] == symbol) & (log_df['status'] == 'open')]

    def log_open_trade(self, ticket_id: int, symbol: str, direction: str, open_price: float, 
                       stop_loss: float, take_profit: float, lot_size: float, reason: str, market_context: str):
        """Logs a new trade when it is opened."""
        try:
            log_df = self._load_log()
            
            # Check if trade with this ticket ID is already logged
            if not log_df[log_df['ticket_id'] == ticket_id].empty:
                logging.warning(f"Trade with ticket_id {ticket_id} already exists in log. Ignoring.")
                return

            new_trade = pd.DataFrame([{
                'ticket_id': ticket_id,
                'symbol': symbol,
                'direction': direction,
                'open_time': datetime.now(),
                'open_price': open_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'lot_size': lot_size,
                'reason': reason,
                'market_context': market_context,
                'close_time': None,
                'close_price': None,
                'pnl': None,
                'status': 'open'
            }])
            
            log_df = pd.concat([log_df, new_trade], ignore_index=True)
            self._save_log(log_df)
            logging.info(f"Logged new open trade {ticket_id} for {symbol}.")
        except Exception as e:
            logging.error(f"Error logging open trade: {str(e)}")

    def log_close_trade(self, ticket_id: int, close_price: float, close_time: datetime, pnl: float, status: str):
        """Updates a trade record with closing information."""
        try:
            log_df = self._load_log()
            trade_index = log_df.index[log_df['ticket_id'] == ticket_id].tolist()

            if not trade_index:
                logging.warning(f"Could not find trade with ticket_id {ticket_id} to close.")
                return

            idx = trade_index[0]
            log_df.loc[idx, 'close_price'] = close_price
            log_df.loc[idx, 'close_time'] = close_time
            log_df.loc[idx, 'pnl'] = pnl
            log_df.loc[idx, 'status'] = status
            
            self._save_log(log_df)
        except Exception as e:
            logging.error(f"Error logging closed trade: {str(e)}")

class OrderManager:
    def __init__( self, client, market_data, indicator_calc, trade_logger, liquidity_detector):
        self.client = client
        self.market_data = market_data
        self.indicator_calc = indicator_calc
        self.trade_logger = trade_logger
        self.liquidity_detector = liquidity_detector

    def calculate_sl_tp(self, symbol, order_type, atr_value, key_levels, setup_swing_points, liquidity_levels, rr_ratio=1.5):
        symbol_tick = mt5.symbol_info_tick(symbol)
        symbol_info = mt5.symbol_info(symbol)
        point = symbol_info.point
        stop_loss = None
        spread_in_price = symbol_info.spread * point
        sl_buffer = (atr_value * 0.2) + spread_in_price

        if order_type == 'buy':
            current_price = symbol_tick.ask
            swing_lows = setup_swing_points.get('lows')
            if swing_lows is None or swing_lows.empty:
                logging.warning(f"[{symbol}] Cannot set SL for BUY: No H1 swing lows found.")
                return False, None, None, None
            last_swing_low = swing_lows['low'].iloc[-1]
            stop_loss = last_swing_low - sl_buffer
        else: # SELL
            current_price = symbol_tick.bid
            swing_highs = setup_swing_points.get('highs')
            if swing_highs is None or swing_highs.empty:
                logging.warning(f"[{symbol}] Cannot set SL for SELL: No H1 swing highs found.")
                return False, None, None, None
            last_swing_high = swing_highs['high'].iloc[-1]
            stop_loss = last_swing_high + sl_buffer

        if (order_type == 'buy' and stop_loss >= current_price) or \
           (order_type == 'sell' and stop_loss <= current_price):
            logging.warning(f"[{symbol}] Invalid SL. SL {stop_loss} is on wrong side of Price {current_price}.")
            return False, None, None, None
                
        stop_loss_pips = abs(current_price - stop_loss) / point
        if stop_loss_pips < 1:
            return False, None, None, None

        # --- Take Profit Calculation using Liquidity ---
        trade_bias = 'bullish' if order_type == 'buy' else 'bearish'
        
        # Get the best liquidity target
        target_liquidity = self.liquidity_detector.get_target_for_bias(
            bias=trade_bias,
            liquidity_levels=liquidity_levels,
            entry_price=current_price,
            min_rr=rr_ratio,
            sl_price=stop_loss
        )

        if not target_liquidity:
            logging.warning(f"[{symbol}] No suitable liquidity target found for {trade_bias} trade. Invalidating setup.")
            return False, None, None, None
        
        take_profit = target_liquidity['level']
        logging.info(f"[{symbol}] Liquidity target found for {trade_bias} trade: {target_liquidity['description']} at {take_profit:.5f}")

        # Final validation of TP
        if (order_type == 'buy' and take_profit <= current_price) or \
           (order_type == 'sell' and take_profit >= current_price):
            logging.warning(f"[{symbol}] Invalid TP calculated. TP {take_profit} is on the wrong side of or equal to Price {current_price}.")
            return False, None, None, None

        return True, stop_loss, take_profit, stop_loss_pips
        
    def calculate_lot_size(self, symbol, account_balance, risk_percentage, stop_loss_pips):
        risk_amount = account_balance * (risk_percentage / 100)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None: return None
        
        contract_size = symbol_info.trade_contract_size
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        
        pip_value = tick_value if tick_size == point else (tick_value / tick_size) * point
        
        # Correctly calculate pips value for the risk amount
        risk_in_pips_value = stop_loss_pips * pip_value
        if risk_in_pips_value == 0: return None
        
        lot_size = risk_amount / risk_in_pips_value
        
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step

        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_size = max(min(lot_size, max_lot), min_lot)

        return lot_size
        
    def place_order(self, symbol, lots_size, order_type, stop_loss, take_profit):        
        symbol_tick = mt5.symbol_info_tick(symbol)
        if not symbol_tick: return None
                
        magic = 234000
        comment = "TradeManager market order"
        
        if order_type == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            entry_price = symbol_tick.ask
        elif order_type == "sell":
            mt5_order_type = mt5.ORDER_TYPE_SELL
            entry_price = symbol_tick.bid
        else: return None
                             
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots_size,
            "type": mt5_order_type, "price": entry_price, "sl": stop_loss, "tp": take_profit,
            "magic": magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
                    
        result = mt5.order_send(request)  
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to place {order_type} order: {result.comment} Lot size: {lots_size}")
        return result
        
async def run_main_loop(client, symbols, timeframes):
    market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
    trade_managers = {symbol: TradeManager(client, market_data_dict[symbol]) for symbol in symbols}

    while True:
        start_time = time.time()
        tasks = []
        for symbol in symbols:
            # Schedule signal checking and position management to run concurrently for each symbol
            tasks.append(trade_managers[symbol].check_for_signals(symbol))
            tasks.append(trade_managers[symbol].manage_open_positions(symbol))
        
        await asyncio.gather(*tasks)
            
        end_time = time.time()
        
        cycle_duration = end_time - start_time
        logging.info(f"Main loop cycle complete. Took {cycle_duration:.2f} seconds.")
        
        sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
        logging.info(f"Next main loop cycle in {sleep_time:.2f} seconds.")
        await asyncio.sleep(sleep_time)


async def main():
    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("Failed to connect to MetaTrader 5")
        return

    symbols = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'EURCAD', 'AUDCHF', 'AUDCAD', 'EURGBP', 'EURAUD', 
               'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDUSD']
    timeframes = (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)

    try:
        await run_main_loop(client, symbols, timeframes)
    except asyncio.CancelledError:
        logging.info("Main task was cancelled. Shutting down...")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}\n{traceback.format_exc()}")
    finally:
        if client.is_initialized(): mt5.shutdown()
        logging.info("Trading script terminated")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Shutting down...")
    except RuntimeError as e:
        logging.error(f"RuntimeError: {e}")