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

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PERIOD = 1000
CHECK_INTERVAL = CONFIG['trading_settings']['main_loop_interval_seconds']

TIMEFRAME_MAP = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1
}

class DivergenceType(Enum):
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"

class MetaTrader5Client:
    def __init__(self):
        self.initialized = mt5.initialize()
        if self.initialized:
            logging.info("MetaTrader5 initialized successfully.")
        else:
            logging.error("Failed to initialize MetaTrader5.")

    def __del__(self):
        mt5.shutdown()
        logging.info("MetaTrader5 connection shut down.")

    def is_initialized(self):
        return self.initialized

    def get_account_info(self):
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
    def __init__(self, client, market_data, config):
        self.client = client
        self.market_data = market_data
        self.config = config
        self.timeframes = [TIMEFRAME_MAP[tf] for tf in self.config['trading_settings']['timeframes']]
        
        self.trade_logger = TradeLogger(filename=self.config['misc']['trade_log_filename'])
        self.indicator_calc = IndicatorCalculator(self.config['indicator_parameters'])
        self.liquidity_detector = LiquidityDetector(self.config['liquidity_detector_settings'])
        self.order_manager = OrderManager(client, market_data, self.indicator_calc, self.trade_logger, self.liquidity_detector)
        self.divergence_detector = DivergenceDetector(order=5, k=2)        
        
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1]
        self.tf_lower = min(self.timeframes)
        
        self.max_positions = self.config['trading_settings']['max_total_positions']
        
    def analyze_market_structure(self, data, timeframe):
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
                        
    def analyze_timeframe_alignment(self, data_dict, indicators_dict):
        try:
            alignment = {}
            for tf in [self.tf_higher, self.tf_medium, self.tf_lower]:
                data = data_dict[tf]
                indicators = indicators_dict[tf]                
        
                if tf == self.tf_higher or tf == self.tf_medium:
                    structure_analysis = self.analyze_market_structure(data, tf)
                    if tf == self.tf_higher:
                        alignment[tf] = {'structure': structure_analysis}
                    else:
                        alignment[tf] = {
                            'structure': structure_analysis,
                            'trend': self.determine_overall_trend(data, indicators)
                        }
                elif tf == self.tf_lower:
                    alignment[tf] = {'momentum': self._analyze_momentum(indicators)}
            return alignment
        except Exception as e:
            logging.error(f"Error in timeframe alignment analysis: {str(e)}")
            return None
            
    def _analyze_momentum(self, indicators):
        try:
            rsi = indicators['rsi'].iloc[-1]
            rsi_trend = 'bullish' if rsi > 50 else 'bearish'
            adx = indicators['adx'].iloc[-1]
            trend_strength = 'strong' if adx > 25 else 'weak'
            return {'rsi_trend': rsi_trend, 'trend_strength': trend_strength}
        except Exception as e:
            logging.error(f"Error analyzing momentum: {str(e)}")
            return None
        
    def determine_overall_trend(self, data, indicators):        
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

            alignment = self.analyze_timeframe_alignment(data, indicators)
            if not alignment: return
    
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
                open_mt5_tickets = set()
            else:
                open_mt5_tickets = {pos.ticket for pos in positions}

            # Find trades that are in our log as 'open' but not in MT5 anymore
            for index, trade in open_logged_trades.iterrows():
                ticket_id = trade['ticket_id']
                if ticket_id not in open_mt5_tickets:
                    deals = mt5.history_deals_get(position=ticket_id)
                    if deals:
                        closing_deal = deals[-1]
                        close_price = closing_deal.price
                        close_time = pd.to_datetime(closing_deal.time, unit='s')
                        pnl = closing_deal.profit
                        
                        status = "closed_manual"
                        if abs(close_price - trade['take_profit']) < 0.0001:
                           status = "closed_tp"
                        elif abs(close_price - trade['stop_loss']) < 0.0001:
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
            
            key_levels = alignment[self.tf_higher]['structure']['key_levels']            
    
            is_valid, stop_loss, take_profit, stop_loss_points = self.order_manager.calculate_sl_tp(
                symbol=symbol,
                order_type=trade_setup['direction'],
                indicators=indicators,
                trade_setup=trade_setup,
                liquidity_levels=liquidity_levels,
                key_levels=key_levels,
                rr_ratio=1.5
            )
            
            if not is_valid:

                logging.info(f"[{symbol}] Trade setup invalidated by order parameter calculation.")
                return None

            lot_size = self.order_manager.calculate_lot_size(
                symbol=symbol,
                account_balance=account_info.balance,
                risk_percentage=1.0,
                stop_loss_points=stop_loss_points
            )
            
            if lot_size is None or lot_size <= 0:
                logging.error(f"[{symbol}] Calculated lot size is invalid: {lot_size}")
                return None

            logging.info(f"[{symbol}] Final Order Params: Lot={lot_size}, SL={stop_loss}, TP={take_profit}")
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
        
    def _evaluate_immediate_breakout_setup(self, symbol, data, indicators, alignment):
        try:
            med_tf_data = data[self.tf_medium]
            if len(med_tf_data) < 21: return None

            last_candle_time = med_tf_data.index[-1]
            hour = last_candle_time.hour
            is_london_open = (7 <= hour < 10)
            is_ny_open = (13 <= hour < 16)
            if not (is_london_open or is_ny_open):
                return None

            med_indicators = indicators[self.tf_medium]
            range_high, range_low = self._identify_consolidation_range(med_tf_data, last_candle_time, hours=5)
            if range_high is None or range_high == range_low:
                return None

            last_candle = med_tf_data.iloc[-1]
            last_close = last_candle['close']
            atr = med_indicators['atr'].iloc[-1]
            
            direction = None
            if last_close > range_high:
                direction = 'buy'
            elif last_close < range_low:
                direction = 'sell'
            else:
                return None

            score = 0
            factors = {}
            required_score = 4.0

            factors['strong_candle'] = self._is_strong_breakout_candle(direction, last_candle, atr, med_indicators)
            if factors['strong_candle']: score += 1.5

            avg_volume = med_tf_data['tick_volume'].rolling(window=20).mean().iloc[-2]
            factors['volume_confirmed'] = last_candle['tick_volume'] > (avg_volume * 1.75)
            if factors['volume_confirmed']: score += 1.5

            if direction == 'buy':
                factors['decisive_close'] = last_close > (range_high + atr * 0.2)
            else: # sell
                factors['decisive_close'] = last_close < (range_low - atr * 0.2)
            if factors['decisive_close']: score += 1.0

            if direction == 'buy':
                factors['rsi_momentum'] = med_indicators['rsi'].iloc[-1] > 55
            else: # sell
                factors['rsi_momentum'] = med_indicators['rsi'].iloc[-1] < 45
            if factors['rsi_momentum']: score += 1.0

            factors['adx_trending'] = med_indicators['adx'].iloc[-1] > 20
            if factors['adx_trending']: score += 0.5
            
            if score >= required_score:
                logging.info(f"[{symbol}] VALID {direction.upper()} BREAKOUT (Score: {score}/{required_score}). Factors: {factors}")
                return {'valid': True, 'symbol': symbol, 'direction': direction,
                        'setup_type': f'daytrade_breakout_{direction}_score_{score}',
                        'breakout_level': range_high if direction == 'buy' else range_low, 
                        'breakout_candle': last_candle.copy(),
                        'breakout_range_high': range_high,
                        'breakout_range_low': range_low
                    }

            return None
        except Exception as e:
            logging.error(f"Error evaluating immediate breakout for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return None
        
    def _is_strong_breakout_candle(self, direction: str, candle: pd.Series, atr: float, indicators: pd.DataFrame) -> bool:
        try:
            is_doji = indicators['cdl_doji'].iloc[-1] != 0
            is_spinning_top = indicators['cdl_spinning_top'].iloc[-1] != 0
            if is_doji or is_spinning_top:
                logging.info(f"[{self.market_data.symbol}] Breakout candle rejected: Is a Doji or Spinning Top.")
                return False

            if direction == 'buy':
                is_rejection = indicators['cdl_shooting_star'].iloc[-1] != 0
                if is_rejection:
                    logging.info(f"[{self.market_data.symbol}] Breakout candle rejected: Is a Shooting Star on a buy attempt.")
                    return False
            else: # sell
                is_rejection = indicators['cdl_hammer'].iloc[-1] != 0
                if is_rejection:
                    logging.info(f"[{self.market_data.symbol}] Breakout candle rejected: Is a Hammer on a sell attempt.")
                    return False

            is_marubozu = indicators['cdl_marubozu'].iloc[-1]
            is_engulfing = indicators['cdl_engulfing'].iloc[-1]
            
            is_pattern_bullish = (is_marubozu == 100) or (is_engulfing == 100)
            is_pattern_bearish = (is_marubozu == -100) or (is_engulfing == -100)

            body = abs(candle['close'] - candle['open'])
            is_size_significant = body > (atr * 0.7)

            if direction == 'buy':
                if is_pattern_bullish and is_size_significant:
                    return True
            elif direction == 'sell':
                if is_pattern_bearish and is_size_significant:
                    return True
            return False
            
        except Exception as e:
            logging.error(f"Error in _is_strong_breakout_candle: {e}")
        return False
        
    def _evaluate_retest_setup(self, symbol, data, indicators, alignment):
        try:
            med_tf_data = data[self.tf_medium]
            med_indicators = indicators[self.tf_medium]
            if len(med_tf_data) < 50:
                return None

            higher_trend = alignment.get(self.tf_higher, {}).get('structure', {}).get('trend')
            if higher_trend not in ['bullish', 'bearish']:
                return None

            end_of_range_search = med_tf_data.index[-15]
            range_high, range_low = self._identify_consolidation_range(
                med_tf_data.loc[:end_of_range_search], med_tf_data.index[-1], hours=12
            )
            if range_high is None or range_high == range_low:
                return None

            breakout_candle = None
            breakout_index = -1
            search_window = med_tf_data.tail(30)
            
            for i in range(len(search_window) - 5):
                candle = search_window.iloc[i]
                prev_candle = search_window.iloc[i-1] if i > 0 else None
                if prev_candle is None: continue
                    
                if (higher_trend == 'bullish' and candle['close'] > range_high and prev_candle['close'] <= range_high) or \
                (higher_trend == 'bearish' and candle['close'] < range_low and prev_candle['close'] >= range_low):
                    breakout_candle = candle
                    breakout_index = -30 + i
                    break

            if breakout_candle is None:
                return None

            avg_volume_before_breakout = med_tf_data['tick_volume'].iloc[:breakout_index].tail(20).mean()
            breakout_atr = med_indicators['atr'].iloc[breakout_index]
            breakout_body_size = abs(breakout_candle['close'] - breakout_candle['open'])
            is_strong_volume = breakout_candle['tick_volume'] > avg_volume_before_breakout * 1.3
            is_impulsive_candle = breakout_body_size > breakout_atr * 0.7
            if not (is_strong_volume and is_impulsive_candle):
                return None

            pullback_candles = med_tf_data.iloc[breakout_index + 1 : -1]

            if len(pullback_candles) < 2:
                return None

            for candle in pullback_candles:
                if abs(candle['close'] - candle['open']) > breakout_atr:
                    return None
                
            rejection_candle = med_tf_data.iloc[-1]
            rejection_indicators = med_indicators.iloc[-1]
            direction = None
            retest_level = None
            is_confirmed_rejection = False
            confirmation_signals = []

            if higher_trend == 'bullish':
                retest_level = range_high
                if not (rejection_candle['low'] <= retest_level and rejection_candle['close'] > retest_level):
                    return None
                
                if rejection_indicators['cdl_engulfing'] == 100:
                    confirmation_signals.append('bullish_engulfing')
                    
                rejection_body = abs(rejection_candle['close'] - rejection_candle['open'])
                rejection_lower_wick = rejection_candle['open'] - rejection_candle['low'] if rejection_candle['close'] > rejection_candle['open'] else rejection_candle['close'] - rejection_candle['low']
                if rejection_lower_wick > rejection_body * 1.5 and rejection_body > 0.00001:
                    confirmation_signals.append('pin_bar')
                    
                if rejection_indicators['rsi'] < 40:
                    confirmation_signals.append('rsi_oversold_bounce')

                if len(confirmation_signals) >= 1:
                    is_confirmed_rejection = True
                    direction = 'buy'

            elif higher_trend == 'bearish':
                retest_level = range_low
                if not (rejection_candle['high'] >= retest_level and rejection_candle['close'] < retest_level):
                    return None

                if rejection_indicators['cdl_engulfing'] == -100:
                    confirmation_signals.append('bearish_engulfing')
                    
                rejection_body = abs(rejection_candle['close'] - rejection_candle['open'])
                rejection_upper_wick = rejection_candle['high'] - rejection_candle['open'] if rejection_candle['close'] < rejection_candle['open'] else rejection_candle['high'] - rejection_candle['close']
                if rejection_upper_wick > rejection_body * 1.5 and rejection_body > 0.00001:
                    confirmation_signals.append('pin_bar')
                    
                if rejection_indicators['rsi'] > 60:
                    confirmation_signals.append('rsi_overbought_rejection')

                if len(confirmation_signals) >= 1:
                    is_confirmed_rejection = True
                    direction = 'sell'

            if is_confirmed_rejection:
                logging.info(f"[{symbol}] VALID {direction.upper()} RETEST. Level: {retest_level:.5f}, Signals: {confirmation_signals}")
                return {
                    'valid': True, 'symbol': symbol, 'direction': direction,
                    'setup_type': f'retest_{direction}_{len(confirmation_signals)}_signals',
                    'retest_level': retest_level,
                    'retest_candle': rejection_candle.copy(),
                    'confirmation_signals': confirmation_signals
                }

            return None
            
        except Exception as e:
            logging.error(f"Error evaluating retest setup for {symbol}: {str(e)}\n{traceback.format_exc()}")
            return None
        
    def _identify_consolidation_range(self, data: pd.DataFrame, end_time: pd.Timestamp, hours=6) -> tuple:
        """
        Identifies the high and low of a recent consolidation period
        """
        try:
            start_time = end_time - pd.Timedelta(hours=hours)
            consolidation_data = data.loc[start_time:end_time].iloc[:-1] 

            if len(consolidation_data) < (hours / 2):
                return None, None

            range_high = consolidation_data['high'].max()
            range_low = consolidation_data['low'].min()
            
            atr = self.indicator_calc.calculate_indicators(data)['atr'].iloc[-1]
            range_size = range_high - range_low
            
            if range_size > (atr * 6.0):
                return None, None
                
            if range_size < atr:
                return None, None

            return range_high, range_low
        except Exception as e:
            logging.error(f"Error identifying consolidation range: {e}")
            return None, None

    def _evaluate_trade_setup(self, symbol, data, indicators, alignment):
        try:
            retest_setup = self._evaluate_retest_setup(symbol, data, indicators, alignment)
            if retest_setup and retest_setup.get('valid'):
                return retest_setup

            immediate_breakout_setup = self._evaluate_immediate_breakout_setup(symbol, data, indicators, alignment)
            if immediate_breakout_setup and immediate_breakout_setup.get('valid'):
                return immediate_breakout_setup

            market_regime = self._get_market_regime(symbol, indicators)
            
            if market_regime == 'trending':
                return self._evaluate_trending_setup(symbol, data, indicators, alignment)
            elif market_regime == 'ranging':
                return self._evaluate_ranging_setup(symbol, data, indicators, alignment)
            
            return None
        except Exception as e:
            logging.error(f"Error in master trade evaluation for {symbol}: {str(e)}\n{traceback.format_exc()}")
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
                'higher_tf_trend': higher_trend, 'setup_type': f'quality_trend_pullback_{confluence_note}',
                'swing_points': alignment[self.tf_medium]['structure']['swing_points'],
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

            ranging_setup_base = {
                'valid': True, 'symbol': symbol, 'higher_tf_trend': 'ranging', 'setup_type': 'ranging',
                'swing_points': alignment[self.tf_medium]['structure']['swing_points']
            }

            if abs(current_price - resistance_levels[-1]) < price_range * 0.1:
                if self._validate_sell(symbol, indicators[self.tf_lower]):
                    ranging_setup_base['direction'] = 'sell'
                    return ranging_setup_base
            elif abs(current_price - support_levels[0]) < price_range * 0.1:
                if self._validate_buy(symbol, indicators[self.tf_lower]):
                    ranging_setup_base['direction'] = 'buy'
                    return ranging_setup_base
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
        try:
            positions = mt5.positions_get()
            
            if positions is None:
                return True, "No positions found"
                
            total_positions = len(positions)
            
            if total_positions >= self.max_positions:
                return False, f"Max overall positions limit reached ({self.max_positions})"
                
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            symbol_position_count = len(symbol_positions)
            
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
    def __init__(self, client, market_data, indicator_calc, trade_logger, liquidity_detector):
        self.client = client
        self.market_data = market_data
        self.indicator_calc = indicator_calc
        self.trade_logger = trade_logger
        self.liquidity_detector = liquidity_detector

    def _normalize_price(self, symbol: str, price: float) -> float:
        symbol_info = mt5.symbol_info(symbol)
        return round(price, symbol_info.digits) if symbol_info else price

    def _get_lot_size_for_risk(self, symbol: str, account_balance: float, risk_percentage: float, stop_loss_points: float) -> float:
        try:
            risk_amount = account_balance * (risk_percentage / 100)
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info: return None
            value_per_point = symbol_info.trade_tick_value / symbol_info.trade_tick_size * symbol_info.point
            if value_per_point <= 0: return None
            loss_for_one_lot = stop_loss_points * value_per_point
            if loss_for_one_lot <= 0: return None
            lot_size = risk_amount / loss_for_one_lot
            lot_step = symbol_info.volume_step
            lot_size = math.floor(lot_size / lot_step) * lot_step
            return max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
        except Exception as e:
            logging.error(f"[{symbol}] CRITICAL ERROR in _get_lot_size_for_risk: {e}")
            return None

    def _find_next_structural_level(self, initial_level: float, direction: str, key_levels: list) -> float:
        """
        Finds the next key support/resistance level beyond the initial one to hide the SL behind.
        """
        if direction == 'buy':
            potential_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'support' and lvl['price'] < initial_level]
            if potential_levels:
                return max(potential_levels)
        else:
            potential_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'resistance' and lvl['price'] > initial_level]
            if potential_levels:
                return min(potential_levels)
        
        return initial_level

    def calculate_sl_tp(self, symbol: str, order_type: str, indicators: dict, 
                    trade_setup: dict, liquidity_levels: dict, key_levels: list, 
                    rr_ratio=1.5, atr_tp_multiplier=2.5) -> tuple:
        try:
            symbol_info = mt5.symbol_info(symbol)
            symbol_tick = mt5.symbol_info_tick(symbol)
            if not symbol_info or not symbol_tick: 
                return False, None, None, None

            point = symbol_info.point
            stops_level = symbol_info.trade_stops_level
            current_price = symbol_tick.ask if order_type == 'buy' else symbol_tick.bid
            atr_value = indicators[self.market_data.timeframes[1]]['atr'].iloc[-1]
            
            setup_type = trade_setup.get('setup_type', '')
            initial_sl_point = None
            
            if 'ranging' in setup_type:
                support_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'support']
                resistance_levels = [lvl['price'] for lvl in key_levels if lvl['type'] == 'resistance']
                if not support_levels or not resistance_levels: return False, None, None, None
                
                range_high = max(resistance_levels)
                range_low = min(support_levels)

                if order_type == 'buy':
                    initial_sl_point = range_low
                    take_profit = self._normalize_price(symbol, range_high)
                else: # sell
                    initial_sl_point = range_high
                    take_profit = self._normalize_price(symbol, range_low)
            
            else:
                if 'breakout' in setup_type:
                    initial_sl_point = trade_setup.get('breakout_range_low') if order_type == 'buy' else trade_setup.get('breakout_range_high')
                elif 'retest' in setup_type:
                    retest_candle = trade_setup.get('retest_candle')
                    if retest_candle is None: return False, None, None, None
                    initial_sl_point = retest_candle['low'] if order_type == 'buy' else retest_candle['high']
                
                if initial_sl_point is None:
                    logging.info(f"[{symbol}] Ignoring non-specialized setup type '{setup_type}' for SL/TP.")
                    return False, None, None, None
                
                atr_tp = current_price + (atr_value * atr_tp_multiplier) if order_type == 'buy' else current_price - (atr_value * atr_tp_multiplier)

                liquidity_tp = None
                target_liquidity = self.liquidity_detector.get_target_for_bias(
                    bias=('bullish' if order_type == 'buy' else 'bearish'),
                    liquidity_levels=liquidity_levels, entry_price=current_price, min_rr=rr_ratio, sl_price=None # SL not needed yet
                )
                if target_liquidity:
                    liquidity_tp = target_liquidity['level']

                final_tp = None
                if order_type == 'buy':
                    final_tp = min(atr_tp, liquidity_tp) if liquidity_tp else atr_tp
                else: # sell
                    final_tp = max(atr_tp, liquidity_tp) if liquidity_tp else atr_tp
                
                take_profit = self._normalize_price(symbol, final_tp)

            final_sl_level = self._find_next_structural_level(initial_sl_point, order_type, key_levels)
            sl_buffer = self._normalize_price(symbol, atr_value * 0.2)
            ideal_sl = final_sl_level - sl_buffer if order_type == 'buy' else final_sl_level + sl_buffer

            if (order_type == 'buy' and ideal_sl >= current_price) or \
               (order_type == 'sell' and ideal_sl <= current_price):
                logging.warning(f"[{symbol}] Trade invalidated: Ideal SL ({ideal_sl}) is on the wrong side of current price ({current_price}).")
                return False, None, None, None
            
            stop_loss = self._normalize_price(symbol, ideal_sl)
            risk_distance = abs(current_price - stop_loss)
            reward_distance = abs(current_price - take_profit)

            if risk_distance == 0 or reward_distance / risk_distance < 1.0:
                logging.warning(f"[{symbol}] Trade invalidated: Final R:R is less than 1:1 (Risk: {risk_distance:.5f}, Reward: {reward_distance:.5f}).")
                return False, None, None, None

            stop_distance_points = risk_distance / point
            if stop_distance_points < stops_level:
                logging.warning(f"[{symbol}] Trade invalidated: Stop distance ({stop_distance_points:.1f}) is less than minimum stops level ({stops_level}).")
                return False, None, None, None

            logging.info(f"[{symbol}] Validated Order Params for {setup_type}: SL={stop_loss}, TP={take_profit}")
            return True, stop_loss, take_profit, stop_distance_points

        except Exception as e:
            logging.error(f"[{symbol}] CRITICAL ERROR in calculate_sl_tp: {e}\n{traceback.format_exc()}")
            return False, None, None, None
        
    def calculate_lot_size(self, symbol, account_balance, risk_percentage, stop_loss_points):
        if stop_loss_points is None or stop_loss_points <= 0:
            logging.error(f"[{symbol}] Invalid stop loss points ({stop_loss_points}) for lot size calculation.")
            return None
        return self._get_lot_size_for_risk(symbol, account_balance, risk_percentage, stop_loss_points)
        
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
    trade_managers = {symbol: TradeManager(client, market_data_dict[symbol], CONFIG) for symbol in symbols}

    while True:
        start_time = time.time()
        tasks = []
        for symbol in symbols:
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