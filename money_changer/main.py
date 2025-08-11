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
from news_analyzer import NewsAnalyzer

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PERIOD = CONFIG['trading_settings']['max_period']
CHECK_INTERVAL = CONFIG['trading_settings']['main_loop_interval_seconds']

TIMEFRAME_MAP = {
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1
}

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
    def __init__(self, timeframe, config):
        self.timeframe = timeframe
        self.lookback_period = config['lookback_period'] 
        
        self.prominence_factor = config['prominence_factor']

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

    def analyze_price_action(self, data, swing_points, trend_lookback, consolidation_threshold):
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
            if range_percentage <= consolidation_threshold:
                price_action_info['consolidation'] = True

        return price_action_info

class TradeManager:
    def __init__(self, client, market_data, config):
        self.client = client
        self.market_data = market_data
        self.config = config
        self.timeframes = [TIMEFRAME_MAP[tf] for tf in self.config['trading_settings']['timeframes']]
        
        self.trade_logger = TradeLogger(filename=self.config['misc']['trade_log_filename'])
        self.indicator_calc = IndicatorCalculator(self.config['indicator_parameters'])
        self.liquidity_detector = LiquidityDetector(self.config['liquidity_detector_settings'])
        self.news_analyzer = NewsAnalyzer(csv_filepath=self.config['misc']['news_calendar_filename'], config=self.config)
        self.order_manager = OrderManager(client, market_data, self.indicator_calc, self.trade_logger, self.liquidity_detector, self.config)
        
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1]
        self.tf_lower = min(self.timeframes)
        
        self.max_positions = self.config['trading_settings']['max_total_positions']
        
    def analyze_market_structure(self, data, timeframe):
        swing_detector = SwingPointDetector(timeframe=timeframe, config=self.config['swing_point_detector_settings'])
        try:
            swing_points = swing_detector.find_swing_points(data)
            if not swing_points:
                return None
                
            price_action = swing_detector.analyze_price_action(data, swing_points, self.config['price_action_settings']['trend_lookback'], self.config['price_action_settings']['consolidation_threshold'])
            
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
            trend_strength = 'strong' if adx > 20 else 'weak' 
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
        
    def _is_breakout_risk_high(self, symbol: str, data: dict, indicators: dict, direction: str) -> bool:
        """
        Analyzes market conditions to determine if there is a high risk of a breakout,
        which would invalidate a ranging trade.
        Returns True if risk is high, False otherwise.
        """
        try:
            h1_indicators = indicators[self.tf_medium] # H1 data
            m15_data = data[self.tf_lower] # M15 data

            if h1_indicators['bb_upper'].iloc[-1] < h1_indicators['kc_upper'].iloc[-1] and \
               h1_indicators['bb_lower'].iloc[-1] > h1_indicators['kc_lower'].iloc[-1]:
                logging.warning(f"[{symbol}] Ranging trade vetoed: High-risk volatility squeeze detected on H1 chart.")
                return True

            volume_period = self.config['breakout_risk_filter_settings']['volume_period']
            volume_factor = self.config['breakout_risk_filter_settings']['volume_increase_factor']
            
            avg_volume = m15_data['tick_volume'].rolling(window=volume_period).mean().iloc[-2]
            last_volume = m15_data['tick_volume'].iloc[-1]

            if last_volume > (avg_volume * volume_factor):
                logging.warning(f"[{symbol}] Ranging trade vetoed: Significant volume spike ({last_volume} vs avg {avg_volume:.0f}) detected on M15.")
                return True

            last_3_candles = data[self.tf_medium].iloc[-3:]
            tight_range_threshold = h1_indicators['atr'].iloc[-1] * 0.5
            
            if (last_3_candles['high'].max() - last_3_candles['low'].min()) < tight_range_threshold:
                logging.warning(f"[{symbol}] Ranging trade vetoed: Price is consolidating too tightly at the level (H1).")
                return True

            return False

        except Exception as e:
            logging.error(f"Error in breakout risk filter for {symbol}: {e}")
            return True

    async def check_for_signals(self, symbol):
        try:
            if self.news_analyzer.check_for_high_impact_news(symbol):
                logging.info(f"[{symbol}] Halting new trade checks due to nearby high-impact news.")
                return 
        
            data, indicators = self._prepare_data(symbol)
            if not data or not indicators: return

            alignment = self.analyze_timeframe_alignment(data, indicators)
            if not alignment: return
    
            liquidity_levels = self.liquidity_detector.get_liquidity_levels(data[self.tf_medium], {}, daily_df=data[self.tf_higher])

            trade_setup = self._evaluate_ranging_setup(symbol, data, indicators, alignment)
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
                logging.info(f"[{symbol}] Successfully opened {trade_setup['direction']} trade {trade_result.order}.")

        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}\n{traceback.format_exc()}")
    
    async def manage_open_positions(self, symbol: str):
        """
        Manages all open positions for a symbol. It checks for trades that have been closed 
        (e.g., by SL/TP) and also proactively closes profitable trades before high-impact news.
        """
        try:
            open_logged_trades = self.trade_logger.get_open_trades(symbol)
            if not open_logged_trades.empty:
                positions = mt5.positions_get(symbol=symbol)
                open_mt5_tickets = {pos.ticket for pos in positions} if positions else set()

                for _, trade in open_logged_trades.iterrows():
                    ticket_id = int(trade['ticket_id'])
                    if ticket_id not in open_mt5_tickets:
                        deals = mt5.history_deals_get(position=ticket_id)
                        if deals:
                            closing_deal = deals[-1]
                            close_price = closing_deal.price
                            close_time = pd.to_datetime(closing_deal.time, unit='s')
                            pnl = closing_deal.profit
                            
                            status = "closed_manual"
                            if abs(close_price - trade['take_profit']) < 1e-5:
                               status = "closed_tp"
                            elif abs(close_price - trade['stop_loss']) < 1e-5:
                               status = "closed_sl"

                            self.trade_logger.log_close_trade(ticket_id, close_price, close_time, pnl, status)
                            logging.info(f"[{symbol}] Logged closure for trade {ticket_id}. PnL: {pnl:.2f}")

        except Exception as e:
            logging.error(f"Error checking for closed trades for {symbol}: {e}\n{traceback.format_exc()}")

        try:
            if not self.news_analyzer.check_for_high_impact_news(symbol):
                return

            open_positions = mt5.positions_get(symbol=symbol)
            if not open_positions:
                return

            for position in open_positions:
                if position.profit > 0:
                    logging.warning(f"Attempting to close profitable trade {position.ticket} for {symbol} due to upcoming high-impact news.")
                    self.order_manager.close_position(position, "Closed before news")
        
        except Exception as e:
            logging.error(f"Error managing trades before news for {symbol}: {e}\n{traceback.format_exc()}")

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
                key_levels=key_levels
            )
            
            if not is_valid:
                logging.info(f"[{symbol}] Trade setup invalidated by order parameter calculation.")
                return None

            lot_size = self.order_manager.calculate_lot_size(
                symbol=symbol,
                account_balance=account_info.balance,
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
            
            # Check for a potential sell signal
            if abs(current_price - resistance_levels[-1]) < price_range * self.config['ranging_strategy_settings']['level_proximity_factor']:
                if self._validate_sell(symbol, indicators[self.tf_lower]):                   
                    ranging_setup_base['direction'] = 'sell'
                    ranging_setup_base['key_level_price'] = resistance_levels[-1]
                    return ranging_setup_base
            
            elif abs(current_price - support_levels[0]) < price_range * self.config['ranging_strategy_settings']['level_proximity_factor']:
                if self._validate_buy(symbol, indicators[self.tf_lower]):
                    ranging_setup_base['direction'] = 'buy'
                    ranging_setup_base['key_level_price'] = support_levels[0]
                    return ranging_setup_base
                    
            return None
        except Exception as e:
            logging.error(f"Error evaluating ranging setup for {symbol}: {str(e)}")
            return None
    
    def _has_reversal_pattern(self, direction: str, data: pd.DataFrame, symbol: str) -> bool:
        """
        Checks the last candle for a suite of high-probability REVERSAL signals
        using precise directional checks. Returns True only if volume is also above average.
        """
        try:
            if len(data) < 2:
                return False
            
            last = data.iloc[-2]
            avg_vol = data['tick_volume'].rolling(window=20).mean().iloc[-2]
            if last['tick_volume'] < avg_vol:
                return False 

            if direction == 'buy':
                patterns = [
                    last['cdl_engulfing'] == 100,
                    last['cdl_hammer'] == 100,
                    last['cdl_piercing'] == 100,
                    last['cdl_morningstar'] == 100,
                    last['cdl_morningdojistar'] == 100,
                    last['cdl_invertedhammer'] == 100,
                ]
            else:  # sell
                patterns = [
                    last['cdl_engulfing'] == -100,
                    last['cdl_hangingman'] == -100,
                    last['cdl_darkcloudcover'] == -100,
                    last['cdl_shootingstar'] == -100,
                    last['cdl_eveningstar'] == -100,
                    last['cdl_eveningdojistar'] == -100,
                ]
                

            return any(patterns)
        except Exception as e:
            logging.error(f"Error checking for reversal pattern: {e}")
            return False
        
    def _validate_buy(self, symbol: str, data: pd.DataFrame) -> bool:
        try:
            has_pattern = self._has_reversal_pattern('buy', data, symbol)
            if not has_pattern:
                return False 
            
            rsi = data['rsi'].iloc[-2]
            is_good_rsi = rsi < (100 - self.config['ranging_strategy_settings']['rsi_threshold'])
        
            if has_pattern and is_good_rsi:
                logging.info(f"Buy entry validated for {symbol} with RSI: {rsi}")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Error validating buy entry for {symbol}: {str(e)}")
            return False

    def _validate_sell(self, symbol: str, data: pd.DataFrame) -> bool:
        try:
            has_pattern = self._has_reversal_pattern('sell', data, symbol)
            if not has_pattern:
                return False
            
            rsi = data['rsi'].iloc[-2]
            is_good_rsi = rsi > self.config['ranging_strategy_settings']['rsi_threshold']
            
            if has_pattern and is_good_rsi:
                logging.info(f"Sell entry validated for {symbol} with RSI: {rsi}")
                return True
            
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
            
            max_per_symbol = self.config['trading_settings']['max_positions_per_symbol']
            if symbol_position_count >= max_per_symbol:
                return False, f"Max positions limit reached for {symbol} ({symbol_position_count}/{max_per_symbol})"
                
            return True, f"Position limit ok (Total: {total_positions}/{self.max_positions}, {symbol}: {symbol_position_count}/{max_per_symbol})"
                
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
        try:
            if not os.path.exists(self.filename):
                df = pd.DataFrame(columns=self.columns)
                df.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error initializing trade log: {str(e)}")

    def _load_log(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=self.columns)
        except Exception as e:
            logging.error(f"Error loading trade log: {e}")
            return pd.DataFrame(columns=self.columns)

    def _save_log(self, df: pd.DataFrame):
        try:
            df.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error saving trade log: {e}")
            
    def get_open_trades(self, symbol: str) -> pd.DataFrame:
        log_df = self._load_log()
        if log_df.empty:
            return pd.DataFrame()
        return log_df[(log_df['symbol'] == symbol) & (log_df['status'] == 'open')]

    def log_open_trade(self, ticket_id: int, symbol: str, direction: str, open_price: float, 
                       stop_loss: float, take_profit: float, lot_size: float, reason: str, market_context: str):
        try:
            log_df = self._load_log()
            
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
        except Exception as e:
            logging.error(f"Error logging open trade: {str(e)}")

    def log_close_trade(self, ticket_id: int, close_price: float, close_time: datetime, pnl: float, status: str):
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
    def __init__(self, client, market_data, indicator_calc, trade_logger, liquidity_detector, config):
        self.client = client
        self.market_data = market_data
        self.indicator_calc = indicator_calc
        self.trade_logger = trade_logger
        self.liquidity_detector = liquidity_detector
        self.config = config

    def _normalize_price(self, symbol: str, price: float) -> float:
        symbol_info = mt5.symbol_info(symbol)
        return round(price, symbol_info.digits) if symbol_info else price

    def _get_lot_size_for_risk(self, symbol: str, account_balance: float, risk_percentage: float, stop_loss_points: float) -> float:
        try:
            risk_amount = account_balance * (risk_percentage / 100)
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info: return None
            tick_value = mt5.symbol_info(symbol).trade_tick_value
            tick_size = mt5.symbol_info(symbol).trade_tick_size
            point = mt5.symbol_info(symbol).point
            
            value_per_point = tick_value / tick_size * point
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
                    trade_setup: dict, liquidity_levels: dict, key_levels: list) -> tuple:
        try:
            rr_ratio = self.config['risk_management']['min_risk_reward_ratio']
            sl_buffer_atr_factor = self.config['risk_management']['sl_buffer_atr_factor']

            symbol_info = mt5.symbol_info(symbol)
            symbol_tick = mt5.symbol_info_tick(symbol)
            if not symbol_info or not symbol_tick:
                logging.error(f"[{symbol}] Could not get symbol info or tick.")
                return False, None, None, None

            point = symbol_info.point
            stops_level = symbol_info.trade_stops_level
            current_price = symbol_tick.ask if order_type == 'buy' else symbol_tick.bid

            # --- Get Data from the Correct Timeframes ---
            h1_indicators = indicators[self.market_data.timeframes[1]]
            h1_swing_points = trade_setup['swing_points']
            h1_atr = h1_indicators['atr'].iloc[-1]

            # D1 for the strategic reason for the trade
            signal_level = trade_setup.get('key_level_price')
            if not signal_level:
                logging.error(f"[{symbol}] Trade setup is missing the strategic 'key_level_price'.")
                return False, None, None, None

            ideal_sl = 0.0
            ideal_tp = 0.0

            if order_type == 'buy':
                # REASON: Buying at a D1 Support Level ('signal_level')
                # SL: Place it just below that D1 support, using H1 ATR for the buffer.
                ideal_sl = signal_level - (h1_atr * sl_buffer_atr_factor)

                # TP: Target the next logical obstacle, which is the last H1 swing high.
                last_swing_high = h1_swing_points['highs']['high'].iloc[-1]
                ideal_tp = last_swing_high

            else: # sell
                # REASON: Selling at a D1 Resistance Level ('signal_level')
                # SL: Place it just above that D1 resistance, using H1 ATR for the buffer.
                ideal_sl = signal_level + (h1_atr * sl_buffer_atr_factor)

                # TP: Target the next logical obstacle, which is the last H1 swing low.
                last_swing_low = h1_swing_points['lows']['low'].iloc[-1]
                ideal_tp = last_swing_low

            stop_loss = self._normalize_price(symbol, ideal_sl)
            take_profit = self._normalize_price(symbol, ideal_tp)

            # --- Final Validation Logic (This part remains the same) ---
            if (order_type == 'buy' and stop_loss >= current_price) or \
            (order_type == 'sell' and stop_loss <= current_price):
                logging.warning(f"[{symbol}] Trade invalidated: Calculated SL ({stop_loss}) is on the wrong side of current price ({current_price}). This can happen in fast markets.")
                return False, None, None, None

            risk_distance = abs(current_price - stop_loss)
            reward_distance = abs(take_profit - current_price)

            if risk_distance == 0 or (reward_distance / risk_distance) < rr_ratio:
                logging.warning(f"[{symbol}] Trade invalidated: Final R:R ({(reward_distance / risk_distance):.2f}) is less than required {rr_ratio}:1.")
                return False, None, None, None

            stop_distance_points = risk_distance / point
            if stop_distance_points < stops_level:
                logging.warning(f"[{symbol}] Trade invalidated: Stop distance ({stop_distance_points:.1f} pts) is less than minimum stops level ({stops_level} pts).")
                return False, None, None, None

            logging.info(f"[{symbol}] Validated Order Params for {trade_setup.get('setup_type', '')}: SL={stop_loss}, TP={take_profit}")
            return True, stop_loss, take_profit, stop_distance_points

        except Exception as e:
            logging.error(f"[{symbol}] CRITICAL ERROR in calculate_sl_tp: {e}\n{traceback.format_exc()}")
            return False, None, None, None
        
    def calculate_lot_size(self, symbol, account_balance, stop_loss_points):
        if stop_loss_points is None or stop_loss_points <= 0:
            logging.error(f"[{symbol}] Invalid stop loss points ({stop_loss_points}) for lot size calculation.")
            return None
        risk_percentage = self.config['risk_management']['risk_per_trade_percent']
        return self._get_lot_size_for_risk(symbol, account_balance, risk_percentage, stop_loss_points)
        
    def place_order(self, symbol, lots_size, order_type, stop_loss, take_profit):        
        symbol_tick = mt5.symbol_info_tick(symbol)
        if not symbol_tick: return None
                
        magic = self.config['misc']['mt5_magic_number']
        comment = self.config['misc']['trade_comment']
        
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
            logging.error(f"Failed to place {order_type} order for {symbol}: {result.comment} (Code: {result.retcode})")
        return result
    
    def close_position(self, position, comment: str):
        """Creates and sends a request to close a specific position."""
        symbol = position.symbol
        
        # Determine the correct order type for closing
        close_type = mt5.ORDER_TYPE_BUY if position.type == mt5.POSITION_TYPE_SELL else mt5.ORDER_TYPE_SELL
        
        # Determine the correct closing price
        price = mt5.symbol_info_tick(symbol).ask if position.type == mt5.POSITION_TYPE_SELL else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": self.config['misc']['mt5_magic_number'],
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Successfully sent close order for position {position.ticket}. Comment: {comment}")
            return True
        else:
            logging.error(f"Failed to close position {position.ticket}: {result.comment}")
            return False
        
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
        logging.info(f"Main loop cycle complete. Duration: {cycle_duration:.2f} seconds.")
        
        sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
        logging.info(f"Sleeping for {sleep_time:.2f} seconds.")
        await asyncio.sleep(sleep_time)

async def main():
    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("Failed to connect to MetaTrader 5")
        return

    symbols = CONFIG['trading_settings']['symbols']
    timeframe_strings = CONFIG['trading_settings']['timeframes']
    timeframes = tuple(TIMEFRAME_MAP[tf] for tf in timeframe_strings)

    try:
        await run_main_loop(client, symbols, timeframes)
    except asyncio.CancelledError:
        logging.info("Main task was cancelled. Shutting down...")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}\n{traceback.format_exc()}")
    finally:
        if client.is_initialized(): mt5.shutdown()
        logging.info("Trading script terminated.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Shutting down...")
    except RuntimeError as e:
        logging.error(f"RuntimeError: {e}")