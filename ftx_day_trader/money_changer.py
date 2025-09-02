import time
import logging
import math
import MetaTrader5 as mt5
import pandas as pd
import traceback
import asyncio
import numpy as np
from scipy.signal import argrelextrema
from typing import Tuple
import json
import talib

from indicators import IndicatorCalculator

# TradeLogger stub remains the same
class TradeLogger:
    def __init__(self, filename):
        self.filename = filename

    def log_trade(self, *args, **kwargs):
        logging.info(f"Trade logged: {args} {kwargs}")

# Load CONFIG
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PERIOD = CONFIG['trading_settings']['max_period']
CHECK_INTERVAL = CONFIG['trading_settings']['main_loop_interval_seconds']

TIMEFRAME_MAP = {
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
}

TIMEFRAME_MINUTES = {
    mt5.TIMEFRAME_M1: 1,
    mt5.TIMEFRAME_M5: 5,
    mt5.TIMEFRAME_M15: 15,
    mt5.TIMEFRAME_M30: 30,
    mt5.TIMEFRAME_H1: 60,
    mt5.TIMEFRAME_H4: 240,
    mt5.TIMEFRAME_D1: 1440
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
        logging.info("Getting account info.")
        return mt5.account_info()

class MarketData:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes
        self.num_candles = {tf: None for tf in timeframes}
        
    def calculate_num_candles(self, timeframe):
        minutes = TIMEFRAME_MINUTES.get(timeframe)
        if minutes is None:
            minutes = 60
        num_candles = int((MAX_PERIOD * 60) / minutes * 2)
        self.num_candles[timeframe] = max(200, num_candles)      

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
        self.order_manager = OrderManager(client, market_data, self.indicator_calc, self.trade_logger, self.config)
        
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
            
            # Pass pre-computed swing points to avoid recalculation
            price_levels = self.indicator_calc.identify_support_resistance_levels(swing_points)
            
            key_levels = []
            if price_levels.size > 0:
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
        
    async def check_for_signals(self, symbol):
        try:
            # Fetch data
            higher_tf_data = self.market_data.fetch_data(self.tf_higher)
            medium_tf_data = self.market_data.fetch_data(self.tf_medium)
            lower_tf_data = self.market_data.fetch_data(self.tf_lower)
            
            if higher_tf_data is None or medium_tf_data is None or lower_tf_data is None:
                logging.warning(f"[{symbol}] Insufficient data for analysis.")
                return
            
            # Analyze ATR
            atr = talib.ATR(medium_tf_data['high'], medium_tf_data['low'], medium_tf_data['close'], timeperiod=self.config['risk_management']['atr_period']).iloc[-2]

            # Analyze structures
            higher_tf_structure = self.analyze_market_structure(higher_tf_data, self.tf_higher)
            medium_tf_structure = self.analyze_market_structure(medium_tf_data, self.tf_medium)

            if not higher_tf_structure or not medium_tf_structure:
                return

            # 1. Higher Timeframe: Determines Trend Direction
            trend = higher_tf_structure['trend']
            if trend not in ['bullish', 'bearish']:
                return

            # 2. Medium Timeframe: Identifies the Pullback/Setup
            if trend == 'bullish':
                medium_tf_swing_lows = medium_tf_structure['swing_points']['lows']
                if medium_tf_swing_lows.empty:
                    return
                pullback_level = medium_tf_swing_lows.iloc[-1]['low']
                direction = 'buy'
            else: # bearish
                medium_tf_swing_highs = medium_tf_structure['swing_points']['highs']
                if medium_tf_swing_highs.empty:
                    return
                pullback_level = medium_tf_swing_highs.iloc[-1]['high']
                direction = 'sell'
            
            lower_tf_with_indicators = self.indicator_calc.calculate_indicators(lower_tf_data)
            is_entry_signal = self.indicator_calc.check_entry_patterns(
                lower_tf_with_indicators, 
                direction, 
                pullback_level=pullback_level, 
                atr=atr
            )

            if is_entry_signal:
                logging.info(f"[{symbol}] Valid entry signal found for {direction}!")
                entry_price = lower_tf_data.iloc[-2]['close']
                
                # Use the Medium Timeframe (the setup timeframe) for all Stop Loss calculations
                sl, tp = self.order_manager.calculate_sl_tp(
                    symbol, 
                    direction, 
                    entry_price, 
                    medium_tf_data,
                    medium_tf_structure['swing_points']
                )

                if sl is None or tp is None:
                    return
                
                trade_setup = {'valid': True, 'direction': direction, 'entry_price': entry_price, 'sl': sl, 'tp': tp}
                await self._execute_trade(symbol, trade_setup)

        except Exception as e:
            logging.error(f"Error in check_for_signals for {symbol}: {str(e)}\n{traceback.format_exc()}")

    async def _execute_trade(self, symbol, trade_setup):
        try:
            can_trade, reason = self.check_position_limit(symbol)
            if not can_trade:
                logging.debug(f"[{symbol}] Trade execution halted: {reason}")
                return
            if self._has_opposing_position(symbol, trade_setup['direction']):
                logging.debug(f"[{symbol}] Trade execution halted: Opposing position exists.")
                return

            trade_result = self.order_manager.place_order(
                symbol, trade_setup['direction'], trade_setup['entry_price'], trade_setup['sl'], trade_setup['tp']
            )
            if trade_result:
                self.trade_logger.log_trade(symbol, trade_setup['direction'], trade_result)
            
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}\n{traceback.format_exc()}")
            
    def _has_opposing_position(self, symbol, direction):
        symbol_positions = mt5.positions_get(symbol=symbol)
        if not symbol_positions: return False
        for position in symbol_positions:
            existing_type = 'buy' if position.type == mt5.POSITION_TYPE_BUY else 'sell'
            if existing_type != direction: return True
        return False

    def check_position_limit(self, symbol: str) -> Tuple[bool, str]:
        try:
            positions = mt5.positions_get()
            if positions is None: return True, "No positions found"
            total_positions = len(positions)
            if total_positions >= self.max_positions:
                return False, f"Max overall positions limit reached ({self.max_positions})"
            symbol_positions = [pos for pos in positions if pos.symbol == symbol]
            symbol_position_count = len(symbol_positions)
            max_per_symbol = self.config['trading_settings']['max_positions_per_symbol']
            if symbol_position_count >= max_per_symbol:
                return False, f"Max positions limit reached for {symbol} ({symbol_position_count}/{max_per_symbol})"
            return True, f"Position limit ok"
        except Exception as e:
            logging.error(f"Error checking position limit: {str(e)}")
            return False, f"Error checking position limit: {str(e)}"

class OrderManager:
    def __init__(self, client, market_data, indicator_calc, trade_logger, config):
        self.client = client
        self.market_data = market_data
        self.indicator_calc = indicator_calc
        self.trade_logger = trade_logger
        self.config = config

    def calculate_sl_tp(self, symbol, direction, entry_price, m15_data, medium_tf_swing_points):
        """
        Calculates SL and TP based on the most extreme price of the last N swing points.
        """
        if m15_data is None or m15_data.empty or medium_tf_swing_points is None:
            logging.warning(f"[{symbol}] Insufficient data for SL/TP calculation.")
            return None, None

        symbol_info = mt5.symbol_info(symbol)
        symbol_tick = mt5.symbol_info_tick(symbol)
        if not symbol_info or not symbol_tick:
            logging.error(f"[{symbol}] Could not get symbol info for SL/TP calculation.")
            return None, None
        
        risk_params = self.config['risk_management']
        num_swings = risk_params.get('num_swings_for_sl', 3)
        
        try:
            atr_period = risk_params.get('atr_period', 14)
            atr = talib.ATR(m15_data['high'], m15_data['low'], m15_data['close'], timeperiod=atr_period).iloc[-2]
            sl_buffer_atr_multiplier = risk_params.get('structural_sl_buffer_atr', 1.5)
            sl_buffer = atr * sl_buffer_atr_multiplier
        except Exception as e:
            logging.error(f"[{symbol}] Could not calculate ATR. Cannot place trade. Error: {e}")
            return None, None

        sl = None
        risk = 0

        if direction == 'buy':
            # Stop loss must go below the lowest of the last N MEDIUM timeframe swing lows.
            recent_lows = medium_tf_swing_points['lows']
            if recent_lows.empty:
                logging.warning(f"[{symbol}] Cannot place buy trade: No structural swing lows found on medium timeframe.")
                return None, None
            
            # Get the lowest price from the last 'num_swings' swing lows
            structural_sl_point = recent_lows['low'].tail(num_swings).min()
            
            sl = structural_sl_point - sl_buffer
            risk = entry_price - sl

        elif direction == 'sell':
            # Stop loss must go above the highest of the last N MEDIUM timeframe swing highs.
            recent_highs = medium_tf_swing_points['highs']
            if recent_highs.empty:
                logging.warning(f"[{symbol}] Cannot place sell trade: No structural swing highs found on medium timeframe.")
                return None, None

            # Get the highest price from the last 'num_swings' swing highs
            structural_sl_point = recent_highs['high'].tail(num_swings).max()

            sl = structural_sl_point + sl_buffer
            # Adjust for spread on sell stop orders
            spread = symbol_tick.ask - symbol_tick.bid
            sl += spread
            risk = sl - entry_price
        
        if risk <= 0:
            logging.warning(f"[{symbol}] Trade aborted. Initial SL calculation is invalid.")
            return None, None      

        # Calculate Take Profit based on a fixed Risk-to-Reward ratio
        rr_ratio = risk_params.get('rr_ratio', 2.0)
        tp = entry_price + (risk * rr_ratio) if direction == 'buy' else entry_price - (risk * rr_ratio)
        
        # Round final values to the symbol's correct number of decimal places
        digits = symbol_info.digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        
        logging.info(f"[{symbol}] Calculated valid SL: {sl}, TP: {tp} based on structural point from last {num_swings} swings.")
        return sl, tp
        
    def calculate_lot_size(self, symbol, entry_price, sl):
        account = self.client.get_account_info()
        if not account: return None
        balance = account.balance
        risk_amount = balance * (self.config['risk_management']['risk_per_trade_percent'] / 100)
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: return None
        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        if tick_size == 0: return None
        risk_pips = abs(entry_price - sl) / tick_size
        if risk_pips == 0 or tick_value == 0: return None
        lot_size = risk_amount / (risk_pips * tick_value)
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        return lot_size
    
    def place_order(self, symbol, direction, signal_entry_price, sl, tp):        
        symbol_tick = mt5.symbol_info_tick(symbol)
        if not symbol_tick: return None
        magic = self.config['misc']['mt5_magic_number']
        comment = self.config['misc']['trade_comment']
        
        if direction == "buy":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            execution_price = symbol_tick.ask
        elif direction == "sell":
            mt5_order_type = mt5.ORDER_TYPE_SELL
            execution_price = symbol_tick.bid
        else: return None
        
        lot_size = self.calculate_lot_size(symbol, signal_entry_price, sl)
        if lot_size is None or lot_size <= 0:
            logging.error(f"[{symbol}] Invalid lot size calculated: {lot_size}")
            return None
                             
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
            "type": mt5_order_type, "price": execution_price, "sl": sl, "tp": tp,
            "magic": magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
                    
        result = mt5.order_send(request)  
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to place {direction} order for {symbol}: {result.comment}")
        else:
            logging.info(f"Placed {direction} order for {symbol}")
        return result
        
async def run_main_loop(client, symbols, timeframes):
    market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
    trade_managers = {symbol: TradeManager(client, market_data_dict[symbol], CONFIG) for symbol in symbols}

    while True:
        start_time = time.time()
        tasks = [trade_managers[symbol].check_for_signals(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        cycle_duration = time.time() - start_time
        logging.info(f"Main loop cycle complete. Took {cycle_duration:.2f} seconds.")
        sleep_time = max(0, CHECK_INTERVAL - cycle_duration)
        logging.info(f"Next main loop cycle in {sleep_time:.2f} seconds.")
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
    finally:
        if client.is_initialized(): mt5.shutdown()
        logging.info("Trading script terminated")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Shutting down...")