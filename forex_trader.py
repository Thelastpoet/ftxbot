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
import json

from talib import ATR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_PERIOD = 1000
CHECK_INTERVAL = 300

class MetaTrader5Client:
    def __init__(self):
        self.initialized = mt5.initialize()
        if self.initialized:
            logging.info("MetaTrader5 initialized successfully.")
        else:
            logging.error("Failed to initialize MetaTrader5.")

    def __del__(self):
        if self.initialized:
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
        self.num_candles[timeframe] = MAX_PERIOD * 2

    def fetch_data(self, timeframe):
        if self.num_candles[timeframe] is None:
            self.calculate_num_candles(timeframe)
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.num_candles[timeframe])
        if rates is None:
            logging.error(f"No rates for {self.symbol} on {timeframe}")
            return None
        df = pd.DataFrame(rates)
        df = df[df['tick_volume'] != 0].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

class SwingPointDetector:
    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period

    def find_swing_points(self, data):
        if not isinstance(data, pd.DataFrame) or not {'high', 'low'}.issubset(data.columns) or data.empty or len(data) < self.lookback_period:
            return None
        try:
            high_indices = argrelextrema(data['high'].values, np.greater_equal, order=self.lookback_period)[0]
            low_indices = argrelextrema(data['low'].values, np.less_equal, order=self.lookback_period)[0]
            return {'highs': data.iloc[high_indices], 'lows': data.iloc[low_indices]}
        except Exception as e:
            logging.error(f"Error finding swing points: {e}")
            return None

    def analyze_market_structure(self, swing_points):
        if not swing_points or swing_points['highs'].empty or swing_points['lows'].empty or len(swing_points['highs']) < 2 or len(swing_points['lows']) < 2:
            return {'trend': 'Insufficient Data'}
        highs, lows = swing_points['highs']['high'], swing_points['lows']['low']
        is_bullish = highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]
        is_bearish = highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]
        if is_bullish: return {'trend': 'bullish'}
        elif is_bearish: return {'trend': 'bearish'}
        else: return {'trend': 'ranging'}

class TradeManager:
    def __init__(self, client, market_data):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        self.trade_logger = TradeLogger()
        self.swing_detector = SwingPointDetector()
        self.order_manager = OrderManager(client)
        self.max_positions = 20
        self.tf_higher, self.tf_medium, self.tf_lower = max(self.timeframes), sorted(self.timeframes)[1], min(self.timeframes)

    def _prepare_analysis_data(self, data_dict):
        analysis_data = {}
        for tf, data in data_dict.items():
            if data is None or data.empty: return None
            swing_points = self.swing_detector.find_swing_points(data)
            structure = self.swing_detector.analyze_market_structure(swing_points)
            atr_series = ATR(data['high'], data['low'], data['close'], timeperiod=14)
            if swing_points is None or structure is None or atr_series is None: return None
            analysis_data[tf] = {'structure': structure, 'swing_points': swing_points, 'data': data, 'atr': atr_series}
        return analysis_data

    async def check_for_signals(self, symbol):
        logging.info(f"[{symbol}] Starting analysis cycle...")
        try:
            data_dict = {tf: self.market_data.fetch_data(tf) for tf in self.timeframes}
            if any(d is None for d in data_dict.values()): return

            market_analysis = self._prepare_analysis_data(data_dict)
            if not market_analysis:
                logging.warning(f"[{symbol}] Could not prepare analysis data. Skipping cycle.")
                return
            
            trade_setup = self._evaluate_price_action_setup(market_analysis)
            if trade_setup:
                await self._execute_trade(symbol, trade_setup, market_analysis)
        except Exception as e:
            logging.error(f"CRITICAL ERROR in check_for_signals for {symbol}: {str(e)}\n{traceback.format_exc()}")

    def _evaluate_price_action_setup(self, market_analysis):
        symbol = self.market_data.symbol
        primary_trend = market_analysis[self.tf_higher]['structure']['trend']
        if primary_trend not in ['bullish', 'bearish']:
            logging.info(f"[{symbol}] SCREEN 1 FAILED: D1 trend is '{primary_trend}'. Standing aside.")
            return None

        h1_analysis = market_analysis[self.tf_medium]
        h1_swing_points = h1_analysis['swing_points']
        current_price = h1_analysis['data']['close'].iloc[-1]
        atr = h1_analysis['atr'].iloc[-1]
        
        setup_zone = None
        if primary_trend == 'bullish':
            relevant_highs = h1_swing_points['highs'][h1_swing_points['highs']['high'] < current_price]
            if relevant_highs.empty:
                logging.info(f"[{symbol}] SCREEN 2 FAILED: D1 is bullish, but no broken H1 swing high found to act as support.")
                return None
            pullback_zone_price = relevant_highs.iloc[-1]['high']
            if abs(current_price - pullback_zone_price) < atr:
                setup_zone = {'price': pullback_zone_price, 'type': 'support'}
        elif primary_trend == 'bearish':
            relevant_lows = h1_swing_points['lows'][h1_swing_points['lows']['low'] > current_price]
            if not relevant_lows.empty:
                logging.info(f"[{symbol}] SCREEN 2 FAILED: D1 is bearish, but no broken H1 swing low found to act as resistance.")
                return None
            pullback_zone_price = relevant_lows.iloc[-1]['low']
            if abs(current_price - pullback_zone_price) < atr:
                setup_zone = {'price': pullback_zone_price, 'type': 'resistance'}
        
        if not setup_zone:
            logging.info(f"[{symbol}] SCREEN 2 FAILED: Price is not currently in the H1 pullback zone.")
            return None

        m15_analysis = market_analysis[self.tf_lower]
        m15_swing_points = m15_analysis['swing_points']
        m15_current_price = m15_analysis['data']['close'].iloc[-1]

        entry_trigger = False
        if primary_trend == 'bullish' and m15_swing_points and not m15_swing_points['highs'].empty:
            last_micro_high = m15_swing_points['highs']['high'].iloc[-1]
            if m15_current_price > last_micro_high:
                entry_trigger = True
        elif primary_trend == 'bearish' and m15_swing_points and not m15_swing_points['lows'].empty:
            last_micro_low = m15_swing_points['lows']['low'].iloc[-1]
            if m15_current_price < last_micro_low:
                entry_trigger = True

        if not entry_trigger:
            logging.info(f"[{symbol}] SCREEN 3 FAILED: Price is in H1 zone, but M15 trigger has not occurred.")
            return None
        
        logging.info(f"[{symbol}] SUCCESS: ALL 3 SCREENS PASSED. VALID TRADE SETUP FOUND: {primary_trend.upper()}")
        return {'valid': True, 'direction': 'buy' if primary_trend == 'bullish' else 'sell', 'setup_type': 'price_action_pullback'}

    async def _execute_trade(self, symbol, trade_setup, market_analysis):
        try:
            can_trade, reason = self.check_position_limit(symbol)
            if not can_trade:
                logging.info(f"[{symbol}] Trade execution halted: {reason}")
                return
            order_params = self.order_manager.calculate_price_action_sl_tp(symbol, trade_setup['direction'],
                                                                           market_analysis[self.tf_lower]['swing_points'],
                                                                           market_analysis[self.tf_higher]['swing_points'])
            if not order_params or not order_params['is_valid']:
                logging.info(f"[{symbol}] Trade invalidated by OrderManager (Risk/Reward or SL placement).")
                return
                
            trade_result = self.order_manager.place_order(symbol, order_params['lot_size'], trade_setup['direction'],
                                                          order_params['stop_loss'], order_params['take_profit'])

            if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Successfully opened trade {trade_result.order}.")
                # --- THIS IS THE CRITICAL FIX ---
                primary_trend = market_analysis[self.tf_higher]['structure']['trend']
                market_context = {'d1_trend': primary_trend}
                self.trade_logger.log_open_trade(ticket_id=trade_result.order, symbol=symbol, direction=trade_setup['direction'],
                                                 open_price=trade_result.price, stop_loss=order_params['stop_loss'],
                                                 take_profit=order_params['take_profit'], lot_size=order_params['lot_size'],
                                                 reason=trade_setup['setup_type'], market_context=json.dumps(market_context))
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {str(e)}\n{traceback.format_exc()}")

    def check_position_limit(self, symbol: str) -> Tuple[bool, str]:
        try:
            positions = mt5.positions_get()
            if positions is None: return True, "No positions"
            if len(positions) >= self.max_positions: return False, f"Max positions ({self.max_positions}) reached"
            if len([p for p in positions if p.symbol == symbol]) >= 2: return False, f"Max positions for {symbol} reached"
            return True, "OK"
        except Exception as e:
            logging.error(f"Error checking position limit: {e}")
            return False, "Error"

    async def manage_open_positions(self, symbol: str):
        try:
            open_trades = self.trade_logger.get_open_trades(symbol)
            if open_trades.empty: return
            open_mt5_tickets = {pos.ticket for pos in mt5.positions_get(symbol=symbol) or []}
            for _, trade in open_trades.iterrows():
                if trade['ticket_id'] not in open_mt5_tickets:
                    deals = mt5.history_deals_get(position=trade['ticket_id'])
                    if deals:
                        deal = deals[-1]
                        status = "closed_manual"
                        if abs(deal.price - trade['take_profit']) < 1e-5: status = "closed_tp"
                        elif abs(deal.price - trade['stop_loss']) < 1e-5: status = "closed_sl"
                        self.trade_logger.log_close_trade(ticket_id=trade['ticket_id'], close_price=deal.price,
                                                          close_time=pd.to_datetime(deal.time, unit='s'), pnl=deal.profit, status=status)
                        logging.info(f"[{symbol}] Logged closure for trade {trade['ticket_id']}.")
        except Exception as e:
            logging.error(f"Error managing positions for {symbol}: {e}\n{traceback.format_exc()}")

class OrderManager:
    def __init__(self, client, risk_percentage=1.0, rr_ratio=2.0):
        self.client = client
        self.risk_percentage = risk_percentage
        self.rr_ratio = rr_ratio

    def calculate_price_action_sl_tp(self, symbol, direction, m15_swing_points, d1_swing_points):
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info: return {'is_valid': False}
        point = symbol_info.point
        
        sl_level = None
        if direction == 'buy' and m15_swing_points and not m15_swing_points['lows'].empty:
            sl_level = m15_swing_points['lows']['low'].iloc[-1] - (10 * point)
        elif direction == 'sell' and m15_swing_points and not m15_swing_points['highs'].empty:
            sl_level = m15_swing_points['highs']['high'].iloc[-1] + (10 * point)
        
        if sl_level is None: return {'is_valid': False}

        tick = mt5.symbol_info_tick(symbol)
        if not tick: return {'is_valid': False}
        current_price = tick.ask if direction == 'buy' else tick.bid
        
        if (direction == 'buy' and sl_level >= current_price) or (direction == 'sell' and sl_level <= current_price):
            return {'is_valid': False}

        stop_loss_pips = abs(current_price - sl_level) / point
        if stop_loss_pips < 5: return {'is_valid': False}

        take_profit = current_price + ((stop_loss_pips * self.rr_ratio) * point) if direction == 'buy' else current_price - ((stop_loss_pips * self.rr_ratio) * point)

        lot_size = self._calculate_lot_size(symbol, stop_loss_pips)
        if not lot_size or lot_size <= 0: return {'is_valid': False}

        return {'is_valid': True, 'stop_loss': sl_level, 'take_profit': take_profit, 'lot_size': lot_size}

    def _calculate_lot_size(self, symbol, stop_loss_pips):
        account_info = self.client.get_account_info()
        symbol_info = mt5.symbol_info(symbol)
        if not account_info or not symbol_info: return None
        risk_amount = account_info.balance * (self.risk_percentage / 100)
        pip_value_per_lot = symbol_info.trade_tick_value
        if symbol_info.trade_tick_size != symbol_info.point:
             pip_value_per_lot = (pip_value_per_lot / symbol_info.trade_tick_size) * symbol_info.point
        risk_in_currency = stop_loss_pips * pip_value_per_lot
        if risk_in_currency <= 0: return None
        lot_size = risk_amount / risk_in_currency
        lot_step = symbol_info.volume_step
        lot_size = math.floor(lot_size / lot_step) * lot_step
        return max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)

    def place_order(self, symbol, lots_size, direction, stop_loss, take_profit):
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return None
        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 'buy' else tick.bid
        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots_size, "type": order_type,
                   "price": price, "sl": stop_loss, "tp": take_profit, "magic": 234001, "comment": "PA_Trader_v2.2",
                   "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed, retcode={result.retcode}, comment={result.comment}")
        return result

class TradeLogger:
    def __init__(self, filename='trade_log.csv'):
        self.filename = filename
        self.columns = [ 'ticket_id', 'symbol', 'direction', 'open_time', 'open_price', 'stop_loss', 'take_profit', 'lot_size', 'reason', 'market_context', 'close_time', 'close_price', 'pnl', 'status' ]
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=self.columns).to_csv(self.filename, index=False)

    def _load_log(self) -> pd.DataFrame:
        try: return pd.read_csv(self.filename)
        except (FileNotFoundError, pd.errors.EmptyDataError): return pd.DataFrame(columns=self.columns)

    def log_open_trade(self, **kwargs):
        log_df = self._load_log()
        if not log_df[log_df['ticket_id'] == kwargs['ticket_id']].empty: return
        kwargs['open_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kwargs['status'] = 'open'
        pd.concat([log_df, pd.DataFrame([kwargs])], ignore_index=True).to_csv(self.filename, index=False)

    def log_close_trade(self, ticket_id, close_price, close_time, pnl, status):
        log_df = self._load_log()
        idx = log_df.index[log_df['ticket_id'] == ticket_id].tolist()
        if not idx: return
        log_df.loc[idx[0], ['close_price', 'close_time', 'pnl', 'status']] = [close_price, close_time.strftime("%Y-%m-%d %H:%M:%S"), pnl, status]
        log_df.to_csv(self.filename, index=False)

    def get_open_trades(self, symbol: str):
        log_df = self._load_log()
        if log_df.empty: return pd.DataFrame()
        return log_df[(log_df['symbol'] == symbol) & (log_df['status'] == 'open')]

async def run_main_loop(client, symbols, timeframes):
    trade_managers = {symbol: TradeManager(client, MarketData(symbol, timeframes)) for symbol in symbols}
    while True:
        start_time = time.time()
        tasks = [manager.check_for_signals(symbol) for symbol, manager in trade_managers.items()]
        tasks.extend([manager.manage_open_positions(symbol) for symbol, manager in trade_managers.items()])
        await asyncio.gather(*tasks)
        cycle_duration = time.time() - start_time
        logging.info(f"--- Main loop cycle complete in {cycle_duration:.2f}s. Waiting {CHECK_INTERVAL}s for next cycle. ---")
        await asyncio.sleep(CHECK_INTERVAL)

async def main():
    client = MetaTrader5Client()
    if not client.is_initialized(): return
    symbols = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'EURCAD', 'AUDCHF', 'AUDCAD', 'EURGBP', 'EURAUD', 'EURCHF', 'EURNZD', 'AUDNZD', 'GBPCHF', 'CADCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDUSD']
    timeframes = (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)
    try:
        await run_main_loop(client, symbols, timeframes)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.info("Trading script terminated by user.")
    finally:
        logging.info("Shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        logging.error(f"RuntimeError in main execution: {e}")