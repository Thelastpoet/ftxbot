import time
import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import traceback
import asyncio
import os
import json

# Import our custom modules and config
import config
from price_action_analyzer import PriceActionAnalyzer
from order_manager import OrderManager

logging.basicConfig(level=logging.INFO, filename=config.LOG_FILE_NAME,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

    def is_initialized(self):
        return self.initialized

    def get_account_info(self):
        return mt5.account_info()

class MarketData:
    def __init__(self, symbol, timeframes):
        self.symbol = symbol
        self.timeframes = timeframes

    def fetch_data(self, timeframe):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, 1000) # Fetch ample data
        if rates is None: return None
        df = pd.DataFrame(rates)
        df = df[df['tick_volume'] != 0].reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

class TradeLogger:
    def __init__(self, filename):
        self.filename = filename
        self.columns = ['ticket_id', 'symbol', 'direction', 'open_time', 'open_price', 'stop_loss', 
                        'take_profit', 'lot_size', 'reason', 'market_context', 'close_time', 
                        'close_price', 'pnl', 'status']
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=self.columns).to_csv(self.filename, index=False)

    def log_open_trade(self, **kwargs):
        log_df = pd.read_csv(self.filename)
        if not log_df[log_df['ticket_id'] == kwargs['ticket_id']].empty: return
        kwargs['open_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kwargs['status'] = 'open'
        pd.concat([log_df, pd.DataFrame([kwargs])], ignore_index=True).to_csv(self.filename, index=False)

    def log_close_trade(self, ticket_id, close_price, close_time, pnl, status):
        log_df = pd.read_csv(self.filename)
        idx = log_df.index[log_df['ticket_id'] == ticket_id].tolist()
        if not idx: return
        log_df.loc[idx[0], ['close_price', 'close_time', 'pnl', 'status']] = \
            [close_price, close_time.strftime("%Y-%m-%d %H:%M:%S"), pnl, status]
        log_df.to_csv(self.filename, index=False)

    def get_open_trades(self, symbol: str):
        try:
            log_df = pd.read_csv(self.filename)
            return log_df[(log_df['symbol'] == symbol) & (log_df['status'] == 'open')]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

class TradeManager:
    def __init__(self, client, market_data, analyzer, order_manager, trade_logger, cfg):
        self.client = client
        self.market_data = market_data
        self.analyzer = analyzer
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.config = cfg
        self.tf_higher, self.tf_medium, self.tf_lower = max(cfg.TIME_FRAMES), sorted(cfg.TIME_FRAMES)[1], min(cfg.TIME_FRAMES)

    async def check_for_signals(self):
        symbol = self.market_data.symbol
        logging.info(f"[{symbol}] Starting analysis cycle...")
        
        # 1. Fetch data for all timeframes
        data_dict = {tf: self.market_data.fetch_data(tf) for tf in self.config.TIME_FRAMES}
        if any(d is None for d in data_dict.values()): return

        # 2. Get analysis for all timeframes
        analysis = {tf: self.analyzer.get_analysis(data) for tf, data in data_dict.items()}
        if any(a is None for a in analysis.values()): return

        # 3. Dynamic Strategy Selection
        primary_trend = analysis[self.tf_higher]['structure']['trend']
        
        trade_setup = None
        if primary_trend in ['bullish', 'bearish']:
            trade_setup = self._evaluate_trend_pullback_setup(analysis, primary_trend)
        elif primary_trend == 'ranging':
            # This is where a range-bound strategy would be called.
            logging.info(f"[{symbol}] Market phase is 'ranging'. No trend strategy applicable.")
        
        if trade_setup:
            await self._execute_trade(symbol, trade_setup)

    def _evaluate_trend_pullback_setup(self, analysis, primary_trend):
        symbol = self.market_data.symbol
        # Screen 2: H1 Setup - Find a Zone of Value
        h1_analysis = analysis[self.tf_medium]
        current_price = h1_analysis['data']['close'].iloc[-1]
        atr = h1_analysis['atr'].iloc[-1]
        h1_swings = h1_analysis['swing_points']

        setup_zone = None
        # --- FLEXIBLE SETUP LOGIC ---
        if primary_trend == 'bullish':
            # Check for pullback to last broken high (Polarity)
            broken_highs = h1_swings['highs'][h1_swings['highs']['high'] < current_price]
            if not broken_highs.empty and abs(current_price - broken_highs.iloc[-1]['high']) < atr * self.config.PULLBACK_ZONE_ATR_FACTOR:
                setup_zone = 'Polarity Support'
            # Check for pullback to last swing low (Trend Continuation)
            elif not h1_swings['lows'].empty and abs(current_price - h1_swings['lows'].iloc[-1]['low']) < atr * self.config.PULLBACK_ZONE_ATR_FACTOR:
                setup_zone = 'Continuation Support'

        elif primary_trend == 'bearish':
            # Check for pullback to last broken low (Polarity)
            broken_lows = h1_swings['lows'][h1_swings['lows']['low'] > current_price]
            if not broken_lows.empty and abs(current_price - broken_lows.iloc[-1]['low']) < atr * self.config.PULLBACK_ZONE_ATR_FACTOR:
                setup_zone = 'Polarity Resistance'
            # Check for pullback to last swing high (Trend Continuation)
            elif not h1_swings['highs'].empty and abs(current_price - h1_swings['highs'].iloc[-1]['high']) < atr * self.config.PULLBACK_ZONE_ATR_FACTOR:
                setup_zone = 'Continuation Resistance'

        if not setup_zone:
            logging.info(f"[{symbol}] D1 trend is '{primary_trend}', but price is not in a valid H1 setup zone.")
            return None

        # Screen 3: M15 Entry - Find a Confirmation of Intent
        m15_analysis = analysis[self.tf_lower]
        m15_swings = m15_analysis['swing_points']
        m15_price = m15_analysis['data']['close'].iloc[-1]

        entry_trigger = None
        # --- FLEXIBLE TRIGGER LOGIC ---
        if primary_trend == 'bullish':
            # Trigger A: Break of micro-structure
            if not m15_swings['highs'].empty and m15_price > m15_swings['highs'].iloc[-1]['high']:
                entry_trigger = 'Structure Break'
            # Trigger B: Order flow shift
            elif m15_analysis['reversal_candle'] == 'bullish_engulfing':
                entry_trigger = 'Bullish Engulfing'
        
        elif primary_trend == 'bearish':
            # Trigger A: Break of micro-structure
            if not m15_swings['lows'].empty and m15_price < m15_swings['lows'].iloc[-1]['low']:
                entry_trigger = 'Structure Break'
            # Trigger B: Order flow shift
            elif m15_analysis['reversal_candle'] == 'bearish_engulfing':
                entry_trigger = 'Bearish Engulfing'

        if not entry_trigger:
            logging.info(f"[{symbol}] Price in H1 zone '{setup_zone}', but no M15 entry trigger found.")
            return None

        logging.info(f"[{symbol}] SUCCESS: VALID TRADE SETUP: {primary_trend.upper()} | H1 Zone: {setup_zone} | M15 Trigger: {entry_trigger}")
        return {'direction': 'buy' if primary_trend == 'bullish' else 'sell',
                'reason': f"{setup_zone} + {entry_trigger}"}

    async def _execute_trade(self, symbol, trade_setup):
        try:
            # Position Limit Check
            positions = mt5.positions_get()
            if positions is not None:
                if len(positions) >= self.config.MAX_OPEN_POSITIONS_TOTAL: return
                if len([p for p in positions if p.symbol == symbol]) >= self.config.MAX_OPEN_POSITIONS_PER_SYMBOL: return
            
            # SL/TP Calculation
            m15_swings = self.analyzer.get_analysis(self.market_data.fetch_data(self.tf_lower))['swing_points']
            direction = trade_setup['direction']
            point = mt5.symbol_info(symbol).point
            
            sl_level = None
            if direction == 'buy' and not m15_swings['lows'].empty:
                sl_level = m15_swings['lows'].iloc[-1]['low'] - (self.config.STOP_LOSS_BUFFER_POINTS * point)
            elif direction == 'sell' and not m15_swings['highs'].empty:
                sl_level = m15_swings['highs'].iloc[-1]['high'] + (self.config.STOP_LOSS_BUFFER_POINTS * point)
            
            if sl_level is None: return

            current_price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
            stop_loss_pips = abs(current_price - sl_level) / point
            take_profit = current_price + (stop_loss_pips * self.config.RISK_REWARD_RATIO * point) if direction == 'buy' else \
                          current_price - (stop_loss_pips * self.config.RISK_REWARD_RATIO * point)

            # Execution
            trade_result = self.order_manager.place_order(symbol, direction, sl_level, take_profit)
            if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                logging.info(f"[{symbol}] Successfully opened trade {trade_result.order}.")
                

        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {e}\n{traceback.format_exc()}")
            
    async def manage_open_positions(self):
        symbol = self.market_data.symbol
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

async def main_runner(client):
    analyzer = PriceActionAnalyzer(swing_lookback_period=config.SWING_LOOKBACK_PERIOD)
    trade_logger = TradeLogger(config.TRADE_LOG_FILE_NAME)

    managers = [TradeManager(client, MarketData(symbol, config.TIME_FRAMES), analyzer,
                             OrderManager(client, config), trade_logger, config)
                for symbol in config.SYMBOLS]

    while True:
        start_time = time.time()
        tasks = [mgr.check_for_signals() for mgr in managers]
        tasks.extend([mgr.manage_open_positions() for mgr in managers])
        await asyncio.gather(*tasks)
        
        cycle_duration = time.time() - start_time
        logging.info(f"--- Main loop cycle complete in {cycle_duration:.2f}s. Waiting {config.CHECK_INTERVAL_SECONDS}s. ---")
        await asyncio.sleep(config.CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    client = MetaTrader5Client()
    if client.is_initialized():
        try:
            asyncio.run(main_runner(client))
        except (asyncio.CancelledError, KeyboardInterrupt):
            logging.info("Trading script terminated by user.")
        finally:
            logging.info("Shutting down...")