import time
import logging
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import traceback
import asyncio
import numpy as np
import os
from typing import Tuple, Dict
import json
from dataclasses import asdict
import pytz

from technical_analysis import IndicatorCalculator
from context_engine import ContextEngine, MarketNarrative
from smc import smc

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.num_candles[timeframe] = 2000

    def fetch_data(self, timeframe):
        if self.num_candles.get(timeframe) is None:
            self.calculate_num_candles(timeframe)

        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, self.num_candles[timeframe])
        if rates is None:
            logging.error(f"No rates available for {self.symbol} on timeframe {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df = df[df['tick_volume'] != 0]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')
        df.set_index('time', inplace=True)
        df.name = self.symbol
        return df

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
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=self.columns).to_csv(self.filename, index=False)

    def _load_log(self):
        try:
            return pd.read_csv(self.filename)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=self.columns)

    def get_open_trades(self, symbol: str):
        log_df = self._load_log()
        if log_df.empty:
            return pd.DataFrame()
        return log_df[(log_df['symbol'] == symbol) & (log_df['status'] == 'open')]

    def log_open_trade(self, **kwargs):
        try:
            log_df = self._load_log()
            if not log_df.empty and not log_df[log_df['ticket_id'] == kwargs.get('ticket_id')].empty:
                logging.warning(f"Trade {kwargs.get('ticket_id')} already logged. Ignoring.")
                return

            kwargs['open_time'] = datetime.now()
            kwargs['status'] = 'open'
            new_trade = pd.DataFrame([kwargs])
            log_df = pd.concat([log_df, new_trade], ignore_index=True)
            log_df.to_csv(self.filename, index=False)
            logging.info(f"Logged new open trade {kwargs.get('ticket_id')} for {kwargs.get('symbol')}.")
        except Exception as e:
            logging.error(f"Error logging open trade: {e}")

    def log_close_trade(self, ticket_id: int, close_price: float, close_time: datetime, pnl: float, status: str):
        try:
            log_df = self._load_log()
            trade_index = log_df.index[log_df['ticket_id'] == ticket_id].tolist()
            if not trade_index:
                logging.warning(f"Could not find trade with ticket_id {ticket_id} to close.")
                return
            
            idx = trade_index[0]
            log_df.loc[idx, ['close_price', 'close_time', 'pnl', 'status']] = [close_price, close_time, pnl, status]
            log_df.to_csv(self.filename, index=False)
        except Exception as e:
            logging.error(f"Error logging closed trade: {e}")

class OrderManager:
    def __init__(self, client: mt5, risk_config: Dict, mt5_config: Dict):
        self.client = client
        self.risk_per_trade_percent = risk_config['risk_per_trade_percent'] / 100.0 # Convert 1.0 to 0.01
        self.min_rr_ratio = risk_config['minimum_risk_reward_ratio']
        self.magic_number = mt5_config['magic_number']

    def calculate_contextual_order_parameters(self, symbol: str, direction: str, hypothesis: MarketNarrative) -> Dict:
        try:
            account_info = self.client.account_info()
            symbol_info = self.client.symbol_info(symbol)
            symbol_tick = self.client.symbol_info_tick(symbol)
            if not all([account_info, symbol_info, symbol_tick]):
                logging.error(f"[{symbol}] Could not retrieve all required info for order calculation.")
                return None

            point = symbol_info.point
            current_price = symbol_tick.ask if direction == 'buy' else symbol_tick.bid
            stop_loss = hypothesis.invalidation_level
            take_profit = hypothesis.liquidity_target

            rates = self.client.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
            atr_df = pd.DataFrame(rates) if rates is not None else pd.DataFrame()
            if not atr_df.empty and len(atr_df) > 1:
                atr_df['tr'] = pd.concat([(atr_df['high'] - atr_df['low']), (atr_df['high'] - atr_df['close'].shift()).abs(), (atr_df['low'] - atr_df['close'].shift()).abs()], axis=1).max(axis=1)
                atr_buffer = atr_df['tr'].ewm(alpha=1/14, adjust=False).mean().iloc[-1] * 0.1
                stop_loss += -atr_buffer if direction == 'buy' else atr_buffer

            if (direction == 'buy' and current_price <= stop_loss) or \
               (direction == 'sell' and current_price >= stop_loss):
                logging.warning(f"[{symbol}] Trade invalidated. Entry price ({current_price}) is already beyond calculated SL ({stop_loss}).")
                return None

            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk <= 0 or (reward / risk) < self.min_rr_ratio:
                logging.warning(f"[{symbol}] Trade invalidated. Poor Risk:Reward ({(reward/risk):.2f}). Required: {self.min_rr_ratio}")
                return None

            stop_loss_points = risk / point
            risk_amount = account_info.balance * self.risk_per_trade_percent
            value_per_point = symbol_info.trade_tick_value / symbol_info.trade_tick_size * symbol_info.point
            loss_for_one_lot = stop_loss_points * value_per_point
            lot_size = round(risk_amount / loss_for_one_lot, 2) if loss_for_one_lot > 0 else 0.0
            lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)

            if lot_size < symbol_info.volume_min:
                logging.warning(f"[{symbol}] Calculated lot size {lot_size} is below minimum {symbol_info.volume_min}.")
                return None

            return {
                'lot_size': lot_size,
                'stop_loss': round(stop_loss, symbol_info.digits),
                'take_profit': round(take_profit, symbol_info.digits)
            }
        except Exception as e:
            logging.error(f"Error in calculate_contextual_order_parameters: {e}\n{traceback.format_exc()}")
            return None

    def check_position_limit(self, symbol: str, max_positions: int) -> Tuple[bool, str]:
        try:
            positions = self.client.positions_get()
            if positions is None: return True, "No positions found"
            if len(positions) >= max_positions: return False, f"Max overall positions limit reached ({max_positions})"
            return True, "OK"
        except Exception as e:
            logging.error(f"Error checking position limit: {e}")
            return False, str(e)

    def has_opposing_position(self, symbol: str, direction: str) -> bool:
        try:
            positions = self.client.positions_get(symbol=symbol)
            if not positions: return False
            for pos in positions:
                if (direction == 'buy' and pos.type == self.client.POSITION_TYPE_SELL) or \
                   (direction == 'sell' and pos.type == self.client.POSITION_TYPE_BUY):
                    return True
            return False
        except Exception as e:
            logging.error(f"Error checking opposing position: {e}")
            return True

    def place_order(self, symbol: str, direction: str, lot_size: float, stop_loss: float, take_profit: float):
        order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if direction == 'buy' else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
            "type": order_type, "price": price, "sl": stop_loss, "tp": take_profit,
            "magic": self.magic_number,
            "comment": "Context-Driven Trade",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed for {symbol}: {result.comment}")
        return result

class TradeManager:
    def __init__(self, client: MetaTrader5Client, market_data: MarketData, config: Dict):
        self.client = client
        self.market_data = market_data
        self.timeframes = market_data.timeframes
        
        self.trade_logger = TradeLogger(filename=config['logging']['trade_log_file'])
        self.indicator_calc = IndicatorCalculator()
        self.context_engine = ContextEngine(strategy_config=config['amd_strategy_parameters'])
        self.order_manager = OrderManager(mt5, config['risk_management'], config['metatrader_settings'])
        
        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1]
        self.tf_lower = min(self.timeframes)
        self.max_positions = config['asset_management']['max_overall_positions']

    def _prepare_data(self, symbol: str) -> Tuple[Dict, Dict]:
        try:
            data = {tf: self.market_data.fetch_data(tf) for tf in self.timeframes}
            if any(d is None or d.empty for d in data.values()): return None, None
            
            indicators = {tf: self.indicator_calc.calculate_indicators(d) for tf, d in data.items()}
            if any(i is None or i.empty for i in indicators.values()): return None, None

            min_len = min(len(df) for df in indicators.values())
            data = {tf: d.iloc[-min_len:] for tf, d in data.items()}
            indicators = {tf: i.iloc[-min_len:] for tf, i in indicators.items()}
            
            return data, indicators
        except Exception as e:
            logging.error(f"Error preparing data for {symbol}: {e}")
            return None, None

    async def check_for_signals(self, symbol: str):
        try:
            data, indicators = self._prepare_data(symbol)
            if not data or not indicators:
                return

            hypothesis = self.context_engine.process_data(data[self.tf_lower], indicators[self.tf_higher])

            if hypothesis.clarity_score != "High" or "MANIPULATION_CONFIRMED" not in hypothesis.session_profile:
                logging.info(f"[{symbol}] STANDING ASIDE. No high-clarity AMD reversal context. Narrative: {hypothesis.narrative_summary}")
                return

            trade_setup = self._evaluate_trade_setup(data, indicators, hypothesis)

            if not trade_setup:
                logging.info(f"[{symbol}] AMD Hypothesis ({hypothesis.daily_bias}) is valid, but awaiting specialist entry confirmation.")
                return

            await self._execute_trade(symbol, trade_setup, hypothesis)

        except Exception as e:
            logging.error(f"Error in check_for_signals for {symbol}: {e}\n{traceback.format_exc()}")

    def _evaluate_trade_setup(self, data: dict, indicators: dict, hypothesis: MarketNarrative):
        return self._evaluate_amd_reversal_setup(data, indicators, hypothesis)

    def _evaluate_amd_reversal_setup(self, data: dict, indicators: dict, hypothesis: MarketNarrative) -> Dict:
        """
        Orchestrates the specialist AMD entry evaluation by breaking it into logical steps.
        This refactored version addresses issues of single responsibility, repetition, and error handling.
        """
        SWING_LENGTH = self.context_engine.ltf_swing_length
        NUM_ZONES_TO_CHECK = self.context_engine.zones_to_check
        
        # --- Initial Setup ---
        ltf_df = data[self.tf_lower].copy()
        direction = hypothesis.daily_bias.lower()
        symbol = self.market_data.symbol
        logging.info(f"[{symbol}] Starting AMD evaluation for {direction} setup.")
        
        # --- Workflow ---
        manipulation_candle = self._find_manipulation_extreme(ltf_df, direction, hypothesis)
        if manipulation_candle is None:
            return None

        mss_level = self._confirm_market_structure_shift(ltf_df, direction, manipulation_candle, SWING_LENGTH)
        if mss_level is None:
            return None
        logging.info(f"[{symbol}] Market Structure Shift CONFIRMED above {mss_level:.5f}. Searching for entry.")

        entry_zone = self._find_entry_zone(ltf_df, direction, SWING_LENGTH, NUM_ZONES_TO_CHECK)
        if not entry_zone:
            logging.info(f"[{symbol}] MSS confirmed, but no valid FVG or Order Block was found for entry.")
            return None
            
        trade_setup = self._confirm_entry_trigger(ltf_df, indicators[self.tf_lower], direction, entry_zone)
        if not trade_setup:
            return None
            
        logging.info(f"[{symbol}] AMD ENTRY CONFIRMED! Price reacted inside the {entry_zone['type']} with a valid rejection candle.")
        return trade_setup

    def _find_manipulation_extreme(self, ltf_df: pd.DataFrame, direction: str, hypothesis: MarketNarrative) -> pd.Series:
        """Finds the extreme candle of the manipulation leg based on the level identified by the ContextEngine."""
        symbol = self.market_data.symbol
        
        # Get today's date and the London start time from the context engine
        current_day = self.context_engine.current_day
        london_start_time = self.context_engine.SESSION_WINDOWS['london']['start']
        london_session_start_dt = datetime.combine(current_day, london_start_time).replace(tzinfo=pytz.UTC)

        # Filter the DataFrame to only include candles from today's London session onwards
        todays_london_leg = ltf_df[ltf_df.index >= london_session_start_dt]

        if todays_london_leg.empty:
            logging.warning(f"[{symbol}] No data available for today's London session to find manipulation extreme.")
            return None

        extreme_level = hypothesis.invalidation_level
        extreme_candle = None

        try:
            if direction == 'bullish':  # Manipulation was a sweep of lows, setting up a BUY
                # Find the first candle that made this low
                extreme_candles = todays_london_leg[todays_london_leg['low'] == extreme_level]
                if not extreme_candles.empty:
                    extreme_candle = extreme_candles.iloc[0]  # Take the first occurrence
            else:  # Bearish, manipulation was a sweep of highs, setting up a SELL
                extreme_candles = todays_london_leg[todays_london_leg['high'] == extreme_level]
                if not extreme_candles.empty:
                    extreme_candle = extreme_candles.iloc[0]  # Take the first occurrence

            if extreme_candle is None:
                # Use a tolerance for floating point comparison
                tolerance = 0.00001 
                if direction == 'bullish':
                    extreme_candles = todays_london_leg[abs(todays_london_leg['low'] - extreme_level) < tolerance]
                else:
                    extreme_candles = todays_london_leg[abs(todays_london_leg['high'] - extreme_level) < tolerance]
                
                if not extreme_candles.empty:
                    extreme_candle = extreme_candles.iloc[0]
                else:
                    logging.warning(f"[{symbol}] Could not find the extreme manipulation candle for level {extreme_level:.5f} in today's data.")
                    return None
            
            logging.info(f"[{symbol}] Found manipulation extreme candle ({'lowest low' if direction == 'bullish' else 'highest high'}): {extreme_level:.5f} at {extreme_candle.name}")
            return extreme_candle
            
        except Exception as e:
            logging.error(f"[{symbol}] Error finding extreme candle: {e}")
            return None

    def _confirm_market_structure_shift(self, ltf_df: pd.DataFrame, direction: str, manipulation_candle: pd.Series, swing_length: int) -> float:
        """Confirms if price has broken the key swing point that preceded the manipulation."""
        symbol = self.market_data.symbol
        
        full_swings = smc.swing_highs_lows(ltf_df, swing_length=swing_length)
        if len(full_swings) != len(ltf_df):
            logging.error(f"[{symbol}] Mismatch in length between OHLC data ({len(ltf_df)}) and swings ({len(full_swings)}). Cannot confirm MSS.")
            return None
        full_swings.index = ltf_df.index
        full_swings.dropna(inplace=True)

        swings_before_extreme = full_swings[full_swings.index < manipulation_candle.name]
        
        swing_type_to_break = 1 if direction == 'bullish' else -1
        relevant_swings = swings_before_extreme[swings_before_extreme['HighLow'] == swing_type_to_break]

        if relevant_swings.empty:
            log_msg = "prior swing high" if direction == 'bullish' else "prior swing low"
            logging.info(f"[{symbol}] No {log_msg} found to define an MSS level.")
            return None
            
        mss_level = relevant_swings['Level'].iloc[-1]
        current_close = ltf_df['close'].iloc[-1]
        
        mss_confirmed = (direction == 'bullish' and current_close > mss_level) or \
                        (direction == 'bearish' and current_close < mss_level)
        
        if not mss_confirmed:
            log_dir = 'above' if direction == 'bullish' else 'below'
            logging.info(f"[{symbol}] Awaiting Market Structure Shift; price needs to break {log_dir} {mss_level:.5f}.")
            return None
            
        return mss_level
    
    def _find_entry_zone(self, ltf_df: pd.DataFrame, direction: str, swing_length: int, num_zones: int) -> Dict:
        symbol = self.market_data.symbol
        fvg_zone = self._find_fvg_entry(ltf_df, direction, num_zones)
        if fvg_zone:
            logging.info(f"[{symbol}] Valid FVG found as entry zone between [{fvg_zone['bottom']:.5f} - {fvg_zone['top']:.5f}].")
            return fvg_zone
        logging.info(f"[{symbol}] No suitable FVG found. Searching for Order Blocks as fallback...")
        ob_zone = self._find_ob_entry(ltf_df, direction, swing_length, num_zones)
        if ob_zone:
            logging.info(f"[{symbol}] No FVG found. Using Order Block between [{ob_zone['bottom']:.5f} - {ob_zone['top']:.5f}].")
            return ob_zone
        return None

    def _find_fvg_entry(self, ltf_df: pd.DataFrame, direction: str, num_zones: int) -> Dict:
        symbol = self.market_data.symbol
        try:
            fvgs = smc.fvg(ltf_df)
            valid_fvgs = fvgs.dropna(subset=['FVG'])
            unmitigated_fvgs = valid_fvgs[valid_fvgs['MitigatedIndex'].isna()]
        except Exception as e:
            logging.error(f"[{symbol}] Error calculating FVGs: {e}")
            return None
        if unmitigated_fvgs.empty: return None
        fvg_type = 1 if direction == 'bullish' else -1
        entry_fvgs = unmitigated_fvgs[unmitigated_fvgs['FVG'] == fvg_type].tail(num_zones)
        if entry_fvgs.empty: return None
        last_fvg = entry_fvgs.iloc[-1]
        return {'type': 'FVG', 'top': last_fvg['Top'], 'bottom': last_fvg['Bottom']}

    def _find_ob_entry(self, ltf_df: pd.DataFrame, direction: str, swing_length: int, num_zones: int) -> Dict:
        symbol = self.market_data.symbol
        try:
            swings = smc.swing_highs_lows(ltf_df, swing_length=swing_length)
            swings.index = ltf_df.index
            order_blocks = smc.ob(ltf_df, swings)
            valid_obs = order_blocks.dropna(subset=['OB'])
            unmitigated_obs = valid_obs[valid_obs['MitigatedIndex'].isna()]
        except Exception as e:
            logging.error(f"[{symbol}] Error calculating Order Blocks: {e}")
            return None
        if unmitigated_obs.empty: return None
        ob_type = 1 if direction == 'bullish' else -1
        entry_obs = unmitigated_obs[unmitigated_obs['OB'] == ob_type].tail(num_zones)
        if entry_obs.empty: return None
        last_ob = entry_obs.iloc[-1]
        return {'type': 'OB', 'top': last_ob['Top'], 'bottom': last_ob['Bottom']}
        
    def _confirm_entry_trigger(self, ltf_df: pd.DataFrame, ltf_indicators: pd.DataFrame, direction: str, entry_zone: Dict) -> Dict:
        symbol = self.market_data.symbol
        if ltf_df.empty or ltf_indicators.empty: return None
        last_candle_close = ltf_df['close'].iloc[-1]
        last_candle_indicators = ltf_indicators.iloc[-1]
        is_in_zone = entry_zone['bottom'] <= last_candle_close <= entry_zone['top']
        if not is_in_zone: return None
        hammer = last_candle_indicators.get('cdl_hammer', 0)
        shooting_star = last_candle_indicators.get('cdl_shooting_star', 0)
        engulfing = last_candle_indicators.get('cdl_engulfing', 0)
        is_rejection_candle = False
        if direction == 'bullish' and (hammer != 0 or engulfing == 100): is_rejection_candle = True
        elif direction == 'bearish' and (shooting_star != 0 or engulfing == -100): is_rejection_candle = True
        if is_in_zone and is_rejection_candle:
            return { 'valid': True, 'direction': direction, 'setup_type': f"amd_{entry_zone['type'].lower()}_rejection_entry_{direction}" }
        return None
    
    async def _execute_trade(self, symbol: str, trade_setup: dict, hypothesis: MarketNarrative):
        try:
            can_trade, reason = self.order_manager.check_position_limit(symbol, self.max_positions)
            if not can_trade:
                logging.info(f"[{symbol}] Trade execution halted: {reason}")
                return
            if self.order_manager.has_opposing_position(symbol, trade_setup['direction']):
                logging.info(f"[{symbol}] Trade execution halted: Opposing position exists.")
                return

            order_params = self._calculate_order_parameters(symbol, trade_setup, hypothesis)
            if not order_params:
                logging.error(f"[{symbol}] Failed to calculate order parameters.")
                return

            trade_result = self.order_manager.place_order(
                symbol=symbol,
                direction=trade_setup['direction'],
                lot_size=order_params['lot_size'],
                stop_loss=order_params['stop_loss'],
                take_profit=order_params['take_profit']
            )

            if trade_result and trade_result.retcode == mt5.TRADE_RETCODE_DONE:
                market_context = {'hypothesis': asdict(hypothesis)}
                def convert_numpy(obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
                    return obj

                self.trade_logger.log_open_trade(
                    ticket_id=trade_result.order, symbol=symbol, direction=trade_setup['direction'],
                    open_price=trade_result.price, stop_loss=order_params['stop_loss'],
                    take_profit=order_params['take_profit'], lot_size=order_params['lot_size'],
                    reason=trade_setup['setup_type'],
                    market_context=json.dumps(market_context, default=convert_numpy, indent=2)
                )
        except Exception as e:
            logging.error(f"Error executing trade for {symbol}: {e}\n{traceback.format_exc()}")
            
    def _calculate_order_parameters(self, symbol: str, trade_setup: dict, hypothesis: MarketNarrative):
        try:
            return self.order_manager.calculate_contextual_order_parameters(
                symbol=symbol, direction=trade_setup['direction'], hypothesis=hypothesis )
        except Exception as e:
            logging.error(f"Error calculating order parameters for {symbol}: {e}\n{traceback.format_exc()}")
            return None
        
async def run_main_loop(client, config: Dict):
    symbols = config['asset_management']['symbols_to_trade']
    timeframes = (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)
    check_interval = config['metatrader_settings']['check_interval_seconds']

    market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
    trade_managers = {symbol: TradeManager(client, market_data_dict[symbol], config) for symbol in symbols}

    while True:
        start_time = time.time()
        tasks = [manager.check_for_signals(symbol) for symbol, manager in trade_managers.items()]
        await asyncio.gather(*tasks)
            
        cycle_duration = time.time() - start_time
        logging.info(f"Main loop cycle complete. Took {cycle_duration:.2f} seconds.")
        
        sleep_time = max(0, check_interval - cycle_duration)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

async def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logging.info("Configuration file 'config.json' loaded successfully.")
    except FileNotFoundError:
        logging.error("FATAL: 'config.json' not found. Please create it next to main.py.")
        return
    except json.JSONDecodeError:
        logging.error("FATAL: 'config.json' is not a valid JSON file. Please check its syntax.")
        return

    client = MetaTrader5Client()
    if not client.is_initialized():
        return

    try:
        await run_main_loop(client, config)
    except asyncio.CancelledError:
        logging.info("Main task was cancelled.")
    finally:
        logging.info("Trading script terminating.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}\n{traceback.format_exc()}")