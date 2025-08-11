from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# External libraries (these must be available in production)
import MetaTrader5 as mt5
from session_manager import SessionManager
from technical_analysis import IndicatorCalculator
from context_engine import ContextEngine, MarketNarrative
from smc import smc

# Logging config
LOG_FILENAME = os.environ.get("TRADEBOT_LOG", "tradebot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILENAME), logging.StreamHandler(sys.stdout)],
)
UTC = pytz.UTC


# -----------------------------
# Utilities
# -----------------------------

def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o))
    except Exception:
        return json.dumps(str(obj))


# -----------------------------
# MetaTrader wrapper
# -----------------------------
class MetaTrader5Client:
    def __init__(self):
        self.initialized = False
        try:
            self.initialized = bool(mt5.initialize())
            if not self.initialized:
                logging.error("MetaTrader5.initialize() failed or returned falsy.")
            else:
                logging.info("MetaTrader5 initialized successfully.")
        except Exception:
            logging.exception("Failed to initialize MetaTrader5")
            self.initialized = False

    def shutdown(self):
        try:
            if self.initialized:
                mt5.shutdown()
                logging.info("MetaTrader5 shutdown completed.")
        except Exception:
            logging.exception("Error during MetaTrader5.shutdown()")

    def is_initialized(self) -> bool:
        return self.initialized

    # Thin wrappers for frequently used calls
    def account_info(self):
        try:
            return mt5.account_info()
        except Exception:
            logging.exception("account_info() failed")
            return None

    def symbol_info(self, symbol: str):
        try:
            return mt5.symbol_info(symbol)
        except Exception:
            logging.exception("symbol_info() failed for %s", symbol)
            return None

    def symbol_info_tick(self, symbol: str):
        try:
            return mt5.symbol_info_tick(symbol)
        except Exception:
            logging.exception("symbol_info_tick() failed for %s", symbol)
            return None

    def positions_get(self, **kwargs):
        try:
            res = mt5.positions_get(**kwargs)
            return res if res is not None else []
        except Exception:
            logging.exception("positions_get() failed")
            return []

    def copy_rates_from_pos(self, symbol: str, timeframe: int, pos: int, count: int):
        try:
            return mt5.copy_rates_from_pos(symbol, timeframe, pos, count)
        except Exception:
            logging.exception("copy_rates_from_pos failed for %s tf=%s", symbol, timeframe)
            return None

    def order_send(self, request: Dict[str, Any]):
        try:
            return mt5.order_send(request)
        except Exception:
            logging.exception("order_send failed for request: %s", request)
            return None


# -----------------------------
# Market Data
# -----------------------------
class MarketData:
    def __init__(self, symbol: str, timeframes: Tuple[int, ...]):
        self.symbol = symbol
        self.timeframes = timeframes
        self.default_candles = {tf: 2000 for tf in timeframes}

    def fetch_data(self, timeframe: int) -> Optional[pd.DataFrame]:
        try:
            num = self.default_candles.get(timeframe, 2000)
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, num)
            if rates is None or len(rates) == 0:
                logging.warning("No rates for %s tf=%s", self.symbol, timeframe)
                return None
            df = pd.DataFrame(rates)
            # keep tick_volume -> volume for clarity
            if 'tick_volume' in df.columns and 'volume' not in df.columns:
                df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            # timezone aware
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
            df.name = self.symbol
            return df
        except Exception:
            logging.exception("fetch_data failed for %s tf=%s", self.symbol, timeframe)
            return None


# -----------------------------
# Trade Logger
# -----------------------------
class TradeLogger:
    def __init__(self, filename: str = 'trade_log.csv'):
        self.filename = filename
        self.columns = [
            'ticket_id', 'symbol', 'direction', 'open_time', 'open_price',
            'stop_loss', 'take_profit', 'lot_size', 'reason', 'market_context',
            'close_time', 'close_price', 'pnl', 'status'
        ]
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=self.columns).to_csv(self.filename, index=False, quoting=csv.QUOTE_MINIMAL)

    def _load_log(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.filename)
        except Exception:
            logging.exception("Failed to load trade log, returning empty DataFrame")
            return pd.DataFrame(columns=self.columns)

    def log_open_trade(self, **kwargs):
        try:
            df = self._load_log()
            ticket = kwargs.get('ticket_id')
            if ticket is not None:
                ticket_str = str(ticket)
                if ('ticket_id' in df.columns) and (df['ticket_id'].astype(str) == ticket_str).any():
                    logging.warning("Ticket %s already logged; skipping.", ticket)
                    return
            kwargs['open_time'] = now_utc_iso()
            kwargs['status'] = 'open'
            if 'market_context' in kwargs:
                kwargs['market_context'] = compact_json(kwargs['market_context'])
            row = pd.DataFrame([kwargs])
            df = pd.concat([df, row], ignore_index=True, sort=False)
            df.to_csv(self.filename, index=False, quoting=csv.QUOTE_MINIMAL)
            logging.info("Logged open trade %s", kwargs.get('ticket_id'))
        except Exception:
            logging.exception("log_open_trade failed")

    def log_close_trade(self, ticket_id: Any, **kwargs):
        try:
            df = self._load_log()
            mask = (df['ticket_id'].astype(str) == str(ticket_id))
            idxs = df.index[mask].tolist()
            if not idxs:
                logging.warning("Ticket %s not found in log", ticket_id)
                return
            i = idxs[0]
            for k, v in kwargs.items():
                if k == 'market_context':
                    df.at[i, k] = compact_json(v)
                else:
                    df.at[i, k] = v
            df.at[i, 'status'] = kwargs.get('status', 'closed')
            df.at[i, 'close_time'] = now_utc_iso()
            df.to_csv(self.filename, index=False, quoting=csv.QUOTE_MINIMAL)
            logging.info("Logged close trade %s", ticket_id)
        except Exception:
            logging.exception("log_close_trade failed for %s", ticket_id)


# -----------------------------
# Order Manager
# -----------------------------
class OrderManager:
    def __init__(self, client: MetaTrader5Client, risk_config: Dict, mt5_config: Dict):
        self.client = client
        self.risk_per_trade_percent = float(risk_config.get('risk_per_trade_percent', 1.0)) / 100.0
        self.min_rr_ratio = float(risk_config.get('minimum_risk_reward_ratio', 1.5))
        self.magic = int(mt5_config.get('magic_number', 123456))

    def _normalize_direction(self, direction: str) -> str:
        d = (direction or '').strip().lower()
        if d in ('bullish', 'buy', 'long'):
            return 'buy'
        if d in ('bearish', 'sell', 'short'):
            return 'sell'
        raise ValueError(f"Unknown direction: {direction}")

    def _floor_to_step(self, value: float, step: float) -> float:
        if step <= 0:
            return round(value, 2)
        steps = math.floor(value / step)
        return max(0.0, steps * step)

    def calculate_contextual_order_parameters(self, symbol: str, direction: str, hypothesis: MarketNarrative) -> Optional[Dict]:
        try:
            account_info = self.client.account_info()
            symbol_info = self.client.symbol_info(symbol)
            symbol_tick = self.client.symbol_info_tick(symbol)
            if not all([account_info, symbol_info, symbol_tick]):
                logging.error("Missing account/symbol info for %s", symbol)
                return None

            # Normalize direction early
            direction_norm = self._normalize_direction(direction)

            point = getattr(symbol_info, 'point', 0.00001)
            digits = getattr(symbol_info, 'digits', 5)
            volume_min = getattr(symbol_info, 'volume_min', 0.01)
            volume_max = getattr(symbol_info, 'volume_max', 100.0)
            volume_step = getattr(symbol_info, 'volume_step', 0.01) or 0.01

            current_price = symbol_tick.ask if direction_norm == 'buy' else symbol_tick.bid
            stop_loss = float(hypothesis.invalidation_level)
            take_profit = float(hypothesis.liquidity_target)

            # ATR buffer (optional)
            rates = self.client.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 20)
            if rates is not None and len(rates) > 1:
                atr_df = pd.DataFrame(rates)
                atr_df['tr'] = pd.concat([
                    (atr_df['high'] - atr_df['low']).abs(),
                    (atr_df['high'] - atr_df['close'].shift()).abs(),
                    (atr_df['low'] - atr_df['close'].shift()).abs()], axis=1).max(axis=1)
                atr = atr_df['tr'].ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                # small buffer towards direction
                stop_loss = stop_loss + (-atr * 0.1 if direction_norm == 'buy' else atr * 0.1)

            # Validate price vs stop
            if (direction_norm == 'buy' and current_price <= stop_loss) or (direction_norm == 'sell' and current_price >= stop_loss):
                logging.warning("Entry price invalid vs stop for %s (price=%s stop=%s)", symbol, current_price, stop_loss)
                return None

            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            if risk <= 0 or (reward / risk) < self.min_rr_ratio:
                logging.warning("Poor R:R for %s (reward/risk=%.2f required=%.2f)", symbol, (reward / risk) if risk>0 else 0.0, self.min_rr_ratio)
                return None

            # Compute lot sizing
            stop_loss_points = risk / point

            # risk amount in currency
            risk_amount = float(account_info.balance) * self.risk_per_trade_percent

            # Value per point for 1 lot: prefer symbol_info.trade_tick_value if present; otherwise approximate
            tick_value = getattr(symbol_info, 'trade_tick_value', None)
            tick_size = getattr(symbol_info, 'trade_tick_size', None)
            if tick_value and tick_size:
                value_per_point = (tick_value / tick_size) * point
            else:
                # fallback conservative estimate: 1 lot ~ 10 USD per pip => per point scale
                value_per_point = 10.0 * (1.0 / point)

            loss_for_one_lot = stop_loss_points * value_per_point
            if loss_for_one_lot <= 0:
                logging.error("loss_for_one_lot <= 0 for %s", symbol)
                return None

            raw_lot = risk_amount / loss_for_one_lot
            lot = self._floor_to_step(raw_lot, volume_step)

            # If raw_lot >= volume_min but floor made it zero, set to volume_min
            if lot < volume_min:
                if raw_lot >= volume_min:
                    lot = volume_min
                else:
                    logging.warning("Computed lot %s below minimum %s for %s", raw_lot, volume_min, symbol)
                    return None

            if lot > volume_max:
                logging.info("Clamping lot %s to max %s for %s", lot, volume_max, symbol)
                lot = volume_max

            lot = self._floor_to_step(lot, volume_step)
            if lot <= 0:
                logging.error("Final lot <= 0 after rounding for %s", symbol)
                return None

            return {
                'lot_size': float(lot),
                'stop_loss': round(stop_loss, digits),
                'take_profit': round(take_profit, digits),
            }

        except Exception:
            logging.exception("Error in calculate_contextual_order_parameters for %s", symbol)
            return None

    def check_position_limit(self, symbol: str, max_positions: int) -> Tuple[bool, str]:
        try:
            positions = self.client.positions_get()
            if positions is None:
                # treat as empty (do not block trading if API returned None unexpectedly)
                positions = []
            if len(positions) >= max_positions:
                return False, f"Max overall positions limit reached ({max_positions})"
            return True, "OK"
        except Exception:
            logging.exception("Error checking position limit")
            return False, "error_checking_positions"

    def has_opposing_position(self, symbol: str, direction: str) -> bool:
        try:
            positions = self.client.positions_get(symbol=symbol)
            if not positions:
                return False
            for pos in positions:
                pos_type = getattr(pos, 'type', None)
                # mt5 defines POSITION_TYPE_BUY=0, POSITION_TYPE_SELL=1 typically
                if isinstance(pos_type, int):
                    if direction == 'buy' and pos_type == mt5.POSITION_TYPE_SELL:
                        return True
                    if direction == 'sell' and pos_type == mt5.POSITION_TYPE_BUY:
                        return True
                else:
                    # fallback: inspect side attribute pricing
                    volume = getattr(pos, 'volume', None)
                    if volume is not None and volume < 0:
                        # negative volume possibly indicates sell; treat conservatively
                        if direction == 'buy':
                            return True
            return False
        except Exception:
            logging.exception("Error checking opposing positions; failing open (allowing trade)")
            return False

    def place_order(self, symbol: str, direction: str, lot_size: float, stop_loss: float, take_profit: float):
        try:
            order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
            tick = self.client.symbol_info_tick(symbol)
            if tick is None:
                logging.error("symbol_info_tick returned None for %s", symbol)
                return None
            price = tick.ask if direction == 'buy' else tick.bid

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': float(lot_size),
                'type': order_type,
                'price': float(price),
                'sl': float(stop_loss) if stop_loss is not None else 0.0,
                'tp': float(take_profit) if take_profit is not None else 0.0,
                'deviation': 20,
                'magic': self.magic,
                'comment': 'Context-Driven Trade',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            result = self.client.order_send(request)
            if result is None:
                logging.error("order_send returned None for %s", symbol)
                return None
            if getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                logging.error("Order send failed for %s: retcode=%s comment=%s", symbol, getattr(result, 'retcode', None), getattr(result, 'comment', None))
            return result
        except Exception:
            logging.exception("place_order failed for %s", symbol)
            return None


# -----------------------------
# Trade Manager (AMD specialist)
# -----------------------------
class TradeManager:
    def __init__(self, client: MetaTrader5Client, market_data: MarketData, config: Dict, session_manager: SessionManager):
        self.client = client
        self.market_data = market_data
        self.session_manager = session_manager
        self.timeframes = market_data.timeframes

        self.trade_logger = TradeLogger(filename=config['logging']['trade_log_file'])
        self.indicator_calc = IndicatorCalculator()
        self.context_engine = ContextEngine(strategy_config=config['amd_strategy_parameters'])
        self.order_manager = OrderManager(client, config['risk_management'], config['metatrader_settings'])

        self.tf_higher = max(self.timeframes)
        self.tf_medium = sorted(self.timeframes)[1] if len(self.timeframes) >= 3 else self.timeframes[0]
        self.tf_lower = min(self.timeframes)
        self.max_positions = config['asset_management']['max_overall_positions']
        self.active_setup: Optional[Dict] = None

    def _prepare_data(self, symbol: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        try:
            data = {tf: self.market_data.fetch_data(tf) for tf in self.timeframes}
            if any(d is None or d.empty for d in data.values()):
                return None, None

            indicators = {tf: self.indicator_calc.calculate_indicators(d) for tf, d in data.items()}
            if any(i is None or i.empty for i in indicators.values()):
                return None, None

            # align lengths
            min_len = min(len(df) for df in indicators.values())
            data = {tf: d.iloc[-min_len:] for tf, d in data.items()}
            indicators = {tf: i.iloc[-min_len:] for tf, i in indicators.items()}
            return data, indicators
        except Exception:
            logging.exception("_prepare_data failed for %s", self.market_data.symbol)
            return None, None

    async def check_for_signals(self, symbol: str):
        try:
            if self.active_setup:
                await self.monitor_existing_setup(symbol)
                return 
            
            await self.find_new_setup(symbol)

        except Exception:
            logging.exception(f"check_for_signals failed for {symbol}")
            
    async def monitor_existing_setup(self, symbol: str):
        """Monitors and acts on a pre-existing trading hypothesis with detailed logging."""
        now = datetime.now(UTC)
        setup = self.active_setup
        hypothesis: MarketNarrative = setup["hypothesis"]
        entry_zone = setup["entry_zone"]
        status = setup["status"]

        # --- A. Check for Expiry or Invalidation first ---
        if now >= setup["expires_at"]:
            logging.info(f"[{symbol}] STATUS: EXPIRED. Setup timed out at {setup['expires_at']}. Discarding.")
            self.active_setup = None
            return

        data, indicators = self._prepare_data(symbol)
        if not data or not indicators:
            logging.warning(f"[{symbol}] Could not fetch data to monitor active setup. Will retry next cycle.")
            return
        
        last_candle = data[self.tf_lower].iloc[-1]
        current_price = last_candle['close']
        
        is_invalidated = (hypothesis.daily_bias == 'bullish' and current_price < hypothesis.invalidation_level) or \
                         (hypothesis.daily_bias == 'bearish' and current_price > hypothesis.invalidation_level)
        
        if is_invalidated:
            logging.warning(f"[{symbol}] STATUS: INVALIDATED. Price {current_price:.5f} broke invalidation level {hypothesis.invalidation_level:.5f}. Discarding.")
            self.active_setup = None
            return

        # --- B. Entry Logic State Machine with Detailed Logging ---
        has_entered_zone = (hypothesis.daily_bias == 'bullish' and last_candle['low'] <= entry_zone['top']) or \
                           (hypothesis.daily_bias == 'bearish' and last_candle['high'] >= entry_zone['bottom'])

        # STATE 1: Awaiting entry into the zone
        if status == "awaiting_zone_entry":
            if has_entered_zone:
                logging.info(f"[{symbol}] STATUS: ZONE ENTERED. Price touched {entry_zone['type']} at {last_candle['high'] if hypothesis.daily_bias == 'bearish' else last_candle['low']:.5f}. Now monitoring for rejection candle.")
                setup["status"] = "zone_entered"
                rejection_timeout_minutes = self.context_engine.strategy_config.get("rejection_timeout_minutes", 60)
                setup["rejection_expires_at"] = now + pd.Timedelta(minutes=rejection_timeout_minutes)
            else:
                # This is the new, continuous log message
                price_needed = entry_zone['bottom'] if hypothesis.daily_bias == 'bearish' else entry_zone['top']
                logging.info(f"[{symbol}] STATUS: MONITORING. Waiting for price to pull back to zone. "
                             f"Current High: {last_candle['high']:.5f}, "
                             f"Current Low: {last_candle['low']:.5f}, "
                             f"Zone Entry: ~{price_needed:.5f}")
            return # Wait for next cycle

        # STATE 2: Waiting for a rejection candle inside the zone
        if status == "zone_entered":
            if "rejection_expires_at" in setup and now >= setup["rejection_expires_at"]:
                logging.info(f"[{symbol}] STATUS: REJECTION TIMEOUT. No confirmation candle formed in time. Discarding.")
                self.active_setup = None
                return

            trade_trigger = self._confirm_entry_trigger(
                ltf_df=data[self.tf_lower],
                ltf_indicators=indicators[self.tf_lower],
                direction=hypothesis.daily_bias.lower(),
                entry_zone=entry_zone
            )
            
            if trade_trigger and trade_trigger.get('valid'):
                logging.info(f"[{symbol}] STATUS: ENTRY CONFIRMED. High-probability rejection signal found. Executing trade.")
                setup["status"] = "entry_confirmed"
                await self._execute_trade(symbol, trade_trigger, hypothesis)
                self.active_setup = None # Clear after execution
            else:
                # The other new, continuous log message
                logging.info(f"[{symbol}] STATUS: IN ZONE. Price is inside the FVG. Waiting for a valid rejection candle.")


    async def find_new_setup(self, symbol: str):
        """Looks for a new, valid trading hypothesis."""
        data, indicators = self._prepare_data(symbol)
        if not data or not indicators:
            return

        hypothesis: MarketNarrative = self.context_engine.process_data(data[self.tf_lower], indicators[self.tf_higher])

        # We only care about high-clarity hypotheses that have passed the manipulation stage
        if hypothesis.clarity_score != 'High' or 'MANIPULATION_CONFIRMED' not in hypothesis.session_profile:
            return
            
        # We need to find the entry zone to monitor
        trade_setup = self._evaluate_trade_setup(data, indicators, hypothesis)
        if not trade_setup or not trade_setup.get("entry_zone"):
            return

        # --- If we get here, we have a valid new setup to start monitoring! ---
        expiry_minutes = self.context_engine.strategy_config.get("setup_expiry_minutes", 120)
        self.active_setup = {
            "hypothesis": hypothesis,
            "status": "awaiting_zone_entry", 
            "entry_zone": trade_setup["entry_zone"],
            "expires_at": datetime.now(UTC) + pd.Timedelta(minutes=expiry_minutes),
            "symbol": symbol
        }
        logging.info(f"[{symbol}] NEW high-clarity setup found. Bias: {hypothesis.daily_bias}. Awaiting entry into zone: {trade_setup['entry_zone']['type']}.")
    
    def _evaluate_trade_setup(self, data: Dict, indicators: Dict, hypothesis: MarketNarrative) -> Optional[Dict]:
        """
        Evaluates if a valid structure with a monitorable entry zone exists.
        Does NOT look for the final entry trigger candle.
        Returns a dictionary with the entry_zone if a valid setup is found.
        """
        SWING_LENGTH = self.context_engine.ltf_swing_length
        NUM_ZONES_TO_CHECK = self.context_engine.zones_to_check

        ltf_df = data[self.tf_lower].copy()
        direction_raw = hypothesis.daily_bias.lower()
        symbol = self.market_data.symbol

        manipulation_candle = self._find_manipulation_extreme(ltf_df, direction_raw, hypothesis)
        if manipulation_candle is None:
            return None

        mss_level = self._confirm_market_structure_shift(ltf_df, direction_raw, manipulation_candle, SWING_LENGTH)
        if mss_level is None:
            return None

        # Its only job is to find the zone.
        entry_zone = self._find_entry_zone(ltf_df, direction_raw, SWING_LENGTH, NUM_ZONES_TO_CHECK, manipulation_candle.name)
        if not entry_zone:
            return None

        # If we found a valid structure with a zone, return it for monitoring.
        return {"entry_zone": entry_zone}

    def _find_manipulation_extreme(self, ltf_df: pd.DataFrame, direction: str, hypothesis: MarketNarrative) -> Optional[pd.Series]:
        symbol = self.market_data.symbol
        try:
            current_day = self.context_engine.current_day
            london_start_time = self.context_engine.SESSION_WINDOWS['london']['start']
            london_session_start_dt = datetime.combine(current_day, london_start_time).replace(tzinfo=UTC)
            todays = ltf_df[ltf_df.index >= london_session_start_dt]
            if todays.empty:
                logging.warning("[%s] No data for today's London session.", symbol)
                return None

            extreme_level = hypothesis.invalidation_level
            if direction == 'bullish':
                found = todays[todays['low'] == extreme_level]
            else:
                found = todays[todays['high'] == extreme_level]

            if found.empty:
                # tolerance using symbol point
                symbol_info = self.client.symbol_info(symbol)
                tol = getattr(symbol_info, 'point', 0.00001) * 0.5
                if direction == 'bullish':
                    found = todays[(todays['low'] - extreme_level).abs() < tol]
                else:
                    found = todays[(todays['high'] - extreme_level).abs() < tol]

            if found.empty:
                logging.warning("[%s] Could not find manipulation extreme for level %s", symbol, extreme_level)
                return None

            cand = found.iloc[0]
            logging.info("[%s] Found manipulation extreme at %s", symbol, cand.name)
            return cand
        except Exception:
            logging.exception("_find_manipulation_extreme failed for %s", symbol)
            return None

    def _confirm_market_structure_shift(self, ltf_df, direction, manipulation_candle, swing_length):
        symbol = self.market_data.symbol
        try:
            full_swings = smc.swing_highs_lows(ltf_df, swing_length=swing_length)

            # Keep same-length index for simple comparisons (we still trust smc output shape)
            full_swings.index = ltf_df.index
            full_swings.dropna(inplace=True)

            # Resolve manipulation time
            if isinstance(manipulation_candle, pd.Series):
                manip_time = manipulation_candle.name
            elif hasattr(manipulation_candle, "name"):
                manip_time = manipulation_candle.name
            else:
                manip_time = manipulation_candle

            if isinstance(manip_time, (np.ndarray, list, tuple)):
                manip_time = manip_time[0]
            manip_time = pd.to_datetime(manip_time)

            swings_before = full_swings[full_swings.index < manip_time]

            swing_type_to_break = 1 if direction == 'bullish' else -1
            relevant = swings_before[swings_before['HighLow'] == swing_type_to_break]
            if relevant.empty:
                logging.info("[%s] No prior swing to define MSS.", symbol)
                return None

            mss_level = relevant['Level'].iloc[-1]

            # --- Robust MSS break check: look for any candle after manipulation that breached the level ---
            post_manip = ltf_df[ltf_df.index > manip_time]
            if post_manip.empty:
                logging.info("[%s] Waiting for MSS break at %s (no candles after manipulation yet).", symbol, mss_level)
                return None

            # For bullish MSS we accept if any high or close exceeds the MSS level.
            if direction == 'bullish':
                broke = (post_manip['high'] > mss_level).any() or (post_manip['close'] > mss_level).any()
            else:
                broke = (post_manip['low'] < mss_level).any() or (post_manip['close'] < mss_level).any()

            if not broke:
                # more informative logging
                last_close = post_manip['close'].iloc[-1]
                last_high = post_manip['high'].iloc[-1]
                last_low = post_manip['low'].iloc[-1]
                logging.info(
                    "[%s] Waiting for MSS break at %s (last_close=%s last_high=%s last_low=%s).",
                    symbol, mss_level, last_close, last_high, last_low
                )
                return None

            logging.info("[%s] MSS break confirmed at level %s", symbol, float(mss_level))
            return mss_level
        except Exception:
            logging.exception("_confirm_market_structure_shift failed for %s", symbol)
            return None

    def _find_entry_zone(self, ltf_df: pd.DataFrame, direction: str, swing_length: int, num_zones: int, after_time: pd.Timestamp) -> Optional[Dict]:
        symbol = self.market_data.symbol
        fvg = self._find_fvg_entry(ltf_df, direction, num_zones, after_time)
        if fvg:
            return fvg
        return self._find_ob_entry(ltf_df, direction, swing_length, num_zones, after_time)

    def _find_fvg_entry(self, ltf_df: pd.DataFrame, direction: str, num_zones: int, after_time: pd.Timestamp) -> Optional[Dict]:
        symbol = self.market_data.symbol
        try:
            relevant_df = ltf_df[ltf_df.index > after_time]
            if relevant_df.empty:
                # This is a normal condition, so no log is needed unless debugging.
                return None

            # 1. Find all FVGs in the relevant timeframe
            all_fvgs = smc.fvg(relevant_df)
            
            # 2. Filter for only the ones that have not been filled/mitigated yet
            unmitigated_fvgs = all_fvgs[all_fvgs['MitigatedIndex'].isna()].dropna(subset=['FVG'])
            
            if unmitigated_fvgs.empty:
                logging.info(f"[{symbol}] FVG Search: No unmitigated FVGs found after manipulation event.")
                return None

            # 3. Determine the type of FVG we need based on our bias
            fvg_type_needed = 1 if direction == 'bullish' else -1
            
            # 4. Filter for FVGs that match our directional bias
            directional_fvgs = unmitigated_fvgs[unmitigated_fvgs['FVG'] == fvg_type_needed]
            
            if directional_fvgs.empty:
                logging.info(f"[{symbol}] FVG Search: Found {len(unmitigated_fvgs)} unmitigated FVG(s), but none match the required {direction} bias.")
                return None

            # 5. We want the most recent FVG that matches our criteria.
            selected_fvg_row = directional_fvgs.tail(1).iloc[0]
            
            fvg_details = {
                'type': 'FVG', 
                'top': selected_fvg_row['Top'], 
                'bottom': selected_fvg_row['Bottom']
            }

            # 6. Log the successful find with all the critical details. THIS IS THE KEY LOG.
            logging.info(
                f"[{symbol}] FVG Search SUCCESS: Identified valid entry zone. "
                f"Top={fvg_details['top']:.5f}, Bottom={fvg_details['bottom']:.5f}"
            )

            return fvg_details
        except Exception:
            logging.exception(f"_find_fvg_entry failed for {symbol}")
            return None

    def _find_ob_entry(self, ltf_df: pd.DataFrame, direction: str, swing_length: int, num_zones: int, after_time: pd.Timestamp) -> Optional[Dict]:
        symbol = self.market_data.symbol
        try:
            relevant_df = ltf_df[ltf_df.index > after_time]
            if relevant_df.empty:
                return None

            # 1. We need swing points to define order blocks
            swings = smc.swing_highs_lows(relevant_df, swing_length=swing_length)
            swings.index = relevant_df.index # Align index for the smc.ob function

            # 2. Find all order blocks in the relevant timeframe
            all_obs = smc.ob(relevant_df, swings)
            
            # 3. Filter for valid and unmitigated OBs
            valid_obs = all_obs.dropna(subset=['OB'])
            unmitigated_obs = valid_obs[valid_obs['MitigatedIndex'].isna()]
            
            if unmitigated_obs.empty:
                logging.info(f"[{symbol}] OB Search: No unmitigated Order Blocks found after manipulation event.")
                return None

            # 4. Filter for OBs that match our directional bias
            ob_type_needed = 1 if direction == 'bullish' else -1
            directional_obs = unmitigated_obs[unmitigated_obs['OB'] == ob_type_needed]

            if directional_obs.empty:
                logging.info(f"[{symbol}] OB Search: Found {len(unmitigated_obs)} unmitigated OB(s), but none match the required {direction} bias.")
                return None

            # 5. Select the most recent valid OB
            selected_ob_row = directional_obs.tail(1).iloc[0]
            
            ob_details = {
                'type': 'OB', 
                'top': selected_ob_row['Top'], 
                'bottom': selected_ob_row['Bottom']
            }

            # 6. Log the successful find with critical details
            logging.info(
                f"[{symbol}] OB Search SUCCESS: Identified valid entry zone. "
                f"Top={ob_details['top']:.5f}, Bottom={ob_details['bottom']:.5f}"
            )

            return ob_details
        except Exception:
            logging.exception(f"_find_ob_entry failed for {symbol}")
            return None

    def _confirm_entry_trigger(self, ltf_df: pd.DataFrame, ltf_indicators: pd.DataFrame, direction: str, entry_zone: Dict) -> Optional[Dict]:
        symbol = self.market_data.symbol
        try:
            if ltf_df.empty or ltf_indicators.empty:
                return None

            last_candle = ltf_df.iloc[-1]
            last_ind = ltf_indicators.iloc[-1]

            # More robust check: Has the candle's wick tapped into the zone?
            in_zone = False
            if direction == 'bullish':
                # Price dipped into the zone from above
                if last_candle['low'] <= entry_zone['top']:
                    in_zone = True
            elif direction == 'bearish':
                # Price rallied into the zone from below
                if last_candle['high'] >= entry_zone['bottom']:
                    in_zone = True

            if not in_zone:
                return None

            logging.info("[%s] Price has entered the %s zone [%.5f - %.5f]", symbol, entry_zone['type'], entry_zone['bottom'], entry_zone['top'])

            hammer = last_ind.get('cdl_hammer', 0)
            shooting_star = last_ind.get('cdl_shooting_star', 0)
            engulfing = last_ind.get('cdl_engulfing', 0)

            is_rejection = False
            if direction == 'bullish' and (hammer == 100 or engulfing == 100):
                is_rejection = True
                logging.info("[%s] Bullish rejection candle (Hammer or Engulfing) confirmed in zone.", symbol)
            elif direction == 'bearish' and (shooting_star == 100 or engulfing == -100):
                is_rejection = True
                logging.info("[%s] Bearish rejection candle (Shooting Star or Engulfing) confirmed in zone.", symbol)

            if is_rejection:
                return {
                    'valid': True,
                    'direction': direction,
                    'setup_type': f"amd_{entry_zone['type'].lower()}_rejection_entry_{direction}"
                }
            return None
        except Exception:
            logging.exception("_confirm_entry_trigger failed for %s", symbol)
            return None

    async def _execute_trade(self, symbol: str, trade_setup: Dict, hypothesis: MarketNarrative):
        try:
            can_trade, reason = self.order_manager.check_position_limit(symbol, self.max_positions)
            if not can_trade:
                logging.info("[%s] Trade halted: %s", symbol, reason)
                return

            direction_norm = 'buy' if trade_setup['direction'] == 'bullish' else 'sell' if trade_setup['direction'] == 'bearish' else trade_setup['direction']
            if self.order_manager.has_opposing_position(symbol, direction_norm):
                logging.info("[%s] Opposing position exists; skipping", symbol)
                return

            order_params = self._calculate_order_parameters(symbol, trade_setup, hypothesis)
            if not order_params:
                logging.error("[%s] Order params calculation failed; skipping", symbol)
                return

            trade_result = self.order_manager.place_order(symbol, direction_norm, order_params['lot_size'], order_params['stop_loss'], order_params['take_profit'])

            if trade_result is None:
                logging.error("[%s] order_send returned None or error", symbol)
                return

            if getattr(trade_result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                market_context = {'hypothesis': asdict(hypothesis)}
                self.trade_logger.log_open_trade(
                    ticket_id=getattr(trade_result, 'order', None) or getattr(trade_result, 'ticket', None),
                    symbol=symbol,
                    direction=direction_norm,
                    open_price=getattr(trade_result, 'price', None),
                    stop_loss=order_params['stop_loss'],
                    take_profit=order_params['take_profit'],
                    lot_size=order_params['lot_size'],
                    reason=trade_setup.get('setup_type'),
                    market_context=market_context
                )
                logging.info("[%s] Trade executed; ticket=%s", symbol, getattr(trade_result, 'order', None))
            else:
                logging.error("[%s] Trade failed: retcode=%s comment=%s", symbol, getattr(trade_result, 'retcode', None), getattr(trade_result, 'comment', None))

        except Exception:
            logging.exception("_execute_trade failed for %s", symbol)

    def _calculate_order_parameters(self, symbol: str, trade_setup: Dict, hypothesis: MarketNarrative) -> Optional[Dict]:
        try:
            return self.order_manager.calculate_contextual_order_parameters(symbol, trade_setup['direction'], hypothesis)
        except Exception:
            logging.exception("_calculate_order_parameters failed for %s", symbol)
            return None


# -----------------------------
# Main loop
# -----------------------------
async def run_main_loop(client: MetaTrader5Client, config: Dict):
    symbols = config['asset_management']['symbols_to_trade']
    timeframes = (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1)
    check_interval = config['metatrader_settings']['check_interval_seconds']

    session_manager = SessionManager(config)
    market_data_dict = {symbol: MarketData(symbol, timeframes) for symbol in symbols}
    trade_managers = {
        symbol: TradeManager(client, market_data_dict[symbol], config, session_manager)
        for symbol in symbols
    }

    while True:
        start_time = time.time()
        tasks = [manager.check_for_signals(symbol) for symbol, manager in trade_managers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception("Symbol task failed: %s", r)

        cycle_duration = time.time() - start_time
        logging.info("Main loop cycle complete. Took %.2f seconds.", cycle_duration)

        sleep_time = max(0, check_interval - cycle_duration)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)


async def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        logging.info("Loaded config.json")
    except FileNotFoundError:
        logging.error("config.json not found; aborting")
        return

    client = MetaTrader5Client()
    if not client.is_initialized():
        logging.error("MT5 failed to initialize; aborting")
        return

    try:
        await run_main_loop(client, config)
    except asyncio.CancelledError:
        logging.info("Main task cancelled")
    except Exception:
        logging.exception("Unhandled exception in main loop")
    finally:
        client.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.exception("Fatal error in __main__")
