#!/usr/bin/env python3
"""
Forex Trading Bot - Core
"""

import asyncio
import logging
import json
import argparse
import sys
from datetime import datetime, timezone, timedelta

from pathlib import Path
from typing import Optional, Dict

from utils import resolve_pip_size
from strategy import PurePriceActionStrategy, TradingSignal
from state_manager import StateManager, save_strategy_state, load_strategy_state
from logging_utils import configure_logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path: str = 'config.json', args: Optional[argparse.Namespace] = None):
        self.config_path = Path(config_path)
        self.args = args
        self.load_config()

    def load_config(self):
        try:
            data = json.loads(self.config_path.read_text(encoding='utf-8'))
            trading = data.get('trading_settings', {})
            self.main_loop_interval = trading.get('main_loop_interval_seconds', 5)
            self.lookback_period = trading.get('lookback_period', 20)
            self.swing_window = trading.get('swing_window', 5)
            self.breakout_threshold = trading.get('breakout_threshold_pips', 7)
            self.breakout_threshold_atr_mult = trading.get('breakout_threshold_atr_mult', None)
            self.breakout_window_bars = trading.get('breakout_window_bars', 1)
            self.max_extension_pips = trading.get('max_extension_pips', None)
            self.max_extension_atr_mult = trading.get('max_extension_atr_mult', None)
            self.min_stop_loss_pips = trading.get('min_stop_loss_pips', 20)
            self.stop_loss_buffer_pips = trading.get('stop_loss_buffer_pips', 15)
            # ATR for dynamic SL buffer
            self.atr_period = trading.get('atr_period', 14)
            self.atr_sl_k = trading.get('atr_sl_k', 0.6)
            self.min_sl_buffer_pips = trading.get('min_sl_buffer_pips', 10)
            self.max_sl_pips = trading.get('max_sl_pips', None)
            self.spread_guard_pips_default = trading.get('spread_guard_pips', None)
            self.sr_lookback_period = trading.get('sr_lookback_period', 80)
            self.sr_proximity_pips = trading.get('sr_proximity_pips', 10)
            self.tp_buffer_pips = trading.get('tp_buffer_pips', 2)
            # Trend filter settings
            self.use_trend_filter = trading.get('use_trend_filter', True)
            self.trend_ema_period = trading.get('trend_ema_period', 200)
            self.use_ema_slope_filter = trading.get('use_ema_slope_filter', True)
            self.ema_slope_period = trading.get('ema_slope_period', 20)
            self.min_ema_slope_pips_per_bar = trading.get('min_ema_slope_pips_per_bar', 0.1)

            risk = data.get('risk_management', {})
            self.risk_per_trade = risk.get('risk_per_trade', 0.01)
            self.fixed_lot_size = risk.get('fixed_lot_size', None)
            self.max_drawdown = risk.get('max_drawdown_percentage', 0.05)
            self.risk_reward_ratio = risk.get('risk_reward_ratio', 2.0)
            self.min_rr = risk.get('min_rr', 1.0)
            self.use_equity = risk.get('use_equity', True)

            # Symbols
            self.symbols = []
            for s in data.get('symbols', []) or []:
                # Prefer explicit entry/trend timeframes; fall back to legacy timeframes list
                legacy_tfs = s.get('timeframes') or []
                entry_tf = s.get('entry_timeframe') or (legacy_tfs[0] if legacy_tfs else 'M15')
                trend_tf = s.get('trend_timeframe') or (legacy_tfs[1] if len(legacy_tfs) > 1 else 'H1')
                entry = {
                    'name': s.get('name'),
                    'entry_timeframe': entry_tf,
                    'trend_timeframe': trend_tf,
                    # Per-symbol overrides (only essential ones)
                    'pip_unit': s.get('pip_unit'),
                    'min_stop_loss_pips': s.get('min_stop_loss_pips'),
                    'stop_loss_buffer_pips': s.get('stop_loss_buffer_pips'),
                    'breakout_threshold_pips': s.get('breakout_threshold_pips'),
                    'breakout_threshold_atr_mult': s.get('breakout_threshold_atr_mult'),
                    'risk_reward_ratio': s.get('risk_reward_ratio'),
                    'spread_guard_pips': s.get('spread_guard_pips'),
                    'max_sl_pips': s.get('max_sl_pips'),
                    'breakout_window_bars': s.get('breakout_window_bars'),
                    'max_extension_pips': s.get('max_extension_pips'),
                    'max_extension_atr_mult': s.get('max_extension_atr_mult'),
                    'sr_lookback_period': s.get('sr_lookback_period'),
                    'sr_proximity_pips': s.get('sr_proximity_pips'),
                    'tp_buffer_pips': s.get('tp_buffer_pips'),
                }
                # Remove None values
                entry = {k: v for k, v in entry.items() if v is not None}
                entry.setdefault('name', s.get('name'))
                entry.setdefault('entry_timeframe', 'M15')
                entry.setdefault('trend_timeframe', 'H1')
                self.symbols.append(entry)

            # CLI overrides
            if self.args:
                if self.args.risk_per_trade is not None:
                    self.risk_per_trade = float(self.args.risk_per_trade)
                if self.args.symbol:
                    self.symbols = [{'name': self.args.symbol, 'entry_timeframe': 'M15', 'trend_timeframe': 'H1'}]
                if self.args.timeframe:
                    for sym in self.symbols:
                        sym['entry_timeframe'] = self.args.timeframe

            logger.info(f"Config loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.set_defaults()

    def set_defaults(self):
        self.main_loop_interval = 5
        self.lookback_period = 20
        self.swing_window = 5
        self.breakout_threshold = 7
        self.breakout_threshold_atr_mult = None
        self.breakout_window_bars = 1
        self.max_extension_pips = None
        self.max_extension_atr_mult = None
        self.min_stop_loss_pips = 20
        self.stop_loss_buffer_pips = 15
        self.atr_period = 14
        self.atr_sl_k = 0.6
        self.min_sl_buffer_pips = 10
        self.max_sl_pips = None
        self.spread_guard_pips_default = None
        self.sr_lookback_period = 80
        self.sr_proximity_pips = 10
        self.tp_buffer_pips = 2
        self.use_trend_filter = True
        self.trend_ema_period = 200
        self.use_ema_slope_filter = True
        self.ema_slope_period = 20
        self.min_ema_slope_pips_per_bar = 0.1
        self.risk_per_trade = 0.01
        self.fixed_lot_size = None
        self.max_drawdown = 0.05
        self.risk_reward_ratio = 2.0
        self.min_rr = 1.0
        self.symbols = [{'name': 'EURUSD', 'entry_timeframe': 'M15', 'trend_timeframe': 'H1'}]

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.trade_logger = None
        self.trailing_manager = None

        # Persist state files under a dedicated folder
        self.state_manager = StateManager('symbol_state')
        self.running = False
        self.symbol_fillings = {}

    def _is_rollover_period(self) -> bool:
        """Check if current time is during daily Forex rollover (avoid trading).

        Rollover occurs around 17:00 New York time (21:00-22:00 UTC depending on DST).
        During this ~70 minute window, liquidity disappears and spreads explode.
        """
        now_utc = datetime.now(timezone.utc)
        hour, minute = now_utc.hour, now_utc.minute
        # Rollover window: 21:55 - 23:05 UTC (covers DST variations)
        if hour == 21 and minute >= 55:
            return True
        if hour == 22:
            return True
        if hour == 23 and minute <= 5:
            return True
        return False

    async def initialize(self):
        try:
            from mt5_client import MetaTrader5Client
            from market_data import MarketData
            from risk_manager import RiskManager
            from trade_logger import TradeLogger
            from trailing_stop import create_live_trailing_manager

            self.mt5_client = MetaTrader5Client()
            if not self.mt5_client.initialized:
                raise RuntimeError('Failed to initialize MT5 connection')

            self.market_data = MarketData(self.mt5_client, self.config)
            self.strategy = PurePriceActionStrategy(self.config, self.mt5_client)
            self.risk_manager = RiskManager(self.config, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')
            self.trailing_manager = create_live_trailing_manager(self.mt5_client)

            # Load strategy state (duplicate prevention)
            try:
                if load_strategy_state(self.state_manager, self.strategy):
                    logger.info("Strategy: Duplicate prevention state restored")
            except Exception as e:
                logger.warning(f"Strategy: Failed to restore state: {e}")

            # Strategy snapshot
            logger.info(
                "Strategy init: breakout_thr=%s rr=%.2f lookback=%s swing=%s",
                getattr(self.config, 'breakout_threshold', 7),
                float(getattr(self.config, 'risk_reward_ratio', 2.0)),
                getattr(self.config, 'lookback_period', 20),
                getattr(self.config, 'swing_window', 5),
            )

            # Preflight introspection for configured symbols
            for sc in (self.config.symbols or []):
                sym = sc.get('name')
                try:
                    info = self.mt5_client.get_symbol_info(sym)
                    tick = self.mt5_client.get_symbol_info_tick(sym)
                    if not info:
                        logger.warning(f"{sym}: no symbol_info")
                        continue
                    filling = self.mt5_client.preferred_filling_mode(sym)
                    self.symbol_fillings[sym] = filling
                    # Log concise spec snapshot
                    logger.debug(
                        f"{sym}: digits={getattr(info,'digits',None)} point={getattr(info,'point',None)} "
                        f"tick_size={getattr(info,'trade_tick_size',None)} tick_value={getattr(info,'trade_tick_value',None)} "
                        f"stops_level={getattr(info,'trade_stops_level',None)} vol(min/step/max)="
                        f"{getattr(info,'volume_min',None)}/{getattr(info,'volume_step',None)}/{getattr(info,'volume_max',None)} "
                        f"filling_mode={getattr(info,'filling_mode',None)} pref_fill={filling}"
                    )
                except Exception as e:
                    logger.warning(f"{sym}: preflight error: {e}")

            # Reconcile persisted trades with live MT5 positions
            try:
                self._reconcile_state()
            except Exception as e:
                logger.warning(f"State reconcile failed: {e}")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def _classify_close_deal(self, closing_deal):
        mt5 = self.mt5_client.mt5
        reason_code = getattr(closing_deal, 'reason', None)
        status = 'CLOSED'
        reason_label = None
        try:
            if reason_code is not None:
                if reason_code == getattr(mt5, 'DEAL_REASON_SL', -1):
                    status, reason_label = 'CLOSED_SL', 'SL'
                elif reason_code == getattr(mt5, 'DEAL_REASON_TP', -1):
                    status, reason_label = 'CLOSED_TP', 'TP'
                elif reason_code in (
                    getattr(mt5, 'DEAL_REASON_SO', None),
                    getattr(mt5, 'DEAL_REASON_STOPOUT', None),
                ):
                    status, reason_label = 'CLOSED_STOPOUT', 'STOPOUT'
                elif reason_code in (
                    getattr(mt5, 'DEAL_REASON_CLIENT', None),
                    getattr(mt5, 'DEAL_REASON_MOBILE', None),
                    getattr(mt5, 'DEAL_REASON_WEB', None),
                ):
                    status, reason_label = 'CLOSED_MANUAL', 'MANUAL'
                elif reason_code == getattr(mt5, 'DEAL_REASON_EXPERT', None):
                    status, reason_label = 'CLOSED_STRATEGY', 'EXPERT'
                elif reason_code == getattr(mt5, 'DEAL_REASON_DEALER', None):
                    status, reason_label = 'CLOSED_DEALER', 'DEALER'
        except Exception:
            pass

        if reason_label is None:
            try:
                ctext = str(getattr(closing_deal, 'comment', '') or '').lower()
                if 'sl' in ctext:
                    status, reason_label = 'CLOSED_SL', 'COMMENT_SL'
                elif 'tp' in ctext:
                    status, reason_label = 'CLOSED_TP', 'COMMENT_TP'
                else:
                    status, reason_label = 'CLOSED_MANUAL', 'COMMENT'
            except Exception:
                status, reason_label = 'CLOSED', None

        try:
            close_time = datetime.fromtimestamp(
                getattr(closing_deal, 'time', 0), tz=timezone.utc
            ) if getattr(closing_deal, 'time', None) else None
        except Exception:
            close_time = None

        return status, reason_label, reason_code, close_time

    def _reconcile_state(self):
        if not self.mt5_client or not self.mt5_client.is_connected():
            logger.warning("State reconcile skipped: MT5 not connected")
            return

        positions = self.mt5_client.get_all_positions()
        if positions is None:
            logger.warning("State reconcile skipped: no positions returned")
            return

        open_positions = {getattr(p, 'ticket', None): p for p in positions if getattr(p, 'ticket', None) is not None}
        open_tickets = set(open_positions.keys())

        open_trades = [t for t in self.trade_logger.trades if t.get('status') == 'OPEN']
        trades_by_pos = {t.get('position_id'): t for t in open_trades if t.get('position_id') is not None}

        updated = False

        # Ensure every MT5 position has a corresponding OPEN trade entry
        for pid, pos in open_positions.items():
            trade = trades_by_pos.get(pid)
            if trade is None:
                entry_time = datetime.fromtimestamp(getattr(pos, 'time', 0), tz=timezone.utc) if getattr(pos, 'time', None) else datetime.now(timezone.utc)
                trade = {
                    'timestamp': entry_time,
                    'symbol': getattr(pos, 'symbol', None),
                    'order_type': 'BUY' if int(getattr(pos, 'type', 0)) == 0 else 'SELL',
                    'entry_price': float(getattr(pos, 'price_open', 0.0) or 0.0),
                    'requested_price': None,
                    'signal_price': None,
                    'volume': float(getattr(pos, 'volume', 0.0) or 0.0),
                    'stop_loss': float(getattr(pos, 'sl', 0.0) or 0.0),
                    'take_profit': float(getattr(pos, 'tp', 0.0) or 0.0),
                    'order_ticket': pid,
                    'deal_ticket': None,
                    'position_id': pid,
                    'reason': 'RECONCILED',
                    'signal_time': None,
                    'breakout_level': None,
                    'status': 'OPEN'
                }
                self.trade_logger.trades.append(trade)
                updated = True
            else:
                # Update with MT5 truth
                trade['symbol'] = getattr(pos, 'symbol', trade.get('symbol'))
                trade['order_type'] = 'BUY' if int(getattr(pos, 'type', 0)) == 0 else 'SELL'
                trade['entry_price'] = float(getattr(pos, 'price_open', trade.get('entry_price') or 0.0) or 0.0)
                trade['volume'] = float(getattr(pos, 'volume', trade.get('volume') or 0.0) or 0.0)
                if getattr(pos, 'sl', None) is not None:
                    trade['stop_loss'] = float(pos.sl)
                if getattr(pos, 'tp', None) is not None:
                    trade['take_profit'] = float(pos.tp)
                trade['position_id'] = pid
                trade['status'] = 'OPEN'
                if trade.get('timestamp') is None and getattr(pos, 'time', None):
                    trade['timestamp'] = datetime.fromtimestamp(pos.time, tz=timezone.utc)
                updated = True

        # Close any trades marked OPEN but not found in MT5
        for trade in open_trades:
            pid = trade.get('position_id')
            if pid is not None and pid in open_tickets:
                continue

            closing_deal = None
            if pid is not None:
                deals = self.mt5_client.get_history_deals_by_position(pid)
                if deals:
                    closing_deals = [
                        d for d in deals
                        if d.entry == self.mt5_client.mt5.DEAL_ENTRY_OUT or d.entry == self.mt5_client.mt5.DEAL_ENTRY_INOUT
                    ]
                    if closing_deals:
                        closing_deal = closing_deals[-1]

            if closing_deal is None and (trade.get('deal_ticket') or trade.get('order_ticket')):
                try:
                    from_date = datetime.now(timezone.utc) - timedelta(days=30)
                    to_date = datetime.now(timezone.utc)
                    history_deals = self.mt5_client.get_history_deals(from_date, to_date)
                    for deal in (history_deals or []):
                        if getattr(deal, 'deal', None) == trade.get('deal_ticket') or getattr(deal, 'order', None) == trade.get('order_ticket'):
                            if deal.entry == self.mt5_client.mt5.DEAL_ENTRY_OUT or deal.entry == self.mt5_client.mt5.DEAL_ENTRY_INOUT:
                                closing_deal = deal
                                break
                except Exception:
                    closing_deal = None

            if closing_deal:
                status, reason_label, reason_code, close_time = self._classify_close_deal(closing_deal)
                trade['exit_price'] = closing_deal.price
                trade['profit'] = closing_deal.profit
                trade['status'] = status
                trade['close_reason_code'] = reason_code
                trade['close_reason'] = reason_label
                trade['close_time'] = close_time
            else:
                trade['status'] = 'CLOSED_UNKNOWN'
                trade['close_reason'] = 'MISSING_ON_RESTART'
                trade['close_time'] = datetime.now(timezone.utc)
            updated = True

        if updated:
            self.trade_logger.persist_all()

    async def execute_trade(self, signal: TradingSignal, symbol: str):
        try:
            sym_info = self.mt5_client.get_symbol_info(symbol)
            if not sym_info:
                return None
            pip = resolve_pip_size(symbol, sym_info, self.config)
            if pip <= 0:
                return None

            tick = self.mt5_client.get_symbol_info_tick(symbol)
            if not tick:
                return None
            exec_price = tick.ask if signal.type == 0 else tick.bid

            # Adjust risk calc using execution price
            actual_sl_pips = abs(exec_price - signal.stop_loss) / pip

            # Verify SL is not too tight
            min_sl_pips = float(self.config.min_stop_loss_pips)
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    min_sl_pips = float(sc.get('min_stop_loss_pips', min_sl_pips))
                    break
            if actual_sl_pips < min_sl_pips * 0.95:  # 5% tolerance
                logger.info(f"{symbol}: SL too tight ({actual_sl_pips:.1f}p < {min_sl_pips:.1f}p)")
                return None

            if not self.risk_manager.check_risk_limits():
                return None

            volume = self.risk_manager.calculate_position_size(symbol, actual_sl_pips)
            if volume <= 0:
                return None

            if not self.risk_manager.validate_trade_parameters(symbol, volume, signal.stop_loss, signal.take_profit, signal.type):
                return None

            # Final spread check before execution (spreads can spike in milliseconds)
            current_spread_pips = abs(float(tick.ask) - float(tick.bid)) / pip
            spread_guard = self.config.spread_guard_pips_default
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    spread_guard = sc.get('spread_guard_pips', spread_guard)
                    break
            if spread_guard is not None and float(spread_guard) > 0 and current_spread_pips > float(spread_guard):
                logger.info(f"{symbol}: Spread {current_spread_pips:.1f}p > guard {spread_guard}p at execution - rejected")
                return None

            result = self.mt5_client.place_order(
                symbol=symbol,
                order_type=signal.type,
                volume=volume,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"CORE_{signal.reason}",
                type_filling_override=self.symbol_fillings.get(symbol)
            )
            if result and getattr(result, 'retcode', None) == self.mt5_client.mt5.TRADE_RETCODE_DONE:
                order_ticket = getattr(result, 'order', None)
                deal_ticket = getattr(result, 'deal', None)

                # Get position ID - MT5 might return it directly, or we need to find it
                position_id = getattr(result, 'position', None)

                # If not in result, query MT5 immediately using order or deal ticket
                if position_id is None and (order_ticket or deal_ticket):
                    try:
                        import time
                        time.sleep(0.1)  # Brief delay for MT5 to register position
                        positions = self.mt5_client.get_positions(symbol)
                        for pos in (positions or []):
                            # Match by deal ticket (position opened by this deal)
                            pos_ticket = getattr(pos, 'ticket', None)
                            if pos_ticket == deal_ticket or pos_ticket == order_ticket:
                                position_id = pos_ticket
                                break
                            # Also check position identifier
                            if getattr(pos, 'identifier', None) == deal_ticket:
                                position_id = getattr(pos, 'ticket', None)
                                break
                    except Exception as e:
                        logger.warning(f"{symbol}: Failed to get position_id: {e}")

                trade = {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': symbol,
                    'order_type': 'BUY' if signal.type == 0 else 'SELL',
                    'entry_price': getattr(result, 'price', None) or float(exec_price),
                    'requested_price': float(exec_price),
                    'signal_price': signal.entry_price,
                    'volume': volume,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'order_ticket': order_ticket,
                    'deal_ticket': deal_ticket,
                    'position_id': position_id,
                    'reason': signal.reason,
                    'signal_time': signal.timestamp,
                    'breakout_level': signal.breakout_level,
                    'status': 'OPEN'
                }
                self.trade_logger.log_trade(trade)
                if self.trailing_manager and position_id is not None:
                    try:
                        self.trailing_manager.register_position(
                            position_id=position_id,
                            symbol=symbol,
                            direction=trade['order_type'],
                            entry_price=trade['entry_price'] if trade['entry_price'] is not None else float(exec_price),
                            entry_time=trade['timestamp'],
                            stop_loss=trade['stop_loss'],
                            take_profit=trade['take_profit'],
                            pip_size=pip,
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Failed to register trailing stop: {e}")
                ep = trade['entry_price'] if trade['entry_price'] is not None else exec_price
                logger.info(f"EXECUTED {symbol} {trade['order_type']} @{ep:.5f} vol={volume:.2f} order={order_ticket} deal={deal_ticket} pos={position_id}")
                return result
            else:
                err = getattr(result, 'comment', 'No result') if result else 'No result'
                logger.error(f"Order failed for {symbol}: {err}")
                return None
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)
            return None

    async def process_symbol(self, symbol_config: Dict):
        symbol = symbol_config['name']
        entry_tf = symbol_config.get('entry_timeframe', 'M15')
        trend_tf = symbol_config.get('trend_timeframe', 'H1')

        # Skip trading during daily rollover (21:55 - 23:05 UTC)
        if self._is_rollover_period():
            return

        try:
            # Determine how many bars needed for analysis (250 ensures 200 EMA can calculate)
            bars_needed = {
                'M1': 250, 'M5': 250, 'M15': 250, 'M30': 250,
                'H1': 250, 'H4': 250, 'D1': 250
            }.get(str(entry_tf).upper(), 250)

            candles = await self.market_data.fetch_data(symbol, entry_tf, bars_needed)
            if candles is None:
                return

            # Fetch higher timeframe data for trend filter
            trend_data = None
            if trend_tf and str(trend_tf).upper() != str(entry_tf).upper():
                trend_bars_needed = {
                    'M1': 250, 'M5': 250, 'M15': 250, 'M30': 250,
                    'H1': 250, 'H4': 250, 'D1': 250
                }.get(str(trend_tf).upper(), 250)
                trend_data = await self.market_data.fetch_data(symbol, trend_tf, trend_bars_needed)

            signal = self.strategy.generate_signal(candles, symbol, trend_data=trend_data)
            if signal:
                # Skip if same-direction position already exists
                try:
                    open_positions = self.mt5_client.get_positions(symbol)
                    for p in open_positions or []:
                        if int(getattr(p, 'type', -1)) == int(signal.type):
                            return  # Already have position in this direction
                except Exception:
                    pass
                await self.execute_trade(signal, symbol)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def monitor_open_trades(self):
        try:
            open_trades = [t for t in self.trade_logger.trades if t.get('status') == 'OPEN']
            if not open_trades:
                return

            for trade in open_trades:
                symbol = trade.get('symbol')
                position_id = trade.get('position_id')
                deal_ticket = trade.get('deal_ticket')
                order_ticket = trade.get('order_ticket')

                # If position_id is still None, try to find it from history or current positions
                if position_id is None:
                    if deal_ticket or order_ticket:
                        try:
                            # First check current positions
                            positions = self.mt5_client.get_positions(symbol) if symbol else self.mt5_client.get_all_positions()
                            for pos in (positions or []):
                                pos_ticket = getattr(pos, 'ticket', None)
                                if pos_ticket == deal_ticket or pos_ticket == order_ticket:
                                    position_id = pos_ticket
                                    trade['position_id'] = position_id
                                    self.trade_logger.persist_all()
                                    logger.debug(f"Found position_id {position_id} for trade {order_ticket}")
                                    break
                                # Check identifier field
                                if getattr(pos, 'identifier', None) == deal_ticket:
                                    position_id = getattr(pos, 'ticket', None)
                                    trade['position_id'] = position_id
                                    self.trade_logger.persist_all()
                                    logger.debug(f"Found position_id {position_id} via identifier for trade {order_ticket}")
                                    break
                        except Exception as e:
                            logger.warning(f"Error finding position_id for {order_ticket}: {e}")

                # If we still don't have position_id, try to find closed position in history
                if position_id is None and deal_ticket:
                    try:
                        # Query history to find the position
                        from_date = datetime.now(timezone.utc) - timedelta(days=7)
                        to_date = datetime.now(timezone.utc)
                        history_deals = self.mt5_client.get_history_deals(from_date, to_date)

                        for deal in (history_deals or []):
                            if getattr(deal, 'deal', None) == deal_ticket or getattr(deal, 'order', None) == order_ticket:
                                position_id = getattr(deal, 'position_id', None)
                                if position_id:
                                    trade['position_id'] = position_id
                                    self.trade_logger.persist_all()
                                    logger.debug(f"Found position_id {position_id} from history for trade {order_ticket}")
                                    break
                    except Exception as e:
                        logger.warning(f"Error searching history for position_id: {e}")

                # Skip if we still can't find position_id
                if position_id is None:
                    logger.debug(f"Cannot find position_id for trade {order_ticket}, skipping monitor")
                    continue

                # Check if position is still open by querying MT5 directly
                position_still_open = False
                position_obj = None
                try:
                    current_positions = self.mt5_client.get_positions(symbol) if symbol else self.mt5_client.get_all_positions()
                    for p in (current_positions or []):
                        if getattr(p, 'ticket', None) == position_id:
                            position_still_open = True
                            position_obj = p
                            break
                except Exception as e:
                    logger.warning(f"Error checking if position {position_id} is open: {e}")
                    continue

                if not position_still_open:
                    # The trade is no longer open, so we need to update its status
                    deals = self.mt5_client.get_history_deals_by_position(position_id)
                    if deals:
                        closing_deals = [d for d in deals if d.entry == self.mt5_client.mt5.DEAL_ENTRY_OUT or d.entry == self.mt5_client.mt5.DEAL_ENTRY_INOUT]
                        if closing_deals:
                            closing_deal = closing_deals[-1]
                            close_price = closing_deal.price
                            profit = closing_deal.profit
                            # Classify closure using deal reason enums when available (fallback to comment parsing)
                            mt5 = self.mt5_client.mt5
                            reason_code = getattr(closing_deal, 'reason', None)
                            status = 'CLOSED'
                            reason_label = None
                            try:
                                if reason_code is not None:
                                    if reason_code == getattr(mt5, 'DEAL_REASON_SL', -1):
                                        status, reason_label = 'CLOSED_SL', 'SL'
                                    elif reason_code == getattr(mt5, 'DEAL_REASON_TP', -1):
                                        status, reason_label = 'CLOSED_TP', 'TP'
                                    elif reason_code in (
                                        getattr(mt5, 'DEAL_REASON_SO', None),
                                        getattr(mt5, 'DEAL_REASON_STOPOUT', None),
                                    ):
                                        status, reason_label = 'CLOSED_STOPOUT', 'STOPOUT'
                                    elif reason_code in (
                                        getattr(mt5, 'DEAL_REASON_CLIENT', None),
                                        getattr(mt5, 'DEAL_REASON_MOBILE', None),
                                        getattr(mt5, 'DEAL_REASON_WEB', None),
                                    ):
                                        status, reason_label = 'CLOSED_MANUAL', 'MANUAL'
                                    elif reason_code == getattr(mt5, 'DEAL_REASON_EXPERT', None):
                                        status, reason_label = 'CLOSED_STRATEGY', 'EXPERT'
                                    elif reason_code == getattr(mt5, 'DEAL_REASON_DEALER', None):
                                        status, reason_label = 'CLOSED_DEALER', 'DEALER'
                            except Exception:
                                pass

                            # Fallback: parse comment text if reason didn’t resolve
                            if reason_label is None:
                                try:
                                    ctext = str(getattr(closing_deal, 'comment', '') or '').lower()
                                    if 'sl' in ctext:
                                        status, reason_label = 'CLOSED_SL', 'COMMENT_SL'
                                    elif 'tp' in ctext:
                                        status, reason_label = 'CLOSED_TP', 'COMMENT_TP'
                                    else:
                                        status, reason_label = 'CLOSED_MANUAL', 'COMMENT'
                                except Exception:
                                    status, reason_label = 'CLOSED', None

                            # Optional: close time from deal timestamp (best effort)
                            try:
                                close_time = datetime.fromtimestamp(getattr(closing_deal, 'time', 0), tz=timezone.utc) if getattr(closing_deal, 'time', None) else None
                            except Exception:
                                close_time = None

                            self.trade_logger.update_trade(
                                ticket=trade.get('order_ticket'),
                                exit_price=close_price,
                                profit=profit,
                                status=status,
                                close_reason_code=reason_code,
                                close_reason=reason_label,
                                close_time=close_time,
                                position_id=position_id
                            )
                            logger.info(f"Trade {trade.get('order_ticket')} closed ({status}) profit {profit}")
                    if self.trailing_manager:
                        try:
                            self.trailing_manager.unregister_position(position_id)
                        except Exception:
                            pass
                    continue

                # Trailing stop management for open positions
                if self.trailing_manager:
                    try:
                        # Register position for trailing if missing
                        if self.trailing_manager.get_position_state(position_id) is None:
                            sym_info = self.mt5_client.get_symbol_info(symbol) if symbol else None
                            pip = resolve_pip_size(symbol, sym_info, self.config) if sym_info else 0.0
                            entry_price = trade.get('entry_price')
                            stop_loss = trade.get('stop_loss')
                            entry_time = trade.get('timestamp')
                            # Parse persisted timestamp if needed
                            if isinstance(entry_time, str):
                                try:
                                    entry_time = datetime.fromisoformat(entry_time)
                                except Exception:
                                    entry_time = None

                            # Fall back to position time if available
                            if entry_time is None and position_obj is not None:
                                try:
                                    ptime = getattr(position_obj, 'time', None)
                                    if ptime:
                                        entry_time = datetime.fromtimestamp(ptime, tz=timezone.utc)
                                except Exception:
                                    entry_time = None

                            entry_time = entry_time or datetime.now(timezone.utc)
                            # Prefer actual SL from MT5 if present
                            current_sl = None
                            try:
                                if position_obj is not None and getattr(position_obj, 'sl', None):
                                    current_sl = float(position_obj.sl)
                            except Exception:
                                current_sl = None

                            if current_sl is None:
                                current_sl = trade.get('trailing_current_sl') or stop_loss

                            if pip > 0 and entry_price is not None and stop_loss is not None:
                                self.trailing_manager.register_position(
                                    position_id=position_id,
                                    symbol=symbol,
                                    direction=trade.get('order_type', ''),
                                    entry_price=float(entry_price),
                                    entry_time=entry_time,
                                    stop_loss=float(stop_loss),
                                    take_profit=trade.get('take_profit'),
                                    pip_size=pip,
                                    current_sl=current_sl,
                                    state=trade.get('trailing_state'),
                                    highest_profit_pips=trade.get('trailing_mfe_pips'),
                                    sl_updates=trade.get('trailing_sl_updates'),
                                    last_update_time=trade.get('trailing_last_update_time'),
                                )

                        tick = self.mt5_client.get_symbol_info_tick(symbol) if symbol else None
                        if tick:
                            direction = str(trade.get('order_type', '')).upper()
                            current_price = float(tick.bid) if direction == 'BUY' else float(tick.ask)
                            update = self.trailing_manager.update_position(
                                position_id, current_price, datetime.now(timezone.utc)
                            )
                            new_sl = update.get('new_sl') if isinstance(update, dict) else None
                            if new_sl is not None:
                                if not self.trailing_manager.update_sl_live(position_id, new_sl):
                                    logger.debug(f"{symbol}: Failed trailing SL update for {position_id}")
                            if update.get('should_close_time'):
                                self.trailing_manager.close_position_live(position_id, reason="TIME_EXIT")

                            # Persist trailing state changes
                            if isinstance(update, dict):
                                prev_state = trade.get('trailing_state')
                                cur_state = update.get('state')
                                if new_sl is not None or (cur_state and cur_state != prev_state):
                                    state_obj = self.trailing_manager.get_position_state(position_id)
                                    fields = {
                                        'trailing_state': cur_state,
                                        'trailing_current_sl': update.get('current_sl'),
                                        'trailing_profit_pips': update.get('profit_pips'),
                                        'trailing_mfe_pips': update.get('mfe_pips'),
                                        'trailing_last_update_time': datetime.now(timezone.utc),
                                    }
                                    if state_obj is not None:
                                        fields['trailing_sl_updates'] = state_obj.sl_updates
                                    self.trade_logger.update_trade(ticket=trade.get('order_ticket'), **fields)
                    except Exception as e:
                        logger.warning(f"{symbol}: Trailing stop error for {position_id}: {e}")
        except Exception as e:
            logger.error(f"Error monitoring open trades: {e}", exc_info=True)

    async def run(self):
        self.running = True
        logger.info("Starting trading loop")
        loop_count = 0

        while self.running:
            try:
                # Ensure MT5 connectivity
                if not self.mt5_client.is_connected():
                    logger.warning("MT5 disconnected; reconnecting...")
                    self.mt5_client.reconnect()
                    if not self.mt5_client.is_connected():
                        await asyncio.sleep(self.config.main_loop_interval)
                        continue

                await self.monitor_open_trades()
                for sym in self.config.symbols:
                    await self.process_symbol(sym)

                # Periodic state persistence
                loop_count += 1
                if loop_count % 10 == 0:
                    try:
                        if self.strategy:
                            save_strategy_state(self.state_manager, self.strategy)
                    except Exception as e:
                        logger.debug(f"State save error: {e}")

                await asyncio.sleep(self.config.main_loop_interval)
            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                await asyncio.sleep(self.config.main_loop_interval)

    async def shutdown(self):
        logger.info("Bot shutdown")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Forex Trading Bot (Core)')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--risk-per-trade', type=float)
    parser.add_argument('--symbol', type=str)
    parser.add_argument('--timeframe', type=str)
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    return parser.parse_args()


async def main():
    args = parse_arguments()
    configure_logging(args.log_level)
    cfg = Config(args.config, args)
    bot = TradingBot(cfg)
    try:
        await bot.initialize()
        await bot.run()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
