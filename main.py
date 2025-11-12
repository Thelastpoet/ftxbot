#!/usr/bin/env python3
"""
Forex Trading Bot - Core
"""

import asyncio
import logging
import json
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from utils import resolve_pip_size
from strategy import PurePriceActionStrategy, TradingSignal

def configure_logging(console_level: str = 'INFO') -> None:
    """Configure production-friendly logging for console and file.

    - Console: concise format at requested level (default INFO).
    - File: full detail at DEBUG in forex_bot.log.
    - Suppress noisy third-party debug (e.g., asyncio proactor message).
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicates on reload
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler: verbose
    fh = logging.FileHandler('forex_bot.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler: concise
    ch = logging.StreamHandler()
    ch.setLevel(level_map.get(str(console_level).upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    ))

    root.addHandler(fh)
    root.addHandler(ch)

    # Tame noisy libraries
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('MetaTrader5').setLevel(logging.WARNING)

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
            self.min_stop_loss_pips = trading.get('min_stop_loss_pips', 20)
            self.stop_loss_buffer_pips = trading.get('stop_loss_buffer_pips', 15)
            # Patterns and ATR (TA-Lib) integration
            self.enable_patterns = trading.get('enable_patterns', True)
            self.pattern_window = trading.get('pattern_window', 3)
            self.pattern_score_threshold = trading.get('pattern_score_threshold', 0.6)
            self.pattern_strong_threshold = trading.get('pattern_strong_threshold', 0.8)
            self.allowed_patterns = trading.get('allowed_patterns', None)
            self.dup_relax_on_strong_pattern = trading.get('dup_relax_on_strong_pattern', True)
            self.headroom_relax_pct_on_strong = trading.get('headroom_relax_pct_on_strong', 0.25)
            self.obstacle_buffer_relax_pct_on_strong = trading.get('obstacle_buffer_relax_pct_on_strong', 0.3)
            self.atr_source = trading.get('atr_source', 'talib')
            # ATR/headroom controls
            self.atr_period = trading.get('atr_period', 14)
            self.atr_sl_k = trading.get('atr_sl_k', 0.6)
            self.min_sl_buffer_pips = trading.get('min_sl_buffer_pips', 10)
            self.max_sl_pips = trading.get('max_sl_pips', None)
            self.min_headroom_rr = trading.get('min_headroom_rr', 1.2)
            self.max_rr_cap = trading.get('max_rr_cap', None)
            # Optional global guards (used as defaults when per-symbol not set)
            self.spread_guard_pips_default = trading.get('spread_guard_pips', None)
            self.duplicate_breakout_distance_pips_default = trading.get('duplicate_breakout_distance_pips', None)
            self.duplicate_breakout_window_seconds_default = trading.get('duplicate_breakout_window_seconds', None)

            risk = data.get('risk_management', {})
            self.risk_per_trade = risk.get('risk_per_trade', 0.01)
            self.fixed_lot_size = risk.get('fixed_lot_size', None)
            self.max_drawdown = risk.get('max_drawdown_percentage', 0.05)
            self.risk_reward_ratio = risk.get('risk_reward_ratio', 2.0)
            self.use_equity = risk.get('use_equity', True)

            # AMD settings
            amd = data.get('amd_settings', {})
            self.amd_enabled = amd.get('enabled', False)
            self.amd_enable_asian_trading = amd.get('enable_asian_trading', False)
            self.amd_enable_london_trading = amd.get('enable_london_trading', True)
            self.amd_enable_ny_trading = amd.get('enable_ny_trading', True)
            self.amd_asian_session_hours = tuple(amd.get('asian_session_hours', [0, 8]))
            self.amd_london_session_hours = tuple(amd.get('london_session_hours', [8, 16]))
            self.amd_ny_session_hours = tuple(amd.get('ny_session_hours', [13, 22]))
            # Additional AMD toggles
            self.amd_trade_unknown_days = amd.get('trade_unknown_days', False)
            self.amd_late_ny_hour_utc = amd.get('late_ny_hour_utc', 19)
            self.amd_volatility_factor_min = amd.get('volatility_factor_min', 0.5)
            self.amd_volatility_factor_max = amd.get('volatility_factor_max', 2.0)
            self.amd_fallback_narrow_range_pips = amd.get('fallback_narrow_range_pips', 50)
            self.level_merge_proximity_pips = amd.get('level_merge_proximity_pips', 10)
            # AMD logging preferences
            self.amd_log_on_change = amd.get('log_on_change', True)
            self.amd_log_each_loop = amd.get('log_each_loop', False)
            # Breakout/pattern relax controls during AMD EXPANSION
            self.amd_breakout_threshold_volatility_scaled = amd.get('breakout_threshold_volatility_scaled', True)
            self.amd_breakout_relax_in_expansion_factor = amd.get('breakout_relax_in_expansion_factor', 0.85)
            self.amd_breakout_min_pips_floor = amd.get('breakout_min_pips_floor', 0.0)
            self.amd_patterns_required_in_expansion = amd.get('patterns_required_in_expansion', False)
            # Accumulation detection
            accum = amd.get('accumulation_detection', {})
            self.amd_max_range_vs_adr_ratio = accum.get('max_range_vs_adr_ratio', 0.4)
            self.amd_min_overlap_bars = accum.get('min_overlap_bars', 5)
            self.amd_max_price_from_open_pips = accum.get('max_price_from_open_pips', 30)
            # Manipulation detection
            manip = amd.get('manipulation_detection', {})
            self.amd_manipulation_enabled = manip.get('enabled', True)
            self.amd_sweep_threshold_pips = manip.get('sweep_threshold_pips', 5)
            self.amd_reversal_confirmation_bars = manip.get('reversal_confirmation_bars', 3)
            self.amd_reversal_threshold_pips = manip.get('reversal_threshold_pips', 15)
            # Directional bias
            bias = amd.get('directional_bias', {})
            self.amd_directional_bias_enabled = bias.get('enabled', True)
            self.amd_allow_countertrend_in_expansion = bias.get('allow_countertrend_in_expansion', False)
            self.amd_disable_trades_in_distribution = bias.get('disable_trades_in_distribution', False)
            # Distribution detection config passthrough
            self.amd_distribution_detection = amd.get('distribution_detection', {})
            # MTF trend confirmation config passthrough
            self.amd_mtf_trend_confirmation = amd.get('mtf_trend_confirmation', {})

            # Symbols
            self.symbols = []
            for s in data.get('symbols', []) or []:
                entry = dict(s)
                entry.setdefault('name', s.get('name'))
                entry.setdefault('timeframes', s.get('timeframes', ['M15']))
                self.symbols.append(entry)

            # CLI overrides
            if self.args:
                if self.args.risk_per_trade is not None:
                    self.risk_per_trade = float(self.args.risk_per_trade)
                if self.args.symbol:
                    self.symbols = [{'name': self.args.symbol, 'timeframes': ['M15']}]
                if self.args.timeframe:
                    for sym in self.symbols:
                        sym['timeframes'] = [self.args.timeframe]

            logger.info(f"Config loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.set_defaults()

    def set_defaults(self):
        self.main_loop_interval = 5
        self.lookback_period = 20
        self.swing_window = 5
        self.breakout_threshold = 7
        self.min_stop_loss_pips = 20
        self.stop_loss_buffer_pips = 15
        self.enable_patterns = True
        self.pattern_window = 3
        self.pattern_score_threshold = 0.6
        self.pattern_strong_threshold = 0.8
        self.allowed_patterns = None
        self.dup_relax_on_strong_pattern = True
        self.headroom_relax_pct_on_strong = 0.25
        self.obstacle_buffer_relax_pct_on_strong = 0.3
        self.atr_source = 'talib'
        self.atr_period = 14
        self.atr_sl_k = 0.6
        self.min_sl_buffer_pips = 10
        self.max_sl_pips = None
        self.min_headroom_rr = 1.2
        self.max_rr_cap = None
        self.spread_guard_pips_default = None
        self.duplicate_breakout_distance_pips_default = None
        self.duplicate_breakout_window_seconds_default = None
        self.risk_per_trade = 0.01
        self.fixed_lot_size = None
        self.max_drawdown = 0.05
        self.risk_reward_ratio = 2.0
        self.symbols = [{'name': 'EURUSD', 'timeframes': ['M15']}]
        # AMD defaults (disabled)
        self.amd_enabled = False
        # No momentum filter config; price-action + MTF context only

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.trade_logger = None
        self.amd_analyzer = None
        self.running = False
        self.symbol_fillings = {}

    async def initialize(self):
        try:
            from mt5_client import MetaTrader5Client
            from market_data import MarketData
            from risk_manager import RiskManager
            from trade_logger import TradeLogger

            self.mt5_client = MetaTrader5Client()
            if not self.mt5_client.initialized:
                raise RuntimeError('Failed to initialize MT5 connection')

            self.market_data = MarketData(self.mt5_client, self.config)
            self.strategy = PurePriceActionStrategy(self.config)
            self.risk_manager = RiskManager(self.config, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')

            # Initialize AMD analyzer if enabled
            if self.config.amd_enabled:
                from amd_context import AMDAnalyzer
                self.amd_analyzer = AMDAnalyzer(self.config)
                logger.info("AMD context analyzer enabled")

            # Log current trading session (UTC) on startup
            try:
                from sessions import get_current_session
                now_utc = datetime.now(timezone.utc)
                session = get_current_session(
                    now_utc,
                    self.config.amd_asian_session_hours,
                    self.config.amd_london_session_hours,
                    self.config.amd_ny_session_hours,
                )
                logger.info(f"Current session (UTC): {session}")
            except Exception:
                pass

            # Strategy snapshot
            try:
                logger.info(
                    "Strategy init: patterns=%s window=%s thr=%s/%s atr=%s rr=%.2f lookback=%s swing=%s",
                    'ON' if getattr(self.config, 'enable_patterns', True) else 'OFF',
                    getattr(self.config, 'pattern_window', 3),
                    getattr(self.config, 'pattern_score_threshold', 0.6),
                    getattr(self.config, 'pattern_strong_threshold', 0.8),
                    getattr(self.config, 'atr_source', 'talib'),
                    float(getattr(self.config, 'risk_reward_ratio', 2.0)),
                    getattr(self.config, 'lookback_period', 20),
                    getattr(self.config, 'swing_window', 5),
                )
            except Exception:
                pass

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
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def execute_trade(self, signal: TradingSignal, symbol: str, amd_context: Optional[Dict] = None):
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
            # Drift guard relative to signal
            try:
                drift_pips = abs(exec_price - signal.entry_price) / pip
                drift_max_pips = None
                drift_fraction_of_sl = 0.5
                for sc in getattr(self.config, 'symbols', []) or []:
                    if sc.get('name') == symbol:
                        drift_max_pips = sc.get('drift_max_pips', None)
                        drift_fraction_of_sl = float(sc.get('drift_fraction_of_sl', drift_fraction_of_sl))
                        break
                allowed_drift = actual_sl_pips * drift_fraction_of_sl
                if drift_max_pips is not None:
                    allowed_drift = min(float(drift_max_pips), allowed_drift)
                if drift_pips > allowed_drift:
                    logger.debug(f"{symbol}: Execution drift {drift_pips:.1f}p exceeds allowed {allowed_drift:.1f}p")
                    return None
            except Exception:
                pass

            # Per-symbol min SL override with small slack to account for drift
            min_sl_pips = float(self.config.min_stop_loss_pips)
            min_slack_fraction = 0.05
            for sc in getattr(self.config, 'symbols', []) or []:
                if sc.get('name') == symbol:
                    min_sl_pips = float(sc.get('min_stop_loss_pips', min_sl_pips))
                    min_slack_fraction = float(sc.get('min_stop_loss_slack_fraction', min_slack_fraction))
                    break
            if actual_sl_pips < (min_sl_pips * (1.0 - min_slack_fraction)):
                logger.debug(f"{symbol}: SL too tight after price drift ({actual_sl_pips:.1f}p < {min_sl_pips*(1-min_slack_fraction):.1f}p)")
                return None

            if not self.risk_manager.check_risk_limits():
                return None

            volume = self.risk_manager.calculate_position_size(symbol, actual_sl_pips)
            if volume <= 0:
                return None

            if not self.risk_manager.validate_trade_parameters(symbol, volume, signal.stop_loss, signal.take_profit, signal.type):
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
                    'order_ticket': getattr(result, 'order', None),
                    'deal_ticket': getattr(result, 'deal', None),
                    'position_id': getattr(result, 'position', None),
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'breakout_level': signal.breakout_level,
                    'pattern_score': getattr(signal, 'pattern_score', None),
                    'pattern_dir': getattr(signal, 'pattern_dir', None),
                    'pattern_primary': getattr(signal, 'pattern_primary', None),
                    'amd_phase': amd_context.get('phase') if amd_context else None,
                    'amd_session': amd_context.get('session') if amd_context else None,
                    'amd_asia_type': amd_context.get('asia_type') if amd_context else None,
                    'amd_direction': amd_context.get('allowed_direction') if amd_context else None,
                    'status': 'OPEN'
                }
                self.trade_logger.log_trade(trade)
                ep = trade['entry_price'] if trade['entry_price'] is not None else exec_price
                logger.info(f"EXECUTED {symbol} {trade['order_type']} @{ep:.5f} vol={volume:.2f} order={trade['order_ticket']} deal={trade['deal_ticket']}")
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
        tfs = (symbol_config.get('timeframes') or ['M15'])
        timeframe = tfs[0]

        try:
            # Base timeframe data
            candles = await self.market_data.fetch_data(symbol, timeframe, max(100, self.config.lookback_period * 3))
            if candles is None:
                return

            # Fetch additional timeframes for multi-timeframe context (do not trade them here)
            mtf_context = {}
            try:
                for tf in tfs[1:]:
                    df_tf = await self.market_data.fetch_data(symbol, tf, max(100, self.config.lookback_period * 3))
                    if df_tf is not None:
                        mtf_context[tf] = df_tf
            except Exception:
                mtf_context = {}

            # Compute AMD context if enabled
            amd_context = None
            if self.amd_analyzer is not None:
                try:
                    amd_context = self.amd_analyzer.analyze(candles, symbol, timeframe, mtf_context=mtf_context)
                except Exception as e:
                    logger.warning(f"{symbol}: AMD analysis failed: {e}")
                    amd_context = None

            # AMD summary logging (terminal observability)
            if amd_context is not None:
                try:
                    sess = amd_context.get('session', 'UNKNOWN')
                    phase = amd_context.get('phase', 'UNKNOWN')
                    asia_type = amd_context.get('asia_type', 'UNKNOWN')
                    vr = amd_context.get('volatility_factor', None)
                    vtxt = f"v={vr:.2f}" if isinstance(vr, (int, float)) else "v=n/a"
                    rng = amd_context.get('asia_range_pips')
                    rngtxt = f"rng={rng:.1f}p" if isinstance(rng, (int, float)) else "rng=n/a"
                    mt = amd_context.get('manipulation_type', None)
                    shc = amd_context.get('sweep_high_confirmed', False)
                    slc = amd_context.get('sweep_low_confirmed', False)
                    shcnt = amd_context.get('sweep_high_count', 0)
                    slcnt = amd_context.get('sweep_low_count', 0)
                    bias = amd_context.get('allowed_direction', None)
                    htf = amd_context.get('higher_tf_trend', None)
                    reason = amd_context.get('reason', '')

                    summary = (
                        f"AMD {symbol} {timeframe} sess={sess} phase={phase} asia={asia_type} "
                        f"{rngtxt} {vtxt} sweeps:H{shcnt}/L{slcnt} conf:H={bool(shc)}/L={bool(slc)} "
                        f"manip={mt or 'none'} bias={bias or 'ANY'} htf={htf or 'n/a'} reason={reason}"
                    )

                    # Log only on change by default, optionally each loop
                    if not hasattr(self, '_amd_last_state'):
                        self._amd_last_state = {}
                    key = (symbol, timeframe)
                    state_tuple = (sess, phase, asia_type, bool(shc), bool(slc), mt, bias, htf)
                    last = self._amd_last_state.get(key)
                    log_each = getattr(self.config, 'amd_log_each_loop', False)
                    log_on_change = getattr(self.config, 'amd_log_on_change', True)
                    if log_each or (log_on_change and last != state_tuple):
                        logger.info(summary)
                        self._amd_last_state[key] = state_tuple
                except Exception:
                    pass

            signal = self.strategy.generate_signal(candles, symbol, mtf_context=mtf_context if mtf_context else None, amd_context=amd_context)
            if signal:
                # Skip if an existing position with same direction and our magic exists
                try:
                    open_positions = self.mt5_client.get_positions(symbol)
                    same_dir_exists = False
                    for p in open_positions or []:
                        # 0=BUY,1=SELL per MT5
                        if int(getattr(p, 'type', -1)) == int(signal.type) and int(getattr(p, 'magic', 0)) == 234000:
                            same_dir_exists = True
                            break
                    if same_dir_exists:
                        return
                except Exception:
                    pass
                await self.execute_trade(signal, symbol, amd_context=amd_context)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def monitor_open_trades(self):
        try:
            open_trades = [t for t in self.trade_logger.trades if t.get('status') == 'OPEN']
            if not open_trades:
                return

            open_positions = self.mt5_client.get_all_positions()

            for trade in open_trades:
                # Find the position for this trade
                position_id = trade.get('position_id')
                deal_ticket = trade.get('deal_ticket')

                # If position_id is None, try to find it from the open positions using deal_ticket
                if position_id is None and deal_ticket is not None:
                    for p in (open_positions or []):
                        # Match by checking if this position was opened by our deal
                        if getattr(p, 'identifier', None) == deal_ticket or getattr(p, 'ticket', None) == deal_ticket:
                            position_id = getattr(p, 'ticket', None)
                            # Update the trade with the found position_id for future reference
                            trade['position_id'] = position_id
                            self.trade_logger.persist_all()
                            break

                # Skip monitoring if we still can't find a valid position_id
                if position_id is None:
                    continue

                # Check if position is still open
                position_still_open = any(getattr(p, 'ticket', None) == position_id for p in (open_positions or []))

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
                                close_time=close_time
                            )
                            logger.info(f"Trade {trade.get('order_ticket')} closed ({status}) profit {profit}")
        except Exception as e:
            logger.error(f"Error monitoring open trades: {e}", exc_info=True)

    async def run(self):
        self.running = True
        logger.info("Starting core trading loop")
        while self.running:
            try:
                await self.monitor_open_trades()
                for sym in self.config.symbols:
                    await self.process_symbol(sym)
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
