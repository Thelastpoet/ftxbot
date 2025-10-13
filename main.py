#!/usr/bin/env python3
"""
Forex Trading Bot - Main Application
Pure Price Action Strategy with MetaTrader 5 Integration
"""

import asyncio
import logging
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

from strategy import TradingSignal
from market_session import MarketSession
from utils import get_pip_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the trading bot"""
    
    def __init__(self, config_path: str = 'config.json', args: Optional[argparse.Namespace] = None):
        self.config_path = Path(config_path)
        self.args = args
        self.load_config()
        
    def load_config(self):
        """Load configuration from JSON file and apply CLI overrides"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Trading settings
            trading = config_data.get('trading_settings', {})
            self.max_period = trading.get('max_period', 50)
            self.main_loop_interval = trading.get('main_loop_interval_seconds', 60)
            self.order_retry_attempts = trading.get('order_retry_attempts', 3)
            self.lookback_period = trading.get('lookback_period', 20)
            self.swing_window = trading.get('swing_window', 5)
            self.breakout_threshold = trading.get('breakout_threshold_pips', 5)
            self.min_peak_rank = trading.get('min_peak_rank', 2)
            self.proximity_threshold = trading.get('proximity_threshold_pips', 10)
            self.min_stop_loss_pips = trading.get('min_stop_loss_pips', 20)
            self.stop_loss_buffer_pips = trading.get('stop_loss_buffer_pips', 15)
            self.min_range_pips = trading.get('min_range_pips', 10)
            self.atr_period = trading.get('atr_period', 14)
            self.atr_tp_multiplier = trading.get('atr_tp_multiplier', 4.0)
            self.session_based_tp = trading.get('session_based_tp', True)
            self.volatility_adjustment = trading.get('volatility_adjustment', True)

            # Adaptive SL configuration (top-level keys with sensible defaults)
            self.stop_loss_atr_multiplier = float(config_data.get('stop_loss_atr_multiplier', 1.0))
            self.spread_floor_multiplier = float(config_data.get('spread_floor_multiplier', 1.1))
            
            # Risk management
            risk = config_data.get('risk_management', {})
            self.risk_per_trade = risk.get('risk_per_trade', 0.02)
            self.fixed_lot_size = risk.get('fixed_lot_size', None)
            self.max_drawdown = risk.get('max_drawdown_percentage', 0.1)
            self.risk_reward_ratio = risk.get('risk_reward_ratio', 2.0)

            # Liquidity Context settings (accept at top-level or under risk_management)
            self.liquidity_context_enabled = bool(config_data.get(
                'liquidity_context_enabled', risk.get('liquidity_context_enabled', True)))
            self.liquidity_context_enforce = bool(config_data.get(
                'liquidity_context_enforce', risk.get('liquidity_context_enforce', False)))
            self.liquidity_equal_level_tolerance_pips = float(config_data.get(
                'liquidity_equal_level_tolerance_pips', risk.get('liquidity_equal_level_tolerance_pips', 6)))
            self.liquidity_cushion_atr_mult = float(config_data.get(
                'liquidity_cushion_atr_mult', risk.get('liquidity_cushion_atr_mult', 0.6)))
            self.liquidity_cushion_atr_mult_repeat = float(config_data.get(
                'liquidity_cushion_atr_mult_repeat', risk.get('liquidity_cushion_atr_mult_repeat', 0.8)))
            self.liquidity_min_cushion_pips = float(config_data.get(
                'liquidity_min_cushion_pips', risk.get('liquidity_min_cushion_pips', 4.0)))
            self.liquidity_recent_sweep_max_age_bars = int(config_data.get(
                'liquidity_recent_sweep_max_age_bars', risk.get('liquidity_recent_sweep_max_age_bars', 15)))
            self.liquidity_sweep_lookback_bars = int(config_data.get(
                'liquidity_sweep_lookback_bars', risk.get('liquidity_sweep_lookback_bars', 40)))
            self.liquidity_wick_to_body_ratio_min = float(config_data.get(
                'liquidity_wick_to_body_ratio_min', risk.get('liquidity_wick_to_body_ratio_min', 1.5)))
            self.liquidity_use_close_back = bool(config_data.get(
                'liquidity_use_close_back', risk.get('liquidity_use_close_back', True)))
            self.liquidity_require_confirmation_on_dual_pools = bool(config_data.get(
                'liquidity_require_confirmation_on_dual_pools', risk.get('liquidity_require_confirmation_on_dual_pools', True)))
            self.liquidity_max_sl_distance_pips = config_data.get(
                'liquidity_max_sl_distance_pips', risk.get('liquidity_max_sl_distance_pips', None))

            # Order deviation (points) default
            self.order_deviation_points = int(config_data.get('order_deviation_points', 20))

            # M1 confirmation (configurable at top-level under "m1_confirmation" or via trading_settings fallbacks)
            m1 = config_data.get('m1_confirmation', {}) or {}
            self.m1_confirmation_enabled = bool(m1.get('enabled', trading.get('m1_confirmation_enabled', False)))
            self.m1_confirmation_candles = int(m1.get('candles', trading.get('m1_confirmation_candles', 1)))
            self.m1_confirmation_buffer_pips = float(m1.get('buffer_pips', trading.get('m1_confirmation_buffer_pips', 0.5)))
            self.m1_confirmation_dynamic_buffer = bool(m1.get('dynamic_buffer', True))
            # If min_buffer not provided, default to buffer_pips
            self.m1_confirmation_min_buffer_pips = float(m1.get('min_buffer_pips', self.m1_confirmation_buffer_pips))
            self.m1_confirmation_spread_multiplier = float(m1.get('spread_multiplier', 1.2))

            # Basket take profit (close only profitable positions on net threshold)
            btp = config_data.get('basket_take_profit', {}) or {}
            self.basket_take_profit = {
                'enabled': bool(btp.get('enabled', False)),
                'mode': str(btp.get('mode', 'equity')),  # 'equity' | 'sum'
                'threshold_amount': float(btp.get('threshold_amount', 25.0)),
                'percent_of_balance': (float(btp['percent_of_balance'])
                                       if btp.get('percent_of_balance') is not None else None),
                'close_only_profitable': bool(btp.get('close_only_profitable', True)),
                'cooldown_seconds': int(btp.get('cooldown_seconds', 60)),
            }

            # Toggle new position manager integration
            self.enable_position_manager = bool(config_data.get('enable_position_manager', False))

            # Position management (session-aware, protective)
            pm = config_data.get('position_management', {}) or {}
            self.position_management = {
                'manage_timeframe': pm.get('manage_timeframe', 'M15'),
                'breakeven': {
                    'enabled': pm.get('breakeven', {}).get('enabled', True),
                    'trigger_r_multiple': pm.get('breakeven', {}).get('trigger_r_multiple', 1.0),
                    'buffer_pips': pm.get('breakeven', {}).get('buffer_pips', 0.3),
                },
                'trailing': {
                    'mode': pm.get('trailing', {}).get('mode', 'atr'),
                    'atr_period': pm.get('trailing', {}).get('atr_period', 14),
                    'atr_multiplier': pm.get('trailing', {}).get('atr_multiplier', 1.25),
                    'start_after_r_multiple': pm.get('trailing', {}).get('start_after_r_multiple', 1.0),
                    'step_trigger_pips': pm.get('trailing', {}).get('step_trigger_pips', 10.0),
                    'step_distance_pips': pm.get('trailing', {}).get('step_distance_pips', 8.0),
                },
                'session_exit': {
                    'enabled': pm.get('session_exit', {}).get('enabled', True),
                    'boundary_minutes': pm.get('session_exit', {}).get('boundary_minutes', 10),
                    'respect_blackouts': pm.get('session_exit', {}).get('respect_blackouts', True),
                    'min_r_to_keep': pm.get('session_exit', {}).get('min_r_to_keep', 0.5),
                    'tighten_atr_multiplier': pm.get('session_exit', {}).get('tighten_atr_multiplier', 1.0),
                },
                'guardrails': {
                    'min_seconds_between_mods': pm.get('guardrails', {}).get('min_seconds_between_mods', 15),
                    'min_improvement_pips': pm.get('guardrails', {}).get('min_improvement_pips', 0.2),
                    'max_spread_pips_for_mods': pm.get('guardrails', {}).get('max_spread_pips_for_mods', 4.0),
                },
            }
            
            # Symbols and per-symbol overrides
            symbols_data = config_data.get('symbols', [])
            self.symbols = []
            for sym in symbols_data:
                # Preserve any additional per-symbol overrides if provided
                entry = dict(sym)
                entry.setdefault('name', sym.get('name'))
                entry.setdefault('timeframes', sym.get('timeframes', ['M15']))
                self.symbols.append(entry)
            
            # Trading sessions
            sessions_data = config_data.get('trading_sessions', [])
            self.trading_sessions = []
            for session in sessions_data:
                self.trading_sessions.append({
                    'name': session['name'],
                    'start_time': datetime.strptime(session['start_time'], '%H:%M').time(),
                    'end_time': datetime.strptime(session['end_time'], '%H:%M').time()
                })
            
            # Apply command-line overrides
            if self.args:
                if self.args.risk_per_trade:
                    self.risk_per_trade = self.args.risk_per_trade
                if self.args.symbol:
                    self.symbols = [{'name': self.args.symbol, 'timeframes': ['M15']}]
                if self.args.timeframe:
                    for sym in self.symbols:
                        sym['timeframes'] = [self.args.timeframe]
                        
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            self.set_defaults()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            self.set_defaults()
    
    def set_defaults(self):
        """Set default configuration values"""
        self.max_period = 50
        self.main_loop_interval = 60
        self.order_retry_attempts = 3
        self.lookback_period = 20
        self.swing_window = 5
        self.breakout_threshold = 5
        self.min_peak_rank = 2
        self.proximity_threshold = 10
        self.risk_per_trade = 0.02
        self.fixed_lot_size = None
        self.max_drawdown = 0.1
        self.risk_reward_ratio = 1.25
        self.symbols = [{'name': 'EURUSD', 'timeframes': ['M15']}]
        self.trading_sessions = []
        self.min_stop_loss_pips = 20
        self.stop_loss_buffer_pips = 15
        self.min_range_pips = 10
        self.atr_period = 14
        self.atr_tp_multiplier = 4.0
        self.session_based_tp = True
        self.volatility_adjustment = True
        # Adaptive SL defaults
        self.stop_loss_atr_multiplier = 1.0
        self.spread_floor_multiplier = 1.1
        # Basket take profit defaults
        self.basket_take_profit = {
            'enabled': False,
            'mode': 'equity',
            'threshold_amount': 25.0,
            'percent_of_balance': None,
            'close_only_profitable': True,
            'cooldown_seconds': 60,
        }
        # Disable position manager by default until it's production-ready
        self.enable_position_manager = False
        # Position management defaults (protective, non-destructive)
        self.position_management = {
            'manage_timeframe': 'M15',
            'breakeven': {
                'enabled': True,
                'trigger_r_multiple': 1.0,
                'buffer_pips': 0.3,
            },
            'trailing': {
                'mode': 'atr',
                'atr_period': 14,
                'atr_multiplier': 1.25,
                'start_after_r_multiple': 1.0,
                'step_trigger_pips': 10.0,
                'step_distance_pips': 8.0,
            },
            'session_exit': {
                'enabled': True,
                'boundary_minutes': 10,
                'respect_blackouts': True,
                'min_r_to_keep': 0.5,
                'tighten_atr_multiplier': 1.0,
            },
            'guardrails': {
                'min_seconds_between_mods': 15,
                'min_improvement_pips': 0.2,
                'max_spread_pips_for_mods': 4.0,
            },
        }

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.position_manager = None
        self.trade_logger = None
        self.consumed_breakouts = {}
        self.recent_closures = {} 
        self.peak_equity = None
        self.load_memory_state()
        self.session_manager = MarketSession(config)
        self.running = False
        self.initial_balance = None
        self._last_basket_tp_ts = None
        
    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Import components (delayed to allow for proper module structure)
            from mt5_client import MetaTrader5Client
            from market_data import MarketData
            from strategy import PurePriceActionStrategy
            from risk_manager import RiskManager
            from trade_logger import TradeLogger

            # Initialize components
            self.mt5_client = MetaTrader5Client()
            if not self.mt5_client.initialized:
                raise Exception("Failed to initialize MetaTrader5 connection")
            
            self.market_data = MarketData(self.mt5_client, self.config)
            self.strategy = PurePriceActionStrategy(self.config)
            self.risk_manager = RiskManager(self.config, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')
            
            # Position manager is disabled by default; only load when explicitly enabled
            if getattr(self.config, 'enable_position_manager', False):
                try:
                    from position_manager import (
                        PositionManager,
                        PositionManagerConfig,
                        BreakEvenConfig,
                        TrailingConfig,
                        SessionExitConfig,
                        GuardrailsConfig,
                    )

                    # Build PositionManager config from loaded settings
                    pm_cfg = getattr(self.config, 'position_management', {}) or {}
                    breakeven = pm_cfg.get('breakeven', {})
                    trailing = pm_cfg.get('trailing', {})
                    session_exit = pm_cfg.get('session_exit', {})
                    guardrails = pm_cfg.get('guardrails', {})

                    pm_config = PositionManagerConfig(
                        breakeven=BreakEvenConfig(
                            enabled=bool(breakeven.get('enabled', True)),
                            trigger_r_multiple=float(breakeven.get('trigger_r_multiple', 1.0)),
                            buffer_pips=float(breakeven.get('buffer_pips', 0.3)),
                        ),
                        trailing=TrailingConfig(
                            mode=str(trailing.get('mode', 'atr')),
                            atr_period=int(trailing.get('atr_period', 14)),
                            atr_multiplier=float(trailing.get('atr_multiplier', 1.25)),
                            start_after_r_multiple=float(trailing.get('start_after_r_multiple', 1.0)),
                            step_trigger_pips=float(trailing.get('step_trigger_pips', 10.0)),
                            step_distance_pips=float(trailing.get('step_distance_pips', 8.0)),
                        ),
                        session_exit=SessionExitConfig(
                            enabled=bool(session_exit.get('enabled', True)),
                            boundary_minutes=int(session_exit.get('boundary_minutes', 10)),
                            respect_blackouts=bool(session_exit.get('respect_blackouts', True)),
                            min_r_to_keep=float(session_exit.get('min_r_to_keep', 0.5)),
                            tighten_atr_multiplier=float(session_exit.get('tighten_atr_multiplier', 1.0)),
                        ),
                        guardrails=GuardrailsConfig(
                            min_seconds_between_mods=int(guardrails.get('min_seconds_between_mods', 15)),
                            min_improvement_pips=float(guardrails.get('min_improvement_pips', 0.2)),
                            max_spread_pips_for_mods=float(guardrails.get('max_spread_pips_for_mods', 4.0)),
                        ),
                        manage_timeframe=str(pm_cfg.get('manage_timeframe', 'M15')),
                    )

                    self.position_manager = PositionManager(
                        self.mt5_client,
                        self.market_data,
                        self.trade_logger,
                        config=pm_config,
                        session_manager=self.session_manager,
                    )
                    logger.info("Position manager initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize PositionManager: {e}")
            else:
                logger.info("Position manager disabled by configuration; skipping integration")
            
            # Get initial account balance
            account_info = self.mt5_client.get_account_info()
            if account_info:
                self.initial_balance = account_info.balance
                self.peak_equity = account_info.equity
                logger.info(f"Initial account balance: {self.initial_balance}")
            
            logger.info("Trading bot initialized successfully")

            # Warm up M1 history so confirmation has data (live only)
            try:
                import asyncio as _asyncio
                tasks = []
                for sym_cfg in self.config.symbols:
                    tasks.append(self.market_data.fetch_data(sym_cfg['name'], 'M1', max(200, self.config.max_period * 5)))
                if tasks:
                    await _asyncio.gather(*tasks)
                    logger.info("M1 history warmup complete")
            except Exception as _e:
                logger.debug(f"M1 warmup skipped: {_e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def check_drawdown(self) -> bool:
        """Check if maximum drawdown from peak equity has been reached."""
        if self.mt5_client is None:
            return False

        account_info = self.mt5_client.get_account_info()
        if not account_info:
            return False

        # Track peak equity and compute drawdown from peak (safer than from initial balance)
        try:
            current_equity = float(getattr(account_info, 'equity', account_info.balance))
        except Exception:
            current_equity = float(account_info.balance)

        if self.peak_equity is None:
            self.peak_equity = current_equity
        else:
            self.peak_equity = max(self.peak_equity, current_equity)

        if self.peak_equity <= 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= self.config.max_drawdown:
            logger.warning(f"Maximum drawdown reached from peak: {drawdown:.2%}")
            return True

        return False

    async def _maybe_basket_take_profit(self) -> None:
        """If configured and threshold met, close profitable open positions.

        Modes:
        - 'equity': trigger when (equity - balance) >= threshold
        - 'sum':    trigger when sum(position P/L incl. commission+swap) >= threshold
        Threshold can be absolute (threshold_amount) or percent_of_balance.
        """
        cfg = getattr(self.config, 'basket_take_profit', None) or {}
        if not cfg.get('enabled'):
            return

        # Cooldown guard
        try:
            cooldown = int(cfg.get('cooldown_seconds', 60))
        except Exception:
            cooldown = 60
        if self._last_basket_tp_ts is not None:
            elapsed = (datetime.now() - self._last_basket_tp_ts).total_seconds()
            if elapsed < cooldown:
                return

        # Account info for equity/balance and percent threshold
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            return

        # Determine effective threshold
        threshold = float(cfg.get('threshold_amount', 25.0))
        pct = cfg.get('percent_of_balance')
        if pct is not None:
            try:
                threshold = max(threshold, float(pct) * float(account_info.balance))
            except Exception:
                pass

        mode = str(cfg.get('mode', 'equity')).lower()
        current = 0.0
        try:
            if mode == 'sum':
                positions = self.mt5_client.get_all_positions() or []
                total = 0.0
                for p in positions:
                    # Include commission and swap for a more net view
                    total += float(getattr(p, 'profit', 0.0) or 0.0)
                    total += float(getattr(p, 'commission', 0.0) or 0.0)
                    total += float(getattr(p, 'swap', 0.0) or 0.0)
                current = total
            else:
                # equity mode
                current = float(account_info.equity) - float(account_info.balance)
        except Exception:
            return

        if current < threshold:
            return

        # Trigger: close only profitable positions if requested
        positions = self.mt5_client.get_all_positions() or []
        if not positions:
            return

        close_only_profitable = bool(cfg.get('close_only_profitable', True))
        closed = 0
        attempted = 0
        for p in positions:
            try:
                pnl = float(getattr(p, 'profit', 0.0) or 0.0)
                if close_only_profitable and pnl <= 0:
                    continue
                attempted += 1
                res = self.mt5_client.close_position(int(getattr(p, 'ticket')))
                if res is not None:
                    closed += 1
            except Exception:
                # Continue closing others
                continue

        if attempted > 0:
            self._last_basket_tp_ts = datetime.now()
            logger.info(
                f"Basket TP triggered (mode={mode}, value={current:.2f} >= {threshold:.2f}). "
                f"Closed {closed}/{attempted} positions (profitable_only={close_only_profitable})."
            )
    
    def _breakout_is_duplicate(self, symbol, signal, *, window_seconds=900, distance_pips=5) -> bool:
        """True if a breakout for `symbol` was traded recently in the same direction and within `distance_pips`.

        Applies per-symbol overrides when available:
        - duplicate_breakout_window_seconds
        - duplicate_breakout_distance_pips
        """
        try:
            if symbol not in self.consumed_breakouts or not self.consumed_breakouts[symbol]:
                return False

            sym_info = self.mt5_client.get_symbol_info(symbol)
            if not sym_info:
                return False
            pip_size = get_pip_size(sym_info)
            if pip_size <= 0:
                return False

            # Per-symbol overrides
            win_eff = window_seconds
            dist_eff = distance_pips
            try:
                for sc in self.config.symbols:
                    if sc.get('name') == symbol:
                        win_eff = int(sc.get('duplicate_breakout_window_seconds', win_eff))
                        dist_eff = float(sc.get('duplicate_breakout_distance_pips', dist_eff))
                        break
            except Exception:
                pass

            sig_dir = 'bullish' if getattr(signal, 'type', None) == 0 else 'bearish'
            level   = getattr(signal, 'breakout_level', None)
            if level is None:
                return False

            now = datetime.now()
            for lvl, dir_, ts in self.consumed_breakouts[symbol]:
                if (now - ts).total_seconds() < win_eff and dir_ == sig_dir:
                    if abs(level - lvl) / pip_size < dist_eff:
                        logger.info(f"{symbol}: duplicate breakout ({sig_dir}, {dist_eff}p/ {win_eff}s) - skipping.")
                        return True
            return False
        except Exception:
            return False

    def _remember_breakout(self, symbol, signal) -> None:
        """Mark breakout as consumed and persist memory."""
        sig_dir = 'bullish' if getattr(signal, 'type', None) == 0 else 'bearish'
        level   = getattr(signal, 'breakout_level', None)
        if level is None:
            return
        if symbol not in self.consumed_breakouts:
            self.consumed_breakouts[symbol] = []
        self.consumed_breakouts[symbol].append((level, sig_dir, datetime.now()))
        self.save_memory_state()
    
    async def execute_trade(self, signal: TradingSignal, symbol: str):
        """
        Execute trade with live price validation for FOREX
        """
        try:
            # Get symbol info for pip calculations
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return
            
            pip_size = get_pip_size(symbol_info)
            
            # Get current tick
            tick = self.mt5_client.get_symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick for {symbol}")
                return
            
            # Get execution price based on order type
            if signal.type == 0:  # BUY
                execution_price = tick.ask
            else:  # SELL
                execution_price = tick.bid
            
            # CRITICAL: Check price drift from signal (relative to risk)
            price_drift = abs(execution_price - signal.entry_price)
            # Estimate SL distance in pips with current execution price
            provisional_sl_pips = abs(execution_price - signal.stop_loss) / pip_size

            # Apply per-symbol drift policy if configured
            drift_frac = 0.25
            drift_cap = 10.0
            try:
                for sc in self.config.symbols:
                    if sc.get('name') == symbol:
                        drift_frac = float(sc.get('drift_fraction_of_sl', drift_frac))
                        drift_cap = float(sc.get('drift_max_pips', drift_cap))
                        break
            except Exception:
                pass

            # Allow drift up to fraction of SL, bounded by cap and a small floor
            dynamic_max_drift_pips = max(2.0, min(drift_cap, drift_frac * provisional_sl_pips))

            if price_drift > dynamic_max_drift_pips * pip_size:
                logger.warning(
                    f"{symbol}: Price drifted too far\n"
                    f"  Signal: {signal.entry_price:.5f}\n"
                    f"  Current: {execution_price:.5f}\n"
                    f"  Drift: {price_drift/pip_size:.1f} pips (limit {dynamic_max_drift_pips:.1f})\n"
                    f"  TRADE SKIPPED"
                )
                return
            
            # Recalculate position size with ACTUAL entry price
            actual_sl_distance = abs(execution_price - signal.stop_loss)
            actual_sl_pips = actual_sl_distance / pip_size
            
            # Validate minimum stop loss against microstructure floor (broker + spread cushion)
            try:
                point = getattr(symbol_info, 'point', 0.0) or 0.0
                pip_points = (pip_size / point) if point and point > 0 else 0.0
                stops_level_points = getattr(symbol_info, 'trade_stops_level', 0) or 0
                stops_level_pips = (stops_level_points / pip_points) if pip_points > 0 else 0.0
            except Exception:
                stops_level_pips = 0.0

            spread_pips = ((tick.ask - tick.bid) / pip_size) if pip_size else 0.0

            # Resolve per-symbol spread floor multiplier
            spread_floor_mult = float(getattr(self.config, 'spread_floor_multiplier', 1.1))
            try:
                for sc in self.config.symbols:
                    if sc.get('name') == symbol:
                        spread_floor_mult = float(sc.get('spread_floor_multiplier', spread_floor_mult))
                        break
            except Exception:
                pass

            micro_floor_pips = max(stops_level_pips, spread_pips * spread_floor_mult)
            if actual_sl_pips < micro_floor_pips:
                logger.warning(
                    f"{symbol}: SL too tight after drift ({actual_sl_pips:.1f} pips < floor {micro_floor_pips:.1f} pips) - skipping"
                )
                return
            
            # Calculate base position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=actual_sl_pips
            )

            # Correlation-aware size reduction (soft)
            if position_size > 0:
                try:
                    position_size = self.risk_manager.adjust_position_size_for_correlation(
                        symbol=symbol,
                        base_position_size=position_size,
                        overrides=None,
                    )
                except Exception:
                    pass
            
            if position_size <= 0:
                logger.warning(f"{symbol}: Invalid position size calculated")
                return
            
            # Final validation
            if not self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                volume=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_type=signal.type,
            ):
                logger.warning(f"{symbol}: Trade parameters validation failed")
                return
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits(symbol, signal.type):
                logger.info(f"{symbol}: Risk limits prevent trading")
                return
            
            # Resolve per-symbol or global order deviation (points)
            deviation_points = None
            try:
                deviation_points = int(getattr(self.config, 'order_deviation_points', 20))
                for sc in self.config.symbols:
                    if sc.get('name') == symbol and sc.get('order_deviation_points') is not None:
                        deviation_points = int(sc.get('order_deviation_points'))
                        break
            except Exception:
                deviation_points = 20

            # EXECUTE THE TRADE
            result = self.mt5_client.place_order(
                symbol=symbol,
                order_type=signal.type,
                volume=position_size,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"LIVE_{signal.reason}",
                deviation=deviation_points,
            )
            
            if result and result.retcode == self.mt5_client.mt5.TRADE_RETCODE_DONE:
                # Log successful trade
                trade_details = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'order_type': 'BUY' if signal.type == 0 else 'SELL',
                    'entry_price': result.price,
                    'signal_price': signal.entry_price,  # Track signal vs execution
                    'volume': position_size,
                    'stop_loss': signal.stop_loss,
                    'status': 'OPEN',
                    'order_id': getattr(result, 'order', None),
                    'take_profit': signal.take_profit,
                    'ticket': result.order,
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'drift_pips': price_drift / pip_size,
                    # Diagnostics for adaptive SL analysis
                    'stop_loss_pips': actual_sl_pips,
                    'pip_size': pip_size,
                    'spread_pips': spread_pips,
                    'broker_stops_level_pips': stops_level_pips,
                    'atr_pips': getattr(signal, 'atr_pips', None),
                    'sl_basis': getattr(signal, 'sl_basis', None),
                }
                
                self.trade_logger.log_trade(trade_details)
                
                logger.info(
                    f"*** TRADE EXECUTED ***\n"
                    f"  Symbol: {symbol}\n"
                    f"  Type: {'BUY' if signal.type == 0 else 'SELL'}\n"
                    f"  Executed: {result.price:.5f}\n"
                    f"  Signal was: {signal.entry_price:.5f}\n"
                    f"  Drift: {price_drift/pip_size:.1f} pips\n"
                    f"  Volume: {position_size:.2f}\n"
                    f"  Risk: {actual_sl_pips:.1f} pips\n"
                    f"  Ticket: {result.order}"
                )
                return result
            else:
                error_msg = getattr(result, 'comment', 'Unknown error') if result else 'No result'
                logger.error(f"{symbol}: Trade execution failed - {error_msg}")
                return None
            
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}", exc_info=True)
            return None
                
    async def process_symbol(self, symbol_config: Dict):
        """
        Process a single symbol for trading opportunities
        Multi-timeframe coordination with trend alignment
        """
        symbol = symbol_config['name']
        
        # Early check: if we already have a position for this symbol, skip all calculations
        existing_positions = self.mt5_client.get_positions(symbol)
        if existing_positions and len(existing_positions) >= 1:
            logger.debug(f"Position already exists for {symbol}, skipping signal generation")
            return
        
        # Check if symbol is in cooldown with IMPROVED logic (different times for TP/SL)
        if symbol in self.recent_closures:
            closure_time, exit_price, status = self.recent_closures[symbol]
            time_since_close = (datetime.now() - closure_time).total_seconds()
            
            # Different cooldowns based on how trade closed
            if status == 'CLOSED_TP':
                cooldown_period = 900  # 15 minutes after TP (market digesting the move)
            elif status == 'CLOSED_SL':
                cooldown_period = 300  # 5 minutes after SL (failed move, can retry sooner)
            else:  # CLOSED_MANUAL or CLOSED_UNKNOWN
                cooldown_period = 600  # 10 minutes default
            
            if time_since_close < cooldown_period:
                logger.info(f"{symbol} in cooldown for {cooldown_period-time_since_close:.0f}s more (closed {status} {time_since_close/60:.1f} min ago)")
                return
        
        # NEW: Check consumed breakouts BEFORE signal generation to save computation
        if symbol in self.consumed_breakouts:
            current_time = datetime.now()
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if symbol_info:
                pip_size = get_pip_size(symbol_info)
                
                # Get current price for comparison
                tick = self.mt5_client.get_symbol_info_tick(symbol)
                if tick:
                    # Clean old entries and check if current price is near recently traded levels
                    active_breakouts = []
                    for level, direction, timestamp in self.consumed_breakouts[symbol]:
                        age = (current_time - timestamp).total_seconds()
                        
                        if age < 1800:  # Keep for 30 minutes
                            active_breakouts.append((level, direction, timestamp))
                            
                            # Check if we're still near this consumed breakout level
                            if age < 900:  # Block re-entry for 15 minutes
                                current_price = tick.bid if direction == 'bearish' else tick.ask
                                distance_pips = abs(current_price - level) / pip_size
                                
                                # If price is still near the consumed level, skip
                                if distance_pips < 20:  # Within 20 pips of consumed level
                                    logger.info(
                                        f"{symbol}: Already traded {direction} breakout at {level:.5f} "
                                        f"({age/60:.1f} min ago). Current price {current_price:.5f} still near level. Skipping."
                                    )
                                    return
                    
                    # Update with only non-expired breakouts
                    self.consumed_breakouts[symbol] = active_breakouts
        
        try:
            # Fetch multi-timeframe data
            mtf_data = await self.market_data.fetch_multi_timeframe_data(symbol)
            
            if not mtf_data:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Use H1 for trend, M15 for signal timing
            if 'H1' in mtf_data and 'M15' in mtf_data:
                # Get H1 trend direction using existing function
                h1_trend = self.market_data.identify_trend(mtf_data['H1'])
                logger.debug(f"{symbol} H1 trend: {h1_trend}")
                
                # Generate M15 signal with trend context
                m15_signal = self.strategy.generate_signal(mtf_data['M15'], symbol, trend=h1_trend)
                
                if m15_signal:
                    if self._breakout_is_duplicate(symbol, m15_signal):
                        return
                    
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if m15_signal.type == 0 else 'SELL'}, "
                        f"Confidence={m15_signal.confidence:.2f}, "
                        f"H1 Trend={h1_trend}"
                    )
                    
                    # Check if within trading session
                    phase = self.session_manager.get_phase()
                    if self.session_manager.is_trade_window():
                        result = await self.execute_trade(m15_signal, symbol)
                        
                        if result:  # If trade was successfully opened
                            # Track this breakout as consumed
                            self._remember_breakout(symbol, m15_signal)
                    else:
                        logger.info(f"Outside trade window, current phase={phase.name}, session={phase.session}")
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            # FALLBACK: Single timeframe processing (if only one timeframe available)
            elif 'M15' in mtf_data:
                # Use M15 data for both trend and signal
                m15_data = mtf_data['M15']
                m15_trend = self.market_data.identify_trend(m15_data)
                
                signal = self.strategy.generate_signal(m15_data, symbol, trend=m15_trend)
                
                if signal:
                    # Check if this breakout level is already consumed (same as above)
                    if self._breakout_is_duplicate(symbol, signal):
                        return
                    
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )
                    
                    phase = self.session_manager.get_phase()
                    if self.session_manager.is_trade_window():
                        result = await self.execute_trade(signal, symbol)
                        
                        if result:
                            # Track consumed breakout
                            self._remember_breakout(symbol, signal)
                    else:
                        logger.info(f"Outside trade window, current phase={phase.name}, session={phase.session}")
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            elif 'H1' in mtf_data:
                # Use H1 data for both trend and signal
                h1_data = mtf_data['H1']
                h1_trend = self.market_data.identify_trend(h1_data)
                
                signal = self.strategy.generate_signal(h1_data, symbol, trend=h1_trend)
                
                if signal:
                    # Check consumed breakouts (same logic)
                    if self._breakout_is_duplicate(symbol, signal):
                        return
                    
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )
                    
                    phase = self.session_manager.get_phase()
                    if self.session_manager.is_trade_window():
                        result = await self.execute_trade(signal, symbol)
                        
                        if result:
                            self._remember_breakout(symbol, signal)
                    else:
                        logger.info(f"Outside trade window, current phase={phase.name}, session={phase.session}")
                else:
                    logger.debug(f"No signal generated for {symbol}")
                        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
    async def sync_trade_status(self):
        """Trade status synchronization (single-pass, position-scoped)."""
        if not self.trade_logger:
            return

        try:
            open_trades = [t for t in self.trade_logger.trades if t.get('status') == 'OPEN']
            if not open_trades:
                return

            # Live positions right now
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {pos.ticket for pos in current_positions}

            for trade in open_trades:
                ticket = trade.get('ticket')
                if not ticket:
                    continue

                # If the ticket is still live, skip
                if ticket in current_tickets:
                    continue

                # Position was closed — reconcile from deals-by-position (authoritative)
                logger.info(f"Position {ticket} closed, fetching details...")

                # Be tolerant to either datetime or ISO string in our own log
                ts = trade.get('timestamp')
                start_time = ts if isinstance(ts, datetime) else None  # not needed for by-position query

                deals = self.mt5_client.get_history_deals_by_position(ticket) or []
                if not deals:
                    # No deals found — keep as unknown but do not crash
                    logger.warning(f"No deal history found for position {ticket}. Marking CLOSED_UNKNOWN.")
                    self.trade_logger.update_trade(
                        ticket=ticket,
                        exit_price=None,
                        profit=0.0,
                        status='CLOSED_UNKNOWN'
                    )
                    continue

                # Out-deals (DEAL_ENTRY_OUT == 1)
                out_deals = [d for d in deals if getattr(d, 'entry', None) == 1]

                exit_price = None
                profit = 0.0
                status = 'CLOSED_UNKNOWN'

                if out_deals:
                    # Use the last closing deal by time
                    def _deal_time(d):
                        return getattr(d, 'time_msc', getattr(d, 'time', 0))
                    last = max(out_deals, key=_deal_time)

                    exit_price = getattr(last, 'price', None)

                    # Sum realized PnL across out deals (include commission/swap like you did)
                    for d in out_deals:
                        profit += float(getattr(d, 'profit', 0) + getattr(d, 'commission', 0) + getattr(d, 'swap', 0))

                    reason = getattr(last, 'reason', 0)
                    if reason == 4:
                        status = 'CLOSED_SL'
                    elif reason == 5:
                        status = 'CLOSED_TP'
                    else:
                        status = 'CLOSED_MANUAL'
                else:
                    # No explicit out deal: keep UNKNOWN and try a soft inference vs SL/TP
                    # (rare edge cases: partials or netting transfers)
                    exit_price = getattr(deals[-1], 'price', None) if deals else None

                # If still unknown but we have an exit price, infer near SL/TP with a tolerance
                if status == 'CLOSED_UNKNOWN' and exit_price:
                    sym_info = self.mt5_client.get_symbol_info(trade['symbol'])
                    pip_size = get_pip_size(sym_info) if sym_info else 0.0001
                    sl = trade.get('stop_loss')
                    tp = trade.get('take_profit')
                    if sl and abs(exit_price - sl) < 5 * pip_size:
                        status = 'CLOSED_SL'
                    elif tp and abs(exit_price - tp) < 5 * pip_size:
                        status = 'CLOSED_TP'
                    else:
                        status = 'CLOSED_MANUAL'

                # Persist closure
                self.trade_logger.update_trade(
                    ticket=ticket,
                    exit_price=exit_price,
                    profit=profit,
                    status=status
                )

                # Remember recent closure for cooldowns
                symbol = trade.get('symbol')
                if symbol:
                    self.recent_closures[symbol] = (datetime.now(), exit_price, status)
                    self.save_memory_state()

                logger.info(f"Recorded {trade.get('symbol')} closure at {exit_price} ({status})")

        except Exception as e:
            logger.error(f"Error syncing trade status: {e}", exc_info=True)
            
    def save_memory_state(self):
        """Save consumed breakouts and recent closures to file"""
        memory_state = {
            'consumed_breakouts': {},
            'recent_closures': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Convert consumed breakouts to JSON-serializable format
        for symbol, breakouts in self.consumed_breakouts.items():
            memory_state['consumed_breakouts'][symbol] = [
                {
                    'level': level,
                    'direction': direction,
                    'timestamp': timestamp.isoformat()
                }
                for level, direction, timestamp in breakouts
            ]
        
        # Convert recent closures to JSON-serializable format
        for symbol, (timestamp, exit_price, status) in self.recent_closures.items():
            memory_state['recent_closures'][symbol] = {
                'timestamp': timestamp.isoformat(),
                'exit_price': exit_price,
                'status': status
            }
        
        # Save to file
        with open('bot_memory.json', 'w') as f:
            json.dump(memory_state, f, indent=2)
        
        logger.debug("Memory state saved to bot_memory.json")

    def load_memory_state(self):
        """Load consumed breakouts and recent closures from file"""
        if not os.path.exists('bot_memory.json'):
            logger.info("No memory state file found, starting fresh")
            return
        
        try:
            with open('bot_memory.json', 'r') as f:
                content = f.read()
                
                # Check if file is empty or just whitespace
                if not content.strip():
                    logger.info("Memory state file is empty, starting fresh")
                    return
                    
                memory_state = json.loads(content)
            
            current_time = datetime.now()
            
            # Load consumed breakouts (only those less than 30 minutes old)
            for symbol, breakouts in memory_state.get('consumed_breakouts', {}).items():
                self.consumed_breakouts[symbol] = []
                for b in breakouts:
                    timestamp = datetime.fromisoformat(b['timestamp'])
                    age_seconds = (current_time - timestamp).total_seconds()
                    
                    # Only load if less than 30 minutes old
                    if age_seconds < 1800:
                        self.consumed_breakouts[symbol].append(
                            (b['level'], b['direction'], timestamp)
                        )
                        logger.info(f"Loaded consumed breakout: {symbol} {b['direction']} "
                                f"at {b['level']} ({age_seconds/60:.1f} min old)")
            
            # Load recent closures (only those less than 5 minutes old)
            for symbol, closure in memory_state.get('recent_closures', {}).items():
                timestamp = datetime.fromisoformat(closure['timestamp'])
                age_seconds = (current_time - timestamp).total_seconds()
                
                # Only load if less than 5 minutes old
                if age_seconds < 300:
                    self.recent_closures[symbol] = (
                        timestamp,
                        closure['exit_price'],
                        closure['status']
                    )
                    logger.info(f"Loaded recent closure: {symbol} at {closure['exit_price']} "
                            f"({age_seconds:.0f}s ago)")
            
            logger.info(f"Memory state loaded: {len(self.consumed_breakouts)} symbols with consumed breakouts, "
                    f"{len(self.recent_closures)} recent closures")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in memory state file: {e}")
            logger.info("Starting with fresh memory state")
            # Don't clear the dictionaries - they're already initialized
        except Exception as e:
            logger.error(f"Error loading memory state: {e}")
            # Don't clear the dictionaries - they're already initialized
        
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting main trading loop")
        
        while self.running:
            try:
                # Check for maximum drawdown
                if await self.check_drawdown():
                    logger.error("Maximum drawdown reached, stopping bot")
                    break
                
                # Sync trade status with MT5
                await self.sync_trade_status()

                # Basket take-profit check (optional)
                try:
                    await self._maybe_basket_take_profit()
                except Exception as e:
                    logger.debug(f"Basket TP check skipped: {e}")

                # Manage open positions (session-aware protections, BE/trailing/partials)
                if self.position_manager:
                    try:
                        await self.position_manager.manage_open_positions()
                    except Exception as e:
                        logger.debug(f"Position management skipped: {e}")
                # Process each symbol
                for symbol_config in self.config.symbols:
                    await self.process_symbol(symbol_config)
                
                # Wait for next iteration
                await asyncio.sleep(self.config.main_loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(self.config.main_loop_interval)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot")
        self.running = False
        
        self.save_memory_state()
        
        # Positions should be managed according to their stop loss and take profit
        if self.mt5_client:
            positions = self.mt5_client.get_all_positions()
            if positions:
                logger.info(f"Note: {len(positions)} positions remain open and will be managed by their SL/TP")
                for position in positions:
                    logger.info(f"Open position: {position.symbol} - Volume: {position.volume} - Profit: {position.profit}")
        
        logger.info("Trading bot shutdown complete")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Forex Trading Bot with Pure Price Action Strategy')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--risk-per-trade', type=float,
                       help='Override risk percentage per trade')
    parser.add_argument('--symbol', type=str,
                       help='Override trading symbol')
    parser.add_argument('--timeframe', type=str,
                       help='Override primary timeframe')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = Config(args.config, args)
    
    # Create and run bot
    bot = TradingBot(config)
    
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
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
