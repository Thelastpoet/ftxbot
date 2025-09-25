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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import sys

from strategy import TradingSignal
from utils import get_pip_size, is_trading_paused
from symbol_runtime import SymbolRuntimeContext, load_symbol_profile

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
            self.enable_news_filter = trading.get('enable_news_filter', True)
            
            # Risk management
            risk = config_data.get('risk_management', {})
            self.risk_per_trade = risk.get('risk_per_trade', 0.02)
            self.fixed_lot_size = risk.get('fixed_lot_size', None)
            self.max_drawdown = risk.get('max_drawdown_percentage', 0.1)
            self.risk_reward_ratio = risk.get('risk_reward_ratio', 2.0)
            self.correlation_threshold = risk.get('correlation_threshold', 0.7)
            self.correlation_lookback_period = risk.get('correlation_lookback_period', 30)
            self.risk_management_settings = risk
            
            # Symbols and timeframes
            symbols_data = config_data.get('symbols', [])
            self.symbols = []
            for sym in symbols_data:
                self.symbols.append({
                    'name': sym['name'],
                    'timeframes': sym.get('timeframes', ['M15'])
                })
            
            # Trading sessions
            sessions_data = config_data.get('trading_sessions', [])
            self.trading_sessions = []
            for session in sessions_data:
                self.trading_sessions.append({
                    'name': session['name'],
                    'start_time': datetime.strptime(session['start_time'], '%H:%M').time(),
                    'end_time': datetime.strptime(session['end_time'], '%H:%M').time()
                })
            
            paths_cfg = config_data.get('paths', {})
            base_dir = self.config_path.parent
            self.symbol_config_dir = (base_dir / paths_cfg.get('symbol_config_dir', 'symbol_configs')).resolve()
            self.calibrator_state_dir = (base_dir / paths_cfg.get('calibrator_state_dir', 'calibrator_states')).resolve()
            self.optimizer_state_dir = (base_dir / paths_cfg.get('optimizer_state_dir', 'optimizer_states')).resolve()
            self.calibration_settings = dict(config_data.get('calibration', {}))

            # Apply command-line overrides
            if self.args:
                if self.args.risk_per_trade:
                    self.risk_per_trade = self.args.risk_per_trade
                if self.args.symbol:
                    self.symbols = [{'name': self.args.symbol, 'timeframes': ['M15']}]
                if self.args.timeframe:
                    for sym in self.symbols:
                        sym['timeframes'] = [self.args.timeframe]
                        
            if isinstance(getattr(self, 'risk_management_settings', None), dict):
                self.risk_management_settings.update({
                    'risk_per_trade': self.risk_per_trade,
                    'fixed_lot_size': self.fixed_lot_size,
                    'max_drawdown_percentage': self.max_drawdown,
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'correlation_threshold': self.correlation_threshold,
                    'correlation_lookback_period': self.correlation_lookback_period,
                })

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
        self.correlation_threshold = 0.7
        self.correlation_lookback_period = 30
        self.risk_management_settings = {
            'risk_per_trade': self.risk_per_trade,
            'fixed_lot_size': self.fixed_lot_size,
            'max_drawdown_percentage': self.max_drawdown,
            'risk_reward_ratio': self.risk_reward_ratio,
            'correlation_threshold': self.correlation_threshold,
            'correlation_lookback_period': self.correlation_lookback_period,
        }
        self.symbols = [{'name': 'EURUSD', 'timeframes': ['M15']}]
        self.trading_sessions = []
        self.min_stop_loss_pips = 20
        self.stop_loss_buffer_pips = 15
        self.min_range_pips = 10
        self.atr_period = 14
        self.atr_tp_multiplier = 4.0
        self.session_based_tp = True
        self.volatility_adjustment = True
        self.enable_news_filter = True

        base_dir = self.config_path.parent
        self.symbol_config_dir = (base_dir / 'symbol_configs').resolve()
        self.calibrator_state_dir = (base_dir / 'calibrator_states').resolve()
        self.optimizer_state_dir = (base_dir / 'optimizer_states').resolve()
        self.calibration_settings = {}

class TradingSession:
    """Manages trading session times"""
    
    def __init__(self, config: Config):
        self.config = config
        # Log session info on creation for clarity
        try:
            self.log_session_info()
        except Exception:
            pass
        
    def is_trading_time(self) -> bool:
        """Check if current time is within any defined trading session"""
        if not self.config.trading_sessions:
            return True  # Trade 24/7 if no sessions defined
        
        current_time = datetime.now().time()
        
        for session in self.config.trading_sessions:
            start = session['start_time']
            end = session['end_time']
            
            # Handle sessions that cross midnight
            if start <= end:
                if start <= current_time <= end:
                    logger.debug(f"Within {session['name']} session")
                    return True
            else:
                if current_time >= start or current_time <= end:
                    logger.debug(f"Within {session['name']} session")
                    return True
        
        return False

    def log_session_info(self):
        """Log configured sessions and local time zone info"""
        try:
            now_local = datetime.now().astimezone()
            tzname = now_local.tzname()
            offset = now_local.utcoffset()
            logger.info(
                f"Session timezone: {tzname} (UTC offset {offset}) | Local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if not self.config.trading_sessions:
                logger.info("No trading sessions configured (24/7 mode)")
                return
            for sess in self.config.trading_sessions:
                logger.info(
                    f"Session configured: {sess['name']} {sess['start_time']}â€“{sess['end_time']} (interpreted in local time)"
                )
        except Exception as e:
            logger.debug(f"Failed to log session info: {e}")

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.trade_logger = None
        self.consumed_breakouts = {}
        self.recent_closures = {} 
        self.symbol_contexts: Dict[str, SymbolRuntimeContext] = {}
        self.load_memory_state()
        self.session_manager = TradingSession(config)
        self.running = False
        self.initial_balance = None
        
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
            # Disable M1 confirmation for now
            try:
                self.strategy.m1_confirmation_enabled = False
            except Exception:
                pass
            self.risk_manager = RiskManager(self.config.risk_management_settings, self.mt5_client)
            from pathlib import Path as _P; self.trade_logger = TradeLogger(str((_P(__file__).parent / 'trades.log').resolve()))

            self._init_symbol_contexts()
            
            # Get initial account balance
            account_info = self.mt5_client.get_account_info()
            if account_info:
                self.initial_balance = account_info.balance
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
    
    def _init_symbol_contexts(self) -> None:
        """Initialize per-symbol runtime contexts."""
        if not self.strategy:
            return

        symbol_config_dir = Path(self.config.symbol_config_dir)
        calibrator_state_dir = Path(self.config.calibrator_state_dir)
        optimizer_state_dir = Path(self.config.optimizer_state_dir)

        symbol_config_dir.mkdir(parents=True, exist_ok=True)
        calibrator_state_dir.mkdir(parents=True, exist_ok=True)
        optimizer_state_dir.mkdir(parents=True, exist_ok=True)

        base_defaults = {
            'min_stop_loss_pips': float(getattr(self.strategy, 'min_stop_loss_pips', getattr(self.config, 'min_stop_loss_pips', 20))),
            'stop_loss_buffer_pips': float(getattr(self.strategy, 'stop_loss_buffer_pips', getattr(self.config, 'stop_loss_buffer_pips', 10))),
            'risk_reward_ratio': float(getattr(self.config, 'risk_reward_ratio', 1.5)),
            'min_confidence': float(getattr(self.strategy, 'min_confidence', 0.5)),
        }

        self.symbol_contexts.clear()
        for sym_cfg in self.config.symbols:
            symbol = sym_cfg['name']
            profile_path = symbol_config_dir / f"{symbol}.json"
            profile = load_symbol_profile(symbol, dict(base_defaults), profile_path)
            trade_history = self._closed_trades_for_symbol(symbol)
            context = SymbolRuntimeContext(
                profile=profile,
                calibration_cfg=self.config.calibration_settings,
                calibrator_state_dir=calibrator_state_dir,
                optimizer_state_dir=optimizer_state_dir,
                trade_history=trade_history,
            )
            self.symbol_contexts[symbol] = context

    def _closed_trades_for_symbol(self, symbol: str) -> List[Dict]:
        if not self.trade_logger:
            return []
        results: List[Dict] = []
        for trade in self.trade_logger.get_all_trades():
            if trade.get('symbol') != symbol:
                continue
            status = str(trade.get('status', '')).upper()
            if status.startswith('CLOSED'):
                results.append(trade)
        return results

    def _is_recent_breakout_duplicate(self, symbol: str, signal: TradingSignal, pip_size: Optional[float]) -> bool:
        if symbol not in self.consumed_breakouts or not pip_size:
            return False
        now = datetime.now()
        filtered = []
        duplicate = False
        for level, direction, timestamp in self.consumed_breakouts.get(symbol, []):
            age = (now - timestamp).total_seconds()
            if age < 1800:
                filtered.append((level, direction, timestamp))
            if duplicate:
                continue
            if age < 900:
                signal_direction = 'bullish' if signal.type == 0 else 'bearish'
                if direction == signal_direction:
                    level_distance_pips = abs(signal.breakout_level - level) / pip_size
                    if level_distance_pips < 5:
                        logger.info(
                            f"{symbol}: Breakout at {level:.5f} already traded {age/60:.1f} min ago; skipping duplicate."
                        )
                        duplicate = True
        self.consumed_breakouts[symbol] = filtered
        return duplicate

    async def _handle_generated_signal(
        self,
        symbol: str,
        signal: TradingSignal,
        context: SymbolRuntimeContext,
        symbol_params: Dict[str, float],
        risk_overrides: Dict[str, float],
        pip_size: Optional[float],
        trend_context: Optional[str] = None,
    ) -> None:
        if not signal:
            return
        risk_overrides = risk_overrides or {}
        if self._is_recent_breakout_duplicate(symbol, signal, pip_size):
            return
        rr_for_threshold = None
        if isinstance(signal.parameters, dict):
            rr_for_threshold = signal.parameters.get('risk_reward_ratio')
        if rr_for_threshold is None:
            rr_for_threshold = float(symbol_params.get('risk_reward_ratio', getattr(self.config, 'risk_reward_ratio', 1.5)))
        gating = context.evaluate_signal(signal.features, rr_for_threshold)
        if not gating.get('accepted', True):
            logger.info(
                f"{symbol}: Calibrator vetoed signal (p={gating.get('probability', 0.0):.2f} < thr={gating.get('threshold', 0.0):.2f})"
            )
            return
        merged_params = dict(symbol_params)
        if isinstance(signal.parameters, dict):
            merged_params.update(signal.parameters)
        merged_params['calibrator'] = gating
        signal.parameters = merged_params
        if not isinstance(signal.features, dict):
            signal.features = {}
        trend_suffix = f", {trend_context}" if trend_context else ''
        logger.info(
            f"Signal generated for {symbol}: Type={'BUY' if signal.type == 0 else 'SELL'}, "
            f"Confidence={signal.confidence:.2f}{trend_suffix}, "
            f"Calib={gating.get('probability', 0.0):.2f}/{gating.get('threshold', 0.0):.2f}"
        )
        if not self.session_manager.is_trading_time():
            logger.info("Outside trading session, signal ignored")
            return
        result = await self.execute_trade(signal, symbol, risk_overrides)
        if result:
            if symbol not in self.consumed_breakouts:
                self.consumed_breakouts[symbol] = []
            now = datetime.now()
            direction = 'bullish' if signal.type == 0 else 'bearish'
            self.consumed_breakouts[symbol].append((signal.breakout_level, direction, now))
            self.consumed_breakouts[symbol] = [
                (lvl, dirn, ts) for lvl, dirn, ts in self.consumed_breakouts[symbol]
                if (now - ts).total_seconds() < 1800
            ]
            self.save_memory_state()

    async def check_drawdown(self) -> bool:
        """Check if maximum drawdown has been reached"""
        if not self.initial_balance:
            return False
        
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            return False
        
        current_balance = account_info.balance
        drawdown = (self.initial_balance - current_balance) / self.initial_balance
        
        if drawdown >= self.config.max_drawdown:
            logger.warning(f"Maximum drawdown reached: {drawdown:.2%}")
            return True
        
        return False
    
    async def execute_trade(self, signal: TradingSignal, symbol: str, risk_overrides: Optional[Dict[str, float]] = None):
        """
        Execute trade with live price validation for FOREX
        """
        risk_overrides = risk_overrides or {}
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
            # Allow drift up to 25% of SL, bounded [2, 10] pips
            dynamic_max_drift_pips = max(2.0, min(10.0, 0.25 * provisional_sl_pips))

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
            
            # Validate minimum stop loss
            min_sl_floor = self.config.min_stop_loss_pips
            if isinstance(signal.parameters, dict):
                try:
                    min_sl_floor = float(signal.parameters.get('min_stop_loss_pips', min_sl_floor))
                except (TypeError, ValueError):
                    pass
            if actual_sl_pips < min_sl_floor:
                logger.warning(
                    f"{symbol}: SL too tight after drift ({actual_sl_pips:.1f} pips < {min_sl_floor:.1f}) - skipping"
                )
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=actual_sl_pips,
                overrides=risk_overrides
            )
            
            if position_size <= 0:
                logger.warning(f"{symbol}: Invalid position size calculated")
                return
            
            # Final validation
            if not self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                volume=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            ):
                logger.warning(f"{symbol}: Trade parameters validation failed")
                return
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits(symbol, signal.type, overrides=risk_overrides):
                logger.info(f"{symbol}: Risk limits prevent trading")
                return
            
            # EXECUTE THE TRADE
            result = self.mt5_client.place_order(
                symbol=symbol,
                order_type=signal.type,
                volume=position_size,
                sl=signal.stop_loss,
                tp=signal.take_profit,
                comment=f"LIVE_{signal.reason}"
            )
            
            if result and result.retcode == self.mt5_client.mt5.TRADE_RETCODE_DONE:
                # Log successful trade
                parameter_snapshot = dict(signal.parameters) if isinstance(signal.parameters, dict) else {}
                calibrator_meta = parameter_snapshot.pop('calibrator', None)
                features_snapshot = signal.features if isinstance(signal.features, dict) else {}
                if calibrator_meta is not None:
                    calibrator_meta = dict(calibrator_meta)
                    if 'features' not in calibrator_meta and features_snapshot:
                        calibrator_meta['features'] = features_snapshot
                elif features_snapshot:
                    calibrator_meta = {'features': features_snapshot}

                
                # Try to capture the live position ticket for robust closure tracking
                try:
                    pos_list = self.mt5_client.get_positions(symbol)
                    if pos_list and len(pos_list) >= 1:
                        pos_id = None
                        for pos in pos_list:
                            if getattr(pos, 'type', None) == signal.type:
                                pos_id = getattr(pos, 'ticket', None)
                                break
                        if pos_id is None:
                            pos_id = getattr(pos_list[0], 'ticket', None)
                        if pos_id is not None:
                            trade_details['position_ticket'] = int(pos_id)
                except Exception:
                    pass

                trade_details = {
                    'timestamp': datetime.now(),

                    'symbol': symbol,
                    'order_type': 'BUY' if signal.type == 0 else 'SELL',
                    'entry_price': result.price,
                    'signal_price': signal.entry_price,  # Track signal vs execution
                    'volume': position_size,
                    'stop_loss': signal.stop_loss,
                    'stop_loss_pips': signal.stop_loss_pips,
                    'take_profit': signal.take_profit,
                    'ticket': result.order,
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'drift_pips': price_drift / pip_size,
                    'pip_size': pip_size,
                }

                if parameter_snapshot:
                    trade_details['parameters'] = parameter_snapshot
                if calibrator_meta:
                    trade_details['calibrator'] = calibrator_meta
                elif features_snapshot:
                    trade_details['features'] = features_snapshot

                applied_risk = risk_overrides.get('risk_per_trade', self.risk_manager.risk_per_trade)
                try:
                    trade_details['risk_per_trade'] = float(applied_risk) if applied_risk is not None else None
                except (TypeError, ValueError):
                    trade_details['risk_per_trade'] = applied_risk
                if risk_overrides:
                    trade_details['risk_overrides'] = dict(risk_overrides)

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

        context = self.symbol_contexts.get(symbol) if hasattr(self, 'symbol_contexts') else None
        if context is None:
            logger.debug(f"{symbol}: No runtime context available, skipping")
            return

        symbol_params = context.get_parameters()
        risk_overrides = context.get_risk_overrides()
        if hasattr(self.strategy, 'set_symbol_params'):
            try:
                self.strategy.set_symbol_params(symbol, symbol_params)
            except Exception:
                pass

        # Early check: if we already have a position for this symbol, skip all calculations
        existing_positions = self.mt5_client.get_positions(symbol)
        if existing_positions and len(existing_positions) >= 1:
            logger.debug(f"Position already exists for {symbol}, skipping signal generation")
            return

        symbol_info = self.mt5_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {symbol}")
            return
        pip_size = get_pip_size(symbol_info)

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
                    await self._handle_generated_signal(
                        symbol,
                        m15_signal,
                        context,
                        symbol_params,
                        risk_overrides,
                        pip_size,
                        trend_context=f"H1 Trend={h1_trend}",
                    )
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            # FALLBACK: Single timeframe processing (if only one timeframe available)
            elif 'M15' in mtf_data:
                # Use M15 data for both trend and signal
                m15_data = mtf_data['M15']
                m15_trend = self.market_data.identify_trend(m15_data)
                
                signal = self.strategy.generate_signal(m15_data, symbol, trend=m15_trend)
                
                if signal:
                    await self._handle_generated_signal(
                        symbol,
                        signal,
                        context,
                        symbol_params,
                        risk_overrides,
                        pip_size,
                    )
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            elif 'H1' in mtf_data:
                # Use H1 data for both trend and signal
                h1_data = mtf_data['H1']
                h1_trend = self.market_data.identify_trend(h1_data)
                
                signal = self.strategy.generate_signal(h1_data, symbol, trend=h1_trend)
                
                if signal:
                    # Check consumed breakouts (same logic)
                    if symbol in self.consumed_breakouts:
                        symbol_info = self.mt5_client.get_symbol_info(symbol)
                        if symbol_info:
                            pip_size = get_pip_size(symbol_info)
                            for level, direction, timestamp in self.consumed_breakouts[symbol]:
                                age = (datetime.now() - timestamp).total_seconds()
                                if age < 900:
                                    signal_direction = 'bullish' if signal.type == 0 else 'bearish'
                                    if direction == signal_direction:
                                        level_distance_pips = abs(signal.breakout_level - level) / pip_size
                                        if level_distance_pips < 5:
                                            logger.info(f"{symbol}: Duplicate breakout detected, skipping.")
                                            return
                    
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )
                    
                    if self.session_manager.is_trading_time():
                        result = await self.execute_trade(signal, symbol, risk_overrides)
                        
                        if result:
                            if symbol not in self.consumed_breakouts:
                                self.consumed_breakouts[symbol] = []
                            
                            direction = 'bullish' if signal.type == 0 else 'bearish'
                            self.consumed_breakouts[symbol].append(
                                (signal.breakout_level, direction, datetime.now())
                            )
                            
                            self.save_memory_state()
                    else:
                        logger.info("Outside trading session, signal ignored")
                else:
                    logger.debug(f"No signal generated for {symbol}")
                        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
    async def sync_trade_status(self):
        """Trade status synchronization"""
        if not self.trade_logger:
            return
            
        try:
            open_trades = [t for t in self.trade_logger.get_all_trades() 
                        if t.get('status') == 'OPEN']
            
            if not open_trades:
                return
                
            # Get current positions
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {pos.ticket for pos in current_positions}
            
            for trade in open_trades:
                ticket = trade.get('position_ticket') or trade.get('ticket')
                if ticket and ticket not in current_tickets:
                    # Position was closed
                    logger.info(f"Position {ticket} closed, fetching details...")
                    
                    # Get the closing deal information
                    end_time = datetime.now()
                    start_time = datetime.fromisoformat(trade['timestamp'])
                    
                    # Get history of deals
                    deals = self.mt5_client.get_history_deals(start_time, end_time)
                    
                    exit_price = 0
                    profit = 0
                    commission = 0
                    
                    # Find ALL deals related to this position (prefer position_ticket)
                    deals = None
                    try:
                        deals = self.mt5_client.get_history_deals_by_position(ticket)
                    except Exception:
                        deals = None
                    
                    if deals:
                        for deal in deals:
                            if hasattr(deal, 'entry') and deal.entry == 1:  # DEAL_ENTRY_OUT
                                exit_price = deal.price
                                profit += deal.profit + getattr(deal, 'commission', 0) + getattr(deal, 'swap', 0)
                                
                                # Determine closure reason
                                reason = getattr(deal, 'reason', 0)
                                if reason == 4:
                                    status = 'CLOSED_SL'
                                elif reason == 5:
                                    status = 'CLOSED_TP'
                                else:
                                    status = 'CLOSED_MANUAL'
                                break
                    
                    status = 'CLOSED_UNKNOWN'  # Default status
                    if exit_price == 0:
                        # Fallback if no exit deal found (e.g., manual closure, partial close)
                        logger.warning(f"No exit deal found for position {ticket}, trying to infer status.")
                        
                        # Check history again with a wider time range
                        history_orders = self.mt5_client.get_history_orders(ticket)
                        if history_orders:
                            for order in history_orders:
                                if order.type == 1:  # Sell order for a buy position
                                    exit_price = order.price_done
                                    break
                                elif order.type == 0:  # Buy order for a sell position
                                    exit_price = order.price_done
                                    break
                        
                        if exit_price != 0:
                            logger.info(f"Inferred exit price {exit_price} from order history.")
                        else:
                            logger.error(f"Could not determine exit price for {ticket}. Status remains UNKNOWN.")

                    # Determine closure status from deal reason if possible
                    if deals:
                        for deal in deals:
                            if deal.entry == 1:  # Out-deal
                                reason = getattr(deal, 'reason', 0)
                                if reason == 4: # DEAL_REASON_SL
                                    status = 'CLOSED_SL'
                                elif reason == 5: # DEAL_REASON_TP
                                    status = 'CLOSED_TP'
                                else:
                                    status = 'CLOSED_MANUAL'
                                break # Exit after finding the main closing deal
                    
                    # If status is still unknown, try to infer from price
                    if status == 'CLOSED_UNKNOWN' and exit_price != 0:
                        pip_size = get_pip_size(self.mt5_client.get_symbol_info(trade['symbol']))
                        
                        # Check against SL/TP with a tolerance
                        sl_diff = abs(exit_price - trade['stop_loss'])
                        tp_diff = abs(exit_price - trade['take_profit'])
                        
                        logger.info(f"Inferring close reason for {ticket}: exit_price={exit_price}, tp={trade['take_profit']}, sl={trade['stop_loss']}")
                        logger.info(f"TP diff: {tp_diff}, SL diff: {sl_diff}, tolerance: {10 * pip_size}")

                        if sl_diff < 10 * pip_size:
                            status = 'CLOSED_SL'
                        elif tp_diff < 10 * pip_size:
                            status = 'CLOSED_TP'
                        else:
                            status = 'CLOSED_MANUAL' # If not near SL/TP, likely manual


                    self.trade_logger.update_trade(
                        ticket,
                        exit_price,
                        profit - commission,
                        status
                    )
                    
                    symbol = trade.get('symbol')
                    if symbol and hasattr(self, 'symbol_contexts'):
                        context = self.symbol_contexts.get(symbol)
                        if context:
                            try:
                                enriched_trade = dict(trade)
                                enriched_trade['exit_price'] = exit_price
                                enriched_trade['profit'] = profit - commission
                                enriched_trade['status'] = status
                                enriched_trade['exit_time'] = datetime.now().isoformat()
                                if 'duration' not in enriched_trade:
                                    entry_ts = trade.get('timestamp')
                                    try:
                                        if isinstance(entry_ts, str):
                                            entry_dt = datetime.fromisoformat(entry_ts)
                                        else:
                                            entry_dt = entry_ts
                                        enriched_trade['duration'] = str(datetime.now() - entry_dt)
                                    except Exception:
                                        pass
                                context.record_trade_result(enriched_trade)
                            except Exception:
                                logger.debug(f"{symbol}: Failed to record trade result for optimizer")
                    if exit_price and symbol:
                        self.recent_closures[symbol] = (
                            datetime.now(),
                            exit_price,
                            status  # This will be 'CLOSED_TP', 'CLOSED_SL', or 'CLOSED_MANUAL'
                        )
                        self.save_memory_state()
                        
                        logger.info(f"Recorded {symbol} closure at {exit_price} ({status})")
                                    
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
        
        # Save to file atomically
        memory_path = Path('bot_memory.json')
        tmp_path = memory_path.with_suffix('.json.tmp')
        try:
            tmp_path.write_text(json.dumps(memory_state, indent=2))
            tmp_path.replace(memory_path)
            logger.debug("Memory state saved to bot_memory.json")
        except Exception as exc:
            logger.error(f"Failed to persist memory state: {exc}", exc_info=True)

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
                # Check for news-related trading pause
                if self.config.enable_news_filter:
                    paused, reason = is_trading_paused(datetime.now(timezone.utc))
                    if paused:
                        logger.debug(f"Trading is paused: {reason}. Skipping this cycle.")
                        await asyncio.sleep(self.config.main_loop_interval)
                        continue

                # Check for maximum drawdown
                if await self.check_drawdown():
                    logger.error("Maximum drawdown reached, stopping bot")
                    break
                
                # Sync trade status with MT5
                await self.sync_trade_status()                
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
