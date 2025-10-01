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
from utils import get_tick_size, get_symbol_precision, is_trading_paused
from symbol_runtime import SymbolRuntimeContext, load_symbol_profile
from market_session import MarketSession

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
        """Load configuration from JSON file and apply CLI overrides."""
        # Helper to force required keys to exist
        def _req(d, key, section_name):
            if key not in d:
                raise ValueError(f"Missing required key '{key}' in section '{section_name}' of {self.config_path}")
            return d[key]

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # --- Trading settings (required) ---
            trading = config_data.get('trading_settings') or {}
            self.max_period               = _req(trading, 'max_period', 'trading_settings')
            self.main_loop_interval       = _req(trading, 'main_loop_interval_seconds', 'trading_settings')
            self.order_retry_attempts     = _req(trading, 'order_retry_attempts', 'trading_settings')
            self.lookback_period          = _req(trading, 'lookback_period', 'trading_settings')
            self.swing_window             = _req(trading, 'swing_window', 'trading_settings')
            self.breakout_threshold       = _req(trading, 'breakout_threshold_pips', 'trading_settings')
            self.min_peak_rank            = _req(trading, 'min_peak_rank', 'trading_settings')
            self.proximity_threshold      = _req(trading, 'proximity_threshold_pips', 'trading_settings')
            self.min_stop_loss_pips       = _req(trading, 'min_stop_loss_pips', 'trading_settings')
            self.stop_loss_buffer_pips    = _req(trading, 'stop_loss_buffer_pips', 'trading_settings')
            self.stop_loss_atr_multiplier = float(trading.get('stop_loss_atr_multiplier', 0.8))
            self.min_range_pips           = _req(trading, 'min_range_pips', 'trading_settings')
            self.atr_period               = _req(trading, 'atr_period', 'trading_settings')
            self.atr_tp_multiplier        = _req(trading, 'atr_tp_multiplier', 'trading_settings')
            self.session_based_tp         = _req(trading, 'session_based_tp', 'trading_settings')
            self.volatility_adjustment    = _req(trading, 'volatility_adjustment', 'trading_settings')
            self.enable_news_filter       = _req(trading, 'enable_news_filter', 'trading_settings')

            # --- Risk management (required) ---
            risk = config_data.get('risk_management') or {}
            self.risk_per_trade               = _req(risk, 'risk_per_trade', 'risk_management')
            self.fixed_lot_size               = risk.get('fixed_lot_size')  # may be None by design
            self.max_drawdown                 = _req(risk, 'max_drawdown_percentage', 'risk_management')
            self.risk_reward_ratio            = _req(risk, 'risk_reward_ratio', 'risk_management')
            self.correlation_threshold        = _req(risk, 'correlation_threshold', 'risk_management')
            self.correlation_lookback_period  = _req(risk, 'correlation_lookback_period', 'risk_management')
            self.risk_management_settings     = dict(risk)  # keep the dict for RiskManager init

            # --- Symbols (required) ---
            symbols_data = config_data.get('symbols')
            if not isinstance(symbols_data, list) or not symbols_data:
                raise ValueError(f"'symbols' must be a non-empty list in {self.config_path}")
            # enforce presence of name + timeframes in each symbol
            self.symbols = []
            for sym in symbols_data:
                name = _req(sym, 'name', 'symbols[]')
                tfs  = _req(sym, 'timeframes', 'symbols[]')
                if not isinstance(tfs, list) or not tfs:
                    raise ValueError("Each symbol must have a non-empty 'timeframes' list")
                self.symbols.append({'name': name, 'timeframes': tfs})

            # --- Trading sessions (optional) ---
            sessions_data = config_data.get('trading_sessions') or []
            self.trading_sessions = []
            for session in sessions_data:
                # if you want sessions enforced, swap to _req(...) here
                name  = session.get('name')
                start = session.get('start_time')
                end   = session.get('end_time')
                if name and start and end:
                    self.trading_sessions.append({
                        'name': name,
                        'start_time': datetime.strptime(start, '%H:%M').time(),
                        'end_time': datetime.strptime(end, '%H:%M').time(),
                    })

            # --- Paths (optional but recommended in JSON) ---
            paths_cfg = config_data.get('paths') or {}
            base_dir = self.config_path.parent
            self.symbol_config_dir   = (base_dir / (paths_cfg.get('symbol_config_dir') or 'symbol_configs')).resolve()
            self.calibrator_state_dir= (base_dir / (paths_cfg.get('calibrator_state_dir') or 'calibrator_states')).resolve()
            self.optimizer_state_dir = (base_dir / (paths_cfg.get('optimizer_state_dir') or 'optimizer_states')).resolve()

            # --- Calibration & pattern settings (optional) ---
            self.calibration_settings = dict(config_data.get('calibration') or {})

            pattern_cfg = config_data.get('pattern_settings') or {}
            # Using bool() on .get(...) avoids embedding a hard-coded default
            self.pattern_only_momentum              = bool(pattern_cfg.get('pattern_only_momentum'))
            self.pattern_mitigate_direction         = True if pattern_cfg.get('mitigate_direction') is None else bool(pattern_cfg.get('mitigate_direction'))
            self.pattern_indecision_penalty_per_hit = float(pattern_cfg.get('indecision_penalty_per_hit') or 0.06)
            self.pattern_indecision_penalty_cap     = float(pattern_cfg.get('indecision_penalty_cap') or 0.12)
            self.pattern_strength_map               = dict(pattern_cfg.get('strength_map') or {})

            # --- Apply CLI overrides (optional) ---
            if self.args:
                if getattr(self.args, 'risk_per_trade', None) is not None:
                    self.risk_per_trade = self.args.risk_per_trade
                    self.risk_management_settings['risk_per_trade'] = self.risk_per_trade
                if getattr(self.args, 'symbol', None):
                    self.symbols = [{'name': self.args.symbol, 'timeframes': ['M15']}]
                if getattr(self.args, 'timeframe', None):
                    for sym in self.symbols:
                        sym['timeframes'] = [self.args.timeframe]

            # Keep risk_management_settings in sync with any flattened attrs
            if isinstance(self.risk_management_settings, dict):
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
            # No silent fallbacks fail fast so configs stay the single source of truth
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}") from e

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
        self.session_manager = MarketSession(config)
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
            # M1 confirmation removed from system
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
            'stop_loss_atr_multiplier': float(getattr(self.strategy, 'stop_loss_atr_multiplier', getattr(self.config, 'stop_loss_atr_multiplier', 0.8))),
            'risk_reward_ratio': float(getattr(self.config, 'risk_reward_ratio', 1.5)),
            'min_confidence': float(getattr(self.strategy, 'min_confidence', 0.6)),
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
        
        phase = self.session_manager.get_phase()
        if not self.session_manager.is_trade_window():
            logger.info(
                f"Outside trade window, current phase={phase.name}, session={phase.session}"
            )
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
    
    async def execute_trade(self, signal: TradingSignal, symbol: str, risk_overrides: Optional[Dict[str, float]] = None):
        """
        Execute trade with live price validation for FOREX
        """
        risk_overrides = risk_overrides or {}
        try:
            # Get symbol info for tick/pip calculations
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return
            
            precision = get_symbol_precision(symbol_info, overrides=risk_overrides)
            tick_size = precision.tick_size or 0.0
            pip_size = precision.pip_size or tick_size
            if tick_size <= 0 or pip_size <= 0:
                logger.error(f"{symbol}: Unable to determine pip/tick precision")
                return
            
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
            
            # --- Drift validation ---
            price_drift = abs(execution_price - signal.entry_price)

            # Calculate provisional stop loss distances
            provisional_sl_ticks = abs(execution_price - signal.stop_loss) / tick_size
            provisional_sl_pips  = abs(execution_price - signal.stop_loss) / pip_size

            # Allow drift up to 25% of SL, bounded [2, 10] pips
            dynamic_max_drift_pips = max(2.0, min(10.0, 0.25 * provisional_sl_pips))

            if price_drift > dynamic_max_drift_pips * pip_size:
                logger.warning(
                    f"{symbol}: Price drifted too far\n"
                    f"  Signal: {signal.entry_price:.5f}\n"
                    f"  Current: {execution_price:.5f}\n"
                    f"  Drift: {price_drift/tick_size:.1f} ticks (approx {price_drift/pip_size:.1f} pips)\n"
                    f"  Limit: {dynamic_max_drift_pips:.1f} pips\n"
                    "  TRADE SKIPPED"
                )
            # --- Recalculate SL distance at actual entry price ---
            actual_sl_distance = abs(execution_price - signal.stop_loss)
            sl_ticks = actual_sl_distance / tick_size
            sl_pips  = actual_sl_distance / pip_size
            
            # Validate minimum stop loss
            min_sl_floor = (
                float(signal.parameters.get('min_stop_loss_pips', self.config.min_stop_loss_pips))
                if isinstance(signal.parameters, dict) else self.config.min_stop_loss_pips
            )

            if not self.risk_manager.validate_stop_loss(
                symbol_info, execution_price, signal.stop_loss, min_sl_floor, overrides=risk_overrides
            ):
                return
            
            # --- Position sizing ---
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=sl_pips,  # pass display pips to risk manager
                overrides=risk_overrides
            )
            
            if position_size <= 0:
                logger.warning(f"{symbol}: Invalid position size calculated")
                return
            
            # --- Final parameter validation ---
            if not self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                volume=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            ):
                logger.warning(f"{symbol}: Trade parameters validation failed")
                return
            
            # --- Risk limits ---
            if not self.risk_manager.check_risk_limits(symbol, signal.type, overrides=risk_overrides):
                logger.info(f"{symbol}: Risk limits prevent trading")
                return
            
            # --- EXECUTE THE TRADE ---
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

                # Capture live position ticket for robust closure tracking
                pos_id = None
                try:
                    pos_list = self.mt5_client.get_positions(symbol)
                    if pos_list and len(pos_list) >= 1:
                        for pos in pos_list:
                            if getattr(pos, 'type', None) == signal.type:
                                pos_id = getattr(pos, 'ticket', None)
                                break
                        if pos_id is None:
                            pos_id = getattr(pos_list[0], 'ticket', None)
                except Exception as _e:
                    logger.debug(f"Failed to capture position ticket for {symbol}: {_e}")

                trade_details = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'order_type': 'BUY' if signal.type == 0 else 'SELL',
                    'entry_price': result.price,
                    'signal_price': signal.entry_price,  # Track signal vs execution
                    'volume': position_size,
                    'stop_loss': signal.stop_loss,
                    'stop_loss_ticks': sl_ticks,
                    'stop_loss_pips': sl_pips,
                    'take_profit': signal.take_profit,
                    'ticket': result.order,
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'drift_ticks': price_drift / tick_size,
                    'drift_pips': price_drift / pip_size,
                    'tick_size': tick_size,
                }

                if pos_id is not None:
                    try:
                        trade_details['position_ticket'] = int(pos_id)
                    except Exception:
                        trade_details['position_ticket'] = pos_id

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
                    f"Symbol: {symbol}\n"
                    f"  Type: {'BUY' if signal.type == 0 else 'SELL'}\n"
                    f"  Executed: {result.price:.5f}\n"
                    f"  Signal was: {signal.entry_price:.5f}\n"
                    f"  Drift: {price_drift/tick_size:.1f} ticks (~{price_drift/pip_size:.1f} pips)\n"
                    f"  Volume: {position_size:.2f}\n"
                    f"  Risk: {sl_ticks:.1f} ticks (~{sl_pips:.1f} pips)\n"
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
        
        precision = get_symbol_precision(symbol_info, overrides=risk_overrides)
        tick_size = precision.tick_size or 0.0
        pip_size = precision.pip_size or tick_size


        # Check if symbol is in cooldown (different times for TP/SL closures)
        if symbol in self.recent_closures:
            closure_time, exit_price, status = self.recent_closures[symbol]
            time_since_close = (datetime.now() - closure_time).total_seconds()

            if status == 'CLOSED_TP':
                cooldown_period = 900  # 15 minutes after TP
            elif status == 'CLOSED_SL':
                cooldown_period = 300  # 5 minutes after SL
            else:  # CLOSED_MANUAL or CLOSED_UNKNOWN
                cooldown_period = 600  # 10 minutes default

            if time_since_close < cooldown_period:
                logger.info(
                    f"{symbol} in cooldown for {cooldown_period-time_since_close:.0f}s more "
                    f"(closed {status} {time_since_close/60:.1f} min ago)"
                )
                return

        try:
            # Fetch multi-timeframe data
            mtf_data = await self.market_data.fetch_multi_timeframe_data(symbol)

            if not mtf_data:
                logger.warning(f"No data available for {symbol}")
                return

            # --- Case 1: H1 + M15 available ---
            if 'H1' in mtf_data and 'M15' in mtf_data:
                h1_trend = self.market_data.identify_trend(mtf_data['H1'])
                logger.debug(f"{symbol} H1 trend: {h1_trend}")

                m15_signal = self.strategy.generate_signal(mtf_data['M15'], symbol, trend=h1_trend)

                if m15_signal:
                    if self._is_recent_breakout_duplicate(symbol, m15_signal, pip_size):
                        return
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

            # --- Case 2: Only M15 available ---
            elif 'M15' in mtf_data:
                m15_data = mtf_data['M15']
                m15_trend = self.market_data.identify_trend(m15_data)

                signal = self.strategy.generate_signal(m15_data, symbol, trend=m15_trend)

                if signal:
                    if self._is_recent_breakout_duplicate(symbol, signal, pip_size):
                        return
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

            # --- Case 3: Only H1 available ---
            elif 'H1' in mtf_data:
                h1_data = mtf_data['H1']
                h1_trend = self.market_data.identify_trend(h1_data)

                signal = self.strategy.generate_signal(h1_data, symbol, trend=h1_trend)

                if signal:
                    if self._is_recent_breakout_duplicate(symbol, signal, pip_size):
                        return

                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if signal.type == 0 else 'SELL'}, "
                        f"Confidence={signal.confidence:.2f}"
                    )

                    if self.session_manager.is_trade_window():
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
                        phase = self.session_manager.get_phase()
                        logger.info(
                            f"Outside trade window, current phase={phase.name}, session={phase.session}"
                        )
                else:
                    logger.debug(f"No signal generated for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
    async def sync_trade_status(self):
        """Trade status synchronization"""
        if not self.trade_logger:
            return
            
        try:
            # Treat any trade not marked CLOSED as open candidate (handles restarts)
            open_trades = [
                t for t in self.trade_logger.get_all_trades()
                if not str(t.get('status', 'OPEN')).upper().startswith('CLOSED')
            ]
            logger.debug(f"sync_trade_status: {len(open_trades)} candidate trades to reconcile")
            
            if not open_trades:
                return
                
            # Get current positions
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {getattr(pos, 'ticket', None) for pos in current_positions}
            
            for trade in open_trades:
                # Prefer position_ticket; if missing, try to map by symbol to a live position
                ticket = trade.get('position_ticket') or trade.get('ticket')
                if 'position_ticket' not in trade or not trade.get('position_ticket'):
                    try:
                        sym = trade.get('symbol')
                        match = next((p for p in current_positions if getattr(p, 'symbol', None) == sym), None)
                        if match:
                            ticket = getattr(match, 'ticket', ticket)
                    except Exception as _e:
                        logger.debug(f"Symbol-to-position mapping skipped: {_e}")
                if ticket and ticket not in current_tickets:
                    # Position was closed
                    logger.info(f"Position {ticket} no longer in open positions. Resolving closure from history...")
                    
                    end_time = datetime.now()
                    start_time = datetime.fromisoformat(trade['timestamp'])
                    
                    exit_price = 0.0
                    profit = 0.0
                    commission = 0.0
                    status = 'CLOSED_UNKNOWN'

                    # Try MT5 deals by position first
                    deals_by_pos = None
                    try:
                        deals_by_pos = self.mt5_client.get_history_deals_by_position(int(ticket))
                    except Exception as _e:
                        logger.debug(f"get_history_deals_by_position({ticket}) failed: {_e}")

                    if deals_by_pos:
                        exit_deals = [d for d in deals_by_pos if getattr(d, 'entry', None) == 1]
                        if exit_deals:
                            d = sorted(exit_deals, key=lambda x: getattr(x, 'time', 0))[-1]
                            exit_price = float(getattr(d, 'price', 0.0) or 0.0)
                            profit = float(getattr(d, 'profit', 0.0) or 0.0)
                            commission = float(getattr(d, 'commission', 0.0) or 0.0) + float(getattr(d, 'swap', 0.0) or 0.0)
                            r = getattr(d, 'reason', 0)
                            if r == 4:
                                status = 'CLOSED_SL'
                            elif r == 5:
                                status = 'CLOSED_TP'
                            else:
                                status = 'CLOSED_MANUAL'
                        else:
                            logger.info(f"No exit deal found in deals_by_position for {ticket}")

                    # Fallback: scan the general history window and filter by symbol
                    if exit_price == 0.0:
                        all_deals = []
                        try:
                            all_deals = self.mt5_client.get_history_deals(start_time, end_time)
                        except Exception as _e:
                            logger.debug(f"get_history_deals window failed: {_e}")
                        symbol = trade.get('symbol')
                        if all_deals and symbol:
                            exit_candidates = [d for d in all_deals if getattr(d, 'entry', None) == 1 and getattr(d, 'symbol', None) == symbol]
                            if exit_candidates:
                                d = sorted(exit_candidates, key=lambda x: getattr(x, 'time', 0))[-1]
                                exit_price = float(getattr(d, 'price', 0.0) or 0.0)
                                profit = float(getattr(d, 'profit', 0.0) or 0.0)
                                commission = float(getattr(d, 'commission', 0.0) or 0.0) + float(getattr(d, 'swap', 0.0) or 0.0)
                                r = getattr(d, 'reason', 0)
                                if r == 4:
                                    status = 'CLOSED_SL'
                                elif r == 5:
                                    status = 'CLOSED_TP'
                                else:
                                    status = 'CLOSED_MANUAL'
                            else:
                                logger.warning(f"No exit deals found for {symbol} within history window")

                    if exit_price == 0.0:
                        logger.warning(f"Unable to determine exit for ticket {ticket}; will retry next cycle")
                        continue
                    
                    # If status is still unknown, try to infer from price proximity to SL/TP
                    if status == 'CLOSED_UNKNOWN' and exit_price != 0:
                        sym_info = self.mt5_client.get_symbol_info(trade['symbol'])
                        tick_size = get_tick_size(sym_info) if sym_info else 0.0001
                        sl = trade.get('stop_loss')
                        tp = trade.get('take_profit')
                        sl_diff = abs(exit_price - sl) if sl is not None else float('inf')
                        tp_diff = abs(exit_price - tp) if tp is not None else float('inf')
                        tol = 10 * tick_size
                        logger.info(f"Inferring close reason for {ticket}: exit_price={exit_price}, tp={tp}, sl={sl}")
                        logger.info(f"TP diff: {tp_diff}, SL diff: {sl_diff}, tolerance: {tol}")

                        if tp is not None and tp_diff < tol:
                            status = 'CLOSED_TP'
                        elif sl is not None and sl_diff < tol:
                            status = 'CLOSED_SL'
                        else:
                            status = 'CLOSED_MANUAL'


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





