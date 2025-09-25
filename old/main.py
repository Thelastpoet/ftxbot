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
from calibrator import OnlineLogisticCalibrator, CalibratorConfig

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
            self.volume_zscore_threshold = trading.get('volume_zscore_threshold', 1.0)
            # Supervision behavior: 'none' | 'advisory' | 'active' (default: none)
            self.supervision_mode = trading.get('supervision_mode', 'none')
            # M1 confirmation (wasn't previously wired into Config)
            self.m1_confirmation_enabled = trading.get('m1_confirmation_enabled', False)
            self.m1_confirmation_candles = trading.get('m1_confirmation_candles', 1)
            self.m1_confirmation_buffer_pips = trading.get('m1_confirmation_buffer_pips', 0.5)
            
            # Risk management
            self.risk_management = config_data.get('risk_management', {})
            self.risk_per_trade = self.risk_management.get('risk_per_trade', 0.02)
            self.fixed_lot_size = self.risk_management.get('fixed_lot_size', None)
            self.max_drawdown = self.risk_management.get('max_drawdown_percentage', 0.1)
            self.risk_reward_ratio = self.risk_management.get('risk_reward_ratio', 2.0)
            
            # Calibration (lightweight ML) settings
            calib = config_data.get('calibration', {})
            self.calib_enabled = calib.get('enabled', False)
            self.calib_margin = calib.get('margin', 0.05)
            self.calib_lr = calib.get('learning_rate', 0.01)
            self.calib_l2 = calib.get('l2', 1e-3)
            self.calib_state_file = calib.get('state_file', 'calibrator_state.json')
            self.calib_ttl_minutes = calib.get('ttl_minutes', 60)
            
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
        self.supervision_mode = 'none'
        # M1 confirmation defaults
        self.m1_confirmation_enabled = False
        self.m1_confirmation_candles = 1
        self.m1_confirmation_buffer_pips = 0.5
        # Calibration defaults
        self.calib_enabled = False
        self.calib_margin = 0.05
        self.calib_lr = 0.01
        self.calib_l2 = 1e-3
        self.calib_state_file = 'calibrator_state.json'
        self.calib_ttl_minutes = 60

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
                    f"Session configured: {sess['name']} {sess['start_time']} -> {sess['end_time']} (interpreted in local time)"
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
        self.load_memory_state()
        self.session_manager = TradingSession(config)
        # Calibrator (lightweight ML) setup
        try:
            calib_cfg = CalibratorConfig(
                enabled=getattr(self.config, 'calib_enabled', False),
                margin=getattr(self.config, 'calib_margin', 0.05),
                learning_rate=getattr(self.config, 'calib_lr', 0.01),
                l2=getattr(self.config, 'calib_l2', 1e-3),
                state_file=getattr(self.config, 'calib_state_file', 'calibrator_state.json'),
            )
            self.calibrator = OnlineLogisticCalibrator(calib_cfg)
            if calib_cfg.enabled:
                logger.info(f"Calibrator: ENABLED | margin={calib_cfg.margin} | state={calib_cfg.state_file}")
            else:
                logger.info("Calibrator: DISABLED")
        except Exception as _e:
            self.calibrator = None
            logger.debug(f"Calibrator init skipped: {_e}")
        # Session-phase engine (lightweight, additive)
        try:
            self.market_session = MarketSession(config)
        except Exception:
            self.market_session = None
        self.running = False
        self.initial_balance = None
        # Log current session phase and reference times
        try:
            if self.market_session:
                phase = self.market_session.get_phase()
                mins = self.market_session.minutes_to_boundary()
                refs = self.market_session.get_reference_times()
                logger.info(
                    f"Current phase: {phase.name} ({phase.session}) | weight={phase.weight:.2f} | ttl={phase.ttl_minutes}m | blackout={phase.is_blackout} | next boundary in {mins}m"
                )
                logger.info(
                    f"London: {refs['london'].strftime('%Y-%m-%d %H:%M:%S')} | NewYork: {refs['new_york'].strftime('%Y-%m-%d %H:%M:%S')} | UTC: {refs['utc'].strftime('%Y-%m-%d %H:%M:%S')}"
                )
            # Explicitly log supervision mode
            logger.info(f"Supervision mode: {self.config.supervision_mode.upper()} (no auto-close/SL changes unless ACTIVE)")
        except Exception:
            pass
        
    async def initialize(self):
        """Initialize all bot components"""
        try:
            # Import components (delayed to allow for proper module structure)
            from mt5_client import MetaTrader5Client
            from market_data import MarketData
            from strategy import PriceActionStrategy
            from risk_manager import RiskManager
            from trade_logger import TradeLogger
            
            # Initialize components
            self.mt5_client = MetaTrader5Client()
            if not self.mt5_client.initialized:
                raise Exception("Failed to initialize MetaTrader5 connection")
            
            self.market_data = MarketData(self.mt5_client, self.config)
            self.strategy = PriceActionStrategy(self.config)
            # Log whether M1 confirmation is enabled via config
            try:
                m1_on = getattr(self.strategy, 'm1_confirmation_enabled', False)
                candles = getattr(self.strategy, 'm1_confirmation_candles', 1)
                buf = getattr(self.strategy, 'm1_confirmation_buffer_pips', 0.5)
                logger.info(f"M1 confirmation: {'ENABLED' if m1_on else 'DISABLED'} (candles={candles}, buffer_pips={buf})")
            except Exception:
                pass
            self.risk_manager = RiskManager(self.config.risk_management, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')
            
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

    def log_signal_row(self, symbol: str, signal: TradingSignal, rr: float, p_hat: float, p_star: float, prefilter: bool = False) -> None:
        """Append a signal decision row to signals.csv for analysis/calibration.

        Columns: timestamp, symbol, type, rr, p_hat, p_star, confidence, strength, momentum,
                 dir_match, trend_match, spread_impact, prefilter
        """
        try:
            from datetime import timezone
            path = 'signals.csv'
            exists = os.path.exists(path)
            cols = [
                'timestamp','symbol','type','rr','p_hat','p_star','confidence',
                'strength','momentum','dir_match','trend_match','spread_impact','prefilter'
            ]
            row = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'type': ('BUY' if signal.type == 0 else 'SELL'),
                'rr': f"{rr:.4f}",
                'p_hat': f"{p_hat:.4f}",
                'p_star': f"{p_star:.4f}",
                'confidence': f"{getattr(signal, 'confidence', 0.0):.4f}",
                'strength': f"{float((signal.features or {}).get('strength', 0.0)):.4f}",
                'momentum': f"{float((signal.features or {}).get('momentum', 0.0)):.4f}",
                'dir_match': f"{float((signal.features or {}).get('dir_match', 0.0)):.4f}",
                'trend_match': f"{float((signal.features or {}).get('trend_match', 0.0)):.4f}",
                'spread_impact': f"{float((signal.features or {}).get('spread_impact', 0.0)):.4f}",
                'prefilter': '1' if prefilter else '0'
            }
            # Write CSV (manual to avoid adding deps)
            with open(path, 'a', encoding='utf-8') as f:
                if not exists:
                    f.write(','.join(cols) + "\n")
                f.write(','.join(str(row[c]) for c in cols) + "\n")
        except Exception:
            pass
    
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
            
            # Check price drift from signal (relative to risk)
            price_drift = abs(execution_price - signal.entry_price)
            provisional_sl_pips = abs(execution_price - signal.stop_loss) / pip_size
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
            if actual_sl_pips < self.config.min_stop_loss_pips:
                logger.warning(
                    f"{symbol}: SL too tight after drift ({actual_sl_pips:.1f} pips) - skipping"
                )
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                stop_loss_pips=actual_sl_pips
            )
            
            if position_size <= 0:
                logger.warning(f"{symbol}: Invalid position size calculated")
                return
            
            # Final validation
            if not self.risk_manager.validate_trade_parameters(
                symbol=symbol,
                volume=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_type=signal.type
            ):
                logger.warning(f"{symbol}: Trade parameters validation failed")
                return
            
            # Check risk limits
            if not self.risk_manager.check_risk_limits(symbol, signal.type):
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
                # Optional: compute calibrator fields
                calib_rr = 0.0
                calib_p_hat = None
                calib_p_star = None
                if getattr(self, 'calibrator', None) and getattr(self.calibrator, 'cfg', None) and self.calibrator.cfg.enabled:
                    try:
                        risk = abs(signal.entry_price - signal.stop_loss)
                        reward = abs(signal.take_profit - signal.entry_price)
                        calib_rr = (reward / risk) if risk > 0 else 0.0
                        calib_p_hat = self.calibrator.predict_proba(signal.features or {})
                        calib_p_star = self.calibrator.decision_threshold(calib_rr)
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
                    'take_profit': signal.take_profit,
                    'ticket': result.order,
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'drift_pips': price_drift / pip_size,
                    'features': getattr(signal, 'features', None),
                    'calib_rr': calib_rr,
                    'calib_p_hat': calib_p_hat,
                    'calib_p_star': calib_p_star,
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
        
        # Check if symbol is in cooldown
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
                logger.debug(f"{symbol} in cooldown for {cooldown_period-time_since_close:.0f}s more (closed {status} {time_since_close/60:.1f} min ago)")
                return
        
        # Check consumed breakouts BEFORE signal generation to save computation
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
                
                # Generate M15 signal with trend context and session phase
                phase = self.market_session.get_phase() if getattr(self, 'market_session', None) else None
                m15_signal = self.strategy.generate_signal(mtf_data['M15'], symbol, trend=h1_trend, phase=phase)

                if m15_signal:
                    # Disable for now not trade if signal is against H1 trend
                    """
                    if not (m15_signal.features or {}).get('trend_match', 0.0):
                        logger.debug(f"{symbol}: Signal ignored, does not match H1 trend.")
                        return
                    """    
                    # Optional calibrator gating based on probability threshold
                    if getattr(self, 'calibrator', None) and getattr(self.calibrator, 'cfg', None) and self.calibrator.cfg.enabled:
                        try:
                            rr = 0.0
                            try:
                                risk = abs(m15_signal.entry_price - m15_signal.stop_loss)
                                reward = abs(m15_signal.take_profit - m15_signal.entry_price)
                                rr = (reward / risk) if risk > 0 else 0.0
                            except Exception:
                                rr = 0.0
                            p_hat = self.calibrator.predict_proba(m15_signal.features or {})
                            p_star = self.calibrator.decision_threshold(rr)
                            # Log to signals.csv
                            try:
                                self.log_signal_row(symbol, m15_signal, rr, p_hat, p_star, prefilter=False)
                            except Exception:
                                pass
                            if p_hat < p_star:
                                logger.info(
                                    f"{symbol}: Calibrator gated trade (p={p_hat:.2f} < p*={p_star:.2f}, R={rr:.2f})"
                                )
                                return
                        except Exception as _e:
                            logger.debug(f"Calibrator gating error: {_e}")
                    # ADDITIONAL CHECK: Is this breakout level already consumed?
                    if symbol in self.consumed_breakouts:
                        for level, direction, timestamp in self.consumed_breakouts[symbol]:
                            age = (datetime.now() - timestamp).total_seconds()
                            if age < 900:  # Within 15 minutes
                                signal_direction = 'bullish' if m15_signal.type == 0 else 'bearish'
                                if direction == signal_direction:
                                    # Check if this is the same breakout level
                                    level_distance_pips = abs(m15_signal.breakout_level - level) / pip_size
                                    if level_distance_pips < 5:  # Same level (within 5 pips)
                                        logger.info(
                                            f"{symbol}: Signal generated but breakout at {level:.5f} "
                                            f"already consumed {age/60:.1f} min ago. Skipping duplicate trade."
                                        )
                                        return
                    
                    logger.info(
                        f"Signal generated for {symbol}: "
                        f"Type={'BUY' if m15_signal.type == 0 else 'SELL'}, "
                        f"Confidence={m15_signal.confidence:.2f}, "
                        f"H1 Trend={h1_trend}"
                    )
                    
                    # Check if within trading session (coarse) and not in blackout (rollover)
                    if self.session_manager.is_trading_time() and (self.market_session.is_trade_window() if getattr(self, 'market_session', None) else True):
                        result = await self.execute_trade(m15_signal, symbol)
                        # If calibrator enabled and trade executed (non-None result), optionally we could
                        # later update with outcome when closed. For now we only log the signal row above.
                        
                        if result:  # If trade was successfully opened
                            # Track this breakout as consumed
                            if symbol not in self.consumed_breakouts:
                                self.consumed_breakouts[symbol] = []
                            
                            direction = 'bullish' if m15_signal.type == 0 else 'bearish'
                            self.consumed_breakouts[symbol].append(
                                (m15_signal.breakout_level, direction, datetime.now())
                            )
                            
                            self.save_memory_state()
                            
                            # Clean old entries (older than 30 minutes)
                            current_time = datetime.now()
                            self.consumed_breakouts[symbol] = [
                                (lvl, dir, ts) for lvl, dir, ts in self.consumed_breakouts[symbol]
                                if (current_time - ts).total_seconds() < 1800
                            ]
                    else:
                        logger.info("Outside trading session, signal ignored")
                else:
                    logger.debug(f"No signal generated for {symbol}")
            
            # FALLBACK: Single timeframe processing (if only one timeframe available)
            elif 'M15' in mtf_data:
                # Use M15 data for both trend and signal
                m15_data = mtf_data['M15']
                m15_trend = self.market_data.identify_trend(m15_data)
                
                phase = self.market_session.get_phase() if getattr(self, 'market_session', None) else None
                signal = self.strategy.generate_signal(m15_data, symbol, trend=m15_trend, phase=phase)
                
                if signal:
                    # Check if this breakout level is already consumed (same as above)
                    if symbol in self.consumed_breakouts:
                        symbol_info = self.mt5_client.get_symbol_info(symbol)
                        if symbol_info:
                            pip_size = get_pip_size(symbol_info)
                            for level, direction, timestamp in self.consumed_breakouts[symbol]:
                                age = (datetime.now() - timestamp).total_seconds()
                                if age < 900:  # Within 15 minutes
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
                    
                    if self.session_manager.is_trading_time() and (self.market_session.is_trade_window() if getattr(self, 'market_session', None) else True):
                        result = await self.execute_trade(signal, symbol)
                        
                        if result:
                            # Track consumed breakout
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
            
            elif 'H1' in mtf_data:
                # Use H1 data for both trend and signal
                h1_data = mtf_data['H1']
                h1_trend = self.market_data.identify_trend(h1_data)
                
                phase = self.market_session.get_phase() if getattr(self, 'market_session', None) else None
                signal = self.strategy.generate_signal(h1_data, symbol, trend=h1_trend, phase=phase)
                
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
                    
                    if self.session_manager.is_trading_time() and (self.market_session.is_trade_window() if getattr(self, 'market_session', None) else True):
                        result = await self.execute_trade(signal, symbol)
                        
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
            open_trades = [t for t in self.trade_logger.trades 
                        if t.get('status') == 'OPEN']
            
            if not open_trades:
                return
                
            # Get current positions
            current_positions = self.mt5_client.get_all_positions()
            current_tickets = {pos.ticket for pos in current_positions}
            
            for trade in open_trades:
                ticket = trade.get('ticket')
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
                    
                    # Find ALL deals related to this position
                    deals = self.mt5_client.get_history_deals_by_position(ticket)
                    
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
                        history_orders = self.mt5_client.get_history_orders_by_position(ticket)
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
                        if abs(exit_price - trade['stop_loss']) < 5 * pip_size:
                            status = 'CLOSED_SL'
                        elif abs(exit_price - trade['take_profit']) < 5 * pip_size:
                            status = 'CLOSED_TP'
                        else:
                            status = 'CLOSED_MANUAL' # If not near SL/TP, likely manual


                    self.trade_logger.update_trade(
                        ticket,
                        exit_price,
                        profit - commission,
                        status
                    )

                    # Update calibrator with stricter +1R-before-SL label
                    try:
                        if getattr(self, 'calibrator', None) and getattr(self.calibrator, 'cfg', None) and self.calibrator.cfg.enabled:
                            ttl = int(getattr(self.config, 'calib_ttl_minutes', 60))
                            label = self._label_one_r_before_sl(trade, ttl)
                            feat = trade.get('features') if isinstance(trade, dict) else None
                            if feat is not None and label is not None:
                                self.calibrator.update(feat, int(label))
                                logger.info(f"Calibrator updated: +1R label={label} (ttl={ttl}m)")
                            else:
                                logger.debug("Calibrator not updated (missing features/label)")
                    except Exception as _e:
                        logger.debug(f"Calibrator update skipped: {_e}")
                    
                    symbol = trade.get('symbol')
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

    def _label_one_r_before_sl(self, trade: Dict, ttl_minutes: int) -> Optional[int]:
        """Return 1 if +1R is reached before SL within ttl_minutes of entry; else 0.

        BUY uses bid highs/lows; SELL approximates ask by bid + spread*point.
        Ties are counted as loss-first (conservative).
        """
        try:
            from datetime import timedelta
            symbol = trade.get('symbol')
            if not symbol:
                return None
            entry_ts_raw = trade.get('timestamp')
            if not entry_ts_raw:
                return None
            if isinstance(entry_ts_raw, str):
                entry_time = datetime.fromisoformat(entry_ts_raw)
            else:
                entry_time = entry_ts_raw
            order_type = trade.get('order_type') or ('BUY' if trade.get('type', 0) == 0 else 'SELL')
            entry_price = float(trade.get('entry_price', 0.0))
            sl = float(trade.get('stop_loss', 0.0))
            if entry_price <= 0 or sl <= 0:
                return None
            to_time = entry_time + timedelta(minutes=int(ttl_minutes))
            rates = self.mt5_client.copy_rates_range(symbol, 'M1', entry_time, to_time)
            if rates is None or len(rates) == 0:
                return None
            import pandas as pd
            df = pd.DataFrame(rates)
            sym_info = self.mt5_client.get_symbol_info(symbol)
            point = float(getattr(sym_info, 'point', 0.00001)) if sym_info else 0.00001

            if order_type == 'BUY':
                risk = entry_price - sl
                if risk <= 0:
                    return None
                target = entry_price + risk
                for _, row in df.iterrows():
                    bid_low = float(row['low'])
                    bid_high = float(row['high'])
                    if bid_low <= sl:
                        return 0
                    if bid_high >= target:
                        return 1
                return 0
            else:  # SELL
                risk = sl - entry_price
                if risk <= 0:
                    return None
                target = entry_price - risk
                for _, row in df.iterrows():
                    bid_high = float(row['high'])
                    bid_low = float(row['low'])
                    spread_points = float(row['spread']) if 'spread' in row else 0.0
                    ask_high = bid_high + spread_points * point
                    ask_low = bid_low + spread_points * point
                    if ask_high >= sl:
                        return 0
                    if ask_low <= target:
                        return 1
                return 0
        except Exception:
            return None
    
    async def supervise_positions(self):
        """Session-aware TTL supervision for open positions.

        Policies (lightweight, additive):
        - If elapsed >= phase.ttl_minutes and R < 0.0: close position (failed to follow through).
        - If elapsed >= phase.ttl_minutes and 0.0 <= R < 0.5: move SL to breakeven (with broker-distance buffer).
        - If elapsed >= 0.5 * ttl and R <= 0.0: tighten SL halfway toward entry to reduce risk.
        - Near session boundary (<=10m) and R >= 0.8: lock at +0.5R via SL.
        """
        try:
            mode = getattr(self.config, 'supervision_mode', 'advisory').lower()
            if mode == 'none':
                return
            if not self.mt5_client:
                return
            positions = self.mt5_client.get_all_positions()
            if not positions:
                return

            phase = self.market_session.get_phase() if getattr(self, 'market_session', None) else None
            ttl_minutes = getattr(phase, 'ttl_minutes', None) or 60
            boundary_mins = self.market_session.minutes_to_boundary() if getattr(self, 'market_session', None) else 999

            for p in positions:
                symbol = p.symbol
                symbol_info = self.mt5_client.get_symbol_info(symbol)
                if not symbol_info:
                    continue
                pip_size = get_pip_size(symbol_info)

                # Fetch matching trade record for entry time; fallback to now if missing
                entry_time = None
                entry_price = float(p.price_open)
                stop_loss = float(getattr(p, 'sl', 0.0)) if getattr(p, 'sl', 0.0) else None
                take_profit = float(getattr(p, 'tp', 0.0)) if getattr(p, 'tp', 0.0) else None
                try:
                    if self.trade_logger and self.trade_logger.trades:
                        for t in reversed(self.trade_logger.trades):
                            if t.get('ticket') == p.ticket:
                                ts = t.get('timestamp')
                                if ts:
                                    entry_time = datetime.fromisoformat(ts)
                                # Prefer logged SL/TP if defined
                                stop_loss = float(t.get('stop_loss', stop_loss or 0.0)) or stop_loss
                                take_profit = float(t.get('take_profit', take_profit or 0.0)) or take_profit
                                break
                except Exception:
                    pass

                if entry_time is None:
                    # Best-effort fallback: approximate from current time
                    entry_time = datetime.now()

                elapsed_min = max(0.0, (datetime.now() - entry_time).total_seconds() / 60.0)

                # Compute R progress
                risk_distance = None
                if stop_loss is not None and stop_loss != 0.0:
                    risk_distance = abs(entry_price - stop_loss)
                if not risk_distance or risk_distance <= 0:
                    # Can't supervise without SL reference
                    continue

                price_current = float(p.price_current)
                if p.type == 0:  # BUY
                    progress = price_current - entry_price
                else:  # SELL
                    progress = entry_price - price_current
                r_multiple = progress / risk_distance

                # Broker minimum stop distance
                try:
                    min_stop_distance = float(symbol_info.trade_stops_level) * float(symbol_info.point)
                except Exception:
                    min_stop_distance = 0.0

                # Helper to set SL safely
                def set_sl(target_sl: float) -> bool:
                    # Ensure SL on correct side and satisfies broker min distance
                    tick = self.mt5_client.get_symbol_info_tick(symbol)
                    if not tick:
                        return False
                    if p.type == 0:  # BUY: SL below current ask
                        safe_max = float(tick.ask) - max(min_stop_distance, pip_size)
                        new_sl = min(target_sl, safe_max)
                        if new_sl >= safe_max:
                            new_sl = safe_max - pip_size/10.0
                        if new_sl >= entry_price:
                            new_sl = entry_price - pip_size  # breakeven just below
                        if new_sl <= 0:
                            return False
                    else:  # SELL: SL above current bid
                        safe_min = float(tick.bid) + max(min_stop_distance, pip_size)
                        new_sl = max(target_sl, safe_min)
                        if new_sl <= safe_min:
                            new_sl = safe_min + pip_size/10.0
                        if new_sl <= entry_price:
                            new_sl = entry_price + pip_size  # breakeven just above
                    if mode == 'active':
                        res = self.mt5_client.modify_position(p.ticket, sl=new_sl, tp=take_profit)
                        if res:
                            logger.info(
                                f"TTL supervise: SL set -> {symbol} ticket {p.ticket} {new_sl:.5f} (R={r_multiple:.2f}, elapsed={elapsed_min:.0f}m, phase={getattr(phase,'name', 'n/a')})"
                            )
                            return True
                        return False
                    else:
                        logger.info(
                            f"TTL supervise (advisory): would set SL -> {symbol} ticket {p.ticket} {new_sl:.5f} (R={r_multiple:.2f}, elapsed={elapsed_min:.0f}m, phase={getattr(phase,'name', 'n/a')})"
                        )
                        return True

                # 1) End of TTL actions
                if elapsed_min >= ttl_minutes:
                    if r_multiple < 0.0:
                        # No follow-through and negative: close (or advise)
                        if mode == 'active':
                            close_res = self.mt5_client.close_position(p.ticket)
                            if close_res:
                                logger.info(
                                    f"TTL supervise: closed -> {symbol} ticket {p.ticket} (R={r_multiple:.2f}, elapsed={elapsed_min:.0f}m >= ttl {ttl_minutes}m)"
                                )
                        else:
                            logger.info(
                                f"TTL supervise (advisory): would CLOSE -> {symbol} ticket {p.ticket} (R={r_multiple:.2f}, elapsed={elapsed_min:.0f}m >= ttl {ttl_minutes}m)"
                            )
                        continue
                    if r_multiple < 0.5:
                        # Protect by moving to breakeven
                        be_sl = entry_price - pip_size if p.type == 0 else entry_price + pip_size
                        set_sl(be_sl)
                        continue

                # 2) Half TTL and non-positive: reduce risk by moving SL halfway to entry
                if elapsed_min >= (ttl_minutes * 0.5) and r_multiple <= 0.0 and stop_loss is not None:
                    mid_sl = (entry_price + stop_loss) / 2.0
                    set_sl(mid_sl)

                # 3) Near boundary and decent profit: lock partial R
                if boundary_mins <= 10 and r_multiple >= 0.8 and stop_loss is not None:
                    # Lock at +0.5R
                    lock_distance = 0.5 * risk_distance
                    if p.type == 0:
                        target_sl = entry_price + lock_distance
                    else:
                        target_sl = entry_price - lock_distance
                    set_sl(target_sl)

        except Exception as e:
            logger.error(f"Error in supervise_positions: {e}", exc_info=True)
            
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
                # Session-aware TTL supervision (only when enabled)
                if getattr(self.config, 'supervision_mode', 'none').lower() != 'none':
                    await self.supervise_positions()
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
