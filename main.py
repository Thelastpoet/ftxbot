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

import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from utils import resolve_pip_size
from strategy import PurePriceActionStrategy, TradingSignal
from state_manager import StateManager, save_strategy_state, load_strategy_state
from history_analysis import HistoricalAnalyzer, HistoricalSnapshot

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
            self.pattern_use_trading_tf = trading.get('pattern_use_trading_tf', False)
            self.htf_pattern_timeframe = trading.get('htf_pattern_timeframe', None)
            self.htf_pattern_window = trading.get('htf_pattern_window', self.pattern_window)
            self.htf_pattern_score_threshold = trading.get('htf_pattern_score_threshold', self.pattern_score_threshold)
            self.require_htf_pattern_alignment = trading.get('require_htf_pattern_alignment', True)
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

            # Historical analysis defaults
            self.historical_analysis = data.get('historical_analysis', {}) or {}

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
        self.pattern_use_trading_tf = False
        self.htf_pattern_timeframe = None
        self.htf_pattern_window = self.pattern_window
        self.htf_pattern_score_threshold = self.pattern_score_threshold
        self.require_htf_pattern_alignment = False
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
        self.historical_analysis = {}
        # No momentum filter config; price-action + MTF context only

class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.mt5_client = None
        self.market_data = None
        self.strategy = None
        self.risk_manager = None
        self.trade_logger = None
        self.historical_analyzer = HistoricalAnalyzer(config) if getattr(config, 'historical_analysis', None) else None
        self._historical_snapshots: Dict[str, HistoricalSnapshot] = {}
        self._historical_last_refresh: Dict[str, datetime] = {}
        
        # Persist state files under a dedicated folder
        self.state_manager = StateManager('symbol_state')
        self.running = False
        self.symbol_fillings = {}

        # Statistics tracking
        self._last_hour = None
        self._hourly_stats = {
            'symbols_processed': 0,
            'breakouts_detected': 0,
            'breakouts_bullish': 0,
            'breakouts_bearish': 0,
            'signals_generated': 0,
            'signals_blocked': 0,
            'block_reasons': {},
            'trades_executed': 0,
            'spring_confirmations': 0,
            'upthrust_confirmations': 0
        }

    async def _get_historical_snapshot(self, symbol: str, pip_size: float, trading_df: Optional[pd.DataFrame]) -> Optional[HistoricalSnapshot]:
        if not self.historical_analyzer or not getattr(self.config, 'historical_analysis', {}):
            return None
        try:
            params = self.historical_analyzer.resolve_params(symbol)
        except Exception:
            params = getattr(self.config, 'historical_analysis', {}) or {}
        if not params.get('enabled', False):
            return None

        refresh_minutes = float(
            params.get(
                'refresh_interval_minutes',
                getattr(self.config, 'historical_analysis', {}).get('refresh_interval_minutes', 60),
            )
            or 60.0
        )
        now = datetime.now(timezone.utc)
        last = self._historical_last_refresh.get(symbol)
        if last and (now - last) < timedelta(minutes=refresh_minutes):
            return self._historical_snapshots.get(symbol)

        daily_tf = params.get('daily_timeframe', 'D1')
        macro_bars = int(params.get('macro_window_bars', 240) or 240)
        try:
            daily_df = await self.market_data.fetch_data(symbol, daily_tf, macro_bars + 5)
        except Exception as exc:
            logger.warning("%s: failed to fetch %s history for analysis: %s", symbol, daily_tf, exc)
            return self._historical_snapshots.get(symbol)
        if daily_df is None:
            return self._historical_snapshots.get(symbol)

        def _fetcher(sym: str, tf: str, bars: int):
            if sym == symbol and tf == daily_tf:
                return daily_df
            try:
                rates = self.mt5_client.copy_rates_from_pos(sym, tf, 0, bars)
                if rates is None or len(rates) == 0:
                    return None
                return self.market_data._process_rates(rates)
            except Exception as exc:
                logger.warning("%s: failed to fetch %s (%s) for historical analysis", sym, tf, exc)
                return None

        intraday_df = trading_df
        intraday_tf = params.get('intraday_timeframe')
        intraday_bars = params.get('intraday_bars')
        if intraday_tf and intraday_tf != daily_tf:
            try:
                bars = int(intraday_bars or self._approx_bars_per_day(intraday_tf) or 100)
                intraday_df = await self.market_data.fetch_data(symbol, intraday_tf, bars)
            except Exception as exc:
                logger.warning("%s: failed to fetch %s intraday data (%s); falling back to trading timeframe", symbol, intraday_tf, exc)
                intraday_df = trading_df

        intraday_today = self._intraday_today(intraday_df, intraday_tf or daily_tf)

        snapshot = self.historical_analyzer.refresh_symbol(
            symbol,
            fetcher=_fetcher,
            pip_size=pip_size,
            intraday_df=intraday_today,
        )
        if snapshot:
            self._historical_snapshots[symbol] = snapshot
            self._historical_last_refresh[symbol] = now
            return snapshot
        return self._historical_snapshots.get(symbol)

    def _intraday_today(self, df: Optional[pd.DataFrame], timeframe: str) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return None
        try:
            idx = pd.to_datetime(df.index)
        except Exception:
            return None
        if len(idx) == 0:
            return None
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        last_ts = idx[-1]
        start_of_day = last_ts.normalize()
        try:
            mask = idx >= start_of_day
            subset = df.loc[mask]
            if subset is not None and len(subset):
                return subset
        except Exception:
            pass
        approx = self._approx_bars_per_day(timeframe)
        if approx:
            return df.tail(int(approx))
        return None

    def _approx_bars_per_day(self, tf: str) -> Optional[int]:
        tfu = (tf or '').upper()
        mapping = {
            'M1': 1440,
            'M5': 288,
            'M15': 96,
            'M30': 48,
            'H1': 24,
            'H4': 6,
        }
        return mapping.get(tfu)

    def _log_talib_status(self):
        """Log whether TA-Lib is importable for pattern gates."""
        try:
            import talib  # type: ignore
            ver = getattr(talib, '__version__', 'unknown')
            logger.info(f"TA-Lib available (version {ver}); pattern gates active")
        except Exception:
            logger.warning("TA-Lib not available; pattern gates will fail-open (no blocking)")

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
            self.strategy = PurePriceActionStrategy(self.config, self.mt5_client)
            self.risk_manager = RiskManager(self.config, self.mt5_client)
            self.trade_logger = TradeLogger('trades.log')
            self._log_talib_status()

            # Load strategy state (duplicate prevention)
            try:
                if load_strategy_state(self.state_manager, self.strategy):
                    logger.info("Strategy: Duplicate prevention state restored")
            except Exception as e:
                logger.warning(f"Strategy: Failed to restore state: {e}")

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
                    'confidence': signal.confidence,
                    'signal_time': signal.timestamp,
                    'breakout_level': signal.breakout_level,
                    'pattern_score': getattr(signal, 'pattern_score', None),
                    'pattern_dir': getattr(signal, 'pattern_dir', None),
                    'pattern_primary': getattr(signal, 'pattern_primary', None),
                    'pattern_timeframe': getattr(signal, 'pattern_timeframe', None),
                    'hist_trend_bias': getattr(signal, 'hist_trend_bias', None),
                    'hist_breakout_rate': getattr(signal, 'hist_breakout_rate', None),
                    'hist_adr_progress': getattr(signal, 'hist_adr_progress', None),
                    'status': 'OPEN'
                }
                self.trade_logger.log_trade(trade)
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
        tfs = (symbol_config.get('timeframes') or ['M15'])
        timeframe = tfs[0]

        try:
            # Determine how many bars are needed to include today's Asia session
            def _required_bars_for_timeframe(tf: str) -> int:
                tfu = (tf or '').upper()
                if tfu == 'M1':
                    return 1500   # ~25h
                if tfu == 'M5':
                    return 350    # ~29h
                if tfu == 'M15':
                    return 120    # ~30h
                if tfu == 'M30':
                    return 60     # ~30h
                if tfu == 'H1':
                    return 36     # ~36h
                if tfu == 'H4':
                    return 12     # ~48h
                if tfu == 'D1':
                    return 7      # 1 week
                return 300        # safe default

            # Base timeframe data
            candles = await self.market_data.fetch_data(symbol, timeframe, _required_bars_for_timeframe(timeframe))
            if candles is None:
                return
            intraday_today = self._intraday_today(candles, timeframe)

            symbol_info = None
            pip_for_hist = None
            try:
                symbol_info = self.mt5_client.get_symbol_info(symbol)
                if symbol_info:
                    pip_for_hist = resolve_pip_size(symbol, symbol_info, self.config)
            except Exception:
                pip_for_hist = None

            # Fetch additional timeframes for multi-timeframe context (do not trade them here)
            mtf_context = {}
            try:
                for tf in tfs[1:]:
                    df_tf = await self.market_data.fetch_data(symbol, tf, _required_bars_for_timeframe(tf))
                    if df_tf is not None:
                        mtf_context[tf] = df_tf
            except Exception:
                mtf_context = {}

            hist_snapshot = None
            if pip_for_hist and pip_for_hist > 0:
                hist_snapshot = await self._get_historical_snapshot(symbol, pip_for_hist, intraday_today)

            signal = self.strategy.generate_signal(
                candles,
                symbol,
                mtf_context=mtf_context if mtf_context else None,
                historical_snapshot=hist_snapshot,
            )
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
                try:
                    current_positions = self.mt5_client.get_positions(symbol) if symbol else self.mt5_client.get_all_positions()
                    position_still_open = any(getattr(p, 'ticket', None) == position_id for p in (current_positions or []))
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
                                close_time=close_time
                            )
                            logger.info(f"Trade {trade.get('order_ticket')} closed ({status}) profit {profit}")
        except Exception as e:
            logger.error(f"Error monitoring open trades: {e}", exc_info=True)

    def _check_hourly_stats(self):
        """Check if hour has changed and log hourly statistics."""
        try:
            now = datetime.now(timezone.utc)
            current_hour = now.hour

            if self._last_hour is not None and current_hour != self._last_hour:
                # Log previous hour's stats
                logger.info(f"")
                logger.info(f"[HOURLY STATS {self._last_hour:02d}:00-{current_hour:02d}:00 UTC]")
                logger.info(f"  Symbols processed: {self._hourly_stats['symbols_processed']}")
                logger.info(f"  Breakouts detected: {self._hourly_stats['breakouts_detected']} (bullish: {self._hourly_stats['breakouts_bullish']}, bearish: {self._hourly_stats['breakouts_bearish']})")
                logger.info(f"  Signals generated: {self._hourly_stats['signals_generated']}")
                logger.info(f"  Signals blocked: {self._hourly_stats['signals_blocked']}")

                if self._hourly_stats['block_reasons']:
                    for reason, count in sorted(self._hourly_stats['block_reasons'].items(), key=lambda x: x[1], reverse=True)[:3]:
                        logger.info(f"    - {reason}: {count}")

                logger.info(f"  Trades executed: {self._hourly_stats['trades_executed']}")

                open_count = len([t for t in self.trade_logger.trades if t.get('status') == 'OPEN'])
                logger.info(f"  Open positions: {open_count}")
                logger.info(f"")

                # Reset stats for new hour
                self._hourly_stats = {
                    'symbols_processed': 0,
                    'breakouts_detected': 0,
                    'breakouts_bullish': 0,
                    'breakouts_bearish': 0,
                    'signals_generated': 0,
                    'signals_blocked': 0,
                    'block_reasons': {},
                    'trades_executed': 0,
                    'spring_confirmations': 0,
                    'upthrust_confirmations': 0
                }

            self._last_hour = current_hour
        except Exception as e:
            logger.debug(f"Error checking hourly stats: {e}")

    async def run(self):
        self.running = True
        logger.info("Starting core trading loop")

        # Initialize tracking
        self._check_hourly_stats()

        loop_count = 0  # For periodic state persistence

        while self.running:
            try:
                # Ensure MT5 connectivity before processing
                if not self.mt5_client.is_connected():
                    logger.warning("MT5 disconnected; attempting reconnect before processing loop")
                    self.mt5_client.reconnect()
                    if not self.mt5_client.is_connected():
                        await asyncio.sleep(self.config.main_loop_interval)
                        continue

                # Check for hour transitions
                self._check_hourly_stats()

                await self.monitor_open_trades()
                for sym in self.config.symbols:
                    await self.process_symbol(sym)

                # Periodic state persistence (every 10 loops = ~50 seconds at 5s interval)
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
