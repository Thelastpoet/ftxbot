"""
Backtesting Module for Pure Price Action Breakout Strategy

Reuses the real bot logic (strategy.py, trailing_stop.py) with a simulated broker.
Run with: python backtester.py

Results are saved to backtester_results/ directory.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple

import pandas as pd

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).parent))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from strategy import PurePriceActionStrategy, TradingSignal
from trailing_stop import TrailingStopManager, TrailingState
from utils import resolve_pip_size
from risk_manager import RiskManager

# Import the Config class from main.py (same as live bot uses)
from main import Config as LiveConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Backtest configuration that wraps the live Config class.
    Adds backtest-specific settings (dates, spread simulation, etc.).
    """
    # Backtest period
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    end_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Account simulation
    initial_balance: float = 10000.0

    # Spread simulation (in pips, added to historical bars)
    simulated_spread_pips: float = 1.5

    # Slippage simulation (in pips)
    slippage_pips: float = 0.5

    # Commission per lot per side (total = 2x this per round trip)
    commission_per_lot: float = 3.5

    # The live config object (holds all strategy/risk settings)
    _live_config: Any = None

    # Expose commonly accessed attributes directly
    symbols: List[Dict] = field(default_factory=list)
    diagnostics: Any = None

    def __getattr__(self, name):
        """Delegate attribute access to live config for strategy settings."""
        if name.startswith('_') or name in ['start_date', 'end_date', 'initial_balance',
                                             'simulated_spread_pips', 'slippage_pips',
                                             'commission_per_lot', 'symbols']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if self._live_config is not None:
            return getattr(self._live_config, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def from_config_file(cls, config_path: str = "config.json", **overrides) -> "BacktestConfig":
        """Load config using the same Config class as main.py."""
        # Create live config (same as live bot)
        live_config = LiveConfig(config_path=config_path)

        # Create backtest config
        bt_config = cls()
        bt_config._live_config = live_config
        bt_config.symbols = live_config.symbols

        # Apply backtest-specific overrides
        for key, value in overrides.items():
            if hasattr(bt_config, key) and not key.startswith('_'):
                setattr(bt_config, key, value)
            elif hasattr(live_config, key):
                setattr(live_config, key, value)

        return bt_config


@dataclass
class BacktestDiagnostics:
    """Collects rejection and execution diagnostics during backtests."""
    reject_counts: Dict[str, int] = field(default_factory=dict)
    reject_by_symbol: Dict[str, Dict[str, int]] = field(default_factory=dict)
    signal_counts: Dict[str, int] = field(default_factory=dict)
    exec_reject_counts: Dict[str, int] = field(default_factory=dict)
    exec_reject_by_symbol: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def _inc(self, bucket: Dict[str, int], key: str):
        bucket[key] = bucket.get(key, 0) + 1

    def _inc_by_symbol(self, bucket: Dict[str, Dict[str, int]], symbol: str, key: str):
        if symbol not in bucket:
            bucket[symbol] = {}
        bucket[symbol][key] = bucket[symbol].get(key, 0) + 1

    def record_reject(self, reason: str, symbol: str):
        self._inc(self.reject_counts, reason)
        self._inc_by_symbol(self.reject_by_symbol, symbol, reason)

    def record_signal(self, symbol: str):
        self._inc(self.signal_counts, symbol)

    def record_exec_reject(self, reason: str, symbol: str):
        self._inc(self.exec_reject_counts, reason)
        self._inc_by_symbol(self.exec_reject_by_symbol, symbol, reason)


# ============================================================================
# Mock MT5 Types (for backtesting without live MT5)
# ============================================================================

SymbolInfo = namedtuple('SymbolInfo', [
    'name', 'point', 'digits', 'spread', 'volume_min', 'volume_max', 'volume_step',
    'trade_tick_size', 'trade_tick_value', 'trade_stops_level'
])

TickInfo = namedtuple('TickInfo', ['bid', 'ask', 'last', 'time'])

AccountInfo = namedtuple('AccountInfo', ['balance', 'equity', 'margin', 'margin_level'])


# Symbol specifications (common pairs)
SYMBOL_SPECS = {
    # tick_value here represents the per-pip value for 1 lot (USD)
    'EURUSD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 10.0},
    'GBPUSD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 10.0},
    'USDJPY': {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1},
    'USDCHF': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 11.0},
    'AUDUSD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 10.0},
    'USDCAD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 7.5},
    'NZDUSD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 10.0},
    'EURJPY': {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1},
    'GBPJPY': {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1},
    'AUDJPY': {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1},
    'NZDJPY': {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1},
    'EURGBP': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 12.5},
    'EURAUD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 6.5},
    'EURCAD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 7.5},
    'GBPAUD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 6.5},
    'GBPCAD': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 7.5},
    'GBPCHF': {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 11.0},
    'XAUUSD': {'point': 0.01, 'digits': 2, 'pip_size': 0.01, 'tick_value': 1.0},
}


# ============================================================================
# Backtest Broker (Mock MT5 Client)
# ============================================================================

class BacktestBroker:
    """
    Simulated broker that mimics MT5 client interface.
    Allows PurePriceActionStrategy to work without modification.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.margin = 0.0

        # Current bar data for each symbol/timeframe
        self._current_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._current_bar_idx: Dict[str, int] = {}
        self._current_time: datetime = config.start_date

        # Order tracking
        self._next_ticket = 1000
        self._positions: Dict[int, Dict] = {}

    def set_current_data(self, symbol: str, timeframe: str, data: pd.DataFrame, bar_idx: int):
        """Set the current visible data for a symbol/timeframe."""
        if symbol not in self._current_data:
            self._current_data[symbol] = {}
        self._current_data[symbol][timeframe] = data.iloc[:bar_idx + 1]
        self._current_bar_idx[symbol] = bar_idx
        if len(data) > bar_idx:
            self._current_time = data.index[bar_idx]

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol specifications."""
        specs = SYMBOL_SPECS.get(symbol)
        if not specs:
            # Try to derive from symbol name
            if 'JPY' in symbol:
                specs = {'point': 0.001, 'digits': 3, 'pip_size': 0.01, 'tick_value': 9.1}
            elif 'XAU' in symbol:
                specs = {'point': 0.01, 'digits': 2, 'pip_size': 0.01, 'tick_value': 1.0}
            else:
                specs = {'point': 0.00001, 'digits': 5, 'pip_size': 0.0001, 'tick_value': 10.0}

        point = float(specs['point'])
        pip_size = float(specs.get('pip_size', point * 10.0))
        pip_value = float(specs.get('tick_value', 10.0))
        # Convert per-pip value to MT5-style trade_tick_value (per point)
        trade_tick_size = point
        trade_tick_value = pip_value * trade_tick_size / pip_size if pip_size > 0 else pip_value
        trade_stops_level = int(specs.get('trade_stops_level', 0))

        return SymbolInfo(
            name=symbol,
            point=point,
            digits=specs['digits'],
            spread=int(self.config.simulated_spread_pips * 10),
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_tick_size=trade_tick_size,
            trade_tick_value=trade_tick_value,
            trade_stops_level=trade_stops_level
        )

    def get_symbol_info_tick(self, symbol: str) -> Optional[TickInfo]:
        """Get current tick (bid/ask) from the current bar."""
        if symbol not in self._current_data:
            return None

        # Get entry timeframe data
        entry_tf = self._get_entry_timeframe(symbol)
        data = self._current_data[symbol].get(entry_tf)
        if data is None or data.empty:
            return None

        last_bar = data.iloc[-1]
        close = float(last_bar['close'])

        # Simulate spread
        specs = SYMBOL_SPECS.get(symbol, {'pip_size': 0.0001})
        pip = specs.get('pip_size', 0.0001)
        half_spread = (self.config.simulated_spread_pips * pip) / 2

        bid = close - half_spread
        ask = close + half_spread

        return TickInfo(bid=bid, ask=ask, last=close, time=self._current_time)

    def _get_entry_timeframe(self, symbol: str) -> str:
        """Get entry timeframe for symbol from config."""
        for s in self.config.symbols:
            if s.get('name') == symbol:
                return s.get('entry_timeframe', 'M15')
        return 'M15'

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        margin_level = (self.equity / self.margin * 100) if self.margin > 0 else 0
        return AccountInfo(
            balance=self.balance,
            equity=self.equity,
            margin=self.margin,
            margin_level=margin_level
        )


# ============================================================================
# Data Loading
# ============================================================================

TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
    'H1': 60, 'H4': 240, 'D1': 1440
}

if MT5_AVAILABLE:
    MT5_TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
else:
    MT5_TIMEFRAMES = {}


def load_mt5_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Load historical data from MT5."""
    if not MT5_AVAILABLE:
        return None

    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return None

    tf = MT5_TIMEFRAMES.get(timeframe)
    if tf is None:
        logger.error(f"Unknown timeframe: {timeframe}")
        return None

    # Request extra bars for lookback
    rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        logger.warning(f"No data for {symbol} {timeframe}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    if 'volume' in df.columns:
        df = df[df['volume'] > 0]
    df.sort_index(inplace=True)

    return df


def load_csv_data(symbol: str, timeframe: str, data_dir: str = "historical_data") -> Optional[pd.DataFrame]:
    """Load historical data from CSV file."""
    filename = f"{data_dir}/{symbol}_{timeframe}.csv"
    if not os.path.exists(filename):
        return None

    try:
        df = pd.read_csv(filename, parse_dates=['time'], index_col='time')
        df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index
        if 'volume' in df.columns:
            df = df[df['volume'] > 0]
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None


def load_data(symbol: str, timeframe: str, start: datetime, end: datetime,
              data_dir: str = "historical_data") -> Optional[pd.DataFrame]:
    """Load data from MT5 or CSV fallback."""
    # Try MT5 first
    df = load_mt5_data(symbol, timeframe, start, end)
    if df is not None:
        return df

    # Fallback to CSV
    df = load_csv_data(symbol, timeframe, data_dir)
    if df is not None:
        # Filter to date range
        mask = (df.index >= start) & (df.index <= end)
        return df[mask]

    return None


def align_timeframes(entry_data: pd.DataFrame, structure_data: pd.DataFrame,
                     trend_data: pd.DataFrame, entry_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get structure and trend data visible at entry_idx time.
    Returns completed bars only (excludes current forming bar).
    """
    if entry_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    current_time = entry_data.index[entry_idx]

    # Structure data: all completed bars before current entry time
    if structure_data is not None and not structure_data.empty:
        struct_visible = structure_data[structure_data.index < current_time]
    else:
        struct_visible = pd.DataFrame()

    # Trend data: all completed bars before current entry time
    if trend_data is not None and not trend_data.empty:
        trend_visible = trend_data[trend_data.index < current_time]
    else:
        trend_visible = pd.DataFrame()

    return struct_visible, trend_visible


# ============================================================================
# Trade Simulation
# ============================================================================

@dataclass
class BacktestTrade:
    """Represents a single backtest trade."""
    ticket: int
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    signal_reason: str
    breakout_level: float

    # Exit info (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "TRAILING", "TIME_EXIT"

    # P&L
    profit_pips: float = 0.0
    profit_money: float = 0.0
    commission: float = 0.0

    # Trailing stop tracking
    trailing_state: str = "INITIAL"
    trailing_sl: Optional[float] = None
    mfe_pips: float = 0.0  # Maximum Favorable Excursion
    mae_pips: float = 0.0  # Maximum Adverse Excursion


class BacktestEngine:
    """
    Core backtesting engine that simulates trade execution.
    Uses real PurePriceActionStrategy and TrailingStopManager.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.broker = BacktestBroker(config)
        self.strategy = PurePriceActionStrategy(config, mt5_client=self.broker)
        self.risk_manager = RiskManager(config, self.broker)
        self.trailing_manager = TrailingStopManager()
        if self.config.diagnostics is None:
            self.config.diagnostics = BacktestDiagnostics()
        self.diagnostics = self.config.diagnostics

        self.trades: List[BacktestTrade] = []
        self.open_positions: Dict[int, BacktestTrade] = {}
        self._next_ticket = 1000

        # Equity curve tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.peak_equity = config.initial_balance
        self.max_drawdown = 0.0
        self._sync_broker_account()

    def run(self, symbols: Optional[List[str]] = None) -> List[BacktestTrade]:
        """Run backtest for specified symbols."""
        if symbols is None:
            symbols = [s['name'] for s in self.config.symbols]

        logger.info(f"Starting backtest: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial balance: ${self.config.initial_balance:,.2f}")

        # Load all data upfront
        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Calculate lookback needed (extra bars for indicators)
        lookback_days = 30  # Extra days for ATR, EMA calculations
        data_start = self.config.start_date - timedelta(days=lookback_days)

        for symbol in symbols:
            symbol_config = next((s for s in self.config.symbols if s['name'] == symbol), {})
            entry_tf = symbol_config.get('entry_timeframe', 'M15')
            structure_tf = symbol_config.get('structure_timeframe', 'H1')
            trend_tf = symbol_config.get('trend_timeframe', 'H4')

            all_data[symbol] = {}

            for tf in [entry_tf, structure_tf, trend_tf]:
                data = load_data(symbol, tf, data_start, self.config.end_date)
                if data is not None and not data.empty:
                    all_data[symbol][tf] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol} {tf}")
                else:
                    logger.warning(f"No data for {symbol} {tf}")

        # Check if we have data
        valid_symbols = [s for s in symbols if s in all_data and all_data[s]]
        if not valid_symbols:
            logger.error("No data available for any symbol")
            return []

        # Simulate bar by bar on entry timeframe
        for symbol in valid_symbols:
            self._run_symbol(symbol, all_data[symbol])

        # Close any remaining open positions at end
        for trade in list(self.open_positions.values()):
            self._close_position(trade, self.config.end_date, trade.entry_price, "END_OF_TEST")

        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return self.trades

    def _sync_broker_account(self):
        """Keep broker account info in sync with engine state."""
        self.broker.balance = float(self.balance)
        self.broker.equity = float(self.equity)

    @staticmethod
    def _is_rollover_period(current_time: datetime) -> bool:
        """Match live bot rollover window: 21:55 - 23:05 UTC."""
        hour = current_time.hour
        minute = current_time.minute
        if hour == 21 and minute >= 55:
            return True
        if hour == 22:
            return True
        if hour == 23 and minute <= 5:
            return True
        return False

    def _pip_value(self, symbol: str) -> float:
        """Compute pip value per lot using MT5-style tick values."""
        info = self.broker.get_symbol_info(symbol)
        if not info or getattr(info, 'trade_tick_size', 0) <= 0:
            return 0.0
        pip_size = resolve_pip_size(symbol, info, self.config)
        per_unit = float(info.trade_tick_value) / float(info.trade_tick_size)
        return float(per_unit * pip_size)

    def _run_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]):
        """Run backtest for a single symbol."""
        symbol_config = next((s for s in self.config.symbols if s['name'] == symbol), {})
        entry_tf = symbol_config.get('entry_timeframe', 'M15')
        structure_tf = symbol_config.get('structure_timeframe', 'H1')
        trend_tf = symbol_config.get('trend_timeframe', 'H4')

        entry_data = data.get(entry_tf)
        structure_data = data.get(structure_tf, pd.DataFrame())
        trend_data = data.get(trend_tf, pd.DataFrame())

        if entry_data is None or entry_data.empty:
            return

        # Get pip size for this symbol
        pip_size = self._get_pip_size(symbol)

        # Find start index (after warmup period)
        start_time = self.config.start_date
        start_idx = entry_data.index.searchsorted(start_time)
        start_idx = max(start_idx, self.config.lookback_period + 50)  # Ensure enough history

        logger.info(f"Running {symbol}: {len(entry_data) - start_idx} bars to process")

        for bar_idx in range(start_idx, len(entry_data)):
            current_time = entry_data.index[bar_idx]
            current_bar = entry_data.iloc[bar_idx]

            # Update broker state
            self.broker.set_current_data(symbol, entry_tf, entry_data, bar_idx)
            self.broker._current_time = current_time

            # Get aligned higher timeframe data
            struct_visible, trend_visible = align_timeframes(
                entry_data, structure_data, trend_data, bar_idx
            )

            # Update open positions (SL/TP always apply; trailing skipped during rollover)
            allow_trailing = not self._is_rollover_period(current_time)
            self._update_positions(symbol, current_bar, current_time, pip_size, allow_trailing=allow_trailing)

            if self._is_rollover_period(current_time):
                self._update_equity(current_time)
                continue

            # Sync equity before sizing any new trades (no curve record)
            self._update_equity(current_time, record=False)

            # Check for new signals
            visible_data = entry_data.iloc[:bar_idx + 1]

            # Add structure data to broker for strategy access
            if not struct_visible.empty:
                self.broker.set_current_data(symbol, structure_tf, struct_visible, len(struct_visible) - 1)
            if not trend_visible.empty:
                self.broker.set_current_data(symbol, trend_tf, trend_visible, len(trend_visible) - 1)

            signal = self.strategy.generate_signal(
                visible_data,
                symbol,
                trend_data=trend_visible if not trend_visible.empty else None,
                structure_data=struct_visible if not struct_visible.empty else None,
                trend_timeframe=trend_tf
            )

            if signal:
                self.diagnostics.record_signal(symbol)
                direction = "BUY" if signal.type == 0 else "SELL"
                if not self._has_open_position(symbol, direction=direction):
                    self._execute_signal(symbol, signal, current_time, pip_size)

            # Record equity
            self._update_equity(current_time, record=True)

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol."""
        info = self.broker.get_symbol_info(symbol)
        if info:
            return resolve_pip_size(symbol, info, self.config)
        specs = SYMBOL_SPECS.get(symbol, {})
        return float(specs.get('pip_size', 0.0001))

    def _has_open_position(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if there's an open position for symbol (optionally specific direction)."""
        for trade in self.open_positions.values():
            if trade.symbol == symbol:
                if direction is None or trade.direction == direction:
                    return True
        return False

    def _execute_signal(self, symbol: str, signal: TradingSignal, current_time: datetime, pip_size: float):
        """Execute a trading signal."""
        direction = "BUY" if signal.type == 0 else "SELL"
        def exec_reject(reason: str):
            if self.diagnostics:
                self.diagnostics.record_exec_reject(reason, symbol)

        sym_info = self.broker.get_symbol_info(symbol)
        if not sym_info:
            exec_reject("EXEC_NO_SYMBOL_INFO")
            return

        pip = float(pip_size)
        if pip <= 0:
            exec_reject("EXEC_BAD_PIP")
            return

        tick = self.broker.get_symbol_info_tick(symbol)
        if not tick:
            exec_reject("EXEC_NO_TICK")
            return

        exec_price = tick.ask if signal.type == 0 else tick.bid
        actual_sl_pips = abs(exec_price - signal.stop_loss) / pip

        # Enforce minimum SL distance (same as live)
        min_sl_pips = float(getattr(self.config, 'min_stop_loss_pips', 0.0))
        for sc in getattr(self.config, 'symbols', []) or []:
            if sc.get('name') == symbol:
                min_sl_pips = float(sc.get('min_stop_loss_pips', min_sl_pips))
                break
        if min_sl_pips > 0 and actual_sl_pips < min_sl_pips * 0.95:
            exec_reject("EXEC_SL_TOO_TIGHT")
            return

        if not self.risk_manager.check_risk_limits():
            exec_reject("EXEC_RISK_LIMIT")
            return

        volume = self.risk_manager.calculate_position_size(symbol, actual_sl_pips)
        if volume <= 0:
            exec_reject("EXEC_BAD_SIZE")
            return

        if not self.risk_manager.validate_trade_parameters(
            symbol, volume, signal.stop_loss, signal.take_profit, signal.type
        ):
            exec_reject("EXEC_INVALID_PARAMS")
            return

        # Final spread guard before execution (live-equivalent)
        current_spread_pips = abs(float(tick.ask) - float(tick.bid)) / pip
        spread_guard = getattr(self.config, 'spread_guard_pips_default', None)
        for sc in getattr(self.config, 'symbols', []) or []:
            if sc.get('name') == symbol:
                spread_guard = sc.get('spread_guard_pips', spread_guard)
                break
        if spread_guard is not None and float(spread_guard) > 0 and current_spread_pips > float(spread_guard):
            exec_reject("EXEC_SPREAD_GUARD")
            return

        # Apply slippage to entry
        slippage = self.config.slippage_pips * pip
        entry_price = exec_price + slippage if direction == "BUY" else exec_price - slippage

        # Calculate commission
        commission = self.config.commission_per_lot * volume * 2  # Round trip

        ticket = self._next_ticket
        self._next_ticket += 1

        trade = BacktestTrade(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_time=current_time,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            volume=volume,
            signal_reason=signal.reason,
            breakout_level=signal.breakout_level,
            commission=commission,
            trailing_sl=signal.stop_loss
        )

        self.open_positions[ticket] = trade

        # Register with trailing stop manager
        self.trailing_manager.register_position(
            position_id=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=current_time,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            pip_size=pip_size
        )

        logger.debug(f"Opened {direction} {symbol} @ {entry_price:.5f} SL:{signal.stop_loss:.5f} TP:{signal.take_profit:.5f}")

    def _calculate_position_size(self, symbol: str, sl_pips: float, pip_size: float) -> float:
        """Calculate position size based on risk."""
        if self.config.fixed_lot_size:
            return self.config.fixed_lot_size

        risk_amount = self.equity * self.config.risk_per_trade

        pip_value = self._pip_value(symbol)

        if sl_pips <= 0 or pip_value <= 0:
            return 0.0

        position_size = risk_amount / (sl_pips * pip_value)

        # Round to 0.01 lots and clamp
        position_size = round(position_size / 0.01) * 0.01
        position_size = max(0.01, min(position_size, 10.0))

        return position_size

    def _update_positions(self, symbol: str, bar: pd.Series, current_time: datetime, pip_size: float, allow_trailing: bool):
        """Update open positions: check SL/TP hits and update trailing stops."""
        high = float(bar['high'])
        low = float(bar['low'])
        close = float(bar['close'])
        half_spread = (self.config.simulated_spread_pips * pip_size) / 2.0

        for ticket, trade in list(self.open_positions.items()):
            if trade.symbol != symbol:
                continue

            current_sl = trade.trailing_sl or trade.stop_loss

            # Update MFE/MAE
            if trade.direction == "BUY":
                bid_high = high - half_spread
                bid_low = low - half_spread
                profit_pips = (bid_high - trade.entry_price) / pip_size
                loss_pips = (trade.entry_price - bid_low) / pip_size
            else:
                ask_low = low + half_spread
                ask_high = high + half_spread
                profit_pips = (trade.entry_price - ask_low) / pip_size
                loss_pips = (ask_high - trade.entry_price) / pip_size

            trade.mfe_pips = max(trade.mfe_pips, profit_pips)
            trade.mae_pips = max(trade.mae_pips, loss_pips)

            # Check SL hit
            if trade.direction == "BUY":
                bid_low = low - half_spread
                if bid_low <= current_sl:
                    exit_price = current_sl - (self.config.slippage_pips * pip_size)
                    self._close_position(trade, current_time, exit_price, "SL" if current_sl == trade.stop_loss else "TRAILING")
                    continue
            else:
                ask_high = high + half_spread
                if ask_high >= current_sl:
                    exit_price = current_sl + (self.config.slippage_pips * pip_size)
                    self._close_position(trade, current_time, exit_price, "SL" if current_sl == trade.stop_loss else "TRAILING")
                    continue

            # Check TP hit
            if trade.direction == "BUY":
                bid_high = high - half_spread
                if bid_high >= trade.take_profit:
                    self._close_position(trade, current_time, trade.take_profit, "TP")
                    continue
            else:
                ask_low = low + half_spread
                if ask_low <= trade.take_profit:
                    self._close_position(trade, current_time, trade.take_profit, "TP")
                    continue

            if allow_trailing:
                tick = self.broker.get_symbol_info_tick(symbol)
                if tick:
                    current_price = float(tick.bid) if trade.direction == "BUY" else float(tick.ask)
                else:
                    current_price = close
                # Update trailing stop
                result = self.trailing_manager.update_position(ticket, current_price, current_time)
                if result.get('new_sl'):
                    trade.trailing_sl = result['new_sl']
                    trade.trailing_state = result.get('state', trade.trailing_state)

                # Check time exit
                if result.get('should_close_time'):
                    self._close_position(trade, current_time, current_price, "TIME_EXIT")

    def _close_position(self, trade: BacktestTrade, exit_time: datetime, exit_price: float, reason: str):
        """Close a position and calculate P&L."""
        pip_size = self._get_pip_size(trade.symbol)

        # Calculate profit
        if trade.direction == "BUY":
            profit_pips = (exit_price - trade.entry_price) / pip_size
        else:
            profit_pips = (trade.entry_price - exit_price) / pip_size

        pip_value = self._pip_value(trade.symbol)

        profit_money = profit_pips * pip_value * trade.volume - trade.commission

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.profit_pips = profit_pips
        trade.profit_money = profit_money

        # Update balance
        self.balance += profit_money
        self._sync_broker_account()

        # Move to completed trades
        self.trades.append(trade)
        del self.open_positions[trade.ticket]
        self.trailing_manager.unregister_position(trade.ticket)

        logger.debug(f"Closed {trade.direction} {trade.symbol} @ {exit_price:.5f} ({reason}) P/L: {profit_pips:.1f} pips (${profit_money:.2f})")

    def _update_equity(self, current_time: datetime, record: bool = True):
        """Update equity and drawdown tracking; optionally record curve point."""
        # Calculate unrealized P&L
        unrealized = 0.0
        for trade in self.open_positions.values():
            tick = self.broker.get_symbol_info_tick(trade.symbol)
            if tick:
                pip_size = self._get_pip_size(trade.symbol)
                pip_value = self._pip_value(trade.symbol)

                if trade.direction == "BUY":
                    pips = (tick.bid - trade.entry_price) / pip_size
                else:
                    pips = (trade.entry_price - tick.ask) / pip_size

                unrealized += pips * pip_value * trade.volume

        self.equity = self.balance + unrealized
        self._sync_broker_account()
        if record:
            self.equity_curve.append((current_time, self.equity))

        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_dd)


# ============================================================================
# Results Generation
# ============================================================================

@dataclass
class BacktestResults:
    """Backtest performance statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_profit: float = 0.0
    total_pips: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_pips_per_trade: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0

    avg_winner: float = 0.0
    avg_loser: float = 0.0
    avg_winner_pips: float = 0.0
    avg_loser_pips: float = 0.0

    largest_winner: float = 0.0
    largest_loser: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    avg_trade_duration: timedelta = field(default_factory=timedelta)
    avg_mfe_pips: float = 0.0
    avg_mae_pips: float = 0.0

    # By exit reason
    tp_exits: int = 0
    sl_exits: int = 0
    trailing_exits: int = 0
    time_exits: int = 0

    # By symbol
    by_symbol: Dict[str, Dict] = field(default_factory=dict)

    # Equity curve
    initial_balance: float = 0.0
    final_balance: float = 0.0
    return_pct: float = 0.0


def calculate_results(trades: List[BacktestTrade], engine: BacktestEngine) -> BacktestResults:
    """Calculate backtest statistics from trade list."""
    results = BacktestResults()

    if not trades:
        return results

    results.total_trades = len(trades)
    results.initial_balance = engine.config.initial_balance
    results.final_balance = engine.balance
    results.return_pct = ((engine.balance - engine.config.initial_balance) / engine.config.initial_balance) * 100
    results.max_drawdown_pct = engine.max_drawdown * 100

    winners = [t for t in trades if t.profit_money > 0]
    losers = [t for t in trades if t.profit_money <= 0]

    results.winning_trades = len(winners)
    results.losing_trades = len(losers)
    results.win_rate = (len(winners) / len(trades)) * 100 if trades else 0

    results.total_profit = sum(t.profit_money for t in trades)
    results.total_pips = sum(t.profit_pips for t in trades)
    results.avg_profit_per_trade = results.total_profit / len(trades) if trades else 0
    results.avg_pips_per_trade = results.total_pips / len(trades) if trades else 0

    results.gross_profit = sum(t.profit_money for t in winners)
    results.gross_loss = abs(sum(t.profit_money for t in losers))
    results.profit_factor = results.gross_profit / results.gross_loss if results.gross_loss > 0 else float('inf')

    results.avg_winner = results.gross_profit / len(winners) if winners else 0
    results.avg_loser = results.gross_loss / len(losers) if losers else 0
    results.avg_winner_pips = sum(t.profit_pips for t in winners) / len(winners) if winners else 0
    results.avg_loser_pips = abs(sum(t.profit_pips for t in losers)) / len(losers) if losers else 0

    # Largest winner/loser should only consider actual winners/losers
    if winners:
        results.largest_winner = max(t.profit_money for t in winners)
    if losers:
        results.largest_loser = min(t.profit_money for t in losers)  # Most negative

    # Average trade duration
    durations = [(t.exit_time - t.entry_time) for t in trades if t.exit_time]
    if durations:
        avg_seconds = sum(d.total_seconds() for d in durations) / len(durations)
        results.avg_trade_duration = timedelta(seconds=avg_seconds)

    results.avg_mfe_pips = sum(t.mfe_pips for t in trades) / len(trades) if trades else 0
    results.avg_mae_pips = sum(t.mae_pips for t in trades) / len(trades) if trades else 0

    # By exit reason
    for t in trades:
        if t.exit_reason == "TP":
            results.tp_exits += 1
        elif t.exit_reason == "SL":
            results.sl_exits += 1
        elif t.exit_reason == "TRAILING":
            results.trailing_exits += 1
        elif t.exit_reason == "TIME_EXIT":
            results.time_exits += 1

    # By symbol
    symbols = set(t.symbol for t in trades)
    for symbol in symbols:
        sym_trades = [t for t in trades if t.symbol == symbol]
        sym_winners = [t for t in sym_trades if t.profit_money > 0]
        results.by_symbol[symbol] = {
            'total': len(sym_trades),
            'winners': len(sym_winners),
            'win_rate': (len(sym_winners) / len(sym_trades)) * 100 if sym_trades else 0,
            'profit': sum(t.profit_money for t in sym_trades),
            'pips': sum(t.profit_pips for t in sym_trades)
        }

    return results


def save_results(trades: List[BacktestTrade], results: BacktestResults,
                 engine: BacktestEngine, output_dir: str = "backtester_results"):
    """Save backtest results to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BACKTEST RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Period: {engine.config.start_date.date()} to {engine.config.end_date.date()}\n")
        f.write(f"Initial Balance: ${results.initial_balance:,.2f}\n")
        f.write(f"Final Balance: ${results.final_balance:,.2f}\n")
        f.write(f"Return: {results.return_pct:.2f}%\n")
        f.write(f"Max Drawdown: {results.max_drawdown_pct:.2f}%\n\n")

        f.write("-" * 40 + "\n")
        f.write("TRADE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Trades: {results.total_trades}\n")
        f.write(f"Winners: {results.winning_trades} ({results.win_rate:.1f}%)\n")
        f.write(f"Losers: {results.losing_trades}\n\n")

        f.write(f"Total Profit: ${results.total_profit:,.2f}\n")
        f.write(f"Total Pips: {results.total_pips:.1f}\n")
        f.write(f"Avg Profit/Trade: ${results.avg_profit_per_trade:.2f}\n")
        f.write(f"Avg Pips/Trade: {results.avg_pips_per_trade:.1f}\n\n")

        f.write(f"Gross Profit: ${results.gross_profit:,.2f}\n")
        f.write(f"Gross Loss: ${results.gross_loss:,.2f}\n")
        f.write(f"Profit Factor: {results.profit_factor:.2f}\n\n")

        f.write(f"Avg Winner: ${results.avg_winner:.2f} ({results.avg_winner_pips:.1f} pips)\n")
        f.write(f"Avg Loser: ${results.avg_loser:.2f} ({results.avg_loser_pips:.1f} pips)\n")
        f.write(f"Largest Winner: ${results.largest_winner:.2f}\n")
        f.write(f"Largest Loser: ${results.largest_loser:.2f}\n\n")

        f.write(f"Avg MFE: {results.avg_mfe_pips:.1f} pips\n")
        f.write(f"Avg MAE: {results.avg_mae_pips:.1f} pips\n")
        f.write(f"Avg Duration: {results.avg_trade_duration}\n\n")

        f.write("-" * 40 + "\n")
        f.write("EXIT REASONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Take Profit: {results.tp_exits}\n")
        f.write(f"Stop Loss: {results.sl_exits}\n")
        f.write(f"Trailing Stop: {results.trailing_exits}\n")
        f.write(f"Time Exit: {results.time_exits}\n\n")

        f.write("-" * 40 + "\n")
        f.write("BY SYMBOL\n")
        f.write("-" * 40 + "\n")
        for symbol, stats in sorted(results.by_symbol.items()):
            f.write(f"{symbol}: {stats['total']} trades, {stats['win_rate']:.1f}% win, "
                   f"${stats['profit']:.2f}, {stats['pips']:.1f} pips\n")

    logger.info(f"Saved summary to {summary_file}")

    # Save trades to JSON
    trades_file = os.path.join(output_dir, f"trades_{timestamp}.json")
    trades_data = []
    for t in trades:
        trades_data.append({
            'ticket': t.ticket,
            'symbol': t.symbol,
            'direction': t.direction,
            'entry_time': t.entry_time.isoformat(),
            'entry_price': t.entry_price,
            'stop_loss': t.stop_loss,
            'take_profit': t.take_profit,
            'volume': t.volume,
            'signal_reason': t.signal_reason,
            'breakout_level': t.breakout_level,
            'exit_time': t.exit_time.isoformat() if t.exit_time else None,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'profit_pips': t.profit_pips,
            'profit_money': t.profit_money,
            'commission': t.commission,
            'trailing_state': t.trailing_state,
            'mfe_pips': t.mfe_pips,
            'mae_pips': t.mae_pips
        })

    with open(trades_file, 'w') as f:
        json.dump(trades_data, f, indent=2)

    logger.info(f"Saved trades to {trades_file}")

    # Save equity curve to CSV
    equity_file = os.path.join(output_dir, f"equity_{timestamp}.csv")
    if engine.equity_curve:
        df = pd.DataFrame(engine.equity_curve, columns=['time', 'equity'])
        df.to_csv(equity_file, index=False)
        logger.info(f"Saved equity curve to {equity_file}")

    # Save diagnostics
    diagnostics_file = os.path.join(output_dir, f"diagnostics_{timestamp}.txt")
    diag = getattr(engine, "diagnostics", None)
    if diag:
        def fmt_top(bucket: Dict[str, int], limit: int = 10):
            items = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
            return items[:limit]

        total_signals = sum(diag.signal_counts.values())
        total_exec = len(trades)
        with open(diagnostics_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("BACKTEST DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total Signals: {total_signals}\n")
            f.write(f"Executed Trades: {total_exec}\n")
            conv = (total_exec / total_signals * 100) if total_signals > 0 else 0
            f.write(f"Signal->Trade Conversion: {conv:.2f}%\n\n")

            f.write("-" * 40 + "\n")
            f.write("REJECTION REASONS (Strategy)\n")
            f.write("-" * 40 + "\n")
            if diag.reject_counts:
                for reason, count in fmt_top(diag.reject_counts, limit=20):
                    f.write(f"{reason}: {count}\n")
            else:
                f.write("No strategy rejections recorded.\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("REJECTION REASONS (Execution)\n")
            f.write("-" * 40 + "\n")
            if diag.exec_reject_counts:
                for reason, count in fmt_top(diag.exec_reject_counts, limit=20):
                    f.write(f"{reason}: {count}\n")
            else:
                f.write("No execution rejections recorded.\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("SIGNALS BY SYMBOL\n")
            f.write("-" * 40 + "\n")
            for symbol, count in sorted(diag.signal_counts.items()):
                f.write(f"{symbol}: {count}\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("TOP REJECTIONS BY SYMBOL (Strategy)\n")
            f.write("-" * 40 + "\n")
            for symbol, bucket in sorted(diag.reject_by_symbol.items()):
                top = fmt_top(bucket, limit=5)
                top_str = ", ".join([f"{r}={c}" for r, c in top]) if top else "None"
                f.write(f"{symbol}: {top_str}\n")

            f.write("\n" + "-" * 40 + "\n")
            f.write("TOP REJECTIONS BY SYMBOL (Execution)\n")
            f.write("-" * 40 + "\n")
            for symbol, bucket in sorted(diag.exec_reject_by_symbol.items()):
                top = fmt_top(bucket, limit=5)
                top_str = ", ".join([f"{r}={c}" for r, c in top]) if top else "None"
                f.write(f"{symbol}: {top_str}\n")

        logger.info(f"Saved diagnostics to {diagnostics_file}")

    return summary_file, trades_file


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run backtester with command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Pure Price Action Strategy")
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default='2024-01-01')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--balance', type=float, help='Initial balance', default=10000)
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to backtest (space or comma separated). Defaults to all in config.')
    parser.add_argument('--spread', type=float, help='Simulated spread in pips', default=1.5)
    parser.add_argument('--slippage', type=float, help='Slippage in pips', default=0.5)
    parser.add_argument('--output', type=str, help='Output directory', default='backtester_results')
    parser.add_argument('--config', type=str, help='Config file path', default='config.json')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    # Load config
    config = BacktestConfig.from_config_file(
        args.config,
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.balance,
        simulated_spread_pips=args.spread,
        slippage_pips=args.slippage
    )

    # Determine symbols
    symbols = [s['name'] for s in config.symbols]
    if args.symbols:
        # Support space or comma-separated symbols
        requested: List[str] = []
        for token in args.symbols:
            requested.extend([t.strip() for t in str(token).split(',') if t.strip()])
        requested = [s for s in requested if s]
        if requested:
            name_map = {s['name']: s for s in config.symbols}
            missing = [s for s in requested if s not in name_map]
            selected = [s for s in requested if s in name_map]
            if missing:
                logger.warning(f"Requested symbols not in config: {', '.join(missing)}")
            if not selected:
                raise SystemExit("No valid symbols specified. Check --symbols and config.")
            # Restrict config to selected symbols so per-symbol overrides stay aligned
            config.symbols = [name_map[s] for s in selected]
            symbols = selected

    print("\n" + "=" * 60)
    print("PURE PRICE ACTION BACKTESTER")
    print("=" * 60)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Balance: ${config.initial_balance:,.2f}")
    print(f"Spread: {config.simulated_spread_pips} pips | Slippage: {config.slippage_pips} pips")
    print("=" * 60 + "\n")

    # Run backtest
    engine = BacktestEngine(config)
    trades = engine.run(symbols)

    # Calculate and display results
    results = calculate_results(trades, engine)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Trades: {results.total_trades}")
    print(f"Win Rate: {results.win_rate:.1f}%")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Total Profit: ${results.total_profit:,.2f} ({results.total_pips:.1f} pips)")
    print(f"Final Balance: ${results.final_balance:,.2f} ({results.return_pct:+.2f}%)")
    print(f"Max Drawdown: {results.max_drawdown_pct:.2f}%")
    print("=" * 60 + "\n")

    # Save results
    summary_file, trades_file = save_results(trades, results, engine, args.output)

    print(f"Results saved to {args.output}/")

    # Cleanup MT5 if used
    if MT5_AVAILABLE and mt5.terminal_info():
        mt5.shutdown()


if __name__ == "__main__":
    main()
