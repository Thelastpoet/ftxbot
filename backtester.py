"""
Professional Backtesting Engine for Forex Trading Bot

Features:
- Bar-by-bar simulation with realistic execution
- Spread and slippage modeling
- Comprehensive statistics (win rate, profit factor, Sharpe, max drawdown)
- Trade-by-trade logging for analysis
- Parameter optimization support
- Out-of-sample validation

Usage:
    python backtester.py --symbol EURUSD --start 2023-01-01 --end 2024-12-31
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import math

import pandas as pd
import numpy as np

from trailing_stop import SYMBOL_TRAILING_CONFIGS, DEFAULT_TRAILING_CONFIG, create_backtest_trailing_manager

# Import MT5 for historical data
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("WARNING: MetaTrader5 not available. Use --data-file for CSV input.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str = "M15"
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"

    # Account settings
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.01  # 1%

    # Strategy parameters (to be optimized)
    lookback_period: int = 20
    swing_window: int = 5
    breakout_threshold_pips: float = 5.0
    breakout_threshold_atr_mult: Optional[float] = None  # If set, uses ATR*mult instead of fixed pips
    min_stop_loss_pips: float = 20.0
    stop_loss_buffer_pips: float = 15.0
    risk_reward_ratio: float = 2.0
    min_rr: float = 1.0

    # ATR settings
    atr_period: int = 14
    atr_sl_k: float = 0.6
    min_sl_buffer_pips: float = 10.0
    max_sl_pips: Optional[float] = None

    # Filters (research-backed defaults)
    use_trend_filter: bool = True  # Enable 200 EMA trend filter
    trend_ema_period: int = 200
    use_session_filter: bool = False
    session_start_utc: int = 8  # London open
    session_end_utc: int = 17   # NY afternoon
    use_retest_confirmation: bool = False
    retest_bars_max: int = 5  # Max bars to wait for retest

    # Execution modeling
    spread_pips: float = 1.5  # Average spread
    slippage_pips: float = 0.5  # Average slippage

    # Trailing stop settings
    use_trailing_stop: bool = False  # Enable trailing stop mode
    trailing_break_even_pips: float = 15.0  # Move to BE after this profit
    trailing_break_even_offset_pips: float = 1.0  # Offset from entry
    trailing_trigger_pips: float = 25.0  # Start trailing after this profit
    trailing_distance_pips: float = 15.0  # Trail this far behind price
    trailing_max_hours: float = 48.0  # Close after N hours (0=disabled)

    # Output
    output_dir: str = "backtest_results"


@dataclass
class Trade:
    """Represents a single trade in backtest."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    symbol: str = ""
    direction: str = ""  # "BUY" or "SELL"
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_TIME, CLOSED_TRAIL
    reason: str = ""
    breakout_level: float = 0.0

    # Metadata for analysis
    atr_at_entry: Optional[float] = None
    trend_direction: Optional[str] = None  # "UP", "DOWN", "FLAT"
    session: Optional[str] = None  # "LONDON", "NY", "OVERLAP", "ASIAN", "OFF"

    # Trailing stop tracking
    original_sl: Optional[float] = None  # Original SL before trailing
    current_sl: Optional[float] = None  # Current SL (may be trailed)
    trailing_state: str = "INITIAL"  # INITIAL, BREAK_EVEN, TRAILING
    max_favorable_pips: float = 0.0  # Maximum favorable excursion
    max_adverse_pips: float = 0.0  # Maximum adverse excursion
    sl_updates: int = 0  # Number of SL modifications


@dataclass
class BacktestResult:
    """Complete backtest results and statistics."""
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)

    # Core metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profitability
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    # Advanced metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)

    # Time analysis
    avg_trade_duration_hours: float = 0.0
    trades_per_month: float = 0.0


# =============================================================================
# Data Fetching
# =============================================================================

class DataFetcher:
    """Fetches historical data from MT5 or CSV files."""

    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1 if mt5 else 1,
        'M5': mt5.TIMEFRAME_M5 if mt5 else 5,
        'M15': mt5.TIMEFRAME_M15 if mt5 else 15,
        'M30': mt5.TIMEFRAME_M30 if mt5 else 30,
        'H1': mt5.TIMEFRAME_H1 if mt5 else 60,
        'H4': mt5.TIMEFRAME_H4 if mt5 else 240,
        'D1': mt5.TIMEFRAME_D1 if mt5 else 1440,
    }

    def __init__(self):
        self.mt5_initialized = False
        if mt5:
            self.mt5_initialized = mt5.initialize()
            if self.mt5_initialized:
                logger.info(f"MT5 initialized: {mt5.terminal_info().name}")

    def fetch_mt5_data(self, symbol: str, timeframe: str,
                       start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical data from MT5."""
        if not self.mt5_initialized:
            logger.error("MT5 not initialized")
            return None

        tf = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol: {symbol}")
            return None

        # Fetch rates
        rates = mt5.copy_rates_range(symbol, tf, start, end)

        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol} {timeframe} from {start} to {end}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
        return df

    def fetch_csv_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load historical data from CSV file."""
        try:
            df = pd.read_csv(filepath, parse_dates=['time'], index_col='time')
            df.index = df.index.tz_localize('UTC')
            logger.info(f"Loaded {len(df)} bars from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol specifications from MT5."""
        if not self.mt5_initialized:
            # Return defaults for common forex pairs
            return self._get_default_symbol_info(symbol)

        info = mt5.symbol_info(symbol)
        if info is None:
            return self._get_default_symbol_info(symbol)

        pip_size = info.point * 10 if info.digits in (5, 3) else info.point
        # MT5's trade_tick_value gives us the value of 1 tick (point) movement
        # Convert to pip value: pip_value = tick_value * (pip_size / point)
        tick_value = getattr(info, 'trade_tick_value', None)
        if tick_value and info.point > 0:
            pip_value_per_lot = tick_value * (pip_size / info.point)
        else:
            # Fallback to approximate values based on symbol
            pip_value_per_lot = self._get_default_symbol_info(symbol).get('pip_value_per_lot', 10.0)

        return {
            'point': info.point,
            'digits': info.digits,
            'pip_size': pip_size,
            'contract_size': info.trade_contract_size,
            'pip_value_per_lot': pip_value_per_lot,
            'volume_min': info.volume_min,
            'volume_step': info.volume_step,
            'volume_max': info.volume_max,
        }

    def _get_default_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Return default symbol info for common pairs.

        pip_value_per_lot: USD value of 1 pip movement per 1 standard lot
        - For XXX/USD pairs: $10 per pip per lot (exact)
        - For USD/XXX pairs: ~$10 per pip per lot (varies by rate)
        - For XXX/JPY pairs: ~$6.67 per pip per lot (1000 JPY / ~150 USD/JPY)
        - For cross pairs: varies by base currency conversion
        """
        symbol_upper = symbol.upper()

        # JPY pairs - pip size 0.01, pip value ~$6.67/lot
        if 'JPY' in symbol_upper:
            return {
                'point': 0.001,
                'digits': 3,
                'pip_size': 0.01,
                'contract_size': 100000,
                'pip_value_per_lot': 6.67,  # 1000 JPY / 150 USD/JPY rate
                'volume_min': 0.01,
                'volume_step': 0.01,
                'volume_max': 100.0,
            }
        # Gold (XAUUSD) - aligned with config.json pip_unit=0.01
        elif 'XAU' in symbol_upper:
            return {
                'point': 0.01,
                'digits': 2,
                'pip_size': 0.01,  # Gold pip = $0.01 movement (matches config pip_unit)
                'contract_size': 100,
                'pip_value_per_lot': 1.0,  # $1 per pip for 100oz at $0.01/pip
                'volume_min': 0.01,
                'volume_step': 0.01,
                'volume_max': 100.0,
            }
        # Pairs ending in USD (EURUSD, GBPUSD, etc.) - exact $10/pip/lot
        elif symbol_upper.endswith('USD'):
            return {
                'point': 0.00001,
                'digits': 5,
                'pip_size': 0.0001,
                'contract_size': 100000,
                'pip_value_per_lot': 10.0,
                'volume_min': 0.01,
                'volume_step': 0.01,
                'volume_max': 100.0,
            }
        # USD/XXX pairs (USDCAD, USDCHF, etc.) - approx $10/pip/lot
        elif symbol_upper.startswith('USD'):
            return {
                'point': 0.00001,
                'digits': 5,
                'pip_size': 0.0001,
                'contract_size': 100000,
                'pip_value_per_lot': 10.0,  # Approximate, varies slightly
                'volume_min': 0.01,
                'volume_step': 0.01,
                'volume_max': 100.0,
            }
        # Cross pairs (EURGBP, AUDCAD, etc.) - use approximate values
        else:
            # Most crosses have pip value between $8-12
            return {
                'point': 0.00001,
                'digits': 5,
                'pip_size': 0.0001,
                'contract_size': 100000,
                'pip_value_per_lot': 10.0,  # Approximate average
                'volume_min': 0.01,
                'volume_step': 0.01,
                'volume_max': 100.0,
            }

    def shutdown(self):
        """Cleanup MT5 connection."""
        if self.mt5_initialized and mt5:
            mt5.shutdown()


# =============================================================================
# Strategy Logic (mirrors strategy.py but for backtesting)
# =============================================================================

class BacktestStrategy:
    """
    Breakout strategy implementation for backtesting.
    Mirrors the live strategy but with added filter options.
    """

    def __init__(self, config: BacktestConfig, symbol_info: Dict[str, Any]):
        self.config = config
        self.symbol_info = symbol_info
        self.pip = symbol_info['pip_size']

        # State tracking
        self._last_breakout_bar = {}

    def calculate_ema(self, closes: pd.Series, period: int) -> pd.Series:
        """Calculate EMA for trend filter."""
        return closes.ewm(span=period, adjust=False).mean()

    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close']

        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period, min_periods=period).mean()

    def find_swing_points(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find swing high and low indices."""
        window = self.config.swing_window
        highs = data['high'].values
        lows = data['low'].values
        n = len(data)

        swing_highs = []
        swing_lows = []

        for i in range(window, n - window):
            # Swing high
            left_h = highs[i - window:i]
            right_h = highs[i + 1:i + window + 1]
            if len(left_h) > 0 and len(right_h) > 0:
                if highs[i] > left_h.max() and highs[i] > right_h.max():
                    swing_highs.append(i)

            # Swing low
            left_l = lows[i - window:i]
            right_l = lows[i + 1:i + window + 1]
            if len(left_l) > 0 and len(right_l) > 0:
                if lows[i] < left_l.min() and lows[i] < right_l.min():
                    swing_lows.append(i)

        return swing_highs, swing_lows

    def calculate_sr_levels(self, data: pd.DataFrame,
                            swing_highs: List[int],
                            swing_lows: List[int]) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels from swing points."""
        proximity = 10.0 * self.pip

        resistance = []
        support = []
        max_levels = 5

        # Get recent resistance levels
        for i in reversed(swing_highs[-50:]):
            level = float(data.iloc[i]['high'])
            if not any(abs(level - x) <= proximity for x in resistance):
                resistance.append(level)
            if len(resistance) >= max_levels:
                break

        # Get recent support levels
        for i in reversed(swing_lows[-50:]):
            level = float(data.iloc[i]['low'])
            if not any(abs(level - x) <= proximity for x in support):
                support.append(level)
            if len(support) >= max_levels:
                break

        return sorted(resistance), sorted(support)

    def detect_breakout(self, close: float, resistance: List[float],
                        support: List[float], threshold: float) -> Optional[Tuple[str, float]]:
        """Detect breakout through S/R levels."""
        # Bullish breakout
        broken_resistance = [r for r in resistance if close > r + threshold]
        if broken_resistance:
            return ('bullish', max(broken_resistance))

        # Bearish breakout
        broken_support = [s for s in support if close < s - threshold]
        if broken_support:
            return ('bearish', min(broken_support))

        return None

    def check_trend_filter(self, data: pd.DataFrame, direction: str,
                           h1_data: Optional[pd.DataFrame] = None) -> bool:
        """Check if trade aligns with trend (200 EMA filter).

        Uses H1 data for trend if provided, otherwise falls back to trading TF.
        """
        if not self.config.use_trend_filter:
            return True

        # Use H1 data for trend if available
        trend_df = h1_data if h1_data is not None and len(h1_data) >= self.config.trend_ema_period else data

        if len(trend_df) < self.config.trend_ema_period:
            return True  # Allow if insufficient data

        ema = self.calculate_ema(trend_df['close'], self.config.trend_ema_period)
        last_close = trend_df['close'].iloc[-1]
        last_ema = ema.iloc[-1]

        if pd.isna(last_ema):
            return True  # Allow if EMA not available

        if direction == 'bullish':
            return last_close > last_ema
        else:
            return last_close < last_ema

    def is_rollover_period(self, timestamp: datetime) -> bool:
        """Check if timestamp is during daily Forex rollover (21:55-23:05 UTC).

        During rollover, liquidity disappears and spreads explode.
        """
        hour = timestamp.hour
        minute = timestamp.minute
        # Rollover window: 21:55 - 23:05 UTC
        if hour == 21 and minute >= 55:
            return True
        if hour == 22:
            return True
        if hour == 23 and minute <= 5:
            return True
        return False

    def check_session_filter(self, timestamp: datetime) -> Tuple[bool, str]:
        """Check if current time is within trading session."""
        hour = timestamp.hour

        # Determine session
        if 0 <= hour < 7:
            session = "ASIAN"
        elif 7 <= hour < 8:
            session = "LONDON_EARLY"
        elif 8 <= hour < 12:
            session = "OVERLAP"
        elif 12 <= hour < 16:
            session = "NY"
        elif 16 <= hour < 21:
            session = "NY_LATE"
        else:
            session = "OFF"

        if not self.config.use_session_filter:
            return True, session

        # Only trade during London and NY
        allowed = self.config.session_start_utc <= hour < self.config.session_end_utc
        return allowed, session

    def calculate_stop_loss(self, direction: str, entry: float, level: float,
                           atr: Optional[float], support: List[float],
                           resistance: List[float]) -> Optional[float]:
        """Calculate stop loss based on structure and ATR.

        Returns None if no valid structure exists (no structure = no trade).
        """
        buf_pips = self.config.stop_loss_buffer_pips
        min_sl_pips = self.config.min_stop_loss_pips

        buf_price = buf_pips * self.pip
        min_buf_price = self.config.min_sl_buffer_pips * self.pip
        atr_price = float(atr) if atr else 0.0
        dyn_extra = max(buf_price, min_buf_price, self.config.atr_sl_k * atr_price)

        min_sl = min_sl_pips * self.pip

        if direction == 'bullish':
            supports_below = [s for s in support if s < level]
            if not supports_below:
                return None  # NO STRUCTURE = NO TRADE
            sl_struct = max(supports_below) - dyn_extra
            sl_min = entry - min_sl
            return min(sl_struct, sl_min)
        else:
            resistances_above = [r for r in resistance if r > level]
            if not resistances_above:
                return None  # NO STRUCTURE = NO TRADE
            sl_struct = min(resistances_above) + dyn_extra
            sl_min = entry + min_sl
            return max(sl_struct, sl_min)

    def generate_signal(self, data: pd.DataFrame, symbol: str,
                        current_bar_idx: int,
                        h1_data: Optional[pd.DataFrame] = None,
                        h1_bar_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal for a given bar.

        Args:
            data: Full historical data (trading timeframe)
            symbol: Trading symbol
            current_bar_idx: Index of current bar (we use data up to this point)
            h1_data: H1 timeframe data for trend filter (optional)
            h1_bar_idx: Current bar index in H1 data (optional)

        Returns:
            Signal dict if valid signal, None otherwise
        """
        # Use only completed bars (up to but not including current)
        lookback = min(current_bar_idx, self.config.lookback_period + 50)
        if lookback < max(20, self.config.lookback_period):
            return None

        analysis_data = data.iloc[current_bar_idx - lookback:current_bar_idx]

        if len(analysis_data) < max(20, self.config.lookback_period):
            return None

        # Find swing points
        swing_highs, swing_lows = self.find_swing_points(analysis_data)
        if not swing_highs and not swing_lows:
            return None

        # Calculate S/R levels
        resistance, support = self.calculate_sr_levels(
            analysis_data, swing_highs, swing_lows
        )

        # Calculate ATR early (needed for both breakout threshold and SL buffer)
        atr_series = self.calculate_atr(analysis_data, self.config.atr_period)
        atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

        # Compute breakout threshold - use ATR-based if configured, else fixed pips
        if self.config.breakout_threshold_atr_mult is not None and atr is not None and atr > 0:
            threshold = self.config.breakout_threshold_atr_mult * atr
        else:
            threshold = self.config.breakout_threshold_pips * self.pip

        # Get last close
        last_close = float(analysis_data['close'].iloc[-1])

        # Detect breakout
        breakout = self.detect_breakout(last_close, resistance, support, threshold)
        if not breakout:
            return None

        direction, level = breakout

        # Duplicate prevention
        bar_time = analysis_data.index[-1]
        key = (symbol, direction)
        if self._last_breakout_bar.get(key) == bar_time:
            return None
        self._last_breakout_bar[key] = bar_time

        # Rollover filter - skip trades during 21:55-23:05 UTC
        if self.is_rollover_period(bar_time):
            return None

        # Trend filter (use H1 data if available)
        h1_slice = None
        if h1_data is not None and h1_bar_idx is not None and h1_bar_idx > 0:
            # Get H1 data up to current H1 bar (need 200+ bars for EMA)
            h1_lookback = min(h1_bar_idx, 250)
            h1_slice = h1_data.iloc[h1_bar_idx - h1_lookback:h1_bar_idx]

        if not self.check_trend_filter(analysis_data, direction, h1_slice):
            return None

        # Session filter
        session_ok, session = self.check_session_filter(bar_time)
        if not session_ok:
            return None

        # Calculate entry (with spread for simulation) - ATR already computed above
        spread = self.config.spread_pips * self.pip
        if direction == 'bullish':
            entry = last_close + spread / 2  # Simulate ask
        else:
            entry = last_close - spread / 2  # Simulate bid

        # Calculate SL
        sl = self.calculate_stop_loss(direction, entry, level, atr, support, resistance)
        if sl is None:
            return None  # No valid structure for SL
        sl_pips = abs(entry - sl) / self.pip

        # Max SL check
        if self.config.max_sl_pips and sl_pips > self.config.max_sl_pips:
            return None

        # Min SL check
        if sl_pips < self.config.min_stop_loss_pips * 0.95:
            return None

        # Calculate TP
        rr = self.config.risk_reward_ratio
        tp_dist = sl_pips * rr * self.pip
        if direction == 'bullish':
            tp = entry + tp_dist
        else:
            tp = entry - tp_dist

        # Calculate actual RR
        tp_pips = abs(tp - entry) / self.pip
        actual_rr = tp_pips / sl_pips if sl_pips > 0 else 0

        if actual_rr < self.config.min_rr:
            return None

        # Determine trend for metadata
        ema = self.calculate_ema(analysis_data['close'], self.config.trend_ema_period)
        last_ema = ema.iloc[-1] if len(ema) >= self.config.trend_ema_period else None
        if last_ema:
            trend = "UP" if last_close > last_ema else "DOWN"
        else:
            trend = "FLAT"

        return {
            'direction': direction,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'actual_rr': actual_rr,
            'level': level,
            'atr': atr,
            'trend': trend,
            'session': session,
            'bar_time': bar_time,
        }


# =============================================================================
# Backtest Engine
# =============================================================================

class BacktestEngine:
    """Main backtesting engine."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_fetcher = DataFetcher()
        self.trades: List[Trade] = []
        self.equity = config.initial_balance
        self.peak_equity = config.initial_balance
        self.equity_curve: List[float] = [config.initial_balance]
        self.max_drawdown = 0.0
        # Initialize trailing manager
        self.trailing_manager = create_backtest_trailing_manager()

    def reset(self):
        """Reset engine state for a new symbol backtest."""
        self.trades = []
        self.equity = self.config.initial_balance
        self.peak_equity = self.config.initial_balance
        self.equity_curve = [self.config.initial_balance]
        self.max_drawdown = 0.0

    def calculate_position_size(self, sl_pips: float, symbol_info: Dict) -> float:
        """Calculate position size based on risk."""
        risk_amount = self.equity * self.config.risk_per_trade

        # Use pip_value_per_lot if available, otherwise calculate from pip_size
        pip_value_per_lot = symbol_info.get('pip_value_per_lot')
        if pip_value_per_lot is None:
            pip_value_per_lot = symbol_info['pip_size'] * symbol_info['contract_size']

        if sl_pips > 0 and pip_value_per_lot > 0:
            lots = risk_amount / (sl_pips * pip_value_per_lot)
            # Round to lot step
            step = symbol_info['volume_step']
            lots = round(lots / step) * step
            lots = max(symbol_info['volume_min'],
                      min(lots, symbol_info['volume_max']))
            return lots
        return 0.0

    def simulate_trade_exit(self, trade: Trade, bar: pd.Series,
                           symbol_info: Dict) -> bool:
        """
        Check if trade hits SL or TP on given bar, with optional trailing stop logic.
        Returns True if trade closed.
        """
        high = bar['high']
        low = bar['low']
        close = bar['close']
        bar_time = bar.name
        pip = symbol_info['pip_size']
        slippage = self.config.slippage_pips * pip

        # Initialize original SL if needed
        if trade.original_sl is None:
            trade.original_sl = trade.stop_loss
            trade.current_sl = trade.stop_loss

        # --- Delegate to Trailing Manager if Enabled ---
        if self.config.use_trailing_stop:
            # Update manager with current price (using close for calculation stability)
            update_res = self.trailing_manager.update_position(
                position_id=id(trade),
                current_price=close,
                current_time=bar_time
            )
            
            # Apply updates back to Trade object
            if update_res['new_sl']:
                trade.current_sl = update_res['new_sl']
                trade.sl_updates += 1
            
            # Sync state strings
            if update_res['state']:
                trade.trailing_state = update_res['state']
            
            # Sync stats
            trade.max_favorable_pips = update_res['mfe_pips']
            # MAE is not tracked by manager in same way, keep local calc below
            
            # Check for Time Exit signal from Manager
            if update_res['should_close_time']:
                trade.exit_price = close
                trade.status = "CLOSED_TIME"
                trade.exit_time = bar_time
                self.trailing_manager.unregister_position(id(trade))
                return True

        # --- Standard Exit Logic (SL/TP) ---
        
        # Use active SL
        active_sl = trade.current_sl if trade.current_sl else trade.stop_loss

        # Update MAE locally (Manager tracks MFE)
        if trade.direction == "BUY":
            current_adverse_pips = (trade.entry_price - low) / pip
        else:
            current_adverse_pips = (high - trade.entry_price) / pip
        trade.max_adverse_pips = max(trade.max_adverse_pips, current_adverse_pips)

        if trade.direction == "BUY":
            # Check SL (Low touches SL)
            if low <= active_sl:
                trade.exit_price = active_sl - slippage
                # Determine exit reason
                if trade.trailing_state != "INITIAL":
                    trade.status = "CLOSED_TRAIL"
                else:
                    trade.status = "CLOSED_SL"
                trade.exit_time = bar_time
                if self.config.use_trailing_stop:
                    self.trailing_manager.unregister_position(id(trade))
                return True
                
            # Check TP (High touches TP)
            # Only check TP if it's reachable and we aren't in "infinite trail" mode
            # But the manager handles logic. If use_fixed_tp is False, TP is huge.
            if trade.take_profit and high >= trade.take_profit:
                 trade.exit_price = trade.take_profit - slippage
                 trade.status = "CLOSED_TP"
                 trade.exit_time = bar_time
                 if self.config.use_trailing_stop:
                    self.trailing_manager.unregister_position(id(trade))
                 return True

        else: # SELL
            # Check SL (High touches SL)
            if high >= active_sl:
                trade.exit_price = active_sl + slippage
                if trade.trailing_state != "INITIAL":
                    trade.status = "CLOSED_TRAIL"
                else:
                    trade.status = "CLOSED_SL"
                trade.exit_time = bar_time
                if self.config.use_trailing_stop:
                    self.trailing_manager.unregister_position(id(trade))
                return True

            # Check TP (Low touches TP)
            if trade.take_profit and low <= trade.take_profit:
                trade.exit_price = trade.take_profit + slippage
                trade.status = "CLOSED_TP"
                trade.exit_time = bar_time
                if self.config.use_trailing_stop:
                    self.trailing_manager.unregister_position(id(trade))
                return True

        return False

    def calculate_pnl(self, trade: Trade, symbol_info: Dict) -> float:
        """Calculate P&L for closed trade."""
        if trade.exit_price is None:
            return 0.0

        pip = symbol_info['pip_size']

        if trade.direction == "BUY":
            pips = (trade.exit_price - trade.entry_price) / pip
        else:
            pips = (trade.entry_price - trade.exit_price) / pip

        trade.pnl_pips = pips

        # Use pip_value_per_lot if available for accurate P&L
        pip_value_per_lot = symbol_info.get('pip_value_per_lot')
        if pip_value_per_lot is None:
            pip_value_per_lot = pip * symbol_info['contract_size']

        trade.pnl = pips * pip_value_per_lot * trade.volume

        return trade.pnl

    def run_backtest(self, symbol: str, data: pd.DataFrame,
                     h1_data: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            symbol: Trading symbol
            data: Historical OHLCV data (trading timeframe)
            h1_data: H1 data for trend filter (optional but recommended)

        Returns:
            BacktestResult with all statistics
        """
        logger.info(f"Running backtest for {symbol}: {len(data)} bars")
        if h1_data is not None:
            logger.info(f"Using H1 data for trend filter: {len(h1_data)} bars")

        symbol_info = self.data_fetcher.get_symbol_info(symbol)
        strategy = BacktestStrategy(self.config, symbol_info)

        open_trade: Optional[Trade] = None

        # Walk through data bar by bar
        for i in range(max(50, self.config.lookback_period), len(data)):
            bar = data.iloc[i]
            bar_time = bar.name

            # Find corresponding H1 bar index
            h1_bar_idx = None
            if h1_data is not None and len(h1_data) > 0:
                # Find H1 bars up to current bar time
                h1_mask = h1_data.index <= bar_time
                h1_bar_idx = h1_mask.sum()

            # Check if open trade should close
            if open_trade and open_trade.status == "OPEN":
                if self.simulate_trade_exit(open_trade, bar, symbol_info):
                    pnl = self.calculate_pnl(open_trade, symbol_info)
                    self.equity += pnl
                    self.equity_curve.append(self.equity)

                    # Update drawdown tracking
                    if self.equity > self.peak_equity:
                        self.peak_equity = self.equity
                    dd = (self.peak_equity - self.equity) / self.peak_equity
                    if dd > self.max_drawdown:
                        self.max_drawdown = dd

                    self.trades.append(open_trade)
                    open_trade = None

            # Generate new signal if no open trade
            if open_trade is None:
                signal = strategy.generate_signal(data, symbol, i, h1_data, h1_bar_idx)

                if signal:
                    volume = self.calculate_position_size(
                        signal['sl_pips'], symbol_info
                    )

                    if volume > 0:
                        open_trade = Trade(
                            entry_time=signal['bar_time'],
                            symbol=symbol,
                            direction="BUY" if signal['direction'] == 'bullish' else "SELL",
                            entry_price=signal['entry'],
                            stop_loss=signal['sl'],
                            take_profit=signal['tp'],
                            volume=volume,
                            status="OPEN",
                            reason=f"{signal['direction']}_breakout",
                            breakout_level=signal['level'],
                            atr_at_entry=signal['atr'],
                            trend_direction=signal['trend'],
                            session=signal['session'],
                        )
                        
                        # Register with trailing manager
                        if self.config.use_trailing_stop:
                            self.trailing_manager.register_position(
                                position_id=id(open_trade), # Use object ID as unique ID
                                symbol=symbol,
                                direction=open_trade.direction,
                                entry_price=open_trade.entry_price,
                                entry_time=open_trade.entry_time,
                                stop_loss=open_trade.stop_loss,
                                take_profit=open_trade.take_profit,
                                pip_size=symbol_info['pip_size']
                            )

        # Close any remaining open trade at end
        if open_trade and open_trade.status == "OPEN":
            last_bar = data.iloc[-1]
            open_trade.exit_price = last_bar['close']
            open_trade.exit_time = last_bar.name
            open_trade.status = "CLOSED_TIME"
            self.calculate_pnl(open_trade, symbol_info)
            self.equity += open_trade.pnl
            self.trades.append(open_trade)

        return self._calculate_statistics()

    def _calculate_statistics(self) -> BacktestResult:
        """Calculate all backtest statistics."""
        result = BacktestResult(config=self.config)
        result.trades = self.trades
        result.equity_curve = self.equity_curve

        if not self.trades:
            return result

        # Basic counts
        closed_trades = [t for t in self.trades if t.status.startswith("CLOSED")]
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl <= 0]

        result.total_trades = len(closed_trades)
        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = len(winners) / len(closed_trades) if closed_trades else 0

        # Profitability
        result.gross_profit = sum(t.pnl for t in winners)
        result.gross_loss = abs(sum(t.pnl for t in losers))
        result.net_profit = result.gross_profit - result.gross_loss
        result.profit_factor = (result.gross_profit / result.gross_loss
                                if result.gross_loss > 0 else float('inf'))

        # Averages
        result.avg_win = result.gross_profit / len(winners) if winners else 0
        result.avg_loss = result.gross_loss / len(losers) if losers else 0
        result.expectancy = result.net_profit / len(closed_trades) if closed_trades else 0

        # Drawdown
        result.max_drawdown = self.max_drawdown * self.config.initial_balance
        result.max_drawdown_pct = self.max_drawdown

        # Time analysis
        durations = []
        for t in closed_trades:
            if t.exit_time and t.entry_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration)
        result.avg_trade_duration_hours = np.mean(durations) if durations else 0

        # Trades per month
        if closed_trades:
            first_trade = min(t.entry_time for t in closed_trades)
            last_trade = max(t.exit_time for t in closed_trades if t.exit_time)
            months = (last_trade - first_trade).days / 30
            result.trades_per_month = len(closed_trades) / months if months > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                result.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        return result

    def shutdown(self):
        """Cleanup resources."""
        self.data_fetcher.shutdown()


# =============================================================================
# Reporting
# =============================================================================

def print_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)

    print(f"\nSymbols: {result.config.symbols}")
    print(f"Period: {result.config.start_date} to {result.config.end_date}")
    print(f"Timeframe: {result.config.timeframe}")
    print(f"Initial Balance: ${result.config.initial_balance:,.2f}")

    print("\n" + "-" * 40)
    print("PERFORMANCE SUMMARY")
    print("-" * 40)

    final_equity = result.equity_curve[-1] if result.equity_curve else result.config.initial_balance
    total_return = (final_equity - result.config.initial_balance) / result.config.initial_balance * 100

    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Net Profit: ${result.net_profit:,.2f}")
    print(f"Total Return: {total_return:.2f}%")

    print(f"\nTotal Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.1%}")

    print(f"\nProfit Factor: {result.profit_factor:.2f}")
    print(f"Average Win: ${result.avg_win:.2f}")
    print(f"Average Loss: ${result.avg_loss:.2f}")
    print(f"Expectancy: ${result.expectancy:.2f} per trade")

    print(f"\nMax Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1%})")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

    print(f"\nAvg Trade Duration: {result.avg_trade_duration_hours:.1f} hours")
    print(f"Trades per Month: {result.trades_per_month:.1f}")

    # Filter analysis
    if result.config.use_trend_filter:
        print("\n[Trend Filter: ENABLED]")
    if result.config.use_session_filter:
        print(f"[Session Filter: ENABLED ({result.config.session_start_utc}:00-{result.config.session_end_utc}:00 UTC)]")
    if result.config.use_retest_confirmation:
        print("[Retest Confirmation: ENABLED]")
    if result.config.use_trailing_stop:
        print(f"[Trailing Stop: ENABLED (BE@{result.config.trailing_break_even_pips}p, Trail@{result.config.trailing_trigger_pips}p/{result.config.trailing_distance_pips}p)]")

    print("\n" + "=" * 70)

    # Trade breakdown by outcome
    if result.trades:
        sl_trades = [t for t in result.trades if t.status == "CLOSED_SL"]
        tp_trades = [t for t in result.trades if t.status == "CLOSED_TP"]
        trail_trades = [t for t in result.trades if t.status == "CLOSED_TRAIL"]
        time_trades = [t for t in result.trades if t.status == "CLOSED_TIME"]

        outcome_str = f"{len(tp_trades)} TP, {len(sl_trades)} SL"
        if trail_trades:
            outcome_str += f", {len(trail_trades)} TRAIL"
        if time_trades:
            outcome_str += f", {len(time_trades)} TIME"
        print(f"\nTrade Outcomes: {outcome_str}")


def save_results(result: BacktestResult, output_dir: str):
    """Save backtest results to files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trades to CSV
    if result.trades:
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'volume': t.volume,
                'pnl': t.pnl,
                'pnl_pips': t.pnl_pips,
                'status': t.status,
                'trend': t.trend_direction,
                'session': t.session,
                # Trailing stop data
                'original_sl': t.original_sl,
                'current_sl': t.current_sl,
                'trailing_state': t.trailing_state,
                'max_favorable_pips': t.max_favorable_pips,
                'max_adverse_pips': t.max_adverse_pips,
                'sl_updates': t.sl_updates,
            })

        trades_df = pd.DataFrame(trades_data)
        trades_file = out_path / f"trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Trades saved to {trades_file}")

    # Save equity curve
    equity_file = out_path / f"equity_{timestamp}.csv"
    pd.DataFrame({'equity': result.equity_curve}).to_csv(equity_file, index=False)

    # Save summary
    summary = {
        'timestamp': timestamp,
        'config': asdict(result.config),
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'net_profit': result.net_profit,
        'max_drawdown_pct': result.max_drawdown_pct,
        'sharpe_ratio': result.sharpe_ratio,
        'expectancy': result.expectancy,
    }

    summary_file = out_path / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {summary_file}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Forex Strategy Backtester')

    # Data source
    parser.add_argument('--config', type=str,
                       help='Load symbols from config.json file')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Symbol to backtest')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Multiple symbols to backtest')
    parser.add_argument('--timeframe', type=str, default='M15',
                       help='Timeframe (M1, M5, M15, M30, H1, H4, D1)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-file', type=str,
                       help='CSV file with historical data (alternative to MT5)')

    # Account
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial account balance')
    parser.add_argument('--risk', type=float, default=0.01,
                       help='Risk per trade (0.01 = 1%)')

    # Strategy parameters
    parser.add_argument('--lookback', type=int, default=20,
                       help='Lookback period for S/R detection')
    parser.add_argument('--breakout-threshold', type=float, default=5,
                       help='Breakout threshold in pips (fallback if ATR mult not set)')
    parser.add_argument('--breakout-threshold-atr', type=float, default=None,
                       help='Breakout threshold as ATR multiplier (e.g., 0.3 = 30%% of ATR)')
    parser.add_argument('--rr', type=float, default=2.0,
                       help='Risk:Reward ratio')
    parser.add_argument('--min-sl', type=float, default=20,
                       help='Minimum stop loss in pips')

    # Filters
    parser.add_argument('--trend-filter', action='store_true',
                       help='Enable 200 EMA trend filter (now enabled by default)')
    parser.add_argument('--no-trend-filter', action='store_true',
                       help='Disable 200 EMA trend filter')
    parser.add_argument('--session-filter', action='store_true',
                       help='Enable London/NY session filter')
    parser.add_argument('--retest', action='store_true',
                       help='Enable retest confirmation')

    # Execution modeling
    parser.add_argument('--spread', type=float, default=1.5,
                       help='Average spread in pips')
    parser.add_argument('--slippage', type=float, default=0.5,
                       help='Average slippage in pips')

    # Trailing stop settings
    parser.add_argument('--trailing', action='store_true',
                       help='Enable trailing stop mode')
    parser.add_argument('--trail-be-pips', type=float, default=15.0,
                       help='Move to break-even after this profit (pips)')
    parser.add_argument('--trail-be-offset', type=float, default=1.0,
                       help='Break-even offset from entry (pips)')
    parser.add_argument('--trail-trigger-pips', type=float, default=25.0,
                       help='Start trailing after this profit (pips)')
    parser.add_argument('--trail-distance-pips', type=float, default=15.0,
                       help='Trail this far behind price (pips)')
    parser.add_argument('--trail-max-hours', type=float, default=48.0,
                       help='Close after this many hours (0=disabled)')

    # Output
    parser.add_argument('--output', type=str, default='backtest_results',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    return parser.parse_args()


def load_symbols_from_config(config_path: str) -> Tuple[List[str], Dict[str, Dict], Dict[str, Any]]:
    """Load symbol names, per-symbol parameters, and global settings from config.json.

    Returns:
        Tuple of (symbol_names, symbol_params, global_config)
    """
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)

        symbols = []
        symbol_params = {}
        global_config = {}

        # Load global settings
        trading = data.get('trading_settings', {})
        risk = data.get('risk_management', {})

        if trading.get('lookback_period'): global_config['lookback_period'] = int(trading['lookback_period'])
        if trading.get('swing_window'): global_config['swing_window'] = int(trading['swing_window'])
        if trading.get('breakout_threshold_pips'): global_config['breakout_threshold_pips'] = float(trading['breakout_threshold_pips'])
        if trading.get('breakout_threshold_atr_mult'): global_config['breakout_threshold_atr_mult'] = float(trading['breakout_threshold_atr_mult'])
        if trading.get('min_stop_loss_pips'): global_config['min_stop_loss_pips'] = float(trading['min_stop_loss_pips'])
        if trading.get('stop_loss_buffer_pips'): global_config['stop_loss_buffer_pips'] = float(trading['stop_loss_buffer_pips'])
        if risk.get('risk_reward_ratio'): global_config['risk_reward_ratio'] = float(risk['risk_reward_ratio'])
        if risk.get('min_rr'): global_config['min_rr'] = float(risk['min_rr'])
        if risk.get('risk_per_trade'): global_config['risk_per_trade'] = float(risk['risk_per_trade'])

        for s in data.get('symbols', []):
            name = s.get('name')
            if not name:
                continue
            symbols.append(name)

            # Extract per-symbol parameter overrides
            params = {}
            if s.get('min_stop_loss_pips') is not None:
                params['min_stop_loss_pips'] = float(s['min_stop_loss_pips'])
            if s.get('stop_loss_buffer_pips') is not None:
                params['stop_loss_buffer_pips'] = float(s['stop_loss_buffer_pips'])
            if s.get('breakout_threshold_pips') is not None:
                params['breakout_threshold_pips'] = float(s['breakout_threshold_pips'])
            if s.get('breakout_threshold_atr_mult') is not None:
                params['breakout_threshold_atr_mult'] = float(s['breakout_threshold_atr_mult'])
            if s.get('risk_reward_ratio') is not None:
                params['risk_reward_ratio'] = float(s['risk_reward_ratio'])
            if s.get('min_rr') is not None:
                params['min_rr'] = float(s['min_rr'])
            if s.get('spread_guard_pips') is not None:
                params['spread_pips'] = float(s['spread_guard_pips'])

            if params:
                symbol_params[name] = params

        return symbols, symbol_params, global_config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return [], {}, {}


def print_portfolio_summary(results: List[BacktestResult], config: BacktestConfig):
    """Print combined portfolio summary for multi-symbol backtests."""
    if len(results) <= 1:
        return

    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY (All Symbols Combined)")
    print("=" * 70)

    total_trades = sum(r.total_trades for r in results)
    total_wins = sum(r.winning_trades for r in results)
    total_losses = sum(r.losing_trades for r in results)
    total_gross_profit = sum(r.gross_profit for r in results)
    total_gross_loss = sum(r.gross_loss for r in results)
    total_net_profit = sum(r.net_profit for r in results)

    win_rate = total_wins / total_trades if total_trades > 0 else 0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
    expectancy = total_net_profit / total_trades if total_trades > 0 else 0

    # Find worst drawdown across symbols
    worst_dd_pct = max(r.max_drawdown_pct for r in results) if results else 0

    print(f"\nSymbols Tested: {len(results)}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Balance per Symbol: ${config.initial_balance:,.2f}")

    print("\n" + "-" * 40)
    print("COMBINED METRICS")
    print("-" * 40)

    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {total_wins}")
    print(f"Losing Trades: {total_losses}")
    print(f"Win Rate: {win_rate:.1%}")

    print(f"\nTotal Net Profit: ${total_net_profit:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expectancy: ${expectancy:.2f} per trade")
    print(f"Worst Max Drawdown: {worst_dd_pct:.1%}")

    print("\n" + "-" * 40)
    print("PER-SYMBOL BREAKDOWN")
    print("-" * 40)
    print(f"{'Symbol':<12} {'Trades':>8} {'Win%':>8} {'Net P&L':>12} {'PF':>8} {'MaxDD':>8}")
    print("-" * 56)

    for r in sorted(results, key=lambda x: x.net_profit, reverse=True):
        symbol = r.config.symbols[0] if r.config.symbols else "?"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "inf"
        print(f"{symbol:<12} {r.total_trades:>8} {r.win_rate:>7.1%} ${r.net_profit:>10,.2f} {pf_str:>8} {r.max_drawdown_pct:>7.1%}")

    print("=" * 70)


def main():
    args = parse_args()

    # Determine symbols: --config > --symbols > --symbol
    symbol_params = {}  # Per-symbol parameter overrides
    global_config = {}
    if args.config:
        symbols, symbol_params, global_config = load_symbols_from_config(args.config)
        if not symbols:
            logger.error("No symbols found in config file")
            sys.exit(1)
        logger.info(f"Loaded {len(symbols)} symbols from {args.config}")
        if symbol_params:
            logger.info(f"Per-symbol overrides for: {', '.join(symbol_params.keys())}")
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = [args.symbol]

    # Trend filter: default is now True, --no-trend-filter disables it
    use_trend = not getattr(args, 'no_trend_filter', False)
    if args.trend_filter:  # Explicit enable overrides
        use_trend = True

    config = BacktestConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        risk_per_trade=global_config.get('risk_per_trade', args.risk),
        lookback_period=global_config.get('lookback_period', args.lookback),
        breakout_threshold_pips=global_config.get('breakout_threshold_pips', args.breakout_threshold),
        breakout_threshold_atr_mult=global_config.get('breakout_threshold_atr_mult', args.breakout_threshold_atr),
        risk_reward_ratio=global_config.get('risk_reward_ratio', args.rr),
        min_stop_loss_pips=global_config.get('min_stop_loss_pips', args.min_sl),
        stop_loss_buffer_pips=global_config.get('stop_loss_buffer_pips', 15.0),
        min_rr=global_config.get('min_rr', 1.0),
        use_trend_filter=use_trend,
        use_session_filter=args.session_filter,
        use_retest_confirmation=args.retest,
        spread_pips=args.spread,
        slippage_pips=args.slippage,
        # Trailing stop settings
        use_trailing_stop=args.trailing,
        trailing_break_even_pips=args.trail_be_pips,
        trailing_break_even_offset_pips=args.trail_be_offset,
        trailing_trigger_pips=args.trail_trigger_pips,
        trailing_distance_pips=args.trail_distance_pips,
        trailing_max_hours=args.trail_max_hours,
        output_dir=args.output,
    )

    logger.info(f"Starting backtest: {len(config.symbols)} symbols on {config.timeframe}")
    logger.info(f"Symbols: {', '.join(config.symbols)}")
    logger.info(f"Period: {config.start_date} to {config.end_date}")
    logger.info(f"Filters: trend={config.use_trend_filter}, session={config.use_session_filter}")

    engine = BacktestEngine(config)

    try:
        # Fetch data
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        all_results = []

        for symbol in config.symbols:
            # Reset engine state for each symbol
            engine.reset()

            if args.data_file:
                data = engine.data_fetcher.fetch_csv_data(args.data_file)
                h1_data = None  # CSV mode doesn't support H1 fetch
            else:
                data = engine.data_fetcher.fetch_mt5_data(
                    symbol, config.timeframe, start_dt, end_dt
                )
                # Fetch H1 data for trend filter (if not already on H1)
                h1_data = None
                if config.timeframe.upper() != 'H1' and config.use_trend_filter:
                    h1_data = engine.data_fetcher.fetch_mt5_data(
                        symbol, 'H1', start_dt, end_dt
                    )

            if data is None or len(data) < 100:
                logger.error(f"Insufficient data for {symbol}")
                continue

            # Create per-symbol config with any parameter overrides
            sym_overrides = symbol_params.get(symbol, {})

            symbol_config = BacktestConfig(
                symbols=[symbol],
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_balance=config.initial_balance,
                risk_per_trade=config.risk_per_trade,
                lookback_period=config.lookback_period,
                breakout_threshold_pips=sym_overrides.get('breakout_threshold_pips', config.breakout_threshold_pips),
                breakout_threshold_atr_mult=sym_overrides.get('breakout_threshold_atr_mult', config.breakout_threshold_atr_mult),
                risk_reward_ratio=sym_overrides.get('risk_reward_ratio', config.risk_reward_ratio),
                min_rr=sym_overrides.get('min_rr', config.min_rr),
                min_stop_loss_pips=sym_overrides.get('min_stop_loss_pips', config.min_stop_loss_pips),
                stop_loss_buffer_pips=sym_overrides.get('stop_loss_buffer_pips', config.stop_loss_buffer_pips),
                use_trend_filter=config.use_trend_filter,
                use_session_filter=config.use_session_filter,
                use_retest_confirmation=config.use_retest_confirmation,
                spread_pips=sym_overrides.get('spread_pips', config.spread_pips),
                slippage_pips=config.slippage_pips,
                # Trailing stop settings
                use_trailing_stop=config.use_trailing_stop,
                trailing_break_even_pips=config.trailing_break_even_pips,
                trailing_break_even_offset_pips=config.trailing_break_even_offset_pips,
                trailing_trigger_pips=config.trailing_trigger_pips,
                trailing_distance_pips=config.trailing_distance_pips,
                trailing_max_hours=config.trailing_max_hours,
                output_dir=config.output_dir,
            )

            if sym_overrides:
                logger.info(f"{symbol}: Using overrides {sym_overrides}")

            engine.config = symbol_config

            result = engine.run_backtest(symbol, data, h1_data)
            all_results.append(result)

            if not args.quiet:
                print_report(result)

            save_results(result, config.output_dir)

        # Print portfolio summary for multi-symbol runs
        if len(all_results) > 1:
            print_portfolio_summary(all_results, config)

        logger.info("Backtest complete")

    finally:
        engine.shutdown()

    return all_results


if __name__ == "__main__":
    main()
