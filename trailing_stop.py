"""
Trailing Stop Module for Breakout Strategy

Based on "R-Multiple" Logic:
- Replaces arbitrary fixed pips with Structure-Based Risk (R).
- 1R = Distance from Entry to Initial Stop Loss.
- Adapts automatically to volatility (ATR) and market structure of ANY pair.

Logic:
1. Entry: Initial Risk established (e.g., 50 pips).
2. Earn It: Move to Break-Even only after profit >= 1.0R (e.g., +50 pips).
3. Trail: Trail behind price by 0.5R (e.g., 25 pips) to allow breathing room.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrailingState(Enum):
    """State of trailing stop for a position."""
    INITIAL = "INITIAL"           # SL at original level
    BREAK_EVEN = "BREAK_EVEN"     # SL moved to break-even
    TRAILING = "TRAILING"         # SL actively trailing price


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop behavior.
    
    Supports both Dynamic Risk (R-Multiples) and Legacy Fixed Pips.
    """
    # --- Dynamic Risk Settings (Preferred) ---
    use_dynamic_risk: bool = True   # If True, overrides pip settings below
    break_even_trigger_r: float = 1.0  # Move to BE when profit is 1.0x Risk
    trail_trigger_r: float = 1.5       # Start trailing when profit is 1.5x Risk
    trail_distance_r: float = 0.5      # Trail behind price by 0.5x Risk

    # --- Legacy Fixed Pip Settings (Fallback) ---
    break_even_trigger_pips: float = 15.0
    break_even_offset_pips: float = 1.0
    trail_trigger_pips: float = 25.0
    trail_distance_pips: float = 15.0

    # --- Common Settings ---
    max_trade_hours: float = 0.0    # Close after N hours (0=disabled)
    use_fixed_tp: bool = False      # If False, TP is set very far
    max_tp_pips: float = 200.0      # Max TP when not using fixed (safety)
    enabled: bool = True


# Standard "Trend Following" Configuration (Robust for all pairs)
# - Validates trend by waiting for 1R profit.
# - Gives room to breathe with 0.5R trail.
STANDARD_R_CONFIG = TrailingStopConfig(
    use_dynamic_risk=True,
    break_even_trigger_r=1.0,  # Earn the right to be risk-free
    trail_trigger_r=1.2,       # Start trailing shortly after BE
    trail_distance_r=0.5,      # Wide trail to ride trends
    max_trade_hours=48.0
)

# Tighter config for mean-reverting or lower volatility expectations
TIGHT_R_CONFIG = TrailingStopConfig(
    use_dynamic_risk=True,
    break_even_trigger_r=0.8,
    trail_trigger_r=1.0,
    trail_distance_r=0.4,
    max_trade_hours=24.0
)

# Per-symbol settings 
# Now we mostly use the Standard R-Config as it auto-adapts to volatility.
SYMBOL_TRAILING_CONFIGS: Dict[str, TrailingStopConfig] = {
    'EURUSD': STANDARD_R_CONFIG,
    'GBPUSD': STANDARD_R_CONFIG,
    'USDJPY': STANDARD_R_CONFIG,
    'GBPJPY': STANDARD_R_CONFIG,
    'XAUUSD': STANDARD_R_CONFIG,
    'AUDUSD': STANDARD_R_CONFIG,
    'USDCAD': STANDARD_R_CONFIG,
    'EURGBP': STANDARD_R_CONFIG,
    'EURJPY': STANDARD_R_CONFIG,
    'GBPCAD': STANDARD_R_CONFIG,
    'AUDCAD': TIGHT_R_CONFIG,     # Often ranges, maybe tighter is better
    'EURCHF': TIGHT_R_CONFIG,     # Low vol
}

DEFAULT_TRAILING_CONFIG = STANDARD_R_CONFIG


@dataclass
class TrailingStopState:
    """Tracks trailing stop state for an open position."""
    position_id: Any
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    original_sl: float
    original_tp: Optional[float]
    current_sl: float
    pip_size: float

    # State tracking
    state: TrailingState = TrailingState.INITIAL
    highest_profit_pips: float = 0.0
    sl_updates: int = 0
    last_update_time: Optional[datetime] = None
    
    # Cache calculated risk for performance
    initial_risk_pips: float = 0.0

    def __post_init__(self):
        # Calculate Initial Risk (1R)
        self.initial_risk_pips = abs(self.entry_price - self.original_sl) / self.pip_size
        if self.initial_risk_pips <= 0:
            self.initial_risk_pips = 100.0 # Fallback safety to avoid div/0 or 0 logic

    def profit_pips(self, current_price: float) -> float:
        """Calculate current profit in pips."""
        if self.direction == "BUY":
            return (current_price - self.entry_price) / self.pip_size
        else:
            return (self.entry_price - current_price) / self.pip_size

    def update_mfe(self, current_price: float):
        """Update maximum favorable excursion."""
        profit = self.profit_pips(current_price)
        if profit > self.highest_profit_pips:
            self.highest_profit_pips = profit


class TrailingStopManager:
    """Manages trailing stops using Risk-Based logic."""

    def __init__(self, mt5_client=None):
        self.mt5_client = mt5_client
        self.positions: Dict[Any, TrailingStopState] = {}
        self.configs: Dict[str, TrailingStopConfig] = SYMBOL_TRAILING_CONFIGS.copy()

    def get_config(self, symbol: str) -> TrailingStopConfig:
        return self.configs.get(symbol, DEFAULT_TRAILING_CONFIG)

    def set_config(self, symbol: str, config: TrailingStopConfig):
        self.configs[symbol] = config

    def register_position(self, position_id, symbol, direction, entry_price, 
                         entry_time, stop_loss, take_profit, pip_size) -> TrailingStopState:
        state = TrailingStopState(
            position_id=position_id,
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            entry_time=entry_time,
            original_sl=stop_loss,
            original_tp=take_profit,
            current_sl=stop_loss,
            pip_size=pip_size
        )
        self.positions[position_id] = state
        logger.debug(f"Registered {symbol} pos {position_id}. Risk: {state.initial_risk_pips:.1f} pips")
        return state

    def unregister_position(self, position_id: Any):
        if position_id in self.positions:
            del self.positions[position_id]

    def calculate_new_sl(self, state: TrailingStopState, current_price: float, current_time: datetime) -> Optional[float]:
        """Calculate new SL using dynamic R-multiples or fixed pips."""
        config = self.get_config(state.symbol)
        if not config.enabled:
            return None

        state.update_mfe(current_price)
        profit_pips = state.profit_pips(current_price)
        pip = state.pip_size

        # Determine Thresholds
        if config.use_dynamic_risk:
            # Dynamic Risk Logic
            r_pips = state.initial_risk_pips
            be_trigger = config.break_even_trigger_r * r_pips
            trail_trigger = config.trail_trigger_r * r_pips
            trail_dist = config.trail_distance_r * r_pips
        else:
            # Legacy Fixed Pips
            be_trigger = config.break_even_trigger_pips
            trail_trigger = config.trail_trigger_pips
            trail_dist = config.trail_distance_pips

        # BE Offset is always fixed pips (usually spread coverage)
        be_offset = config.break_even_offset_pips

        new_sl = None
        new_state = state.state

        # 1. Break-Even Logic
        if state.state == TrailingState.INITIAL:
            if profit_pips >= be_trigger:
                if state.direction == "BUY":
                    new_sl = state.entry_price + (be_offset * pip)
                else:
                    new_sl = state.entry_price - (be_offset * pip)
                new_state = TrailingState.BREAK_EVEN
                logger.info(f"{state.symbol} {state.position_id}: BE Triggered (+{profit_pips:.1f} pips / {profit_pips/state.initial_risk_pips:.1f}R)")

        # 2. Trailing Logic
        # Check if we should start trailing (from either Initial or BE state)
        if state.state in (TrailingState.INITIAL, TrailingState.BREAK_EVEN):
            if profit_pips >= trail_trigger:
                # Calculate potential trail level
                if state.direction == "BUY":
                    trail_level = current_price - (trail_dist * pip)
                else:
                    trail_level = current_price + (trail_dist * pip)

                # Validate against current SL
                if self._is_better_sl(state.direction, trail_level, state.current_sl):
                    new_sl = trail_level
                    new_state = TrailingState.TRAILING
                    logger.info(f"{state.symbol} {state.position_id}: Trail Started (+{profit_pips:.1f} pips)")

        elif state.state == TrailingState.TRAILING:
            # Continue trailing
            if state.direction == "BUY":
                trail_level = current_price - (trail_dist * pip)
            else:
                trail_level = current_price + (trail_dist * pip)

            if self._is_better_sl(state.direction, trail_level, state.current_sl):
                new_sl = trail_level

        # Update State
        if new_state != state.state:
            state.state = new_state

        return new_sl

    def _is_better_sl(self, direction: str, new_sl: float, current_sl: float) -> bool:
        if direction == "BUY":
            return new_sl > current_sl
        else:
            return new_sl < current_sl

    def check_time_exit(self, state: TrailingStopState, current_time: datetime) -> bool:
        config = self.get_config(state.symbol)
        if config.max_trade_hours <= 0:
            return False
        
        elapsed = current_time - state.entry_time
        if elapsed >= timedelta(hours=config.max_trade_hours):
            return True
        return False

    def update_position(self, position_id: Any, current_price: float, current_time: datetime) -> Dict[str, Any]:
        if position_id not in self.positions:
            return {'new_sl': None, 'should_close_time': False, 'state': None}

        state = self.positions[position_id]
        new_sl = self.calculate_new_sl(state, current_price, current_time)
        should_close_time = self.check_time_exit(state, current_time)

        if new_sl is not None:
            state.current_sl = new_sl
            state.sl_updates += 1
            state.last_update_time = current_time

        return {
            'new_sl': new_sl,
            'should_close_time': should_close_time,
            'state': state.state.value,
            'current_sl': state.current_sl,
            'profit_pips': state.profit_pips(current_price),
            'mfe_pips': state.highest_profit_pips
        }

    # Live trading methods
    def update_sl_live(self, position_id: Any, new_sl: float) -> bool:
        if not self.mt5_client or position_id not in self.positions: return False
        try:
            state = self.positions[position_id]
            return self.mt5_client.modify_position(ticket=position_id, sl=new_sl, tp=state.original_tp)
        except Exception as e:
            logger.error(f"Error live SL: {e}")
            return False

    def close_position_live(self, position_id: Any, reason: str = "TIME_EXIT") -> bool:
        if not self.mt5_client: return False
        try:
            if self.mt5_client.close_position(ticket=position_id, comment=f"TRAILING_{reason}"):
                self.unregister_position(position_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error live close: {e}")
            return False

    def get_position_state(self, position_id: Any) -> Optional[TrailingStopState]:
        return self.positions.get(position_id)

    def get_all_positions(self) -> Dict[Any, TrailingStopState]:
        return self.positions.copy()


def create_backtest_trailing_manager() -> TrailingStopManager:
    return TrailingStopManager(mt5_client=None)

def create_live_trailing_manager(mt5_client) -> TrailingStopManager:
    return TrailingStopManager(mt5_client=mt5_client)