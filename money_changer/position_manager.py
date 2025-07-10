"""
ICT Position Management Component
Advanced position management following Inner Circle Trader methodology.
Handles position scaling, risk management, and institutional alignment.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class PositionStage(Enum):
    """Position lifecycle stages following ICT methodology."""
    ENTRY = "entry"
    RISK_FREE = "risk_free" 
    SCALING = "scaling"
    RUNNER = "runner"
    MANAGEMENT = "management"
    EXIT = "exit"

class ExitReason(Enum):
    """Reasons for position exits."""
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target" 
    PARTIAL_SCALE = "partial_scale"
    BREAK_EVEN = "break_even"
    STRUCTURE_BREAK = "structure_break"
    SESSION_END = "session_end"
    REVERSAL_SIGNAL = "reversal_signal"
    MANUAL = "manual"

@dataclass
class PositionPlan:
    """Complete position management plan following ICT principles."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    initial_stop: float
    initial_volume: float
    initial_risk: float
    
    # Scaling levels (ICT pyramid approach)
    scale_levels: List[Dict]  # [{'price': float, 'volume': float, 'target': str}]
    
    # Exit targets (ICT liquidity targeting)
    exit_targets: List[Dict]  # [{'price': float, 'volume_pct': float, 'type': str}]
    
    # Risk management
    break_even_trigger: float  # Price level to move SL to break even
    risk_free_trigger: float   # Price level to move SL to risk-free
    max_risk_per_add: float    # Max additional risk per scale-in
    
    # Session context
    entry_session: str         # Asian/London/NY
    session_bias: str          # bullish/bearish
    narrative_model: str       # Judas/OTE/FVG etc.

@dataclass 
class PositionState:
    """Current state of a managed position."""
    ticket: int
    symbol: str
    direction: str
    current_volume: float
    average_entry: float
    current_stop: float
    current_targets: List[Dict]
    
    stage: PositionStage
    profit_pips: float
    profit_amount: float
    risk_amount: float
    rr_ratio: float
    
    # Scaling tracking
    scale_ins: List[Dict]      # History of additions
    scale_outs: List[Dict]     # History of partial exits
    
    # Management flags
    is_risk_free: bool
    break_even_triggered: bool
    trailing_active: bool
    
    # Context
    entry_time: datetime
    last_update: datetime
    management_log: List[str]

class PositionManager:
    """
    Advanced ICT Position Management System
    
    Implements institutional-grade position management following 
    Inner Circle Trader methodology with:
    - Risk-based scaling and pyramiding
    - Liquidity-based exit targeting  
    - Session-aware management
    - Market structure alignment
    """
    
    def __init__(self, mt5_client, config, liquidity_detector, signal_generator):
        self.mt5_client = mt5_client
        self.config = config
        self.liquidity_detector = liquidity_detector
        self.signal_generator = signal_generator
        
        # Active positions being managed
        self.managed_positions: Dict[int, PositionState] = {}
        self.position_plans: Dict[int, PositionPlan] = {}
        
        # Configuration
        self.max_positions_per_symbol = getattr(config, 'MAX_POSITIONS_PER_SYMBOL', 1)
        self.max_total_risk = getattr(config, 'MAX_TOTAL_RISK_PERCENT', 5.0)
        self.scale_in_enabled = getattr(config, 'ENABLE_SCALING', True)
        self.partial_exit_enabled = getattr(config, 'ENABLE_PARTIAL_EXITS', True)
        
        # ICT Position Management Settings
        self.break_even_ratio = getattr(config, 'BREAK_EVEN_RR', 0.5)  # Move to BE at 0.5R
        self.risk_free_ratio = getattr(config, 'RISK_FREE_RR', 1.0)    # Risk-free at 1R
        self.first_target_ratio = getattr(config, 'FIRST_TARGET_RR', 1.5)  # First exit at 1.5R
        self.runner_percentage = getattr(config, 'RUNNER_PERCENTAGE', 25)  # Keep 25% as runner
        
        logger.info("ICT Position Manager initialized")
    
    def create_position_plan(self, symbol: str, signal: str, entry_price: float, 
                           sl_price: float, tp_price: float, volume: float,
                           narrative, market_data) -> PositionPlan:
        """
        Create comprehensive position management plan following ICT methodology.
        """
        initial_risk = abs(entry_price - sl_price) * volume
        
        # Calculate ICT-based scaling levels
        scale_levels = self._calculate_scale_levels(
            symbol, signal, entry_price, sl_price, narrative, market_data
        )
        
        # Calculate liquidity-based exit targets
        exit_targets = self._calculate_exit_targets(
            symbol, signal, entry_price, tp_price, narrative, market_data
        )
        
        # Risk management levels
        risk_range = abs(entry_price - sl_price)
        break_even_trigger = (
            entry_price + (risk_range * self.break_even_ratio) if signal == "BUY"
            else entry_price - (risk_range * self.break_even_ratio)
        )
        
        risk_free_trigger = (
            entry_price + (risk_range * self.risk_free_ratio) if signal == "BUY"
            else entry_price - (risk_range * self.risk_free_ratio)
        )
        
        plan = PositionPlan(
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            initial_stop=sl_price,
            initial_volume=volume,
            initial_risk=initial_risk,
            scale_levels=scale_levels,
            exit_targets=exit_targets,
            break_even_trigger=break_even_trigger,
            risk_free_trigger=risk_free_trigger,
            max_risk_per_add=initial_risk * 0.5,  # Max 50% of initial risk per add
            entry_session=narrative.killzone_name or "Unknown",
            session_bias=narrative.daily_bias,
            narrative_model=narrative.entry_model
        )
        
        logger.info(f"{symbol}: Created position plan - {len(scale_levels)} scale levels, "
                   f"{len(exit_targets)} exit targets")
        return plan
    
    def register_position(self, ticket: int, plan: PositionPlan) -> bool:
        """Register a new position for management."""
        try:
            # Get position details from MT5
            positions = self.mt5_client.get_current_positions(plan.symbol)
            position = next((p for p in positions if p.ticket == ticket), None)
            
            if not position:
                logger.error(f"Position {ticket} not found in MT5")
                return False
            
            # Create position state
            state = PositionState(
                ticket=ticket,
                symbol=plan.symbol,
                direction=plan.direction,
                current_volume=position.volume,
                average_entry=position.price_open,
                current_stop=position.sl,
                current_targets=[{'price': position.tp, 'volume_pct': 100.0, 'type': 'initial'}],
                stage=PositionStage.ENTRY,
                profit_pips=0.0,
                profit_amount=0.0,
                risk_amount=plan.initial_risk,
                rr_ratio=0.0,
                scale_ins=[],
                scale_outs=[],
                is_risk_free=False,
                break_even_triggered=False,
                trailing_active=False,
                entry_time=datetime.now(timezone.utc),
                last_update=datetime.now(timezone.utc),
                management_log=[f"Position registered: {plan.narrative_model}"]
            )
            
            self.managed_positions[ticket] = state
            self.position_plans[ticket] = plan
            
            logger.info(f"{plan.symbol}: Position {ticket} registered for ICT management")
            return True
            
        except Exception as e:
            logger.error(f"Error registering position {ticket}: {e}")
            return False
    
    def update_positions(self, market_data_provider) -> Dict[str, int]:
        """
        Main position management update cycle.
        Returns summary of actions taken.
        """
        actions = {
            'positions_updated': 0,
            'break_even_moves': 0,
            'risk_free_moves': 0,
            'partial_exits': 0,
            'scale_ins': 0,
            'full_exits': 0
        }
        
        for ticket in list(self.managed_positions.keys()):
            try:
                if self._update_single_position(ticket, market_data_provider):
                    actions['positions_updated'] += 1
                    
                    # Track specific actions (simplified for summary)
                    state = self.managed_positions.get(ticket)
                    if state:
                        if state.break_even_triggered:
                            actions['break_even_moves'] += 1
                        if state.is_risk_free:
                            actions['risk_free_moves'] += 1
                            
            except Exception as e:
                logger.error(f"Error updating position {ticket}: {e}")
                continue
        
        if actions['positions_updated'] > 0:
            logger.info(f"Updated {actions['positions_updated']} positions: "
                       f"BE:{actions['break_even_moves']}, RF:{actions['risk_free_moves']}")
        
        return actions
    
    def _update_single_position(self, ticket: int, market_data_provider) -> bool:
        """Update a single position following ICT management rules."""
        if ticket not in self.managed_positions:
            return False
        
        state = self.managed_positions[ticket]
        plan = self.position_plans[ticket]
        
        # Get current market data
        ticker = self.mt5_client.get_symbol_ticker(state.symbol)
        if not ticker:
            return False
        
        current_price = ticker.ask if state.direction == "BUY" else ticker.bid
        
        # Update profit metrics
        self._update_profit_metrics(state, current_price)
        
        # Get current MT5 position to check if still exists
        positions = self.mt5_client.get_current_positions(state.symbol)
        mt5_position = next((p for p in positions if p.ticket == ticket), None)
        
        if not mt5_position:
            # Position closed externally
            self._handle_external_closure(ticket)
            return False
        
        # Update position details from MT5
        state.current_volume = mt5_position.volume
        state.current_stop = mt5_position.sl
        
        # Get market data for structure analysis
        ohlc_df = market_data_provider.get_ohlc(state.symbol, "M15", 200)
        if ohlc_df is None or len(ohlc_df) < 50:
            return False
        
        # Apply ICT management rules
        management_actions = []
        
        # 1. Break-even management
        if not state.break_even_triggered and self._should_move_to_break_even(state, plan, current_price):
            if self._move_stop_to_break_even(ticket, state, plan):
                management_actions.append("Moved to break-even")
                state.break_even_triggered = True
                state.stage = PositionStage.RISK_FREE
        
        # 2. Risk-free management  
        if not state.is_risk_free and self._should_move_to_risk_free(state, plan, current_price):
            if self._move_stop_to_risk_free(ticket, state, plan):
                management_actions.append("Moved to risk-free")
                state.is_risk_free = True
        
        # 3. Partial exit management
        if self.partial_exit_enabled and self._should_take_partial_profit(state, plan, current_price):
            if self._execute_partial_exit(ticket, state, plan, current_price):
                management_actions.append("Partial exit executed")
        
        # 4. Scaling management (if enabled)
        if self.scale_in_enabled and self._should_scale_in(state, plan, current_price, ohlc_df):
            if self._execute_scale_in(ticket, state, plan, current_price):
                management_actions.append("Scaled into position")
        
        # 5. Trailing stop management
        if state.is_risk_free and self._should_trail_stop(state, plan, current_price, ohlc_df):
            if self._update_trailing_stop(ticket, state, plan, current_price, ohlc_df):
                management_actions.append("Trailing stop updated")
        
        # 6. Structure-based exit
        if self._should_exit_on_structure(state, plan, current_price, ohlc_df):
            if self._execute_structure_exit(ticket, state, plan):
                management_actions.append("Structure-based exit")
                return False  # Position closed
        
        # Log management actions
        if management_actions:
            log_entry = f"{datetime.now().strftime('%H:%M')} - {', '.join(management_actions)}"
            state.management_log.append(log_entry)
            logger.info(f"{state.symbol} #{ticket}: {log_entry}")
        
        state.last_update = datetime.now(timezone.utc)
        return True
    
    def _calculate_scale_levels(self, symbol: str, signal: str, entry_price: float,
                               sl_price: float, narrative, market_data) -> List[Dict]:
        """Calculate ICT-based scaling levels using order blocks and FVGs."""
        scale_levels = []
        
        if not self.scale_in_enabled:
            return scale_levels
        
        # Get analysis for potential scale levels
        analysis = self.signal_generator.analyzer.analyze(market_data, symbol)
        if not analysis:
            return scale_levels
        
        direction_filter = 'bullish' if signal == "BUY" else 'bearish'
        risk_range = abs(entry_price - sl_price)
        
        # Scale Level 1: Nearby Order Block (highest priority)
        order_blocks = analysis.get('order_blocks', [])
        for ob in order_blocks:
            if ob['type'] == direction_filter:
                # For BUY: look for bullish OB below entry
                # For SELL: look for bearish OB above entry
                ob_price = (ob['top'] + ob['bottom']) / 2
                
                if signal == "BUY" and entry_price * 0.997 <= ob_price <= entry_price * 0.9985:
                    scale_levels.append({
                        'price': ob_price,
                        'volume_ratio': 0.5,  # 50% of initial size
                        'trigger_type': 'order_block',
                        'max_risk': risk_range * 0.5
                    })
                elif signal == "SELL" and entry_price * 1.0015 >= ob_price >= entry_price * 1.003:
                    scale_levels.append({
                        'price': ob_price, 
                        'volume_ratio': 0.5,
                        'trigger_type': 'order_block',
                        'max_risk': risk_range * 0.5
                    })
        
        # Scale Level 2: Fair Value Gap (medium priority)
        fvgs = analysis.get('fair_value_gaps', [])
        for fvg in fvgs:
            if fvg['type'] == direction_filter:
                fvg_price = (fvg['top'] + fvg['bottom']) / 2
                
                # Check if FVG is in reasonable scaling distance
                distance = abs(fvg_price - entry_price) / entry_price
                if 0.0005 <= distance <= 0.002:  # 5-20 pips for major pairs
                    scale_levels.append({
                        'price': fvg_price,
                        'volume_ratio': 0.3,  # 30% of initial size
                        'trigger_type': 'fair_value_gap',
                        'max_risk': risk_range * 0.3
                    })
        
        # Scale Level 3: OTE Zone (if available)
        ote_zones = analysis.get('ote_zones', [])
        for ote in ote_zones:
            if ote['direction'] == signal.lower():
                # Use the "sweet spot" of OTE
                ote_price = ote.get('sweet', (ote['high'] + ote['low']) / 2)
                
                # Only if it's a reasonable pullback
                if signal == "BUY" and ote_price < entry_price:
                    scale_levels.append({
                        'price': ote_price,
                        'volume_ratio': 0.25,  # 25% of initial size
                        'trigger_type': 'ote_zone',
                        'max_risk': risk_range * 0.25
                    })
                elif signal == "SELL" and ote_price > entry_price:
                    scale_levels.append({
                        'price': ote_price,
                        'volume_ratio': 0.25,
                        'trigger_type': 'ote_zone', 
                        'max_risk': risk_range * 0.25
                    })
        
        # Sort by distance from entry (closest first)
        scale_levels.sort(key=lambda x: abs(x['price'] - entry_price))
        
        # Limit to max 2 scale levels to prevent over-exposure
        return scale_levels[:2]
    
    def _calculate_exit_targets(self, symbol: str, signal: str, entry_price: float,
                               initial_tp: float, narrative, market_data) -> List[Dict]:
        """Calculate ICT liquidity-based exit targets."""
        exit_targets = []
        
        # Get liquidity analysis
        analysis = self.signal_generator.analyzer.analyze(market_data, symbol)
        if not analysis:
            # Fallback to initial TP
            return [{'price': initial_tp, 'volume_pct': 100.0, 'type': 'initial_target'}]
        
        liquidity_levels = analysis.get('liquidity_levels', {'buy_side': [], 'sell_side': []})
        
        # Target liquidity based on bias
        if signal == "BUY":
            # Target buy-side liquidity (above price)
            targets = [liq for liq in liquidity_levels['buy_side'] if liq['level'] > entry_price]
        else:
            # Target sell-side liquidity (below price)  
            targets = [liq for liq in liquidity_levels['sell_side'] if liq['level'] < entry_price]
        
        # Sort by priority and distance
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        targets.sort(key=lambda x: (priority_order.get(x['priority'], 999), 
                                   abs(x['level'] - entry_price)))
        
        # Create exit plan
        if len(targets) >= 2:
            # First target: 50% at nearest high-priority liquidity
            exit_targets.append({
                'price': targets[0]['level'],
                'volume_pct': 50.0,
                'type': f"liquidity_{targets[0]['type']}",
                'description': targets[0]['description']
            })
            
            # Second target: 25% at next liquidity level
            exit_targets.append({
                'price': targets[1]['level'], 
                'volume_pct': 25.0,
                'type': f"liquidity_{targets[1]['type']}",
                'description': targets[1]['description']
            })
            
            # Runner: 25% at extended target or initial TP
            runner_target = targets[2]['level'] if len(targets) > 2 else initial_tp
            exit_targets.append({
                'price': runner_target,
                'volume_pct': 25.0,
                'type': 'runner',
                'description': 'Runner position'
            })
        
        elif len(targets) == 1:
            # Single target: 75% at liquidity, 25% runner
            exit_targets.append({
                'price': targets[0]['level'],
                'volume_pct': 75.0,
                'type': f"liquidity_{targets[0]['type']}",
                'description': targets[0]['description']
            })
            
            exit_targets.append({
                'price': initial_tp,
                'volume_pct': 25.0, 
                'type': 'runner',
                'description': 'Runner to initial target'
            })
        
        else:
            # No liquidity targets found - use staged exits with initial TP
            risk_range = abs(entry_price - narrative.manipulation_level) if hasattr(narrative, 'manipulation_level') else abs(entry_price - initial_tp) / 2
            
            # Conservative staged exit
            first_target = entry_price + (risk_range * 1.5) if signal == "BUY" else entry_price - (risk_range * 1.5)
            
            exit_targets = [
                {'price': first_target, 'volume_pct': 50.0, 'type': '1.5R_target', 'description': '1.5R Profit'},
                {'price': initial_tp, 'volume_pct': 50.0, 'type': 'final_target', 'description': 'Final Target'}
            ]
        
        return exit_targets
    
    def _should_move_to_break_even(self, state: PositionState, plan: PositionPlan, current_price: float) -> bool:
        """Check if position should move to break-even."""
        if state.break_even_triggered:
            return False
        
        if plan.direction == "BUY":
            return current_price >= plan.break_even_trigger
        else:
            return current_price <= plan.break_even_trigger
    
    def _should_move_to_risk_free(self, state: PositionState, plan: PositionPlan, current_price: float) -> bool:
        """Check if position should move to risk-free."""
        if state.is_risk_free:
            return False
        
        if plan.direction == "BUY":
            return current_price >= plan.risk_free_trigger
        else:
            return current_price <= plan.risk_free_trigger
    
    def _move_stop_to_break_even(self, ticket: int, state: PositionState, plan: PositionPlan) -> bool:
        """Move stop loss to break-even."""
        try:
            new_sl = state.average_entry
            
            # Add small buffer to account for spread
            symbol_info = self.mt5_client.get_symbol_info(state.symbol)
            if symbol_info:
                buffer = symbol_info.point * 5  # 5 point buffer
                if plan.direction == "BUY":
                    new_sl -= buffer
                else:
                    new_sl += buffer
            
            # Modify position
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": state.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": state.current_targets[0]['price'] if state.current_targets else 0
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                state.current_stop = new_sl
                logger.info(f"{state.symbol} #{ticket}: Stop moved to break-even at {new_sl:.5f}")
                return True
            else:
                logger.error(f"Failed to move stop to break-even: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error moving stop to break-even: {e}")
            return False
    
    def _move_stop_to_risk_free(self, ticket: int, state: PositionState, plan: PositionPlan) -> bool:
        """Move stop loss to risk-free (small profit)."""
        try:
            # Risk-free = entry + small profit (0.25R)
            risk_range = abs(state.average_entry - plan.initial_stop)
            risk_free_profit = risk_range * 0.25
            
            if plan.direction == "BUY":
                new_sl = state.average_entry + risk_free_profit
            else:
                new_sl = state.average_entry - risk_free_profit
            
            # Modify position
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": state.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": state.current_targets[0]['price'] if state.current_targets else 0
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                state.current_stop = new_sl
                logger.info(f"{state.symbol} #{ticket}: Stop moved to risk-free at {new_sl:.5f}")
                return True
            else:
                logger.error(f"Failed to move stop to risk-free: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error moving stop to risk-free: {e}")
            return False
    
    def _should_take_partial_profit(self, state: PositionState, plan: PositionPlan, current_price: float) -> bool:
        """Check if should take partial profit at target levels."""
        if not state.current_targets:
            return False
        
        # Check first available target
        first_target = state.current_targets[0]
        
        if plan.direction == "BUY":
            return current_price >= first_target['price']
        else:
            return current_price <= first_target['price']
    
    def _execute_partial_exit(self, ticket: int, state: PositionState, plan: PositionPlan, current_price: float) -> bool:
        """Execute partial exit at target level."""
        if not state.current_targets:
            return False
        
        try:
            target = state.current_targets[0]
            exit_volume = state.current_volume * (target['volume_pct'] / 100.0)
            
            # Round to lot step
            symbol_info = self.mt5_client.get_symbol_info(state.symbol)
            if symbol_info:
                lot_step = symbol_info.volume_step
                exit_volume = round(exit_volume / lot_step) * lot_step
                exit_volume = max(symbol_info.volume_min, min(exit_volume, state.current_volume))
            
            # Execute partial close
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": state.symbol,
                "volume": float(exit_volume),
                "type": mt5.ORDER_TYPE_SELL if plan.direction == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": current_price,
                "deviation": 20,
                "comment": f"ICT_PARTIAL_{target['type']}"
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Update state
                state.current_volume -= exit_volume
                state.scale_outs.append({
                    'time': datetime.now(timezone.utc),
                    'price': current_price,
                    'volume': exit_volume,
                    'type': target['type'],
                    'reason': f"Target: {target['description']}"
                })
                
                # Remove this target
                state.current_targets.pop(0)
                
                logger.info(f"{state.symbol} #{ticket}: Partial exit {exit_volume} lots at {current_price:.5f} "
                           f"({target['type']}) - Remaining: {state.current_volume} lots")
                return True
            else:
                logger.error(f"Partial exit failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing partial exit: {e}")
            return False
    
    def _should_scale_in(self, state: PositionState, plan: PositionPlan, current_price: float, ohlc_df) -> bool:
        """Check if should scale into position."""
        if not plan.scale_levels or len(state.scale_ins) >= len(plan.scale_levels):
            return False
        
        # Check total risk limit
        total_risk = state.risk_amount + sum(si['risk'] for si in state.scale_ins)
        max_total_risk = plan.initial_risk * 2.0  # Max 2x initial risk
        
        if total_risk >= max_total_risk:
            return False
        
        # Check next scale level
        next_scale_index = len(state.scale_ins)
        next_scale = plan.scale_levels[next_scale_index]
        
        # Price condition
        if plan.direction == "BUY":
            price_hit = current_price <= next_scale['price']
        else:
            price_hit = current_price >= next_scale['price']
        
        if not price_hit:
            return False
        
        # Additional confirmation: check if we're at a valid ICT level
        return self._confirm_scale_level(next_scale, current_price, ohlc_df)
    
    def _confirm_scale_level(self, scale_level, current_price, ohlc_df) -> bool:
        """Confirm scale level is still valid using ICT concepts."""
        # Simple confirmation - check if we're near the expected level
        distance = abs(current_price - scale_level['price']) / current_price
        return distance <= 0.0005  # Within 5 pips
    
    def _execute_scale_in(self, ticket: int, state: PositionState, plan: PositionPlan, current_price: float) -> bool:
        """Execute scale-in addition to position."""
        try:
            next_scale_index = len(state.scale_ins)
            scale_level = plan.scale_levels[next_scale_index]
            
            # Calculate add volume
            add_volume = plan.initial_volume * scale_level['volume_ratio']
            
            # Round to lot step
            symbol_info = self.mt5_client.get_symbol_info(state.symbol)
            if symbol_info:
                lot_step = symbol_info.volume_step
                add_volume = round(add_volume / lot_step) * lot_step
                add_volume = max(symbol_info.volume_min, add_volume)
            
            # Execute add
            order_type = mt5.ORDER_TYPE_BUY if plan.direction == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": state.symbol,
                "volume": float(add_volume),
                "type": order_type,
                "price": current_price,
                "deviation": 20,
                "comment": f"ICT_ADD_{scale_level['trigger_type']}"
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Calculate new average entry
                total_volume = state.current_volume + add_volume
                new_avg_entry = ((state.average_entry * state.current_volume) + 
                               (current_price * add_volume)) / total_volume
                
                # Update state
                state.current_volume = total_volume
                state.average_entry = new_avg_entry
                state.scale_ins.append({
                    'time': datetime.now(timezone.utc),
                    'price': current_price,
                    'volume': add_volume,
                    'type': scale_level['trigger_type'],
                    'risk': scale_level['max_risk']
                })
                
                logger.info(f"{state.symbol} #{ticket}: Scaled in {add_volume} lots at {current_price:.5f} "
                           f"({scale_level['trigger_type']}) - Total: {state.current_volume} lots, "
                           f"Avg: {new_avg_entry:.5f}")
                return True
            else:
                logger.error(f"Scale-in failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing scale-in: {e}")
            return False
    
    def _should_trail_stop(self, state: PositionState, plan: PositionPlan, current_price: float, ohlc_df) -> bool:
        """Check if should update trailing stop."""
        if not state.is_risk_free or not state.current_targets:
            return False
        
        # Only trail if we have runner position
        return len(state.current_targets) == 1 and state.current_targets[0]['type'] == 'runner'
    
    def _update_trailing_stop(self, ticket: int, state: PositionState, plan: PositionPlan, 
                             current_price: float, ohlc_df) -> bool:
        """Update trailing stop based on ICT structure."""
        try:
            # Use recent swing low/high as trailing reference
            lookback = min(20, len(ohlc_df))
            recent_data = ohlc_df.tail(lookback)
            
            if plan.direction == "BUY":
                # Trail using recent swing low
                trail_reference = recent_data['low'].min()
                # Add buffer
                symbol_info = self.mt5_client.get_symbol_info(state.symbol)
                buffer = symbol_info.point * 10 if symbol_info else 0.0001
                new_stop = trail_reference - buffer
                
                # Only move up
                if new_stop > state.current_stop:
                    success = self._modify_stop_loss(ticket, state, new_stop)
                    if success:
                        logger.info(f"{state.symbol} #{ticket}: Trailing stop updated to {new_stop:.5f}")
                    return success
            else:
                # Trail using recent swing high
                trail_reference = recent_data['high'].max()
                symbol_info = self.mt5_client.get_symbol_info(state.symbol)
                buffer = symbol_info.point * 10 if symbol_info else 0.0001
                new_stop = trail_reference + buffer
                
                # Only move down
                if new_stop < state.current_stop:
                    success = self._modify_stop_loss(ticket, state, new_stop)
                    if success:
                        logger.info(f"{state.symbol} #{ticket}: Trailing stop updated to {new_stop:.5f}")
                    return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return False
    
    def _should_exit_on_structure(self, state: PositionState, plan: PositionPlan, 
                                 current_price: float, ohlc_df) -> bool:
        """Check if should exit due to structure break."""
        # Simplified structure check - look for significant counter-move
        if len(ohlc_df) < 10:
            return False
        
        recent_data = ohlc_df.tail(10)
        
        if plan.direction == "BUY":
            # Check for bearish structure break
            recent_low = recent_data['low'].min()
            prev_low = ohlc_df.iloc[-11:-1]['low'].min()
            return recent_low < prev_low  # Lower low formed
        else:
            # Check for bullish structure break  
            recent_high = recent_data['high'].max()
            prev_high = ohlc_df.iloc[-11:-1]['high'].max()
            return recent_high > prev_high  # Higher high formed
    
    def _execute_structure_exit(self, ticket: int, state: PositionState, plan: PositionPlan) -> bool:
        """Execute full exit due to structure break."""
        try:
            current_price = self.mt5_client.get_symbol_ticker(state.symbol)
            if not current_price:
                return False
            
            price = current_price.bid if plan.direction == "BUY" else current_price.ask
            
            # Close entire position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": state.symbol,
                "volume": state.current_volume,
                "type": mt5.ORDER_TYPE_SELL if plan.direction == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "comment": "ICT_STRUCTURE_EXIT"
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                state.management_log.append(f"Structure exit at {price:.5f}")
                self._close_position_tracking(ticket, ExitReason.STRUCTURE_BREAK)
                logger.info(f"{state.symbol} #{ticket}: Full exit due to structure break at {price:.5f}")
                return True
            else:
                logger.error(f"Structure exit failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing structure exit: {e}")
            return False
    
    def _modify_stop_loss(self, ticket: int, state: PositionState, new_stop: float) -> bool:
        """Modify stop loss for position."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": state.symbol,
                "position": ticket,
                "sl": new_stop,
                "tp": state.current_targets[0]['price'] if state.current_targets else 0
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                state.current_stop = new_stop
                return True
            else:
                logger.error(f"Stop modification failed: {result.comment if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying stop loss: {e}")
            return False
    
    def _update_profit_metrics(self, state: PositionState, current_price: float):
        """Update profit metrics for position."""
        try:
            symbol_info = self.mt5_client.get_symbol_info(state.symbol)
            if not symbol_info:
                return
            
            # Calculate profit in price terms
            if state.direction == "BUY":
                price_diff = current_price - state.average_entry
            else:
                price_diff = state.average_entry - current_price
            
            # Convert to pips
            state.profit_pips = price_diff / symbol_info.point
            
            # Calculate profit amount
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            state.profit_amount = (price_diff / tick_size) * tick_value * state.current_volume
            
            # Calculate R:R ratio
            initial_risk = abs(state.average_entry - state.current_stop)
            if initial_risk > 0:
                state.rr_ratio = abs(price_diff) / initial_risk
            else:
                state.rr_ratio = 0.0
                
        except Exception as e:
            logger.error(f"Error updating profit metrics: {e}")
    
    def _handle_external_closure(self, ticket: int):
        """Handle position closed externally (stop loss, take profit, etc.)."""
        if ticket in self.managed_positions:
            state = self.managed_positions[ticket]
            state.management_log.append("Position closed externally")
            self._close_position_tracking(ticket, ExitReason.MANUAL)
    
    def _close_position_tracking(self, ticket: int, reason: ExitReason):
        """Close position tracking and log final statistics."""
        if ticket not in self.managed_positions:
            return
        
        state = self.managed_positions[ticket]
        plan = self.position_plans[ticket]
        
        # Log final statistics
        logger.info(f"\n--- POSITION CLOSED: {state.symbol} #{ticket} ---")
        logger.info(f"Entry Model: {plan.narrative_model}")
        logger.info(f"Final Profit: {state.profit_amount:.2f} ({state.profit_pips:.1f} pips)")
        logger.info(f"Final R:R: {state.rr_ratio:.2f}")
        logger.info(f"Scale-ins: {len(state.scale_ins)}")
        logger.info(f"Partial exits: {len(state.scale_outs)}")
        logger.info(f"Exit reason: {reason.value}")
        logger.info("Management log:")
        for log_entry in state.management_log[-5:]:  # Last 5 entries
            logger.info(f"  {log_entry}")
        logger.info("--- END POSITION LOG ---\n")
        
        # Remove from tracking
        del self.managed_positions[ticket]
        del self.position_plans[ticket]
    
    def get_position_summary(self) -> Dict:
        """Get summary of all managed positions."""
        if not self.managed_positions:
            return {'total_positions': 0}
        
        summary = {
            'total_positions': len(self.managed_positions),
            'total_profit': sum(state.profit_amount for state in self.managed_positions.values()),
            'total_pips': sum(state.profit_pips for state in self.managed_positions.values()),
            'avg_rr_ratio': np.mean([state.rr_ratio for state in self.managed_positions.values()]),
            'risk_free_positions': sum(1 for state in self.managed_positions.values() if state.is_risk_free),
            'break_even_positions': sum(1 for state in self.managed_positions.values() if state.break_even_triggered),
            'positions_by_stage': {}
        }
        
        # Count by stage
        for state in self.managed_positions.values():
            stage = state.stage.value
            summary['positions_by_stage'][stage] = summary['positions_by_stage'].get(stage, 0) + 1
        
        return summary
    
    def force_close_all_positions(self, reason: str = "Manual close"):
        """Force close all managed positions."""
        logger.info(f"Force closing {len(self.managed_positions)} managed positions: {reason}")
        
        for ticket in list(self.managed_positions.keys()):
            try:
                state = self.managed_positions[ticket]
                current_price = self.mt5_client.get_symbol_ticker(state.symbol)
                
                if current_price:
                    price = current_price.bid if state.direction == "BUY" else current_price.ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": state.symbol,
                        "volume": state.current_volume,
                        "type": mt5.ORDER_TYPE_SELL if state.direction == "BUY" else mt5.ORDER_TYPE_BUY,
                        "position": ticket,
                        "price": price,
                        "deviation": 20,
                        "comment": f"ICT_FORCE_CLOSE"
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Force closed {state.symbol} #{ticket}")
                        self._close_position_tracking(ticket, ExitReason.MANUAL)
                    else:
                        logger.error(f"Failed to force close {ticket}: {result.comment if result else 'Unknown error'}")
                        
            except Exception as e:
                logger.error(f"Error force closing position {ticket}: {e}")