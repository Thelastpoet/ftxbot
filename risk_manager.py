"""
Risk Manager - Core version
"""

import logging
from typing import Any

from utils import resolve_pip_size

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk and position sizing (core)."""

    def __init__(self, config, mt5_client):
        self.config = config
        self.mt5_client = mt5_client
        self.risk_per_trade = getattr(config, 'risk_per_trade', 0.01)
        self.fixed_lot_size = getattr(config, 'fixed_lot_size', None)
        self.max_drawdown = getattr(config, 'max_drawdown', 0.05)
        self.use_equity = bool(getattr(config, 'use_equity', True))

    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk parameters.
        Returns position size in lots.
        """
        if self.fixed_lot_size is not None:
            return float(self.fixed_lot_size)

        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info for position sizing")
                return 0.0

            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0

            balance = float(account_info.balance)
            equity = float(getattr(account_info, 'equity', balance))
            base = equity if self.use_equity else balance
            risk_amount = base * float(self.risk_per_trade)

            pip_value = self._calculate_pip_value(symbol_info, symbol)
            if stop_loss_pips > 0 and pip_value > 0:
                position_size = risk_amount / (stop_loss_pips * pip_value)

                # Round to broker lot step & clamp to min/max
                step = getattr(symbol_info, 'volume_step', 0.01) or 0.01
                vmin = getattr(symbol_info, 'volume_min', step)
                vmax = getattr(symbol_info, 'volume_max', 100.0)
                position_size = round(position_size / step) * step
                position_size = max(vmin, min(position_size, vmax))
                logger.info(f"Calculated position size for {symbol}: {position_size} lots")
                return float(position_size)

            logger.warning(f"Invalid SL or pip value for {symbol}")
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _calculate_pip_value(self, symbol_info: Any, symbol: str) -> float:
        """Calculate pip value per standard lot using MT5 native values."""
        try:
            if getattr(symbol_info, 'trade_tick_size', 0) > 0:
                pip_size = resolve_pip_size(symbol, symbol_info, self.config)
                per_unit = float(symbol_info.trade_tick_value) / float(symbol_info.trade_tick_size)
                return float(per_unit * pip_size)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating pip value: {e}")
            return 0.0

    def check_risk_limits(self) -> bool:
        """Basic drawdown and margin checks."""
        try:
            info = self.mt5_client.get_account_info()
            if not info:
                return False

            balance = float(info.balance)
            equity = float(info.equity)
            if balance > 0:
                dd = (balance - equity) / balance
                if dd >= float(self.max_drawdown):
                    logger.warning(f"Max drawdown exceeded: {dd:.2%}")
                    return False

            margin_level = float(getattr(info, 'margin_level', 0.0) or 0.0)
            if margin_level > 0 and margin_level < 200:
                logger.warning(f"Low margin level: {margin_level:.1f}%")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def validate_trade_parameters(self, symbol: str, volume: float, stop_loss: float, take_profit: float, order_type: int) -> bool:
        """Validate trade parameters before execution."""
        try:
            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                return False

            if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
                logger.error(f"Invalid volume: {volume} (min: {symbol_info.volume_min}, max: {symbol_info.volume_max})")
                return False

            tick = self.mt5_client.get_symbol_info_tick(symbol)
            if not tick:
                return False

            # Side-aware current price
            current_price = float(tick.ask) if int(order_type) == 0 else float(tick.bid)
            # Validate SL/TP are on correct sides of current price
            if int(order_type) == 0:
                if not (take_profit > current_price > stop_loss):
                    logger.error("Invalid price placement for BUY: expected TP > price > SL")
                    return False
            else:
                if not (take_profit < current_price < stop_loss):
                    logger.error("Invalid price placement for SELL: expected TP < price < SL")
                    return False
            min_stop_distance = float(symbol_info.trade_stops_level) * float(symbol_info.point)

            if abs(current_price - stop_loss) < min_stop_distance:
                logger.error("Stop loss too close to current price")
                return False

            if abs(current_price - take_profit) < min_stop_distance:
                logger.error("Take profit too close to current price")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return False
