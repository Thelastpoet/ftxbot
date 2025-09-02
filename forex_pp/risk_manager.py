"""
Risk Manager Module
Handles position sizing and risk management calculations
"""

import logging
from typing import Dict, Any

from utils import get_pip_size

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk and position sizing"""

    def __init__(self, config, mt5_client):
        self.config = config
        self.mt5_client = mt5_client
        self.risk_per_trade = config.risk_per_trade
        self.fixed_lot_size = config.fixed_lot_size
        self.max_drawdown = config.max_drawdown

    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk parameters

        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips

        Returns:
            Position size in lots
        """
        # If fixed lot size is specified, use it
        if self.fixed_lot_size is not None:
            return self.fixed_lot_size

        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info for position sizing")
                return 0.0

            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0

            account_balance = account_info.balance
            risk_amount = account_balance * self.risk_per_trade

            pip_value = self._calculate_pip_value(symbol_info)

            if stop_loss_pips > 0 and pip_value > 0:
                # Risk per lot = stop_loss_pips * pip_value
                position_size = risk_amount / (stop_loss_pips * pip_value)

                # Round to symbol’s lot step
                lot_step = symbol_info.volume_step
                position_size = round(position_size / lot_step) * lot_step

                # Apply min/max constraints
                position_size = max(symbol_info.volume_min,
                                    min(position_size, symbol_info.volume_max))

                logger.info(f"Calculated position size for {symbol}: {position_size} lots")
                return position_size
            else:
                logger.warning(f"Invalid stop loss or pip value for {symbol}")
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _calculate_pip_value(self, symbol_info: Any) -> float:
        """
        Calculate pip value per standard lot using MT5 native values

        Args:
            symbol_info: Symbol information from MT5

        Returns:
            Pip value per lot
        """
        try:
            if symbol_info.trade_tick_size > 0:
                pip_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                return pip_value
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error calculating pip value: {e}")
            return 0.0

    def calculate_stop_loss_pips(self, entry_price: float, stop_loss: float, symbol_info: Any) -> float:
        """
        Calculate stop loss in clean pip units

        Args:
            entry_price: Trade entry price
            stop_loss: Stop loss price
            symbol_info: MT5 symbol info

        Returns:
            Stop loss distance in pips
        """
        try:
            if not symbol_info:
                return 0.0
            
            pip_size = get_pip_size(symbol_info)
            
            if pip_size <= 0:
                return 0.0
            
            return abs(entry_price - stop_loss) / pip_size
        except Exception as e:
            logger.error(f"Error calculating stop loss pips: {e}")
            return 0.0

    def check_risk_limits(self, symbol: str) -> bool:
        """Check if trading is allowed based on risk limits"""
        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return False

            equity = account_info.equity
            balance = account_info.balance

            if balance > 0:
                current_drawdown = (balance - equity) / balance
                if current_drawdown >= self.max_drawdown:
                    logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%}")
                    return False

            if account_info.margin_level > 0 and account_info.margin_level < 200:
                logger.warning(f"Low margin level: {account_info.margin_level}%")
                return False

            positions = self.mt5_client.get_positions(symbol)
            if len(positions) >= 2:
                logger.info(f"Position already open for {symbol}, skipping new signal")
                return False

            all_positions = self.mt5_client.get_all_positions()
            if len(all_positions) >= 20:
                logger.warning(f"Maximum total positions reached: {len(all_positions)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics"""
        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return {}

            positions = self.mt5_client.get_all_positions()
            total_exposure = sum(p.volume * p.price_current for p in positions)

            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'margin_level': account_info.margin_level if account_info.margin > 0 else 0,
                'profit': account_info.profit,
                'drawdown': (account_info.balance - account_info.equity) / account_info.balance if account_info.balance > 0 else 0,
                'open_positions': len(positions),
                'total_exposure': total_exposure
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def adjust_position_size_for_correlation(self, symbol: str, base_position_size: float) -> float:
        """Adjust position size based on correlation with existing positions"""
        try:
            positions = self.mt5_client.get_all_positions()
            correlated_count = 0
            for position in positions:
                if self._are_correlated(symbol, position.symbol):
                    correlated_count += 1

            if correlated_count > 0:
                reduction_factor = 1.0 / (1 + correlated_count * 0.5)
                adjusted_size = base_position_size * reduction_factor
                logger.info(f"Position size adjusted for correlation: {base_position_size} -> {adjusted_size}")
                return adjusted_size

            return base_position_size

        except Exception as e:
            logger.error(f"Error adjusting position size for correlation: {e}")
            return base_position_size

    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are correlated (simple currency overlap check)"""
        if len(symbol1) >= 6 and len(symbol2) >= 6:
            base1, quote1 = symbol1[:3], symbol1[3:6]
            base2, quote2 = symbol2[:3], symbol2[3:6]
            return (base1 == base2 or base1 == quote2 or
                    quote1 == base2 or quote1 == quote2)
        return False

    def validate_trade_parameters(self, symbol: str, volume: float, stop_loss: float, take_profit: float) -> bool:
        """Validate trade parameters before execution"""
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

            current_price = tick.ask
            min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

            if abs(current_price - stop_loss) < min_stop_distance:
                logger.error(f"Stop loss too close to current price")
                return False

            if abs(current_price - take_profit) < min_stop_distance:
                logger.error(f"Take profit too close to current price")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating trade parameters: {e}")
            return False
