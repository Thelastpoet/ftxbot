"""
Risk Manager Module
Handles position sizing and risk management calculations
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any

from utils import get_pip_size

logger = logging.getLogger(__name__)

class RiskManager:
    """Manages risk and position sizing"""

    def __init__(self, risk_management: dict, mt5_client):
        self.config = risk_management
        self.mt5_client = mt5_client
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        self.fixed_lot_size = self.config.get('fixed_lot_size')
        self.max_drawdown = self.config.get('max_drawdown_percentage', 0.05)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.correlation_lookback_period = self.config.get('correlation_lookback_period', 30)

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

                # Adjust for correlation
                position_size = self.adjust_position_size_for_correlation(symbol, position_size)

                # Round to symbol’s lot step
                lot_step = symbol_info.volume_step
                position_size = round(position_size / lot_step) * lot_step

                # Apply min/max constraints
                position_size = max(symbol_info.volume_min,
                                    min(position_size, symbol_info.volume_max))

                logger.debug(f"Calculated position size for {symbol}: {position_size} lots")
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
                # Convert native tick value per tick-size to value per pip
                pip_size = get_pip_size(symbol_info)
                per_price_unit = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                pip_value = per_price_unit * pip_size
                return float(pip_value)
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

    def check_risk_limits(self, symbol: str, signal_type: int = None) -> bool:
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

            all_positions = self.mt5_client.get_all_positions()
            if len(all_positions) >= 20:
                logger.warning(f"Maximum total positions reached: {len(all_positions)}")
                return False
            
            if signal_type is not None:
                if not self.check_correlation_risk(symbol, signal_type):
                    return False
            
            return True

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
        
    def check_correlation_risk(self, new_symbol: str, proposed_signal_type: int) -> bool:
        """
        Prevent multiple correlated positions to avoid risk multiplication
        Returns True if safe to trade, False if correlation risk too high
        """
        try:
            # Get existing open positions
            existing_positions = self.mt5_client.get_all_positions()
            if not existing_positions:
                return True  # No positions, safe to trade
            
            for position in existing_positions:
                # Skip if same symbol (already checked elsewhere)
                if position.symbol == new_symbol:
                    continue
                
                current_corr = self._get_correlation(new_symbol, position.symbol)
                if current_corr is None:
                    continue

                # Check if same or opposite direction
                same_direction = (
                    (position.type == 0 and proposed_signal_type == 0) or  # Both long
                    (position.type == 1 and proposed_signal_type == 1)     # Both short
                )

                if same_direction and abs(current_corr) > self.correlation_threshold:
                    logger.debug(
                        f"HIGH CORRELATION RISK: {new_symbol} has {current_corr:.2f} "
                        f"correlation with open {position.symbol} position"
                    )
                    return False

                # Opposite direction with negative correlation is also risky
                if not same_direction and current_corr < -self.correlation_threshold:
                    logger.debug(
                        f"INVERSE CORRELATION RISK: {new_symbol} has {current_corr:.2f} "
                        f"correlation with opposite {position.symbol} position"
                    )
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False  # Be conservative on error

    def _get_correlation(self, symbol1: str, symbol2: str) -> float | None:
        """Calculate Pearson correlation between two symbols"""
        try:
            # Get recent data for correlation calculation
            data1 = self.mt5_client.copy_rates_from_pos(symbol1, 'M15', 0, self.correlation_lookback_period)
            if data1 is None:
                logger.warning(f"Cannot get data for {symbol1} correlation check")
                return None
            
            df1 = pd.DataFrame(data1)
            close1 = df1['close'].values
            
            data2 = self.mt5_client.copy_rates_from_pos(symbol2, 'M15', 0, self.correlation_lookback_period)
            if data2 is None:
                logger.warning(f"Cannot get data for {symbol2} correlation check")
                return None
            
            df2 = pd.DataFrame(data2)
            close2 = df2['close'].values
            
            # Calculate correlation on aligned windows using Pearson coefficient
            N = min(len(close1), len(close2), self.correlation_lookback_period)
            if N >= 10:
                a = close1[-N:]
                b = close2[-N:]
                # Ensure finite values
                if np.std(a) > 0 and np.std(b) > 0:
                    corr_matrix = np.corrcoef(a, b)
                    return float(corr_matrix[0, 1])
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics"""
        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return {}

            positions = self.mt5_client.get_all_positions()
            total_exposure = 0.0
            for p in positions:
                try:
                    sym_info = self.mt5_client.get_symbol_info(p.symbol)
                    if sym_info and hasattr(sym_info, 'trade_contract_size'):
                        contract_size = float(sym_info.trade_contract_size)
                    else:
                        # Fallback: assume 100000 for FX if unknown
                        contract_size = 100000.0
                    # Notional exposure in quote currency (approx): lots * contract_size * price
                    total_exposure += float(p.volume) * contract_size * float(p.price_current)
                except Exception:
                    total_exposure += float(p.volume) * float(p.price_current)

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
            if not positions:
                return base_position_size

            total_correlation_risk = 0.0
            correlated_count = 0

            for position in positions:
                if position.symbol == symbol:
                    continue

                correlation = self._get_correlation(symbol, position.symbol)
                if correlation is not None and abs(correlation) > self.correlation_threshold:
                    total_correlation_risk += abs(correlation)
                    correlated_count += 1

            if correlated_count > 0:
                # Average the correlation risk
                avg_correlation = total_correlation_risk / correlated_count
                # Reduce position size based on the number of correlated pairs and their average strength
                reduction_factor = 1.0 / (1 + correlated_count * avg_correlation)
                adjusted_size = base_position_size * reduction_factor
                
                logger.debug(f"Position size adjusted for correlation: {base_position_size:.2f} -> {adjusted_size:.2f} "
                            f"(due to {correlated_count} correlated positions)")
                return adjusted_size

            return base_position_size

        except Exception as e:
            logger.error(f"Error adjusting position size for correlation: {e}")
            return base_position_size

    def validate_trade_parameters(self, symbol: str, volume: float, stop_loss: float, take_profit: float, order_type: int = None) -> bool:
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

            # Minimum broker stop distance in price units
            min_stop_distance = float(symbol_info.trade_stops_level) * float(symbol_info.point)

            # If order_type provided, enforce directional SL/TP placement and min distances
            if order_type is not None:
                if order_type == 0:  # BUY
                    current_price = float(tick.ask)
                    # SL must be below current price by at least min distance
                    if not (stop_loss < current_price - min_stop_distance):
                        logger.error("Stop loss invalid for BUY (too close or above current price)")
                        return False
                    # TP must be above current price by at least min distance
                    if not (take_profit > current_price + min_stop_distance):
                        logger.error("Take profit invalid for BUY (too close or below current price)")
                        return False
                else:  # SELL
                    current_price = float(tick.bid)
                    # SL must be above current price by at least min distance
                    if not (stop_loss > current_price + min_stop_distance):
                        logger.error("Stop loss invalid for SELL (too close or below current price)")
                        return False
                    # TP must be below current price by at least min distance
                    if not (take_profit < current_price - min_stop_distance):
                        logger.error("Take profit invalid for SELL (too close or above current price)")
                        return False
            else:
                # Fallback absolute-distance checks if direction not provided
                current_price = float(tick.ask)
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
