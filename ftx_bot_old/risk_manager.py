"""
Risk Manager Module
Handles position sizing and risk management calculations
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, Optional

from utils import get_pip_size

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk and position sizing with optional per-symbol overrides."""

    def __init__(self, risk_management: dict, mt5_client):
        self.config = risk_management or {}
        self.mt5_client = mt5_client
        self.risk_per_trade = self._get_config_value('risk_per_trade', 0.01)
        self.fixed_lot_size = self._get_config_value('fixed_lot_size')
        self.max_drawdown = self._get_config_value('max_drawdown_percentage', 0.05)
        self.correlation_threshold = self._get_config_value('correlation_threshold', 0.7)
        self.correlation_lookback_period = self._get_config_value('correlation_lookback_period', 30)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        source = self.config
        if isinstance(source, dict):
            return source.get(key, default)
        getter = getattr(source, 'get', None)
        if callable(getter):
            try:
                return getter(key, default)
            except TypeError:
                pass
        if hasattr(source, key):
            return getattr(source, key)
        if hasattr(source, '__dict__') and key in source.__dict__:
            return source.__dict__.get(key, default)
        return default

    def _resolve_setting(
        self,
        key: str,
        overrides: Optional[Dict[str, float]],
        *,
        attr_key: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        if overrides and key in overrides and overrides[key] is not None:
            try:
                return float(overrides[key])
            except (TypeError, ValueError):
                return overrides[key]
        attribute = attr_key or key
        return getattr(self, attribute, default)

    # ------------------------------------------------------------------
    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate position size based on risk parameters."""
        overrides = overrides or {}
        fixed_lot = self._resolve_setting('fixed_lot_size', overrides, default=self.fixed_lot_size)
        if fixed_lot is not None:
            return float(fixed_lot)

        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                logger.error("Failed to get account info for position sizing")
                return 0.0

            symbol_info = self.mt5_client.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0

            risk_per_trade = self._resolve_setting('risk_per_trade', overrides, default=self.risk_per_trade)
            if not risk_per_trade or risk_per_trade <= 0:
                logger.warning("risk_per_trade not positive; skipping position sizing")
                return 0.0

            account_balance = account_info.balance
            risk_amount = account_balance * float(risk_per_trade)

            pip_value = self._calculate_pip_value(symbol_info)

            if stop_loss_pips > 0 and pip_value > 0:
                position_size = risk_amount / (stop_loss_pips * pip_value)
                position_size = self.adjust_position_size_for_correlation(symbol, position_size, overrides)

                lot_step = symbol_info.volume_step or 0.01
                position_size = round(position_size / lot_step) * lot_step

                position_size = max(symbol_info.volume_min,
                                    min(position_size, symbol_info.volume_max))

                logger.debug(
                    "%s position size: %.4f lots (risk_per_trade=%.4f)",
                    symbol,
                    position_size,
                    float(risk_per_trade),
                )
                return float(position_size)
            else:
                logger.warning(f"Invalid stop loss or pip value for {symbol}")
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _calculate_pip_value(self, symbol_info: Any) -> float:
        """Calculate pip value per standard lot using MT5 native values."""
        try:
            if symbol_info.trade_tick_size > 0:
                pip_size = get_pip_size(symbol_info)
                per_price_unit = symbol_info.trade_tick_value / symbol_info.trade_tick_size
                pip_value = per_price_unit * pip_size
                return float(pip_value)
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating pip value: {e}")
            return 0.0

    def check_risk_limits(
        self,
        symbol: str,
        signal_type: int = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Check if trading is allowed based on risk limits."""
        overrides = overrides or {}
        try:
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return False

            equity = account_info.equity
            balance = account_info.balance

            max_drawdown = self._resolve_setting(
                'max_drawdown_percentage', overrides, attr_key='max_drawdown', default=self.max_drawdown
            )
            if balance > 0 and max_drawdown is not None:
                current_drawdown = (balance - equity) / balance
                if current_drawdown >= float(max_drawdown):
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
                if not self.check_correlation_risk(symbol, signal_type, overrides=overrides):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def check_correlation_risk(
        self,
        new_symbol: str,
        proposed_signal_type: int,
        overrides: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Prevent multiple highly correlated positions."""
        overrides = overrides or {}
        try:
            existing_positions = self.mt5_client.get_all_positions()
            if not existing_positions:
                return True

            threshold = float(self._resolve_setting(
                'correlation_threshold', overrides, attr_key='correlation_threshold', default=self.correlation_threshold
            ))

            for position in existing_positions:
                if position.symbol == new_symbol:
                    continue

                current_corr = self._get_correlation(new_symbol, position.symbol, overrides)
                if current_corr is None:
                    continue

                same_direction = (
                    (position.type == 0 and proposed_signal_type == 0)
                    or (position.type == 1 and proposed_signal_type == 1)
                )

                if same_direction and abs(current_corr) > threshold:
                    logger.debug(
                        "HIGH CORRELATION RISK: %s vs %s corr=%.2f",
                        new_symbol,
                        position.symbol,
                        current_corr,
                    )
                    return False

                if not same_direction and current_corr < -threshold:
                    logger.debug(
                        "INVERSE CORRELATION RISK: %s vs %s corr=%.2f",
                        new_symbol,
                        position.symbol,
                        current_corr,
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return False

    def adjust_position_size_for_correlation(
        self,
        symbol: str,
        base_position_size: float,
        overrides: Optional[Dict[str, float]] = None,
    ) -> float:
        """Adjust position size based on correlation with existing positions."""
        overrides = overrides or {}
        try:
            positions = self.mt5_client.get_all_positions()
            if not positions:
                return base_position_size

            threshold = float(self._resolve_setting(
                'correlation_threshold', overrides, attr_key='correlation_threshold', default=self.correlation_threshold
            ))

            total_correlation_risk = 0.0
            correlated_count = 0

            for position in positions:
                if position.symbol == symbol:
                    continue

                correlation = self._get_correlation(symbol, position.symbol, overrides)
                if correlation is not None and abs(correlation) > threshold:
                    total_correlation_risk += abs(correlation)
                    correlated_count += 1

            if correlated_count > 0:
                avg_correlation = total_correlation_risk / correlated_count
                reduction_factor = 1.0 / (1 + correlated_count * avg_correlation)
                adjusted_size = base_position_size * reduction_factor

                logger.debug(
                    "Position size adjusted for correlation: %.2f -> %.2f (n=%d)",
                    base_position_size,
                    adjusted_size,
                    correlated_count,
                )
                return adjusted_size

            return base_position_size

        except Exception as e:
            logger.error(f"Error adjusting position size for correlation: {e}")
            return base_position_size

    def _get_correlation(
        self,
        symbol1: str,
        symbol2: str,
        overrides: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Calculate Pearson correlation between two symbols."""
        overrides = overrides or {}
        try:
            lookback = int(self._resolve_setting(
                'correlation_lookback_period', overrides,
                attr_key='correlation_lookback_period',
                default=self.correlation_lookback_period,
            ) or self.correlation_lookback_period)

            data1 = self.mt5_client.copy_rates_from_pos(symbol1, 'M15', 0, lookback)
            if data1 is None:
                logger.warning(f"Cannot get data for {symbol1} correlation check")
                return None

            df1 = pd.DataFrame(data1)
            close1 = df1['close'].values

            data2 = self.mt5_client.copy_rates_from_pos(symbol2, 'M15', 0, lookback)
            if data2 is None:
                logger.warning(f"Cannot get data for {symbol2} correlation check")
                return None

            df2 = pd.DataFrame(data2)
            close2 = df2['close'].values

            N = min(len(close1), len(close2), lookback)
            if N >= 10:
                a = close1[-N:]
                b = close2[-N:]
                if np.std(a) > 0 and np.std(b) > 0:
                    corr_matrix = np.corrcoef(a, b)
                    return float(corr_matrix[0, 1])
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate current risk metrics."""
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
                        contract_size = 100000.0
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
    def validate_trade_parameters(self, symbol: str, volume: float, stop_loss: float, take_profit: float, order_type: int = None) -> bool:
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

            min_stop_distance = float(symbol_info.trade_stops_level) * float(symbol_info.point)

            if order_type is not None:
                if order_type == 0:  # BUY
                    current_price = float(tick.ask)
                    if not (stop_loss < current_price - min_stop_distance):
                        logger.error("Stop loss invalid for BUY (too close or above current price)")
                        return False
                    if not (take_profit > current_price + min_stop_distance):
                        logger.error("Take profit invalid for BUY (too close or below current price)")
                        return False
                else:  # SELL
                    current_price = float(tick.bid)
                    if not (stop_loss > current_price + min_stop_distance):
                        logger.error("Stop loss invalid for SELL (too close or below current price)")
                        return False
                    if not (take_profit < current_price - min_stop_distance):
                        logger.error("Take profit invalid for SELL (too close or above current price)")
                        return False
            else:
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
