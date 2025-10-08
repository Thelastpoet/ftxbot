"""
Risk Manager Module
Handles position sizing and risk management calculations
"""

from talib import CORREL
import numpy as np
import pandas as pd
import logging
from typing import  Any, Dict, Optional, Tuple

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
        self.correlation_threshold = self._get_config_value('correlation_threshold', 0.7)
        self.correlation_lookback_period = self._get_config_value('correlation_lookback_period', 30)

        # Correlation workspace cache to avoid duplicate MT5 fetches per decision
        self._correlation_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self._correlation_cache_anchor: Optional[str] = None
        
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
    
    def _series_returns(self, closes: np.ndarray) -> np.ndarray:
        # log-returns are standard; fall back to pct change if needed
        x = np.asarray(closes, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 3:
            return np.array([], dtype=float)
        rets = np.diff(np.log(x))
        return rets

    def _ewma_corr(self, r1: np.ndarray, r2: np.ndarray, lam: float = 0.94) -> Optional[float]:
        """
        RiskMetrics-style EWMA correlation on returns.
        lam is decay (daily classic=0.94). For M15 you can use 0.97–0.99.
        """
        n = min(r1.size, r2.size)
        if n < 10:
            return None
        r1 = r1[-n:]
        r2 = r2[-n:]

        # de-mean with EWMA means
        w = np.power(lam, np.arange(n-1, -1, -1, dtype=float))
        w /= w.sum()
        m1 = (w * r1).sum()
        m2 = (w * r2).sum()
        d1 = r1 - m1
        d2 = r2 - m2
        cov = float((w * d1 * d2).sum())
        v1 = float((w * d1 * d1).sum())
        v2 = float((w * d2 * d2).sum())
        if v1 <= 0 or v2 <= 0:
            return None
        return cov / (v1**0.5 * v2**0.5)
    
    def _start_correlation_window(self, anchor_symbol: str) -> None:
        """Reset correlation cache for the current symbol decision."""
        self._correlation_cache = {}
        self._correlation_cache_anchor = anchor_symbol

    def _correlation_key(self, symbol1: str, symbol2: str) -> Tuple[str, str]:
        return tuple(sorted((symbol1, symbol2)))

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

    def check_risk_limits(self, symbol: str, signal_type: int = None, overrides: Optional[Dict[str, float]] = None,) -> bool:
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
        """Prevent highly correlated duplicates; soft cases are handled via size reduction."""
        overrides = overrides or {}
        try:
            self._start_correlation_window(new_symbol)
            positions = self.mt5_client.get_all_positions()
            if not positions:
                return True

            hard = float(self._resolve_setting('correlation_threshold', overrides,
                                            attr_key='correlation_threshold', default=self.correlation_threshold))
            # Optional softer threshold (below 'hard') to inform sizing only
            soft = float(self._resolve_setting('correlation_soft_threshold', overrides, default=max(0.5, hard - 0.15)))

            for p in positions:
                if p.symbol == new_symbol:
                    continue

                corr = self._get_correlation(new_symbol, p.symbol, overrides)
                if corr is None:
                    continue

                same_dir = ((p.type == 0 and proposed_signal_type == 0) or
                            (p.type == 1 and proposed_signal_type == 1))

                # HARD block only for extreme same-direction correlation
                if same_dir and abs(corr) >= hard:
                    logger.debug("HARD CORR BLOCK: %s vs %s corr=%.2f >= %.2f",
                                new_symbol, p.symbol, corr, hard)
                    return False

                # Opposite-direction, strongly negative correlation can also be risky (pairs trade),
                # but we allow it and let sizing handle reduction.
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
        """Continuously reduce size as correlated overlaps grow; avoid full block unless 'hard' hit."""
        overrides = overrides or {}
        try:
            if self._correlation_cache_anchor != symbol:
                self._start_correlation_window(symbol)
            positions = self.mt5_client.get_all_positions()
            if not positions:
                return base_position_size

            hard = float(self._resolve_setting('correlation_threshold', overrides,
                                            attr_key='correlation_threshold', default=self.correlation_threshold))
            soft = float(self._resolve_setting('correlation_soft_threshold', overrides, default=max(0.5, hard - 0.15)))

            # k controls reduction speed; 0.0=off, 1.0=aggressive
            k = float(self._resolve_setting('correlation_size_slope', overrides, default=0.8))

            reduction = 1.0
            for p in positions:
                if p.symbol == symbol:
                    continue
                corr = self._get_correlation(symbol, p.symbol, overrides)
                if corr is None:
                    continue

                # Only reduce when correlation meaningfully high (beyond soft)
                x = abs(corr)
                if x >= soft:
                    # linear taper from soft -> hard: at hard, reduction could be near zero
                    span = max(1e-9, hard - soft)
                    frac = min(1.0, (x - soft) / span)  # 0..1
                    reduction *= max(0.0, 1.0 - k * frac)

            adjusted = base_position_size * reduction
            # keep exchange/broker volume rules
            if positions:
                sym_info = self.mt5_client.get_symbol_info(symbol)
                if sym_info:
                    step = sym_info.volume_step or 0.01
                    adjusted = round(adjusted / step) * step
                    adjusted = max(sym_info.volume_min, min(adjusted, sym_info.volume_max))

            logger.debug("Correlation size adjust: %.4f -> %.4f (soft=%.2f, hard=%.2f)",
                        base_position_size, adjusted, soft, hard)
            return float(adjusted)
        except Exception as e:
            logger.error(f"Error adjusting position size for correlation: {e}")
            return base_position_size

    def _get_correlation(self, symbol1: str, symbol2: str, overrides: Optional[Dict[str, float]] = None) -> Optional[float]:
        """Correlation between symbols using log-returns with EWMA weighting."""
        overrides = overrides or {}
        key = self._correlation_key(symbol1, symbol2)
        if key in self._correlation_cache:
            return self._correlation_cache[key]
        try:
            lookback = int(self._resolve_setting(
                'correlation_lookback_period', overrides,
                attr_key='correlation_lookback_period',
                default=self.correlation_lookback_period,
            ) or self.correlation_lookback_period)

            # Pull closes (same TF as before)
            d1 = self.mt5_client.copy_rates_from_pos(symbol1, 'M15', 0, lookback)
            d2 = self.mt5_client.copy_rates_from_pos(symbol2, 'M15', 0, lookback)
            if d1 is None or d2 is None:
                logger.warning(f"Cannot get data for {symbol1} or {symbol2} correlation check")
                self._correlation_cache[key] = None
                return None

            c1 = pd.DataFrame(d1)['close'].values
            c2 = pd.DataFrame(d2)['close'].values

            r1 = self._series_returns(c1)
            r2 = self._series_returns(c2)
            # EWMA decay: use slightly slower decay for intraday (e.g., 0.97–0.99)
            lam = float(self._resolve_setting('correlation_ewma_lambda', overrides, default=0.97))
            corr = self._ewma_corr(r1, r2, lam=lam)
            value = float(corr) if corr is not None else 0.0
            self._correlation_cache[key] = value
            return value
        
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            self._correlation_cache[key] = None
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