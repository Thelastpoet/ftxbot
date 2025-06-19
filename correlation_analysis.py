"""
Correlation Analysis Module for Live Trading
Production-ready correlation tracking and risk management
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
import MetaTrader5 as mt5
from collections import defaultdict
from scipy import stats


class CorrelationManager:
    """
    Manages correlation analysis for multi-pair trading with focus on risk management.
    Designed to integrate seamlessly with existing trading bot architecture.
    """
    
    def __init__(self, symbols, market_context, lookback_periods=[20, 50, 100]):
        """
        Initialize correlation manager
        
        Args:
            symbols: List of trading symbols
            market_context: MarketContext instance for session info
            lookback_periods: Periods for correlation calculation
        """
        self.symbols = symbols
        self.market_context = market_context
        self.lookback_periods = sorted(lookback_periods)  # Ensure ascending order
        
        # Primary data storage
        self.correlation_matrices = {}  # {timeframe: {period: correlation_matrix}}
        self.correlation_history = defaultdict(list)  # Historical correlations for stability
        
        # Cache for performance
        self.returns_cache = {}  # {symbol_timeframe: returns_series}
        self.last_calculation_time = 0
        self.calculation_interval = 900  # Recalculate every 15 minutes
        
        # Trading-specific thresholds
        self.high_correlation_threshold = 0.75
        self.significant_correlation_threshold = 0.60
        self.max_correlation_exposure = 2.5  # Max equivalent positions
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def update(self, market_data):
        """
        Update correlation calculations with new market data
        
        Args:
            market_data: Dict of {symbol: {timeframe: DataFrame}}
        """
        current_time = datetime.now(timezone.utc)
        
        # Check if update needed
        if (current_time.timestamp() - self.last_calculation_time) < self.calculation_interval:
            return False
            
        try:
            # Calculate returns for all symbols and timeframes
            self._update_returns_cache(market_data)
            
            # Calculate correlation matrices for each timeframe
            for timeframe in [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]:
                self._calculate_timeframe_correlations(timeframe)
                
            # Update correlation history for stability analysis
            self._update_correlation_history()
            
            self.last_calculation_time = current_time.timestamp()
            self.logger.info(f"Correlation update completed at {current_time}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating correlations: {e}")
            return False
    
    def _update_returns_cache(self, market_data):
        """Calculate and cache returns for all symbols"""
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
                
            for timeframe, df in market_data[symbol].items():
                if df is None or len(df) < 2:
                    continue
                    
                # Calculate log returns
                returns = np.log(df['close'] / df['close'].shift(1)).dropna()
                
                # Store in cache with proper key
                cache_key = f"{symbol}_{timeframe}"
                self.returns_cache[cache_key] = returns
                
    def _calculate_timeframe_correlations(self, timeframe):
        """Calculate correlation matrices for a specific timeframe"""
        if timeframe not in self.correlation_matrices:
            self.correlation_matrices[timeframe] = {}
            
        for period in self.lookback_periods:
            # Gather returns for all symbols
            returns_dict = {}
            
            for symbol in self.symbols:
                cache_key = f"{symbol}_{timeframe}"
                if cache_key in self.returns_cache:
                    returns = self.returns_cache[cache_key]
                    if len(returns) >= period:
                        returns_dict[symbol] = returns.iloc[-period:]
            
            # Need at least 2 symbols to calculate correlation
            if len(returns_dict) >= 2:
                # Create DataFrame and calculate correlation
                returns_df = pd.DataFrame(returns_dict)
                corr_matrix = returns_df.corr(method='pearson')
                
                # Store the correlation matrix
                self.correlation_matrices[timeframe][period] = corr_matrix
                
    def _update_correlation_history(self):
        """Track correlation history for stability analysis"""
        current_time = datetime.now(timezone.utc).timestamp()
        
        # Focus on H1 timeframe with 50-period for stability tracking
        timeframe = mt5.TIMEFRAME_H1
        period = 50
        
        if (timeframe in self.correlation_matrices and 
            period in self.correlation_matrices[timeframe]):
            
            corr_matrix = self.correlation_matrices[timeframe][period]
            
            # Track each pair's correlation
            for i in range(len(corr_matrix.index)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    symbol1 = corr_matrix.index[i]
                    symbol2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if not np.isnan(correlation):
                        key = self._get_pair_key(symbol1, symbol2)
                        
                        # Add to history
                        self.correlation_history[key].append({
                            'time': current_time,
                            'correlation': correlation
                        })
                        
                        # Keep only last 48 data points (12 hours at 15-min updates)
                        if len(self.correlation_history[key]) > 48:
                            self.correlation_history[key].pop(0)
    
    def get_correlation(self, symbol1, symbol2, timeframe=mt5.TIMEFRAME_H1, period=50):
        """
        Get current correlation between two symbols
        
        Returns:
            float: Correlation coefficient (-1 to 1) or None if not available
        """
        if (timeframe in self.correlation_matrices and 
            period in self.correlation_matrices[timeframe]):
            
            corr_matrix = self.correlation_matrices[timeframe][period]
            
            if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
                return corr_matrix.loc[symbol1, symbol2]
            elif symbol2 in corr_matrix.index and symbol1 in corr_matrix.columns:
                return corr_matrix.loc[symbol2, symbol1]
                
        return None
    
    def get_correlation_stability(self, symbol1, symbol2):
        """
        Calculate how stable the correlation is between two symbols
        
        Returns:
            dict: {'stability': float (0-1), 'avg_correlation': float, 'std': float}
        """
        key = self._get_pair_key(symbol1, symbol2)
        
        if key not in self.correlation_history or len(self.correlation_history[key]) < 10:
            return {'stability': 0.5, 'avg_correlation': 0, 'std': 1.0}  # Default neutral
            
        correlations = [h['correlation'] for h in self.correlation_history[key]]
        avg_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        
        # Stability score: lower std = more stable
        # Normalize to 0-1 range where 1 is most stable
        stability = max(0, 1 - (std_corr / 0.3))  # 0.3 std = 0 stability
        
        return {
            'stability': stability,
            'avg_correlation': avg_corr,
            'std': std_corr
        }
    
    def assess_new_trade_correlation_risk(self, symbol, direction, open_positions):
        """
        Assess correlation risk of taking a new trade
        
        Args:
            symbol: Symbol to trade
            direction: 'buy' or 'sell'
            open_positions: List of {'symbol': str, 'direction': str, 'lots': float}
            
        Returns:
            dict: Risk assessment with actionable recommendations
        """
        assessment = {
            'risk_level': 'low',  # low, medium, high, extreme
            'effective_exposure': 1.0,  # How many equivalent positions this would create
            'recommended_size_adjustment': 1.0,  # Multiplier for position size
            'correlated_positions': [],
            'warnings': [],
            'proceed': True
        }
        
        if not open_positions:
            return assessment
            
        # Get current session for context
        session_info = self.market_context.get_trading_session()
        session_volatility = session_info.get('volatility_multiplier', 1.0)
        
        # Check correlation with each open position
        total_effective_exposure = 1.0  # Start with the new position
        highest_correlation = 0
        
        for position in open_positions:
            # Get correlation on multiple timeframes for robustness
            correlations = []
            for tf in [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]:
                corr = self.get_correlation(symbol, position['symbol'], tf, period=50)
                if corr is not None:
                    correlations.append(corr)
                    
            if not correlations:
                continue
                
            avg_correlation = np.mean(correlations)
            
            # Determine if this increases risk
            # Same direction + positive correlation = increased risk
            # Opposite direction + negative correlation = increased risk
            if direction == position['direction']:
                effective_correlation = avg_correlation
            else:
                effective_correlation = -avg_correlation
                
            # Calculate position weight (could be enhanced with actual dollar exposure)
            position_weight = position.get('lots', 1.0)
            
            if effective_correlation > self.significant_correlation_threshold:
                # This position adds to our risk
                correlation_impact = effective_correlation * position_weight
                total_effective_exposure += correlation_impact
                
                assessment['correlated_positions'].append({
                    'symbol': position['symbol'],
                    'correlation': round(avg_correlation, 3),
                    'impact': round(correlation_impact, 2)
                })
                
                if effective_correlation > self.high_correlation_threshold:
                    assessment['warnings'].append(
                        f"High correlation ({avg_correlation:.2f}) with {position['symbol']}"
                    )
                    
            highest_correlation = max(highest_correlation, abs(effective_correlation))
        
        # Determine risk level and recommendations
        assessment['effective_exposure'] = round(total_effective_exposure, 2)
        
        if total_effective_exposure > 3.0:
            assessment['risk_level'] = 'extreme'
            assessment['proceed'] = False
            assessment['warnings'].append("Would exceed maximum correlation exposure")
            
        elif total_effective_exposure > self.max_correlation_exposure:
            assessment['risk_level'] = 'high'
            assessment['recommended_size_adjustment'] = 0.5
            assessment['warnings'].append("High correlation exposure - recommend reduced size")
            
        elif total_effective_exposure > 2.0:
            assessment['risk_level'] = 'medium'
            assessment['recommended_size_adjustment'] = 0.75
            
        # Adjust for market session
        if session_volatility > 1.2 and highest_correlation > 0.7:
            assessment['recommended_size_adjustment'] *= 0.8
            assessment['warnings'].append("High volatility session - additional size reduction")
            
        return assessment
    
    def get_correlation_regime(self):
        """
        Identify current market correlation regime
        
        Returns:
            dict: Current regime and trading implications
        """
        # Use H1 timeframe for regime detection
        if (mt5.TIMEFRAME_H1 not in self.correlation_matrices or 
            50 not in self.correlation_matrices[mt5.TIMEFRAME_H1]):
            return {
                'regime': 'unknown',
                'avg_correlation': 0,
                'implications': 'Insufficient data for regime detection'
            }
            
        corr_matrix = self.correlation_matrices[mt5.TIMEFRAME_H1][50]
        
        # Calculate average absolute correlation
        correlations = []
        high_corr_pairs = 0
        total_pairs = 0
        
        for i in range(len(corr_matrix.index)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    total_pairs += 1
                    if abs(corr) > self.high_correlation_threshold:
                        high_corr_pairs += 1
                        
        if not correlations:
            return {
                'regime': 'unknown',
                'avg_correlation': 0,
                'implications': 'No valid correlations found'
            }
            
        avg_correlation = np.mean(correlations)
        high_corr_ratio = high_corr_pairs / total_pairs if total_pairs > 0 else 0
        
        # Classify regime
        if avg_correlation > 0.65 or high_corr_ratio > 0.4:
            regime = 'high_correlation'
            implications = 'Risk-on/off market - reduce position sizes, avoid correlated trades'
        elif avg_correlation < 0.35:
            regime = 'low_correlation'
            implications = 'Divergent market - normal position sizing, seek best setups'
        else:
            regime = 'normal_correlation'
            implications = 'Mixed market - standard risk management applies'
            
        return {
            'regime': regime,
            'avg_correlation': round(avg_correlation, 3),
            'high_correlation_ratio': round(high_corr_ratio, 3),
            'implications': implications,
            'timestamp': datetime.now(timezone.utc)
        }
    
    def find_divergence_opportunities(self, min_historical_correlation=0.7, min_divergence=0.3):
        """
        Find pairs that normally correlate but are currently diverging
        
        Returns:
            List of potential mean reversion opportunities
        """
        opportunities = []
        
        for pair_key, history in self.correlation_history.items():
            if len(history) < 20:
                continue
                
            # Get historical and current correlation
            historical_corr = np.mean([h['correlation'] for h in history[:-5]])
            current_corr = history[-1]['correlation']
            
            # Check for divergence from normally correlated pairs
            if (abs(historical_corr) > min_historical_correlation and 
                abs(current_corr) < abs(historical_corr) - min_divergence):
                
                symbols = pair_key.split('_')
                stability_info = self.get_correlation_stability(symbols[0], symbols[1])
                
                opportunities.append({
                    'pair': pair_key,
                    'symbol1': symbols[0],
                    'symbol2': symbols[1],
                    'historical_correlation': round(historical_corr, 3),
                    'current_correlation': round(current_corr, 3),
                    'divergence': round(abs(historical_corr) - abs(current_corr), 3),
                    'stability': round(stability_info['stability'], 2)
                })
                
        # Sort by divergence magnitude
        opportunities.sort(key=lambda x: x['divergence'], reverse=True)
        
        return opportunities
    
    def _get_pair_key(self, symbol1, symbol2):
        """Create consistent key for symbol pairs"""
        return '_'.join(sorted([symbol1, symbol2]))
    
    def get_risk_adjusted_position_size(self, base_risk_percent, symbol, direction, open_positions):
        """
        Calculate position size adjusted for correlation risk
        
        Args:
            base_risk_percent: Base risk percentage (e.g., 0.01 for 1%)
            symbol: Symbol to trade
            direction: Trade direction
            open_positions: Current open positions
            
        Returns:
            tuple: (adjusted_risk_percent, adjustment_details)
        """
        # Assess correlation risk
        risk_assessment = self.assess_new_trade_correlation_risk(
            symbol, direction, open_positions
        )
        
        # Get current correlation regime
        regime = self.get_correlation_regime()
        
        # Start with base risk
        adjusted_risk = base_risk_percent
        
        # Apply correlation-based adjustment
        adjusted_risk *= risk_assessment['recommended_size_adjustment']
        
        # Apply regime-based adjustment
        if regime['regime'] == 'high_correlation':
            adjusted_risk *= 0.8  # Additional 20% reduction in high correlation regime
        elif regime['regime'] == 'low_correlation':
            adjusted_risk *= 1.1  # Slight increase when correlations are low
            
        # Ensure we don't exceed maximum risk
        max_risk = base_risk_percent * 1.5  # Never more than 150% of base
        adjusted_risk = min(adjusted_risk, max_risk)
        
        details = {
            'base_risk': base_risk_percent,
            'adjusted_risk': round(adjusted_risk, 4),
            'correlation_adjustment': risk_assessment['recommended_size_adjustment'],
            'regime': regime['regime'],
            'effective_exposure': risk_assessment['effective_exposure'],
            'warnings': risk_assessment['warnings']
        }
        
        return adjusted_risk, details