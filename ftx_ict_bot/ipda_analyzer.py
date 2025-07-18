"""
IPDA Analyzer - Integrated with LiquidityDetector
Clean architecture following single responsibility principle.
"""
import logging

logger = logging.getLogger(__name__)

class IPDAAnalyzer:
    """
    Implements IPDA (Interbank Price Delivery Algorithm) based on ICT's official teachings.
    Integrates with existing LiquidityDetector for comprehensive liquidity analysis.
    
    ICT Official Teachings (2016-2017 Mentorship):
    - "IPDA Data Ranges provide the Algorithm with a context to (60-40-20) Look Back and Cast-Forward(20-40-60)"
    - "Each new day we shift the range forward"
    - "IPDA uses PD Array within IPDA data ranges to determine whether the market is in Premium or Discount"
    - "When the market is Bearish we work from a Premium all the way down to a Discount"
    """
    
    def __init__(self, config, liquidity_detector):
        self.config = config
        self.liquidity_detector = liquidity_detector  # Use existing LiquidityDetector
        self.IPDA_PERIODS = [20, 40, 60]  # ICT's official IPDA periods
        
    def analyze_ipda_data_ranges(self, daily_df, symbol, ohlc_df=None, session_context=None):
        """
        Analyze IPDA data ranges and integrate with comprehensive liquidity analysis.
        
        Returns:
        - IPDA-specific analysis (ranges, bias)
        - Enhanced liquidity levels (existing + IPDA targets)
        - Integrated analysis for trading decisions
        """
        if daily_df is None or len(daily_df) < 60:
            logger.warning(f"{symbol}: Insufficient daily data for IPDA analysis (need 60+ days)")
            return {}
            
        current_date = daily_df.index[-1]
        current_price = daily_df['close'].iloc[-1]
        
        # 1. IPDA-specific analysis (ranges and bias)
        ipda_ranges = {}
        for period in self.IPDA_PERIODS:
            if len(daily_df) >= period:
                range_data = self._analyze_period_range(daily_df, period)
                ipda_ranges[f'{period}_day'] = range_data
        
        # 2. Determine IPDA bias using ICT's official methodology
        ipda_bias = self._determine_ipda_bias_official(ipda_ranges, current_price)
        
        # 3. Get comprehensive liquidity using existing LiquidityDetector
        if ohlc_df is not None and session_context is not None:
            # Enhance LiquidityDetector with IPDA targets
            comprehensive_liquidity = self._get_enhanced_liquidity_with_ipda(
                ohlc_df, session_context, daily_df, ipda_ranges, current_price
            )
        else:
            comprehensive_liquidity = {'buy_side': [], 'sell_side': []}
            logger.warning(f"{symbol}: No intraday data provided for comprehensive liquidity analysis")
        
        # 4. Create integrated analysis
        ipda_analysis = {
            'symbol': symbol,
            'current_date': current_date,
            'current_price': current_price,
            'ipda_ranges': ipda_ranges,
            'ipda_bias': ipda_bias,
            'comprehensive_liquidity': comprehensive_liquidity,
            'algorithmic_context': self._determine_algorithmic_context(ipda_ranges, ipda_bias),
            'integration_summary': self._create_integration_summary(ipda_bias, comprehensive_liquidity)
        }
        
        logger.info(f"{symbol}: IPDA Analysis Complete - Bias: {ipda_bias}")
        return ipda_analysis
    
    def _analyze_period_range(self, daily_df, period):
        """Analyze specific IPDA period range following ICT's exact methodology."""
        period_data = daily_df.tail(period)
        
        range_high = period_data['high'].max()
        range_low = period_data['low'].min()
        range_size = range_high - range_low
        current_price = daily_df['close'].iloc[-1]
        
        # ICT's Premium/Discount Analysis
        if range_size > 0:
            position_in_range = (current_price - range_low) / range_size
        else:
            position_in_range = 0.5
        
        # ICT's official Premium/Discount zones
        if position_in_range >= 0.7:
            zone = 'premium'
        elif position_in_range >= 0.5:
            zone = 'equilibrium_to_premium'
        elif position_in_range >= 0.3:
            zone = 'equilibrium_to_discount'
        else:
            zone = 'discount'
            
        return {
            'period': period,
            'range_high': range_high,
            'range_low': range_low,
            'range_size': range_size,
            'current_position': position_in_range,
            'premium_discount_zone': zone,
            'high_date': period_data['high'].idxmax(),
            'low_date': period_data['low'].idxmin(),
            'significance': self._get_period_significance(period)
        }
    
    def _determine_ipda_bias_official(self, ipda_ranges, current_price):
        """
        Determine IPDA bias using ICT's official methodology:
        "When the market is Bearish we work from a Premium all the way down to a Discount"
        "When the market is Bullish we work from a Discount all the way up to a Premium"
        """
        # Use 60-day range as primary context (most significant per ICT)
        if '60_day' in ipda_ranges:
            primary_range = ipda_ranges['60_day']
            zone = primary_range['premium_discount_zone']
            
            # ICT's official bias determination
            if zone == 'premium':
                return 'bearish'  # "work from Premium down to Discount"
            elif zone == 'discount':
                return 'bullish'  # "work from Discount up to Premium"
            elif zone == 'equilibrium_to_premium':
                return 'bearish_leaning'
            elif zone == 'equilibrium_to_discount':
                return 'bullish_leaning'
            else:
                return 'neutral'
        
        # Fallback to 40-day if 60-day not available
        elif '40_day' in ipda_ranges:
            range_data = ipda_ranges['40_day']
            zone = range_data['premium_discount_zone']
            
            if zone in ['premium', 'equilibrium_to_premium']:
                return 'bearish_leaning'
            elif zone in ['discount', 'equilibrium_to_discount']:
                return 'bullish_leaning'
            else:
                return 'neutral'
                
        return 'neutral'
    
    def _get_enhanced_liquidity_with_ipda(self, ohlc_df, session_context, daily_df, ipda_ranges, current_price):
        """
        Get comprehensive liquidity using existing LiquidityDetector + IPDA enhancements.
        """
        # 1. Get existing comprehensive liquidity
        base_liquidity = self.liquidity_detector.get_liquidity_levels(ohlc_df, session_context, daily_df)
        
        # 2. Add IPDA-specific liquidity targets
        ipda_liquidity = self._add_ipda_liquidity_targets(ipda_ranges, current_price)
        
        # 3. Merge and prioritize
        enhanced_liquidity = {
            'buy_side': base_liquidity.get('buy_side', []) + ipda_liquidity.get('buy_side', []),
            'sell_side': base_liquidity.get('sell_side', []) + ipda_liquidity.get('sell_side', [])
        }
        
        # 4. Sort by priority and remove duplicates
        enhanced_liquidity['buy_side'] = self._finalize_liquidity_list(enhanced_liquidity['buy_side'], current_price, 'above')
        enhanced_liquidity['sell_side'] = self._finalize_liquidity_list(enhanced_liquidity['sell_side'], current_price, 'below')
        
        return enhanced_liquidity
    
    def _add_ipda_liquidity_targets(self, ipda_ranges, current_price):
        """Add IPDA-specific liquidity targets to the comprehensive analysis."""
        ipda_liquidity = {'buy_side': [], 'sell_side': []}
        
        for period_name, range_data in ipda_ranges.items():
            period = range_data['period']
            priority = self._get_ipda_priority(period)
            
            # Buyside liquidity (above current price)
            if range_data['range_high'] > current_price:
                ipda_liquidity['buy_side'].append({
                    'level': range_data['range_high'],
                    'type': f'ipda_{period}_day_high',
                    'priority': priority,
                    'description': f'IPDA {period}-Day Range High',
                    'source': 'ipda_analysis'
                })
            
            # Sellside liquidity (below current price)
            if range_data['range_low'] < current_price:
                ipda_liquidity['sell_side'].append({
                    'level': range_data['range_low'],
                    'type': f'ipda_{period}_day_low',
                    'priority': priority,
                    'description': f'IPDA {period}-Day Range Low',
                    'source': 'ipda_analysis'
                })
        
        return ipda_liquidity
    
    def _get_ipda_priority(self, period):
        """Get priority for IPDA liquidity based on ICT hierarchy: 60>40>20."""
        if period == 60:
            return 'very_high'  # Most important per ICT
        elif period == 40:
            return 'high'
        elif period == 20:
            return 'medium'
        else:
            return 'low'
    
    def _get_period_significance(self, period):
        """Get significance level for IPDA period."""
        if period >= 60:
            return 'very_high'
        elif period >= 40:
            return 'high'
        else:
            return 'medium'
    
    def _finalize_liquidity_list(self, liquidity_list, current_price, side):
        """Clean up liquidity list by removing duplicates and sorting by priority."""
        if not liquidity_list:
            return []
        
        # Filter by price side
        if side == 'above':
            filtered = [l for l in liquidity_list if l['level'] > current_price]
        else:
            filtered = [l for l in liquidity_list if l['level'] < current_price]
        
        # Remove near-duplicate levels (within 0.1% of each other)
        unique_levels = []
        for level_data in filtered:
            is_duplicate = False
            for existing in unique_levels:
                if abs(level_data['level'] - existing['level']) / existing['level'] < 0.001:  # 0.1% threshold
                    # Keep the higher priority one
                    priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
                    if priority_order.get(level_data['priority'], 99) < priority_order.get(existing['priority'], 99):
                        unique_levels.remove(existing)
                        unique_levels.append(level_data)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append(level_data)
        
        # Sort by priority then by distance
        priority_order = {'very_high': 0, 'high': 1, 'medium': 2, 'low': 3}
        unique_levels.sort(key=lambda x: (
            priority_order.get(x['priority'], 99),
            abs(x['level'] - current_price)
        ))
        
        return unique_levels
    
    def _determine_algorithmic_context(self, ipda_ranges, ipda_bias):
        """Determine algorithmic context based on IPDA analysis."""
        if '60_day' in ipda_ranges:
            zone = ipda_ranges['60_day']['premium_discount_zone']
            
            context_map = {
                'premium': 'distribution_mode',
                'equilibrium_to_premium': 'distribution_pending',
                'equilibrium_to_discount': 'accumulation_pending',
                'discount': 'accumulation_mode'
            }
            
            return {
                'mode': context_map.get(zone, 'neutral'),
                'bias': ipda_bias,
                'description': f"IPDA indicates {context_map.get(zone, 'neutral')} with {ipda_bias} bias"
            }
        
        return {'mode': 'insufficient_data', 'bias': ipda_bias, 'description': 'Insufficient IPDA data'}
    
    def _create_integration_summary(self, ipda_bias, comprehensive_liquidity):
        """Create summary of integrated IPDA + liquidity analysis."""
        buyside_targets = len(comprehensive_liquidity.get('buy_side', []))
        sellside_targets = len(comprehensive_liquidity.get('sell_side', []))
        
        # Get highest priority targets
        best_buyside = comprehensive_liquidity.get('buy_side', [{}])[0] if buyside_targets > 0 else None
        best_sellside = comprehensive_liquidity.get('sell_side', [{}])[0] if sellside_targets > 0 else None
        
        return {
            'ipda_bias': ipda_bias,
            'total_buyside_targets': buyside_targets,
            'total_sellside_targets': sellside_targets,
            'primary_target': best_buyside if ipda_bias in ['bullish', 'bullish_leaning'] else best_sellside,
            'target_alignment': self._check_bias_target_alignment(ipda_bias, best_buyside, best_sellside),
            'recommendation': self._generate_recommendation(ipda_bias, best_buyside, best_sellside)
        }
    
    def _check_bias_target_alignment(self, bias, best_buyside, best_sellside):
        """Check if available targets align with IPDA bias."""
        if bias in ['bullish', 'bullish_leaning'] and best_buyside:
            return 'aligned'
        elif bias in ['bearish', 'bearish_leaning'] and best_sellside:
            return 'aligned'
        elif bias == 'neutral':
            return 'neutral'
        else:
            return 'misaligned'
    
    def _generate_recommendation(self, bias, best_buyside, best_sellside):
        """Generate trading recommendation based on integrated analysis."""
        if bias in ['bullish', 'bullish_leaning'] and best_buyside:
            return f"IPDA supports bullish bias targeting {best_buyside.get('description', 'buyside liquidity')}"
        elif bias in ['bearish', 'bearish_leaning'] and best_sellside:
            return f"IPDA supports bearish bias targeting {best_sellside.get('description', 'sellside liquidity')}"
        elif bias == 'neutral':
            return "IPDA bias neutral - wait for clearer directional context"
        else:
            return "IPDA bias and available targets misaligned - exercise caution"