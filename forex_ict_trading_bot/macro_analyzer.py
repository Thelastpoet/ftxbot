"""
ICT Macro Analyzer - Corrected Implementation Based on ICT's Official Teachings
Follows Michael Huddleston's exact macro times from official content.
"""
from datetime import datetime, time, timezone
import logging

logger = logging.getLogger(__name__)

class MacroAnalyzer:
    """
    Analyzes ICT Macros based on official ICT teachings.
    
    ICT Official Teachings:
    - "ICT Macro Times refer to short time intervals during which the algorithm seeks liquidity"
    - "Macros typically occur in 20-minute intervals"
    - "These events happen during the London session, New York's morning, noon, and evening sessions"
    - NO Asian session macros mentioned in official content
    """
    
    def __init__(self, config):
        self.config = config
        
        # ICT's Official Macro Times (from official mentorship content)
        # NO ASIAN MACROS - not in ICT's official teachings
        self.OFFICIAL_MACRO_TIMES = {
            'london_macro_1': {'start': time(2, 33), 'end': time(3, 0)},
            'london_macro_2': {'start': time(4, 3), 'end': time(4, 30)},
            'ny_am_macro_1': {'start': time(8, 50), 'end': time(9, 10)},
            'ny_am_macro_2': {'start': time(9, 50), 'end': time(10, 10)},  # Most important per ICT
            'ny_macro_3': {'start': time(10, 50), 'end': time(11, 10)},
            'ny_lunch_macro': {'start': time(11, 50), 'end': time(12, 10)},
            'ny_pm_macro': {'start': time(13, 10), 'end': time(13, 40)},
        }
        
    def get_current_macro_status(self):
        """
        Determine if we're currently in an ICT macro period.
        Based on ICT's official macro times and teachings.
        """
        current_utc = datetime.now(timezone.utc)
        current_ny = current_utc.astimezone(self.config.NY_TIMEZONE)
        current_time = current_ny.time()
        
        # Check official named macro times first
        for macro_name, times in self.OFFICIAL_MACRO_TIMES.items():
            if times['start'] <= current_time <= times['end']:
                return {
                    'in_macro': True,
                    'macro_name': macro_name,
                    'macro_type': self._get_macro_classification(macro_name),
                    'time_remaining': self._calculate_time_remaining(current_time, times['end']),
                    'significance': self._get_macro_significance(macro_name)
                }
        
        # ICT 2024 teaching: "Every hour: last 10 minutes + first 10 minutes"
        # This is additional to the named macros above
        current_minute = current_time.minute
        if (50 <= current_minute <= 59) or (0 <= current_minute <= 10):
            # Only during London and NY sessions (no Asian per ICT)
            current_hour = current_time.hour
            if self._is_active_session_hour(current_hour):
                return {
                    'in_macro': True,
                    'macro_name': 'hourly_macro',
                    'macro_type': 'liquidity_seeking',
                    'time_remaining': self._calculate_hourly_macro_remaining(current_minute),
                    'significance': 'medium'
                }
        
        return {
            'in_macro': False, 
            'next_macro': self._get_next_macro(current_time)
        }
    
    def _is_active_session_hour(self, hour):
        """
        Check if current hour is during active sessions (London/NY only).
        NO Asian session macros per ICT's official content.
        """
        # London session: 2-5 AM NY time
        # NY session: 8-16 PM NY time  
        return (2 <= hour <= 5) or (8 <= hour <= 16)
    
    def _get_macro_classification(self, macro_name):
        """Classify macro type based on ICT's official teachings."""
        classification_map = {
            'london_macro_1': 'liquidity_hunt',
            'london_macro_2': 'range_expansion', 
            'ny_am_macro_1': 'market_opening',
            'ny_am_macro_2': 'primary_expansion',  # Most important
            'ny_macro_3': 'continuation',
            'ny_lunch_macro': 'rebalance',
            'ny_pm_macro': 'session_close'
        }
        return classification_map.get(macro_name, 'liquidity_seeking')
    
    def _get_macro_significance(self, macro_name):
        """Determine macro significance based on ICT's teachings."""
        # NY AM Macro 2 (09:50-10:10) is most significant per ICT
        if macro_name == 'ny_am_macro_2':
            return 'very_high'
        elif macro_name in ['london_macro_1', 'ny_am_macro_1']:
            return 'high'
        else:
            return 'medium'
    
    def _calculate_time_remaining(self, current_time, end_time):
        """Calculate time remaining in current macro."""
        current_total_minutes = current_time.hour * 60 + current_time.minute
        end_total_minutes = end_time.hour * 60 + end_time.minute
        
        if end_total_minutes > current_total_minutes:
            return end_total_minutes - current_total_minutes
        else:
            return 0
    
    def _calculate_hourly_macro_remaining(self, current_minute):
        """Calculate remaining time for hourly macro."""
        if 50 <= current_minute <= 59:
            return 60 - current_minute  # Until top of hour
        elif 0 <= current_minute <= 10:
            return 10 - current_minute  # Until end of first 10 minutes
        return 0
    
    def _get_next_macro(self, current_time):
        """Find the next upcoming macro."""
        current_total_minutes = current_time.hour * 60 + current_time.minute
        
        next_macros = []
        for macro_name, times in self.OFFICIAL_MACRO_TIMES.items():
            start_total_minutes = times['start'].hour * 60 + times['start'].minute
            
            if start_total_minutes > current_total_minutes:
                minutes_until = start_total_minutes - current_total_minutes
                next_macros.append({
                    'name': macro_name,
                    'start_time': times['start'],
                    'minutes_until': minutes_until
                })
        
        if next_macros:
            next_macros.sort(key=lambda x: x['minutes_until'])
            return next_macros[0]
        
        # If no more macros today, return first macro of next day
        return {
            'name': 'london_macro_1',
            'start_time': self.OFFICIAL_MACRO_TIMES['london_macro_1']['start'],
            'minutes_until': 'next_day'
        }
    
    def analyze_macro_setup(self, ohlc_df, symbol):
        """
        Analyze price behavior during macro periods based on ICT methodology.
        """
        macro_status = self.get_current_macro_status()
        
        if not macro_status['in_macro']:
            return None
            
        current_price = ohlc_df['close'].iloc[-1]
        macro_name = macro_status['macro_name']
        macro_type = macro_status['macro_type']
        
        analysis = {
            'symbol': symbol,
            'macro_name': macro_name,
            'macro_type': macro_type,
            'current_price': current_price,
            'significance': macro_status.get('significance', 'medium'),
            'expected_behavior': self._get_expected_behavior(macro_type),
            'liquidity_target': None
        }
        
        # Analyze based on macro type
        if macro_type == 'liquidity_hunt':
            analysis.update(self._analyze_liquidity_hunt_setup(ohlc_df))
        elif macro_type == 'primary_expansion':
            analysis.update(self._analyze_primary_expansion_setup(ohlc_df))
        elif macro_type == 'rebalance':
            analysis.update(self._analyze_rebalance_setup(ohlc_df))
            
        return analysis
    
    def _get_expected_behavior(self, macro_type):
        """Get expected price behavior during macro type."""
        behavior_map = {
            'liquidity_hunt': 'Seek and sweep liquidity levels',
            'primary_expansion': 'Strong directional movement with high volume',
            'market_opening': 'Initial volatility and direction setting',
            'continuation': 'Extend existing moves or reverse',
            'rebalance': 'Fill imbalances and seek equilibrium',
            'session_close': 'Position adjustments and volatility',
            'liquidity_seeking': 'Target nearby liquidity pools'
        }
        return behavior_map.get(macro_type, 'General liquidity seeking')
    
    def _analyze_liquidity_hunt_setup(self, ohlc_df):
        """Analyze setup during liquidity hunting macros."""
        recent_data = ohlc_df.tail(20)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        current_price = ohlc_df['close'].iloc[-1]
        
        # Determine likely liquidity target
        distance_to_high = recent_high - current_price
        distance_to_low = current_price - recent_low
        
        if distance_to_high < distance_to_low:
            return {
                'liquidity_target': recent_high,
                'expected_direction': 'bullish_sweep',
                'target_type': 'buy_side_liquidity',
                'setup_quality': 'high' if distance_to_high < recent_data['close'].std() else 'medium'
            }
        else:
            return {
                'liquidity_target': recent_low,
                'expected_direction': 'bearish_sweep', 
                'target_type': 'sell_side_liquidity',
                'setup_quality': 'high' if distance_to_low < recent_data['close'].std() else 'medium'
            }
    
    def _analyze_primary_expansion_setup(self, ohlc_df):
        """Analyze setup during primary expansion macros (most important)."""
        # This is the 09:50-10:10 NY AM macro - highest significance
        recent_data = ohlc_df.tail(10)
        
        return {
            'liquidity_target': None,  # Determined by broader market context
            'expected_direction': 'high_volatility_breakout',
            'target_type': 'institutional_order_flow',
            'setup_quality': 'very_high',  # This is ICT's most important macro
            'volume_expectation': 'high',
            'volatility_expectation': 'very_high'
        }
    
    def _analyze_rebalance_setup(self, ohlc_df):
        """Analyze setup during rebalancing macros."""
        return {
            'liquidity_target': None,
            'expected_direction': 'range_bound_rebalancing',
            'target_type': 'fair_value_gaps',
            'setup_quality': 'medium',
            'behavior': 'Fill imbalances and seek equilibrium levels'
        }