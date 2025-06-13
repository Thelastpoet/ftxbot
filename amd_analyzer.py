import logging
from datetime import datetime, timezone

class AMDAnalyzer:
    
    def __init__(self, market_data):
        self.market_data = market_data
        self.pip_size = market_data.get_pip_size()
        self.asian_range_cache = {}
        
    def get_session_range(self, df, session_name):
        today = datetime.now(timezone.utc).date()
        
        if session_name == 'ASIAN_SESSION':
            start = datetime.combine(today, datetime.min.time()).replace(hour=0, tzinfo=timezone.utc)
            end = datetime.combine(today, datetime.min.time()).replace(hour=7, tzinfo=timezone.utc)
        elif session_name in ['LONDON_OPEN', 'LONDON_MAIN']:
            start = datetime.combine(today, datetime.min.time()).replace(hour=7, tzinfo=timezone.utc)
            end = datetime.combine(today, datetime.min.time()).replace(hour=12, tzinfo=timezone.utc)
        else:
            return None
            
        session_data = df[(df.index >= start) & (df.index < end)]
        
        if session_data.empty:
            return None
            
        return {
            'high': session_data['high'].max(),
            'low': session_data['low'].min(),
            'range': session_data['high'].max() - session_data['low'].min()
        }
    
    def identify_amd_context(self, df, current_session, market_structure):
        now = datetime.now(timezone.utc)
        today = now.date()
        
        asian_range = None
        if today in self.asian_range_cache:
            asian_range = self.asian_range_cache[today]
        elif now.hour >= 7:
            asian_range = self.get_session_range(df, 'ASIAN_SESSION')
            if asian_range:
                self.asian_range_cache[today] = asian_range
        
        amd_context = {
            'phase': None,
            'asian_range': asian_range,
            'setup_bias': None,
            'key_levels': []
        }
        
        if current_session['name'] == 'ASIAN_SESSION':
            if market_structure['regime'] == 'RANGING':
                amd_context['phase'] = 'ACCUMULATION'
                amd_context['setup_bias'] = 'range_trades_only'
                
        elif current_session['name'] in ['LONDON_OPEN', 'LONDON_MAIN']:
            if asian_range:
                current_price = df['close'].iloc[-1]
                
                if current_price > asian_range['high'] or current_price < asian_range['low']:
                    recent_bars = df.iloc[-10:]
                    
                    if current_price > asian_range['high']:
                        reversals = recent_bars[recent_bars['close'] < asian_range['high']]
                        if not reversals.empty:
                            amd_context['phase'] = 'MANIPULATION_CONFIRMED'
                            amd_context['setup_bias'] = 'bearish'
                            amd_context['key_levels'].append(asian_range['low'])
                    else:
                        reversals = recent_bars[recent_bars['close'] > asian_range['low']]
                        if not reversals.empty:
                            amd_context['phase'] = 'MANIPULATION_CONFIRMED'
                            amd_context['setup_bias'] = 'bullish'
                            amd_context['key_levels'].append(asian_range['high'])
                else:
                    amd_context['phase'] = 'AWAITING_MANIPULATION'
                    
        elif current_session['name'] in ['NY_MAIN', 'LONDON_NY_OVERLAP']:
            if asian_range and amd_context.get('phase') == 'MANIPULATION_CONFIRMED':
                amd_context['phase'] = 'DISTRIBUTION'
            elif market_structure['regime'] in ['STRONG_TREND', 'NORMAL_TREND']:
                amd_context['phase'] = 'DISTRIBUTION'
                amd_context['setup_bias'] = market_structure['trend']['direction']
                
        return amd_context


def with_amd_analysis(original_method):
    def wrapper(self, data):
        structure = original_method(self, data)
        
        if not structure:
            return None
            
        session_info = self.market_context.get_trading_session()
        
        amd_analyzer = AMDAnalyzer(self.market_data)
        amd_context = amd_analyzer.identify_amd_context(data, session_info, structure)
        
        structure['session'] = session_info
        structure['amd'] = amd_context
        
        if amd_context['phase'] == 'ACCUMULATION':
            structure['trading_approach'] = 'range_bound'
            structure['avoid_setups'] = ['momentum', 'breakout']
            
        elif amd_context['phase'] == 'MANIPULATION_CONFIRMED':
            structure['trading_approach'] = 'reversal'
            structure['preferred_direction'] = amd_context['setup_bias']
            
        elif amd_context['phase'] == 'DISTRIBUTION':
            structure['trading_approach'] = 'trend_following'
            if amd_context['setup_bias']:
                structure['preferred_direction'] = amd_context['setup_bias']
                
        return structure
        
    return wrapper


def with_amd_scoring(original_method):
    def wrapper(self, setups, market_structure):
        scored_setups = original_method(self, setups, market_structure)
        
        if not scored_setups or 'amd' not in market_structure:
            return scored_setups
            
        amd = market_structure['amd']
        
        for setup in scored_setups:
            amd_adjustment = 0
            
            if amd['phase'] == 'ACCUMULATION':
                if setup['type'] in ['range_extreme', 'fibonacci']:
                    amd_adjustment += 0.15
                elif setup['type'] == 'momentum':
                    amd_adjustment -= 0.25
                    
            elif amd['phase'] == 'MANIPULATION_CONFIRMED':
                if amd['setup_bias'] and setup['direction'] == amd['setup_bias']:
                    amd_adjustment += 0.35
                    if amd['key_levels'] and 'zone' in setup:
                        for level in amd['key_levels']:
                            if abs(setup['zone'].center - level) < market_structure['atr']:
                                amd_adjustment += 0.15
                                break
                else:
                    amd_adjustment -= 0.40
                    
            elif amd['phase'] == 'DISTRIBUTION':
                if setup['type'] in ['momentum', 'ma_bounce']:
                    amd_adjustment += 0.20
                if amd['setup_bias'] and setup['direction'] == amd['setup_bias']:
                    amd_adjustment += 0.25
                elif amd['setup_bias'] and setup['direction'] != amd['setup_bias']:
                    amd_adjustment -= 0.35
                    
            setup['amd_phase'] = amd['phase']
            setup['amd_adjustment'] = amd_adjustment
            setup['score'] = max(0, min(1.0, setup['score'] + amd_adjustment))
            
        scored_setups.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_setups
        
    return wrapper


def with_amd_logging(original_method):
    def wrapper(self, setup, market_structure, result):
        original_method(self, setup, market_structure, result)
        
        if 'amd' in market_structure:
            amd = market_structure['amd']
            logging.info(f"  - AMD Phase: {amd['phase']}")
            if amd['setup_bias']:
                logging.info(f"  - AMD Bias: {amd['setup_bias']}")
            if amd['asian_range']:
                logging.info(f"  - Asian Range: {amd['asian_range']['low']:.5f} - {amd['asian_range']['high']:.5f}")
                
    return wrapper