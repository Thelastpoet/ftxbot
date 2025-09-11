#!/usr/bin/env python3
"""
Backtest for the Forex Trading Bot
Uses the EXACT live system components
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import MetaTrader5 as mt5
import asyncio

# Import EXACT components from live system
from main import Config, TradingBot
from strategy import PurePriceActionStrategy
from market_data import MarketData
from mt5_client import MetaTrader5Client
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtest:
    """Backtest using exact live components"""
    
    def __init__(self, config_path: str = 'config.json'):
        # Use existing Config class from main.py
        self.config = Config(config_path)
        
        # Use existing components exactly as live bot does
        self.mt5_client = MetaTrader5Client()
        self.market_data = MarketData(self.mt5_client, self.config)
        self.strategy = PurePriceActionStrategy(self.config)
        self.risk_manager = RiskManager(self.config, self.mt5_client)
        
        # Backtest tracking
        self.trades = []
        self.open_positions = {}
        self.balance = 10000
        self.initial_balance = 10000
    
    async def run(self, symbol: str, start_date: datetime, end_date: datetime):
        """Run backtest using exact live logic"""
        
        logger.info(f"Backtesting {symbol} from {start_date} to {end_date}")
        
        # Get historical data using MarketData's existing methods
        symbol_config = next((s for s in self.config.symbols if s['name'] == symbol), None)
        if not symbol_config:
            return None
        
        # Fetch data for all timeframes
        all_data = {}
        for tf in symbol_config['timeframes']:
            # Use MT5 to get historical data
            tf_map = {'M15': mt5.TIMEFRAME_M15, 'H1': mt5.TIMEFRAME_H1}
            rates = mt5.copy_rates_range(symbol, tf_map.get(tf), start_date, end_date)
            
            if rates is not None:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                all_data[tf] = df
        
        if not all_data:
            return None
        
        # Walk through time
        primary_data = all_data.get('M15', all_data.get('H1'))
        
        for i in range(self.config.lookback_period, len(primary_data)):
            current_bar = primary_data.iloc[i]
            
            # Check open positions for exit
            self._check_exits(symbol, current_bar)
            
            # Generate signal if no position
            if symbol not in self.open_positions:
                # Prepare data window exactly as live system
                window_data = primary_data.iloc[:i+1].tail(self.config.max_period)
                
                # Get trend from H1 if available
                trend = 'ranging'
                if 'H1' in all_data:
                    h1_window = all_data['H1'][all_data['H1'].index <= current_bar.name].tail(20)
                    if len(h1_window) >= 20:
                        trend = self.market_data.identify_trend(h1_window)
                
                # Use EXACT strategy.generate_signal()
                signal = self.strategy.generate_signal(window_data, symbol, trend)
                
                if signal:
                    # Use EXACT risk_manager.calculate_position_size()
                    position_size = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        stop_loss_pips=signal.stop_loss_pips
                    )
                    
                    if position_size > 0:
                        self._open_position(symbol, signal, position_size, current_bar.name)
        
        # Close remaining positions
        for sym in list(self.open_positions.keys()):
            self._close_position(sym, primary_data.iloc[-1]['close'], 'END')
        
        # Save results
        return self._save_results()
    
    def _check_exits(self, symbol: str, bar: pd.Series):
        """Check if position should exit"""
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        
        if pos['type'] == 'BUY':
            if bar['low'] <= pos['sl']:
                self._close_position(symbol, pos['sl'], 'SL')
            elif bar['high'] >= pos['tp']:
                self._close_position(symbol, pos['tp'], 'TP')
        else:  # SELL
            if bar['high'] >= pos['sl']:
                self._close_position(symbol, pos['sl'], 'SL')
            elif bar['low'] <= pos['tp']:
                self._close_position(symbol, pos['tp'], 'TP')
    
    def _open_position(self, symbol: str, signal, size: float, time):
        """Open position"""
        self.open_positions[symbol] = {
            'type': 'BUY' if signal.type == 0 else 'SELL',
            'entry': signal.entry_price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'size': size,
            'time': time,
            'confidence': signal.confidence
        }
        
        logger.info(f"OPEN: {'BUY' if signal.type == 0 else 'SELL'} {symbol} @ {signal.entry_price:.5f}")
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close position and record trade"""
        if symbol not in self.open_positions:
            return
        
        pos = self.open_positions[symbol]
        pip_size = 0.0001 if 'JPY' not in symbol else 0.01
        
        if pos['type'] == 'BUY':
            pips = (exit_price - pos['entry']) / pip_size
        else:
            pips = (pos['entry'] - exit_price) / pip_size
        
        profit = pips * pos['size'] * 10
        self.balance += profit
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'type': pos['type'],
            'entry': pos['entry'],
            'exit': exit_price,
            'pips': pips,
            'profit': profit,
            'reason': reason,
            'confidence': pos['confidence']
        })
        
        del self.open_positions[symbol]
        logger.info(f"CLOSE: {symbol} @ {exit_price:.5f} ({reason}), Profit: {profit:.2f}")
    
    def _save_results(self):
        """Save backtest results to JSON"""
        wins = [t for t in self.trades if t['profit'] > 0]
        losses = [t for t in self.trades if t['profit'] <= 0]
        
        results = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_profit': self.balance - self.initial_balance,
                'total_trades': len(self.trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(self.trades) if self.trades else 0,
                'avg_win': sum(t['profit'] for t in wins) / len(wins) if wins else 0,
                'avg_loss': sum(t['profit'] for t in losses) / len(losses) if losses else 0,
            },
            'trades': self.trades
        }
        
        # Save to file
        filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
        print(f"\n=== BACKTEST COMPLETE ===")
        print(f"Total trades: {results['summary']['total_trades']}")
        print(f"Win rate: {results['summary']['win_rate']:.2%}")
        print(f"Total profit: ${results['summary']['total_profit']:.2f}")
        print(f"Final balance: ${results['summary']['final_balance']:.2f}")
        
        return results

async def main():
    """Run backtest"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='EURUSD')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--config', default='config.json')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Run backtest
    backtest = Backtest(args.config)
    await backtest.run(args.symbol, start_date, end_date)
    
    # Cleanup
    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(main())