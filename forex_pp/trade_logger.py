"""
Trade Logger Module
Records trade history and performance metrics
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


class TradeLogger:
    """Handles trade logging and performance tracking"""
    
    def __init__(self, log_file: str = 'trades.log'):
        self.log_file = Path(log_file)
        self.csv_file = Path(log_file.replace('.log', '.csv'))
        self.json_file = Path(log_file.replace('.log', '.json'))
        self._setup_logging()
        self.trades = []
        
    def _setup_logging(self):
        """Setup logging files"""
        # Create CSV file with headers if it doesn't exist
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'order_type', 'entry_price', 
                    'volume', 'stop_loss', 'take_profit', 'ticket',
                    'exit_price', 'profit', 'duration', 'status'
                ])
    
    def log_trade(self, trade_details: Dict[str, Any]):
        """
        Log a new trade
        
        Args:
            trade_details: Dictionary containing trade information
        """
        try:
            # Add to in-memory list
            self.trades.append(trade_details)
            
            # Write to CSV
            self._write_to_csv(trade_details)
            
            # Write to JSON
            self._write_to_json(trade_details)
            
            # Write to log file (fix JSON serialization issue)
            trade_copy = trade_details.copy()
            for key, value in trade_copy.items():
                if isinstance(value, datetime):
                    trade_copy[key] = value.isoformat()

            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {json.dumps(trade_copy)}\n")
            
            logger.info(f"Trade logged: {trade_details}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    
    def _write_to_csv(self, trade_details: Dict[str, Any]):
        """Write trade to CSV file"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_details.get('timestamp', datetime.now()),
                trade_details.get('symbol', ''),
                trade_details.get('order_type', ''),
                trade_details.get('entry_price', 0),
                trade_details.get('volume', 0),
                trade_details.get('stop_loss', 0),
                trade_details.get('take_profit', 0),
                trade_details.get('ticket', ''),
                trade_details.get('exit_price', ''),
                trade_details.get('profit', ''),
                trade_details.get('duration', ''),
                trade_details.get('status', 'OPEN')
            ])
    
    def _write_to_json(self, trade_details: Dict[str, Any]):
        """Write trade to JSON file"""
        # Convert datetime to string for JSON serialization
        trade_copy = trade_details.copy()
        if isinstance(trade_copy.get('timestamp'), datetime):
            trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
        
        # Read existing trades
        existing_trades = []
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r') as f:
                    existing_trades = json.load(f)
            except:
                existing_trades = []
        
        # Append new trade
        existing_trades.append(trade_copy)
        
        # Write back
        with open(self.json_file, 'w') as f:
            json.dump(existing_trades, f, indent=2)
    
    def update_trade(self, ticket: int, exit_price: float, profit: float, status: str = 'CLOSED'):
        """
        Update a trade with exit information
        
        Args:
            ticket: Trade ticket number
            exit_price: Exit price
            profit: Trade profit/loss
            status: Trade status
        """
        try:
            # Find trade in memory
            for trade in self.trades:
                if trade.get('ticket') == ticket:
                    trade['exit_price'] = exit_price
                    trade['profit'] = profit
                    trade['status'] = status
                    trade['exit_time'] = datetime.now()
                    
                    if 'timestamp' in trade:
                        duration = datetime.now() - trade['timestamp']
                        trade['duration'] = str(duration)
                    
                    # Update JSON file
                    self._update_json_file()
                    
                    logger.info(f"Trade {ticket} updated: profit={profit}, status={status}")
                    break
                    
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
    
    def _update_json_file(self):
        """Update the JSON file with current trades"""
        trades_copy = []
        for trade in self.trades:
            trade_copy = trade.copy()
            # Convert datetime objects to strings
            for key in ['timestamp', 'exit_time']:
                if key in trade_copy and isinstance(trade_copy[key], datetime):
                    trade_copy[key] = trade_copy[key].isoformat()
            trades_copy.append(trade_copy)
        
        with open(self.json_file, 'w') as f:
            json.dump(trades_copy, f, indent=2)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from trade history
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {}
        
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                'total_trades': len(self.trades),
                'open_trades': len(self.trades),
                'closed_trades': 0
            }
        
        # Calculate metrics
        profits = [t.get('profit', 0) for t in closed_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        metrics = {
            'total_trades': len(self.trades),
            'open_trades': len(self.trades) - len(closed_trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_profit': sum(profits),
            'average_profit': sum(profits) / len(closed_trades) if closed_trades else 0,
            'average_win': sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            'average_loss': sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0,
            'max_win': max(winning_trades) if winning_trades else 0,
            'max_loss': min(losing_trades) if losing_trades else 0
        }
        
        # Calculate max drawdown
        if profits:
            cumulative = []
            cum_sum = 0
            for p in profits:
                cum_sum += p
                cumulative.append(cum_sum)
            
            peak = cumulative[0]
            max_dd = 0
            for value in cumulative:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            metrics['max_drawdown'] = max_dd
        
        return metrics
    
    def generate_report(self, output_file: str = 'performance_report.txt'):
        """
        Generate a performance report
        
        Args:
            output_file: Output file path
        """
        metrics = self.get_performance_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("TRADING PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        report.append("TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"Open Trades: {metrics.get('open_trades', 0)}")
        report.append(f"Closed Trades: {metrics.get('closed_trades', 0)}")
        report.append(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        report.append(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append("")
        report.append("PROFIT METRICS")
        report.append("-" * 40)
        report.append(f"Total Profit: ${metrics.get('total_profit', 0):.2f}")
        report.append(f"Average Profit: ${metrics.get('average_profit', 0):.2f}")
        report.append(f"Average Win: ${metrics.get('average_win', 0):.2f}")
        report.append(f"Average Loss: ${metrics.get('average_loss', 0):.2f}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Max Win: ${metrics.get('max_win', 0):.2f}")
        report.append(f"Max Loss: ${metrics.get('max_loss', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append("=" * 60)
        
        # Write report
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Performance report generated: {output_file}")
        return report_text
    
    def load_trades_from_file(self):
        """Load trades from JSON file"""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r') as f:
                    trades = json.load(f)
                    # Convert timestamp strings back to datetime
                    for trade in trades:
                        if 'timestamp' in trade:
                            trade['timestamp'] = datetime.fromisoformat(trade['timestamp'])
                        if 'exit_time' in trade:
                            trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])
                    self.trades = trades
                    logger.info(f"Loaded {len(trades)} trades from file")
            except Exception as e:
                logger.error(f"Error loading trades from file: {e}")
                self.trades = []