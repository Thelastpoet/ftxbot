"""
Trade Logger Module - Fixed Version
Records trade history and performance metrics with proper datetime handling
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union
from decimal import Decimal

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super(DateTimeEncoder, self).default(obj)

class TradeLogger:
    """Handles trade logging and performance tracking"""
    
    def __init__(self, log_file: str = 'trades.log'):
        self.log_file = Path(log_file)
        self.csv_file = Path(log_file.replace('.log', '.csv'))
        self.json_file = Path(log_file.replace('.log', '.json'))
        self._setup_logging()
        self.trades = []
        self.load_trades_from_file()
        
    def _setup_logging(self):
        """Setup logging files"""
        # Create CSV file with headers if it doesn't exist
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'order_type', 'entry_price', 
                    'volume', 'stop_loss', 'take_profit', 'ticket',
                    'exit_price', 'profit', 'duration', 'status',
                    'reason', 'confidence', 'signal_time'
                ])
    
    def _sanitize_trade_for_json(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize trade details for JSON serialization
        
        Args:
            trade_details: Original trade details
            
        Returns:
            Sanitized trade details with all datetime objects converted to strings
        """
        trade_copy = {}
        for key, value in trade_details.items():
            if isinstance(value, datetime):
                trade_copy[key] = value.isoformat()
            elif isinstance(value, (Decimal, float)):
                # Ensure numbers are JSON serializable
                trade_copy[key] = float(value)
            elif value is None:
                # Handle None values
                trade_copy[key] = None
            else:
                trade_copy[key] = value
        return trade_copy
    
    def log_trade(self, trade_details: Dict[str, Any]):
        """
        Log a new trade
        
        Args:
            trade_details: Dictionary containing trade information
        """
        try:
            # Sanitize the trade details for JSON
            sanitized_trade = self._sanitize_trade_for_json(trade_details)
            
            # Add status if not present
            if 'status' not in sanitized_trade:
                sanitized_trade['status'] = 'OPEN'
            
            # Add to in-memory list (store the sanitized version)
            self.trades.append(sanitized_trade)
            
            # Write to CSV
            self._write_to_csv(sanitized_trade)
            
            # Write to JSON
            self._write_to_json(sanitized_trade)
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                log_entry = f"{datetime.now().isoformat()} - {json.dumps(sanitized_trade, cls=DateTimeEncoder)}\n"
                f.write(log_entry)

            logger.info(f"Trade logged successfully: {sanitized_trade.get('symbol')} {sanitized_trade.get('order_type')} @ {sanitized_trade.get('entry_price')}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}", exc_info=True)
    
    def _write_to_csv(self, trade_details: Dict[str, Any]):
        """Write trade to CSV file"""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_details.get('timestamp', datetime.now().isoformat()),
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
                    trade_details.get('status', 'OPEN'),
                    trade_details.get('reason', ''),
                    trade_details.get('confidence', ''),
                    trade_details.get('signal_time', '')
                ])
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}")
    
    def _write_to_json(self, trade_details: Dict[str, Any]):
        """Write trade to JSON file"""
        try:
            # Read existing trades
            existing_trades = []
            if self.json_file.exists():
                try:
                    with open(self.json_file, 'r') as f:
                        content = f.read()
                        if content.strip():  # Only parse if file has content
                            existing_trades = json.loads(content)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Corrupted JSON file, starting fresh: {e}")
                    existing_trades = []
                    # Backup the corrupted file
                    backup_path = self.json_file.with_suffix('.json.bak')
                    self.json_file.rename(backup_path)
                    logger.info(f"Backed up corrupted JSON to {backup_path}")
            
            # Append new trade (already sanitized)
            existing_trades.append(trade_details)
            
            # Write back with custom encoder
            with open(self.json_file, 'w') as f:
                json.dump(existing_trades, f, indent=2, cls=DateTimeEncoder)
                
        except Exception as e:
            logger.error(f"Error writing to JSON: {e}", exc_info=True)
    
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
                    trade['exit_time'] = datetime.now().isoformat()
                    
                    if 'timestamp' in trade:
                        # Calculate duration
                        if isinstance(trade['timestamp'], str):
                            entry_time = datetime.fromisoformat(trade['timestamp'])
                        else:
                            entry_time = trade['timestamp']
                        duration = datetime.now() - entry_time
                        trade['duration'] = str(duration)
                    
                    # Update JSON file
                    self._update_json_file()
                    
                    # FIX: Also update the CSV file
                    self._update_csv_file()
                    
                    logger.info(f"Trade {ticket} updated: profit={profit}, status={status}")
                    break
                    
        except Exception as e:
            logger.error(f"Error updating trade: {e}", exc_info=True)
    
    def _update_json_file(self):
        """Update the JSON file with current trades"""
        try:
            # All trades should already be sanitized
            with open(self.json_file, 'w') as f:
                json.dump(self.trades, f, indent=2, cls=DateTimeEncoder)
        except Exception as e:
            logger.error(f"Error updating JSON file: {e}", exc_info=True)
            
    def _update_csv_file(self):
        """Rewrite the entire CSV file with updated trades"""
        try:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write all trades (updated ones will have new values)
                for trade in self.trades:
                    writer.writerow([
                        trade.get('timestamp', ''),
                        trade.get('symbol', ''),
                        trade.get('order_type', ''),
                        trade.get('entry_price', ''),
                        trade.get('volume', ''),
                        trade.get('stop_loss', ''),
                        trade.get('take_profit', ''),
                        trade.get('ticket', ''),
                        trade.get('exit_price', ''),
                        trade.get('profit', ''),
                        trade.get('exit_time', ''),
                        trade.get('status', 'OPEN'),
                        trade.get('reason', ''),
                        trade.get('confidence', ''),
                        trade.get('signal_time', '')
                    ])
        except Exception as e:
            logger.error(f"Error updating CSV file: {e}")
    
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
        profits = [float(t.get('profit', 0)) for t in closed_trades]
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
                    content = f.read()
                    if not content.strip():  # Empty file
                        self.trades = []
                        return
                        
                    trades = json.loads(content)
                    # Trades should already have datetime strings, not objects
                    # But we'll store them as strings consistently
                    self.trades = trades
                    logger.info(f"Loaded {len(trades)} trades from file")
            except json.JSONDecodeError as e:
                logger.error(f"Error loading trades from file (corrupted JSON): {e}")
                # Backup corrupted file
                backup_path = self.json_file.with_suffix('.json.bak')
                self.json_file.rename(backup_path)
                logger.info(f"Backed up corrupted JSON to {backup_path}")
                self.trades = []
            except Exception as e:
                logger.error(f"Unexpected error loading trades: {e}", exc_info=True)
                self.trades = []
                
    def repair_json_file(self):
        """
        Attempt to repair a corrupted JSON file
        This method can be called manually if the JSON file is corrupted
        """
        try:
            logger.info("Attempting to repair JSON file...")
            
            # Try to read from the CSV as backup
            if self.csv_file.exists():
                trades = []
                with open(self.csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV row to trade dict
                        trade = {
                            'timestamp': row.get('timestamp', ''),
                            'symbol': row.get('symbol', ''),
                            'order_type': row.get('order_type', ''),
                            'entry_price': float(row.get('entry_price', 0)),
                            'volume': float(row.get('volume', 0)),
                            'stop_loss': float(row.get('stop_loss', 0)),
                            'take_profit': float(row.get('take_profit', 0)),
                            'ticket': int(row.get('ticket', 0)) if row.get('ticket') else 0,
                            'status': row.get('status', 'OPEN')
                        }
                        
                        # Add optional fields if present
                        if row.get('exit_price'):
                            trade['exit_price'] = float(row['exit_price'])
                        if row.get('profit'):
                            trade['profit'] = float(row['profit'])
                        if row.get('duration'):
                            trade['duration'] = row['duration']
                        if row.get('reason'):
                            trade['reason'] = row['reason']
                        if row.get('confidence'):
                            trade['confidence'] = float(row['confidence'])
                        if row.get('signal_time'):
                            trade['signal_time'] = row['signal_time']
                            
                        trades.append(trade)
                
                # Write repaired JSON
                with open(self.json_file, 'w') as f:
                    json.dump(trades, f, indent=2, cls=DateTimeEncoder)
                
                self.trades = trades
                logger.info(f"Successfully repaired JSON file with {len(trades)} trades from CSV")
                return True
                
        except Exception as e:
            logger.error(f"Failed to repair JSON file: {e}", exc_info=True)
            return False