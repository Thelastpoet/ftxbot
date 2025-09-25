"""
Trade Logger Module - Refactored for Robustness and Performance
Records trade history to an append-only JSON Lines file.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and Decimal objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class TradeLogger:
    """
    Handles trade logging and performance tracking using an append-only JSON Lines file.
    This approach is fast, robust, and prevents data corruption.
    """
    
    def __init__(self, log_file: str = 'trades.jsonl'):
        self.log_file = Path(log_file)
        self.trades: Dict[int, Dict[str, Any]] = {}  # Store trades in a dict keyed by ticket for efficient lookup
        self.load_trades()
        
    def log_trade(self, trade_details: Dict[str, Any]):
        """
        Logs an opening trade event.
        
        Args:
            trade_details: Dictionary containing new trade information.
        """
        try:
            ticket = trade_details.get('ticket')
            if not ticket:
                logger.error("Cannot log trade without a ticket.")
                return

            event = {
                "event_type": "open",
                "timestamp": datetime.now().isoformat(),
                "trade": trade_details
            }
            
            # Append to the log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event, cls=DateTimeEncoder) + '\n')

            # Update in-memory state
            self.trades[ticket] = trade_details
            
            logger.info(f"Trade OPEN logged for ticket {ticket}: {trade_details.get('symbol')} {trade_details.get('order_type')}")

        except Exception as e:
            logger.error(f"Error logging open trade for ticket {ticket}: {e}", exc_info=True)
    
    def update_trade(self, ticket: int, exit_price: float, profit: float, status: str):
        """
        Logs a closing or updating trade event.
        
        Args:
            ticket: Trade ticket number.
            exit_price: Exit price of the trade.
            profit: Profit/loss of the trade.
            status: Final status of the trade (e.g., 'CLOSED_SL', 'CLOSED_TP').
        """
        try:
            if ticket not in self.trades:
                logger.warning(f"Cannot update trade for ticket {ticket}: Not found in memory. It might be from a previous session.")
                # Still log the update event, load_trades will handle it
            
            exit_time = datetime.now()
            duration = None
            if self.trades.get(ticket, {}).get('timestamp'):
                entry_time_iso = self.trades[ticket]['timestamp']
                entry_time = datetime.fromisoformat(entry_time_iso) if isinstance(entry_time_iso, str) else entry_time_iso
                duration = str(exit_time - entry_time)

            update_data = {
                "ticket": ticket,
                "exit_price": exit_price,
                "profit": profit,
                "status": status,
                "exit_time": exit_time.isoformat(),
                "duration": duration
            }

            event = {
                "event_type": "update",
                "timestamp": exit_time.isoformat(),
                "update": update_data
            }

            # Append to the log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event, cls=DateTimeEncoder) + '\n')

            # Update in-memory state
            if ticket in self.trades:
                self.trades[ticket].update(update_data)
            else:
                # If a trade is updated that wasn't in memory, we should probably log it as a new entry
                # but this case is unlikely if the bot is running continuously.
                # For now, we rely on the log file as the source of truth.
                logger.info(f"Logged an update for a trade (ticket {ticket}) not currently in active memory.")


            logger.info(f"Trade UPDATE logged for ticket {ticket}: profit={profit}, status={status}")

        except Exception as e:
            logger.error(f"Error updating trade for ticket {ticket}: {e}", exc_info=True)

    def load_trades(self):
        """
        Loads and reconstructs the state of all trades from the JSON Lines log file.
        This ensures data is consistent and resilient to corruption.
        """
        if not self.log_file.exists():
            logger.info("Trade log file not found, starting fresh.")
            return

        temp_trades: Dict[int, Dict[str, Any]] = {}
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("event_type")
                        
                        if event_type == "open":
                            trade = event.get("trade", {})
                            ticket = trade.get("ticket")
                            if ticket:
                                temp_trades[ticket] = trade
                        elif event_type == "update":
                            update = event.get("update", {})
                            ticket = update.get("ticket")
                            if ticket and ticket in temp_trades:
                                temp_trades[ticket].update(update)

                    except json.JSONDecodeError:
                        logger.warning(f"Skipping corrupted line in trade log: {line.strip()}")
                        continue
            
            self.trades = temp_trades
            logger.info(f"Loaded and reconstructed {len(self.trades)} trades from log file.")

        except Exception as e:
            logger.error(f"Failed to load and reconstruct trades from log: {e}", exc_info=True)
            self.trades = {} # Start fresh if loading fails catastrophically

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Returns a list of all trades held in memory."""
        return list(self.trades.values())

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from the current in-memory trade history.
        """
        all_trades = self.get_all_trades()
        if not all_trades:
            return {}
        
        closed_trades = [t for t in all_trades if str(t.get('status', '')).startswith('CLOSED')]
        
        if not closed_trades:
            return {
                'total_trades': len(all_trades),
                'open_trades': len(all_trades) - len(closed_trades),
                'closed_trades': 0
            }
        
        profits = [float(t.get('profit', 0)) for t in closed_trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        metrics = {
            'total_trades': len(all_trades),
            'open_trades': len(all_trades) - len(closed_trades),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_profit': sum(profits),
            'average_profit': sum(profits) / len(closed_trades) if closed_trades else 0,
            'average_win': sum(winning_trades) / len(winning_trades) if winning_trades else 0,
            'average_loss': sum(losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0,
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
            
            peak = cumulative[0] if cumulative else 0
            max_dd = 0
            for value in cumulative:
                if value > peak:
                    peak = value
                
                # Avoid division by zero if peak is zero or negative
                if peak > 0:
                    dd = (peak - value) / peak
                    if dd > max_dd:
                        max_dd = dd
            
            metrics['max_drawdown'] = max_dd
        
        return metrics

    def generate_report(self, output_file: str = 'performance_report.txt'):
        """
        Generate a performance report.
        """
        metrics = self.get_performance_metrics()
        
        report = [
            "=" * 60,
            "TRADING PERFORMANCE REPORT",
            "=" * 60,
            f"Generated: {datetime.now()}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades: {metrics.get('total_trades', 0)}",
            f"Open Trades: {metrics.get('open_trades', 0)}",
            f"Closed Trades: {metrics.get('closed_trades', 0)}",
            f"Winning Trades: {metrics.get('winning_trades', 0)}",
            f"Losing Trades: {metrics.get('losing_trades', 0)}",
            f"Win Rate: {metrics.get('win_rate', 0):.2%}",
            "",
            "PROFIT METRICS",
            "-" * 40,
            f"Total Profit: ${metrics.get('total_profit', 0):.2f}",
            f"Average Profit: ${metrics.get('average_profit', 0):.2f}",
            f"Average Win: ${metrics.get('average_win', 0):.2f}",
            f"Average Loss: ${metrics.get('average_loss', 0):.2f}",
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"Max Win: ${metrics.get('max_win', 0):.2f}",
            f"Max Loss: ${metrics.get('max_loss', 0):.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
            "=" * 60,
        ]
        
        report_text = "\n".join(report)
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Performance report generated: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write performance report: {e}")
            
        return report_text