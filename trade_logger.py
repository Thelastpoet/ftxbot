"""
Trade Logger Module - Robust JSONL event log with consolidation
- Appends 'open' and 'update' events to an append-only JSON Lines file
- Consolidates events into latest trade states for querying/reporting
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -------------------------
# JSON helpers
# -------------------------

class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime and Decimal objects."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # Accept both with/without timezone
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _deepcopy_jsonable(obj: Any) -> Any:
    # Ensure we store a JSON-friendly copy (no datetimes/decimals leaking through)
    return json.loads(json.dumps(obj, cls=DateTimeEncoder))


def _first_non_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


# -------------------------
# Trade Logger
# -------------------------

class TradeLogger:
    """
    Persistent trade logging as JSON Lines (JSONL).
    Each line is one event:
      - {"event_type":"open", "timestamp":"...", "trade": {...}}
      - {"event_type":"update", "timestamp":"...", "update": {...}}

    Public API:
      - log_trade(trade_details)    -> append 'open'
      - update_trade(ticket, exit_price, profit, status, **kw) -> append 'update'
      - get_all_trades()            -> consolidated list of latest trade states
    """

    def __init__(self, log_file: str = "trades.log"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            # Use JSONL (append-only). Start as empty file.
            self.log_path.write_text("", encoding="utf-8")

    # ----------- Append operations -----------

    def log_trade(self, trade_details: Dict[str, Any]) -> None:
        """
        Append a new OPEN event. trade_details should be a flat dict of the trade fields.
        We'll wrap it under {"event_type":"open", "timestamp":..., "trade":{...}}.
        """
        try:
            payload = {
                "event_type": "open",
                "timestamp": datetime.now().isoformat(),
                "trade": _deepcopy_jsonable(trade_details),
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, cls=DateTimeEncoder))
                f.write("\n")

            # Console summary (shows both ticks & pips if available)
            t = trade_details
            sl_ticks = t.get("stop_loss_ticks")
            sl_pips  = t.get("stop_loss_pips")
            drift_ticks = t.get("drift_ticks")
            drift_pips  = t.get("drift_pips")

            msg = [
                f"TRADE LOGGED (OPEN): {t.get('symbol')} {t.get('order_type')} @ {t.get('entry_price')}",
                f"  Volume: {t.get('volume')}"
            ]
            if sl_ticks is not None and sl_pips is not None:
                msg.append(f"  Stop Loss: {sl_ticks:.1f} ticks (~{sl_pips:.1f} pips)")
            if drift_ticks is not None and drift_pips is not None:
                msg.append(f"  Drift: {drift_ticks:.1f} ticks (~{drift_pips:.1f} pips)")
            logger.info("\n".join(msg))

        except Exception as e:
            logger.error(f"Failed to log trade: {e}", exc_info=True)

    def update_trade(
        self,
        ticket: Any,
        exit_price: float,
        profit: float,
        status: str,
        *,
        exit_time: Optional[datetime] = None,
        duration: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append an UPDATE event for a given ticket (or position_ticket).
        If exit_time/duration not provided, they will be inferred if possible.
        """
        try:
            if exit_time is None:
                exit_time = datetime.now()

            update_block: Dict[str, Any] = {
                "ticket": ticket,
                "exit_price": exit_price,
                "profit": profit,
                "status": status,
                "exit_time": exit_time.isoformat(),
            }
            if duration is not None:
                update_block["duration"] = duration
            if isinstance(extra, dict) and extra:
                update_block.update(_deepcopy_jsonable(extra))

            payload = {
                "event_type": "update",
                "timestamp": datetime.now().isoformat(),
                "update": update_block,
            }

            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, cls=DateTimeEncoder))
                f.write("\n")

            logger.info(
                f"TRADE LOGGED (UPDATE): ticket={ticket} status={status} exit_price={exit_price} profit={profit}"
            )

        except Exception as e:
            logger.error(f"Failed to update trade: {e}", exc_info=True)

    # ----------- Read / Consolidate -----------

    def _iter_events(self) -> Iterable[Dict[str, Any]]:
        """Yield each JSON object (event) from the JSONL file."""
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue
        except FileNotFoundError:
            return

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """
        Return a consolidated list of latest trade states.
        - Start from each OPEN event's 'trade' dict.
        - Apply any UPDATE events on top (match by ticket or position_ticket).
        """
        # Map by a canonical key (prefer 'position_ticket', fallback to 'ticket')
        def key_of(trade_like: Dict[str, Any]) -> Optional[str]:
            tkt = trade_like.get("position_ticket")
            if tkt is None:
                tkt = trade_like.get("ticket")
            return str(tkt) if tkt is not None else None

        latest: Dict[str, Dict[str, Any]] = {}

        for ev in self._iter_events():
            et = ev.get("event_type")
            if et == "open":
                trade = ev.get("trade") or {}
                # Normalize
                trade = dict(trade)
                trade.setdefault("status", "OPEN")
                # Some sources keep the entry timestamp inside 'trade'
                # If absent, use event's timestamp
                trade.setdefault("timestamp", ev.get("timestamp"))
                k = key_of(trade)
                if k:
                    latest[k] = trade

            elif et == "update":
                upd = ev.get("update") or {}
                # Find existing trade by ticket / position_ticket
                k = _first_non_none(upd.get("ticket"), upd.get("position_ticket"))
                k = str(k) if k is not None else None
                if not k:
                    continue
                base = latest.get(k, {"ticket": k, "status": "OPEN"})
                # Apply update fields
                for fld in ("exit_price", "profit", "status", "exit_time", "duration"):
                    if fld in upd and upd[fld] is not None:
                        base[fld] = upd[fld]
                # Ensure symbol is present if we can infer (not strictly required)
                if "symbol" not in base and "symbol" in upd:
                    base["symbol"] = upd["symbol"]
                latest[k] = base

            else:
                # Backward compatibility: a flat trade dict line (rare)
                if "symbol" in ev and "entry_price" in ev:
                    trade = dict(ev)
                    trade.setdefault("status", "OPEN")
                    k = key_of(trade)
                    if k:
                        latest[k] = trade

        # Return as list
        return list(latest.values())

    # ----------- Metrics / Reporting -----------

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Compute basic performance metrics from consolidated trades.
        Counts only those with a CLOSED_* status for PnL stats.
        """
        trades = self.get_all_trades()

        total = len(trades)
        closed = [t for t in trades if str(t.get("status", "")).upper().startswith("CLOSED")]
        open_trades = [t for t in trades if not str(t.get("status", "OPEN")).upper().startswith("CLOSED")]

        wins = [t for t in closed if float(t.get("profit", 0) or 0) > 0]
        losses = [t for t in closed if float(t.get("profit", 0) or 0) <= 0]

        gross_profit = sum(float(t.get("profit", 0) or 0) for t in wins)
        gross_loss = sum(abs(float(t.get("profit", 0) or 0)) for t in losses)

        win_rate = (len(wins) / len(closed)) if closed else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        net_profit = gross_profit - gross_loss

        # Simple running equity & max drawdown estimate (based on closed PnL only)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in closed:
            equity += float(t.get("profit", 0) or 0)
            peak = max(peak, equity)
            dd = (peak - equity)
            max_dd = max(max_dd, dd)

        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (-gross_loss / len(losses)) if losses else 0.0
        max_win = max((float(t.get("profit", 0) or 0) for t in closed), default=0.0)
        max_loss = min((float(t.get("profit", 0) or 0) for t in closed), default=0.0)

        return {
            "total_trades": total,
            "closed_trades": len(closed),
            "open_trades": len(open_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": -gross_loss,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "max_drawdown": max_dd,
        }

    def generate_performance_report(self, output_file: str = "performance_report.txt") -> str:
        """
        Produce a simple text report from current logs (closed trades only).
        """
        m = self.get_performance_metrics()
        lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            f"Total Trades: {m.get('total_trades', 0)}",
            f"Closed Trades: {m.get('closed_trades', 0)}",
            f"Open Trades: {m.get('open_trades', 0)}",
            f"Win Rate: {m.get('win_rate', 0):.2%}",
            f"Profit Factor: {m.get('profit_factor', 0):.2f}",
            f"Net Profit: {m.get('net_profit', 0):.2f}",
            f"Gross Profit: {m.get('gross_profit', 0):.2f}",
            f"Gross Loss: {m.get('gross_loss', 0):.2f}",
            f"Avg Win: {m.get('avg_win', 0):.2f}",
            f"Avg Loss: {m.get('avg_loss', 0):.2f}",
            f"Max Win: {m.get('max_win', 0):.2f}",
            f"Max Loss: {m.get('max_loss', 0):.2f}",
            f"Max Drawdown: {m.get('max_drawdown', 0):.2f}",
            "=" * 60,
        ]
        report_text = "\n".join(lines)
        try:
            Path(output_file).write_text(report_text, encoding="utf-8")
            logger.info(f"Performance report generated: {output_file}")
        except Exception as e:
            logger.error(f"Failed to write performance report: {e}")
        return report_text


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
