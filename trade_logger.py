import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional
from datetime import datetime


class _DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that renders datetime objects as ISO strings."""
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class TradeLogger:
    """
    Single-source-of-truth trade logger for a live trading bot.

    - Keeps all trades in memory: self.trades (List[Dict]).
    - Centralizes persistence via persist_all(): writes a JSON snapshot atomically.
    - No JSON re-load on each append/update.
    - Backwards compatible with main.py usage:
        TradeLogger('trades.log')
        log_trade({...})
        update_trade(ticket, exit_price, profit, status)  # positional ok
    """

    def __init__(self, base_path: str,
                 json_path: Optional[str] = None) -> None:
        """
        Args:
            base_path: Passed as 'trades.log' in your code; we derive .json alongside it.
            json_path: Optional explicit path override.
        """
        base = Path(base_path)
        stem = base.stem or "trades"

        # Files used for persistence
        self.json_file: Path = Path(json_path) if json_path else base.with_name(f"{stem}.json")

        # In-memory store (source of truth)
        self.trades: List[Dict[str, Any]] = []

        # Simple lock to serialize writes in async/multi-thread contexts
        self._lock = RLock()

        # One-time load of existing trades (if JSON exists)
        self._load_once()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Append a new trade record and persist snapshots.
        Expected keys typically include:
          timestamp (datetime), symbol, order_type, entry_price, stop_loss, take_profit,
          ticket, reason, confidence, signal_time, drift_pips, volume, etc.
        """
        with self._lock:
            self.trades.append(trade)
            self.persist_all()

    def update_trade(self,
                     ticket: Optional[int] = None,
                     exit_price: Optional[float] = None,
                     profit: Optional[float] = None,
                     status: Optional[str] = None,
                     **fields: Any) -> None:
        """
        Update a trade in-memory, then persist snapshots.

        Backward-compatible with positional usage:
            update_trade(ticket, exit_price, profit, status)

        Also supports keyword usage:
            update_trade(ticket=123, exit_price=1.2345, profit=42.0, status='CLOSED_TP')

        You can pass extra fields via **fields (e.g., commission, swap, close_time).
        """
        with self._lock:
            target = None

            if ticket is not None:
                for t in self.trades:
                    if t.get("ticket") == ticket or t.get("order_ticket") == ticket:
                        target = t
                        break

            # If not found by ticket but some identifying fields were provided,
            # try the first record that matches all of them.
            if target is None and fields:
                for t in self.trades:
                    if all(t.get(k) == v for k, v in fields.items() if v is not None):
                        target = t
                        break

            if target is None:
                # Not found — nothing to update (intentionally silent for live safety)
                return

            if exit_price is not None:
                target["exit_price"] = exit_price
            if profit is not None:
                target["profit"] = profit
            if status is not None:
                target["status"] = status

            # Apply any additional updates provided
            for k, v in fields.items():
                if v is not None:
                    target[k] = v

            self.persist_all()

    def persist_all(self) -> None:
        """
        Single serialization path:
        - Writes full JSON snapshot of self.trades (atomic: tmp + replace).
        """
        self._write_json_snapshot(self.trades)

    # -------------------------------------------------------------------------
    # Internals (I/O)
    # -------------------------------------------------------------------------
    def _load_once(self) -> None:
        """Load existing trades from JSON once at startup (if present and valid)."""
        try:
            if self.json_file.exists():
                raw = self.json_file.read_text(encoding="utf-8")
                if raw.strip():
                    self.trades = json.loads(raw)
                else:
                    self.trades = []
            else:
                self.trades = []
        except Exception:
            # On corrupt JSON, start fresh (optionally back up here)
            self.trades = []

    def _write_json_snapshot(self, rows: List[Dict[str, Any]]) -> None:
        """Atomic JSON snapshot write."""
        tmp = self.json_file.with_suffix(self.json_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, cls=_DateTimeEncoder, ensure_ascii=False)
        os.replace(tmp, self.json_file)
