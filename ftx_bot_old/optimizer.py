"""Symbol-level parameter optimizer.

Adjusts adaptive trading parameters (e.g. min_stop_loss_pips, risk_reward_ratio)
using recent trade outcomes. Persists state per symbol so adjustments survive
restarts and replays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class OptimizerConfig:
    symbol: str
    base_params: Dict[str, float]
    bounds: Dict[str, List[float]]
    state_path: Path
    adaptive_keys: List[str]
    history_window: int = 20
    max_history: int = 200
    cooloff_window: int = 8
    cooloff_win_rate: float = 0.25
    cooloff_expectancy: float = -1.0
    cooloff_duration: int = 5
    promote_win_rate: float = 0.55
    promote_expectancy: float = 0.0


class ParameterOptimizer:
    """Heuristic optimizer with guard rails and cool-off handling."""

    def __init__(self, config: OptimizerConfig) -> None:
        self.config = config
        self.state: Dict[str, Any] = {
            "current": {key: float(config.base_params.get(key, 0.0)) for key in config.adaptive_keys},
            "history": [],
            "seen_tickets": [],
            "cooldown_remaining": 0,
            "best_snapshot": None,
            "stats": {"resets": 0},
        }
        self._load_state()

    @property
    def state_path(self) -> Path:
        return self.config.state_path

    def bootstrap(self, trades: Iterable[Dict]) -> None:
        """Seed optimizer history from existing trades (idempotent)."""
        updated = False
        for trade in trades or []:
            status = str(trade.get("status", "")).upper()
            if not status.startswith("CLOSED"):
                continue
            updated |= self._process_trade(trade, persist=False)
        if updated:
            self._save_state()

    def update(self, trade: Dict) -> bool:
        """Ingest a newly closed trade. Returns True if state changed."""
        changed = self._process_trade(trade, persist=True)
        if changed:
            self._save_state()
        return changed

    def current_parameters(self) -> Dict[str, float]:
        return dict(self.state.get("current", {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_trade(self, trade: Dict, persist: bool) -> bool:
        ticket = trade.get("ticket")
        if ticket in (self.state.get("seen_tickets") or []):
            return False

        status = str(trade.get("status", "")).upper()
        if not status.startswith("CLOSED"):
            return False

        result = self._extract_result(trade, status)
        profit_val = self._safe_float(trade.get("profit")) or 0.0
        entry = {
            "ticket": ticket,
            "timestamp": trade.get("timestamp") or trade.get("exit_time"),
            "result": result,
            "status": status,
            "profit": profit_val,
            "risk_reward_ratio": self._safe_float(self._lookup_param(trade, "risk_reward_ratio")),
            "min_stop_loss_pips": self._safe_float(self._lookup_param(trade, "min_stop_loss_pips")),
            "stop_loss_pips": self._safe_float(trade.get("stop_loss_pips")),
        }

        history: List[Dict] = self.state.setdefault("history", [])
        history.append(entry)
        if len(history) > self.config.max_history:
            del history[: len(history) - self.config.max_history]

        seen = self.state.setdefault("seen_tickets", [])
        seen.append(ticket)

        self._recompute()
        return True

    def _lookup_param(self, trade: Dict, key: str) -> Optional[float]:
        params = trade.get("parameters")
        if isinstance(params, dict) and key in params:
            return params[key]
        return trade.get(key)

    @staticmethod
    def _safe_float(value: Optional[float]) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_result(self, trade: Dict, status: str) -> int:
        profit = trade.get("profit")
        profit_val = self._safe_float(profit) or 0.0
        if status == "CLOSED_TP":
            return 1
        if status == "CLOSED_SL":
            return 0
        return 1 if profit_val > 0 else 0

    def _recompute(self) -> None:
        current = self.state.setdefault("current", {})
        for key in self.config.adaptive_keys:
            current.setdefault(key, float(self.config.base_params.get(key, 0.0)))

        window = self.state.get("history", [])[-self.config.history_window :]
        if not window:
            self._apply_bounds()
            return

        win_rate = mean(entry.get("result", 0) for entry in window)
        avg_profit = mean(entry.get("profit", 0.0) for entry in window)
        sl_hits = sum(1 for entry in window if entry.get("status") == "CLOSED_SL")
        tp_hits = sum(1 for entry in window if entry.get("status") == "CLOSED_TP")
        total = len(window) or 1
        sl_rate = sl_hits / total
        tp_rate = tp_hits / total
        losing_streak = self._losing_streak()

        if self.state.get("cooldown_remaining", 0) > 0:
            self.state["cooldown_remaining"] = max(self.state.get("cooldown_remaining", 0) - 1, 0)
            self._apply_bounds()
            return

        if self._should_reset(win_rate, avg_profit, losing_streak, len(window)):
            self._apply_reset()
            self._apply_bounds()
            return

        self._maybe_promote(win_rate, avg_profit)

        if "min_stop_loss_pips" in current:
            base = float(self.config.base_params.get("min_stop_loss_pips", current["min_stop_loss_pips"]))
            value = current["min_stop_loss_pips"]
            step = max(0.5, base * 0.05)
            if sl_rate > 0.55:
                value += step
            elif sl_rate < 0.30 and win_rate > 0.55:
                value -= step
            else:
                value += 0.25 * (base - value)
            current["min_stop_loss_pips"] = value

        if "risk_reward_ratio" in current:
            base_rr = float(self.config.base_params.get("risk_reward_ratio", current["risk_reward_ratio"]))
            value = current["risk_reward_ratio"]
            rr_step = 0.1
            if win_rate < 0.35 and sl_rate > 0.5:
                value -= rr_step
            elif win_rate > 0.6 and tp_rate > 0.4:
                value += rr_step
            else:
                value += 0.2 * (base_rr - value)
            current["risk_reward_ratio"] = value

        self._apply_bounds()

    def _losing_streak(self) -> int:
        streak = 0
        for entry in reversed(self.state.get("history", [])):
            if entry.get("result") == 0:
                streak += 1
            else:
                break
        return streak

    def _should_reset(self, win_rate: float, avg_profit: float, losing_streak: int, window_len: int) -> bool:
        if losing_streak >= max(3, self.config.cooloff_window):
            return True
        if window_len >= self.config.cooloff_window:
            if win_rate <= self.config.cooloff_win_rate and avg_profit <= self.config.cooloff_expectancy:
                return True
        return False

    def _apply_reset(self) -> None:
        fallback = self.state.get("best_snapshot")
        if not isinstance(fallback, dict):
            fallback = self.config.base_params
        current = self.state.setdefault("current", {})
        for key in self.config.adaptive_keys:
            current[key] = float(fallback.get(key, self.config.base_params.get(key, current.get(key, 0.0))))
        self.state["cooldown_remaining"] = max(int(self.config.cooloff_duration), 0)
        stats = self.state.setdefault("stats", {})
        stats["resets"] = int(stats.get("resets", 0)) + 1

    def _maybe_promote(self, win_rate: float, avg_profit: float) -> None:
        if len(self.state.get("history", [])) < self.config.history_window:
            return
        if win_rate >= self.config.promote_win_rate and avg_profit >= self.config.promote_expectancy:
            self.state["best_snapshot"] = self._current_snapshot()

    def _current_snapshot(self) -> Dict[str, float]:
        current = self.state.setdefault("current", {})
        return {key: float(current.get(key, self.config.base_params.get(key, 0.0))) for key in self.config.adaptive_keys}

    def _apply_bounds(self) -> None:
        current = self.state.setdefault("current", {})
        for key, value in list(current.items()):
            bounds = self._bounds_for(key)
            if bounds:
                lo, hi = bounds
                current[key] = float(min(max(value, lo), hi))

    def _bounds_for(self, key: str) -> Optional[List[float]]:
        raw = self.config.bounds.get(key)
        if not raw or len(raw) != 2:
            if key == "min_stop_loss_pips":
                return [float(self.config.base_params.get(key, 0.0) * 0.5 or 5.0), 80.0]
            if key == "risk_reward_ratio":
                return [0.8, 5.0]
            return None
        return [float(raw[0]), float(raw[1])]

    def _load_state(self) -> None:
        try:
            if not self.state_path.exists():
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                self._apply_bounds()
                return
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                current = data.get("current") or {}
                for key in self.config.adaptive_keys:
                    if key in current:
                        self.state["current"][key] = float(current[key])
                history = data.get("history") or []
                if isinstance(history, list):
                    self.state["history"] = history[-self.config.max_history :]
                seen = data.get("seen_tickets") or []
                if isinstance(seen, list):
                    self.state["seen_tickets"] = seen
                self.state["cooldown_remaining"] = int(data.get("cooldown_remaining", 0))
                best = data.get("best_snapshot")
                if isinstance(best, dict):
                    self.state["best_snapshot"] = {k: float(v) for k, v in best.items()}
                stats = data.get("stats")
                if isinstance(stats, dict):
                    self.state["stats"] = stats
            self._apply_bounds()
        except Exception:
            self.state["history"] = []
            self.state["seen_tickets"] = []
            self.state["cooldown_remaining"] = 0
            self.state["best_snapshot"] = None
            for key in self.config.adaptive_keys:
                self.state["current"][key] = float(self.config.base_params.get(key, 0.0))
            self._apply_bounds()

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_path.open("w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception:
            pass


