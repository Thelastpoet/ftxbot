"""Per-symbol runtime context: config overrides, calibrator, and optimizer."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from calibrator import CalibratorConfig, OnlineLogisticCalibrator
from optimizer import OptimizerConfig, ParameterOptimizer

logger = logging.getLogger(__name__)

@dataclass
class SymbolProfile:
    symbol: str
    base_params: Dict[str, float]
    overrides: Dict[str, float]
    bounds: Dict[str, List[float]]
    adaptive_keys: List[str]
    risk_overrides: Dict[str, float]


class SymbolRuntimeContext:
    """Holds symbol-specific runtime components."""

    def __init__(
        self,
        profile: SymbolProfile,
        calibration_cfg: Optional[Dict[str, Any]],
        calibrator_state_dir: Path,
        optimizer_state_dir: Path,
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.symbol = profile.symbol
        self.profile = profile
        self.risk_overrides = dict(profile.risk_overrides) if profile.risk_overrides else {}
        self.calibration_cfg = calibration_cfg or {}
        self.calibrator_enabled = bool(self.calibration_cfg.get("enabled", False))
        self._calibrator = self._build_calibrator(calibrator_state_dir)
        
        # Ensure adaptive keys include breakout quality knobs
        if 'breakout_threshold_pips' not in profile.adaptive_keys:
            profile.adaptive_keys.append('breakout_threshold_pips')
        if 'min_peak_rank' not in profile.adaptive_keys:
            profile.adaptive_keys.append('min_peak_rank')
        if 'proximity_threshold_pips' not in profile.adaptive_keys:
            profile.adaptive_keys.append('proximity_threshold_pips')
        if 'require_close_breakout' not in profile.adaptive_keys:
            profile.adaptive_keys.append('require_close_breakout')

        optimizer_cfg = OptimizerConfig(
            symbol=self.symbol,
            base_params=self.effective_base,
            bounds=profile.bounds,
            state_path=optimizer_state_dir / f"{self.symbol}.json",
            adaptive_keys=profile.adaptive_keys,
        )
        self.optimizer = ParameterOptimizer(optimizer_cfg)
        if trade_history:
            self.optimizer.bootstrap(trade_history)

    # ------------------------------------------------------------------
    @property
    def effective_base(self) -> Dict[str, float]:
        base = dict(self.profile.base_params)
        base.update(self.profile.overrides)
        return base

    def get_parameters(self) -> Dict[str, float]:
        params = self.effective_base
        params.update(self.optimizer.current_parameters())
        return params

    def get_risk_overrides(self) -> Dict[str, float]:
        return dict(self.risk_overrides) if self.risk_overrides else {}

    def evaluate_signal(self, features: Dict[str, float], risk_reward_ratio: float) -> Dict[str, float]:
        if not self.calibrator_enabled or not self._calibrator or not features:
            return {"accepted": True, "probability": 1.0, "threshold": 0.0}
        probability = float(self._calibrator.predict_proba(features))
        threshold = float(self._calibrator.decision_threshold(risk_reward_ratio))
        return {
            "accepted": probability >= threshold,
            "probability": probability,
            "threshold": threshold,
        }

    def record_trade_result(self, trade: Dict[str, Any]) -> None:
        outcome = self._grade_trade(trade)
        if outcome:
            trade["result_grade"] = outcome.get("grade")
            trade["result_label"] = outcome.get("label")
            if outcome.get("risk_multiple") is not None:
                trade["risk_multiple"] = outcome.get("risk_multiple")
            if outcome.get("target_multiple") is not None:
                trade["target_multiple"] = outcome.get("target_multiple")
        if not self.optimizer.update(trade):
            return
        if not self.calibrator_enabled or not self._calibrator:
            return
        features = self._extract_features(trade)
        if not features:
            return
        label = float(outcome.get("label", 0.0)) if outcome else float(trade.get("result_label", 0.0) or 0.0)
        try:
            self._calibrator.update(features, label)
        except Exception as e:
            logger.error(f"Failed to update calibrator for trade {trade.get('ticket')}: {e}")

    # ------------------------------------------------------------------
    def _build_calibrator(self, state_dir: Path) -> Optional[OnlineLogisticCalibrator]:
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            cfg_kwargs = dict(self.calibration_cfg)
            cfg_kwargs.pop("ttl_minutes", None)  # Not part of CalibratorConfig dataclass
            cfg = CalibratorConfig(**{
                "enabled": cfg_kwargs.get("enabled", False),
                "margin": cfg_kwargs.get("margin", 0.05),
                "learning_rate": cfg_kwargs.get("learning_rate", 0.01),
                "l2": cfg_kwargs.get("l2", 1e-3),
                "state_file": str(state_dir / f"{self.symbol}.json"),
                "feature_names": cfg_kwargs.get("feature_names")
                or CalibratorConfig().feature_names,
            })
            return OnlineLogisticCalibrator(cfg)
        except Exception:
            return None

    def _extract_features(self, trade: Dict[str, Any]) -> Optional[Dict[str, float]]:
        container = trade.get("calibrator")
        raw_features = None
        if isinstance(container, dict):
            raw_features = container.get("features")
        if raw_features is None and isinstance(trade.get("features"), dict):
            raw_features = trade.get("features")
        if not isinstance(raw_features, dict):
            return None
        features: Dict[str, float] = {}
        for key, value in raw_features.items():
            try:
                features[key] = float(value)
            except (TypeError, ValueError):
                continue
        return features or None

    def _grade_trade(self, trade: Dict[str, Any]) -> Optional[Dict[str, float]]:
        status = str(trade.get("status", "")).upper()
        profit = self._safe_float(trade.get("profit")) or 0.0
        rr = self._safe_float(self._param_from_trade(trade, "risk_reward_ratio"))
        stop_loss_pips = self._safe_float(trade.get("stop_loss_pips"))
        if stop_loss_pips is None:
            stop_loss_pips = self._safe_float(self._param_from_trade(trade, "min_stop_loss_pips"))
        pip_size = self._safe_float(trade.get("pip_size"))
        entry_price = self._safe_float(trade.get("entry_price"))
        exit_price = self._safe_float(trade.get("exit_price"))
        order_type = str(trade.get("order_type", "")).upper()
        target_multiple = rr if rr is not None else 1.0

        if status == "CLOSED_TP":
            risk_multiple = target_multiple if target_multiple else 1.0
            return {"label": 1.0, "grade": "tp", "risk_multiple": risk_multiple, "target_multiple": target_multiple}
        if status == "CLOSED_SL":
            return {"label": 0.0, "grade": "sl", "risk_multiple": -1.0, "target_multiple": target_multiple}

        risk_multiple = None
        if (
            pip_size and pip_size > 0
            and stop_loss_pips and stop_loss_pips > 0
            and entry_price is not None
            and exit_price is not None
        ):
            if order_type == "SELL":
                move = entry_price - exit_price
            else:
                move = exit_price - entry_price
            move_pips = move / pip_size
            risk_multiple = move_pips / stop_loss_pips if stop_loss_pips else None

        if risk_multiple is not None:
            label, grade = self._label_from_r_multiple(risk_multiple, target_multiple)
            return {"label": label, "grade": grade, "risk_multiple": risk_multiple, "target_multiple": target_multiple}

        if profit > 0:
            return {"label": 0.35, "grade": "scratch_win", "risk_multiple": None, "target_multiple": target_multiple}
        if profit < 0:
            return {"label": 0.1, "grade": "scratch_loss", "risk_multiple": None, "target_multiple": target_multiple}
        return {"label": 0.0, "grade": "flat", "risk_multiple": None, "target_multiple": target_multiple}

    def _label_from_r_multiple(self, risk_multiple: float, target_multiple: float) -> Tuple[float, str]:
        target = max(target_multiple or 1.0, 1.0)
        if risk_multiple >= target * 0.9:
            return 1.0, "tp_manual"
        if risk_multiple >= target * 0.5:
            return 0.65, "partial_win"
        if risk_multiple >= 0.0:
            return 0.35, "scratch_win"
        if risk_multiple >= -0.5:
            return 0.2, "managed_loss"
        return 0.0, "loss"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _param_from_trade(self, trade: Dict[str, Any], key: str) -> Any:
        params = trade.get("parameters")
        if isinstance(params, dict) and key in params:
            return params[key]
        return trade.get(key)


def load_symbol_profile(symbol: str, base_params: Dict[str, float], profile_path: Path) -> SymbolProfile:
    overrides: Dict[str, float] = {}
    bounds: Dict[str, List[float]] = {}
    adaptive_keys = ["min_stop_loss_pips", "stop_loss_atr_multiplier", "risk_reward_ratio"]
    risk_overrides: Dict[str, float] = {}

    if profile_path.exists():
        try:
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            overrides = data.get("parameters", {}) or {}
            bounds = data.get("bounds", {}) or {}
            adaptive_keys = data.get("adaptive_keys", adaptive_keys)
            risk_overrides = data.get("risk", {}) or {}
        except Exception:
            overrides = {}
            bounds = {}
            risk_overrides = {}

    numeric_overrides = {k: float(v) for k, v in overrides.items() if isinstance(v, (int, float))}
    numeric_risk = {k: float(v) for k, v in risk_overrides.items() if isinstance(v, (int, float))}

    return SymbolProfile(
        symbol=symbol,
        base_params=base_params,
        overrides=numeric_overrides,
        bounds=bounds,
        adaptive_keys=adaptive_keys,
        risk_overrides=numeric_risk,
    )


