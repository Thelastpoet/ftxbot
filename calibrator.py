"""
Lightweight online logistic calibrator for trade decision gating.

Computes p_hat = sigmoid(wÂ·x + b) from a small feature vector and
optionally updates weights online after outcomes are observed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class CalibratorConfig:
    enabled: bool = False
    margin: float = 0.05  # economic safety margin over break-even prob
    learning_rate: float = 0.01
    l2: float = 1e-3
    state_file: str = "calibrator_state.json"
    feature_names: List[str] = field(default_factory=lambda: [
        "strength",       # breakout strength 0..1
        "momentum",       # candle/m1 momentum 0..1
        "dir_match",      # 0/1
        "trend_match",    # 0/1
        "spread_impact",  # dimensionless (spread/ATR, clipped)
    ])


class OnlineLogisticCalibrator:
    """Simple logistic model with online updates and JSON persistence."""

    def __init__(self, cfg: CalibratorConfig) -> None:
        self.cfg = cfg
        self.n = len(cfg.feature_names)
        self.w = np.zeros(self.n, dtype=float)
        self.b = 0.0
        self._load_state()

        # If brand new, seed with small priors roughly reflecting existing heuristics
        if not np.any(self.w):
            # modest positive weights on strength/momentum, small on dir/trend,
            # negative on spread impact
            priors = {
                "strength": 0.8,
                "momentum": 0.9,
                "dir_match": 0.3,
                "trend_match": 0.4,
                "spread_impact": -0.6,
            }
            for i, name in enumerate(self.cfg.feature_names):
                self.w[i] = priors.get(name, 0.0)
            self.b = -0.5

    def _load_state(self) -> None:
        try:
            if os.path.exists(self.cfg.state_file):
                with open(self.cfg.state_file, "r") as f:
                    data = json.load(f)
                w = np.array(data.get("w", []), dtype=float)
                b = float(data.get("b", 0.0))
                if w.shape[0] == self.n:
                    self.w = w
                    self.b = b
        except Exception:
            # Ignore state errors; start fresh
            pass

    def _save_state(self) -> None:
        try:
            with open(self.cfg.state_file, "w") as f:
                json.dump({"w": self.w.tolist(), "b": self.b}, f)
        except Exception:
            pass

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Numerically stable sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        x = np.zeros(self.n, dtype=float)
        for i, name in enumerate(self.cfg.feature_names):
            x[i] = float(features.get(name, 0.0))
        # Clip bounded features to reasonable ranges
        x[0] = float(np.clip(x[0], 0.0, 1.0))  # strength
        x[1] = float(np.clip(x[1], 0.0, 1.0))  # momentum
        x[2] = 1.0 if x[2] >= 0.5 else 0.0     # dir_match
        x[3] = 1.0 if x[3] >= 0.5 else 0.0     # trend_match
        x[4] = float(np.clip(x[4], 0.0, 1.0))  # spread_impact, clip to [0,1]
        return x

    def predict_proba(self, features: Dict[str, float]) -> float:
        """Return p_hat in [0,1] for given features."""
        x = self._vectorize(features)
        z = float(np.dot(self.w, x) + self.b)
        return self._sigmoid(z)

    def update(self, features: Dict[str, float], label: int) -> None:
        """One online gradient step on logistic loss with L2 regularization."""
        try:
            x = self._vectorize(features)
            y = 1.0 if label else 0.0
            p = self._sigmoid(float(np.dot(self.w, x) + self.b))
            # gradients
            err = p - y
            grad_w = err * x + self.cfg.l2 * self.w
            grad_b = err
            # update
            self.w -= self.cfg.learning_rate * grad_w
            self.b -= self.cfg.learning_rate * grad_b
            self._save_state()
        except Exception:
            # Stay silent on update issues to avoid trading disruptions
            pass

    def decision_threshold(self, rr_ratio: float) -> float:
        """Economic decision threshold p* = 1/(1+R) + margin, clipped to [0,1]."""
        if rr_ratio <= 0:
            return min(0.6, max(0.0, 0.5 + self.cfg.margin))
        base = 1.0 / (1.0 + rr_ratio)
        thr = base + self.cfg.margin
        return float(np.clip(thr, 0.0, 1.0))


