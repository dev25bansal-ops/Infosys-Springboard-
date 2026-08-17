"""Stage 2 — Isolation Forest microstructure scorer.

Trained on 12 features (F1+F2). Detects OBI shifts, cancellation spikes,
and depth anomalies.
Latency: ~1 ms. Pass-through: ~20% of suspects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)

# 12 features used by Stage 2 (F1 + F2)
STAGE2_FEATURES = [
    "f1_mid_velocity_50ms", "f1_mid_velocity_200ms", "f1_micro_price",
    "f1_trade_arrival_rate", "f1_cancel_to_trade_ratio",
    "f2_bid_depth_10", "f2_ask_depth_10", "f2_obi_10",
    "f2_weighted_mid_10", "f2_depth_slope",
    # Add 2 from F3 for richer scoring
    "f3_vpin", "f3_kyle_lambda",
]


class Stage2IsolationForest:
    """Isolation Forest on 12 microstructure features."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self._model = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._ticks_processed = 0
        self._ticks_passed = 0

    def fit(self, features_matrix: np.ndarray) -> None:
        """Fit the Isolation Forest on a matrix of normal features.

        Args:
            features_matrix: shape (n_samples, 12), normal market data.
        """
        from sklearn.ensemble import IsolationForest

        # Normalize features
        self._feature_means = features_matrix.mean(axis=0)
        self._feature_stds = features_matrix.std(axis=0)
        self._feature_stds[self._feature_stds == 0] = 1.0

        normalized = (features_matrix - self._feature_means) / self._feature_stds

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(normalized)
        logger.info(
            "Stage 2 IsolationForest trained on %d samples, contamination=%.3f",
            features_matrix.shape[0], self.contamination,
        )

    def score(self, tick: Tick) -> tuple[float, bool]:
        """Returns (anomaly_score in [0,1], should_pass_to_stage3)."""
        if self._model is None:
            # Fallback: use simple z-score heuristic if not trained
            return self._fallback_score(tick)

        features = tick.features
        # NaN/None-guard: a missing feature must never turn the anomaly score NaN
        # (which would silently fail the `> 0.5` gate and suppress the tick).
        vals = []
        for f in STAGE2_FEATURES:
            v = features.get(f, 0.0)
            vals.append(0.0 if v is None or (isinstance(v, float) and v != v) else float(v))
        vec = np.array([vals])
        normalized = (vec - self._feature_means) / self._feature_stds

        # IsolationForest decision_function: higher = more normal
        # Convert to anomaly score in [0, 1]: score = 1 - sigmoid(decision_function)
        raw = self._model.decision_function(normalized)[0]
        anomaly_score = 1.0 / (1.0 + float(np.exp(raw)))

        should_pass = anomaly_score > 0.5
        self._ticks_processed += 1
        if should_pass:
            self._ticks_passed += 1
        return float(anomaly_score), should_pass

    def _fallback_score(self, tick: Tick) -> tuple[float, bool]:
        """Simple z-score heuristic when model isn't trained yet."""
        features = tick.features
        obi = features.get("f2_obi_10", 0.0)
        vel = features.get("f1_mid_velocity_50ms", 0.0)
        # If OBI is extreme or velocity is high, flag as suspect
        score = min(1.0, abs(obi) * 5 + abs(vel) * 100)
        return float(score), score > 0.5

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump({
            "model": self._model,
            "means": self._feature_means,
            "stds": self._feature_stds,
            "config": {"n_estimators": self.n_estimators, "contamination": self.contamination},
        }, path)

    def load(self, path: str | Path) -> None:
        import joblib
        data = joblib.load(path)
        self._model = data["model"]
        self._feature_means = data["means"]
        self._feature_stds = data["stds"]

    @property
    def pass_through_rate(self) -> float:
        if self._ticks_processed == 0:
            return 0.0
        return self._ticks_passed / self._ticks_processed
