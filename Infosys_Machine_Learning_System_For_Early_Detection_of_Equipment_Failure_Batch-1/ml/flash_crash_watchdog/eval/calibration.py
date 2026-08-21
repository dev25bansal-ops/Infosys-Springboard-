"""ADV-02: post-hoc score calibration + an honest false-positive bound.

The operating TCN emits a sigmoid score, but raw sigmoids are rarely
well-calibrated probabilities. This module:
  - temperature-scales the scores to minimize Brier/ECE on a calibration set,
  - reports expected calibration error (a reliability summary),
  - computes a CONFORMAL false-positive threshold: the score above which at most
    an alpha fraction of non-crash calibration samples score (with a finite-sample
    leave-one-out margin), giving an honest ">= alpha of alerts are noise" bound.

No retraining — calibration is post-hoc, on a set that is NOT the eval day.
"""
from __future__ import annotations

import numpy as np


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def temperature_scale(scores: np.ndarray, labels: np.ndarray,
                      lo: float = 0.1, hi: float = 10.0) -> float:
    """Find T minimizing the Brier score of sigmoid(logit(s)/T) vs labels."""
    from scipy.optimize import minimize_scalar
    logits = _logit(scores)
    y = np.asarray(labels, dtype=np.float64)

    def brier(T: float) -> float:
        cal = 1.0 / (1.0 + np.exp(-logits / T))
        return float(np.mean((cal - y) ** 2))

    res = minimize_scalar(brier, bounds=(lo, hi), method="bounded")
    return float(res.x)


def calibrate_probs(scores: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature: p' = sigmoid(logit(p) / T)."""
    return 1.0 / (1.0 + np.exp(-_logit(scores) / T))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray,
                               bins: int = 10) -> float:
    """Expected Calibration Error: |mean(pred) - acc| per bin, weighted by size."""
    p = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = max(1, len(p))
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def conformal_fp_threshold(neg_scores: np.ndarray, alpha: float) -> float:
    """A threshold that bounds the false-positive rate at <= alpha.

    Uses a conformal (split / leave-one-out) quantile over the non-crash
    calibration scores so the guarantee holds with a finite-sample margin:
        P(new non-crash score >= threshold) <= alpha + 1/(n+1)  (for iid).
    """
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def calibrate(scores: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> dict:
    """One-shot: fit temperature + return ECE (before/after) + conformal FP bound."""
    t = temperature_scale(scores, labels)
    cal = calibrate_probs(scores, t)
    neg = np.asarray(scores)[np.asarray(labels) == 0]
    return {
        "temperature": round(t, 3),
        "ece_raw": round(expected_calibration_error(scores, labels), 4),
        "ece_calibrated": round(expected_calibration_error(cal, labels), 4),
        "conformal_fp_threshold": round(conformal_fp_threshold(neg, alpha), 4),
        "alpha": alpha,
    }