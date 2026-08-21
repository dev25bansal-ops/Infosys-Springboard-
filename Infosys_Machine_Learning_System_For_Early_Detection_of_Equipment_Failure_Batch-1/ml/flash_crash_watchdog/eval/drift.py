"""ADV-03: feature-drift / Population Stability Index (advisory only).

PSI quantifies how much a feature's live distribution has shifted vs a reference
(training-time) distribution. Interpretations are advisory — ADVISORY-ONLY, never
auto-tuning: the operating point is a validated static configuration, and a drift
banner is a signal to re-validate, not to silently change thresholds.
"""
from __future__ import annotations

import numpy as np

# Conventional PSI bands.
STABLE = 0.1
MODERATE = 0.25


def psi(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.

    Bin the reference into quantile breaks; compare the observed percentage in
    each bin.  PSI = sum( (obs_pct - ref_pct) * ln(obs_pct / ref_pct) ).
    """
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.size == 0 or obs.size == 0:
        return 0.0
    breaks = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    ref_hist, _ = np.histogram(ref, breaks)
    obs_hist, _ = np.histogram(obs, breaks)
    ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
    obs_pct = np.clip(obs_hist / obs_hist.sum(), 1e-6, None)
    return float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))


def drift_flags(reference: np.ndarray, observed: np.ndarray,
                feature_names: list[str], threshold: float = MODERATE,
                bins: int = 10) -> list[dict]:
    """Per-feature PSI across all columns. Returns rows for features exceeding
    ``threshold``, each tagged with its band (STABLE/MODERATE/DRIFT).
    """
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    assert ref.ndim == 2 and obs.ndim == 2 and ref.shape[1] == obs.shape[1]
    flags = []
    for i, name in enumerate(feature_names):
        p = psi(ref[:, i], obs[:, i], bins=bins)
        if p > threshold:
            band = "DRIFT" if p >= 0.25 else ("MODERATE" if p >= 0.1 else "STABLE")
            flags.append({"feature": name, "psi": round(p, 4), "band": band})
    flags.sort(key=lambda r: -r["psi"])
    return flags