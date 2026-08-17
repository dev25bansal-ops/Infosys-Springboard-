"""ADV-02: score calibration + conformal FP bound tests."""
import numpy as np

from flash_crash_watchdog.eval.calibration import (
    calibrate,
    conformal_fp_threshold,
    expected_calibration_error,
    temperature_scale,
)


def test_temperature_stays_near_one_for_calibrated():
    rng = np.random.default_rng(0)
    # well-calibrated: label ~ Bernoulli(score)
    scores = rng.uniform(0.05, 0.95, 400)
    labels = (rng.random(400) < scores).astype(np.int64)
    t = temperature_scale(scores, labels)
    assert 0.2 < t < 5.0  # roughly calibrated -> T ~ 1, bounded fit should be stable


def test_conformal_threshold_bounds_fp_rate():
    rng = np.random.default_rng(1)
    neg = rng.beta(2, 5, 2000)  # non-crash scores, mostly low
    alpha = 0.05
    thr = conformal_fp_threshold(neg, alpha)
    emp = float(np.mean(neg > thr))
    assert emp <= alpha + 1 / len(neg) + 1e-6, f"empirical FP {emp} above conformal bound"
    assert thr > 0.0


def test_ece_zero_when_perfectly_calibrated():
    # a binned 'perfect' calibration: score equals the empirical rate in each bin
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.0, 1.0, 1000)
    labels = (rng.random(1000) < scores).astype(np.int64)
    # ECE measures |mean(pred)-acc| per bin; for matched Bernoulli it is small
    ece = expected_calibration_error(scores, labels)
    assert 0.0 <= ece < 0.05


def test_calibrate_returns_report():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.05, 0.95, 500)
    labels = (rng.random(500) < scores).astype(np.int64)
    r = calibrate(scores, labels, alpha=0.05)
    for k in ("temperature", "ece_raw", "ece_calibrated", "conformal_fp_threshold", "alpha"):
        assert k in r
    assert 0.0 <= r["ece_calibrated"] <= 1.0