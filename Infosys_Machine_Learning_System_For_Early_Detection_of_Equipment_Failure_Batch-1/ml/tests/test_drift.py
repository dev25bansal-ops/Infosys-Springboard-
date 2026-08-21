"""ADV-03: PSI / drift telemetry tests."""
import numpy as np

from flash_crash_watchdog.eval.drift import drift_flags, psi


def test_psi_same_distribution_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, x) < 0.05


def test_psi_increases_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 2000)
    obs = rng.normal(2.0, 1.0, 2000)  # mean shifted +2 sigma
    assert psi(ref, obs) > psi(ref, ref) + 0.2
    assert psi(ref, obs) > 0.25  # large shift -> DRIFT band


def test_drift_flags_flags_shifted_column_only():
    rng = np.random.default_rng(2)
    names = ["f1", "f2", "f3"]
    ref = np.column_stack([rng.normal(size=2000) for _ in range(3)])
    obs = ref.copy()
    obs[:, 1] += 3.0  # only feature 1 drifts
    flags = drift_flags(ref, obs, names, threshold=0.25)
    flagged = [f["feature"] for f in flags]
    assert "f2" in flagged, f"shifted feature f2 must be flagged, got {flagged}"
    assert "f1" not in flagged
    assert "f3" not in flagged