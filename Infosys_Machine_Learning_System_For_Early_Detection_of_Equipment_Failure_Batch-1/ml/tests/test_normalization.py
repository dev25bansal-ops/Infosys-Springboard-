"""Consistency test for the shared rolling-z normalization (BUG-03).

The offline backtest transform `normalize_z` MUST produce the same per-tick
normalized vectors as the online streaming `Stage3TCN._normalize`, so that
offline scores transfer to the live Stage3TCN.feed path.
"""
import numpy as np

from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.models.stage3_tcn import (
    STAGE3_FEATURES,
    Stage3TCN,
    TCNConfig,
    normalize_z,
)
from flash_crash_watchdog.tick import Tick


def _tick_from_features(fvec: np.ndarray, ts: int) -> Tick:
    book = OrderBookSnapshot(
        timestamp_ms=ts, bids=[PriceLevel(100.0, 1.0)], asks=[PriceLevel(100.0, 1.0)]
    )
    tick = Tick(book=book, symbol="T")
    tick.features = {name: float(v) for name, v in zip(STAGE3_FEATURES, fvec)}
    return tick


def test_offline_normalize_matches_streaming_stage3():
    rng = np.random.default_rng(0)
    n = 800
    F = rng.normal(size=(n, 17)).astype(np.float32)
    F[:, 5] = 42.0  # a constant feature: must map to 0 in both paths

    norm = normalize_z(F, 500)  # shared offline transform

    s3 = Stage3TCN(TCNConfig(sequence_length=500))
    streamed = np.zeros_like(norm)
    for i in range(n):
        s3.feed(_tick_from_features(F[i], i))
        streamed[i] = np.array(s3._window[-1], dtype=np.float32)

    np.testing.assert_allclose(streamed, norm, atol=1e-4)
    # constant feature maps to 0 in both
    np.testing.assert_allclose(streamed[:, 5], 0.0, atol=1e-6)
    np.testing.assert_allclose(norm[:, 5], 0.0, atol=1e-6)


def test_normalize_z_matches_hand_computed_rolling_z():
    """Sanity: normalize_z(window=3) equals a hand-rolled rolling z-score."""
    f = np.arange(10, dtype=np.float64).reshape(10, 1)
    out = normalize_z(f.astype(np.float32), 3).squeeze()
    # row i is z-scored against rows [max(0,i-2), i] with sample std (ddof=1);
    # a single-sample window (std=None) maps to 0 in normalize_z.
    expected = np.zeros(10)
    for i in range(10):
        seg = f[max(0, i - 2): i + 1, 0]
        if len(seg) < 2:
            expected[i] = 0.0
        else:
            expected[i] = (seg[-1] - seg.mean()) / seg.std(ddof=1)
    np.testing.assert_allclose(out, expected, atol=1e-5)