"""Regression tests for wall-clock crash labels (RSR-02).

The crash label horizon must be a *real time* window (next lookahead_ms), not a
fixed tick count. The old `lookahead_ms // 100` heuristic assumed 10 ticks/s,
which mislabeled fast crash days (real tick rate 42/s) and slow normal days (8/s).
"""
import numpy as np
import pandas as pd

from flash_crash_watchdog.data.windows import build_windows_from_df


def _flat_frame(n: int, dt_ms: float, drop_from_idx: int = -1, drop_price: float = 97.0):
    """Detector-schema frame: mid price = 100.0, optionally dropping later."""
    timestamps = np.arange(n, dtype=np.float64) * dt_ms
    price = np.full(n, 100.0)
    if drop_from_idx >= 0:
        price[drop_from_idx:] = drop_price
    return pd.DataFrame(
        {
            "timestamp_ms": timestamps.astype(np.int64),
            "best_bid": price,
            "best_ask": price,
            "bid_size": 1.0,
            "ask_size": 1.0,
        }
    )


def test_wallclock_label_catches_drop_beyond_old_1s_horizon():
    """Fast ticks (20ms = 50/s): a 3% drop ~3s after the window is inside 5s.

    Old heuristic (lookahead_ms // 100 = 50 ticks = 1s) missed it -> label 0.
    Wall-clock (5s) catches it -> label 1.
    """
    n, dt_ms = 500, 20.0
    df = _flat_frame(n, dt_ms, drop_from_idx=300)  # t=6.0s, within 5s of window end (3.98s)
    windows, labels, _ = build_windows_from_df(
        df, window_size=200, stride=10, lookahead_ms=5000, crash_pct=2.0
    )
    assert windows.shape[1] == 200
    assert labels.shape == (len(windows),)
    # First window (ticks 0-199) ends at t=3980ms; horizon [3980, 8980]ms covers
    # the drop at tick 300 (t=6000ms).
    assert labels[0] == 1, "wall-clock label must catch a drop at ~3s within a 5s horizon"


def test_legacy_tick_heuristic_still_misses_late_drop():
    """Without a timestamp column the old heuristic is used and misses the late drop."""
    n, dt_ms = 500, 20.0
    df = _flat_frame(n, dt_ms, drop_from_idx=300).drop(columns=["timestamp_ms"])
    windows, labels, _ = build_windows_from_df(
        df, window_size=200, stride=10, lookahead_ms=5000, crash_pct=2.0
    )
    assert labels[0] == 0, "legacy tick-count lookahead must not see the ~3s drop"


def test_wallclock_label_excludes_drop_outside_horizon():
    """Slow ticks (125ms = 8/s): a 3% drop ~5.6s after the window is OUTSIDE 5s.

    Old heuristic (50 ticks = 6.25s) saw it -> label 1 (false positive).
    Wall-clock (5s) must reject it -> label 0.
    """
    n, dt_ms = 500, 125.0
    # Window 0 ends at tick 199, t=24875ms. 5s horizon -> t<=29875ms -> tick 239.
    # Legacy 50-tick horizon -> tick 249. Drop at tick 244 (t=30500ms, 5.6s out).
    df = _flat_frame(n, dt_ms, drop_from_idx=244, drop_price=97.0)
    windows, labels, _ = build_windows_from_df(
        df, window_size=200, stride=10, lookahead_ms=5000, crash_pct=2.0
    )
    assert labels[0] == 0, "wall-clock label must reject a drop beyond the 5s horizon"


def test_wallclock_label_requires_full_horizon_coverage():
    """Windows whose lookahead horizon is not fully covered by data are skipped."""
    n, dt_ms = 230, 20.0  # window(200) + only 30 future ticks = 600ms < 5s
    df = _flat_frame(n, dt_ms, drop_from_idx=205, drop_price=97.0)
    windows, labels, _ = build_windows_from_df(
        df, window_size=200, stride=10, lookahead_ms=5000, crash_pct=2.0
    )
    # Only windows with a full 5s horizon are emitted; with 30 future ticks none qualify.
    assert len(windows) == 0
