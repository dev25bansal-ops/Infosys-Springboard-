"""Regression tests for the event-based crash labeler (RSR-03).

The old `label_crashes` reset the running peak to the current trough at every
detection, chain-splitting one continuous descent into many overlapping
"crashes". The fix emits ONE label per distinct peak-to-trough descent event,
resolved by a minimum recovery from the trough.
"""
import numpy as np

from flash_crash_watchdog.data.labels import label_crashes
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick


def _ticks(prices: np.ndarray, dt_ms: float = 100.0, start_ms: int = 0):
    ticks = []
    for i, p in enumerate(prices):
        ts = start_ms + int(i * dt_ms)
        book = OrderBookSnapshot(timestamp_ms=ts, bids=[PriceLevel(p, 1.0)], asks=[PriceLevel(p, 1.0)])
        ticks.append(Tick(book=book, symbol="T"))
    return ticks


def test_one_continuous_descent_is_one_event():
    """A single 100->90 drop must produce exactly ONE crash (was ~5+)."""
    prices = np.concatenate([
        np.full(50, 100.0),          # calm
        np.linspace(100.0, 90.0, 100),  # the whole descent
        np.full(50, 90.0),           # flat at trough
    ])
    crashes = label_crashes(_ticks(prices), drop_threshold_pct=2.0, window_ms=60_000)
    assert len(crashes) == 1, f"expected 1 event, got {len(crashes)}"
    c = crashes[0]
    assert c.peak_price == 100.0
    assert c.trough_price == 90.0
    assert abs(c.drop_pct - 10.0) < 1e-6
    # Peak is the first tick (t=0); trough is the first 90.0 at index 149 (t=14900ms).
    assert c.duration_ms == int(np.argmax(prices == 90.0) * 100.0)


def test_two_descents_with_recovery_produce_two_events():
    """100 ->97 ->99 (recovery) ->95 ->97 (recovery) = two distinct events."""
    prices = np.array(
        [100.0] * 40
        + [97.0] * 30          # descent 1
        + [99.0] * 30          # recovery (>1% up from 97)
        + [95.0] * 30          # descent 2
        + [97.0] * 30          # recovery
    )
    crashes = label_crashes(_ticks(prices), drop_threshold_pct=2.0, window_ms=60_000)
    assert len(crashes) == 2, f"expected 2 events, got {len(crashes)}"
    assert crashes[0].peak_price == 100.0 and crashes[0].trough_price == 97.0
    assert crashes[1].peak_price == 99.0 and crashes[1].trough_price == 95.0


def test_drop_below_threshold_is_not_a_crash():
    prices = np.array([100.0] * 40 + [99.0] * 40 + [100.0] * 40)
    crashes = label_crashes(_ticks(prices), drop_threshold_pct=2.0, window_ms=60_000)
    assert crashes == []


def test_pre_crash_margin_extends_start_ts():
    """pre_crash_ms pushes start_ts back from the peak (early-warning margin)."""
    prices = np.concatenate([
        np.linspace(90.0, 100.0, 60),  # rise -> peak at tick 59 (t=5900ms)
        np.linspace(100.0, 95.0, 50),  # descent to 95
        np.full(30, 95.0),
    ])
    ticks = _ticks(prices, dt_ms=100.0)
    crashes = label_crashes(ticks, drop_threshold_pct=2.0, window_ms=60_000, pre_crash_ms=3000)
    assert len(crashes) == 1
    assert crashes[0].start_ts == 5900 - 3000
    # default margin keeps the start exactly at the peak
    c0 = label_crashes(ticks, drop_threshold_pct=2.0, window_ms=60_000)[0]
    assert c0.start_ts == 5900
