"""Post-hoc crash labeling for backtesting."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)


@dataclass
class CrashLabel:
    start_ts: int
    end_ts: int
    peak_price: float
    trough_price: float
    drop_pct: float
    duration_ms: int


def label_crashes(
    ticks: List[Tick],
    drop_threshold_pct: float = 2.0,
    window_ms: int = 60_000,
    recovery_pct: float = 1.0,
    pre_crash_ms: int = 0,
) -> List[CrashLabel]:
    """Decompose the mid-price path into distinct descent *events* (RSR-03).

    One continuous descent produces ONE label: a swing is detected from a running
    peak, the trough is tracked while the price stays below the peak by >=
    drop_threshold_pct, and the event is emitted when the price recovers >=
    recovery_pct from the trough (or fully back to the peak). The old
    implementation reset the peak to the current trough at every detection, so a
    single continuous drop was chain-split into hundreds of overlapping
    "crashes" (e.g. 507-718 for one day), making recall denominators
    slice-dependent.

    Args:
        ticks: chronological tick stream.
        drop_threshold_pct: min drop from the swing peak to count as a crash.
        window_ms: a swing peak expires (and the baseline rolls forward to the
            current mid) if no >=threshold drop occurs within this wall-clock
            window. Bounds "a crash" to a drop from a *recent* peak.
        recovery_pct: how much the price must recover from the trough (percent of
            trough price) to end the event. Larger values merge shallow bounces
            into one descent.
        pre_crash_ms: extend start_ts back from the peak by this much (early-
            warning margin), clamped to the first tick. 0 = event starts exactly
            at the peak.

    Returns:
        One CrashLabel per distinct descent event, in chronological order.
    """
    if not ticks:
        return []

    first_ts = ticks[0].timestamp_ms
    crashes: List[CrashLabel] = []

    swing_peak = 0.0
    swing_peak_ts = first_ts
    in_crash = False
    trough = 0.0
    trough_ts = 0

    def emit() -> None:
        nonlocal in_crash
        if swing_peak > 0 and trough > 0:
            drop_pct = (swing_peak - trough) / swing_peak * 100.0
            crashes.append(CrashLabel(
                start_ts=max(first_ts, swing_peak_ts - pre_crash_ms),
                end_ts=trough_ts,
                peak_price=swing_peak,
                trough_price=trough,
                drop_pct=drop_pct,
                duration_ms=trough_ts - swing_peak_ts,
            ))
        in_crash = False

    for tick in ticks:
        mid = tick.book.mid_price
        if mid is None or mid <= 0:
            continue
        ts = tick.timestamp_ms

        if not in_crash:
            if mid > swing_peak:
                swing_peak = mid
                swing_peak_ts = ts
            elif ts - swing_peak_ts > window_ms:
                # No >=threshold drop since this peak: it's stale, roll baseline up.
                swing_peak = mid
                swing_peak_ts = ts
            if swing_peak > 0:
                drop_pct = (swing_peak - mid) / swing_peak * 100.0
                if drop_pct >= drop_threshold_pct:
                    in_crash = True
                    trough = mid
                    trough_ts = ts
        else:
            if mid < trough:
                trough = mid
                trough_ts = ts
            recovered = (mid >= swing_peak) or (
                trough > 0 and (mid - trough) / trough * 100.0 >= recovery_pct
            )
            if recovered:
                emit()
                swing_peak = mid
                swing_peak_ts = ts

    if in_crash:
        emit()  # descent unresolved by end of data -> still label it

    logger.info("Found %d crash events (threshold=%.1f%%, window=%dms, recovery=%.1f%%)",
                len(crashes), drop_threshold_pct, window_ms, recovery_pct)
    return crashes
