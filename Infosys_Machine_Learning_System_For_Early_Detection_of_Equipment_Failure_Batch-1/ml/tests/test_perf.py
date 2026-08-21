"""TST-25: performance / soak measurements (lightweight, CI-safe).

These assert LOOSE lower bounds so they pass on modest CI machines while still
guarding against catastrophic regressions (e.g. a 100x slowdown). The
measured-on-reference numbers (feature extraction ~4400 ticks/s, TCN ~11ms CPU
forward) are documented targets; the assertions use a ~10x safety margin.
"""
import time

import numpy as np
import torch

from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector
from flash_crash_watchdog.tick import Tick


def _ticks(n: int, start: int = 100_000):
    out = []
    for i in range(n):
        p = 100.0 + i
        book = OrderBookSnapshot(timestamp_ms=start + i, bids=[PriceLevel(p - 0.5, 1.0)],
                                 asks=[PriceLevel(p + 0.5, 1.0)])
        out.append(Tick(book=book, symbol="BTC"))
    return out


def test_feature_extraction_throughput():
    """Loose floor on feature-extraction ticks/s (reference ~4400/s)."""
    ex = FeatureExtractor()
    ticks = _ticks(3000)
    t0 = time.perf_counter()
    for t in ticks:
        ex.extract(t)
    dt = time.perf_counter() - t0
    rate = len(ticks) / dt
    assert len(ex.extract(ticks[0])) == 20
    assert rate >= 200, f"feature extraction too slow: {rate:.0f} ticks/s"


def test_tcn_forward_latency():
    """Loose p95 ceiling on a batch TCN forward (reference ~11ms CPU/single)."""
    model = TCNDetector(TCNConfig(sequence_length=200))
    model.eval()
    x = torch.randn(32, 17, 200)
    lat = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model(x)
            lat.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(lat, 95))
    assert p95 < 1000.0, f"TCN batch forward too slow (p95={p95:.0f}ms)"


def test_sustained_feed_memory_bounded():
    """Feeding many ticks into a Stage3TCN keeps its windows bounded (no growth)."""
    from flash_crash_watchdog.models.stage3_tcn import Stage3TCN
    s3 = Stage3TCN(TCNConfig(sequence_length=200))
    names = FEATURE_NAMES[:17]
    for i in range(30_000):
        t = _ticks(1, start=i)[0]
        t.features = {f: float(i % 7) for f in names}
        s3.feed(t)
    assert len(s3._window) <= 200, "window must stay capped at sequence_length"
    assert len(s3._norm_hist) <= s3._norm_window, "normalization history must stay capped"