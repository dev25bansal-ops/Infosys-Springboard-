"""Tests for the 5-stage detection cascade."""
import pytest

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick


def make_normal_tick(ts: int = 1000) -> Tick:
    """A tick that should NOT trigger an alert."""
    return Tick(
        book=OrderBookSnapshot(
            timestamp_ms=ts,
            bids=[PriceLevel(99.5, 1.0)],
            asks=[PriceLevel(100.5, 1.0)],
        ),
        symbol="BTCUSDT",
    )


def make_anomalous_tick(ts: int = 1000) -> Tick:
    """A tick with extreme OBI that SHOULD trigger Stage 1."""
    return Tick(
        book=OrderBookSnapshot(
            timestamp_ms=ts,
            bids=[PriceLevel(99.5, 50.0)],  # huge bid-side imbalance
            asks=[PriceLevel(100.5, 0.1)],
        ),
        symbol="BTCUSDT",
    )


def test_stage1_rejects_normal_ticks():
    """Stage 1 should reject normal ticks quickly."""
    from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical
    s1 = Stage1Statistical()
    # Feed a lot of normal ticks to build baseline
    for i in range(200):
        score, passed = s1.score(make_normal_tick(ts=1000 + i))
    # The latest tick should have a low score
    assert score < 0.5
    assert not passed


def test_cascade_processes_tick(pipeline_config):
    """The full cascade should process a tick without error."""
    cascade = DetectionCascade.from_config(pipeline_config)
    # Process many normal ticks
    for i in range(100):
        result = cascade.process_tick(make_normal_tick(ts=1000 + i))
        assert result is None  # no alert for normal tick
    assert cascade.stats.ticks_total == 100


def test_cascade_stats(pipeline_config):
    """Cascade stats should be tracked correctly."""
    cascade = DetectionCascade.from_config(pipeline_config)
    for i in range(50):
        cascade.process_tick(make_normal_tick(ts=1000 + i))
    stats = cascade.stats
    assert stats.ticks_total == 50
    assert stats.total_latency_ms > 0


def test_pipeline_config_stage4_disabled(pipeline_config):
    """Hermetic pin (TST-02): the shipped config disables Stage-4, so the live
    cascade is S1->S2->S3->S5 — docs/UI claiming a live 5-stage Transformer
    contradict the actual operating config (ARCH-01)."""
    import yaml
    with open(pipeline_config) as f:
        cfg = yaml.safe_load(f)
    assert cfg["stage4"]["enabled"] is False
