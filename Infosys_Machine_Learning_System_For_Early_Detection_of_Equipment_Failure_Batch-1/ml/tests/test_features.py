"""Tests for the feature engineering modules."""
import pytest

from flash_crash_watchdog.features import FeatureExtractor, FEATURE_NAMES
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick, Trade


def make_tick(mid: float = 100.0, bid_size: float = 1.0, ask_size: float = 1.0,
              ts: int = 1000, trades=None):
    book = OrderBookSnapshot(
        timestamp_ms=ts,
        bids=[PriceLevel(mid - 0.5, bid_size)],
        asks=[PriceLevel(mid + 0.5, ask_size)],
    )
    return Tick(book=book, trades=trades or [], symbol="TEST")


def test_feature_extractor_returns_20_features():
    extractor = FeatureExtractor()
    tick = make_tick()
    features = extractor.extract(tick)
    assert len(features) == 20


def test_feature_names_complete():
    assert len(FEATURE_NAMES) == 20
    # Check all 5 families are present
    families = {name.split("_")[0] for name in FEATURE_NAMES}
    assert families == {"f1", "f2", "f3", "f4", "f5"}


def test_price_action_features():
    from flash_crash_watchdog.features.price_action import PriceActionFeatures
    f1 = PriceActionFeatures()
    # First tick
    f1.update(make_tick(mid=100.0, ts=1000))
    # Second tick — price moved up
    f1.update(make_tick(mid=100.5, ts=1050))
    features = f1.update(make_tick(mid=101.0, ts=1100))
    # Velocity should be positive (price went up)
    assert features["f1_mid_velocity_50ms"] > 0


def test_depth_imbalance_features():
    from flash_crash_watchdog.features.depth_imbalance import DepthImbalanceFeatures
    f2 = DepthImbalanceFeatures()
    # Imbalanced book: heavy bid side
    tick = make_tick(bid_size=10.0, ask_size=1.0)
    features = f2.update(tick)
    assert features["f2_obi_10"] > 0.5  # bullish imbalance
    assert features["f2_bid_depth_10"] == 10.0
    assert features["f2_ask_depth_10"] == 1.0


def test_volatility_features():
    from flash_crash_watchdog.features.volatility import VolatilityFeatures
    f4 = VolatilityFeatures()
    # Generate some price movement
    for i in range(20):
        f4.update(make_tick(mid=100.0 + i * 0.01, ts=1000 + i * 10))
    features = f4.update(make_tick(mid=100.2, ts=1200))
    assert features["f4_realized_vol_1s"] >= 0
    assert 0 < features["f4_variance_ratio"] <= 5  # should be finite


def test_cross_symbol_reference_stores_log_returns():
    """ENH-02: the F5 reference history must store log returns, not raw mids —
    otherwise cross-symbol correlation/cointegration compare mismatched units."""
    import math
    from flash_crash_watchdog.features.cross_symbol import CrossSymbolFeatures
    f5 = CrossSymbolFeatures()
    # self ticks: 100 -> 110 -> 121 (log returns)
    f5.update(make_tick(mid=100.0, ts=0))
    f5.update(make_tick(mid=110.0, ts=100))
    f5.update(make_tick(mid=121.0, ts=200))
    # reference mids: same pattern
    f5.update_reference("REF", 0, 100.0)
    f5.update_reference("REF", 100, 110.0)
    f5.update_reference("REF", 200, 121.0)

    ref_rets = [s.ret for s in f5._ref_history["REF"]]
    assert abs(ref_rets[0]) < 1e-9, "first reference return must be 0 (no prior mid)"
    assert abs(ref_rets[1] - math.log(110.0 / 100.0)) < 1e-9
    assert abs(ref_rets[2] - math.log(121.0 / 110.0)) < 1e-9
    assert ref_rets[2] != 121.0, "reference must store returns, not the raw mid"

    # With consistent units, correlation with self is well-defined and high.
    feats = f5.update(make_tick(mid=121.0, ts=300))
    if feats["f5_pairwise_correlation"] != 0.0:
        assert feats["f5_pairwise_correlation"] > 0.99
