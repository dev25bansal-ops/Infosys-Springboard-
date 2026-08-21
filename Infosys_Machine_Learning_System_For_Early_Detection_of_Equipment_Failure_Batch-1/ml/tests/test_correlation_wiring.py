"""ENH-03: the validated CorrelationBreakdown wired into the Python cascade.

The cascade already feeds the anchor (tick symbol) into ``corr`` per tick; the
missing piece was a way to feed the reference/basket symbols. update_reference
closes that so the Python path can exercise the real Stage-4 correlation signal.
"""
from flash_crash_watchdog.features import FeatureExtractor
from flash_crash_watchdog.features.correlation import CorrelationBreakdown
from flash_crash_watchdog.models.cascade import DetectionCascade
from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig
from flash_crash_watchdog.models.stage4_transformer import Stage4Transformer, TransformerConfig
from flash_crash_watchdog.models.stage5_bayesian import Stage5Bayesian


def _cascade(corr=None) -> DetectionCascade:
    return DetectionCascade(
        feature_extractor=FeatureExtractor(),
        stage1=Stage1Statistical(),
        stage2=Stage2IsolationForest(),
        stage3=Stage3TCN(TCNConfig()),
        stage4=Stage4Transformer(TransformerConfig()),
        stage5=Stage5Bayesian(),
        corr=corr,
    )


def test_update_reference_forwards_to_correlation():
    corr = CorrelationBreakdown()
    c = _cascade(corr=corr)
    for i in range(30):
        c.update_reference("ETHUSDT", 100.0 + i, i)
    assert "ETHUSDT" in corr._returns
    assert len(corr._returns["ETHUSDT"]) >= 29
    # the anchor can also be fed through the reference API
    for i in range(30, 60):
        c.update_reference("BTCUSDT", 100.0 + i, i)
    assert len(corr._returns.get("BTCUSDT", [])) >= 29


def test_update_reference_is_noop_without_corr():
    c = _cascade()  # corr=None by default
    assert c.corr is None
    c.update_reference("ETHUSDT", 100.0, 0)  # must not raise


def test_correlation_evaluate_is_reachable_after_reference_feed():
    """Once anchor+basket returns exist, the correlation emits a bounded score."""
    corr = CorrelationBreakdown()  # defaults: 120-window/300-warmup too big for a test
    # use a small-config detector to reach 'ready' quickly
    from flash_crash_watchdog.features.correlation import CorrelationConfig
    corr = CorrelationBreakdown(CorrelationConfig(
        anchor='BTCUSDT', corr_window_bins=10, baseline_bins=50, warmup_bins=40
    ))
    c = _cascade(corr=corr)
    # alternate anchor + reference mid observations so returns build on both sides
    for i in range(200):
        c.update_reference("BTCUSDT", 100.0 + 0.001 * i, i)
        c.update_reference("ETHUSDT", 100.0 + 0.001 * i, i)
    c.update_reference("ETHUSDT", 101.0, 1000)  # a decoupling tick
    z, score, fire = corr.evaluate()
    assert 0.0 <= score <= 1.0
    assert z == z  # not NaN

def test_pair_correlations_and_decoupling():
    """ADV-04: per-pair correlation + which basket symbols decouple."""
    import numpy as np
    from flash_crash_watchdog.features.correlation import CorrelationConfig, CorrelationBreakdown
    cfg = CorrelationConfig(anchor='BTCUSDT', corr_window_bins=20, floor_corr=0.2)
    corr = CorrelationBreakdown(cfg)
    rng = np.random.default_rng(1)
    n = 60
    for i in range(n):
        d = rng.normal(0, 0.05)   # shared macro "shock"
        corr.update('BTCUSDT', 100 + i * 0.05 + d, i)
        corr.update('ETHUSDT', 100 + i * 0.05 + d, i)  # same drift + same shock
        corr.update('ALGO', 100 - i * 0.05 - d, i)      # opposite shock -> decouples
    pairs = corr.pair_correlations()
    assert set(pairs) == {'ETHUSDT', 'ALGO'}
    assert pairs['ETHUSDT'] > 0.8
    assert pairs['ALGO'] < 0.2
    dec = corr.decoupling_symbols()
    assert 'ALGO' in dec and 'ETHUSDT' not in dec
