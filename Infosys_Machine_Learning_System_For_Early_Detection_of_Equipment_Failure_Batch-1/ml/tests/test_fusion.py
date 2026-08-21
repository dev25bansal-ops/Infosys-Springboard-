"""Stage-5 fusion math tests (TST-05)."""
import math

from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.models.stage5_bayesian import Stage5Bayesian, Stage5Config
from flash_crash_watchdog.tick import Tick


def _tick(ts=1000, symbol="BTCUSDT", features=None):
    book = OrderBookSnapshot(timestamp_ms=ts, bids=[PriceLevel(99.0, 1.0)], asks=[PriceLevel(101.0, 1.0)])
    t = Tick(book=book, symbol=symbol, features=features or {})
    return t


def _posterior(s2, s3, s4):
    lo2 = math.log(max(1e-6, min(1 - 1e-6, s2)) / (1 - max(1e-6, min(1 - 1e-6, s2)))) if s2 == s2 and s2 is not None else math.log(0.5 / 0.5)
    # simpler: rely on the model
    cfg = Stage5Config(alert_threshold=0.7)
    s5 = Stage5Bayesian(cfg)
    a = s5.aggregate(_tick(), s2, s3, s4)
    return a.posterior if a else None


def test_high_scores_fire_alert():
    s5 = Stage5Bayesian(Stage5Config(alert_threshold=0.7))
    alert = s5.aggregate(_tick(), 1.0, 1.0, 1.0)
    assert alert is not None
    assert alert.posterior > 0.7
    assert alert.symbol == "BTCUSDT"
    assert s5.alerts_fired == 1


def test_low_scores_no_alert():
    s5 = Stage5Bayesian(Stage5Config(alert_threshold=0.7))
    assert s5.aggregate(_tick(), 0.0, 0.0, 0.0) is None


def test_fusion_monotonic_in_stage3():
    cfg = Stage5Config(alert_threshold=0.3)  # low enough that both configs fire
    s5 = Stage5Bayesian(cfg)
    a_low = s5.aggregate(_tick(), 0.5, 0.6, 0.5)
    s5b = Stage5Bayesian(cfg)
    a_high = s5b.aggregate(_tick(), 0.5, 0.95, 0.5)
    assert a_low is not None and a_high is not None
    assert a_high.posterior > a_low.posterior


def test_nan_score_is_neutral_not_suppressive():
    """NaN in one stage maps to neutral 0.5 and must not suppress the alert."""
    s5 = Stage5Bayesian(Stage5Config(alert_threshold=0.7))
    a = s5.aggregate(_tick(), 1.0, float('nan'), 1.0)
    assert a is not None, "a NaN stage score must not suppress a high-confidence alert"


def test_scores_are_clamped_to_open_interval():
    # a score of 10 is clamped to ~1-1e-6, producing a finite posterior
    s5 = Stage5Bayesian(Stage5Config(alert_threshold=0.7))
    a = s5.aggregate(_tick(), 10.0, 10.0, 10.0)
    assert a is not None and math.isfinite(a.posterior)
    # a score of -5 is clamped to ~1e-6, holding posterior near zero
    a2 = s5.aggregate(_tick(), -5.0, -5.0, -5.0)
    assert a2 is None or a2.posterior < 0.7