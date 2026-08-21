"""Regression tests guarding session-critical behavior in the detection pipeline.

Each test uses small synthetic inputs and must stay fast (<5s). No network access.
"""
import numpy as np
import pandas as pd
import pytest

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.eval.backtest import run_backtest
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig, TCNDetector
from flash_crash_watchdog.models.stage5_bayesian import Stage5Bayesian
from flash_crash_watchdog.tick import Tick


def _make_tick(ts: int = 0, features: dict | None = None, symbol: str = "BTCUSDT") -> Tick:
    """A normal synthetic tick (balanced book)."""
    return Tick(
        book=OrderBookSnapshot(
            timestamp_ms=ts,
            bids=[PriceLevel(99.5, 1.0)],
            asks=[PriceLevel(100.5, 1.0)],
        ),
        features=features or {},
        symbol=symbol,
    )


# ---------------------------------------------------------------------------
# 1. df_to_ticks — guards the itertuples change
# ---------------------------------------------------------------------------

def test_df_to_ticks_maps_tick_fields():
    """df_to_ticks must map a tiny DataFrame into correct Tick book/trade fields."""
    df = pd.DataFrame([
        {
            "timestamp_ms": 1_000_000,
            "best_bid": 99.5, "best_ask": 100.5,
            "bid_size": 2.0, "ask_size": 3.0,
            "trade_price": 99.7, "trade_size": 1.5, "trade_side": "sell",
        },
        {
            "timestamp_ms": 1_001_000,
            "best_bid": 0, "best_ask": 0,
            "bid_size": 0.0, "ask_size": 0.0,
            "trade_price": float("nan"), "trade_size": 0.0, "trade_side": "buy",
        },
    ])

    ticks = list(df_to_ticks(df, symbol="BTCUSDT"))
    assert len(ticks) == 2

    t0 = ticks[0]
    assert t0.symbol == "BTCUSDT"
    assert t0.timestamp_ms == 1_000_000
    assert t0.book.best_bid == 99.5
    assert t0.book.best_ask == 100.5
    assert t0.book.bids == [PriceLevel(99.5, 2.0)]
    assert t0.book.asks == [PriceLevel(100.5, 3.0)]
    assert len(t0.trades) == 1
    trade = t0.trades[0]
    assert trade.timestamp_ms == 1_000_000
    assert trade.price == 99.7
    assert trade.size == 1.5
    assert trade.side == "sell"

    t1 = ticks[1]
    assert t1.book.bids == []  # best_bid == 0 -> no bid level emitted
    assert t1.book.asks == []
    assert t1.trades == []  # NaN trade_price -> no trade emitted


# ---------------------------------------------------------------------------
# 2. Stage3TCN.feed — contiguous normalized window, capped at sequence_length
# ---------------------------------------------------------------------------

def test_stage3_feed_builds_contiguous_capped_window():
    seq_len = 64
    s3 = Stage3TCN(TCNConfig(sequence_length=seq_len))
    n = 70  # more ticks than the cap
    for i in range(n):
        s3.feed(_make_tick(ts=1000 + i, features={"f1_mid_velocity_50ms": float(i)}))

    assert s3._ticks_processed == n
    # Window must cap at config.sequence_length, never grow unbounded.
    assert len(s3._window) == seq_len
    assert s3._window[0].shape == (17,)  # one entry per STAGE3 feature
    # Window is contiguous (all ticks fed) and normalized (a moving z-score).
    assert all(isinstance(vec, np.ndarray) for vec in s3._window)

    score, should_pass = s3.score_current()
    assert isinstance(score, float)
    assert isinstance(should_pass, bool)
    assert 0.0 <= score <= 1.0

    # Before the cap, the window length exactly equals the number of ticks fed.
    fresh = Stage3TCN(TCNConfig(sequence_length=seq_len))
    for i in range(30):
        fresh.feed(_make_tick(ts=i))
    assert len(fresh._window) == 30
    assert fresh._ticks_processed == 30
    # Below the warmup threshold score_current is a no-op tuple, not a model call.
    assert fresh.score_current() == (0.0, False)


# ---------------------------------------------------------------------------
# 3. TCNDetector.train_on_windows — crash-classifier guard
# ---------------------------------------------------------------------------

def test_tcn_train_requires_positive_labels():
    """Training on labels with no positives must raise (crash-classifier guard)."""
    detector = TCNDetector()
    windows = np.zeros((8, 32, 17), dtype=np.float32)
    labels = np.zeros(8, dtype=np.int64)  # no crash windows at all
    with pytest.raises(ValueError, match="No positive"):
        detector.train_on_windows(windows, labels, epochs=1)


# ---------------------------------------------------------------------------
# 4. DetectionCascade.process_tick — Stage-3 fed on ticks that FAIL Stage-1
# ---------------------------------------------------------------------------

def test_cascade_feeds_stage3_even_when_stage1_rejects():
    """Stage-3 must be advanced on every tick, even those rejected by Stage-1.

    Regression: the TCN is trained on contiguous windows, so if the cascade only
    fed Stage-3 for the ticks that clear the early gates, its window would be
    sparse and its scores would zero out.
    """
    from flash_crash_watchdog.features import FeatureExtractor
    from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical
    from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
    from flash_crash_watchdog.models.stage4_transformer import Stage4Transformer, TransformerConfig

    cascade = DetectionCascade(
        feature_extractor=FeatureExtractor(),
        stage1=Stage1Statistical(),
        stage2=Stage2IsolationForest(),
        stage3=Stage3TCN(TCNConfig(sequence_length=64)),
        stage4=Stage4Transformer(
            TransformerConfig(num_layers=1, num_heads=2, feature_dim=8, num_symbols=4)
        ),
        stage5=Stage5Bayesian(),
    )
    cascade._stage4_enabled = False

    n = 10
    for i in range(n):
        result = cascade.process_tick(_make_tick(ts=1000 + i))
        assert result is None  # warmup ticks fail Stage-1 -> no alert

    assert cascade.stats.ticks_total == n
    assert cascade.stats.alerts_fired == 0
    # The whole point: Stage-3 was fed on every tick even though Stage-1 rejected all.
    assert cascade.s3._ticks_processed == n
    assert len(cascade.s3._window) == n


# ---------------------------------------------------------------------------
# 5. run_backtest — cooldown_ms coalesces alert bursts
# ---------------------------------------------------------------------------

class _NullExtractor:
    """Feature-extractor stub that just installs an empty feature dict."""

    def extract(self, tick):
        tick.features = {}
        return tick.features


class _AlwaysPassStage:
    """A stage that passes every tick with a fixed high score."""

    def __init__(self, score: float = 0.9) -> None:
        self._score = score
        self._ticks_processed = 0
        self._ticks_passed = 0

    def score(self, tick):
        self._ticks_processed += 1
        self._ticks_passed += 1
        return self._score, True


class _AlwaysPassStage3:
    """Stage-3 double: fed every tick, always scores above threshold."""

    def __init__(self, score: float = 0.9) -> None:
        self._score = score
        self._ticks_processed = 0

    def feed(self, tick):
        self._ticks_processed += 1

    def score_current(self):
        return self._score, True


def _alert_every_tick_cascade() -> DetectionCascade:
    """A cascade where every tick fires an alert (deterministic for backtest)."""
    cascade = DetectionCascade(
        feature_extractor=_NullExtractor(),
        stage1=_AlwaysPassStage(),
        stage2=_AlwaysPassStage(),
        stage3=_AlwaysPassStage3(),
        stage4=object(),  # unused: stage 4 disabled below
        stage5=Stage5Bayesian(),
    )
    cascade._stage4_enabled = False
    return cascade


def test_run_backtest_cooldown_coalesces_alerts():
    """cooldown_ms>0 must emit strictly fewer alerts than cooldown_ms=0."""
    n = 40
    df = pd.DataFrame({
        "timestamp_ms": [1_000_000 + i * 1_000 for i in range(n)],
        "best_bid": [100.0] * n,
        "best_ask": [100.1] * n,
        "bid_size": [2.0] * n,
        "ask_size": [3.0] * n,
    })

    without_cooldown = run_backtest(_alert_every_tick_cascade(), df, cooldown_ms=0)
    with_cooldown = run_backtest(_alert_every_tick_cascade(), df, cooldown_ms=5_000)

    assert without_cooldown.alerts_fired == n
    assert with_cooldown.alerts_fired < without_cooldown.alerts_fired
    # 40 ticks spaced 1s apart, 5s cooldown -> one alert per 5s bucket.
    assert with_cooldown.alerts_fired == 8
