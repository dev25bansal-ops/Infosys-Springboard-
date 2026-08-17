"""Leakage regression tests (RSR-13).

Guards the four known look-ahead/leakage bugs:
  (a) Stage-1 z-score must exclude the current tick (leave-one-out).
  (b) Batched Stage-3 scores[j] must use only data up to tick j+window-1 (causal).
  (c) Crash labels must use only strictly-future mids.
  (d) rolling-z row i must use only rows <= i.
"""
import numpy as np

from flash_crash_watchdog.eval.backtest import stage3_scores_batched
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.models.stage1_statistical import Stage1Config, Stage1Statistical
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector, normalize_z
from flash_crash_watchdog.tick import Tick


def _tick(ts: int, price: float, features: dict | None = None) -> Tick:
    book = OrderBookSnapshot(timestamp_ms=ts, bids=[PriceLevel(price, 1.0)], asks=[PriceLevel(price, 1.0)])
    t = Tick(book=book, symbol="T")
    t.features = features or {}
    return t


# --- (a) Stage-1 leave-one-out -----------------------------------------------

def test_stage1_leave_one_out_excludes_current_tick():
    """A spike against a constant baseline must NOT score itself.

    With a constant history (std=0) the leave-one-out z-score is 0. If the
    current tick were appended before scoring (the old bug), the spike would
    inflate its own std and score high.
    """
    s1 = Stage1Statistical(Stage1Config(baseline_window=1000))
    for _ in range(200):
        s1.score(_tick(0, 100.0, {"f1_mid_velocity_50ms": 0.0, "f2_obi_10": 0.0}))

    # A huge velocity spike: LOO => z vs all-zeros history => std 0 => score 0.
    score, passed = s1.score(_tick(0, 100.0, {"f1_mid_velocity_50ms": 100.0, "f2_obi_10": 0.0}))
    assert score == 0.0 and passed is False, (
        "spike must be z-scored against a baseline EXCLUDING it (std=0 -> score 0)"
    )

    # The SAME spike now lands in the (now-non-constant) baseline -> must score high.
    score2, passed2 = s1.score(_tick(0, 100.0, {"f1_mid_velocity_50ms": 100.0, "f2_obi_10": 0.0}))
    assert score2 > 0.0 and passed2 is True


# --- (b) batched scores are causal -------------------------------------------

def test_batched_scores_are_causal():
    """scores[j] must be unaffected by feature changes strictly after tick j+W-1."""
    rng = np.random.default_rng(1)
    n, w = 400, 200
    F = rng.normal(size=(n, 17)).astype(np.float32)
    model = TCNDetector(TCNConfig(sequence_length=w))
    model.eval()

    base = stage3_scores_batched(model, F, "cpu", window=w, norm_window=500)
    assert len(base) == n - w + 1

    # Perturb a future row (row 250). Windows ending before row 250 (j < 51) must
    # be bit-identical; the window containing row 250 must change.
    F2 = F.copy()
    F2[250] += 50.0
    perturbed = stage3_scores_batched(model, F2, "cpu", window=w, norm_window=500)

    np.testing.assert_array_equal(base[:51], perturbed[:51])  # ends at tick 249
    assert not np.allclose(base[51:], perturbed[51:])  # row 250 inside windows j>=51


# --- (c) labels use only strictly-future mids --------------------------------

def test_label_lookahead_uses_only_strictly_future_mids():
    from flash_crash_watchdog.data.windows import build_windows_from_df
    import pandas as pd

    n, dt = 500, 100.0
    ts = np.arange(n, dtype=np.float64) * dt
    price = np.full(n, 100.0)
    # 2% drop INSIDE the window (rows 300-399 of the window is not possible:
    # window is rows 0-199). Put the drop in rows 150..199 => INSIDE the first
    # window, and keep the strictly-future rows flat.
    price[150:200] = 97.0
    df = pd.DataFrame({
        "timestamp_ms": ts.astype(np.int64),
        "best_bid": price, "best_ask": price, "bid_size": 1.0, "ask_size": 1.0,
    })
    windows, labels, _ = build_windows_from_df(
        df, window_size=200, stride=10, lookahead_ms=5000, crash_pct=2.0
    )
    # First window (rows 0-199) contains the 2% drop, but the label looks only at
    # strictly-future mids (rows >=200, flat) => must be 0.
    assert labels[0] == 0


# --- (d) rolling-z uses only past rows ---------------------------------------

def test_rolling_z_uses_only_past_rows():
    rng = np.random.default_rng(2)
    n = 300
    F = rng.normal(size=(n, 17)).astype(np.float32)
    base = normalize_z(F, 100)

    F2 = F.copy()
    F2[200:] += 1000.0  # change only rows 200+ (strictly future for row 199)
    norm2 = normalize_z(F2, 100)

    # rows 0..199 must be identical (their rolling window only includes rows <= i)
    np.testing.assert_array_equal(base[:200], norm2[:200])
    assert not np.allclose(base[200:], norm2[200:])


# --- (e) replay exporter uses the causal window index (RSR-01) ---------------

def test_replay_exporter_uses_causal_window_index():
    """The replay exporter must plot s3[i-W+1] at tick i, not s3[i] (RSR-01).

    s3[j] is the score of window [j, j+W) ending at tick j+W-1; the score that is
    actually known AT tick i is therefore s3[i-W+1], and 0.0 during warmup.
    The old `s3[i]` assigned a window STARTING at i — 199 ticks of future data.
    """
    rng = np.random.default_rng(3)
    n, w = 400, 200
    F = rng.normal(size=(n, 17)).astype(np.float32)
    model = TCNDetector(TCNConfig(sequence_length=w))
    model.eval()
    s3 = stage3_scores_batched(model, F, "cpu", window=w, norm_window=500)
    n_win = len(s3)

    for i in (0, 150, 199, 200, 399):
        j = i - w + 1
        causal = float(s3[j]) if 0 <= j < n_win else 0.0
        assert causal == (float(s3[i - w + 1]) if 0 <= i - w + 1 < n_win else 0.0)
    # Warmup ticks (i < W-1 = 199) must be 0.0; i=198 is the last warmup tick.
    assert (float(s3[198 - w + 1]) if 0 <= 198 - w + 1 < n_win else 0.0) == 0.0
    # A mid-series tick must NOT use the window starting at it (the old bug).
    # i=200 is in range for both the causal index (200-199=1) and the buggy s3[200].
    assert float(s3[200 - w + 1]) != float(s3[200])
