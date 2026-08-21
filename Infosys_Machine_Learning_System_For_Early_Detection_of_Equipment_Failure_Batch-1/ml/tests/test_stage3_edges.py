"""Stage3TCN edge-case tests (TST-08)."""
import numpy as np
import torch

from flash_crash_watchdog.models.stage3_tcn import TCNConfig, Stage3TCN
from flash_crash_watchdog.tick import Tick
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel


def _tick(ts: int):
    book = OrderBookSnapshot(timestamp_ms=ts, bids=[PriceLevel(99.0, 1.0)], asks=[PriceLevel(101.0, 1.0)])
    t = Tick(book=book, symbol="BTC")
    t.features = {f: float(ts % 13) for f in [
        "f1_mid_velocity_50ms", "f1_mid_velocity_200ms", "f1_micro_price",
        "f1_trade_arrival_rate", "f1_cancel_to_trade_ratio",
        "f2_bid_depth_10", "f2_ask_depth_10", "f2_obi_10",
        "f2_weighted_mid_10", "f2_depth_slope",
        "f3_vpin", "f3_kyle_lambda", "f3_effective_spread_bps", "f3_realized_spread_bps",
        "f4_realized_vol_1s", "f4_variance_ratio", "f4_garman_klass",
    ]}
    return t


def test_score_current_warmup_until_full_window():
    s3 = Stage3TCN(TCNConfig(sequence_length=50))
    for _ in range(49):
        assert s3.score_current() == (0.0, False), "must be warmup before the window is full"
        s3.feed(_tick(0))
    # one more tick fills the window -> it scores (score may be anything, finite)
    s3.feed(_tick(1))
    val, should_pass = s3.score_current()
    assert np.isfinite(val)


def test_load_rebuilds_seq_length_from_checkpoint(tmp_path):
    """Stage3TCN.load must adopt the checkpoint's sequence_length (e.g. 200),
    overriding the caller's default (500) — the drift that made offline numbers
    disagree with the live 200-tick path."""
    from flash_crash_watchdog.models.stage3_tcn import TCNConfig as TC, Stage3TCN as S3
    src = S3(TC(sequence_length=200))
    path = tmp_path / "m.pt"
    torch.save({"model_state": src.model.state_dict(), "config": src.config}, path)

    fresh = S3(TC(sequence_length=500))  # caller default mismatches the checkpoint
    assert fresh._max_window == 500
    fresh.load(str(path))
    assert fresh._max_window == 200, "load must adopt the checkpoint's sequence_length"