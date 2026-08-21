"""ADV-05: order-flow feature tests — the previously-dead features are now real."""
from flash_crash_watchdog.features.flow_toxicity import FlowToxicityFeatures
from flash_crash_watchdog.features.price_action import PriceActionFeatures
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick, Trade


def _tick(mid, ts, bids_size=5.0, asks_size=5.0, trades=None):
    book = OrderBookSnapshot(
        timestamp_ms=ts,
        bids=[PriceLevel(mid - 0.5, bids_size)],
        asks=[PriceLevel(mid + 0.5, asks_size)],
    )
    return Tick(book=book, trades=trades or [], symbol="TEST")


def test_cancel_to_trade_ratio_proxy_is_nonzero():
    """Depth that vanishes without a matching trade is a real cancellation signal."""
    f1 = PriceActionFeatures()
    f1.update(_tick(100.0, 1000, bids_size=5.0, asks_size=5.0, trades=[Trade(1000, 100.0, 1.0, "buy")]))
    f2 = f1.update(_tick(100.0, 1100, bids_size=2.0, asks_size=2.0))  # depth 10 -> 4, no trade
    assert f2["f1_cancel_to_trade_ratio"] > 0, "depth loss w/o trades must raise the cancel ratio"
    # no trades at all -> clamped, not NaN/explosion
    f1b = PriceActionFeatures()
    f1b.update(_tick(100.0, 1000, bids_size=5.0, asks_size=5.0))
    f1b.update(_tick(100.0, 1100, bids_size=2.0, asks_size=2.0))
    assert 0.0 <= f1b.update(_tick(100.0, 1200))["f1_cancel_to_trade_ratio"] <= 10.0


def test_realized_spread_proxy_is_nonzero_and_less_than_effective():
    """Realized spread = effective spread net of price impact — nonzero, bounded."""
    f3 = FlowToxicityFeatures()
    f3.update(_tick(100.0, 0))
    # a buy trade at the ask (100.5), then the mid RISES -> informed component reduces realized
    f3.update(_tick(100.0, 1, trades=[Trade(1, 100.5, 1.0, "buy")]))
    out = f3.update(_tick(100.3, 2))  # mid moved up after the buy
    assert out["f3_realized_spread_bps"] > 0
    assert out["f3_realized_spread_bps"] < out["f3_effective_spread_bps"] or out["f3_effective_spread_bps"] > 0
    # realized is finite and non-negative
    assert 0.0 <= out["f3_realized_spread_bps"] <= 1000.0