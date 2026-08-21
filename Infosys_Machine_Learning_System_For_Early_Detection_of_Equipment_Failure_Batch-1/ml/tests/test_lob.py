"""Tests for the LOB (Limit Order Book) module."""
from flash_crash_watchdog.lob import OrderBookReconstructor, OrderBookSnapshot, PriceLevel


def test_price_level():
    level = PriceLevel(price=100.0, size=1.5)
    assert level.price == 100.0
    assert level.size == 1.5


def test_order_book_snapshot_mid_price():
    book = OrderBookSnapshot(
        timestamp_ms=1000,
        bids=[PriceLevel(99.0, 1.0), PriceLevel(98.0, 2.0)],
        asks=[PriceLevel(101.0, 1.5), PriceLevel(102.0, 0.5)],
    )
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0
    assert book.mid_price == 100.0
    assert book.spread == 2.0


def test_order_book_snapshot_micro_price():
    """micro_price weights bid/ask by opposing side's size."""
    book = OrderBookSnapshot(
        timestamp_ms=1000,
        bids=[PriceLevel(99.0, 1.0)],  # small bid size
        asks=[PriceLevel(101.0, 3.0)],  # large ask size
    )
    # micro_price = (99 * 3 + 101 * 1) / (1 + 3) = (297 + 101) / 4 = 99.5
    assert abs(book.micro_price - 99.5) < 0.01


def test_order_book_imbalance():
    # Balanced book
    book = OrderBookSnapshot(
        timestamp_ms=1000,
        bids=[PriceLevel(100.0, 5.0)],
        asks=[PriceLevel(101.0, 5.0)],
    )
    assert abs(book.order_book_imbalance(1)) < 0.01

    # Bid-heavy book (bullish pressure)
    book = OrderBookSnapshot(
        timestamp_ms=1000,
        bids=[PriceLevel(100.0, 9.0)],
        asks=[PriceLevel(101.0, 1.0)],
    )
    assert book.order_book_imbalance(1) > 0.5


def test_reconstructor():
    recon = OrderBookReconstructor()
    recon.update_level("bid", 100.0, 1.0)
    recon.update_level("bid", 99.0, 2.0)
    recon.update_level("ask", 101.0, 1.5)
    recon.update_level("ask", 102.0, 0.5)

    snap = recon.snapshot(timestamp_ms=1000, levels=10)
    assert snap.best_bid == 100.0
    assert snap.best_ask == 101.0
    assert len(snap.bids) == 2
    assert len(snap.asks) == 2

    # Cancel a level
    recon.update_level("bid", 100.0, 0.0)
    snap = recon.snapshot(timestamp_ms=1001, levels=10)
    assert snap.best_bid == 99.0
    assert len(snap.bids) == 1


# --- TST-04: side validation + property tests ---------------------------------

def test_update_level_rejects_unknown_side():
    """A typo side must NOT silently mutate the ask book (was routed there)."""
    recon = OrderBookReconstructor()
    recon.update_level("bid", 100.0, 1.0)
    recon.update_level("ask", 101.0, 1.0)
    recon.update_level("bidss", 99.0, 5.0)   # typo -> must be ignored
    recon.update_level("BUY", 98.0, 5.0)     # non-canonical -> must be ignored
    snap = recon.snapshot(timestamp_ms=0, levels=10)
    assert len(snap.asks) == 1 and snap.asks[0].price == 101.0
    assert len(snap.bids) == 1 and snap.bids[0].price == 100.0


def test_update_level_is_case_insensitive():
    recon = OrderBookReconstructor()
    recon.update_level("BID", 100.0, 1.0)
    recon.update_level("Ask", 101.0, 1.0)
    snap = recon.snapshot(timestamp_ms=0, levels=10)
    assert snap.best_bid == 100.0 and snap.best_ask == 101.0


def test_snapshot_properties():
    """Property: bids sorted desc, asks sorted asc, size-0 removes, truncation."""
    recon = OrderBookReconstructor()
    prices = [100.0, 99.0, 101.0, 98.0, 102.0, 97.0]
    for p in prices[:3]:
        recon.update_level("bid", p, 1.0)
    for p in prices[3:]:
        recon.update_level("ask", p, 1.0)
    # remove one ask and one bid
    recon.update_level("ask", 98.0, 0.0)
    recon.update_level("bid", 99.0, 0.0)
    snap = recon.snapshot(timestamp_ms=0, levels=2)
    assert [b.price for b in snap.bids] == sorted([b.price for b in snap.bids], reverse=True)
    assert [a.price for a in snap.asks] == sorted([a.price for a in snap.asks])
    assert len(snap.bids) <= 2 and len(snap.asks) <= 2
    assert all(s > 0 for s in [b.size for b in snap.bids] + [a.size for a in snap.asks])
    assert 99.0 not in [b.price for b in snap.bids]
    assert 98.0 not in [a.price for a in snap.asks]
