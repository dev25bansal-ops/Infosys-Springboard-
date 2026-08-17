"""Tests for the live Binance stream parser (ENH-08)."""
from flash_crash_watchdog.data.live_stream import BinanceLiveStream


def _stream():
    return BinanceLiveStream(symbol="BTCUSDT", depth_levels=10)


def test_depth_uses_binance_event_time():
    s = _stream()
    depth = {
        "E": 1700000123456,
        "bids": [["100.0", "1.0"]],
        "asks": [["101.0", "1.0"]],
    }
    tick = s._parse_depth(depth)
    assert tick is not None
    assert tick.book.timestamp_ms == 1700000123456  # event time, not local clock


def test_trade_tick_attaches_real_book():
    s = _stream()
    # a depth update first -> real book (best bid 100, ask 101)
    s._parse_depth({"E": 1000, "bids": [["100.0", "5.0"]], "asks": [["101.0", "5.0"]]})
    trade = s._parse_trade({"E": 1005, "p": "100.5", "q": "2.0", "m": False})
    assert trade is not None
    assert len(trade.trades) == 1
    assert trade.trades[0].price == 100.5
    # the book is the REAL last depth book, not a fabricated trade-price book
    assert trade.book.best_bid == 100.0 and trade.book.best_ask == 101.0
    assert trade.book.spread is not None and trade.book.spread > 0


def test_trade_tick_without_prior_depth_uses_empty_book():
    s = _stream()
    trade = s._parse_trade({"T": 1005, "p": "100.5", "q": "2.0", "m": True})
    assert trade is not None
    assert trade.trades[0].side == "sell"  # buyer_maker -> sell aggressor
    assert trade.book.mid_price is None  # no fabricated degenerate book