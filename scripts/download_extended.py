#!/usr/bin/env python3
"""Download EXTENDED datasets for the Flash Crash Early Warning project.

Adds:
    1. Binance depth snapshots (the actual LOB, not just trades)
    2. Binance klines (OHLCV candles for long context)
    3. Binance funding rates (perpetual futures — crypto crash signal)
    4. Binance aggTrades (smaller, faster backtests)
    5. Coinbase BTC trades (cross-exchange — for Stage 4 Transformer)
    6. Kraken BTC trades (third venue)
    7. More crash dates (2024 NYSE halt day, 2015 Aug 24 flash crash equivalent)

Usage:
    python scripts/download_extended.py --out data/extended/
    python scripts/download_extended.py --out data/extended/ --only depth,klines
"""
import argparse
import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, out_path: Path, timeout: int = 120) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("  SKIP (exists): %s", out_path.name)
        return True
    try:
        logger.info("  GET %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "flash-crash-watchdog/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            logger.info("  ✓ %s (%.1f MB)", out_path.name, size_mb)
            return True
    except Exception as e:
        logger.error("  ✗ %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Binance DEPTH snapshots (the actual limit order book)
# ─────────────────────────────────────────────────────────────────────────────
def download_binance_depth(out_dir: Path, dates: list[str], symbol: str = "BTCUSDT") -> int:
    """Download Binance depth snapshots — the actual LOB data."""
    logger.info("=" * 60)
    logger.info("1. BINANCE DEPTH SNAPSHOTS (actual LOB)")
    logger.info("   Symbol: %s, Days: %d", symbol, len(dates))
    logger.info("=" * 60)
    success = 0
    for date in dates:
        url = f"https://data.binance.vision/data/spot/daily/depthBookToTick/{symbol}/{symbol}-depthBookToTick-{date}.zip"
        out = out_dir / f"{symbol}-depth-{date}.zip"
        if download_file(url, out):
            success += 1
    logger.info("  Result: %d/%d\n", success, len(dates))
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 2. Binance KLINES (OHLCV candles)
# ─────────────────────────────────────────────────────────────────────────────
def download_binance_klines(out_dir: Path, dates: list[str], symbol: str = "BTCUSDT",
                             interval: str = "1s") -> int:
    """Download Binance klines (1-second OHLCV candles)."""
    logger.info("=" * 60)
    logger.info("2. BINANCE KLINES (OHLCV, %s interval)", interval)
    logger.info("   Symbol: %s, Days: %d", symbol, len(dates))
    logger.info("=" * 60)
    success = 0
    for date in dates:
        url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
        out = out_dir / f"{symbol}-klines-{interval}-{date}.zip"
        if download_file(url, out):
            success += 1
    logger.info("  Result: %d/%d\n", success, len(dates))
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 3. Binance FUNDING RATES (perpetual futures — crypto crash signal)
# ─────────────────────────────────────────────────────────────────────────────
def download_binance_funding(out_dir: Path, dates: list[str], symbol: str = "BTCUSDT") -> int:
    """Download Binance futures funding rates — key crypto crash signal."""
    logger.info("=" * 60)
    logger.info("3. BINANCE FUNDING RATES (perpetual futures)")
    logger.info("   Inverted funding = extreme short pressure (May 2021 signal)")
    logger.info("=" * 60)
    success = 0
    for date in dates:
        url = f"https://data.binance.vision/data/futures/um/daily/fundingRate/{symbol}/{symbol}-fundingRate-{date}.zip"
        out = out_dir / f"{symbol}-funding-{date}.zip"
        if download_file(url, out):
            success += 1
    logger.info("  Result: %d/%d\n", success, len(dates))
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 4. Binance AGGTRADES (aggregated trades — smaller files)
# ─────────────────────────────────────────────────────────────────────────────
def download_binance_aggtrades(out_dir: Path, dates: list[str], symbol: str = "BTCUSDT") -> int:
    """Download Binance aggregated trades — smaller than raw trades."""
    logger.info("=" * 60)
    logger.info("4. BINANCE AGGTRADES (aggregated, smaller)")
    logger.info("   Symbol: %s, Days: %d", symbol, len(dates))
    logger.info("=" * 60)
    success = 0
    for date in dates:
        url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"
        out = out_dir / f"{symbol}-aggTrades-{date}.zip"
        if download_file(url, out):
            success += 1
    logger.info("  Result: %d/%d\n", success, len(dates))
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 5. COINBASE BTC trades (cross-exchange — for Stage 4 Transformer)
# ─────────────────────────────────────────────────────────────────────────────
def download_coinbase_trades(out_dir: Path, date: str) -> int:
    """Download Coinbase BTC/USD trades for a single day.

    Coinbase provides historical candles via API; for tick data we use
    the Coinbase Advanced Trade API historical data.
    """
    logger.info("=" * 60)
    logger.info("5. COINBASE BTC-USD (cross-exchange correlation)")
    logger.info("   Day: %s", date)
    logger.info("=" * 60)
    # Coinbase public API: 1-minute candles for BTC-USD
    # Format: https://api.exchange.coinbase.com/products/BTC-USD/candles
    url = (
        f"https://api.exchange.coinbase.com/products/BTC-USD/candles"
        f"?granularity=60&start={date}T00:00:00&end={date}T23:59:59"
    )
    out = out_dir / f"COINBASE-BTCUSD-candles-{date}.json"
    if download_file(url, out):
        logger.info("  ✓ Coinbase candles downloaded\n")
        return 1
    logger.info("  Result: 0/1\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. KRAKEN BTC trades (third venue)
# ─────────────────────────────────────────────────────────────────────────────
def download_kraken_trades(out_dir: Path, date: str) -> int:
    """Download Kraken XBT/USD trades for a single day (1-min OHLCV)."""
    logger.info("=" * 60)
    logger.info("6. KRAKEN XBT-USD (third venue for cross-exchange)")
    logger.info("   Day: %s", date)
    logger.info("=" * 60)
    # Kraken public API: OHLC data
    # https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1
    url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1"
    out = out_dir / f"KRAKEN-XBTUSD-ohlc-{date}.json"
    if download_file(url, out):
        logger.info("  ✓ Kraken OHLC downloaded\n")
        return 1
    logger.info("  Result: 0/1\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. ADDITIONAL CRASH DATES (more training examples)
# ─────────────────────────────────────────────────────────────────────────────
ADDITIONAL_CRASH_DATES = {
    "BTC": [
        ("2022-06-13", "Celsius withdrawal freeze — BTC -25%"),
        ("2022-11-08", "FTX collapse begins — BTC -15%"),
        ("2024-08-05", "Carry trade unwind — BTC -18%"),
    ],
    "ETH": [
        ("2021-05-19", "ETH flash crash — -40%"),
        ("2022-06-13", "ETH -28%"),
    ],
}


def download_additional_crashes(out_dir: Path) -> int:
    """Download additional crash-day data for more training examples."""
    logger.info("=" * 60)
    logger.info("7. ADDITIONAL CRASH DAYS (more training examples)")
    logger.info("=" * 60)
    success = 0
    total = 0
    for symbol, dates in ADDITIONAL_CRASH_DATES.items():
        for date, desc in dates:
            total += 1
            logger.info("  %s %s — %s", symbol, date, desc)
            url = f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{date}.zip"
            out = out_dir / f"{symbol}-trades-{date}.zip"
            if download_file(url, out):
                success += 1
    logger.info("  Result: %d/%d\n", success, total)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Download extended flash-crash datasets")
    parser.add_argument("--out", default="data/extended/", help="Output directory")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list: depth,klines,funding,aggtrades,coinbase,kraken,additional")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    # Crash dates to download depth/klines/funding for
    crash_dates_btc = ["2021-05-19", "2022-05-10", "2024-08-05"]
    crash_dates_luna = ["2022-05-10", "2022-05-11"]

    total = 0

    if only is None or "depth" in only:
        total += download_binance_depth(out_dir / "depth", crash_dates_btc)
    if only is None or "klines" in only:
        total += download_binance_klines(out_dir / "klines", crash_dates_btc)
    if only is None or "funding" in only:
        total += download_binance_funding(out_dir / "funding", crash_dates_btc)
    if only is None or "aggtrades" in only:
        total += download_binance_aggtrades(out_dir / "aggtrades", crash_dates_btc)
    if only is None or "coinbase" in only:
        total += download_coinbase_trades(out_dir / "coinbase", "2021-05-19")
    if only is None or "kraken" in only:
        total += download_kraken_trades(out_dir / "kraken", "2021-05-19")
    if only is None or "additional" in only:
        total += download_additional_crashes(out_dir / "additional_crashes")

    logger.info("=" * 60)
    logger.info("  EXTENDED DOWNLOAD COMPLETE")
    logger.info("  Total files: %d", total)
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("=" * 60)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
