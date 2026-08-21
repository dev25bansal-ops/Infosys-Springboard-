#!/usr/bin/env python3
"""FIXED Bybit historical data downloader.

Bybit changed their public data URL pattern. The correct format is:
    https://public.bybit.com/kline_for_metatrader4/{symbol}/{year}-{month:02d}/
    https://public.bybit.com/kline/{symbol}/{YYYY-MM-DD}/{interval}.csv.gz
    https://public.bybit.com/premium_quote/{symbol}/{YYYY-MM-DD}/{interval}.csv.gz
    https://public.bybit.com/trading/{symbol}/{YYYY-MM-DD}.csv.gz

This script tries multiple patterns + the Bybit v5 API as fallback.

Usage:
    python scripts/download_bybit_fixed.py --out data/more/bybit/
    python scripts/download_bybit_fixed.py --out data/more/bybit/ --symbols BTCUSDT,ETHUSDT
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
            if len(data) < 100:
                logger.error("  ✗ Response too small (%d bytes) — likely 404", len(data))
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            size_kb = len(data) / 1024
            logger.info("  ✓ %s (%.1f KB)", out_path.name, size_kb)
            return True
    except Exception as e:
        logger.error("  ✗ %s", e)
        return False


def download_json(url: str, out_path: Path, timeout: int = 30) -> bool:
    """Download JSON from Bybit v5 API."""
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("  SKIP (exists): %s", out_path.name)
        return True
    try:
        logger.info("  GET %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "flash-crash-watchdog/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            if data.get("retCode") != 0:
                logger.error("  ✗ API error: %s", data.get("retMsg", "unknown"))
                return False
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2))
            n = len(data.get("result", {}).get("list", []))
            logger.info("  ✓ %s (%d candles)", out_path.name, n)
            return True
    except Exception as e:
        logger.error("  ✗ %s", e)
        return False


def try_bybit_kline_csv(out_dir: Path, symbol: str, date: str, intervals: list[str]) -> bool:
    """Try downloading Bybit kline CSV files with multiple URL patterns."""
    for interval in intervals:
        urls = [
            # Pattern 1: /kline/{symbol}/{date}/{interval}.csv.gz
            f"https://public.bybit.com/kline/{symbol}/{date}/{interval}.csv.gz",
            # Pattern 2: /kline_for_metatrader4/{symbol}/{date}/{interval}.csv.gz
            f"https://public.bybit.com/kline_for_metatrader4/{symbol}/{date}/{interval}.csv.gz",
            # Pattern 3: /price_quote/{symbol}/{date}/{interval}.csv.gz
            f"https://public.bybit.com/price_quote/{symbol}/{date}/{interval}.csv.gz",
            # Pattern 4: /premium_quote/{symbol}/{date}/{interval}.csv.gz
            f"https://public.bybit.com/premium_quote/{symbol}/{date}/{interval}.csv.gz",
        ]
        for url in urls:
            out = out_dir / f"BYBIT-{symbol}-{interval}-{date}.csv.gz"
            if download_file(url, out):
                return True
    return False


def try_bybit_trading_csv(out_dir: Path, symbol: str, date: str) -> bool:
    """Try downloading Bybit tick-level trading data."""
    urls = [
        f"https://public.bybit.com/trading/{symbol}/{date}.csv.gz",
        f"https://public.bybit.com/trading/{symbol}/{date}.csv",
    ]
    for url in urls:
        out = out_dir / f"BYBIT-{symbol}-trades-{date}.csv.gz"
        if download_file(url, out):
            return True
    return False


def download_bybit_api_klines(out_dir: Path, symbol: str, date: str, interval: str = "1") -> bool:
    """Download klines via Bybit v5 REST API as fallback.

    API: GET /v5/market/kline
    Params: category, symbol, interval, start, end
    Interval: 1,3,5,15,30,60,120,240,360,720,D,W,M
    """
    import datetime
    start_dt = datetime.datetime.strptime(date, "%Y-%m-%d")
    end_dt = start_dt + datetime.timedelta(days=1)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    # Bybit v5 API
    url = (
        f"https://api.bybit.com/v5/market/kline"
        f"?category=linear&symbol={symbol}&interval={interval}"
        f"&start={start_ts}&end={end_ts}&limit=1000"
    )
    out = out_dir / f"BYBIT-{symbol}-{interval}min-{date}.json"
    return download_json(url, out)


def download_bybit(out_dir: Path, symbols: list[str], dates: list[str]) -> int:
    """Download Bybit historical data using multiple strategies."""
    logger.info("=" * 70)
    logger.info("BYBIT HISTORICAL DATA (fixed — multiple URL patterns + API fallback)")
    logger.info("  Symbols: %s", ", ".join(symbols))
    logger.info("  Dates:   %s", ", ".join(dates))
    logger.info("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    total = 0

    for symbol in symbols:
        for date in dates:
            # Strategy 1: Try kline CSV files (multiple intervals)
            total += 1
            logger.info("\n  [%s %s] Strategy 1: kline CSV files", symbol, date)
            if try_bybit_kline_csv(out_dir, symbol, date, ["1min", "5min", "1", "5"]):
                success += 1
                continue

            # Strategy 2: Try trading (tick) CSV
            total += 1
            logger.info("  [%s %s] Strategy 2: trading CSV", symbol, date)
            if try_bybit_trading_csv(out_dir, symbol, date):
                success += 1
                continue

            # Strategy 3: API fallback (always works for recent dates)
            total += 1
            logger.info("  [%s %s] Strategy 3: Bybit v5 API fallback", symbol, date)
            if download_bybit_api_klines(out_dir, symbol, date, interval="1"):
                success += 1
                continue

            logger.error("  [%s %s] All strategies failed", symbol, date)

    logger.info("\n  Result: %d/%d files\n", success, total)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Bybit historical data (fixed)")
    parser.add_argument("--out", default="data/more/bybit/", help="Output directory")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--dates", default="2021-05-19,2022-05-10,2024-08-05",
                        help="Comma-separated dates (YYYY-MM-DD)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    dates = [d.strip() for d in args.dates.split(",")]

    total = download_bybit(Path(args.out), symbols, dates)

    logger.info("=" * 70)
    logger.info("  BYBIT DOWNLOAD COMPLETE — %d files", total)
    logger.info("  Output: %s", Path(args.out).resolve())
    logger.info("=" * 70)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
