#!/usr/bin/env python3
"""FIXED downloader for Binance futures funding rates + depth + equities.

Fixes:
    1. Funding rates: use Binance REST API (works for any historical date)
    2. Depth: try both 'depth' and 'depthBookToTick' paths + monthly fallback
    3. Equities: auto-install yfinance if missing

Usage:
    python scripts/download_fixed.py --out data/more/
"""
import argparse
import json
import logging
import subprocess
import sys
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


def download_json(url: str, out_path: Path, timeout: int = 30) -> bool:
    """Download JSON from an API endpoint."""
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("  SKIP (exists): %s", out_path.name)
        return True
    try:
        logger.info("  GET %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "flash-crash-watchdog/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2))
            logger.info("  ✓ %s (%d records)", out_path.name, len(data) if isinstance(data, list) else 1)
            return True
    except Exception as e:
        logger.error("  ✗ %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 1. FUNDING RATES via Binance REST API (works for any historical date)
# ═══════════════════════════════════════════════════════════════════════════
def download_funding_rates_api(out_dir: Path) -> int:
    """Download funding rates via Binance Futures REST API.

    API: GET /fapi/v1/fundingRate
    Params: symbol, startTime, endTime, limit (max 1000)
    """
    logger.info("=" * 70)
    logger.info("1. BINANCE FUNDING RATES (via REST API)")
    logger.info("   Inverted funding = extreme short pressure (crypto crash signal)")
    logger.info("=" * 70)

    # Convert dates to timestamps (ms)
    import datetime
    crash_periods = {
        "2021-05-19": ("2021-05-19", "2021-05-20", "May 2021 BTC crash"),
        "2022-05-10": ("2022-05-09", "2022-05-13", "May 2022 LUNA crash"),
        "2022-06-13": ("2022-06-13", "2022-06-14", "Celsius freeze"),
        "2024-08-05": ("2024-08-05", "2024-08-06", "Carry trade unwind"),
    }
    symbols = ["BTCUSDT", "ETHUSDT"]

    success = 0
    total = 0
    for symbol in symbols:
        for date_key, (start, end, desc) in crash_periods.items():
            total += 1
            start_ts = int(datetime.datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
            url = (
                f"https://fapi.binance.com/fapi/v1/fundingRate"
                f"?symbol={symbol}&startTime={start_ts}&endTime={end_ts}&limit=1000"
            )
            out = out_dir / f"{symbol}-funding-{date_key}.json"
            logger.info("  %s %s — %s", symbol, date_key, desc)
            if download_json(url, out):
                success += 1
    logger.info("  Result: %d/%d files\n", success, total)
    return success


# ═══════════════════════════════════════════════════════════════════════════
# 2. OPEN INTEREST + LONG/SHORT RATIO via Binance REST API
# ═══════════════════════════════════════════════════════════════════════════
def download_futures_metrics_api(out_dir: Path) -> int:
    """Download open interest history + long/short ratio via Binance API."""
    logger.info("=" * 70)
    logger.info("2. BINANCE FUTURES METRICS (open interest + long/short ratio)")
    logger.info("=" * 70)

    import datetime
    success = 0
    total = 0

    # Open interest history (5-min intervals, last 30 days available)
    # GET /futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=30
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        total += 1
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=15m&limit=1000"
        out = out_dir / f"{symbol}-open-interest-recent.json"
        if download_json(url, out):
            success += 1

        # Top trader long/short ratio (accounts)
        total += 1
        url = f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={symbol}&period=15m&limit=1000"
        out = out_dir / f"{symbol}-longshort-ratio-recent.json"
        if download_json(url, out):
            success += 1

        # Taker buy/sell volume
        total += 1
        url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=15m&limit=1000"
        out = out_dir / f"{symbol}-taker-volume-recent.json"
        if download_json(url, out):
            success += 1

    logger.info("  Result: %d/%d files\n", success, total)
    return success


# ═══════════════════════════════════════════════════════════════════════════
# 3. DEPTH — try multiple paths (daily + monthly, both types)
# ═══════════════════════════════════════════════════════════════════════════
def download_depth_multi(out_dir: Path) -> int:
    """Try multiple URL patterns for Binance depth data."""
    logger.info("=" * 70)
    logger.info("3. BINANCE DEPTH SNAPSHOTS (trying multiple URL patterns)")
    logger.info("=" * 70)

    dates = ["2021-05-19", "2022-05-10", "2024-08-05"]
    success = 0
    total = 0

    for date in dates:
        total += 1
        year, month, day = date.split("-")
        # Try 4 URL patterns in order
        urls = [
            # 1. Daily depthBookToTick
            f"https://data.binance.vision/data/spot/daily/depthBookToTick/BTCUSDT/BTCUSDT-depthBookToTick-{date}.zip",
            # 2. Daily depth (snapshot)
            f"https://data.binance.vision/data/spot/daily/depth/BTCUSDT/BTCUSDT-depth-{date}.zip",
            # 3. Monthly depthBookToTick
            f"https://data.binance.vision/data/spot/monthly/depthBookToTick/BTCUSDT/BTCUSDT-depthBookToTick-{year}-{month}.zip",
            # 4. Monthly depth
            f"https://data.binance.vision/data/spot/monthly/depth/BTCUSDT/BTCUSDT-depth-{year}-{month}.zip",
        ]
        downloaded = False
        for i, url in enumerate(urls):
            out = out_dir / f"BTCUSDT-depth-{date}.zip"
            if download_file(url, out):
                success += 1
                downloaded = True
                break
        if not downloaded:
            logger.warning("  ✗ No depth data available for %s (tried 4 URL patterns)", date)
    logger.info("  Result: %d/%d files\n", success, total)
    return success


# ═══════════════════════════════════════════════════════════════════════════
# 4. EQUITIES via yfinance (auto-install if missing)
# ═══════════════════════════════════════════════════════════════════════════
def ensure_yfinance() -> bool:
    """Ensure yfinance is installed. Returns True if available."""
    try:
        import yfinance  # noqa: F401
        return True
    except ImportError:
        logger.info("  yfinance not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
            logger.info("  ✓ yfinance installed")
            return True
        except Exception as e:
            logger.error("  ✗ Failed to install yfinance: %s", e)
            logger.error("  Run manually: pip install yfinance")
            return False


def download_equities(out_dir: Path) -> int:
    """Download US equity + VIX data for flash-crash dates via yfinance."""
    logger.info("=" * 70)
    logger.info("4. US EQUITY DATA (SPY, QQQ, VIX, XLF, XLE via yfinance)")
    logger.info("   The May 6, 2010 US equities flash crash — the canonical case")
    logger.info("=" * 70)

    if not ensure_yfinance():
        return 0

    import yfinance as yf

    tickers = ["SPY", "QQQ", "^VIX", "XLF", "XLE"]
    periods = {
        "2010-05-06": "May 6, 2010 US equities flash crash (Dow -9.2% in 36 min)",
        "2015-08-24": "Aug 24, 2015 flash crash (China devaluation)",
        "2020-03-12": "March 12, 2020 COVID crash",
        "2024-08-05": "Aug 5, 2024 carry-trade unwind",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    total = 0
    for ticker_symbol in tickers:
        for date, desc in periods.items():
            total += 1
            logger.info("  %s %s — %s", ticker_symbol, date, desc)
            # Download 5 days around the crash date
            start = date
            year, month, day = map(int, date.split("-"))
            # End = date + 5 days
            from datetime import datetime, timedelta
            end_dt = datetime(year, month, day) + timedelta(days=5)
            end = end_dt.strftime("%Y-%m-%d")
            try:
                ticker = yf.Ticker(ticker_symbol)
                # Try 1-minute data first (most granular)
                hist = ticker.history(start=start, end=end, interval="1m")
                if hist.empty:
                    hist = ticker.history(start=start, end=end, interval="5m")
                if hist.empty:
                    hist = ticker.history(start=start, end=end, interval="1h")
                if hist.empty:
                    hist = ticker.history(start=start, end=end, interval="1d")
                if not hist.empty:
                    out = out_dir / f"YFINANCE-{ticker_symbol.replace('^','')}-{date}.csv"
                    hist.to_csv(out)
                    size_kb = out.stat().st_size / 1024
                    logger.info("    ✓ %s (%.1f KB, %d bars)", out.name, size_kb, len(hist))
                    success += 1
                else:
                    logger.warning("    ✗ No data for %s on %s", ticker_symbol, date)
            except Exception as e:
                logger.error("    ✗ %s: %s", ticker_symbol, e)
    logger.info("  Result: %d/%d files\n", success, total)
    return success


# ═══════════════════════════════════════════════════════════════════════════
# 5. BYBIT historical klines (cross-exchange)
# ═══════════════════════════════════════════════════════════════════════════
def download_bybit(out_dir: Path) -> int:
    """Download Bybit historical 1-min klines (cross-exchange)."""
    logger.info("=" * 70)
    logger.info("5. BYBIT HISTORICAL KLINES (cross-exchange — venue #2)")
    logger.info("=" * 70)
    dates = ["2021-05-19", "2022-05-10", "2024-08-05"]
    symbols = ["BTCUSDT", "ETHUSDT"]
    success = 0
    total = 0
    for symbol in symbols:
        for date in dates:
            total += 1
            url = f"https://public.bybit.com/kline/{symbol}/{date}/1min.csv.gz"
            out = out_dir / f"BYBIT-{symbol}-1min-{date}.csv.gz"
            if download_file(url, out):
                success += 1
    logger.info("  Result: %d/%d files\n", success, total)
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Download fixed datasets (funding API + depth + equities)")
    parser.add_argument("--out", default="data/more/", help="Output directory")
    parser.add_argument("--only", default=None,
                        help="Comma-separated: funding,metrics,depth,equities,bybit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    only = set(args.only.split(",")) if args.only else None

    total = 0
    if only is None or "funding" in only:
        total += download_funding_rates_api(out_dir / "futures")
    if only is None or "metrics" in only:
        total += download_futures_metrics_api(out_dir / "metrics")
    if only is None or "depth" in only:
        total += download_depth_multi(out_dir / "depth")
    if only is None or "equities" in only:
        total += download_equities(out_dir / "equities")
    if only is None or "bybit" in only:
        total += download_bybit(out_dir / "bybit")

    logger.info("=" * 70)
    logger.info("  DOWNLOAD COMPLETE — %d files total", total)
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("=" * 70)
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
