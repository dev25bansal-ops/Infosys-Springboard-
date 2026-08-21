"""Download historical Binance data from data.binance.vision."""
from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly"

DATA_TYPES = {
    "trades": "trades",
    "depth": "depthBookToTick",
    "klines": "klines",
    "agg_trades": "aggTrades",
}


def download_file(url: str, out_path: Path) -> bool:
    try:
        logger.info("Downloading %s", url)
        urllib.request.urlretrieve(url, str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("Saved %s (%.1f MB)", out_path.name, size_mb)
        return True
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return False


def download_binance_data(
    symbol: str,
    date: str,
    data_type: str = "trades",
    out_dir: str | Path = "data",
) -> Path | None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "-" in date and len(date.split("-")) == 3:
        url = f"https://data.binance.vision/data/spot/daily/{DATA_TYPES[data_type]}/{symbol}/{symbol}-{DATA_TYPES[data_type]}-{date}.zip"
        filename = f"{symbol}-{data_type}-{date}.zip"
    else:
        url = f"{BASE_URL}/{DATA_TYPES[data_type]}/{symbol}/{symbol}-{DATA_TYPES[data_type]}-{date}.zip"
        filename = f"{symbol}-{data_type}-{date}.zip"

    out_path = out_dir / filename
    if out_path.exists():
        logger.info("File already exists: %s", out_path)
        return out_path

    if download_file(url, out_path):
        return out_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--type", default="trades", choices=list(DATA_TYPES.keys()))
    parser.add_argument("--out", default="data/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = download_binance_data(args.symbol, args.date, args.type, args.out)
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())
