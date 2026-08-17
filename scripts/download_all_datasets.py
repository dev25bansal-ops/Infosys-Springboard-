#!/usr/bin/env python3
"""Download ALL datasets for the Flash Crash Early Warning project.

Downloads:
    1. Binance historical trades — May 2021 BTC flash crash (3 days)
    2. Binance historical trades — May 2022 LUNA crash (4 days)
    3. Binance historical trades — baseline normal days (2 days)
    4. FI-2010 academic benchmark dataset (from Helsinki fairdata.fi)

Total download size: ~2-3 GB
Time: ~10-20 minutes depending on connection

Usage:
    python scripts/download_all_datasets.py --out data/
    python scripts/download_all_datasets.py --out data/ --skip-fi2010  # skip the academic dataset
    python scripts/download_all_datasets.py --out data/ --binance-only  # Binance only
"""
import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Binance May 2021 BTC flash crash (3 days around May 19, 2021)
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_BTC_CRASH_2021 = {
    "name": "May 2021 BTC Flash Crash",
    "description": "BTC -30% in hours, $8B liquidations, Binance outage",
    "symbol": "BTCUSDT",
    "dates": ["2021-05-18", "2021-05-19", "2021-05-20"],
    "type": "trades",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Binance May 2022 LUNA death spiral (4 days)
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_LUNA_CRASH_2022 = {
    "name": "May 2022 LUNA / UST Death Spiral",
    "description": "LUNA -99.9% in 48h, $40B wiped, UST depeg",
    "symbol": "LUNAUSDT",
    "dates": ["2022-05-09", "2022-05-10", "2022-05-11", "2022-05-12"],
    "type": "trades",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: Baseline normal days (for training the "normal" model)
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_BASELINE = {
    "name": "BTC Baseline (normal volatility days)",
    "description": "Normal market days for training baseline / Isolation Forest",
    "symbol": "BTCUSDT",
    "dates": ["2024-01-15", "2024-01-16"],
    "type": "trades",
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4: FI-2010 academic benchmark
# ─────────────────────────────────────────────────────────────────────────────
FI2010_URLS = [
    # 5 stocks × 10 days = 50 files
    # These are the publicly-hosted versions from the original authors
    # Format: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
    # Direct download requires accepting a license — we provide instructions instead
]

DATASETS = [BINANCE_BTC_CRASH_2021, BINANCE_LUNA_CRASH_2022, BINANCE_BASELINE]


def download_file(url: str, out_path: Path, timeout: int = 120) -> bool:
    """Download a file with progress logging."""
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("  SKIP (already exists): %s (%.1f MB)", out_path.name, size_mb)
        return True
    try:
        logger.info("  Downloading %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "flash-crash-watchdog/0.4"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            logger.info("  ✓ Saved %s (%.1f MB)", out_path.name, size_mb)
            return True
    except Exception as e:
        logger.error("  ✗ Failed: %s", e)
        return False


def download_binance_dataset(dataset: dict, out_dir: Path) -> int:
    """Download all dates for a Binance dataset. Returns count of successful downloads."""
    symbol = dataset["symbol"]
    data_type = dataset["type"]
    success = 0
    total = len(dataset["dates"])

    logger.info("=" * 70)
    logger.info("DATASET: %s", dataset["name"])
    logger.info("  %s", dataset["description"])
    logger.info("  Symbol: %s  |  Type: %s  |  Days: %d", symbol, data_type, total)
    logger.info("=" * 70)

    for date in dataset["dates"]:
        url = (
            f"https://data.binance.vision/data/spot/daily/trades/"
            f"{symbol}/{symbol}-trades-{date}.zip"
        )
        out_path = out_dir / f"{symbol}-trades-{date}.zip"
        if download_file(url, out_path):
            success += 1

    logger.info("  Result: %d/%d days downloaded\n", success, total)
    return success


def download_fi2010_instructions(out_dir: Path) -> None:
    """Print instructions for downloading FI-2010 (requires license acceptance)."""
    logger.info("=" * 70)
    logger.info("DATASET: FI-2010 Academic Benchmark")
    logger.info("  5 Finnish stocks, 10 days, ~10ms tick resolution, labeled")
    logger.info("  700+ citations — the standard LOB benchmark")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  FI-2010 requires manual download (license acceptance).")
    logger.info("  Steps:")
    logger.info("    1. Go to: https://etsin.fairdata.fi/dataset/")
    logger.info("       73eb48d7-4dbc-4a10-a52a-da745b47a649")
    logger.info("    2. Click 'Download' and accept the license")
    logger.info("    3. Extract the .zip into: %s", out_dir / "fi2010")
    logger.info("    4. You should have 5 .txt files (one per stock)")
    logger.info("")
    logger.info("  Alternatively, use the mirror at:")
    logger.info("    https://research.ed.ac.uk/en/publications/")
    logger.info("    benchmark-dataset-for-mid-price-forecasting-of-limit-order-book")
    logger.info("")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all flash-crash datasets")
    parser.add_argument("--out", default="data/", help="Output directory")
    parser.add_argument("--binance-only", action="store_true",
                        help="Skip FI-2010 (Binance data only)")
    parser.add_argument("--skip-fi2010", action="store_true",
                        help="Skip FI-2010 instructions")
    parser.add_argument("--skip-btc-2021", action="store_true",
                        help="Skip May 2021 BTC crash")
    parser.add_argument("--skip-luna-2022", action="store_true",
                        help="Skip May 2022 LUNA crash")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline normal days")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  Flash Crash Early Warning — Dataset Downloader")
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("=" * 70)
    logger.info("")

    total_success = 0
    total_attempt = 0

    for dataset in DATASETS:
        if args.skip_btc_2021 and dataset["name"] == BINANCE_BTC_CRASH_2021["name"]:
            continue
        if args.skip_luna_2022 and dataset["name"] == BINANCE_LUNA_CRASH_2022["name"]:
            continue
        if args.skip_baseline and dataset["name"] == BINANCE_BASELINE["name"]:
            continue
        total_attempt += len(dataset["dates"])
        total_success += download_binance_dataset(dataset, out_dir)

    if not args.skip_fi2010 and not args.binance_only:
        download_fi2010_instructions(out_dir)

    logger.info("=" * 70)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("  Binance files: %d/%d successful", total_success, total_attempt)
    logger.info("  Output dir:    %s", out_dir.resolve())
    logger.info("=" * 70)

    if total_success == 0:
        logger.error("No files downloaded. Check your internet connection.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
