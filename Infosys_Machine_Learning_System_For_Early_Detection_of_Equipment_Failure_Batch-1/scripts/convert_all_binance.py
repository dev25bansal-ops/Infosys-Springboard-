#!/usr/bin/env python3
"""Convert ALL downloaded Binance ZIP files to parquet in one batch.

Usage:
    python scripts/convert_all_binance.py --input data/ --output data/parquet/
"""
import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def convert_zip(zip_path: Path, output_path: Path) -> bool:
    """Convert one Binance trades ZIP to parquet. Returns True on success."""
    if output_path.exists():
        logger.info("  SKIP (exists): %s", output_path.name)
        return True
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(
                    io.BytesIO(f.read()),
                    header=None,
                    names=["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"],
                )
        df_out = pd.DataFrame({
            "timestamp_ms": df["time"].astype("int64"),
            "best_bid": df["price"] * 0.9999,
            "best_ask": df["price"] * 1.0001,
            "bid_size": 1.0,
            "ask_size": 1.0,
            "mid_price": df["price"].astype("float64"),
            "trade_price": df["price"].astype("float64"),
            "trade_size": df["qty"].astype("float64"),
            "trade_side": df["isBuyerMaker"].map({True: "sell", False: "buy"}),
        }).sort_values("timestamp_ms").reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_parquet(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("  ✓ %s -> %s (%.1f MB, %d ticks)",
                    zip_path.name, output_path.name, size_mb, len(df_out))
        return True
    except Exception as e:
        logger.error("  ✗ %s: %s", zip_path.name, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert all Binance ZIPs to parquet")
    parser.add_argument("--input", default="data/", help="Directory with .zip files")
    parser.add_argument("--output", default="data/parquet/", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    zips = sorted(input_dir.glob("*-trades-*.zip"))
    if not zips:
        logger.error("No *-trades-*.zip files found in %s", input_dir)
        return 1

    logger.info("Found %d ZIP files to convert", len(zips))
    success = 0
    for zip_path in zips:
        # e.g., BTCUSDT-trades-2021-05-19.zip -> BTCUSDT_2021-05-19.parquet
        stem = zip_path.stem  # BTCUSDT-trades-2021-05-19
        parts = stem.split("-", 1)  # ["BTCUSDT", "trades-2021-05-19"]
        symbol = parts[0]
        date_part = parts[1].replace("trades-", "")  # 2021-05-19
        out_name = f"{symbol}_{date_part}.parquet"
        out_path = output_dir / out_name
        if convert_zip(zip_path, out_path):
            success += 1

    logger.info("=" * 50)
    logger.info("Converted %d/%d files", success, len(zips))
    logger.info("Output: %s", output_dir.resolve())
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
