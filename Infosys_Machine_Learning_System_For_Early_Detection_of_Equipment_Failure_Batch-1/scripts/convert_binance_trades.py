#!/usr/bin/env python3
"""Unzip and convert Binance trade ZIP files to parquet.

Binance trade CSVs have these columns:
    id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch

We convert to the schema expected by the detector:
    timestamp_ms, best_bid, best_ask, bid_size, ask_size, mid_price,
    trade_price, trade_size, trade_side

Usage:
    python scripts/convert_binance_trades.py --input data/BTCUSDT-trades-2021-05-19.zip --output data/BTCUSDT_2021-05-19.parquet
"""
import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd


def convert_zip_to_parquet(zip_path: str, output_path: str) -> None:
    """Convert a Binance trades ZIP to parquet."""
    zip_path = Path(zip_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Reading %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_name = zf.namelist()[0]
        logging.info("  Extracting %s", csv_name)
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                io.BytesIO(f.read()),
                header=None,
                names=["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"],
            )

    logging.info("  Loaded %d trades", len(df))

    # Build the detector schema
    # Each trade is a tick. We approximate the order book with the trade price
    # (real impl would merge with depth data).
    df_out = pd.DataFrame({
        "timestamp_ms": df["time"].astype("int64"),
        "best_bid": df["price"] * 0.9999,  # approximate bid = price - 1 bps
        "best_ask": df["price"] * 1.0001,  # approximate ask = price + 1 bps
        "bid_size": 1.0,  # placeholder
        "ask_size": 1.0,  # placeholder
        "mid_price": df["price"].astype("float64"),
        "trade_price": df["price"].astype("float64"),
        "trade_size": df["qty"].astype("float64"),
        "trade_side": df["isBuyerMaker"].map({True: "sell", False: "buy"}),
    })

    # Sort by timestamp
    df_out = df_out.sort_values("timestamp_ms").reset_index(drop=True)

    logging.info("  Writing %s (%d rows)", output_path, len(df_out))
    df_out.to_parquet(output_path, index=False)
    logging.info("Done. %d ticks -> %s", len(df_out), output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Binance trades ZIP to parquet")
    parser.add_argument("--input", required=True, help="Path to the .zip file")
    parser.add_argument("--output", required=True, help="Output .parquet path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert_zip_to_parquet(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
