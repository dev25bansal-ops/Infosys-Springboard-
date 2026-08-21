"""FI-2010 benchmark dataset loader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)


def load_fi2010(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    all_data = []
    all_labels = []

    for file in files:
        logger.info("Loading %s...", file.name)
        df = pd.read_csv(file, sep="\t", header=None)
        prices_asks = df.iloc[:, :10].values
        sizes_asks = df.iloc[:, 10:20].values
        prices_bids = df.iloc[:, 20:30].values
        sizes_bids = df.iloc[:, 30:40].values
        labels = df.iloc[:, -1].values

        all_data.append(np.concatenate([prices_asks, sizes_asks, prices_bids, sizes_bids], axis=1))
        all_labels.append(labels)

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]

    logger.info("FI-2010 loaded: %d train, %d val samples", len(train_data), len(val_data))
    return train_data, val_data


def fi2010_to_ticks(data_dir: str | Path, symbol: str = "FI2010") -> list[Tick]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    ticks = []
    for file in files:
        df = pd.read_csv(file, sep="\t", header=None)
        for idx, row in df.iterrows():
            bids = [
                PriceLevel(float(row[20 + i]), float(row[30 + i]))
                for i in range(10) if float(row[20 + i]) > 0
            ]
            asks = [
                PriceLevel(float(row[i]), float(row[10 + i]))
                for i in range(10) if float(row[i]) > 0
            ]
            book = OrderBookSnapshot(timestamp_ms=idx, bids=bids, asks=asks)
            ticks.append(Tick(book=book, symbol=symbol))
    logger.info("Converted FI-2010 to %d ticks", len(ticks))
    return ticks
