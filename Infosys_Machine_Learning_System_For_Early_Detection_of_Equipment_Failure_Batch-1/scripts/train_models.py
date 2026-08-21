#!/usr/bin/env python3
"""Train the detector models on real Binance crash data.

Trains:
    1. Stage 2 Isolation Forest on normal traffic (unsupervised)
    2. Stage 3 TCN on labeled crash windows (self-supervised + supervised)

Usage:
    python scripts/train_models.py --data data/parquet/BTCUSDT_2021-05-18.parquet --out models/
    python scripts/train_models.py --data data/parquet/BTCUSDT_2024-01-15.parquet --out models/ --epochs 20

The training data should be a NORMAL day (not a crash day) so the models
learn what "normal" looks like. Then the backtest on crash days will detect
the anomalies.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Insert the ml directory at the FRONT of sys.path
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet
from flash_crash_watchdog.data.windows import build_windows_from_df
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_feature_matrix(df: pd.DataFrame, max_ticks: int = 100_000) -> np.ndarray:
    """Extract a feature matrix from a DataFrame for training.

    Args:
        df: Historical tick data.
        max_ticks: Maximum number of ticks to process (for speed).

    Returns:
        Matrix of shape (n_ticks, 20) — the feature vector per tick.
    """
    logger.info("Extracting features from %d ticks (max %d)...", len(df), max_ticks)

    # Sample if too many ticks
    if len(df) > max_ticks:
        # Sample evenly across the day to get representative data
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df_sample = df.iloc[indices].copy()
        logger.info("Sampled down to %d ticks (evenly spaced)", len(df_sample))
    else:
        df_sample = df

    extractor = FeatureExtractor()
    features_list = []

    for i, tick in enumerate(df_to_ticks(df_sample, symbol="TRAIN")):
        if i % 10000 == 0:
            logger.info("  Processing tick %d/%d...", i, len(df_sample))
        features = extractor.extract(tick)
        features_list.append([features.get(f, 0.0) for f in FEATURE_NAMES])

    matrix = np.array(features_list, dtype=np.float32)
    logger.info("Feature matrix shape: %s", matrix.shape)

    # Replace NaN/Inf with 0
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return matrix


def train_stage2_isolation_forest(feature_matrix: np.ndarray, out_path: Path) -> None:
    """Train the Stage 2 Isolation Forest on normal data."""
    logger.info("=" * 60)
    logger.info("TRAINING STAGE 2 — ISOLATION FOREST")
    logger.info("=" * 60)

    # Use the first 12 features (F1 + F2) for Stage 2
    stage2_features = feature_matrix[:, :12]
    logger.info("Stage 2 input shape: %s", stage2_features.shape)

    model = Stage2IsolationForest(n_estimators=100, contamination=0.05)
    model.fit(stage2_features)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    logger.info("Stage 2 model saved to %s", out_path)


def train_stage3_tcn(df: pd.DataFrame, out_path: Path, epochs: int = 20,
                     max_ticks: int = 100_000) -> None:
    """Train the Stage 3 TCN on labeled crash windows.

    Built from the same tick frame; window labels come from a forward-looking
    2%-drop-in-5s rule (see data.windows.build_windows_from_df). A "normal" day
    has no crash labels, so Stage 3 is skipped with a pointer rather than
    silently trained to an all-zero target.
    """
    logger.info("=" * 60)
    logger.info("TRAINING STAGE 3 — TEMPORAL CONVOLUTIONAL NETWORK")
    logger.info("=" * 60)

    try:
        windows, labels, _feature_names = build_windows_from_df(df, max_ticks=max_ticks)
    except FileNotFoundError as e:
        logger.warning("Skipping Stage 3: %s", e)
        return
    n_pos = int(np.sum(labels))
    if n_pos == 0:
        logger.warning(
            "No crash labels in this day — the TCN is a crash classifier and cannot be "
            "trained on normal-only data. Skip Stage 3 (or run scripts/train_tcn_windows.py "
            "on labeled crash windows).")
        return
    logger.info("Stage 3 windows: %s (%d positive)", windows.shape, n_pos)

    seq_len = windows.shape[1]
    config = TCNConfig(sequence_length=seq_len)
    model = Stage3TCN(config)
    model.train(windows, labels, epochs=epochs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    logger.info("Stage 3 model saved to %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train detector models on real data")
    parser.add_argument("--data", required=True, help="Parquet file of NORMAL market data")
    parser.add_argument("--out", default="models/", help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=20, help="TCN training epochs")
    parser.add_argument("--max-ticks", type=int, default=100_000,
                        help="Max ticks to process (for speed)")
    args = parser.parse_args()

    # Load data
    df = load_parquet(args.data)
    logger.info("Loaded %d ticks from %s", len(df), args.data)

    # Extract features
    feature_matrix = extract_feature_matrix(df, max_ticks=args.max_ticks)

    # Train Stage 2
    out_dir = Path(args.out)
    train_stage2_isolation_forest(feature_matrix, out_dir / "stage2_isolation_forest.joblib")

    # Train Stage 3
    train_stage3_tcn(df, out_dir / "stage3_tcn.pt", epochs=args.epochs,
                     max_ticks=args.max_ticks)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Models saved to: %s", out_dir.resolve())
    logger.info("  Stage 2: stage2_isolation_forest.joblib")
    logger.info("  Stage 3: stage3_tcn.pt")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next step: re-run the backtest on crash data to see alerts!")
    logger.info("  python scripts/run_backtest.py --data data/parquet/BTCUSDT_2021-05-19.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
