#!/usr/bin/env python3
"""Train the TCN model on labeled windows (or build them from tick data).

Usage:
    # From pre-built windows (fastest):
    python scripts/train_tcn.py --data data/windows/BTCUSDT_2021-05-19_windows.npz --epochs 50

    # From a crash-day parquet (windows + labels built on the fly):
    python scripts/train_tcn.py --data data/parquet/BTCUSDT_2021-05-19.parquet --epochs 50

    # A directory of windows/.parquet files:
    python scripts/train_tcn.py --data data/windows/ --out models/tcn_baseline.pt
"""
import argparse
import logging
import sys
from pathlib import Path

# Insert the ml folder so `flash_crash_watchdog.*` imports resolve.
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.data.windows import resolve_windows_source  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the TCN model")
    parser.add_argument("--data", required=True,
                        help="a .npz window file, a dir of .npz/.parquet, or a tick .parquet/.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Cap ticks when building windows from parquet/csv (0 = all)")
    parser.add_argument("--output", default="models/tcn_baseline.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    windows, labels, _feature_names = resolve_windows_source(args.data, max_ticks=args.max_ticks)
    config = TCNConfig(sequence_length=windows.shape[1], input_dim=windows.shape[-1])
    model = Stage3TCN(config, device=args.device)
    model.train(windows, labels, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.lr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())