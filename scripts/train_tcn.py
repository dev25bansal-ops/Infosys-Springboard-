#!/usr/bin/env python3
"""Train the TCN model on FI-2010 or custom data.

Usage:
    python scripts/train_tcn.py --data data/fi2010/ --epochs 50
"""
import argparse
import logging
import sys
from pathlib import Path

# Add ml package to path
# Insert the ml directory at the FRONT of sys.path
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.data.fi2010_loader import load_fi2010
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the TCN model")
    parser.add_argument("--data", required=True, help="FI-2010 directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", default="models/tcn_baseline.pt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_data, val_data = load_fi2010(args.data)
    model = Stage3TCN(TCNConfig())
    history = model.train(train_data, val_data, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
