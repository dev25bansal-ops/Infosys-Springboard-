#!/usr/bin/env python3
"""GAP-01: audit that no held-out validation day leaked into training.

Cross-checks ``configs/operating.yml``:
  - ``validation.train_days``   (inputs to stage3_tcn_prod.pt)
  - ``validation.days``         (the held-out 6-day validation table)

Fails (exit 1) if any held-out day appears in the training set at the (symbol,
date) granularity, or if a training-input window file is missing from
``--windows-dir``.

Usage:
    python scripts/audit_dataset_split.py --config configs/operating.yml --windows-dir data/windows
"""
import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description="GAP-01 train/test day-split audit")
    ap.add_argument("--config", default="configs/operating.yml")
    ap.add_argument("--windows-dir", default="data/windows",
                    help="dir to check training window files exist")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        op = yaml.safe_load(f)
    val = op["validation"]
    train = val.get("train_days", [])
    held = val.get("days", [])

    train_keys = {(d["symbol"], d["date"]) for d in train}
    held_keys = {(d["symbol"], d["date"]) for d in held}
    overlap = train_keys & held_keys

    problems = []
    if overlap:
        problems.append(f"LEAK: held-out day(s) present in training set: {sorted(overlap)}")
    else:
        print(f"OK: {len(train)} train days and {len(held)} held-out days are disjoint at (symbol, date).")

    # Training-input window files must exist.
    win_dir = Path(args.windows_dir)
    for d in train:
        f = d.get("file")
        if f and not (win_dir / f).exists():
            problems.append(f"MISSING training window file: {win_dir / f}")

    if problems:
        for p in problems:
            print("ERROR:", p, file=sys.stderr)
        return 1
    print(f"OK: all {len(train)} training window files exist under {win_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
