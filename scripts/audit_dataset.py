#!/usr/bin/env python3
"""RSR-16: checkpoint/dataset provenance audit.

Verifies that a trained checkpoint's recorded provenance is honest and that its
training inputs do not leak the held-out validation days.

New-style checkpoints (trained by train_tcn_windows.py after RSR-09) carry a
``provenance`` dict: {trainer, seed, torch_version, input_files, label_mode,
normalize}. This script:
  1. reads the provenance (if present),
  2. checks ``label_mode`` is wall-clock (RSR-02) and ``normalize`` is
     rolling-z (BUG-03),
  3. cross-checks the training window files against configs/operating.yml's
     held-out validation days — any overlap is a train/test leak.

Legacy checkpoints (e.g. the current stage3_tcn_prod.pt, trained before
stamping) have no provenance: they are reported as unstamped (exit 0 with a
warning — the day-level split is still enforced by scripts/audit_dataset_split.py).

Usage:
    PYTHONPATH=ml python scripts/audit_dataset.py \
        --model models/stage3_tcn_prod.pt --config configs/operating.yml
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from flash_crash_watchdog.models.stage3_tcn import TCNConfig


def main() -> int:
    ap = argparse.ArgumentParser(description="RSR-16 checkpoint provenance audit")
    ap.add_argument("--model", required=True, help="checkpoint .pt")
    ap.add_argument("--config", default="configs/operating.yml")
    args = ap.parse_args()

    torch.serialization.add_safe_globals([TCNConfig])  # MLOPS-06
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    prov = state.get("provenance") if isinstance(state, dict) else None

    with open(args.config, encoding="utf-8") as f:
        op = yaml.safe_load(f)
    held = {(d["symbol"], d["date"]) for d in op["validation"]["days"]}

    problems = []
    model = Path(args.model).name

    if not isinstance(prov, dict):
        print(f"[audit] {model}: NO provenance stamp (legacy checkpoint). "
              f"Recorded sources cannot be verified; "
              f"day-level split enforced by audit_dataset_split.py instead.")
        return 0

    print(f"[audit] {model}: provenance present")
    for k in ("trainer", "seed", "torch_version", "label_mode", "normalize"):
        if k in prov:
            print(f"    {k}: {prov[k]}")
    files = prov.get("input_files", [])
    print(f"    input_files: {files}")

    if prov.get("label_mode") != "wall-clock":
        problems.append("label_mode is not wall-clock (RSR-02 violated)")
    if prov.get("normalize") != "rolling-z-500":
        problems.append("normalize is not rolling-z-500 (BUG-03 violated)")

    # Infer which (symbol, date) each training window covers from the filename.
    for f in files:
        f = Path(f).name
        for sym, date in held:
            # npz names like LUNAUSDT_2022-05-10_slice_norm500.npz
            if sym in f and date in f:
                problems.append(f"LEAK: training input '{f}' matches held-out ({sym}, {date})")

    if problems:
        for p in problems:
            print(f"    ERROR: {p}", file=sys.stderr)
        return 1
    print("[audit] OK: provenance consistent; no training/held-out overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())