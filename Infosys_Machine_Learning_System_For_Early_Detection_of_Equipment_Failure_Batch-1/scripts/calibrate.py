#!/usr/bin/env python3
"""ADV-02: calibrate the operating TCN on a held-out set + conformal FP bound.

Scores a labeled window set (npz: windows, labels) with a model checkpoint, then
reports temperature scaling, ECE (raw vs calibrated), and a conformal false-
positive threshold (score above which <= alpha of non-crash windows land).

Usage:
    PYTHONPATH=ml python scripts/calibrate.py \
        --model models/stage3_tcn_prod.pt --data data/windows/<held_out>.npz \
        [--alpha 0.05]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.windows import load_windows  # noqa: E402
from flash_crash_watchdog.eval.calibration import calibrate  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help=".npz with windows + labels (held-out/calibration set)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    torch.serialization.add_safe_globals([TCNConfig])  # MLOPS-06
    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to("cpu")
    model.load_state_dict(st["model_state"])
    model.eval()

    windows, labels = load_windows([Path(args.data)])
    x = torch.FloatTensor(windows).permute(0, 2, 1)
    with torch.no_grad():
        scores = model(x)[:, -1].squeeze().cpu().numpy()

    report = calibrate(scores, labels, alpha=args.alpha)
    print(f"model={Path(args.model).name}  set={Path(args.data).name}  n={len(scores)}")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  -> fire alerts only for score >= conformal_fp_threshold to bound FPs at {args.alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())