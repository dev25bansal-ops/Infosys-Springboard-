#!/usr/bin/env python3
"""RSR-14: reproducibility smoke — golden metrics diff in CI.

Runs the canonical evaluator (``run_validation.run_day``) on a FIXED slice of one
day at the operating point and diffs the metrics against a committed golden. Any
drift fails the run, pinning determinism of:
  - event-based crash labels   (RSR-03)
  - shared rolling-z           (BUG-03)
  - canonical TP/FP matching   (RSR-04)

The pipeline is fully deterministic (no RNG in scoring/labeling/matching), so the
golden must reproduce bit-for-bit.

Usage:
    # (re)generate the golden after an intentional pipeline change:
    PYTHONPATH=ml python scripts/repro_smoke.py --write-golden
    # verify (CI):
    PYTHONPATH=ml python scripts/repro_smoke.py
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
ROOT = Path(__file__).resolve().parent.parent

# Import run_validation (a script, not a package) so we share its run_day.
_spec = importlib.util.spec_from_file_location("run_validation", ROOT / "scripts" / "run_validation.py")
_runval = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runval)

from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="RSR-14 reproducibility smoke")
    ap.add_argument("--config", default=str(ROOT / "configs" / "operating.yml"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--data", default=str(ROOT / "data" / "parquet" / "BTCUSDT_2024-01-16.parquet"),
                    help="FIXED slice source (must stay pinned; changing it invalidates the golden)")
    ap.add_argument("--max-ticks", type=int, default=20000)
    ap.add_argument("--golden", default=str(ROOT / "results" / "golden_metrics.json"))
    ap.add_argument("--write-golden", action="store_true",
                    help="write the golden file from the current run")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        op = yaml.safe_load(f)
    op = {**op, **op.get("validation", {})}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.models_dir) / op["model"]
    torch.serialization.add_safe_globals([TCNConfig])  # secure load (MLOPS-06)
    st = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev)
    model.load_state_dict(st["model_state"])
    model.eval()

    import pandas as pd
    df = pd.read_parquet(args.data).iloc[: args.max_ticks]
    m = _runval.run_day(df, model, dev, op)
    keys = ("ticks", "crashes", "alerts", "alerts_per_hour", "true_positives",
            "false_positives", "false_negatives", "precision", "recall", "f1",
            "median_ttd_ms")
    got = {k: m[k] for k in keys}

    golden_path = Path(args.golden)
    if args.write_golden:
        payload = {"data": args.data, "max_ticks": args.max_ticks,
                   "config": args.config, "model": op["model"], "metrics": got}
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote golden -> {golden_path}")
        return 0

    if not golden_path.exists():
        print(f"ERROR: no golden at {golden_path} — run with --write-golden first.", file=sys.stderr)
        return 1
    golden = json.loads(golden_path.read_text())
    if (golden.get("data"), golden.get("max_ticks")) != (args.data, args.max_ticks):
        print(f"ERROR: golden pinned to {golden.get('data')}@{golden.get('max_ticks')} but ran "
              f"{args.data}@{args.max_ticks}. Regenerate the golden or pin the inputs.", file=sys.stderr)
        return 1
    diffs = {k: (golden["metrics"][k], got[k]) for k in keys
             if abs(golden["metrics"][k] - got[k]) > 1e-9}
    if diffs:
        print("REPRO FAILED — metrics drifted:", file=sys.stderr)
        for k, (exp, act) in diffs.items():
            print(f"  {k}: golden={exp} got={act}", file=sys.stderr)
        return 1
    print(f"REPRO OK: metrics identical to golden on {Path(args.data).name}@{args.max_ticks} ticks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
