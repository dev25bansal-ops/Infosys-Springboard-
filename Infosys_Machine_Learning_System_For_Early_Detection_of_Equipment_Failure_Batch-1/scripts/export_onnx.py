#!/usr/bin/env python3
"""NEW-03: export the Stage-3 TCN to ONNX + verify + benchmark.

The TCN is a pure causal Conv1d stack, so it exports cleanly. The ONNX file is a
plain (non-pickle) model artifact — it inherently sidesteps the weights_only
(RCE) concern — and runs on onnxruntime, which is typically several x faster than
the raw PyTorch forward (~11ms CPU).

Verifies the exported ONNX output matches PyTorch, then times onnxruntime vs
torch for a mini-batch.

Usage:
    PYTHONPATH=ml python scripts/export_onnx.py \
        --model models/stage3_tcn_prod.pt --out models/stage3_tcn_prod.onnx [--b 8]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b", type=int, default=8, help="mini-batch for verify/benchmark")
    args = ap.parse_args()

    torch.serialization.add_safe_globals([TCNConfig])  # MLOPS-06
    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to("cpu")
    model.load_state_dict(st["model_state"])
    model.eval()

    dummy = torch.randn(args.b, cfg.input_dim, cfg.sequence_length)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["x"], output_names=["scores"],
        dynamic_axes={"x": {0: "batch"}, "scores": {0: "batch", 1: "time"}},
        opset_version=14,
        dynamo=False,  # legacy TorchScript exporter — no onnxscript dep
    )
    print(f"[onnx] exported -> {args.out}")

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {"x": dummy.numpy()})[0]
    with torch.no_grad():
        y_torch = model(dummy).numpy()
    diff = float(np.abs(y_ort - y_torch).max())
    print(f"[onnx] max |ort - torch| = {diff:.2e}  ({'OK' if diff < 1e-5 else 'MISMATCH'})")

    # benchmark onnxruntime vs torch (CPU)
    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"x": dummy.numpy()})
    ort_ms = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            model(dummy)
    tor_ms = (time.perf_counter() - t0) / reps * 1000
    print(f"[onnx] onnxruntime: {ort_ms:.3f} ms/batch of {args.b}  | torch: {tor_ms:.3f} ms  "
          f"({tor_ms / ort_ms:.1f}x)")
    return 0 if diff < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())