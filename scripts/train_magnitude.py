#!/usr/bin/env python3
"""Train the Stage-3 magnitude head (regress forward min-drop-%).

Reuses the existing TCNDetector architecture: its final Linear+→Sigmoid head
outputs ~[0,1], so it can serve as the scaled-magnitude predictor directly.
We start from models/stage3_tcn_prod.pt, freeze the conv backbone, and re-train
only the head with Huber loss against the scaled forward min-drop-% labels.

Usage:
    PYTHONPATH=ml python scripts/train_magnitude.py \
        --crash data/mw/BTC_0519_val_mag.npz \
        --normal data/mw/BTC_2024_0116_norm_mag.npz \
        --base models/stage3_tcn_prod.pt --out models/stage3_tcn_magnitude.pt
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402
torch.serialization.add_safe_globals([TCNConfig])

logger = logging.getLogger(__name__)


def load_npz(path):
    d = np.load(path, allow_pickle=True)  # trusted local arrays (plain .npz)
    return d["windows"], d["labels"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True, help="magnitude windows .npz (large forward-drop labels)")
    ap.add_argument("--normal", required=True, help="magnitude windows .npz (normal-day, ~0 labels)")
    ap.add_argument("--base", default="models/stage3_tcn_prod.pt")
    ap.add_argument("--out", default="models/stage3_tcn_magnitude.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # RSR-09: deterministic training
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    Xc, yc = load_npz(args.crash)
    Xn, yn = load_npz(args.normal)
    logger.info("crash windows %s (label mean %.3f), normal %s (label mean %.3f)",
                Xc.shape, yc.mean(), Xn.shape, yn.mean())

    # balanced mix: all high-magnitude crash windows + subsample of the rest
    hi_idx = np.where(yc >= 0.12)[0]           # forward drop >=0.6%
    lo_idx = np.where(yc < 0.12)[0]
    n_hi = len(hi_idx)
    n_lo = int(min(len(lo_idx), 2 * n_hi))
    n_norm = int(min(len(Xn), 2 * n_hi))
    rng = np.random.default_rng(0)
    X_hi = Xc[hi_idx]; y_hi = yc[hi_idx]
    sel_lo = rng.choice(lo_idx, size=n_lo, replace=False) if n_lo else np.array([], dtype=int)
    X_lo = Xc[sel_lo]; y_lo = yc[sel_lo]
    sel_n = rng.choice(len(Xn), size=n_norm, replace=False) if n_norm else np.array([], dtype=int)
    Xn_sel = Xn[sel_n]; yn_sel = yn[sel_n]

    X = np.concatenate([X_hi, X_lo, Xn_sel])
    y = np.concatenate([y_hi, y_lo, yn_sel])
    order = rng.permutation(len(y))[: args.max_train]
    X, y = X[order], y[order]
    logger.info("train: %d windows (label mean %.3f, hi %.0f)", len(y), y.mean(), n_hi)

    # torch.load(weights_only=True): checkpoint stores a TCNConfig dataclass;
    # trusted local artifact (see SEC-3). Finetune only the head.
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    cfg = base["config"] if isinstance(base["config"], TCNConfig) else TCNConfig(**base["config"])
    model = TCNDetector(cfg).to(args.device)
    model.load_state_dict(base["model_state"])
    for p in model.network.parameters():
        p.requires_grad_(False)

    Xt = torch.from_numpy(np.ascontiguousarray(X)).permute(0, 2, 1).float()
    yt = torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    loss_f = torch.nn.SmoothL1Loss()  # Huber

    model.train()
    n_batches = max(1, len(Xt) // args.batch)
    for ep in range(args.epochs):
        gen = torch.Generator().manual_seed(ep)
        idx = torch.randperm(len(Xt), generator=gen)
        tl = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb = Xt[b].to(args.device)
            yb = yt[b].to(args.device)
            opt.zero_grad()
            pred = model(xb)[:, -1]   # sigmoid scaled magnitude at last timestep
            loss = loss_f(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item()
        logger.info("epoch %2d  loss %.4f", ep, tl / n_batches)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model_state": model.state_dict()}, out)
    logger.info("saved magnitude model to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
