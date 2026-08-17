#!/usr/bin/env python3
"""Fine-tune the BTC-trained TCN on LUNA data, then re-test on LUNA crash.

Loads the BTC-trained model, fine-tunes on 1 day of LUNA windows,
then runs the backtest on the LUNA crash day.

Usage:
    python scripts/finetune_luna.py \
        --base-model models/stage3_tcn_trained.pt \
        --luna-windows data/windows/LUNAUSDT_2022-05-09_windows.npz \
        --luna-test data/parquet/LUNAUSDT_2022-05-10.parquet \
        --out models/ --epochs 20
"""
import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.models.stage3_tcn import TCNDetector, TCNConfig
torch.serialization.add_safe_globals([TCNConfig])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        pred = preds[:, -1].squeeze()
        BCE = nn.functional.binary_cross_entropy(pred, targets.float(), reduction="none")
        p_t = pred * targets + (1 - pred) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * BCE).mean()


def main() -> int:
    # RSR-09: deterministic training
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    parser = argparse.ArgumentParser(description="Fine-tune TCN on LUNA data")
    parser.add_argument("--base-model", required=True, help="BTC-trained TCN")
    parser.add_argument("--luna-windows", action="append", required=True,
                        help="LUNA windows .npz (can specify multiple)")
    parser.add_argument("--luna-test", required=True, help="LUNA crash test parquet")
    parser.add_argument("--out", default="models/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower LR for fine-tuning")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load base model
    logger.info("Loading base model from %s", args.base_model)
    data = torch.load(args.base_model, map_location=device, weights_only=True)
    config = data["config"]
    model = TCNDetector(config).to(device)
    model.load_state_dict(data["model_state"])
    logger.info("Base model loaded (BTC-trained)")

    # Load LUNA windows (multiple files)
    all_windows = []
    all_labels = []
    for path in args.luna_windows:
        logger.info("Loading LUNA windows from %s", path)
        luna_data = np.load(path, allow_pickle=True)
        all_windows.append(luna_data["windows"])
        all_labels.append(luna_data["labels"])
        logger.info("  %s: %d windows (%d positive)", path, len(luna_data["windows"]), np.sum(luna_data["labels"]))
    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logger.info("Total LUNA windows: %d (%d positive = %.1f%%)",
                len(windows), np.sum(labels), np.sum(labels)/max(1,len(labels))*100)

    # Prepare data
    x = torch.FloatTensor(windows).permute(0, 2, 1).to(device)
    y = torch.LongTensor(labels).to(device)
    dataset = TensorDataset(x, y)

    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Fine-tune with LOWER learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    logger.info("=" * 60)
    logger.info("FINE-TUNING ON LUNA DATA")
    logger.info("  Epochs: %d | LR: %s | Device: %s", args.epochs, args.lr, device)
    logger.info("=" * 60)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                pred_label = (preds[:, -1].squeeze() > 0.5).long()
                val_correct += (pred_label == batch_y).sum().item()
                val_total += len(batch_y)
        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | %.1fs",
                        epoch, args.epochs, train_loss, val_loss, val_acc * 100, time.time() - t0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    logger.info("Best val_loss=%.4f", best_val_loss)

    # Save fine-tuned model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "stage3_tcn_luna_finetuned.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, model_path)
    logger.info("Fine-tuned model saved to %s", model_path)

    # Now run backtest on LUNA crash day
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BACKTEST ON LUNA CRASH DAY")
    logger.info("=" * 60)

    import subprocess
    result = subprocess.run([
        sys.executable, str(Path(__file__).parent / "backtest_windows.py"),
        "--data", args.luna_test,
        "--model", str(model_path),
        "--output", str(out_dir.parent / "results" / "luna_finetuned_backtest.json"),
        "--max-ticks", "500000",
        "--threshold", "0.3",
    ], capture_output=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
