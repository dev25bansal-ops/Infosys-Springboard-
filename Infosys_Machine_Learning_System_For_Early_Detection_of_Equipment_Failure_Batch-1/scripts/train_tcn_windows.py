#!/usr/bin/env python3
"""Train the TCN on real labeled windows (no synthetic data).

Uses the windows extracted by extract_windows.py. Each window is a
(200, 17) tensor labeled 0 (normal) or 1 (crash). The TCN learns to
classify windows — detecting the real pre-crash microstructure pattern.

Usage:
    # Train on crash day windows
    python scripts/train_tcn_windows.py --data data/windows/BTCUSDT_2021-05-19_windows.npz --out models/ --epochs 50

    # Train on both crash + normal days (recommended)
    python scripts/train_tcn_windows.py \
        --data data/windows/BTCUSDT_2021-05-19_windows.npz \
        --data data/windows/BTCUSDT_2024-01-15_windows.npz \
        --out models/ --epochs 50 --device cuda
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance — focuses training on hard examples."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (B, T) — per-timestep score in [0, 1]
        # targets: (B,) — window-level label
        # Use the score at the LAST timestep as the window-level prediction
        pred = preds[:, -1].squeeze()  # (B,)
        BCE = nn.functional.binary_cross_entropy(pred, targets.float(), reduction="none")
        p_t = pred * targets + (1 - pred) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * BCE
        return loss.mean()


def load_windows(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate window data from multiple .npz files."""
    all_windows = []
    all_labels = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        windows = data["windows"]
        labels = data["labels"]
        logger.info("Loaded %s: %d windows (%d positive)", path, len(windows), np.sum(labels))
        all_windows.append(windows)
        all_labels.append(labels)
    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    logger.info("Total: %d windows (%d positive = %.1f%%)",
                len(windows), np.sum(labels), np.sum(labels) / len(labels) * 100)
    return windows, labels


def balance_windows(windows: np.ndarray, labels: np.ndarray,
                    max_neg_per_pos: float) -> tuple[np.ndarray, np.ndarray]:
    """Subsample negatives so the training set is not flooded by the majority class.

    Keeps all positives and at most ``max_neg_per_pos`` negatives per positive.
    Heavy imbalance is the usual cause of a TCN collapsing to the all-negative
    prediction (constant ~0 scores) under focal loss.
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0:
        raise ValueError("No positive windows to balance against — training set has no crash labels.")
    max_neg = int(len(pos_idx) * max_neg_per_pos)
    if len(neg_idx) > max_neg:
        rng = np.random.default_rng(0)
        keep_neg = rng.choice(neg_idx, size=max_neg, replace=False)
        idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(idx)
        windows = windows[idx]
        labels = labels[idx]
        logger.info("Balanced: %d pos + %d neg (%.1f:1)", len(pos_idx), len(keep_neg), max_neg_per_pos)
    return windows, labels


def train_tcn(
    windows: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    channels: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> TCNDetector:
    """Train the TCN on labeled windows.

    Args:
        windows: shape (N, window_size, 17)
        labels: shape (N,) — 0 or 1
        seed: RSR-09 determinism — fixed for reproducible train/val split + init.
    """
    logger.info("=" * 60)
    logger.info("TRAINING TCN ON REAL LABELED WINDOWS")
    logger.info("  Device:    %s", device)
    logger.info("  Windows:   %d", len(windows))
    logger.info("  Positive:  %d (%.1f%%)", np.sum(labels), np.sum(labels) / len(labels) * 100)
    logger.info("  Shape:     %s", windows.shape)
    logger.info("  Epochs:    %d", epochs)
    logger.info("  Batch:     %d", batch_size)
    logger.info("  Channels:  %d/layer", channels)
    logger.info("=" * 60)

    # Config
    window_size = windows.shape[1]
    input_dim = windows.shape[2]
    config = TCNConfig(
        num_channels=(channels,) * 8,
        kernel_size=3,
        input_dim=input_dim,
        dropout=0.1,
        sequence_length=window_size,
    )

    # Model
    model = TCNDetector(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # RSR-09: deterministic data
    generator = torch.Generator().manual_seed(seed)

    # Data — transpose to (N, input_dim, window_size) for Conv1d
    x = torch.FloatTensor(windows).permute(0, 2, 1).to(device)  # (N, C, T)
    y = torch.LongTensor(labels).to(device)
    dataset = TensorDataset(x, y)

    # Split 80/20
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("Train: %d windows | Val: %d windows", n_train, n_val)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)  # (B, T)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                # Predict: use last timestep score
                pred_label = (preds[:, -1].squeeze() > 0.5).long()
                val_correct += (pred_label == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | %.1fs",
                        epoch, epochs, train_loss, val_loss, val_acc * 100, elapsed)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    model.load_state_dict(best_state)
    logger.info("Best val_loss=%.4f, val_acc=%.2f%%", best_val_loss,
                max(history["val_acc"]) * 100)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TCN on real labeled windows")
    parser.add_argument("--data", action="append", required=True,
                        help="Path to .npz window file (can specify multiple)")
    parser.add_argument("--out", default="models/", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=256,
                        help="Channels per TCN layer (256 for A100, 64 for CPU)")
    parser.add_argument("--balance-ratio", type=float, default=0.0,
                        help="Cap negatives at N:1 positives (0 = no balancing)")
    parser.add_argument("--output", default="stage3_tcn_trained.pt",
                        help="Checkpoint filename under --out (default stage3_tcn_trained.pt)")
    parser.add_argument("--device", default="auto",
                        help="cuda, cpu, or auto")
    parser.add_argument("--seed", type=int, default=42,
                        help="RSR-09: RNG seed for deterministic train/val split + init")
    parser.add_argument("--eval-data", default=None,
                        help="ENH-13: an .npz held-out set (windows/labels) to score with the "
                             "trained model as a post-train eval gate (logs precision/recall/F1)")
    args = parser.parse_args()

    # RSR-09: deterministic training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        logger.info("GPU: %s (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
                    

    # Load windows
    windows, labels = load_windows(args.data)
    if args.balance_ratio and args.balance_ratio > 0:
        windows, labels = balance_windows(windows, labels, args.balance_ratio)

    # Train
    model = train_tcn(
        windows, labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        channels=args.channels,
        device=device,
        seed=args.seed,
    )

    # ENH-13: post-train eval gate — score a held-out set with the trained model
    # and report precision/recall/F1 (threshold 0.5). This makes the trainer
    # honest: a checkpoint that regresses can be rejected before it ships.
    if args.eval_data:
        from flash_crash_watchdog.data.windows import load_windows
        ew, el = load_windows([Path(args.eval_data)])
        ex = torch.FloatTensor(ew).permute(0, 2, 1).to(device)
        model.eval()
        with torch.no_grad():
            scores = model(ex)[:, -1].squeeze().cpu().numpy()
        pred = (scores > 0.5).astype(int)
        tp = int(((pred == 1) & (el == 1)).sum())
        fp = int(((pred == 1) & (el == 0)).sum())
        fn = int(((pred == 0) & (el == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        logger.info("ENH-13 eval gate (%s, n=%d, thr 0.5): precision=%.3f recall=%.3f F1=%.3f",
                    Path(args.eval_data).name, len(ew), prec, rec, f1)

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.output
    torch.save({
        "model_state": model.state_dict(),
        "config": model.config,
        # RSR-16 provenance: who/what/where trained this checkpoint.
        "provenance": {
            "trainer": "train_tcn_windows.py",
            "seed": args.seed,
            "torch_version": str(torch.__version__),  # str() — TorchVersion isn't weights_only-safe
            "input_files": [str(Path(p).name) for p in args.data],
            "label_mode": "wall-clock",  # RSR-02 (not tick-count lookahead)
            "normalize": "rolling-z-500",  # BUG-03 (matches Stage3TCN.feed)
        },
    }, model_path)
    logger.info("Model saved to %s", model_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
