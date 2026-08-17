"""Stage 3 — Temporal Convolutional Network (TCN).

8-layer dilated causal convolution network.
Receptive field: ~500 ms. Detects collective temporal anomalies.
Latency: ~8 ms (GPU). Pass-through: ~40%.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)


def normalize_z(f: np.ndarray, window: int) -> np.ndarray:
    """Rolling-z transform (sample std, min_periods=1, constant->0).

    THE shared offline normalization (BUG-03). Matches the training-time
    transform (extract_windows_contiguous --normalize) and the streaming
    ``Stage3TCN._normalize``: for each row ``i``, z-score row ``i`` against the
    mean/std of rows ``[max(0, i-window+1), i]``. Constant features map to 0.
    Every offline backtest must feed this so offline scores match the online
    ``Stage3TCN.feed`` path.

    Args:
        f: (N, D) raw feature matrix.
        window: rolling normalization window (500 for this project).
    Returns:
        (N, D) float32 rolling-z matrix.
    """
    pdf = pd.DataFrame(f)
    mean = pdf.rolling(window, min_periods=1).mean()
    std = pdf.rolling(window, min_periods=1).std()
    return ((pdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)


class FocalLoss(nn.Module):
    """Focal loss for the heavily class-imbalanced crash label.

    Down-weights easy negatives so training concentrates on the rare "crash"
    windows. Compared against the sigmoid score at the LAST timestep of each
    window (the window-level anomaly decision).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (B, T) sigmoid scores; targets: (B,) 0/1 window labels.
        pred = preds[:, -1].squeeze().float()
        tgt = targets.float()
        if pred.ndim == 0:
            pred = pred.unsqueeze(0)
        bce = nn.functional.binary_cross_entropy(pred, tgt, reduction="none")
        p_t = pred * tgt + (1 - pred) * (1 - tgt)
        alpha_t = self.alpha * tgt + (1 - self.alpha) * (1 - tgt)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

# 17 features consumed by Stage 3 (F1-F4)
STAGE3_FEATURES = [
    "f1_mid_velocity_50ms", "f1_mid_velocity_200ms", "f1_micro_price",
    "f1_trade_arrival_rate", "f1_cancel_to_trade_ratio",
    "f2_bid_depth_10", "f2_ask_depth_10", "f2_obi_10",
    "f2_weighted_mid_10", "f2_depth_slope",
    "f3_vpin", "f3_kyle_lambda", "f3_effective_spread_bps", "f3_realized_spread_bps",
    "f4_realized_vol_1s", "f4_variance_ratio", "f4_garman_klass",
]


@dataclass
class TCNConfig:
    num_channels: tuple = (64, 64, 64, 64, 64, 64, 64, 64)  # 8 layers
    kernel_size: int = 3
    input_dim: int = 17
    dropout: float = 0.1
    sequence_length: int = 500  # 500 ms receptive field at 1ms tick
    threshold: float = 0.5      # Stage-3 pass gate (canonical operating value; see pipeline.yml + live mini-stream)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TCNConfig":
        """Build a config from a model YAML file.

        Accepts either a flat file (configs/tcn_baseline.yml) or one with a
        ``stage3:`` section (configs/pipeline.yml); training/pretraining keys
        are ignored here.
        """
        import yaml

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict) and "stage3" in cfg:
            cfg = cfg["stage3"]
        channels = cfg.get("num_channels", (64,) * 8)
        if isinstance(channels, (list, tuple)):
            channels = tuple(int(c) for c in channels)
        return cls(
            num_channels=channels,
            kernel_size=int(cfg.get("kernel_size", 3)),
            input_dim=int(cfg.get("input_dim", 17)),
            dropout=float(cfg.get("dropout", 0.1)),
            sequence_length=int(cfg.get("sequence_length", 500)),
            threshold=float(cfg.get("threshold", 0.5)),
        )


class Chomp1d(nn.Module):
    """Trim the front of a 1D tensor to make the conv causal.

    Removes `padding` timesteps from the end (since Conv1d with padding
    appends to both sides, we keep the first `seq_len` timesteps).
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single dilated causal convolution block with residual connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.chomp2 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNDetector(nn.Module):
    """Temporal Convolutional Network for crash-window classification.

    Trained with supervised focal loss on labeled windows
    (see ``train_on_windows``); the decision is the sigmoid score at the
    most recent timestep of each window.
    """

    def __init__(self, config: TCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()
        layers = []
        num_levels = len(self.config.num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = self.config.input_dim if i == 0 else self.config.num_channels[i - 1]
            out_ch = self.config.num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, self.config.kernel_size,
                                         dilation=dilation, dropout=self.config.dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(self.config.num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: shape (batch, input_dim, sequence_length)
        Returns:
            scores: shape (batch, sequence_length), per-timestep anomaly score in [0, 1]
        """
        # x: (B, C, T) — already in the right format for Conv1d
        out = self.network(x)  # (B, C_last, T)
        out = out.transpose(1, 2)  # (B, T, C_last)
        scores = self.sigmoid(self.head(out)).squeeze(-1)  # (B, T)
        return scores

    def score(self, feature_window: np.ndarray) -> float:
        """Score a single feature window.

        Args:
            feature_window: shape (sequence_length, input_dim)
        Returns:
            anomaly_score in [0, 1]
        """
        if feature_window.shape[0] < 2:
            return 0.0
        # Must match the model's device (model may be on cuda while the window is host memory).
        dev = next(self.parameters()).device
        x = torch.FloatTensor(feature_window).T.unsqueeze(0).to(dev)
        with torch.no_grad():
            scores = self.forward(x)
        # Return the score at the last (most recent) timestep
        return float(scores[0, -1].item())

    @classmethod
    def from_config(cls, source: "TCNConfig | str | Path") -> "TCNDetector":
        """Build a TCNDetector from a TCNConfig or a YAML config path."""
        if isinstance(source, TCNConfig):
            return cls(source)
        return cls(TCNConfig.from_yaml(source))

    def train_on_windows(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        val_windows: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        device: str = "auto",
        seed: int = 0,
    ) -> dict:
        """Supervised classification training on labeled windows (focal loss).

        Args:
            windows: float array (N, window_size, input_dim)
            labels: int array (N,) — 0 normal / 1 crash
            val_windows / val_labels: optional holdout; defaults to an 80/20
                random split of ``windows``.

        Returns:
            history dict with ``train_loss``/``val_loss``/``val_acc`` lists.
            The best (lowest-val-loss) weights are restored onto this model
            before returning, and the model is left in eval() mode.
        """
        import time
        from torch.utils.data import DataLoader, TensorDataset

        if windows.ndim != 3:
            raise ValueError(f"windows must be (N, seq_len, input_dim), got {windows.shape}")
        if labels.ndim != 1 or len(labels) != len(windows):
            raise ValueError(
                f"labels must be 1-D with len == len(windows) ({len(windows)}), got {labels.shape}"
            )
        n_pos = int(np.sum(labels))
        if n_pos == 0:
            raise ValueError(
                "No positive (crash) labels in the training set. The TCN is a crash classifier "
                "and needs labeled windows — run scripts/extract_windows.py on crash-day data "
                "(e.g. data/parquet/BTCUSDT_2021-05-19.parquet) and point --data at the resulting .npz."
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        x = torch.FloatTensor(windows).permute(0, 2, 1)  # (N, C, T) for Conv1d
        y = torch.LongTensor(labels.astype(np.int64))
        dataset = TensorDataset(x, y)

        if val_windows is not None and val_labels is not None:
            vx = torch.FloatTensor(val_windows).permute(0, 2, 1)
            vy = torch.LongTensor(val_labels.astype(np.int64))
            train_ds = dataset
            val_ds = TensorDataset(vx, vy)
        else:
            n_train = int(len(dataset) * 0.8)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset,
                [n_train, len(dataset) - n_train],
                generator=torch.Generator().manual_seed(seed),
            )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        logger.info("TCN training: %d windows (%d positive = %.1f%%) | device=%s",
                    len(windows), n_pos, n_pos / max(1, len(windows)) * 100, device)

        best_val_loss = float("inf")
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            t0 = time.time()
            self.train()
            train_loss = 0.0
            n_batches = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                preds = self(bx)  # (B, T)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(1, n_batches)

            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    preds = self(bx)
                    val_loss += criterion(preds, by).item()
                    val_correct += ((preds[:, -1].squeeze() > 0.5).long() == by).sum().item()
                    val_total += len(by)
            val_loss /= max(1, len(val_loader))
            val_acc = val_correct / max(1, val_total)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%% | %.1fs",
                            epoch, epochs, train_loss, val_loss, val_acc * 100, time.time() - t0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        logger.info("Best val_loss=%.4f, val_acc=%.2f%%", best_val_loss, max(history["val_acc"]) * 100)
        return history


class TCNMagnitudeDetector(nn.Module):
    """Same TCN backbone, but the head REGRESSES forward drop magnitude.

    Backward-compatible addition to :class:`TCNDetector`: identical conv backbone
    (so the prod checkpoint's ``network.*`` weights transfer cleanly), but the
    final ``Linear`` head feeds a magnitude activation that maps the last-timestep
    embedding to a predicted drop in percent, in ``[0, mag_cap]``.

    The supervised target is the mean forward min-drop of the raw mid price out
    FORWARD_TICKS ticks after the window (>=0 => future drawdown), clipped to
    ``mag_cap`` pct (default 5.0). Loss = Huber against the same clipped scale, so
    below-cutting normal-0-drop windows are pushed toward ~0 while crash-onset
    windows are pushed toward their realized drawdown.
    """

    def __init__(self, config: TCNConfig | None = None, mag_cap: float = 5.0) -> None:
        super().__init__()
        self.config = config or TCNConfig()
        self.mag_cap = mag_cap
        layers = []
        for i in range(len(self.config.num_channels)):
            dilation = 2 ** i
            in_ch = self.config.input_dim if i == 0 else self.config.num_channels[i - 1]
            out_ch = self.config.num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, self.config.kernel_size,
                                         dilation=dilation, dropout=self.config.dropout))
        self.network = nn.Sequential(*layers)
        # Regress a scalar magnitude (NOT a binary score): no final sigmoid-as-prob.
        self.head = nn.Linear(self.config.num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-timestep predicted drop magnitude in PERCENT, within (0, mag_cap).

        Args:
            x: (B, input_dim, sequence_length)
        Returns:
            (B, sequence_length) — predicted forward drop % (soft-clamped to mag_cap).
        """
        out = self.network(x)                      # (B, C_last, T)
        out = out.transpose(1, 2)                  # (B, T, C_last)
        logit = self.head(out).squeeze(-1)         # (B, T) raw
        # Smooth saturating map to (0, mag_cap) percent; 0 drop -> ~0.
        return self.mag_cap * torch.sigmoid(logit)

    def score(self, feature_window: np.ndarray) -> float:
        """Predicted forward-drop % for a single (T, input_dim) window."""
        if feature_window.shape[0] < 2:
            return 0.0
        dev = next(self.parameters()).device
        x = torch.FloatTensor(feature_window).T.unsqueeze(0).to(dev)
        with torch.no_grad():
            pred = self.forward(x)
        return float(pred[0, -1].item())

    def train_on_magnitude(
        self,
        windows: np.ndarray,
        targets: np.ndarray,
        val_windows: np.ndarray | None = None,
        val_targets: np.ndarray | None = None,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
        device: str = "auto",
        seed: int = 0,
    ) -> dict:
        """Regression finetune of the magnitude head (Huber loss on clipped scale).

        Args:
            windows: (N, seq_len, input_dim)
            targets: (N,) realized forward drop %, already clipped to mag_cap (>=0).
            freeze_backbone: iff True, only ``self.head`` + optional BN-like params train;
                conv backbone weights stay as loaded (transfer from the prod classifier).
        """
        import time
        from torch.utils.data import DataLoader, TensorDataset

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        if freeze_backbone:
            for p in self.network.parameters():
                p.requires_grad = False
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.SmoothL1Loss()  # Huber: robust to a few fat crash tails

        x = torch.FloatTensor(windows).permute(0, 2, 1)
        y = torch.FloatTensor(np.clip(targets, 0.0, self.mag_cap))
        dataset = TensorDataset(x, y)

        if val_windows is not None and val_targets is not None:
            vx = torch.FloatTensor(val_windows).permute(0, 2, 1)
            vy = torch.FloatTensor(np.clip(val_targets, 0.0, self.mag_cap))
            train_ds, val_ds = dataset, TensorDataset(vx, vy)
        else:
            n_train = int(len(dataset) * 0.85)
            train_ds, val_ds = torch.utils.data.random_split(
                dataset, [n_train, len(dataset) - n_train],
                generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        for epoch in range(epochs):
            t0 = time.time()
            self.train()
            tl = 0.0
            nb = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = self(bx)[:, -1]
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                tl += loss.item(); nb += 1
            tl /= max(1, nb)
            self.eval()
            vloss = 0.0; nv = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    vloss += criterion(self(bx)[:, -1], by).item(); nv += 1
            vloss /= max(1, nv)
            history["train_loss"].append(tl)
            history["val_loss"].append(vloss)
            logger.info("Mag epoch %d/%d | huber train=%.4f val=%.4f | %.1fs",
                        epoch, epochs, tl, vloss, time.time() - t0)
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        logger.info("Best val Huber=%.4f", best_val)
        return history

    def save(self, path: str | Path) -> None:
        torch.save({"model_state": self.state_dict(), "config": self.config,
                    "mag_cap": self.mag_cap}, path)
        logger.info("Saved magnitude TCN to %s", path)


class Stage3TCN:
    """Wrapper that maintains the rolling feature window + calls the TCN."""

    def __init__(self, config: TCNConfig | None = None, device: str = "cpu") -> None:
        self.config = config or TCNConfig()
        self.device = device
        self.model = TCNDetector(self.config).to(device)
        self._window: list[np.ndarray] = []
        self._max_window = self.config.sequence_length
        self._ticks_processed = 0
        self._ticks_passed = 0
        self._threshold = self.config.threshold  # configurable pass gate (per-asset tuned)
        # Rolling-history buffer for feature normalization (see _normalize).
        # MUST match the training normalization window. Models are trained on
        # *_norm500 windows (rolling-z window 500) — inference uses 500 too.
        self._norm_hist: list[np.ndarray] = []
        self._norm_window = 500

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Rolling z-score of one raw feature vector (matches training-time transform).

        Streaming twin of the vectorized ``normalize_z`` (the shared offline
        transform). The TCN is trained on rolling-standardized features
        (extract_windows_contiguous --normalize), otherwise raw price-level
        features saturate the model. Constant features (rolling std ~ 0) map to 0,
        as in training.
        """
        self._norm_hist.append(vec)
        if len(self._norm_hist) > self._norm_window:
            self._norm_hist = self._norm_hist[-self._norm_window:]
        hist = np.asarray(self._norm_hist, dtype=np.float64)
        mean = hist.mean(axis=0)
        # sample std (ddof=1) to match pandas rolling().std() used in training
        std = hist.std(axis=0, ddof=1) if len(hist) >= 2 else hist.std(axis=0)
        out = np.zeros_like(vec, dtype=np.float32)
        mask = std > 1e-8
        if mask.any():
            out[mask] = ((vec[mask] - mean[mask]) / std[mask]).astype(np.float32)
        return out

    def feed(self, tick: Tick) -> None:
        """Advance the rolling feature window + normalization history on EVERY tick.

        The TCN is trained on contiguous 200-tick windows, so the window must be
        fed on every tick regardless of upstream gating. Kept separate from
        ``score_now`` so a sparse detector/gate cannot desync it.
        """
        features = tick.features
        vec = np.array([features.get(f, 0.0) for f in STAGE3_FEATURES], dtype=np.float32)
        self._window.append(self._normalize(vec))
        if len(self._window) > self._max_window:
            self._window = self._window[-self._max_window:]
        self._ticks_processed += 1

    def score_current(self) -> tuple[float, bool]:
        """Score the current (contiguous) window. Does not mutate window state.

        Returns (anomaly_score, should_pass_to_stage4).
        """
        if len(self._window) < self._max_window:
            return 0.0, False  # warmup: only score full (trained-length) windows
        window_array = np.array(self._window)
        score = self.model.score(window_array)
        should_pass = score > self._threshold
        if should_pass:
            self._ticks_passed += 1
        return score, should_pass

    def score(self, tick: Tick) -> tuple[float, bool]:
        """Feed a tick, then score the current window (kept for API compatibility)."""
        self.feed(tick)
        return self.score_current()

    def train(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        val_windows: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        epochs: int = 50,
        **kwargs: dict,
    ) -> dict:
        """Train the wrapped model with supervised focal loss on labeled windows.

        Args:
            windows: (n_samples, seq_len, input_dim) float array
            labels: (n_samples,) 0/1 int array — REQUIRED (must contain crashes)
            val_windows / val_labels: optional holdout (default 80/20 split)
            epochs: number of training epochs
            **kwargs: forwarded to ``TCNDetector.train_on_windows``.

        Raises:
            ValueError: if ``labels`` contains no positive examples (a crash
                classifier can't be learned from a normal-only day).
        """
        return self.model.train_on_windows(
            windows, labels, val_windows, val_labels,
            epochs=epochs, **kwargs,
        )

    def save(self, path: str | Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)
        logger.info("Saved TCN model to %s", path)

    def load(self, path: str | Path) -> None:
        # MLOPS-06: secure load. The checkpoint stores a TCNConfig dataclass, so
        # allowlist it and use weights_only=True (never unpickle arbitrary objects).
        torch.serialization.add_safe_globals([TCNConfig])
        data = torch.load(path, map_location=self.device, weights_only=True)
        cfg = data.get("config") if isinstance(data, dict) and data.get("config") is not None else None
        if cfg is not None:
            # Rebuild the config (and the model) from the checkpoint's own config so
            # architecture (channels/seq_len) always matches, not the caller's defaults.
            self.config = cfg if isinstance(cfg, TCNConfig) else TCNConfig(**cfg)
            self._max_window = self.config.sequence_length
            self.model = TCNDetector(self.config).to(self.device)
        # Reset runtime state so a previously-warmed Stage-3 doesn't inherit a
        # stale/non-contiguous window or normalization history.
        self._window = []
        self._norm_hist = []
        self._ticks_processed = 0
        self._ticks_passed = 0
        if isinstance(data, dict) and "model_state" in data:
            self.model.load_state_dict(data["model_state"])
        else:
            self.model.load_state_dict(data)
        self.model.eval()
        logger.info("Loaded TCN model from %s", path)

    @property
    def pass_through_rate(self) -> float:
        if self._ticks_processed == 0:
            return 0.0
        return self._ticks_passed / self._ticks_processed
