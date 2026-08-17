"""Deterministic-training regression test (RSR-09)."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_tcn_windows", _ROOT / "scripts" / "train_tcn_windows.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _reseed(seed: int):
    # Mirror what the trainer's main() does at startup (RSR-09).
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def test_training_is_deterministic_with_same_seed():
    ttw = _trainer()
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(64, 40, 17)).astype(np.float32)
    labels = (rng.random(64) > 0.9).astype(np.int64)
    assert labels.sum() > 0  # needs positives or the focal guard raises

    _reseed(7)
    m1 = ttw.train_tcn(windows, labels, epochs=2, batch_size=16, channels=8, device="cpu", seed=7)
    _reseed(7)
    m2 = ttw.train_tcn(windows, labels, epochs=2, batch_size=16, channels=8, device="cpu", seed=7)
    s1, s2 = m1.state_dict(), m2.state_dict()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), f"weight drifted with same seed: {k}"


def test_different_seed_gives_different_weights():
    ttw = _trainer()
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(64, 40, 17)).astype(np.float32)
    labels = (rng.random(64) > 0.9).astype(np.int64)

    _reseed(7)
    m1 = ttw.train_tcn(windows, labels, epochs=2, batch_size=16, channels=8, device="cpu", seed=7)
    _reseed(999)
    m2 = ttw.train_tcn(windows, labels, epochs=2, batch_size=16, channels=8, device="cpu", seed=999)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert any(not torch.equal(s1[k], s2[k]) for k in s1), "different seeds must not be identical"