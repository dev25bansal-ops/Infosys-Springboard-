"""CLI smoke tests (TST-09): argument contracts + the README quickstart shape."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ML = ROOT / "ml"


def _run(*args):
    env = dict(os.environ, PYTHONPATH=str(ML))
    return subprocess.run(
        [sys.executable, "-m", "flash_crash_watchdog.cli", *args],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
        timeout=120,
    )


def test_backtest_help_shows_model_alias():
    r = _run("backtest", "--help")
    assert r.returncode == 0, r.stderr
    # ENH-05: --model is now a valid backtest arg (README quickstart uses it).
    assert "--model MODEL" in r.stdout or "--model" in r.stdout


def test_live_help_shows_source_and_dry_run():
    r = _run("live", "--help")
    assert r.returncode == 0, r.stderr
    assert "--source" in r.stdout
    assert "--dry-run" in r.stdout  # NEW-01


def test_backtest_requires_data():
    r = _run("backtest")
    assert r.returncode != 0  # argparse: --data is required


def test_backtest_accepts_model_argument():
    # The documented README command shape (--model) must parse past argparse —
    # it fails only on the (nonexistent) data file, not on "unrecognized arg".
    r = _run("backtest", "--data", "nope.parquet", "--model", "configs/tcn_baseline.yml")
    assert r.returncode != 0  # fails because the data file is missing
    assert "unrecognized arguments" not in r.stderr, "the --model alias must be accepted"


def test_train_help_shows_model_config():
    r = _run("train", "--help")
    assert r.returncode == 0, r.stderr
    assert "--model" in r.stdout