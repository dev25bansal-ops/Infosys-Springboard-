#!/usr/bin/env python3
"""Replay a historical crash through the detector (trained) at speed, then render
an overlay plot: price, Stage-3 score vs the gate threshold, trailing-vol vs the
regime-gate line, and fired alerts — the core of the AF-2 crash-scrub view.

Usage:
    python scripts/replay_crash.py --data <crash.parquet> --speed 200 --models models
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.cascade import DetectionCascade  # noqa: E402
from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet  # noqa: E402

logger = logging.getLogger(__name__)
W = 200


def trailing_vol_bps(prices: list[float], window: int = 200) -> float:
    w = prices[-window:]
    if len(w) < 50:
        return 0.0
    m = float(np.mean(w))
    return float(np.std(w) / m) * 10000.0 if m else 0.0


def render_overlay(times, prices, scores, vols, gate_bps, thr, alerts, out) -> str:
    """Write a PNG overlay: price / s3 score vs thr / trailing-vol vs gate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"matplotlib unavailable: {e}"
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t0 = times[0] if times else 0
    tmin = [(t - t0) / 1000.0 for t in times]

    axs[0].plot(tmin, prices, lw=0.8, color="#1f77b4")
    axs[0].set_ylabel("mid price")
    axs[0].grid(alpha=0.3)

    axs[1].plot(tmin, scores, lw=0.7, color="#d62728")
    axs[1].axhline(thr, color="black", ls="--", lw=1, label=f"thr={thr}")
    axs[1].set_ylabel("Stage-3 score")
    axs[1].legend(loc="upper left", fontsize=8)
    axs[1].grid(alpha=0.3)

    axs[2].plot(tmin, vols, lw=0.7, color="#2ca02c")
    axs[2].axhline(gate_bps, color="black", ls=":", lw=1, label=f"vol gate={gate_bps} bps")
    axs[2].set_ylabel("trailing vol (bps)")
    axs[2].set_xlabel("seconds since start")
    axs[2].legend(loc="upper left", fontsize=8)
    axs[2].grid(alpha=0.3)

    for at in alerts:
        x = (at - t0) / 1000.0
        axs[0].axvline(x, color="red", alpha=0.5, lw=0.8)
    fig.suptitle("Crash replay — overlay (price / Stage-3 / trailing-vol gate)")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return str(out)


async def replay(df, cascade, speed: float, out_png: str, models_dir=None) -> None:
    ticks = list(df_to_ticks(df))
    logger.info("Replaying %d ticks at %sx", len(ticks), speed)
    times, prices, scores, vols, alerts = [], [], [], [], []
    mid_history: list[float] = []
    SAMPLE = max(1, len(ticks) // 2000)  # cap sampled series to ~2000 pts

    for i, tick in enumerate(ticks):
        if i > 0:
            delay_ms = (tick.timestamp_ms - ticks[i - 1].timestamp_ms) / speed
            if 0 < delay_ms < 10_000:
                await asyncio.sleep(delay_ms / 1000)
        mp = tick.book.mid_price
        if mp is not None:
            mid_history.append(float(mp))
            if len(mid_history) > 500:
                mid_history = mid_history[-500:]
        alert = cascade.process_tick(tick)
        if alert is not None:
            alerts.append(alert.timestamp_ms)
            logger.info("ALERT at %d (posterior %.2f)", alert.timestamp_ms, alert.posterior)
        if i % SAMPLE == 0:
            times.append(tick.timestamp_ms)
            prices.append(float(mp) if mp is not None else 0.0)
            scores.append(float(cascade.s3.score_current()[0]) if len(cascade.s3._window) >= 50 else 0.0)
            vols.append(trailing_vol_bps(mid_history))
        if i % 100000 == 0:
            logger.info("  progress %d/%d alerts=%d", i, len(ticks), len(alerts))

    cascade.print_stats()
    thr = float(cascade.s3._threshold)
    gate = float(getattr(cascade, "_gate_bps", 2.0))
    png = render_overlay(times, prices, scores, vols, gate, thr, alerts, out_png)
    logger.info("Overlay plot -> %s", png)
    logger.info("Alerts fired: %d", len(alerts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="configs/pipeline.yml")
    ap.add_argument("--speed", type=float, default=200.0)
    ap.add_argument("--models", default="models", help="dir with trained checkpoints (prod/v2)")
    ap.add_argument("--gate-bps", type=float, default=2.0)
    ap.add_argument("--max-ticks", type=int, default=0, help="cap replay length (full-cascade is ~100 ticks/s)")
    ap.add_argument("--out", default="results/plots/replay_overlay.png")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_parquet(args.data)
    if args.max_ticks > 0 and len(df) > args.max_ticks:
        df = df.iloc[: args.max_ticks]
    cascade = DetectionCascade.from_config(args.config, models_dir=args.models)
    cascade._gate_bps = args.gate_bps
    asyncio.run(replay(df, cascade, args.speed, args.out, args.models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())