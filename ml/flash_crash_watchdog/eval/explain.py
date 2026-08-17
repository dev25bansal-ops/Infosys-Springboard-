"""ADV-01: explanatory alert rationales (exact, dependency-free).

Stage-5 fuses the stage scores via a weighted geometric mean in log-odds:

    log_odds = sum_i  w_i * logit(s_i)     (w_i = normalized weight*confidence)
    posterior = sigmoid(log_odds)

So each stage's contribution to the alert is exactly its additive log-odds term,
relative to a neutral baseline (s=0.5 -> logit 0). ``stage_attribution`` returns
that per-stage split, and ``alert_rationale`` distils it into a human-readable
reason (which stages drove the decision). Attachable to every alert.
"""
from __future__ import annotations

import math


def _to_log_odds(p: float) -> float:
    if p is None or p != p:
        p = 0.5
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def stage_attribution(stage2: float, stage3: float, stage4: float, config) -> list[dict]:
    """Per-stage log-odds contribution to the logit-space posterior.

    Returns rows [{stage: 2|3|4, log_odds, share}] sorted by |log_odds| desc.
    ``share`` is the fraction of the total |log-odds| that the stage accounts for
    (0 when the alert is neutral / no log-odds).
    """
    weights = [
        config.stage2_weight * config.stage2_confidence,
        config.stage3_weight * config.stage3_confidence,
        config.stage4_weight * config.stage4_confidence,
    ]
    total_w = sum(weights)
    w = [x / total_w for x in weights]
    scores = [stage2, stage3, stage4]
    los = [w[i] * _to_log_odds(scores[i]) for i in range(3)]
    total_abs = sum(abs(x) for x in los)
    rows = []
    for i in range(3):
        rows.append({
            "stage": i + 2,
            "log_odds": round(los[i], 4),
            "share": round(abs(los[i]) / total_abs, 4) if total_abs else 0.0,
        })
    rows.sort(key=lambda r: -abs(r["log_odds"]))
    return rows


def alert_rationale(stage2: float, stage3: float, stage4: float, config, top_n: int = 2) -> dict:
    """A compact, human-readable rationale for an alert.

    Returns {drivers: [stage numbers], text, per_stage}. Positive log-odds push
    toward an alert; negative pull back toward normal.
    """
    att = stage_attribution(stage2, stage3, stage4, config)
    drivers = [a["stage"] for a in att[:top_n] if abs(a["log_odds"]) > 1e-6]
    text = "Alert driven by Stage " + ", ".join(f"S{d}" for d in drivers) if drivers else "Alert (neutral signals)"
    return {"drivers": drivers, "text": text, "per_stage": att}