"""ADV-06: market-context features (funding / basis) — OPT-IN, additive.

The operating TCN consumes the 17 LOB features (FEATURE_NAMES[:17]). This module
adds a market-context family (funding-rate bps + its change, plus a basis proxy
once spot-futures pairs are available) that downstream models CAN adopt at the
next retrain — it does NOT change the operating model's inputs today.

The funding data (data/more/context/<SYM>_<date>_funding.json) is 8-hourly, so the
per-tick feature is a hold-last interpolation to the current timestamp.
"""
from __future__ import annotations

from typing import List, Tuple


class FundingContextFeatures:
    """Per-symbol funding-rate context, exposed as opt-in features."""

    def __init__(self) -> None:
        # symbol -> sorted [(fundingTime_ms, rate)]
        self._series: dict[str, List[Tuple[int, float]]] = {}

    def set_funding(self, symbol: str, records) -> None:
        """Load Binance funding records: [{fundingTime, fundingRate}, ...]."""
        pts = sorted(
            (int(r.get("fundingTime", 0)), float(r.get("fundingRate")))
            for r in records if r.get("fundingRate")
        )
        if pts:
            self._series[symbol] = pts

    def context(self, symbol: str, timestamp_ms: int) -> dict:
        """Most recent funding rate (bps) + change vs the previous rate (bps).

        ``f6_basis_proxy`` stays 0 until spot-vs-futures pairs are available
        (only futures klines are downloaded today).
        """
        pts = self._series.get(symbol, [])
        if not pts:
            return {"f6_funding_rate_bps": 0.0, "f6_funding_change_bps": 0.0,
                    "f6_basis_proxy": 0.0}
        cur = pts[0][1]
        for t, r in pts:
            if t <= timestamp_ms:
                cur = r
            else:
                break
        # previous rate for the change
        idx = next((i for i, (t, r) in enumerate(pts) if r == cur), 0)
        prev = pts[idx - 1][1] if idx > 0 else cur
        return {
            "f6_funding_rate_bps": round(cur * 10_000, 3),
            "f6_funding_change_bps": round((cur - prev) * 10_000, 3),
            "f6_basis_proxy": 0.0,
        }