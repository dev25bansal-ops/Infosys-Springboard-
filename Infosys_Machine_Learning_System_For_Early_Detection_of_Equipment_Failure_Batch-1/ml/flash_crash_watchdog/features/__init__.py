"""Feature engineering orchestrator.

Combines all 5 feature families into a single 20-dim feature vector
per tick. Total extraction latency: ~2 ms.
"""
from __future__ import annotations

from flash_crash_watchdog.features.cross_symbol import CrossSymbolFeatures
from flash_crash_watchdog.features.depth_imbalance import DepthImbalanceFeatures
from flash_crash_watchdog.features.flow_toxicity import FlowToxicityFeatures
from flash_crash_watchdog.features.price_action import PriceActionFeatures
from flash_crash_watchdog.features.volatility import VolatilityFeatures
from flash_crash_watchdog.tick import Tick


# Ordered list of all 20 feature names — used by models for indexing
FEATURE_NAMES = [
    # F1 — Price & Action
    "f1_mid_velocity_50ms",
    "f1_mid_velocity_200ms",
    "f1_micro_price",
    "f1_trade_arrival_rate",
    "f1_cancel_to_trade_ratio",
    # F2 — Depth & Imbalance
    "f2_bid_depth_10",
    "f2_ask_depth_10",
    "f2_obi_10",
    "f2_weighted_mid_10",
    "f2_depth_slope",
    # F3 — Flow & Toxicity
    "f3_vpin",
    "f3_kyle_lambda",
    "f3_effective_spread_bps",
    "f3_realized_spread_bps",
    # F4 — Volatility
    "f4_realized_vol_1s",
    "f4_variance_ratio",
    "f4_garman_klass",
    # F5 — Cross-Symbol
    "f5_pairwise_correlation",
    "f5_lead_lag_coefficient",
    "f5_cointegration_residual",
]

# Which families each cascade stage consumes
STAGE_FEATURES = {
    1: ["f1_mid_velocity_50ms", "f1_mid_velocity_200ms", "f2_obi_10"],  # statistical pre-filter
    2: FEATURE_NAMES[:10],  # isolation forest on F1+F2
    3: FEATURE_NAMES[:17],  # TCN on F1-F4
    4: FEATURE_NAMES[17:],  # transformer on F5
}


class FeatureExtractor:
    """Extracts all 20 features per tick, maintaining rolling state."""

    def __init__(self) -> None:
        self._f1 = PriceActionFeatures()
        self._f2 = DepthImbalanceFeatures()
        self._f3 = FlowToxicityFeatures()
        self._f4 = VolatilityFeatures()
        self._f5 = CrossSymbolFeatures()

    def extract(self, tick: Tick) -> dict:
        """Compute the full feature vector for this tick."""
        features = {}
        features.update(self._f1.update(tick))
        features.update(self._f2.update(tick))
        features.update(self._f3.update(tick))
        features.update(self._f4.update(tick))
        features.update(self._f5.update(tick))
        tick.features = features
        return features

    def update_reference_symbol(self, symbol: str, timestamp_ms: int, mid_price: float) -> None:
        """Feed reference symbol data for cross-symbol features (F5)."""
        self._f5.update_reference(symbol, timestamp_ms, mid_price)
