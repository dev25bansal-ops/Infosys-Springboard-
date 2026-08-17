"""ADV-06: funding-context feature tests (opt-in market context)."""
import json

from flash_crash_watchdog.features.funding_context import FundingContextFeatures


def _load(record_file: str, repo_root) -> list:
    return json.loads((repo_root / "data" / "more" / "context" / record_file).read_text())


def test_funding_context_hold_last_interpolation(repo_root):
    f = FundingContextFeatures()
    f.set_funding("BTCUSDT", _load("BTCUSDT_2021-05-19_funding.json", repo_root))
    # after the first funding event (0.00010000) but before the second
    c1 = f.context("BTCUSDT", 1621382400000 + 60_000)
    assert c1["f6_funding_rate_bps"] == 1.0  # 0.0001 * 10000
    # after the crash-time funding (-0.00089697): deeply negative + big change
    c2 = f.context("BTCUSDT", 1621440000018 + 60_000)
    assert abs(c2["f6_funding_rate_bps"] - (-8.97)) < 0.01
    assert abs(c2["f6_funding_change_bps"] - (-8.97 - 3.58)) < 0.02  # vs the prior rate
    # unknown symbol -> zeros (safe no-op)
    c3 = f.context("NOPE", 1621440000018)
    assert c3["f6_funding_rate_bps"] == 0.0


def test_funding_context_no_data_is_zero():
    f = FundingContextFeatures()
    assert f.context("BTCUSDT", 0)["f6_funding_rate_bps"] == 0.0