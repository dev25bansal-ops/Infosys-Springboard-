"""ADV-01: stage-attribution / alert-rationale tests."""
from flash_crash_watchdog.eval.explain import alert_rationale, stage_attribution
from flash_crash_watchdog.models.stage5_bayesian import Stage5Config


def test_stage3_dominates_when_high():
    cfg = Stage5Config()
    att = stage_attribution(0.4, 0.99, 0.5, cfg)
    top = max(att, key=lambda r: abs(r["log_odds"]))
    assert top["stage"] == 3
    # s3 at 0.99 is the decisive driver
    assert top["share"] > 0.5


def test_shares_sum_to_one():
    cfg = Stage5Config()
    att = stage_attribution(0.9, 0.9, 0.9, cfg)
    total = sum(a["share"] for a in att)
    assert abs(total - 1.0) < 1e-6


def test_alert_rationale_lists_driver_stages():
    cfg = Stage5Config()
    r = alert_rationale(0.3, 0.98, 0.5, cfg)
    assert "S3" in r["text"]
    assert r["drivers"] and 3 in r["drivers"]
    assert "per_stage" in r and len(r["per_stage"]) == 3


def test_neutral_signals_yield_no_drivers():
    cfg = Stage5Config()
    r = alert_rationale(0.5, 0.5, 0.5, cfg)
    assert r["drivers"] == []