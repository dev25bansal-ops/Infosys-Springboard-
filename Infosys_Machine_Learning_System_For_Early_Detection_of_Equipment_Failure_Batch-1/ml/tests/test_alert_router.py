"""Tests for real alert delivery (BUG-05 / NEW-01)."""
import json

from flash_crash_watchdog.alert.router import AlertRouter, PAGERDUTY_EVENTS_V2
from flash_crash_watchdog.models.stage5_bayesian import Alert


def _alert(**kw) -> Alert:
    base = dict(
        timestamp_ms=1700000000000,
        symbol="BTCUSDT",
        posterior=0.9,
        stage2_score=0.6,
        stage3_score=0.85,
        stage4_score=0.5,
        affected_symbols=["BTCUSDT"],
        features_snapshot={},
    )
    base.update(kw)
    return Alert(**base)


def test_jsonl_and_console(monkeypatch, tmp_path):
    log = tmp_path / "alerts.jsonl"
    router = AlertRouter(log_path=log)
    router.route(_alert())
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["symbol"] == "BTCUSDT" and rec["posterior"] == 0.9
    assert "iso_ts" in rec and rec["iso_ts"].endswith("+00:00")  # CQ-01 tz-aware


def test_slack_pd_webhook_deliver_payloads(monkeypatch):
    captured = []

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured.append((req.full_url, json.loads(req.data.decode("utf-8"))))
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    router = AlertRouter(
        slack_webhook="https://hooks.slack.com/X",
        pagerduty_key="pd-key-123",
        webhook_url="https://example.com/hook",
    )
    router.route(_alert())

    urls = [u for u, _ in captured]
    assert any("hooks.slack.com" in u for u in urls)
    assert any(u == PAGERDUTY_EVENTS_V2 for u in urls)
    assert any("example.com/hook" in u for u in urls)

    pd_payload = next(p for u, p in captured if u == PAGERDUTY_EVENTS_V2)
    assert pd_payload["routing_key"] == "pd-key-123"
    assert pd_payload["event_action"] == "trigger"
    assert pd_payload["payload"]["summary"].startswith("Flash crash detected on BTCUSDT")
    assert pd_payload["payload"]["custom_details"]["posterior"] == 0.9

    slack_payload = next(p for u, p in captured if "hooks.slack.com" in u)
    assert "FLASH CRASH ALERT" in slack_payload["text"]
    assert router.failures == 0 and router.alerts_sent == 1


def test_dry_run_does_not_send(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry_run must not POST")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    router = AlertRouter(webhook_url="https://example.com/hook", dry_run=True)
    router.route(_alert())
    assert router.failures == 0


def test_failure_counter_on_non_2xx(monkeypatch):
    class _FakeErr:
        status = 500

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _FakeErr())
    router = AlertRouter(webhook_url="https://example.com/hook")
    router.route(_alert())
    assert router.failures == 1


def test_telegram_delivers_send_message(monkeypatch):
    import json as _json
    captured = []

    class _FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured.append((req.full_url, _json.loads(req.data.decode("utf-8"))))
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    router = AlertRouter(telegram_bot_token="tok123", telegram_chat_id="chat456")
    router.route(_alert())
    assert len(captured) == 1
    url, body = captured[0]
    assert "api.telegram.org/bottok123/sendMessage" in url
    assert body["chat_id"] == "chat456"
    assert "FLASH CRASH ALERT" in body["text"]
    assert router.failures == 0


def test_telegram_noop_without_token():
    router = AlertRouter()  # no telegram config
    router.route(_alert())
    assert router.failures == 0
