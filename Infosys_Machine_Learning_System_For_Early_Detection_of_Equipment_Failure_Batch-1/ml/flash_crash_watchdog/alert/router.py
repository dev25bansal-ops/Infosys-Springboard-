"""Alert routing — Slack / PagerDuty / generic webhook / email / console.

BUG-05/NEW-01: Slack and PagerDuty were log-only stubs. This module delivers
alerts for real over stdlib HTTP/SMTP (no extra dependencies): Slack incoming
webhook, PagerDuty Events API v2, an arbitrary JSON webhook, and optional SMTP
email. Every delivery has a failure counter and supports ``dry_run`` (log the
intended request instead of sending).
"""
from __future__ import annotations

import json
import logging
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from flash_crash_watchdog.models.stage5_bayesian import Alert

logger = logging.getLogger(__name__)

PAGERDUTY_EVENTS_V2 = "https://events.pagerduty.com/v2/enqueue"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class AlertRouter:
    """Routes alerts to console + JSONL file + optional webhooks/email."""

    def __init__(
        self,
        log_path: Optional[str | Path] = None,
        slack_webhook: Optional[str] = None,
        pagerduty_key: Optional[str] = None,
        webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        smtp_config: Optional[dict] = None,
        dry_run: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key
        self.webhook_url = webhook_url
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.smtp = smtp_config or {}
        self.dry_run = dry_run
        self.timeout = timeout
        self._alerts_sent = 0
        self._failures = 0

    def route(self, alert: Alert) -> None:
        """Route an alert to all configured destinations."""
        self._alerts_sent += 1
        # CQ-01: timezone-aware (utcfromtimestamp was deprecated in 3.12).
        ts_str = datetime.fromtimestamp(alert.timestamp_ms / 1000, tz=timezone.utc).isoformat()
        logger.warning(
            "FLASH CRASH ALERT  ts=%s  symbol=%s  posterior=%.3f  "
            "s2=%.2f s3=%.2f s4=%.2f  affected=%s",
            ts_str, alert.symbol, alert.posterior,
            alert.stage2_score, alert.stage3_score, alert.stage4_score,
            alert.affected_symbols,
        )
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp_ms": alert.timestamp_ms,
                    "iso_ts": ts_str,
                    "symbol": alert.symbol,
                    "posterior": alert.posterior,
                    "stage2_score": alert.stage2_score,
                    "stage3_score": alert.stage3_score,
                    "stage4_score": alert.stage4_score,
                    "affected_symbols": list(alert.affected_symbols or []),
                }) + "\n")

        self._deliver_slack(alert, ts_str)
        self._deliver_pagerduty(alert, ts_str)
        self._deliver_webhook(alert, ts_str)
        self._deliver_telegram(alert, ts_str)
        self._deliver_email(alert, ts_str)

    # --- delivery helpers ---------------------------------------------------

    def _post_json(self, url: str, payload: dict, label: str) -> bool:
        """POST JSON to ``url``; return True on 2xx. Records failures."""
        if self.dry_run:
            logger.info("[dry-run] would POST %s alert to %s", label, url)
            return True
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if 200 <= resp.status < 300:
                    logger.info("%s alert delivered (%s)", label, resp.status)
                    return True
                logger.error("%s alert delivery failed: HTTP %s", label, resp.status)
        except Exception as e:
            logger.error("%s alert delivery failed: %s", label, e)
        self._failures += 1
        return False

    def _deliver_slack(self, alert: Alert, ts_str: str) -> None:
        if not self.slack_webhook:
            return
        text = (
            f"FLASH CRASH ALERT — {alert.symbol}\n"
            f"posterior={alert.posterior:.3f}  s2={alert.stage2_score:.2f}  "
            f"s3={alert.stage3_score:.2f}  s4={alert.stage4_score:.2f}\n"
            f"ts={ts_str}  affected={list(alert.affected_symbols or [])}"
        )
        self._post_json(self.slack_webhook, {"text": text}, "Slack")

    def _deliver_pagerduty(self, alert: Alert, ts_str: str) -> None:
        if not self.pagerduty_key:
            return
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"Flash crash detected on {alert.symbol} (posterior {alert.posterior:.3f})",
                "source": alert.symbol,
                "severity": "critical",
                "timestamp": ts_str,
                "custom_details": {
                    "posterior": alert.posterior,
                    "stage2_score": alert.stage2_score,
                    "stage3_score": alert.stage3_score,
                    "stage4_score": alert.stage4_score,
                    "affected_symbols": list(alert.affected_symbols or []),
                },
            },
        }
        self._post_json(PAGERDUTY_EVENTS_V2, payload, "PagerDuty")

    def _deliver_webhook(self, alert: Alert, ts_str: str) -> None:
        if not self.webhook_url:
            return
        payload = {
            "event": "flash_crash_alert",
            "timestamp_ms": alert.timestamp_ms,
            "iso_ts": ts_str,
            "symbol": alert.symbol,
            "posterior": alert.posterior,
            "stage2_score": alert.stage2_score,
            "stage3_score": alert.stage3_score,
            "stage4_score": alert.stage4_score,
            "affected_symbols": list(alert.affected_symbols or []),
        }
        self._post_json(self.webhook_url, payload, "webhook")

    def _deliver_telegram(self, alert: Alert, ts_str: str) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        text = (
            f"⚠️ FLASH CRASH ALERT — {alert.symbol}\n"
            f"posterior={alert.posterior:.3f}  s2={alert.stage2_score:.2f}  "
            f"s3={alert.stage3_score:.2f}  s4={alert.stage4_score:.2f}\n"
            f"ts={ts_str}"
        )
        self._post_json(
            TELEGRAM_API.format(token=self.telegram_bot_token),
            {"chat_id": self.telegram_chat_id, "text": text, "disable_notification": False},
            "Telegram",
        )

    def _deliver_email(self, alert: Alert, ts_str: str) -> None:
        host = self.smtp.get("host")
        to_addr = self.smtp.get("to")
        if not host or not to_addr:
            return
        if self.dry_run:
            logger.info("[dry-run] would email alert to %s via %s", to_addr, host)
            return
        msg = EmailMessage()
        msg["Subject"] = f"[flash-crash] {alert.symbol} posterior={alert.posterior:.3f}"
        msg["From"] = self.smtp.get("from", "flash-crash-watchdog@localhost")
        msg["To"] = to_addr
        msg.set_content(
            f"Flash crash detected on {alert.symbol} at {ts_str}\n"
            f"posterior={alert.posterior:.3f}\n"
            f"stage2={alert.stage2_score:.2f} stage3={alert.stage3_score:.2f} "
            f"stage4={alert.stage4_score:.2f}\n"
            f"affected={list(alert.affected_symbols or [])}"
        )
        try:
            port = int(self.smtp.get("port", 587))
            with smtplib.SMTP(host, port, timeout=self.timeout) as s:
                s.starttls()
                if self.smtp.get("user"):
                    s.login(self.smtp["user"], self.smtp.get("password", ""))
                s.send_message(msg)
            logger.info("Email alert delivered to %s", to_addr)
        except Exception as e:
            logger.error("Email alert delivery failed: %s", e)
            self._failures += 1

    @property
    def alerts_sent(self) -> int:
        return self._alerts_sent

    @property
    def failures(self) -> int:
        return self._failures
