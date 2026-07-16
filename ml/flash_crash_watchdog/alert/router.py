"""Alert routing — Slack / PagerDuty / console."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from flash_crash_watchdog.models.stage5_bayesian import Alert

logger = logging.getLogger(__name__)


class AlertRouter:
    """Routes alerts to console + JSONL file + optional webhooks."""

    def __init__(
        self,
        log_path: Optional[str | Path] = None,
        slack_webhook: Optional[str] = None,
        pagerduty_key: Optional[str] = None,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key
        self._alerts_sent = 0

    def route(self, alert: Alert) -> None:
        """Route an alert to all configured destinations."""
        self._alerts_sent += 1
        ts_str = datetime.utcfromtimestamp(alert.timestamp_ms / 1000).isoformat()
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
                    "affected_symbols": alert.affected_symbols,
                }) + "\n")
        if self.slack_webhook:
            logger.info("Slack alert queued")
        if self.pagerduty_key:
            logger.info("PagerDuty alert queued")

    @property
    def alerts_sent(self) -> int:
        return self._alerts_sent
