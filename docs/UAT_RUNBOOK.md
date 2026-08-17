# User Acceptance Test (UAT) Runbook — Flash Crash Watchdog

**Goal:** a non-technical stakeholder can validate that the system (a) detects a
replayed crash with lead time and (b) stays quiet on a calm day — without reading
code.

## 0. Prerequisites
- Services running (`START-WINDOWS.ps1`), dashboard open at http://localhost:3000.
- A replayed crash available: `data/replay/btc-0519.json` (the corrected export).

## 1. Setup check
- [ ] Log in / register on the dashboard.
- [ ] The "live stream connected" toast appears; the score source badge is
      `TCN` (green). If `HEURISTIC`/`WARMUP`, note it (heuristic-only is a
      degraded but valid fallback).

## 2. Calm-day quiet (the anti-spam guarantee)
- [ ] Let the dashboard run on a live/normal market for **5 minutes**.
- [ ] **Expected:** zero alerts fire. If any alert fires on a clearly calm day,
      that is a **FAIL** of the trailing-vol gate (operating value 2 bps).

## 3. Crash detection (replay mode)
Use the **Replay** page to play the `btc-0519` crash day:
- [ ] Play the replay; the price chart shows the BTC crash (43k → 30k).
- [ ] **Expected:** at least one alert fires **before the steepest portion** of
      the drop (lead time measured in the alert's `ttdMs`).
- [ ] Each alert carries a reason (score, trailing-vol bps, regime `CRASH`,
      `TENSE`, or `CALM`).
- [ ] The alert is persisted even if you close and reopen the dashboard tab
      (outbox catch-up) — it appears on reload once you ack it.

## 4. Delivery (if configured)
- [ ] If `SLACK_WEBHOOK` / `PAGERDUTY_KEY` / `SMTP_TO` are set, a fired alert
      arrives on the configured channel.

## 5. Incidents & lifecycle
- [ ] An alert can be **acknowledged / dismissed / escalated** (status change is
      persisted and reloads correctly).

## 6. Health / observability
- [ ] `http://127.0.0.1:8000/health` → `model_loaded: true`.
- [ ] `http://127.0.0.1:3005/health` → `stale: false` while ticks are flowing.
- [ ] `http://127.0.0.1:8000/metrics` and `http://127.0.0.1:3005/metrics` return
      Prometheus counters (tick/alerts/score-requests totals).

## 7. Sign-off
- [ ] Calm day quiet: **passed / failed** (expect passed)
- [ ] Crash detected with lead time: **passed / failed**
- [ ] Persistence + lifecycle: **passed / failed**
- [ ] Delivery (if configured): **passed / failed** / n/a

**Acceptance:** pass on calm-day quiet **and** crash-with-lead-time; record any
failure with the alert's `score`/`trailingVolBps` so it can be triaged against
the operating point (threshold 0.5, gate 2 bps, cooldown 10s).