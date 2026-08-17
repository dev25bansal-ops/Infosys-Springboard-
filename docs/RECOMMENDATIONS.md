# Flash Crash Watchdog — Strategic Recommendations (v3 · 2026-08-09)

**Scope:** `D:\flash-crash-watchdog`. This v3 supersedes v2; it reflects the terminal state after the 2026-08-07/09 sessions.
**Companion:** `vault/Results & Performance`, `vault/Cascade & Models`, project memory.

---

## 0. Executive Summary — where the project actually is

The detector is **research-complete and validated**:

- **Cross-asset, multi-day-validated**: 6 held-out days (BTC-0519, LUNA-0510/0509, ETH-0613/0805, normal BTC-0116) at thr0.5 → crash recall 0.38–1.0, F1 up to 0.75.
- **Spam floor SOLVED** (the standing gap): a **trailing-realized-volatility regime gate** (≥2 bps) zeros calm-day alerts (11→0) with crash recall fully preserved. Wired in the batched tool and the live mini-service.
- **5-stage pipeline audited + fixed**: norm-window mismatch, hardcoded gate, untrained production models, Stage-1 self-inclusive baseline, Stage-3 starvation — all fixed; all 18 tests pass; CI added.
- **Live stack verified end-to-end** via the real `/score` API + mini-stream gate (crash → alerts, normal → 0).
- **Auth hardened**: signed HMAC session cookie (forgery rejected), sidecar CORS-restricted + localhost-bound + optional `x-api-key`.

**What remains is product/engineering polish, not model research**: config/deploy tidiness, a real hosting story, and the optional model frontier (magnitude head / market-context features — the data is now downloaded).

---

## 1. Project Analysis & Strategic Opportunities

### 1.1 Differentiable position
The project now has a **credible, measurable, spam-bounded crash-detection engine** — lead-time-with-proof on real held-out crash days, an auditable pipeline, and honest numbers. That's the moat: competitors (TradingView alerts, funding/OI dashboards, sentiment bots) don't sell reproducible pre-crash lead time with a published validation record.

### 1.2 Strategic plays (ranked by leverage)
1. **VP-A — "Pre-crash lead time, measured & reproducible"** — publish the 6-day validation table + threshold/gate settings as the trust artifact. This is the headline differentiator and it's *true* now.
2. **VP-B — Regime-aware alerts** (the trailing-vol gate as a *feature*): surface "market regime: CALM / TENSE / CRASH" in the UI, with the vol gate as an explainable rule. Great demo, honest.
3. **VP-C — Crash replay / scrub suite** — replay any crash day at ×speed with per-stage overlay (already have the data + batched tooling).
4. **VP-D — White-label non-advice monitoring** — email/Slack/PagerDuty on regime flips; non-advice framing.
5. **VP-E — Cross-asset correlation / regime view** (Stage-4's intended signal) once a multi-symbol live feed exists.

### 1.3 Instrumentation to own the niche
- p50/p95 lead-time histogram per asset (currently ~3–30 s; push toward sub-10 s).
- Alert rate per asset/hour with the cooldown+gate (episodes, not bursts).
- A published "operating point card" per asset (threshold, gate, held-out recall/F1).

---

## 2. Issues & Required Fixes (catalog — current open items)

> **Most historical blockers are DONE.** Remaining catalog:

> **Cleared 2026-08-09/10:** S2 rate-limit + password policy (in-memory limiter on login/register, min-8 password) and Q1 `eval/metrics.py` (shared metrics module) are now DONE; C1/C2/C3 (gate-consistency, sidecar token, SESSION_SECRET) were cleared in the prior pass. **Q2/Q3/Q4:** `docker-compose.yml` marked STALE/non-buildable (Dockerfiles don't exist, legacy `dashboard/` is an empty stub); canonical trainer designated **`train_tcn_windows.py`** (produces the working `stage3_tcn_*.pt`) with `train_magnitude.py` as the extension (older `train_models/gpu/tcn` are legacy); web `*.gitignore` now excludes `.env` so env files won't be tracked. Remaining open: only the deliberately-left legacy `dashboard/` stub (harmless, two placeholder files).

### Software/config
- **C1 — Gate/threshold consistency (LOW · P2 · 0.25 d).** `mini-services/binance-stream` gates at 0.5 (`SYMBOL_ALERT_THRESHOLD`), the checkpoint `TCNConfig.threshold` is 0.6, `pipeline.yml` says 0.5. All currently work (the server returns raw scores), but one audited constant would prevent confusion. **Fix:** pick one canonical gate (0.5) and reference it in the three places.
- **C2 — Mini-stream must send `x-api-key` if `TCN_API_KEY` is enabled (LOW · P2 · 0.25 d).** Documented; add the header to the mini-stream fetch when the env is set.
- **C3 — `SESSION_SECRET` must be set in prod (LOW · P2 · 0.25 d).** The dev fallback is insecure-by-flag.

### Performance
- **P1 — Full-cascade backtest slow (MED · P2 · 1–2 d).** The batched tool solves validation (1 min/day); the full `run_backtest_trained.py` path is still slow because it scores Stage-3 per tick.
- **P2 — No vectorized FeatureExtractor (MED · P3 · 2–3 d).** The real 5–10× on extraction; touch risk moderate. Do only if batch validation throughput matters.

> **✅ Resolved by measurement (2026-08-09).** Profiled the full-cascade path: feature extraction runs at **~4,400 ticks/s**, but per-tick Stage-3 feed+score at **~96/s** (model.forward 11 ms CPU). **The bottleneck is the per-tick Stage-3 model forward, not the FeatureExtractor** — so "vectorize the extractor for 5–10×" is misdirected (extraction is already ~100× faster than scoring). The real lever is **batched GPU Stage-3 inference, which `run_backtest_batched.py` already provides** (1 min/day validation). **P1/P2 effectively closed; no risky extractor rewrite warranted.** The live path scores per-tick at live rates, which is correct.

### Security — mostly done
- **S1 ✅ signed sessions; S3/S4 ✅ sidecar hardened (localhost, CORS restricted, optional token).**
- **S2 — rate limiting / password policy still absent (MED · P1 · 1 d).** Add simple in-memory rate limit on login/register + a min password length. Priority if the app goes public.
- **S5 — `torch.load(weights_only=False)` across train/eval (MED · P2 · 1 d).** Checkpoints store a `TCNConfig` dataclass; migrating config to a plain dict would allow `weights_only=True`. All checkpoints are local/trusted; low urgency.

### Code quality / debt
- **Q1 — `eval/metrics.py` still missing** (README claims it) — consolidate the duplicated evaluators. (MED · P2 · 1–2 d)
- **Q2 — Legacy `dashboard/` empty stub + stale `docker-compose.yml` (no Dockerfiles) (LOW · P2 · 0.5 d).** Remove or fix.
- **Q3 — One canonical trainer** vs the fragmented `train_*` scripts. (LOW · P2 · 1 d)
- **Q4 — `.env` committed** (no secrets today; move to real env). (LOW · P2 · 0.25 d)

### Prioritized backlog (only 6 items)
| # | Item | P | Sev | Effort |
|---|---|---|---|---|
| 1 | S2 rate-limit/password policy | P1 | MED | 1 d |
| 2 | C1 canonical gate constant | P2 | LOW | 0.25 d |
| 3 | Q1 `eval/metrics.py` | P2 | MED | 1–2 d |
| 4 | C3/C2 env notes + token header | P2 | LOW | 0.25 d |
| 5 | Q2/Q4 repo hygiene (docker/stub/.env) | P2 | LOW | 0.5 d |
| 6 | S5 weights_only migration | P2 | MED | 1 d |

---

## 3. Enhancements & Modifications (component-level)

- **Inference sidecar (`ml-inference/server.py`)**: already serves the normalized model; add `/version` (model+threshold+gate) and a `model_key` in the response so the UI can show what's live. (0.5 d)
- **Mini-stream (`index.ts`)**: surface `trailingVolBps` in the `tick`/`alert` payloads (currently computed internally) so the UI can display "regime". Wire the `x-api-key` header behind the env. (0.5 d)
- **Batched backtest**: emit a machine-readable CSV/JSON row (model, gate, thr, per-day table) to feed the leaderboard (§5). (0.5 d)
- **Config**: make the operating point (model `prod.pt`, thr 0.5, gate 2 bps, cooldown 10 s) a single documented `configs/operating.yml` consumed by the batched tool + the live path, so a fresh deploy gets the audited state. (0.5–1 d)

---

## 4. Advanced Features (differentiators)

- **AF-1 — Correlation-breakdown early-warning.** Wire the (currently disabled) Stage-4 multi-symbol feed: alert when an anchor's pairwise correlation with the basket collapses. Requires a live multi-symbol mini-stream. (~3–5 d)
- **AF-2 — Crash-replay / scrub suite.** Load any crash day in the UI, ×1–×1000 scrubber, per-stage overlay + the trailing-vol gate curve. Strong demo + training tool. (~3–4 d)
- **AF-3 — Regime-aware UI state.** "Market regime: CALM/TENSE/CRASH" driven by the trailing-vol gate + Stage-1/2; the alert reason card shows `score`, `trailing-vol`, `threshold`, `gate`. Trust + usability. (~1–2 d)
- **AF-4 — Magnitude-aware head (now that futures/funding data is downloaded).** The proven-fail mode was a head trained only on spot windows; retrain the magnitude head WITH the funding/futures klines (`data/more/context/`) so "genuine deleveraging" vs flat-noise has a real input. Uncertain payoff (~2–3 d).
- **AF-5 — Cross-exchange basis alarm.** Use the downloaded futures klines to alert on basis blowout (futures-vs-spot divergence) as an independent cross-check on the TCN. (~1–2 d)

---

## 5. New Additions

- **M-1 — Model/operating-point registry.** Stamp every Alert with `model_key`, threshold, gate-bps, commit; store rolled-up per-day stats → the audit + leaderboard artifact for VP-A.
- **M-2 — Alert destinations** (Slack/PagerDuty/webhook) — `alert/router.py` exists and is wired to CLI; productize with per-symbol channels + severity. (1 d)
- **M-3 — Leaderboard page in the dashboard** — the 6-day held-out table + latest backtest rows, rendered live. Turns honesty into a feature.
- **M-4 — Funding/basis context features** integrated into `FeatureExtractor` (the 2021–2024 funding + 1h futures klines are downloaded; align to per-tick for crash-vs-normal separation tests).
- **M-5 — Real deployment** — Dockerfiles (proxy/ml/web), a `docker-compose.prod` with auth + Caddy/Traefik, `.env` hygiene. (1–2 d)
- **M-6 — `make data` one-command data load** (download→convert→windows→train) so the README equals truth.

---

## 6. Verification & Testing Strategy (current state + gaps)

**In place:** `ml/tests/` (18 pass: itertuples, feed/score, no-positive guard, gate-fix regression, cooldown); `.github/workflows/ci.yml`; `scripts/verify_live_stack.py` (E2E: crash→alerts, normal→0 via real `/score`); `scripts/run_backtest_batched.py` (1 min/day validation); `scripts/tcn_score_diag.py` (GPU threshold sweep).

**Gaps to close:**
- **Web typecheck/build** — the TS edits (session signing, trailing-vol gate, thresholds) haven't been through a full `next build`/`tsc`; add a `typecheck` job to CI. (P1)
- **Security regression test** — assert a raw-`User.id` cookie is rejected and a signed one is accepted (Node test alongside the routes); rate-limit test after S2. (P1)
- **Perf test** — `/score` p95 (target <20 ms) under load; add to CI as a smoke.
- **UAT/demo** — one scripted live run (Binance connect → dashboard shows alerts on a seeded high-vol event; calm day shows none). (P1)

---

## 7. Beyond the Six Categories

- **Ops/cost**: the stack is cheap (one GPU inference + two small services + SQLite). A weekly retrain job (pull new windows → train → re-validate) keeps the leaderboard fresh.
- **Governance**: the vault + `docs/RECOMMENDATIONS.md` hold the actual-results narrative; give the README a "Status" page (runs, measured numbers, known limits) separate from aspirational claims.
- **Honest limit to state**: the vol gate suppresses *calm-day* chatter; on genuinely volatile non-crash days, alerts can return (the gate can't separate a crash from a volatile trend). Correct behavior, but don't over-claim "zero spam" — claim "zero spam on calm days."
- **Compliance**: keep it non-advice, monitoring-only; add SEC/MiFID language before any automated action; the cooldown+gate already cap false-alert cost.
- **Maintain this doc**: recompute §0's table after any model change; keep the §2 backlog current.

---

*Keep alongside `vault/`; refresh §0 and §2 after each deployment step.*