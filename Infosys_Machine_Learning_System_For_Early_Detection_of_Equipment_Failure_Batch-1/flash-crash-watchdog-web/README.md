# Flash Crash Watchdog - Web Dashboard (v2.1)

Real-time ML-powered flash crash detection web app. Connects to Binance WebSocket, streams BTC/USDT order book data, scores each tick with the **trained PyTorch TCN model** (or a built-in heuristic fallback), and fires alerts to a dashboard.

**Stack:** Next.js 16 - TypeScript - Prisma + SQLite - Socket.io - Tailwind CSS 4 - shadcn/ui - Sonner toasts - bcrypt auth - **Python FastAPI + PyTorch TCN sidecar**.

---

## Quick Start (Windows, 3 commands)

```powershell
# 1. First-time setup (installs all npm + Python deps, creates SQLite DB)
powershell -ExecutionPolicy Bypass -File .\SETUP-WINDOWS.ps1

# 2. Launch everything (opens 3 PowerShell windows + browser)
powershell -ExecutionPolicy Bypass -File .\START-WINDOWS.ps1

# 3. To stop all services later
powershell -ExecutionPolicy Bypass -File .\STOP-WINDOWS.ps1
```

After running SETUP once, you can use the shorter form (if you've set execution policy):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # one-time
.\SETUP-WINDOWS.ps1
.\START-WINDOWS.ps1
.\STOP-WINDOWS.ps1
```

The START script opens:
- **Window 1**: Python TCN sidecar on `http://localhost:8000`
- **Window 2**: Binance stream + Socket.io on `http://localhost:3003`
- **Window 3**: Next.js dashboard on `http://localhost:3000`

...then opens your browser to `http://localhost:3000`.

If you don't have Python installed, run `.\START-WINDOWS.ps1 -SkipPython` - the dashboard will use the built-in heuristic scorer (still works, just not the trained TCN).

---

## Project structure

```
flash-crash-watchdog-web/
+-- src/
|   +-- app/
|   |   +-- page.tsx              # auth gate + socket.io client + toast routing
|   |   +-- layout.tsx            # Sonner + Radix toasters
|   |   +-- api/
|   |       +-- auth/{login,register,me,logout}/route.ts
|   |       +-- alerts/{read}/route.ts
|   +-- components/
|   |   +-- LoginScreen.tsx       # landing page + auth form
|   |   +-- Dashboard.tsx         # full dashboard with cascade funnel + feature bars
|   +-- lib/{db.ts,stores.ts,utils.ts}
+-- mini-services/
|   +-- binance-stream/
|       +-- index.ts              # port 3003 - Binance WS + Socket.io + TCN caller
|       +-- package.json          # uses tsx (no bun required)
+-- ml-inference/
|   +-- server.py                 # FastAPI sidecar that loads the real TCN
|   +-- requirements.txt
+-- models/
|   +-- stage3_tcn_trained.pt     # the trained model (764 KB, 93.3% val acc)
+-- prisma/schema.prisma          # User + Alert models
+-- .env                          # Windows-friendly env vars (relative paths)
+-- .npmrc                        # legacy-peer-deps=true (avoids next-auth conflict)
+-- SETUP-WINDOWS.ps1             # first-time install
+-- START-WINDOWS.ps1             # launch all 3 services
+-- STOP-WINDOWS.ps1              # kill all services
+-- Caddyfile                     # optional - for production deploy
+-- CONNECT_TO_PROJECT.md         # how to wire it to the existing ML repo
```

---

## How the Python sidecar connects

```
Binance WS
              v
   mini-services/binance-stream  (port 3003)
     |  1. Parse depth + trade
     |  2. Buffer 500 ticks
     |  3. POST /score
     |
     v
     |           ml-inference/server.py  (port 8000)
     |             * Loads stage3_tcn_trained.pt
     |             * Builds 17-feature window
     |             * Returns sigmoid score
     |  4. Emit 'tick' to browser
     |  5. Emit 'alert' if score > 0.6
     v
   Browser dashboard
     * Live price + score charts
     * Cascade funnel (5 stages)
     * Feature breakdown bars
     * Sonner toast on alert
     * Browser notification
     * POST /api/alerts to SQLite
```

The binance-stream service automatically falls back to the heuristic scorer if:
- Python sidecar is not running (`USE_TCN=false` env var or Python down)
- Python takes more than 2 seconds to respond
- 3 consecutive failures (auto-retries after 30s)

The dashboard shows a badge next to the anomaly score:
- **TCN** (green) - real PyTorch model is scoring
- **WARMUP** (amber) - Python is buffering the first 50 ticks
- **HEURISTIC** (gray) - fallback scorer in use

---

## What it does

1. **Landing page** - hero copy ("Catch the crash before the chart does"), feature chips, auth form
2. **Auth** - bcrypt password hashing, httpOnly cookie session, 7-day expiry
3. **Dashboard**:
   - 4 stat cards: price, anomaly score, alerts fired, status
   - Live price area chart + anomaly score chart with alert threshold line
   - **5-stage cascade funnel** (Statistical, Isolation Forest, TCN, Transformer, Bayesian)
   - **Live feature breakdown bars** (Velocity, OBI, Volatility, Spread)
   - **Score source badge** (TCN / WARMUP / HEURISTIC)
   - 50-item alert feed with mark-all-read
4. **Real-time alerts**:
   - Score > 0.6 with 10s cooldown then fire alert
   - Sonner toast (red for critical > 0.8, amber for warning)
   - Browser notification
   - Persisted to SQLite via `/api/alerts`

---

## Manual start (if you prefer not to use the .ps1 scripts)

```powershell
# Terminal 1 - Python TCN sidecar
cd ml-inference
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn server:app --port 8000 --reload

# Terminal 2 - Binance stream
cd mini-services\binance-stream
npm install --legacy-peer-deps
npm run dev

# Terminal 3 - Next.js dashboard
npm install --legacy-peer-deps
npm run db:push
npm run dev

# Open http://localhost:3000
```

---

## Connecting to your existing ML repo

This zip is **self-contained** - it includes the trained model and loads the ML package at runtime from `../flash-crash-watchdog/ml/` (one level up).

If you unzipped into `D:\flash-crash-watchdog\`, you should have:
```
D:\flash-crash-watchdog\
+-- ml\flash_crash_watchdog\        <-- your existing ML package (provides features)
+-- models\stage3_tcn_trained.pt    <-- your trained model (if any)
+-- scripts\
+-- flash-crash-watchdog-web\       <-- this zip, unzipped here
    +-- ml-inference\server.py      <-- looks for ML package at ..\flash-crash-watchdog\ml
    +-- models\stage3_tcn_trained.pt <-- backup copy (used if not found in parent)
```

The Python sidecar tries these locations in order:
1. `../flash-crash-watchdog/models/stage3_tcn_trained.pt`
2. `./models/stage3_tcn_trained.pt` (inside the web app folder)
3. `./ml-inference/models/stage3_tcn_trained.pt`

Same for the ML package: it looks at `../flash-crash-watchdog/ml/` first, then falls back. So if you have the existing ML repo at `D:\flash-crash-watchdog\`, everything auto-wires.

If the ML package isn't found, the sidecar runs in fallback mode (heuristic scorer) and logs a warning - the dashboard still works, just without the trained TCN.

See **CONNECT_TO_PROJECT.md** for full details.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ExecutionPolicy` errors when running `.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `powershell -ExecutionPolicy Bypass -File .\script.ps1` |
| `npm error ERESOLVE` peer dep conflict | The `.npmrc` file with `legacy-peer-deps=true` handles this. SETUP also passes `--legacy-peer-deps`. |
| `'tee' is not recognized` | Fixed in v2.1 - the `dev` script is now just `next dev -p 3000` (no tee) |
| `'bun' is not recognized` | Fixed in v2.1 - binance-stream now uses `tsx watch` instead of `bun --hot`. tsx is auto-installed by `npm install` in the binance-stream folder. |
| Dashboard shows "Offline" badge | Binance stream service (port 3003) isn't running. Check Window 2. |
| Score badge shows "HEURISTIC" not "TCN" | Python sidecar (port 8000) isn't running, or model file not found. Check Window 1 for errors. Visit `http://localhost:8000/health` to verify. |
| Score badge shows "WARMUP" forever | First 50 ticks needed. Should switch to TCN within ~5 seconds of Binance data flowing. |
| `EADDRINUSE :3000` | Another Next.js is running. Run `.\STOP-WINDOWS.ps1` or close stale Node windows. |
| `@prisma/client did not initialize yet` | Run `npm run db:generate` then restart the Next.js dev server. SETUP-WINDOWS.ps1 v2.1+ does this automatically. |
| `prisma/client not found` | Run `npm run db:generate`. |
| `Cannot apply unknown utility class 'border-border'` | Tailwind v4 CSS config issue. Fixed in v2.1 — `globals.css` now uses `@theme` directive instead of `tailwind.config.ts` for colors. If you still see this, delete `.next/` folder and restart. |
| `[tcn] fetch failed` in binance-stream | Python sidecar isn't running on port 8000. Check Window 1 for errors. Visit `http://127.0.0.1:8000/health` to verify. Common cause: Python venv creation failed during SETUP. |
| Python `ModuleNotFoundError: No module named 'flash_crash_watchdog'` | The sidecar expects `../flash-crash-watchdog/ml/` to exist. Either put the web app inside your ML repo, OR run with `USE_TCN=false` to use heuristic mode. |
| Binance WebSocket disconnects | Auto-reconnects after 2s. If it keeps failing, check your internet / firewall. |

---

## Production deployment

For production:
1. Set `NODE_ENV=production` in `.env`
2. Run `npm run build && npm run start` instead of `npm run dev`
3. Put all three services behind a reverse proxy (Caddy / Nginx) with TLS
4. Swap SQLite for Postgres by changing `provider = "postgresql"` in `prisma/schema.prisma`
5. For the Python sidecar on GPU, use `gunicorn -k uvicorn.workers.UvicornWorker -w 4`

The included `Caddyfile` shows the gateway config (only needed if you want a single public port).

---

Built for the Flash Crash Early Warning project.
Model: `huggingface.co/Dev2506/flash-crash-watchdog` - 93.3% val accuracy - 287k real windows - Focal Loss.
