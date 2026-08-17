import './env' // BUG-14: load web-root .env into process.env before any config reads
import { createServer } from 'http'
import { Server } from 'socket.io'
import { CorrelationBreakdown } from './correlation'
import { append as outboxAppend, pending as outboxPending, markDelivered as outboxMarkDelivered } from './outbox'

// MLOPS-04: staleness heartbeat — a dark dashboard must be distinguishable from
// calm. A dedicated health HTTP server reports last-tick age; if no ticks arrive
// for STALE_AFTER_MS a `status` event with stale:true is emitted too. (Health is
// on its own port because socket.io with path:'/' consumes every request on the
// main server.)
const STALE_AFTER_MS = 15_000
const HEALTH_PORT = Number(process.env.HEALTH_PORT || 3005)
let lastTickAt = 0

// SEC-7: restrict socket.io CORS to the known dashboard/gateway origins so an
// arbitrary external site can't open the alert feed. Override via ALLOWED_ORIGINS.
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS
  || 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:81,http://127.0.0.1:81'
).split(',').map((s) => s.trim()).filter(Boolean)

const httpServer = createServer()
const io = new Server(httpServer, {
  path: '/',
  cors: {
    origin: (origin: string | undefined, cb: (err: Error | null, allow: boolean) => void) => {
      if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true)
      return cb(new Error('Origin not allowed'), false)
    },
    methods: ['GET', 'POST'],
  },
  pingTimeout: 60000,
  pingInterval: 25000,
})

const PORT = 3003

const TCN_INFERENCE_URL = process.env.TCN_INFERENCE_URL || 'http://127.0.0.1:8001/score'
const TCN_API_KEY = process.env.TCN_API_KEY || ''
const USE_TCN = process.env.USE_TCN !== 'false'
const FALLBACK_AFTER_MS = 2000

let binanceWs: WebSocket | null = null
let priceHistory: { time: number; price: number }[] = []
let tickBuffer: any[] = []
let scoreHistory: { time: number; score: number }[] = []
let currentPrice = 0
let currentScore = 0
let tickCount = 0
let alertsFired = 0
let tcnRequests = 0  // NEW-02 /metrics
let wsReconnects = 0 // NEW-02 /metrics
let lastAlertTime = 0
let tcnAvailable = true
let tcnFailureCount = 0
let lastTcnCall = 0
let lastFlushTs = 0
const TCN_THROTTLE_MS = 500

// STR-09: per-symbol watchlist state (anchor + correlation-basket symbols), so the
// dashboard can render a multi-symbol watchlist without a full depth refactor.
const perSymbolStats: Record<string, { ticks: number; alerts: number; lastPrice: number; lastScore: number }> = {}
function bumpSymbol(sym: string, price: number, score: number) {
  const s = perSymbolStats[sym] ||= { ticks: 0, alerts: 0, lastPrice: 0, lastScore: 0 }
  s.ticks++
  s.lastPrice = price
  s.lastScore = score
}
function symbolAlerts(sym: string) {
  perSymbolStats[sym] ||= { ticks: 0, alerts: 0, lastPrice: 0, lastScore: 0 }
  perSymbolStats[sym].alerts++
}

// STR-09 (full): multi-symbol DEPTH watchlist. Symbols listed here get their own
// depth subscription + heuristic detection (velocity + trailing-vol gate). The
// anchor (BTCUSDT) keeps the TCN path; watchlist symbols use the heuristic
// detector (the sidecar holds one TCN window, so per-symbol TCN scoring is a
// follow-on). Default: empty -> BTC-only (unchanged behavior).
const WATCHLIST = (process.env.WATCHLIST || '').split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
const watchDepth: Record<string, {
  priceHistory: { time: number; price: number }[]; currentPrice: number; alertsFired: number; lastAlertTime: number
}> = {}
function watchState(sym: string) {
  return watchDepth[sym] ||= { priceHistory: [], currentPrice: 0, alertsFired: 0, lastAlertTime: 0 }
}

// AF-1: cross-symbol correlation breakdown. Anchor is BTCUSDT (already in the
// stream); basket symbols are added as trade streams. Emits a distinct
// `correlation-alert` when the anchor-vs-basket correlation stays below the
// floor for the sustain window (a decoupling event — see correlation.ts).
const CORRELATION_SYMBOLS = (process.env.CORRELATION_SYMBOLS || 'ETHUSDT').split(',').filter(Boolean)
const CORR_ANCHOR = 'BTCUSDT'
const corrDetector = new CorrelationBreakdown({
  anchor: CORR_ANCHOR,
  corrWindowBins: 300,   // 5 min of 1s bins
  warmupBins: 300,
  floorCorr: Number(process.env.CORR_FLOOR || 0.4),
  sustainBins: Number(process.env.CORR_SUSTAIN_S || 60),
})
let lastCorrAlertTs = 0
const CORR_COOLDOWN_MS = 60000

// Per-symbol alert thresholds (Stage-3 score), calibrated on the ONLINE
// (per-tick, normalized) path via the held-out backtests: BTC 0.80, LUNA 0.50.
// Env-overridable (ALERT_THRESHOLD_BTC / _LUNA / generic ALERT_THRESHOLD).
const SYMBOL_ALERT_THRESHOLD: Record<string, number> = {
  BTCUSDT: Number(process.env.ALERT_THRESHOLD_BTC || 0.5),
  LUNAUSDT: Number(process.env.ALERT_THRESHOLD_LUNA || 0.5),
  default: Number(process.env.ALERT_THRESHOLD || 0.5),
}

// Regime gate: an alert is only emitted when the trailing realized volatility of
// the last ~25 mid-prices exceeds this (in basis points). The model's calm-day
// false positives sit on ~0-vol flat windows; crash onsets are on 10s-of-bps
// windows — so this suppresses calm-day chatter with ~no crash-recall cost. The
// window is short (see trailingVolBps) so the gate confirms an onset in ~1-2s,
// not after a long pre-drop calm stretch.
const MIN_TRAILING_VOL_BPS = Number(process.env.MIN_TRAILING_VOL_BPS || 2)

function trailingVolBps(prices: number[]): number {
  // SHORT confirmation window (~25 ticks ≈ 1-2s at 10-20Hz) instead of the old
  // 200-tick (~10-20s) window. Both suppress calm-day chatter — a flat price gives
  // ~0 vol at ANY window — but the short window confirms a real onset in ~1-2s
  // instead of only after seconds of price movement. That is the whole fix for
  // "alerts fire after the stock go down": the gate no longer waits out a long
  // pre-drop calm stretch.
  const w = prices.slice(-25)
  if (w.length < 12) return 0
  const mean = w.reduce((a, b) => a + b, 0) / w.length
  if (!mean) return 0
  const v = Math.sqrt(w.reduce((a, b) => a + (b - mean) ** 2, 0) / w.length)
  return (v / mean) * 10000
}

// NEW-01: web-path alert delivery (Slack / PagerDuty / generic webhook), driven
// from the mini-stream at the source. Reads config from the web-root .env
// (loaded by env.ts). Fire-and-forget; failures are logged, never fatal.
const SLACK_WEBHOOK = process.env.SLACK_WEBHOOK || ''
const PAGERDUTY_KEY = process.env.PAGERDUTY_KEY || ''
const ALERT_WEBHOOK = process.env.ALERT_WEBHOOK || ''
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || ''
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID || ''

async function notifyChannels(alert: any): Promise<void> {
  const tasks: Promise<unknown>[] = []
  const score = typeof alert.score === 'number' ? alert.score : 0
  if (SLACK_WEBHOOK) {
    tasks.push(fetch(SLACK_WEBHOOK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: `FLASH CRASH ALERT — ${alert.symbol}\nscore=${score.toFixed(3)} severity=${alert.severity}\n${alert.message || ''}` }),
    }))
  }
  if (PAGERDUTY_KEY) {
    tasks.push(fetch('https://events.pagerduty.com/v2/enqueue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        routing_key: PAGERDUTY_KEY,
        event_action: 'trigger',
        payload: {
          summary: `Flash crash detected on ${alert.symbol} (score ${score.toFixed(3)})`,
          source: alert.symbol,
          severity: alert.severity === 'critical' ? 'critical' : 'warning',
          custom_details: alert,
        },
      }),
    }))
  }
  if (ALERT_WEBHOOK) {
    tasks.push(fetch(ALERT_WEBHOOK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(alert),
    }))
  }
  if (TELEGRAM_BOT_TOKEN && TELEGRAM_CHAT_ID) {
    const text = `⚠️ FLASH CRASH ALERT — ${alert.symbol}\nscore=${score.toFixed(3)} severity=${alert.severity}\n${alert.message || ''}`
    tasks.push(fetch(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chat_id: TELEGRAM_CHAT_ID, text }),
    }))
  }
  const results = await Promise.allSettled(tasks)
  for (const r of results) {
    if (r.status === 'rejected') console.warn('[notify] channel delivery failed:', r.reason)
  }
}

// AF-3: 3-state market regime. CRASH = the actual alert condition (score above
// the per-symbol threshold AND trailing-vol above the gate). TENSE = elevated
// score or vol meaningfully above the calm gate. CALM otherwise.
function regimeLabel(score: number, tvBps: number, threshold: number, gateBps: number): 'CALM' | 'TENSE' | 'CRASH' {
  if (score > threshold && tvBps >= gateBps) return 'CRASH'
  if (score > threshold * 0.6 || tvBps >= gateBps * 5) return 'TENSE'
  return 'CALM'
}

function computeHeuristicScore(price: number, depth: { bids: [string, string][]; asks: [string, string][] }, history: { time: number; price: number }[] = priceHistory) {
  if (history.length < 30) return { score: 0, velocity: 0, obi: 0, volatility: 0, spreadBps: 0, bidDepth: 0, askDepth: 0 }

  const recent = history.slice(-10)
  const baseline = history.slice(-50, -10)
  const recentAvg = recent.reduce((s, p) => s + p.price, 0) / recent.length
  const baselineAvg = baseline.length > 0 ? baseline.reduce((s, p) => s + p.price, 0) / baseline.length : recentAvg
  const velocity = baselineAvg > 0 ? Math.abs((recentAvg - baselineAvg) / baselineAvg) * 100 : 0

  let bidDepth = 0, askDepth = 0
  if (depth.bids.length > 0 && depth.asks.length > 0) {
    for (let i = 0; i < Math.min(10, depth.bids.length); i++) bidDepth += parseFloat(depth.bids[i][1])
    for (let i = 0; i < Math.min(10, depth.asks.length); i++) askDepth += parseFloat(depth.asks[i][1])
  }
  const totalDepth = bidDepth + askDepth
  const obi = totalDepth > 0 ? (bidDepth - askDepth) / totalDepth : 0

  const prices = recent.map(p => p.price)
  const mean = prices.reduce((s, p) => s + p, 0) / prices.length
  const variance = prices.reduce((s, p) => s + (p - mean) ** 2, 0) / prices.length
  const volatility = mean > 0 ? Math.sqrt(variance) / mean * 100 : 0

  let spread = 0
  if (depth.bids.length > 0 && depth.asks.length > 0) {
    const bestBid = parseFloat(depth.bids[0][0])
    const bestAsk = parseFloat(depth.asks[0][0])
    spread = bestAsk > 0 ? ((bestAsk - bestBid) / bestAsk) * 10000 : 0
  }

  const velocityScore = Math.min(1, velocity / 1.5)
  const obiScore = Math.min(1, Math.abs(obi) * 3)
  const volScore = Math.min(1, volatility * 80)
  const spreadScore = Math.min(1, spread / 100)

  const score = Math.max(velocityScore, obiScore * 0.6, volScore * 0.4, spreadScore * 0.2)
  return { score: Math.min(1, score), velocity, obi, volatility, spreadBps: spread, bidDepth, askDepth }
}

async function scoreWithTCN(
  price: number,
  depth: { bids: [string, string][]; asks: [string, string][] },
  trade: { price: number; size: number; side: string } | null,
  timestamp: number,
): Promise<{ score: number; source: string }> {
  const tick = {
    timestamp_ms: timestamp,
    price,
    bids: depth.bids.slice(0, 10),
    asks: depth.asks.slice(0, 10),
    symbol: 'BTCUSDT',
    trade: trade ? { ...trade, timestamp_ms: timestamp } : null,
  }
  tickBuffer.push(tick)
  if (tickBuffer.length > 500) tickBuffer.shift()

  if (!USE_TCN || !tcnAvailable) {
    const h = computeHeuristicScore(price, depth)
    return { score: h.score, source: 'heuristic' }
  }

  const now = Date.now()
  if (now - lastTcnCall < TCN_THROTTLE_MS) {
    // BUG-11: never stamp a heuristic/fallback score as 'tcn'. The throttle
    // path returns the last REAL TCN score if one exists, else an honestly
    // labeled heuristic score (previously the heuristic fallback was tagged
    // 'tcn' whenever any prior TCN call had happened).
    const h = computeHeuristicScore(price, depth)
    if (currentScore > 0 && lastTcnCall > 0) {
      return { score: currentScore, source: 'tcn' }
    }
    return { score: h.score, source: 'heuristic' }
  }
  lastTcnCall = now
  tcnRequests++ // NEW-02 /metrics

  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), FALLBACK_AFTER_MS)

    // Batch: send the ticks that arrived since the last TCN call so the server's
    // Stage-3 window is built from the real (near-contiguous) tick stream, not a
    // single sample per throttle interval. The sidecar feeds each tick cheaply.
    const batch = tickBuffer.filter((t) => t.timestamp_ms > lastFlushTs).slice(-40)
    const ticksToSend = batch.length ? batch : [tickBuffer[tickBuffer.length - 1]]
    if (batch.length) lastFlushTs = batch[batch.length - 1].timestamp_ms
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (TCN_API_KEY) headers['x-api-key'] = TCN_API_KEY
    const res = await fetch(TCN_INFERENCE_URL, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ticks: ticksToSend }),
      signal: controller.signal,
    })
    clearTimeout(timeout)

    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json() as { score: number; ready: boolean; source: string }

    if (data.source === 'tcn' && data.ready) {
      tcnFailureCount = 0
      return { score: data.score, source: 'tcn' }
    }
    return { score: 0, source: data.source || 'warmup' }
  } catch (e) {
    tcnFailureCount++
    const errMsg = (e as Error).message || String(e)
    if (tcnFailureCount === 1 || tcnFailureCount % 10 === 0) {
      console.warn(`[tcn] Failure #${tcnFailureCount}: ${errMsg}`)
      console.warn(`[tcn] URL: ${TCN_INFERENCE_URL}`)
      console.warn(`[tcn] Check: (1) is Python sidecar running on port 8001? (2) visit http://localhost:8001/health`)
    }
    if (tcnFailureCount >= 3) {
      tcnAvailable = false
      console.warn(`[tcn] Unavailable after ${tcnFailureCount} failures - falling back to heuristic for 30s`)
      setTimeout(() => { tcnAvailable = true; tcnFailureCount = 0; console.log('[tcn] Retrying Python sidecar...') }, 30000)
    }
    const h = computeHeuristicScore(price, depth)
    return { score: h.score, source: 'heuristic' }
  }
}

// GAP-04: the message-processing seam — exported so the live-stack test can
// drive the real detection path (scoring -> regime gate -> alert -> outbox)
// deterministically without the Binance WebSocket.
export async function processStreamMessage(msg: any): Promise<void> {
  try {
    if (!msg.stream || !msg.data) return
    const stream = msg.stream as string
    const data = msg.data

      if (stream.includes('depth')) {
        const symbol = stream.split('@')[0].toUpperCase()
        const bids = data.bids || data.b || []
        const asks = data.asks || data.a || []
        if (bids.length > 0 && asks.length > 0) {
          const bestBid = parseFloat(bids[0][0])
          const bestAsk = parseFloat(asks[0][0])
          const midPrice = (bestBid + bestAsk) / 2
          if (midPrice > 0) {
            // STR-09 (full): a NON-anchor watchlist symbol gets its own depth
            // history + heuristic (velocity/trailing-vol) detection. The anchor
            // keeps the TCN path below.
            if (symbol !== CORR_ANCHOR && WATCHLIST.includes(symbol)) {
              const st = watchState(symbol)
              st.currentPrice = midPrice
              st.priceHistory.push({ time: Date.now(), price: midPrice })
              if (st.priceHistory.length > 500) st.priceHistory.shift()
              const f = computeHeuristicScore(midPrice, { bids, asks }, st.priceHistory)
              const tvBps = trailingVolBps(st.priceHistory.map(p => p.price))
              const threshold = SYMBOL_ALERT_THRESHOLD[symbol] ?? SYMBOL_ALERT_THRESHOLD.default
              bumpSymbol(symbol, midPrice, f.score)
              const now = Date.now()
              if (f.score > threshold && tvBps >= MIN_TRAILING_VOL_BPS && (now - st.lastAlertTime) > 10_000) {
                st.lastAlertTime = now
                st.alertsFired++
                const alert = {
                  id: `alert_${now}_${symbol}`, symbol, price: midPrice, score: f.score,
                  type: 'crash', dedupKey: `${symbol}:${Math.floor(now / 10_000)}`,
                  trailingVolBps: Math.round(tvBps * 10) / 10,
                  threshold, gate: MIN_TRAILING_VOL_BPS,
                  regime: regimeLabel(f.score, tvBps, threshold, MIN_TRAILING_VOL_BPS),
                  ttdMs: null, obi: f.obi, vpin: null, volatility: f.volatility,
                  message: `[heuristic] Watch anomaly on ${symbol}: score=${f.score.toFixed(3)}, trailingVol=${tvBps.toFixed(1)}bps, velocity=${f.velocity.toFixed(2)}%`,
                  severity: f.score > 0.8 ? 'critical' : 'warning',
                  createdAt: new Date().toISOString(),
                }
                console.log(`[WATCH-ALERT] ${alert.message}`)
                symbolAlerts(symbol)
                outboxAppend(alert)
                io.emit('alert', alert)
                notifyChannels(alert).catch(() => {})
              }
              return
            }

            currentPrice = midPrice
            lastTickAt = Date.now()  // MLOPS-04 staleness heartbeat
            priceHistory.push({ time: Date.now(), price: midPrice })
            if (priceHistory.length > 500) priceHistory.shift()

            let trade = null
            if (stream.includes('trade')) {
              trade = {
                price: parseFloat(data.p),
                size: parseFloat(data.q),
                side: data.m ? 'sell' : 'buy',
                // Use Binance's trade timestamp (data.T) if available, else Date.now()
                timestamp_ms: data.T || Date.now(),
              }
            }

            const { score, source } = await scoreWithTCN(midPrice, { bids, asks }, trade, Date.now())
            const features = computeHeuristicScore(midPrice, { bids, asks })
            currentScore = score
            scoreHistory.push({ time: Date.now(), score })
            if (scoreHistory.length > 500) scoreHistory.shift()
            tickCount++

            const alertThreshold = SYMBOL_ALERT_THRESHOLD[symbol] ?? SYMBOL_ALERT_THRESHOLD.default
            bumpSymbol(symbol, midPrice, score) // STR-09 watchlist

            if (tickCount % 5 === 0) {
              const tvBpsTick = trailingVolBps(priceHistory.map(p => p.price))
              io.emit('tick', {
                price: midPrice,
                score,
                source,
                trailingVolBps: Math.round(tvBpsTick * 10) / 10,
                regime: regimeLabel(score, tvBpsTick, alertThreshold, MIN_TRAILING_VOL_BPS),
                velocity: features.velocity,
                obi: features.obi,
                volatility: features.volatility,
                spreadBps: features.spreadBps,
                bidDepth: features.bidDepth,
                askDepth: features.askDepth,
                timestamp: Date.now(),
                tickCount,
              })
            }

            const now = Date.now()
            const tvBps = trailingVolBps(priceHistory.map(p => p.price))
            // ONE alert, SCORE-LED, gated on a SHORT-window trailing-vol floor.
            // The score is the trigger — it leads the price turn — and the gate is
            // the spam suppressor: the model's calm-day false positives sit on
            // ~0-vol flat windows, so a vol floor kills them (the 0-calm-day-alert
            // validation figure was measured with this gate). The window is 25 ticks
            // (~1-2s) rather than 200 so the gate confirms a real onset almost
            // immediately instead of after seconds of price movement.
            if (score > alertThreshold && tvBps >= MIN_TRAILING_VOL_BPS && (now - lastAlertTime) > 10_000) {
              lastAlertTime = now
              alertsFired++
              const alert = {
                id: `alert_${now}`,
                symbol,
                price: midPrice,
                score,
                // NEW-04: stable content-addressable key enables idempotent upsert.
                type: 'crash',
                dedupKey: `${symbol}:${Math.floor(now / 10_000)}`,
                trailingVolBps: Math.round(tvBps * 10) / 10,
                threshold: alertThreshold,
                gate: MIN_TRAILING_VOL_BPS,
                regime: regimeLabel(score, tvBps, alertThreshold, MIN_TRAILING_VOL_BPS),
                ttdMs: null,
                obi: features.obi,
                // BUG-08: this used to copy volatility into vpin — a mislabel.
                // No real VPIN is computed here, so leave it null.
                vpin: null,
                volatility: features.volatility,
                message: `[${source}] CRASH WARNING: score=${score.toFixed(3)} (vol ${tvBps.toFixed(1)}bps vs ${MIN_TRAILING_VOL_BPS}bps floor)`,
                severity: score > 0.8 ? 'critical' : 'warning',
                createdAt: new Date().toISOString(),
              }
              console.log(`[ALERT #${alertsFired}] ${alert.message}`)
              // MLOPS-01: persist at the source (durable outbox) so a crash alert
              // survives a closed browser / process restart. The dashboard catches
              // up on reconnect via `outbox-sync` and acks with `alert-acked`.
              outboxAppend(alert)
              symbolAlerts(symbol) // STR-09 watchlist
              io.emit('alert', alert)
              // NEW-01: real multi-channel delivery (Slack/PD/webhook) at the source.
              notifyChannels(alert).catch(() => {})
            }
          }
        }
      }

      // AF-1: feed basket trade streams into the correlation-breakdown detector
      const symUpper = (stream.split('@')[0] || '').toUpperCase()
      if (stream.includes('@trade') && symUpper !== CORR_ANCHOR && CORRELATION_SYMBOLS.includes(symUpper)) {
        const tradePrice = parseFloat(data.p)
        const ts = data.T || Date.now()
        if (tradePrice > 0 && currentPrice > 0) {
          corrDetector.update(CORR_ANCHOR, currentPrice, ts)
          corrDetector.update(symUpper, tradePrice, ts)
          bumpSymbol(symUpper, tradePrice, 0) // STR-09 watchlist
          const r = corrDetector.evaluate()
          const now = Date.now()
          if (r.fire && r.corr != null && (now - lastCorrAlertTs) > CORR_COOLDOWN_MS) {
            lastCorrAlertTs = now
            const alert = {
              id: `corr_${now}`,
              type: 'correlation',
              dedupKey: `corr:${Math.floor(now / 10_000)}`,
              symbol: `${CORR_ANCHOR}↔${CORRELATION_SYMBOLS.join(',')}`,
              price: currentPrice,
              score: r.corr,
              message: `Correlation breakdown: ${CORR_ANCHOR} vs basket corr=${r.corr.toFixed(2)} (below floor ${corrDetector.floor})`,
              severity: 'warning',
              createdAt: new Date().toISOString(),
            }
            console.log(`[CORR-ALERT] ${alert.message}`)
            symbolAlerts(symUpper) // STR-09 watchlist
            io.emit('correlation-alert', alert)
          }
        }
      }
    } catch (e) {}
  }
function connectBinance() {
  const streams = [`${CORR_ANCHOR.toLowerCase()}@depth20@100ms`, `${CORR_ANCHOR.toLowerCase()}@trade`]
  for (const s of CORRELATION_SYMBOLS) streams.push(`${s.toLowerCase()}@trade`)
  // STR-09: full per-symbol depth subscriptions for the watchlist (non-anchor).
  for (const s of WATCHLIST) if (s !== CORR_ANCHOR) streams.push(`${s.toLowerCase()}@depth20@100ms`)
  const url = `wss://stream.binance.com:9443/stream?streams=${streams.join('/')}`
  console.log(`[binance] Connecting to ${url}`)
  binanceWs = new WebSocket(url)

  binanceWs.onopen = () => {
    console.log('[binance] Connected')
    io.emit('status', { connected: true, message: 'Connected to Binance' })
  }

  binanceWs.onmessage = (event) => {
    processStreamMessage(JSON.parse(event.data as string))
  }


  binanceWs.onclose = () => {
    wsReconnects++ // NEW-02 /metrics
    console.log('[binance] Disconnected, reconnecting in 2s...')
    io.emit('status', { connected: false, message: 'Reconnecting...' })
    setTimeout(connectBinance, 2000)
  }
  binanceWs.onerror = (e) => console.error('[binance] Error:', e)
}

io.on('connection', (socket) => {
  console.log(`[io] Client connected: ${socket.id}`)
  socket.emit('status', { connected: binanceWs?.readyState === WebSocket.OPEN, message: 'Connected' })
  // STR-09: push the current per-symbol watchlist to a newly-connected dashboard.
  socket.emit('watchlist', perSymbolStats)
  if (priceHistory.length > 0) {
    socket.emit('history', { priceHistory: priceHistory.slice(-120), scoreHistory: scoreHistory.slice(-120), currentPrice, currentScore, tickCount, alertsFired })
  }
  // MLOPS-01: deliver any alerts that fired while no dashboard was connected.
  const undelivered = outboxPending()
  if (undelivered.length > 0) {
    console.log(`[outbox] replaying ${undelivered.length} undelivered alert(s) to ${socket.id}`)
    socket.emit('outbox-sync', undelivered)
  }
  socket.on('alert-acked', (ids: unknown) => {
    if (Array.isArray(ids) && ids.length > 0) {
      const clean = ids.filter((x): x is string => typeof x === 'string')
      if (clean.length) outboxMarkDelivered(clean)
    }
  })
  socket.on('disconnect', () => console.log(`[io] Client disconnected: ${socket.id}`))
})

// GAP-04: DISABLE_BINANCE_WS=true runs the service with NO live feed — tests
// drive processStreamMessage() directly (the real detection path) instead.
if (process.env.DISABLE_BINANCE_WS !== 'true') {
  connectBinance()
}
httpServer.listen(PORT, () => {
  console.log(`[server] Flash Crash Stream Service running on port ${PORT}`)
  console.log(`[tcn] Python sidecar URL: ${TCN_INFERENCE_URL}`)
  console.log(`[tcn] USE_TCN=${USE_TCN} (set USE_TCN=false to force heuristic-only)`)
})

// MLOPS-04: dedicated health/staleness endpoint (socket.io owns the main port).
// NEW-02: also serves Prometheus /metrics (scraped by configs/prometheus.yml).
createServer((req, res) => {
  if (req.url === '/metrics') {
    const m = [
      '# HELP fcw_ticks_received_total Binance depth ticks processed',
      '# TYPE fcw_ticks_received_total counter',
      `fcw_ticks_received_total ${tickCount}`,
      '# HELP fcw_alerts_fired_total Alerts fired',
      '# TYPE fcw_alerts_fired_total counter',
      `fcw_alerts_fired_total ${alertsFired}`,
      '# HELP fcw_tcn_requests_total TCN inference requests',
      '# TYPE fcw_tcn_requests_total counter',
      `fcw_tcn_requests_total ${tcnRequests}`,
      '# HELP fcw_ws_reconnects_total Binance WS reconnects',
      '# TYPE fcw_ws_reconnects_total counter',
      `fcw_ws_reconnects_total ${wsReconnects}`,
    ].join('\n')
    res.writeHead(200, { 'content-type': 'text/plain; version=0.0.4' })
    res.end(m + '\n')
    return
  }
  const stalenessMs = lastTickAt ? Date.now() - lastTickAt : 0
  res.writeHead(200, { 'content-type': 'application/json' })
  res.end(JSON.stringify({
    ok: true,
    connected: binanceWs?.readyState === WebSocket.OPEN,
    lastTickAt,
    stalenessMs,
    stale: stalenessMs > STALE_AFTER_MS,
    alertsFired,
  }))
}).listen(HEALTH_PORT, () => {
  console.log(`[server] Health endpoint on :${HEALTH_PORT}`)
})

// MLOPS-04: staleness monitor — emit a status event when ticks stop arriving, so
// a supervision process / dashboard can alarm on a silently dead feed.
setInterval(() => {
  if (!lastTickAt) return
  const stalenessMs = Date.now() - lastTickAt
  if (stalenessMs > STALE_AFTER_MS) {
    console.warn(`[heartbeat] STALE: no ticks for ${Math.round(stalenessMs / 1000)}s`)
    io.emit('status', {
      connected: binanceWs?.readyState === WebSocket.OPEN,
      stale: true,
      lastTickAt,
      stalenessMs,
      message: `Feed stale — no data for ${Math.round(stalenessMs / 1000)}s`,
    })
  }
}, 5_000)
