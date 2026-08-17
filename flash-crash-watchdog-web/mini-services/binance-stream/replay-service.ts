/**
 * AF-2: crash-replay streaming service.
 *
 * Loads an exported replay day (data/replay/<label>.json — produced by
 * scripts/export_replay_day.py) and streams its per-tick staging over socket.io
 * with play/pause, a × speed control, and seek. Emits:
 *   replay:load   {label, ticks, threshold, gateBps, alerts}   (on connect/load)
 *   replay:data   {points: [...], alerts: [...]}               (streaming chunks)
 *   replay:state  {playing, speed, index}                      (ack/status)
 *
 * Commands from client: replay:load, replay:play, replay:pause, replay:speed,
 * replay:seek.
 *
 * Run:  cd mini-services/binance-stream && npx tsx replay-service.ts
 */
import './env' // BUG-14: load web-root .env into process.env (REPLAY_PORT etc.)
import { createServer } from 'http'
import { Server } from 'socket.io'
import { readFileSync, existsSync } from 'fs'
import { join, resolve, sep } from 'path'

const PORT = Number(process.env.REPLAY_PORT || 3004)
const DATA_DIR = join(__dirname, '..', '..', '..', 'data', 'replay')

// SEC-4: replay labels are strictly alphanumeric + _ - (e.g. "btc-0519"). Reject
// anything else so a malicious label can never traverse out of data/replay/.
const SAFE_LABEL = /^[A-Za-z0-9_-]+$/

// base stream rate: this many points per second at ×1, scaled by `speed`.
const BASE_RATE = 60

interface ReplayDay {
  label: string
  symbol: string
  threshold: number
  gate_bps: number
  start_ms: number
  ticks: number
  columns: string[]
  series: { t: number; p: number; s3: number; tv: number; s1: number; s2: number }[]
  alerts: { t: number; price: number; s3: number; tv: number }[]
}

let day: ReplayDay | null = null
let index = 0
let playing = false
let speed = 10
let timer: NodeJS.Timeout | null = null

const httpServer = createServer()
// SEC-7: same CORS allowlist as the stream service (dashboard/gateway origins).
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS
  || 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:81,http://127.0.0.1:81'
).split(',').map((s) => s.trim()).filter(Boolean)
const io = new Server(httpServer, {
  cors: {
    origin: (origin: string | undefined, cb: (err: Error | null, allow: boolean) => void) => {
      if (!origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true)
      return cb(new Error('Origin not allowed'), false)
    },
    methods: ['GET', 'POST'],
  },
  pingTimeout: 60000,
})

export function loadDay(label: string): ReplayDay | null {
  // SEC-4: label allowlist + path containment — a traversal like
  // "../../configs/operating" is rejected before touching the filesystem.
  if (typeof label !== 'string' || !SAFE_LABEL.test(label)) return null
  const f = resolve(DATA_DIR, `${label}.json`)
  if (!f.startsWith(resolve(DATA_DIR) + sep)) return null
  if (!existsSync(f)) return null
  try {
    return JSON.parse(readFileSync(f, 'utf-8'))
  } catch {
    return null
  }
}

function emitDay(socket: any) {
  socket.emit('replay:load', day ? {
    label: day.label, symbol: day.symbol, ticks: day.ticks,
    threshold: day.threshold, gateBps: day.gate_bps, startMs: day.start_ms,
  } : { label: null })
}

function tick() {
  if (!day || !playing) return
  // emit up to `speed` points this tick (BASE_RATE/s * speed frames/s ≈ speed pts/s)
  const count = Math.max(1, Math.round(speed))
  const pts: any[] = []
  const alerts: any[] = []
  for (let k = 0; k < count && index < day.series.length; k++) {
    const rec = day.series[index]
    pts.push(rec)
    if (day.alerts.some((a) => a.t === rec.t)) alerts.push({ t: rec.t, price: rec.p, s3: rec.s3 })
    index++
  }
  if (pts.length) io.emit('replay:data', { points: pts, alerts })
  if (index >= day.series.length) {
    playing = false
    io.emit('replay:state', { playing, speed, index, done: true })
    clearInterval(timer!); timer = null
    return
  }
  io.emit('replay:state', { playing, speed, index })
}

function schedule() {
  if (timer) clearInterval(timer)
  timer = setInterval(tick, Math.max(16, Math.round(1000 / BASE_RATE)))
}

io.on('connection', (socket) => {
  emitDay(socket)
  socket.emit('replay:state', { playing, speed, index })

  socket.on('replay:load', (label: string) => {
    day = loadDay(String(label))
    index = 0
    emitDay(socket)
  })
  socket.on('replay:play', () => { if (!day) return; playing = true; schedule() })
  socket.on('replay:pause', () => { playing = false; if (timer) { clearInterval(timer); timer = null } })
  socket.on('replay:speed', (x: number) => { speed = Math.max(1, Math.min(1000, Number(x) || 1)) })
  socket.on('replay:seek', (frac: number) => {
    if (!day) return
    index = Math.max(0, Math.min(day.series.length - 1, Math.round(Number(frac) * day.series.length)))
  })
  socket.on('disconnect', () => { /* keep the server running for other clients */ })
})

httpServer.listen(PORT, () => {
  console.log(`[replay] replay service on :${PORT} (data dir ${DATA_DIR})`)
})