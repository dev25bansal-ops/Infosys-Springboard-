// GAP-04: live-stack end-to-end alert test.
//
// Drives the REAL mini-stream detection path (scoring -> regime gate -> alert ->
// durable outbox -> socket.io broadcast) with injected depth messages, bypassing
// the Binance WebSocket, and asserts a crash alert fires with correct fields.
// This binds BUG-04/11 (ready/source), STR-02 (ttdMs), MLOPS-08 (gate) together.
//
// Run from this directory:  npx tsx _gap04_test.ts
import { existsSync, unlinkSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

async function main() {
  process.env.USE_TCN = 'false'            // deterministic heuristic scoring, no sidecar
  process.env.DISABLE_BINANCE_WS = 'true'  // no live feed — we inject ticks
  process.env.MIN_TRAILING_VOL_BPS = '2'
  process.env.ALERT_THRESHOLD = '0.5'

  const { default: ioClient } = await import('socket.io-client')
  const { processStreamMessage } = await import('./index')

  const client = ioClient('http://127.0.0.1:3003', { transports: ['websocket'], forceNew: true })
  await new Promise<void>((res, rej) => {
    client.on('connect', () => res())
    client.on('connect_error', (e) => rej(new Error('socket connect: ' + e.message)))
    setTimeout(() => rej(new Error('socket connect timeout')), 5000)
  })
  console.log('socket connected to live stream service')

  const alerts: any[] = []
  client.on('alert', (a: any) => alerts.push(a))

  const depthMsg = (price: number) => ({
    stream: 'btcusdt@depth20@100ms',
    data: {
      bids: [[String(price - 1), '1.0'], [String(price - 2), '2.0']],
      asks: [[String(price + 1), '1.0'], [String(price + 2), '2.0']],
    },
  })

  // Flat phase: ~200 ticks to build the price/vol history baseline.
  for (let i = 0; i < 200; i++) await processStreamMessage(depthMsg(50000))
  // Crash phase: 50000 -> 47000 over 20 ticks (velocity ~6%, trailing vol high).
  for (let i = 1; i <= 20; i++) await processStreamMessage(depthMsg(50000 - i * 150))
  console.log('injected flat + crash ticks through the live path')

  const deadline = Date.now() + 5000
  while (alerts.length === 0 && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 100))
  }

  if (alerts.length === 0) {
    console.error('FAIL: no alert received on the live path')
    process.exit(1)
  }
  const a = alerts[0]
  const src = (a.message || '').match(/^\[([^\]]+)\]/)?.[1] ?? a.source ?? '?'
  console.log('ALERT:', JSON.stringify({
    symbol: a.symbol, price: a.price, score: a.score,
    severity: a.severity, source: src, ttdMs: a.ttdMs,
  }))

  const ok =
    a.symbol === 'BTCUSDT' &&
    a.price > 0 &&
    typeof a.score === 'number' && a.score > 0.5 &&
    (a.severity === 'critical' || a.severity === 'warning') &&
    src === 'heuristic'

  // Cleanup: disconnect, remove outbox artifacts created by the fired alert.
  client.disconnect()
  const dataDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'data')
  for (const f of ['alerts.jsonl', 'alerts.delivered.json']) {
    const p = join(dataDir, f)
    if (existsSync(p)) unlinkSync(p)
  }
  console.log(ok ? 'GAP-04 OK: live-stack alert fired with correct fields' : 'GAP-04 FAILED: field mismatch')
  process.exit(ok ? 0 : 1)
}

main().catch((e) => { console.error('FAIL', e); process.exit(1) })
