// MLOPS-01: source-side alert outbox.
//
// Alerts fired by the mini-stream are appended durably (JSONL + fsync) at the
// SOURCE, so a crash alert survives a closed browser and a process restart.
// A reconnecting dashboard receives any undelivered alerts via `outbox-sync`
// and acks them via `alert-acked`; only then are they marked delivered.
//
// Files (under flash-crash-watchdog-web/data/):
//   alerts.jsonl          — append-only alert log (durable)
//   alerts.delivered.json — set of alert ids the dashboard has persisted
import {
  existsSync, mkdirSync, writeFileSync, readFileSync, openSync,
  writeSync, fsyncSync,
} from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

interface OutboxAlert {
  id: string
  symbol: string
  price: number
  score: number
  message: string
  severity: string
  createdAt: string
  trailingVolBps?: number
  threshold?: number
  gate?: number
  type?: string
}

const DATA_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'data')
const ALERTS_FILE = join(DATA_DIR, 'alerts.jsonl')
const DELIVERED_FILE = join(DATA_DIR, 'alerts.delivered.json')

let fd: number | null = null

function ensureFiles(): void {
  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true })
  if (!existsSync(ALERTS_FILE)) writeFileSync(ALERTS_FILE, '', 'utf8')
  if (!existsSync(DELIVERED_FILE)) writeFileSync(DELIVERED_FILE, '[]', 'utf8')
  if (fd === null) fd = openSync(ALERTS_FILE, 'a')
}

function deliveredIds(): Set<string> {
  try {
    return new Set(JSON.parse(readFileSync(DELIVERED_FILE, 'utf8')) as string[])
  } catch {
    return new Set()
  }
}

/** Durably record an alert at the source (fsync'd append). */
export function append(alert: OutboxAlert): void {
  ensureFiles()
  try {
    writeSync(fd as number, JSON.stringify(alert) + '\n', null, 'utf8')
    fsyncSync(fd as number)
  } catch (e) {
    console.error('[outbox] append failed:', e)
  }
}

/** Alerts that have NOT yet been delivered to (and persisted by) a dashboard. */
export function pending(): OutboxAlert[] {
  ensureFiles()
  const done = deliveredIds()
  const out: OutboxAlert[] = []
  try {
    const lines = readFileSync(ALERTS_FILE, 'utf8').split('\n')
    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const a = JSON.parse(line) as OutboxAlert
        if (a.id && !done.has(a.id)) out.push(a)
      } catch {
        /* skip malformed line */
      }
    }
  } catch (e) {
    console.error('[outbox] read failed:', e)
  }
  return out
}

/** Mark alert ids as delivered/persisted by a dashboard. */
export function markDelivered(ids: string[]): void {
  ensureFiles()
  const done = deliveredIds()
  for (const id of ids) done.add(id)
  try {
    writeFileSync(DELIVERED_FILE, JSON.stringify([...done]), 'utf8')
  } catch (e) {
    console.error('[outbox] markDelivered failed:', e)
  }
}
