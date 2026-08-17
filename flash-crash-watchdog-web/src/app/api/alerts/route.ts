import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { verifySession } from '@/lib/session'
import { checkRateLimit, clientIp } from '@/lib/rate-limit'
import { isSameOrigin } from '@/lib/csrf'

export const runtime = 'nodejs'

export async function GET(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ alerts: [], nextCursor: null })
  }
  // NEW-04: cursor pagination + filters (limit/cursor/symbol/severity/from/to).
  const sp = req.nextUrl.searchParams
  const limit = Math.min(200, Math.max(1, Number(sp.get('limit')) || 50))
  const cursor = sp.get('cursor') // createdAt ms of the last alert on the prior page
  const symbol = sp.get('symbol') || undefined
  const severity = sp.get('severity') || undefined
  const fromS = sp.get('from'), toS = sp.get('to')

  const where: Record<string, unknown> = { userId }
  if (symbol) where.symbol = symbol
  if (severity) where.severity = severity
  if (fromS || toS || cursor) {
    const ca: Record<string, Date> = {}
    if (fromS) ca.gte = new Date(Number(fromS))
    if (toS) ca.lte = new Date(Number(toS))
    if (cursor) ca.lt = new Date(Number(cursor))
    where.createdAt = ca
  }

  const rows = await db.alert.findMany({
    where,
    orderBy: { createdAt: 'desc' },
    take: limit + 1, // +1 to detect whether there is another page
  })
  const hasMore = rows.length > limit
  const alerts = hasMore ? rows.slice(0, limit) : rows
  const nextCursor = hasMore && alerts.length > 0
    ? String(alerts[alerts.length - 1].createdAt.getTime())
    : null
  return NextResponse.json({ alerts, nextCursor })
}

export async function POST(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  // SEC-16: cross-origin writes are rejected even with a valid cookie.
  if (!isSameOrigin(req)) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }
  // SEC-6: bound unauthorized/authenticated alert-write volume (100/min is high
  // enough for outbox catch-up, too low for DB-flood abuse).
  if (!checkRateLimit(clientIp(req), 100)) {
    return NextResponse.json({ error: 'Too many alert writes. Try again later.' }, { status: 429 })
  }
  // BUG-08: validate + coerce types instead of `body.x || default`, which
  // silently turned legitimate 0 values (obi, vpin, volatility, price) into
  // null/0 and accepted malformed payloads.
  const raw = await req.json().catch(() => null)
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return NextResponse.json({ error: 'Invalid alert payload' }, { status: 400 })
  }
  const body = raw as Record<string, unknown>
  const num = (v: unknown): number | null => {
    if (typeof v === 'number' && Number.isFinite(v)) return v
    if (typeof v === 'string' && v.trim() !== '') {
      const n = Number(v)
      if (Number.isFinite(n)) return n
    }
    return null
  }
  const str = (v: unknown, d: string): string =>
    typeof v === 'string' && v.trim() ? v : d

  const symbol = str(body.symbol, 'BTCUSDT')
  const type = typeof body.type === 'string' && body.type ? body.type : undefined
  // NEW-04: idempotent upsert — a self-computed dedupKey (symbol + 10s bucket) or
  // a client-supplied one makes duplicate delivery a no-op instead of a dup row.
  const providedDedup = typeof body.dedupKey === 'string' && body.dedupKey ? body.dedupKey : null
  const dedupKey = providedDedup ?? `${symbol}:${Math.floor(Date.now() / 10_000)}`

  const data = {
    userId,
    symbol,
    price: num(body.price) ?? 0,
    score: num(body.score) ?? 0,
    ttdMs: num(body.ttdMs),
    obi: num(body.obi),
    vpin: num(body.vpin),
    volatility: num(body.volatility),
    message: str(body.message, 'Flash crash anomaly detected'),
    severity: str(body.severity, 'warning'),
    type,
    dedupKey,
  }
  try {
    const alert = await db.alert.upsert({
      where: { userId_dedupKey: { userId, dedupKey } },
      update: {}, // idempotent: keep the first delivery
      create: data,
    })
    return NextResponse.json({ alert })
  } catch (e: unknown) {
    // Fall back to a plain create if the dedupKey collides unexpectedly.
    if (typeof e === 'object' && e !== null && (e as { code?: string }).code === 'P2002') {
      const alert = await db.alert.create({ data: { ...data, dedupKey: `${dedupKey}-${Date.now()}` } })
      return NextResponse.json({ alert })
    }
    throw e
  }
}
