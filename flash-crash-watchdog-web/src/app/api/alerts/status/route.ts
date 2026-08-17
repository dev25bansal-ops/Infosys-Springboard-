import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { verifySession } from '@/lib/session'
import { isSameOrigin } from '@/lib/csrf'

export const runtime = 'nodejs'

// ADV-10: bulk incident lifecycle — ack/dismiss/escalate many alerts at once,
// scoped to the authenticated user's alerts.
const VALID_STATUS = new Set(['NEW', 'ACKED', 'DISMISSED', 'ESCALATED'])

export async function POST(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  if (!isSameOrigin(req)) { // SEC-16
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }
  const raw = await req.json().catch(() => null)
  const idsRaw = Array.isArray((raw as { ids?: unknown })?.ids)
    ? (raw as { ids: unknown[] }).ids
    : []
  const ids = idsRaw.filter((x): x is string => typeof x === 'string')
  const status = (raw as { status?: unknown })?.status
  if (!ids.length || typeof status !== 'string' || !VALID_STATUS.has(status)) {
    return NextResponse.json({ error: 'Provide a non-empty ids array and a valid status' }, { status: 400 })
  }
  const result = await db.alert.updateMany({
    where: { id: { in: ids }, userId },
    data: { status },
  })
  return NextResponse.json({ updated: result.count })
}