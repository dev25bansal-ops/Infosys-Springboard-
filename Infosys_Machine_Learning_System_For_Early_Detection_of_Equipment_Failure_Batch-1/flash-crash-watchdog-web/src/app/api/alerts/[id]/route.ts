import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { verifySession } from '@/lib/session'
import { isSameOrigin } from '@/lib/csrf'

export const runtime = 'nodejs'

// ADV-10: incident lifecycle — an owner can update an alert's status.
const VALID_STATUS = new Set(['NEW', 'ACKED', 'DISMISSED', 'ESCALATED'])

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params  // Next 15+: route params are async
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  if (!isSameOrigin(req)) { // SEC-16
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }
  const raw = await req.json().catch(() => null)
  const status = raw?.status
  if (typeof status !== 'string' || !VALID_STATUS.has(status)) {
    return NextResponse.json({ error: `status must be one of ${[...VALID_STATUS].join(', ')}` }, { status: 400 })
  }
  const updated = await db.alert.updateMany({
    where: { id, userId }, // ownership-bound
    data: { status },
  })
  if (updated.count === 0) {
    return NextResponse.json({ error: 'Alert not found' }, { status: 404 })
  }
  const alert = await db.alert.findUnique({ where: { id } })
  return NextResponse.json({ alert })
}