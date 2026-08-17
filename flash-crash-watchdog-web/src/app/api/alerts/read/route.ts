import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { verifySession } from '@/lib/session'

export const runtime = 'nodejs'

export async function POST(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  await db.alert.updateMany({
    where: { userId, isRead: false },
    data: { isRead: true },
  })
  return NextResponse.json({ ok: true })
}
