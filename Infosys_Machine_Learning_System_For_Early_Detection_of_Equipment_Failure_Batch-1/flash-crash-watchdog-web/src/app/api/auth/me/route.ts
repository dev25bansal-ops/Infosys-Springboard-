import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { verifySession } from '@/lib/session'

export const runtime = 'nodejs'

export async function GET(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ user: null }, { status: 401 })
  }
  const user = await db.user.findUnique({
    where: { id: userId },
    select: { id: true, email: true, name: true },
  })
  if (!user) {
    return NextResponse.json({ user: null }, { status: 401 })
  }
  return NextResponse.json({ user })
}
