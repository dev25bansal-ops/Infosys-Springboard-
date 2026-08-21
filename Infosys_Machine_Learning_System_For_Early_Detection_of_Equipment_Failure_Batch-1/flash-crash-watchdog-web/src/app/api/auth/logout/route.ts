import { NextRequest, NextResponse } from 'next/server'
import { revokeSession } from '@/lib/session'

export async function POST(req: NextRequest) {
  // SEC-2: revoke this session server-side (the signed cookie is one-shot).
  await revokeSession(req.cookies.get('session')?.value)
  const res = NextResponse.json({ ok: true })
  res.cookies.delete('session')
  return res
}
