import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import bcrypt from 'bcryptjs'
import { sessionCookieOptions, signSession } from '@/lib/session'
import { checkRateLimit, clientIp } from '@/lib/rate-limit'

export const runtime = 'nodejs'

export async function POST(req: NextRequest) {
  if (!checkRateLimit(clientIp(req))) {
    return NextResponse.json({ error: 'Too many attempts. Try again later.' }, { status: 429 })
  }
  try {
    const { email, password } = await req.json()
    if (!email || !password) {
      return NextResponse.json({ error: 'Email and password are required' }, { status: 400 })
    }
    // BUG-09: emails are case-insensitive — normalize to match registration.
    const normalizedEmail = String(email).trim().toLowerCase()
    const user = await db.user.findUnique({ where: { email: normalizedEmail } })
    if (!user) {
      return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 })
    }
    const valid = await bcrypt.compare(password, user.passwordHash)
    if (!valid) {
      return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 })
    }
    const res = NextResponse.json({ id: user.id, email: user.email, name: user.name })
    res.cookies.set('session', await signSession(user.id), sessionCookieOptions)
    return res
  } catch (e) {
    return NextResponse.json({ error: 'Login failed' }, { status: 500 })
  }
}
