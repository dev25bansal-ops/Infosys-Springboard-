import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import bcrypt from 'bcryptjs'
import { sessionCookieOptions, signSession } from '@/lib/session'
import { checkRateLimit, clientIp, isStrongPassword, MIN_PASSWORD_LENGTH } from '@/lib/rate-limit'

export const runtime = 'nodejs'

export async function POST(req: NextRequest) {
  if (!checkRateLimit(clientIp(req))) {
    return NextResponse.json({ error: 'Too many attempts. Try again later.' }, { status: 429 })
  }
  try {
    const { email, password, name } = await req.json()
    if (!email || !password) {
      return NextResponse.json({ error: 'Email and password are required' }, { status: 400 })
    }
    if (!isStrongPassword(password)) {
      return NextResponse.json(
        { error: `Password must be at least ${MIN_PASSWORD_LENGTH} characters` },
        { status: 400 },
      )
    }
    // BUG-09: emails are case-insensitive — normalize so "A@B.com" == "a@b.com".
    const normalizedEmail = String(email).trim().toLowerCase()
    const existing = await db.user.findUnique({ where: { email: normalizedEmail } })
    if (existing) {
      return NextResponse.json({ error: 'Email already registered' }, { status: 409 })
    }
    const passwordHash = await bcrypt.hash(password, 10)
    let user
    try {
      user = await db.user.create({
        data: { email: normalizedEmail, name: name || null, passwordHash },
      })
    } catch (e: unknown) {
      // BUG-09: TOCTOU — a concurrent register can win the unique race. Map the
      // Prisma unique-constraint error (P2002) to a clean 409, not a 500.
      if (typeof e === 'object' && e !== null && (e as { code?: string }).code === 'P2002') {
        return NextResponse.json({ error: 'Email already registered' }, { status: 409 })
      }
      throw e
    }
    const res = NextResponse.json({ id: user.id, email: user.email, name: user.name })
    res.cookies.set('session', await signSession(user.id), sessionCookieOptions)
    return res
  } catch (e) {
    return NextResponse.json({ error: 'Registration failed' }, { status: 500 })
  }
}
