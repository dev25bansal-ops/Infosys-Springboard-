import crypto from 'crypto'
import { db } from './db'

/**
 * Signed-session helper (SEC-1 / SEC-2).
 *
 * SEC-1: no public secret fallback — production requires a strong SESSION_SECRET
 * (>=32 chars, non-placeholder) and fails closed otherwise; dev uses a
 * per-process random secret.
 *
 * SEC-2: the cookie is "<userId>.<expMs>.<nonce>.<HMAC>", so expiry is enforced
 * server-side (not just a browser maxAge hint) and a nonce makes every token
 * unique. A `Session` row (token hash) is persisted at sign time and checked on
 * every verify, giving real server-side revocation on logout.
 */

export const SESSION_TTL_MS = 60 * 60 * 24 * 7 * 1000 // 7 days, matches maxAge

const KNOWN_PLACEHOLDERS = new Set([
  'dev-insecure-change-me',
  'change-me-to-a-long-random-string',
])

let _devSecret: string | null = null
function devSecret(): string {
  if (_devSecret === null) {
    _devSecret = crypto.randomBytes(32).toString('hex')
    console.warn(
      '[session] SESSION_SECRET not set — using a per-process random secret ' +
      '(sessions invalidate on restart). Set a strong SESSION_SECRET in production.'
    )
  }
  return _devSecret
}

function secret(): string {
  const s = process.env.SESSION_SECRET
  if (process.env.NODE_ENV === 'production') {
    if (!s || KNOWN_PLACEHOLDERS.has(s) || s.length < 32) {
      throw new Error(
        'SESSION_SECRET must be a strong random secret (>=32 chars, not a placeholder) in production'
      )
    }
    return s
  }
  return s && !KNOWN_PLACEHOLDERS.has(s) ? s : devSecret()
}

function sha256Hex(input: string): string {
  return crypto.createHash('sha256').update(input).digest('hex')
}

function timingSafeEqualHex(a: string, b: string): boolean {
  const ba = Buffer.from(a, 'utf8')
  const bb = Buffer.from(b, 'utf8')
  return ba.length === bb.length && crypto.timingSafeEqual(ba, bb)
}

/** Sign a new session for `userId`, persisting a revocable Session row. */
export async function signSession(userId: string): Promise<string> {
  const expMs = Date.now() + SESSION_TTL_MS
  const nonce = crypto.randomBytes(16).toString('hex')
  const payload = `${userId}.${expMs}.${nonce}`
  const sig = crypto.createHmac('sha256', secret()).update(payload).digest('hex')
  const token = `${payload}.${sig}`
  await db.session.create({
    data: { userId, tokenHash: sha256Hex(token), expiresAt: new Date(expMs) },
  })
  return token
}

/** Verify a session token (signature + server-side expiry + revocation). */
export async function verifySession(token: string | undefined): Promise<string | null> {
  if (!token) return null
  const parts = token.split('.')
  if (parts.length !== 4) return null
  const [userId, expMsStr, nonce, sig] = parts as [string, string, string, string]
  const payload = `${userId}.${expMsStr}.${nonce}`
  const expect = crypto.createHmac('sha256', secret()).update(payload).digest('hex')
  if (!timingSafeEqualHex(sig, expect)) return null
  const expMs = Number(expMsStr)
  if (!Number.isFinite(expMs) || Date.now() > expMs) return null
  // Server-side revocation: the Session row must exist and not be revoked.
  try {
    const row = await db.session.findUnique({ where: { tokenHash: sha256Hex(token) } })
    if (!row || row.revokedAt) return null
  } catch {
    return null
  }
  return userId
}

/** Revoke a session server-side (used by logout). */
export async function revokeSession(token: string | undefined): Promise<void> {
  if (!token) return
  try {
    await db.session.updateMany({
      where: { tokenHash: sha256Hex(token), revokedAt: null },
      data: { revokedAt: new Date() },
    })
  } catch {
    // revocation is best-effort; the token still carries an expiry
  }
}

export const sessionCookieOptions = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax' as const,
  maxAge: SESSION_TTL_MS / 1000,
  path: '/',
}
