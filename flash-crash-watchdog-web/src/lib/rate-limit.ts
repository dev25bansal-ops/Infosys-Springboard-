import { NextRequest } from 'next/server'

/**
 * S2: minimal per-IP rate limiter for the auth endpoints.
 *
 * In-memory (per-process) bucketed limiter. Suitable for a single-instance /
 * SQLite deployment; for multi-instance serverless you'd move this to a shared
 * store (Redis). Returns true when the request is allowed, false when the IP
 * has exceeded MAX attempts in WINDOW_MS.
 */

const WINDOW_MS = 60_000
const MAX_PER_WINDOW = 10
const MAX_BUCKETS = 10_000  // guard against unbounded memory growth

const buckets = new Map<string, number[]>()

export const MIN_PASSWORD_LENGTH = 8

export function clientIp(req: NextRequest): string {
  // SEC-5: prefer the proxy-appended entry (LAST) of x-forwarded-for — the first
  // entry is client-controlled and trivially spoofable to rotate past the limit.
  const fwd = req.headers.get('x-forwarded-for')
  if (fwd) {
    const list = fwd.split(',')
    for (let i = list.length - 1; i >= 0; i--) {
      const ip = list[i]?.trim()
      if (ip && ip.toLowerCase() !== 'unknown') return ip
    }
  }
  return req.headers.get('x-real-ip') || 'unknown'
}

export function checkRateLimit(ip: string, max: number = MAX_PER_WINDOW): boolean {
  const now = Date.now()
  const arr = (buckets.get(ip) || []).filter((t) => now - t < WINDOW_MS)
  if (arr.length >= max) {
    buckets.set(ip, arr)
    return false
  }
  arr.push(now)
  buckets.set(ip, arr)
  return true
}

// SEC-5: periodically evict stale buckets so the in-memory map can't grow
// unbounded (with a spoofable key space an attacker could otherwise exhaust RAM).
export function evictStaleBuckets(): void {
  if (buckets.size < MAX_BUCKETS) return
  const now = Date.now()
  for (const [k, v] of buckets) {
    if (v.every((t) => now - t >= WINDOW_MS)) buckets.delete(k)
  }
}
setInterval(evictStaleBuckets, WINDOW_MS)

export function isStrongPassword(pw: string): boolean {
  return typeof pw === 'string' && pw.length >= MIN_PASSWORD_LENGTH
}