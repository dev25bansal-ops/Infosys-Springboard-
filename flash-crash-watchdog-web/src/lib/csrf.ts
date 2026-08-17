// SEC-16: defense-in-depth CSRF guard on cookie-authenticated writes.
//
// Sessions are SameSite=lax (which already blocks cross-site cookie sends on
// POST), but an explicit Origin check hardens the alert-write surface: a
// state-changing request must come from the same origin that serves the app.
import { NextRequest } from 'next/server'

export function isSameOrigin(req: NextRequest): boolean {
  const origin = req.headers.get('origin')
  if (!origin) return true // non-browser clients (server-to-server) carry no Origin
  const host = req.headers.get('host') || ''
  try {
    const o = new URL(origin)
    return o.host === host || host.endsWith('.' + o.host)
  } catch {
    return false
  }
}
