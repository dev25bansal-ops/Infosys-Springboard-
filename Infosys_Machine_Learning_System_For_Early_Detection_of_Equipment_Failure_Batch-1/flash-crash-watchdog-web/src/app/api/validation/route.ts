import { NextRequest, NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'
import { verifySession } from '@/lib/session'

export const runtime = 'nodejs'

// ADV-09 (subset): surface the corrected validation table (produced by
// scripts/run_validation.py -> results/validation.json) to the dashboard, so the
// operating-point results are viewable without re-running heavy Python in the
// web process. A full "run a backtest" service is a follow-on.
const VALIDATION_FILE = join(process.cwd(), '..', 'results', 'validation.json')

export async function GET(req: NextRequest) {
  const userId = await verifySession(req.cookies.get('session')?.value)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  if (!existsSync(VALIDATION_FILE)) {
    return NextResponse.json({ validation: null, hint: 'run scripts/run_validation.py first' })
  }
  try {
    const validation = JSON.parse(readFileSync(VALIDATION_FILE, 'utf-8'))
    return NextResponse.json({ validation })
  } catch {
    return NextResponse.json({ error: 'Could not parse validation.json' }, { status: 500 })
  }
}