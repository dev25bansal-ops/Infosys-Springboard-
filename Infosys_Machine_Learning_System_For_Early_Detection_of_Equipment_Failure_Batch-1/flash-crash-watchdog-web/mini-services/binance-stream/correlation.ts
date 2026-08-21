/**
 * Real-time correlation-breakdown detector (TS port of the validated
 * ml/flash_crash_watchdog/features/correlation.py).
 *
 * Tracks aligned per-second mid-price series for a basket of symbols and
 * computes the rolling anchor-vs-basket return-correlation. Fires when the
 * correlation drops BELOW an absolute floor and STAYS there for `sustain`
 * seconds (a decoupling event — e.g. LUNA 2022-05-11 fired ~5.9h early,
 * calm days stay silent). Pure logic, no socket.io deps — unit-testable.
 */

export interface CorrelationOptions {
  anchor?: string
  binMs?: number          // time-aligned bin (default 1000 ms)
  corrWindowBins?: number // rolling corr window (bins); 300 = 5 min
  warmupBins?: number
  floorCorr?: number      // absolute floor; corr below this = decoupled
  sustainBins?: number    // must stay below floor this long to fire
  maxPairs?: number
}

export class CorrelationBreakdown {
  private anchor: string
  private binMs: number
  private corrWindowBins: number
  private warmupBins: number
  private floorCorr: number
  private sustainBins: number
  private maxPairs: number

  private latestMid: Record<string, number> = {}
  private bins: Record<string, number[]> = {}        // symbol -> aligned mid series
  private returns: Record<string, number[]> = {}     // symbol -> aligned log-return series
  private corrHistory: number[] = []
  private belowCount = 0
  private lastBin = -1

  constructor(opts: CorrelationOptions = {}) {
    this.anchor = opts.anchor ?? 'BTCUSDT'
    this.binMs = opts.binMs ?? 1000
    this.corrWindowBins = opts.corrWindowBins ?? 300
    this.warmupBins = opts.warmupBins ?? Math.max(60, this.corrWindowBins)
    this.floorCorr = opts.floorCorr ?? 0.4
    this.sustainBins = opts.sustainBins ?? 60
    this.maxPairs = opts.maxPairs ?? 8
  }

  /** Record a mid-price observation; flushes an aligned bin when the second ticks. */
  update(symbol: string, mid: number, tsMs: number): void {
    if (mid == null || !(mid > 0)) return
    this.latestMid[symbol] = mid
    const bin = Math.floor(tsMs / this.binMs)
    if (bin === this.lastBin) return
    this.lastBin = bin
    this.flushBin()
  }

  private flushBin(): void {
    const syms = Object.keys(this.latestMid)
    for (const s of syms) {
      const prev = this.bins[s]?.[this.bins[s].length - 1]
      const cur = this.latestMid[s]
      this.push(this.bins, s, cur)
      if (prev != null && prev > 0 && cur > 0) {
        this.push(this.returns, s, Math.log(cur / prev))
      } else {
        this.push(this.returns, s, NaN) // keep alignment
      }
    }
  }

  private push(map: Record<string, number[]>, key: string, val: number): void {
    ;(map[key] ??= []).push(val)
    if (map[key].length > this.corrWindowBins + 1) map[key].shift()
  }

  private currentCorr(): number | null {
    const a = this.returns[this.anchor]
    if (!a || a.length < this.corrWindowBins) return null
    const aArr = a.slice(-this.corrWindowBins)
    const vals: number[] = []
    for (const [sym, rq] of Object.entries(this.returns)) {
      if (sym === this.anchor || rq.length < this.corrWindowBins) continue
      if (vals.length >= this.maxPairs) break
      const bArr = rq.slice(-this.corrWindowBins)
      const c = corr(aArr, bArr)
      if (Number.isFinite(c)) vals.push(c)
    }
    if (!vals.length) return null
    return vals.reduce((x, y) => x + y, 0) / vals.length
  }

  /** Returns { fire, corr, belowCount } — fire when corr below floor for sustainBins. */
  evaluate(): { fire: boolean; corr: number | null; belowCount: number } {
    const c = this.currentCorr()
    if (c == null) return { fire: false, corr: null, belowCount: this.belowCount }
    this.corrHistory.push(c)
    if (this.corrHistory.length > this.warmupBins * 2) this.corrHistory.shift()
    this.belowCount = c < this.floorCorr ? this.belowCount + 1 : 0
    const fire = this.belowCount >= this.sustainBins
    return { fire, corr: c, belowCount: this.belowCount }
  }

  get floor(): number {
    return this.floorCorr
  }
}

/** Pearson correlation of two arrays (NaN-safe; alignment assumed). */
export function corr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length)
  if (n < 2) return NaN
  let ma = 0, mb = 0, va = 0, vb = 0, cov = 0, k = 0
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(a[i]) || Number.isNaN(b[i])) continue
    ma += a[i]; mb += b[i]; k++
  }
  if (k < 2) return NaN
  ma /= k; mb /= k
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(a[i]) || Number.isNaN(b[i])) continue
    const da = a[i] - ma, db = b[i] - mb
    va += da * da; vb += db * db; cov += da * db
  }
  if (va === 0 || vb === 0) return NaN
  return cov / Math.sqrt(va * vb)
}
