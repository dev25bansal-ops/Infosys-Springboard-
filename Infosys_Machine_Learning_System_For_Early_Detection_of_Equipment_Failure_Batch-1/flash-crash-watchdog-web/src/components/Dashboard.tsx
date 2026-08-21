'use client'

import { useEffect, useState } from 'react'
import { useAuthStore, useDashboardStore } from '@/lib/stores'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart, ReferenceLine, BarChart, Bar, Cell } from 'recharts'
import { Activity, TrendingDown, Zap, Bell, LogOut, Wifi, WifiOff, AlertTriangle, CheckCircle, Clock, Layers, BarChart3 } from 'lucide-react'

interface AlertItem {
  id: string
  symbol: string
  price: number
  score: number
  message: string
  severity: string
  createdAt: string
  type?: 'crash' | 'correlation'
  trailingVolBps?: number
  threshold?: number
  gate?: number
}

const CASCADE_STAGES: { name: string; color: string; desc: string; disabled?: boolean }[] = [
  { name: 'Stage 1 · Statistical', color: '#9ca3af', desc: 'z-score pre-filter' },
  { name: 'Stage 2 · Isolation Forest', color: '#f59e0b', desc: 'unsupervised gate' },
  { name: 'Stage 3 · TCN', color: '#ef4444', desc: 'focal-loss crash classifier' },
  { name: 'Stage 4 · Correlation', color: '#8b5cf6', desc: 'cross-asset decoupling (live)' },
  { name: 'Decision · Bayesian', color: '#10b981', desc: 'posterior ≥ 0.5' },
]

// AF-3: calm-day regime gate (bps) — must match operating.yml + live mini-stream.
const REGIME_GATE_BPS = 2

export function Dashboard() {
  const { user, logout } = useAuthStore()
  const {
    alerts, unreadCount, livePrice, liveScore, isConnected, scoreSource,
    priceHistory, scoreHistory, features, tickCount, lastTickAt, regime, trailingVolBps,
    markAllRead,
  } = useDashboardStore()
  const [dbAlerts, setDbAlerts] = useState<AlertItem[]>([])

  useEffect(() => {
    fetch('/api/alerts')
      .then(r => r.json())
      .then(data => setDbAlerts(data.alerts || []))
      .catch(() => {})
  }, [alerts.length])

  const formatPrice = (p: number | null) => {
    if (!p) return '-'
    return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }

  const scoreColor = liveScore > 0.6 ? 'text-red-500' : liveScore > 0.3 ? 'text-amber-500' : 'text-green-500'
  const scoreBg = liveScore > 0.6 ? 'bg-red-50 border-red-200' : liveScore > 0.3 ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200'
  const statusLabel = liveScore > 0.6 ? 'CRASH RISK' : liveScore > 0.3 ? 'ELEVATED' : 'NORMAL'

  const allAlerts = [...alerts, ...dbAlerts.filter(db => !alerts.find(a => a.id === db.id))]
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, 50)

  const chartData = priceHistory.map((p, i) => ({
    time: new Date(p.time).toLocaleTimeString(),
    price: p.price,
    score: scoreHistory[i]?.score || 0,
  }))

  const featureBars = [
    { name: 'Velocity', value: Math.min(100, features.velocity * 50), color: '#ef4444' },
    { name: 'OBI', value: Math.min(100, Math.abs(features.obi) * 200), color: '#f59e0b' },
    { name: 'Volatility', value: Math.min(100, features.volatility * 1000), color: '#8b5cf6' },
    { name: 'Spread', value: Math.min(100, features.spreadBps), color: '#10b981' },
  ]

  const cascadeReached = liveScore > 0.6 ? 4 : liveScore > 0.3 ? 3 : liveScore > 0.15 ? 2 : 1
  const lastTickAgo = lastTickAt ? Math.max(0, Math.floor((Date.now() - lastTickAt) / 1000)) : null

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-red-500" />
              <span className="font-bold text-gray-900">Flash Crash Watchdog</span>
              <Badge variant={isConnected ? 'default' : 'secondary'} className={`ml-2 ${isConnected ? 'bg-emerald-500 hover:bg-emerald-600' : ''}`}>
                {isConnected ? <><Wifi className="w-3 h-3 mr-1" /> Live</> : <><WifiOff className="w-3 h-3 mr-1" /> Offline</>}
              </Badge>
              {lastTickAgo !== null && (
                <span className="hidden sm:inline-flex items-center gap-1 text-xs text-gray-400 ml-2">
                  <Clock className="w-3 h-3" /> {lastTickAgo}s ago - {tickCount} ticks
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <a href="/replay" className="text-sm text-blue-600 hover:underline hidden sm:block">Replay</a>
              <span className="text-sm text-gray-500 hidden sm:block">{user?.email}</span>
              <Button variant="ghost" size="sm" onClick={logout}>
                <LogOut className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="border-gray-200">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 font-medium">BTC Price</span>
                <TrendingDown className="w-3.5 h-3.5 text-gray-400" />
              </div>
              <div className="mt-1 text-xl font-bold text-gray-900">{formatPrice(livePrice)}</div>
            </CardContent>
          </Card>

          <Card className={`border-2 ${scoreBg}`}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 font-medium">Anomaly Score</span>
                <div className="flex items-center gap-1">
                  {scoreSource && (
                    <Badge
                      variant={scoreSource === 'tcn' ? 'default' : 'secondary'}
                      className={`text-[10px] px-1.5 py-0 ${
                        scoreSource === 'tcn'
                          ? 'bg-emerald-500 hover:bg-emerald-600'
                          : scoreSource === 'warmup'
                            ? 'bg-amber-400 hover:bg-amber-500'
                            : 'bg-gray-300 hover:bg-gray-400'
                      }`}
                    >
                      {scoreSource.toUpperCase()}
                    </Badge>
                  )}
                  <Zap className="w-3.5 h-3.5 text-gray-400" />
                </div>
              </div>
              <div className={`mt-1 text-xl font-bold ${scoreColor}`}>
                {(liveScore * 100).toFixed(1)}%
              </div>
            </CardContent>
          </Card>

          <Card className="border-gray-200">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 font-medium">Alerts Fired</span>
                <Bell className="w-3.5 h-3.5 text-gray-400" />
              </div>
              <div className="mt-1 text-xl font-bold text-gray-900">{alerts.length}</div>
            </CardContent>
          </Card>

          <Card className="border-gray-200">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500 font-medium">Status</span>
                {liveScore > 0.5 ? <AlertTriangle className="w-3.5 h-3.5 text-red-500" /> : <CheckCircle className="w-3.5 h-3.5 text-green-500" />}
              </div>
              <div className={`mt-1 text-xl font-bold ${scoreColor}`}>{statusLabel}</div>
            </CardContent>
          </Card>
        </div>

        {/* AF-3: Market-regime status card (trailing-vol gate + score-driven) */}
        <Card className={`border-2 mb-4 ${
          regime === 'CRASH' ? 'border-red-400 bg-red-50'
            : regime === 'TENSE' ? 'border-amber-400 bg-amber-50'
              : 'border-emerald-300 bg-emerald-50'}`}>
          <CardContent className="p-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              {regime === 'CRASH' ? <AlertTriangle className="w-5 h-5 text-red-600" />
                : regime === 'TENSE' ? <Zap className="w-5 h-5 text-amber-600" />
                  : <CheckCircle className="w-5 h-5 text-emerald-600" />}
              <div>
                <div className="text-xs text-gray-500 font-medium">Market regime</div>
                <div className={`text-lg font-bold ${
                  regime === 'CRASH' ? 'text-red-600'
                    : regime === 'TENSE' ? 'text-amber-600' : 'text-emerald-600'}`}>
                  {regime}
                </div>
              </div>
            </div>
            <div className="text-xs text-gray-500 text-right">
              <div>trailing vol <span className="font-medium text-gray-900">{trailingVolBps.toFixed(1)} bps</span></div>
              <div>gate <span className="font-medium text-gray-900">{REGIME_GATE_BPS} bps</span></div>
              <div>regime = score vs threshold + vol gate</div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="lg:col-span-2 border-gray-200">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-700">BTC/USDT Price (Live)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval={20} />
                  <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={60} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => ['$' + v.toFixed(2), 'Price']} />
                  <Area type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={1.5} fill="url(#priceGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="border-gray-200">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-700">Anomaly Score</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval={20} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={40} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => [(v * 100).toFixed(1) + '%', 'Score']} />
                  <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Alert', fontSize: 10, fill: '#ef4444' }} />
                  <Area type="monotone" dataKey="score" stroke="#ef4444" strokeWidth={1.5} fill="url(#scoreGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card className="border-gray-200">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-gray-500" />
                <CardTitle className="text-sm font-medium text-gray-700">Detection Pipeline</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              {CASCADE_STAGES.map((s, i) => {
                const reached = i <= cascadeReached && !s.disabled
                return (
                  <div key={s.name} className="flex items-center gap-3">
                    <div className="w-2 h-8 rounded-sm transition-colors" style={{ backgroundColor: reached ? s.color : '#e5e7eb' }} />
                    <div className="flex-1 flex items-center justify-between">
                      <div>
                        <div className="text-sm font-medium text-gray-900">{s.name}</div>
                        <div className="text-xs text-gray-400">{s.desc}</div>
                      </div>
                      <Badge variant={reached ? 'default' : 'secondary'} className="text-xs">
                        {reached ? 'passed' : 'idle'}
                      </Badge>
                    </div>
                  </div>
                )
              })}
              <div className="pt-2 mt-2 border-t border-gray-100 text-xs text-gray-500">
                Highest stage implied by the current score: <span className="font-semibold text-gray-900">{CASCADE_STAGES[Math.min(cascadeReached, CASCADE_STAGES.length - 1)]?.name || '-'}</span>
                <div className="mt-1">Illustrative (score-based). Stage 4 disabled pending a multi-symbol feed; alerts fire above the calibrated threshold with a cooldown.</div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-gray-200">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-gray-500" />
                <CardTitle className="text-sm font-medium text-gray-700">Live Feature Breakdown</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={featureBars} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
                  <XAxis type="number" domain={[0, 100]} hide />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={70} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v: number) => [v.toFixed(1), 'Strength']} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {featureBars.map((b, i) => (<Cell key={i} fill={b.color} />))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-3 text-xs">
                <FeatureRow label="OBI" value={features.obi.toFixed(4)} />
                <FeatureRow label="Volatility" value={features.volatility.toFixed(4)} />
                {/* spread on liquid BTC is ~0.002 bps — show 3 decimals so it isn't a frozen-looking 0.0 */}
                <FeatureRow label="Spread (bps)" value={features.spreadBps.toFixed(3)} />
                <FeatureRow label="Velocity" value={features.velocity.toFixed(3) + '%'} />
                <FeatureRow label="Bid Depth" value={features.bidDepth.toFixed(2)} />
                <FeatureRow label="Ask Depth" value={features.askDepth.toFixed(2)} />
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="border-gray-200">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-medium text-gray-700">Alert Feed</CardTitle>
              {unreadCount > 0 && (<Badge variant="destructive" className="text-xs">{unreadCount} new</Badge>)}
            </div>
            {unreadCount > 0 && (
              <Button variant="ghost" size="sm" onClick={() => { markAllRead(); fetch('/api/alerts/read', { method: 'POST' }) }}>
                Mark all read
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {allAlerts.length === 0 ? (
              <div className="text-center py-8 text-gray-400 text-sm">
                No alerts yet. The detector is monitoring the market in real time.
                <br />
                Alerts fire when anomaly score exceeds 0.6 with a 10-second cooldown.
              </div>
            ) : (
              <ScrollArea className="h-72">
                <div className="space-y-2">
                  {allAlerts.map((alert) => {
                    const isCorr = alert.type === 'correlation'
                    return (
                      <div key={alert.id} className={`flex items-start gap-3 p-3 rounded-lg border ${
                        isCorr ? 'border-purple-200 bg-purple-50'
                          : alert.severity === 'critical' ? 'border-red-200 bg-red-50'
                            : 'border-amber-200 bg-amber-50'}`}>
                        <div className={`mt-0.5 ${
                          isCorr ? 'text-purple-500'
                            : alert.severity === 'critical' ? 'text-red-500' : 'text-amber-500'}`}>
                          {isCorr ? <Layers className="w-4 h-4" />
                            : alert.severity === 'critical' ? <AlertTriangle className="w-4 h-4" /> : <Bell className="w-4 h-4" />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-medium text-gray-900">{alert.symbol}</span>
                            <span className="text-xs text-gray-500">${alert.price?.toFixed(2)}</span>
                            {isCorr ? (
                              <Badge variant="outline" className="text-xs text-purple-600 border-purple-300">CORRELATION</Badge>
                            ) : (
                              <Badge variant={alert.severity === 'critical' ? 'destructive' : 'secondary'} className="text-xs">
                                Score: {(alert.score * 100).toFixed(0)}%
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-gray-600 mt-0.5">{alert.message}</p>
                          {/* AF-3: alert reason card — score / trailing-vol / threshold / gate */}
                          {!isCorr && (
                            <div className="flex flex-wrap gap-1.5 mt-1 text-[10px] text-gray-500">
                              <span className="rounded bg-gray-100 px-1.5 py-0.5">score {alert.score?.toFixed(2)}</span>
                              <span className="rounded bg-gray-100 px-1.5 py-0.5">tv {alert.trailingVolBps?.toFixed(1) ?? '—'}bps</span>
                              <span className="rounded bg-gray-100 px-1.5 py-0.5">threshold {alert.threshold ?? 0.5}</span>
                              <span className="rounded bg-gray-100 px-1.5 py-0.5">gate {alert.gate ?? 2}bps</span>
                            </div>
                          )}
                          <span className="text-xs text-gray-400">{new Date(alert.createdAt).toLocaleString()}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </main>

      <footer className="border-t border-gray-200 bg-white/50 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 text-xs text-gray-400 text-center">
          Flash Crash Watchdog - Powered by Binance WebSocket + ML Anomaly Detection -{' '}
          <a href="https://huggingface.co/Dev2506/flash-crash-watchdog" target="_blank" rel="noopener noreferrer" className="underline">
            View on Hugging Face
          </a>
        </div>
      </footer>
    </div>
  )
}

function FeatureRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono font-semibold text-gray-900">{value}</span>
    </div>
  )
}
