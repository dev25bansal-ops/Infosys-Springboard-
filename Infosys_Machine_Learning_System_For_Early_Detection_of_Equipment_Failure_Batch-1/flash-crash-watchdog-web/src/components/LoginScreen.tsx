'use client'

import { useState } from 'react'
import { useAuthStore } from '@/lib/stores'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Activity, TrendingDown, Zap, ShieldAlert, Gauge, Layers, Radio } from 'lucide-react'

export function LoginScreen() {
  const { setUser } = useAuthStore()
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const endpoint = mode === 'login' ? '/api/auth/login' : '/api/auth/register'
      const body = mode === 'login' ? { email, password } : { email, password, name }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()

      if (!res.ok) {
        setError(data.error || 'Something went wrong')
        return
      }
      setUser(data)
    } catch {
      setError('Network error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <div className="border-b border-gray-200 bg-white/70 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-12 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-red-500" />
            <TrendingDown className="w-3.5 h-3.5 text-orange-500" />
            <Zap className="w-3.5 h-3.5 text-amber-500" />
            <span className="ml-1.5 font-bold text-gray-900 text-sm">Flash Crash Watchdog</span>
          </div>
          <a href="https://huggingface.co/Dev2506/flash-crash-watchdog" target="_blank" rel="noopener noreferrer" className="text-xs text-gray-500 hover:text-gray-700">
            huggingface.co/Dev2506/flash-crash-watchdog
          </a>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 lg:py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-16 items-center">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-50 border border-red-200 text-red-700 text-xs font-medium">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
              </span>
              LIVE - BTC/USDT order book streaming
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900 leading-[1.05]">
              Catch the crash
              <br />
              <span className="bg-gradient-to-r from-red-500 via-orange-500 to-amber-500 bg-clip-text text-transparent">
                before the chart does.
              </span>
            </h1>

            <p className="text-base sm:text-lg text-gray-600 max-w-xl">
              A 5-stage hybrid detection cascade ingests live Binance depth and trade streams,
              extracts 20 microstructure features, and runs a trained Temporal Convolutional
              Network to flag flash-crash precursors in real time - alerts fire straight
              to your dashboard.
            </p>

            <div className="grid grid-cols-2 gap-3 max-w-xl">
              <FeatureChip icon={<Layers className="w-4 h-4 text-red-500" />} title="5-stage cascade" subtitle="Stat to iForest to TCN to Transformer to Bayes" />
              <FeatureChip icon={<Gauge className="w-4 h-4 text-orange-500" />} title="20 features" subtitle="OBI, VPIN, Kyle lambda, vol, spread" />
              <FeatureChip icon={<ShieldAlert className="w-4 h-4 text-amber-500" />} title="93.3% val acc" subtitle="287k real windows, Focal Loss" />
              <FeatureChip icon={<Radio className="w-4 h-4 text-emerald-500" />} title="Sub-ms ingest" subtitle="Binance WS, auto-reconnect" />
            </div>
          </div>

          <div className="w-full max-w-md mx-auto lg:ml-auto">
            <Card className="shadow-xl border-gray-200">
              <CardHeader>
                <CardTitle className="text-center">
                  {mode === 'login' ? 'Sign In' : 'Create Account'}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  {mode === 'register' && (
                    <div className="space-y-2">
                      <Label htmlFor="name">Name</Label>
                      <Input id="name" type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="John Doe" className="bg-white" />
                    </div>
                  )}
                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required className="bg-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="password">Password</Label>
                    <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="........" required className="bg-white" />
                  </div>

                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}

                  <Button type="submit" className="w-full bg-gray-900 hover:bg-gray-800 text-white" disabled={loading}>
                    {loading ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
                  </Button>
                </form>

                <div className="mt-4 text-center text-sm">
                  <button
                    type="button"
                    onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError('') }}
                    className="text-gray-500 hover:text-gray-700 underline"
                  >
                    {mode === 'login' ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
                  </button>
                </div>

                <div className="mt-6 pt-4 border-t border-gray-100 text-xs text-gray-400 text-center space-y-1">
                  <div>By signing in you accept that alert signals are informational only.</div>
                  <div>Not financial advice. Trade at your own risk.</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <footer className="border-t border-gray-200 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 text-xs text-gray-400 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>Flash Crash Watchdog - Binance WebSocket - TCN Anomaly Detection</div>
          <div>Built on Next.js 16 - Prisma - Socket.io</div>
        </div>
      </footer>
    </div>
  )
}

function FeatureChip({ icon, title, subtitle }: { icon: React.ReactNode; title: string; subtitle: string }) {
  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg bg-white border border-gray-200 shadow-sm">
      <div className="mt-0.5">{icon}</div>
      <div className="min-w-0">
        <div className="text-sm font-semibold text-gray-900">{title}</div>
        <div className="text-xs text-gray-500 leading-tight">{subtitle}</div>
      </div>
    </div>
  )
}
