'use client'

import { useEffect } from 'react'
import { io } from 'socket.io-client'
import { toast } from 'sonner'
import { useAuthStore, useDashboardStore, type TickFeatures } from '@/lib/stores'
import { LoginScreen } from '@/components/LoginScreen'
import { Dashboard } from '@/components/Dashboard'

export default function Home() {
  const { user, loading, setUser } = useAuthStore()
  const {
    setLivePrice, setLiveScore, setConnected, addAlert,
    setFeatures, setTickCount, setScoreSource, setTickMeta,
  } = useDashboardStore()

  useEffect(() => {
    fetch('/api/auth/me')
      .then(r => r.json())
      .then(data => { setUser(data.user || null) })
      .catch(() => setUser(null))
  }, [setUser])

  useEffect(() => {
    if (!user) return

    const envUrl = process.env.NEXT_PUBLIC_SOCKET_URL
    const isCaddyGateway = typeof window !== 'undefined' && window.location.port === '81'
    const SOCKET_URL = envUrl || (isCaddyGateway ? '/' : 'http://localhost:3003')
    const SOCKET_QUERY = isCaddyGateway && !envUrl ? { XTransformPort: '3003' } : undefined
    const s = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      query: SOCKET_QUERY,
      forceNew: true,
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 2000,
      timeout: 15000,
    })

    s.on('connect', () => {
      setConnected(true)
      toast.success('Live stream connected', { description: 'Receiving BTC/USDT order book updates from Binance' })
    })

    s.on('disconnect', () => {
      setConnected(false)
      toast.warning('Stream disconnected', { description: 'Attempting to reconnect...' })
    })

    s.on('status', (data: { connected: boolean; message: string; stale?: boolean; stalenessMs?: number }) => {
      setConnected(data.connected)
      // MLOPS-04: a dark dashboard must be distinguishable from calm.
      if (data.stale) {
        toast.error('Feed stale — no market data', {
          description: data.message || `No data for a while. Check the stream service (port 3003).`,
        })
      }
    })

    s.on('history', (data: {
      priceHistory: { time: number; price: number }[]
      scoreHistory: { time: number; score: number }[]
      currentPrice: number
      currentScore: number
      tickCount: number
      alertsFired: number
    }) => {
      if (data.currentPrice) setLivePrice(data.currentPrice)
      if (data.currentScore !== undefined) setLiveScore(data.currentScore)
      if (data.tickCount) setTickCount(data.tickCount)
    })

    s.on('tick', (data: {
      price: number
      score: number
      source?: string
      velocity: number
      obi: number
      volatility: number
      spreadBps: number
      bidDepth: number
      askDepth: number
      timestamp: number
      tickCount: number
      regime?: 'CALM' | 'TENSE' | 'CRASH'
      trailingVolBps?: number
    }) => {
      setLivePrice(data.price)
      setLiveScore(data.score)
      if (data.source) setScoreSource(data.source)
      if (data.regime) setTickMeta(data.regime, data.trailingVolBps ?? 0)
      const f: TickFeatures = {
        velocity: data.velocity, obi: data.obi, volatility: data.volatility,
        spreadBps: data.spreadBps, bidDepth: data.bidDepth, askDepth: data.askDepth,
      }
      setFeatures(f)
      if (data.tickCount) setTickCount(data.tickCount)
    })

    s.on('alert', (alert: {
      id: string
      symbol: string
      price: number
      score: number
      message: string
      severity: string
      createdAt: string
    }) => {
      addAlert(alert)

      fetch('/api/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alert),
      }).catch(() => {})

      const isCritical = alert.severity === 'critical' || alert.score > 0.8
      if (isCritical) {
        toast.error(`CRASH ALERT - ${alert.symbol}`, {
          description: `Score ${(alert.score * 100).toFixed(0)}% - $${alert.price.toFixed(2)} - ${alert.message.slice(0, 80)}`,
          duration: 10000,
        })
      } else {
        toast.warning(`Anomaly - ${alert.symbol}`, {
          description: `Score ${(alert.score * 100).toFixed(0)}% - ${alert.message.slice(0, 80)}`,
          duration: 7000,
        })
      }

      if (Notification.permission === 'granted') {
        new Notification(`Flash Crash Alert - ${alert.symbol}`, {
          body: `Score ${(alert.score * 100).toFixed(0)}% - $${alert.price.toFixed(2)}`,
        })
      }
    })

    // AF-1: correlation-breakdown alert (distinct type — one asset decoupling from the market)
    s.on('correlation-alert', (alert: {
      id: string
      symbol: string
      price: number
      score: number
      message: string
      severity: string
      createdAt: string
    }) => {
      addAlert({ ...alert, type: 'correlation' })
      fetch('/api/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...alert, type: 'correlation' }),
      }).catch(() => {})
      toast.warning(`CORRELATION BREAKDOWN - ${alert.symbol}`, {
        description: alert.message.slice(0, 90),
        duration: 10000,
      })
    })

    if (Notification.permission === 'default') {
      Notification.requestPermission()
    }

    // MLOPS-01: catch up on alerts that fired while no dashboard was connected
    // (durable source-side outbox). Persist each and ack so the outbox marks
    // them delivered.
    s.on('outbox-sync', (alerts: {
      id: string; symbol: string; price: number; score: number
      message: string; severity: string; createdAt: string
    }[]) => {
      if (!Array.isArray(alerts) || alerts.length === 0) return
      const ids: string[] = []
      for (const alert of alerts) {
        addAlert(alert)
        ids.push(alert.id)
        fetch('/api/alerts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(alert),
        }).catch(() => {})
      }
      s.emit('alert-acked', ids)
      if (alerts.some(a => a.severity === 'critical')) {
        toast.warning(`Caught up ${alerts.length} alert(s) that fired while you were away`, { duration: 8000 })
      }
    })

    return () => { s.disconnect() }
  }, [user, setLivePrice, setLiveScore, setConnected, addAlert, setFeatures, setTickCount, setScoreSource])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-pulse text-gray-400">Loading...</div>
      </div>
    )
  }

  if (!user) return <LoginScreen />
  return <Dashboard />
}
