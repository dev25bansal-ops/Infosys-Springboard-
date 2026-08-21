import { create } from 'zustand'

interface AuthState {
  user: { id: string; email: string; name: string | null } | null
  loading: boolean
  setUser: (user: { id: string; email: string; name: string | null } | null) => void
  setLoading: (loading: boolean) => void
  logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  setUser: (user) => set({ user, loading: false }),
  setLoading: (loading) => set({ loading }),
  logout: () => {
    set({ user: null, loading: false })
    fetch('/api/auth/logout', { method: 'POST' }).catch(() => {})
  },
}))

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

export interface TickFeatures {
  velocity: number
  obi: number
  volatility: number
  spreadBps: number
  bidDepth: number
  askDepth: number
}

interface DashboardState {
  alerts: AlertItem[]
  unreadCount: number
  livePrice: number | null
  liveScore: number
  scoreSource: string
  isConnected: boolean
  regime: 'CALM' | 'TENSE' | 'CRASH'
  trailingVolBps: number
  priceHistory: { time: number; price: number }[]
  scoreHistory: { time: number; score: number }[]
  features: TickFeatures
  tickCount: number
  lastTickAt: number | null
  addAlert: (alert: AlertItem) => void
  setLivePrice: (price: number) => void
  setLiveScore: (score: number) => void
  setScoreSource: (s: string) => void
  setConnected: (connected: boolean) => void
  setFeatures: (f: TickFeatures) => void
  setTickCount: (n: number) => void
  setTickMeta: (regime: 'CALM' | 'TENSE' | 'CRASH', trailingVolBps: number) => void
  markAllRead: () => void
}

const emptyFeatures: TickFeatures = {
  velocity: 0, obi: 0, volatility: 0, spreadBps: 0, bidDepth: 0, askDepth: 0,
}

export const useDashboardStore = create<DashboardState>((set) => ({
  alerts: [],
  unreadCount: 0,
  livePrice: null,
  liveScore: 0,
  scoreSource: '',
  isConnected: false,
  regime: 'CALM',
  trailingVolBps: 0,
  priceHistory: [],
  scoreHistory: [],
  features: emptyFeatures,
  tickCount: 0,
  lastTickAt: null,
  addAlert: (alert) =>
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, 100),
      unreadCount: state.unreadCount + 1,
    })),
  setLivePrice: (price) =>
    set((state) => ({
      livePrice: price,
      priceHistory: [...state.priceHistory, { time: Date.now(), price }].slice(-120),
    })),
  setLiveScore: (score) =>
    set((state) => ({
      liveScore: score,
      scoreHistory: [...state.scoreHistory, { time: Date.now(), score }].slice(-120),
      lastTickAt: Date.now(),
    })),
  setScoreSource: (s) => set({ scoreSource: s }),
  setConnected: (connected) => set({ isConnected: connected }),
  setFeatures: (f) => set({ features: f }),
  setTickCount: (n) => set({ tickCount: n }),
  setTickMeta: (regime, trailingVolBps) => set({ regime, trailingVolBps }),
  markAllRead: () => set({ unreadCount: 0 }),
}))
