'use client'

import { useEffect, useRef, useState } from 'react'
import { io, Socket } from 'socket.io-client'
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

interface Pt { t: number; p: number; s3: number; tv: number; s1: number; s2: number }
interface RAlert { t: number; price: number; s3: number }

const SOCKET_URL = process.env.NEXT_PUBLIC_REPLAY_URL || 'http://localhost:3004'
const DAYS = ['btc-0519', 'eth-0805', 'lu-0510']

export default function ReplayView() {
  const sock = useRef<Socket | null>(null)
  const [points, setPoints] = useState<Pt[]>([])
  const [alerts, setAlerts] = useState<RAlert[]>([])
  const [speed, setSpeed] = useState(10)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)      // 0..100
  const [day, setDay] = useState<string>('btc-0519')
  const [thr, setThr] = useState(0.5)
  const [gate, setGate] = useState(2)
  const [showS1, setShowS1] = useState(false)
  const [showS2, setShowS2] = useState(false)
  const maxPts = 800

  useEffect(() => {
    const s = io(SOCKET_URL, { transports: ['websocket', 'polling'], reconnection: true })
    sock.current = s

    s.on('replay:load', (d: any) => {
      if (d?.label) { setThr(d.threshold); setGate(d.gateBps); setProgress(0); setPoints([]); setAlerts([]) }
    })
    s.on('replay:data', (d: any) => {
      const pts = d.points as Pt[]
      setPoints((prev) => [...prev, ...pts].slice(-maxPts))
      if (d.alerts?.length) setAlerts((prev) => [...prev, ...d.alerts].slice(-50))
    })
    s.on('replay:state', (st: any) => { setPlaying(st.playing); setProgress(st.index / (st.total || 1) * 100 || 0) })
    s.emit('replay:load', day)
    return () => { s.disconnect() }
  }, [])

  const cmd = (label: string, ...a: any[]) => sock.current?.emit(label, ...a)
  const onLoad = (d: string) => { setDay(d); cmd('replay:load', d) }
  const onPlay = () => { setPlaying(!playing); cmd(playing ? 'replay:pause' : 'replay:play') }
  const onSpeed = (x: number) => { setSpeed(x); cmd('replay:speed', x) }
  const onSeek = (frac: number) => { setProgress(frac); cmd('replay:seek', frac / 100) }

  // enrich points with per-stage pass booleans for overlay display
  const data = points.map((pt) => ({ ...pt, s1b: pt.s1 === 1, s2b: pt.s2 === 1 }))

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-3 flex-wrap bg-gray-50 border rounded-lg p-3">
        <select value={day} onChange={(e) => onLoad(e.target.value)} className="border rounded px-2 py-1 text-sm">
          {DAYS.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <button onClick={onPlay} className={`px-3 py-1 rounded text-sm text-white ${playing ? 'bg-amber-500' : 'bg-emerald-500'}`}>
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <label className="text-xs text-gray-600">Speed ×{speed}
          <input type="range" min={1} max={1000} value={speed}
            onChange={(e) => onSpeed(Number(e.target.value))} className="ml-2" />
        </label>
        <label className="text-xs text-gray-600">Seek
          <input type="range" min={0} max={100} value={progress}
            onChange={(e) => onSeek(Number(e.target.value))} className="ml-2" />
        </label>
        <label className="text-xs text-gray-600"><input type="checkbox" checked={showS1} onChange={(e) => setShowS1(e.target.checked)} /> S1</label>
        <label className="text-xs text-gray-600"><input type="checkbox" checked={showS2} onChange={(e) => setShowS2(e.target.checked)} /> S2</label>
        <span className="text-xs text-gray-500 ml-auto">threshold={thr} · vol-gate={gate}bps · {points.length} pts</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="border rounded-lg p-2">
          <div className="text-xs font-medium text-gray-700 mb-1">Price</div>
          <div className="h-32">
            <ResponsiveContainer>
              <AreaChart data={data}>
                <defs><linearGradient id="pg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#1f77b4" stopOpacity={0.4}/><stop offset="95%" stopColor="#1f77b4" stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="t" hide /><YAxis domain={['auto', 'auto']} width={44} />
                <Tooltip /><Area type="monotone" dataKey="p" stroke="#1f77b4" fill="url(#pg)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="border rounded-lg p-2">
          <div className="text-xs font-medium text-gray-700 mb-1">Stage-3 score v threshold</div>
          <div className="h-32">
            <ResponsiveContainer>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="t" hide /><YAxis domain={[0, 1]} width={36} />
                <Tooltip /><ReferenceLine y={thr} stroke="#000" strokeDasharray="4 4" label={{ value: 'thr', fontSize: 10 }} />
                <Line type="monotone" dataKey="s3" stroke="#d62728" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="border rounded-lg p-2">
          <div className="text-xs font-medium text-gray-700 mb-1">Trailing-vol (bps) v gate</div>
          <div className="h-32">
            <ResponsiveContainer>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="t" hide /><YAxis width={44} />
                <Tooltip /><ReferenceLine y={gate} stroke="#000" strokeDasharray="4 4" label={{ value: 'gate', fontSize: 10 }} />
                <Line type="monotone" dataKey="tv" stroke="#2ca02c" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {(showS1 || showS2) && (
        <div className="border rounded-lg p-2 text-xs text-gray-600">
          Per-stage pass overlay: {showS1 && <span className="font-medium">Stage-1 passes at points where the marker line is up &nbsp;</span>}
          {showS2 && <span className="font-medium">Stage-2 similarly.</span>}
          (Cameline counts of passing ticks for the viewed window below.)
          <div className="mt-1">
            S1 passing: <b>{data.filter((p) => p.s1b).length}</b> · S2 passing: <b>{data.filter((p) => p.s2b).length}</b> / {data.length} pts
          </div>
        </div>
      )}

      <div className="border rounded-lg p-2">
        <div className="text-xs font-medium text-gray-700 mb-1">Alerts ({alerts.length})</div>
        {alerts.length === 0 ? <div className="text-xs text-gray-400">None yet</div> : (
          <div className="flex flex-wrap gap-2">
            {alerts.map((a, i) => (
              <span key={i} className="text-xs bg-red-50 border border-red-200 text-red-700 rounded px-2 py-0.5">
                ⚠ {a.price?.toFixed(2)} · s3={a.s3?.toFixed(2)} · t={a.t}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}