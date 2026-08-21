import ReplayView from '@/components/ReplayView'

export default function ReplayPage() {
  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto py-6 px-4">
        <h1 className="text-xl font-semibold text-gray-900 mb-1">
          Crash Replay <span className="text-gray-400 font-normal">/ scrub suite</span>
        </h1>
        <p className="text-sm text-gray-500 mb-4">
          Replay a historical crash day through the trained detector — scrub with
          the ×speed and seek sliders, and watch price, Stage-3 score, the
          trailing-vol regime gate, and per-stage pass overlay.
        </p>
        <ReplayView />
      </div>
    </main>
  )
}