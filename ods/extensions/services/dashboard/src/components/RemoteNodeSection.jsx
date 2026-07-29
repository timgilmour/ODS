import { memo } from 'react'
import { Server } from 'lucide-react'
import { GPUCard } from './GPUCard'

const STATUS_STYLES = {
  online: { dot: 'bg-emerald-400', label: 'online', text: 'text-emerald-300' },
  offline: { dot: 'bg-zinc-500', label: 'offline', text: 'text-zinc-400' },
  error: { dot: 'bg-amber-400', label: 'error', text: 'text-amber-300' },
}

function lastSeenLabel(iso) {
  if (!iso) return null
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso)) / 60000))
  return mins < 1 ? 'last seen just now' : `last seen ${mins}m ago`
}

export const RemoteNodeSection = memo(function RemoteNodeSection({ node }) {
  const style = STATUS_STYLES[node.status] || STATUS_STYLES.offline
  const dimmed = node.status !== 'online'
  return (
    <section className={`mt-8 ${dimmed ? 'opacity-60' : ''}`}>
      <div className="flex items-center gap-3 mb-3">
        <Server size={16} className="text-zinc-400" />
        <h2 className="text-lg font-semibold text-white">
          {node.display_name || node.name}
        </h2>
        <span className="text-xs uppercase tracking-wide text-zinc-500">
          {node.platform}
        </span>
        <span className={`flex items-center gap-1.5 text-xs ${style.text}`}>
          <span className={`h-2 w-2 rounded-full ${style.dot}`} />
          {style.label}
        </span>
        {node.status !== 'online' && node.last_seen && (
          <span className="text-xs text-zinc-500">{lastSeenLabel(node.last_seen)}</span>
        )}
      </div>
      {node.error && (
        <div className="mb-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          {node.error}
        </div>
      )}
      {node.serving?.model && (
        <p className="mb-3 text-sm text-zinc-400">
          serving <span className="font-mono text-white">{node.serving.model}</span>
          {node.serving.endpoint_ok ? ' · endpoint healthy' : ' · endpoint unreachable'}
        </p>
      )}
      {node.gpus.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {node.gpus.map((gpu) => <GPUCard key={gpu.uuid} gpu={gpu} />)}
        </div>
      )}
    </section>
  )
})
