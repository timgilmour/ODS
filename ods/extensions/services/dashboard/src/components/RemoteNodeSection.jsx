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
  // Only an offline node is visually "dead". An errored node is actively
  // reporting a problem worth reading, so it keeps full contrast and relies on
  // the amber badge plus the message box to stand out.
  const dimmed = node.status === 'offline' || !STATUS_STYLES[node.status]
  const gpus = node.gpus ?? []
  const serving = node.serving
  // serving.model is null whenever the probe failed, so gating the whole line
  // on it hid the "node up, inference endpoint down" case entirely.
  const servingLine = serving && (serving.model || serving.endpoint_ok === false
    || serving.container_status)
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
      {servingLine && (
        <p className="mb-3 text-sm text-zinc-400">
          {serving.model
            ? <>serving <span className="font-mono text-white">{serving.model}</span></>
            : <>no model reported</>}
          {serving.endpoint_ok ? ' · endpoint healthy' : ' · endpoint unreachable'}
          {serving.container_status && ` · container ${serving.container_status}`}
        </p>
      )}
      {gpus.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {gpus.map((gpu) => <GPUCard key={gpu.uuid} gpu={gpu} />)}
        </div>
      ) : (
        // A reachable node with an empty GPU list previously rendered an empty
        // card body with no explanation. node.error above carries the agent's
        // own collector message when there is one; this line always states the
        // fact so the state never dead-ends.
        <p className="text-sm text-zinc-500">
          No GPUs reported by this node
          {node.status === 'online' && !node.error && ' (the node reports zero GPUs)'}
        </p>
      )}
    </section>
  )
})
