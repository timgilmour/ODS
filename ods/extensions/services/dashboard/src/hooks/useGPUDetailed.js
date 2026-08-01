import { useState, useEffect, useRef } from 'react'

// Auth: nginx injects Authorization header for all /api/ requests (see nginx.conf).

const POLL_INTERVAL = 5000
// Two consecutive missed polls (plus slack) with no fresh /api/gpu/detailed
// payload = the numbers on screen are no longer live.
const STALE_AFTER_MS = 12000

export function useGPUDetailed() {
  const [detailed, setDetailed] = useState(null)
  const [history, setHistory] = useState(null)
  const [topology, setTopology] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [stale, setStale] = useState(false)
  const fetchInFlight = useRef(false)
  // Topology is an install-time artifact (written by `ods gpu reassign`); a
  // 404 stays a 404 until an operator regenerates it, so stop re-asking
  // every poll. A page reload after a reassign picks it up again.
  const topologyMissing = useRef(false)
  const lastSuccess = useRef(null)

  useEffect(() => {
    const fetchAll = async () => {
      if (document.hidden) return
      if (fetchInFlight.current) return
      fetchInFlight.current = true
      try {
        const [detRes, histRes, topoRes] = await Promise.all([
          fetch('/api/gpu/detailed'),
          fetch('/api/gpu/history'),
          topologyMissing.current ? Promise.resolve(null) : fetch('/api/gpu/topology'),
        ])
        if (detRes.ok) {
          setDetailed(await detRes.json())
          lastSuccess.current = Date.now()
        }
        if (histRes.ok) setHistory(await histRes.json())
        if (topoRes) {
          if (topoRes.ok) setTopology(await topoRes.json())
          else if (topoRes.status === 404) topologyMissing.current = true
        }
        setError(null)
      } catch (err) {
        setError(err.message)
      } finally {
        fetchInFlight.current = false
        setLoading(false)
        setStale(lastSuccess.current != null && Date.now() - lastSuccess.current > STALE_AFTER_MS)
      }
    }

    fetchAll()
    const interval = setInterval(fetchAll, POLL_INTERVAL)
    const onVisibility = () => { if (!document.hidden) fetchAll() }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      clearInterval(interval)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return { detailed, history, topology, loading, error, stale }
}
