import { renderHook, act } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { useGPUDetailed } from './useGPUDetailed'

const ok = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) })
const notFound = () => Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) })
const DETAILED = { gpus: [], gpu_count: 0, backend: 'amd', nodes: [] }

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('useGPUDetailed', () => {
  it('stops re-fetching topology after a 404', async () => {
    vi.useFakeTimers()
    const calls = []
    global.fetch = vi.fn((url) => {
      calls.push(url)
      if (url === '/api/gpu/topology') return notFound()
      return ok(DETAILED)
    })
    renderHook(() => useGPUDetailed())
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })       // initial fetch
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })   // two more polls
    expect(calls.filter((u) => u === '/api/gpu/detailed').length).toBeGreaterThanOrEqual(3)
    expect(calls.filter((u) => u === '/api/gpu/topology').length).toBe(1)
  })

  it('reports stale when polls stop succeeding', async () => {
    vi.useFakeTimers()
    let broken = false
    global.fetch = vi.fn((url) => {
      if (broken) return Promise.reject(new Error('network down'))
      if (url === '/api/gpu/topology') return notFound()
      return ok(DETAILED)
    })
    const { result } = renderHook(() => useGPUDetailed())
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(result.current.stale).toBe(false)
    broken = true
    await act(async () => { await vi.advanceTimersByTimeAsync(15000) })
    expect(result.current.stale).toBe(true)
  })
})
