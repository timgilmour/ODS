import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render } from '../test/test-utils'

// The hook is mocked so these cases exercise GPUMonitor's own rendering
// decisions (which sections appear for which payload) without fetch plumbing.
const useGPUDetailed = vi.fn()
vi.mock('../hooks/useGPUDetailed', () => ({
  useGPUDetailed: () => useGPUDetailed(),
}))

const GPUMonitor = (await import('./GPUMonitor')).default

const remoteGpu = {
  index: 0, uuid: 'GPU-remote', name: 'NVIDIA GB10',
  memory_used_mb: 2048, memory_total_mb: 122880, memory_percent: 1.7,
  utilization_percent: 40, temperature_c: 55, power_w: 90,
  memory_type: 'unified', assigned_services: [],
}

const node = {
  name: 'sparky', display_name: 'DGX Spark GB10', platform: 'nvidia',
  status: 'online', last_seen: new Date().toISOString(),
  gpus: [remoteGpu], serving: null, error: null,
}

const zeroAggregate = {
  name: 'No local GPU', memory_used_mb: 0, memory_total_mb: 0,
  memory_percent: 0, utilization_percent: 0, temperature_c: 0, power_w: null,
  gpu_count: 0, memory_usage_available: false, utilization_available: false,
  temperature_available: false,
}

describe('GPUMonitor remote node sections', () => {
  beforeEach(() => {
    useGPUDetailed.mockReset()
  })

  it('renders remote node sections when the host has no local GPUs', () => {
    useGPUDetailed.mockReturnValue({
      detailed: {
        gpu_count: 0, backend: 'nvidia', gpus: [], aggregate: zeroAggregate,
        assignment: null, split_mode: null, tensor_split: null, nodes: [node],
      },
      history: null, topology: null, loading: false, error: null,
    })

    render(<GPUMonitor />)

    expect(screen.getByText('DGX Spark GB10')).toBeInTheDocument()
    // GPUCard strips the "NVIDIA " brand prefix by design (see GPUCard.jsx).
    expect(screen.getByText('GB10')).toBeInTheDocument()
    expect(screen.queryByText(/GPU data unavailable/i)).not.toBeInTheDocument()
  })

  it('still renders remote node sections in the unavailable branch', () => {
    // A local collector failure must not blank a remote node the API already
    // reported: the last good payload keeps its nodes on screen.
    useGPUDetailed.mockReturnValue({
      detailed: {
        gpu_count: 0, backend: 'nvidia', gpus: [], aggregate: zeroAggregate,
        nodes: [node],
      },
      history: null, topology: null, loading: false, error: 'network down',
    })

    render(<GPUMonitor />)

    expect(screen.getByText(/GPU data unavailable/i)).toBeInTheDocument()
    expect(screen.getByText('DGX Spark GB10')).toBeInTheDocument()
  })

  it('keeps node sections on the overview tab, not under History charts', async () => {
    useGPUDetailed.mockReturnValue({
      detailed: {
        gpu_count: 0, backend: 'nvidia', gpus: [], aggregate: zeroAggregate,
        nodes: [node],
      },
      history: null, topology: null, loading: false, error: null,
    })

    render(<GPUMonitor />)
    expect(screen.getByText('DGX Spark GB10')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'History' }))
    expect(screen.queryByText('DGX Spark GB10')).not.toBeInTheDocument()
  })
})
