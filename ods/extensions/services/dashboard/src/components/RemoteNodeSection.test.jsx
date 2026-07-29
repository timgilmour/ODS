import { screen } from '@testing-library/react'
import { render } from '../test/test-utils'
import { RemoteNodeSection } from './RemoteNodeSection'

const gpu = {
  index: 0, uuid: 'GPU-x', name: 'NVIDIA GB10',
  memory_used_mb: 2048, memory_total_mb: 122880, memory_percent: 1.7,
  utilization_percent: 40, temperature_c: 55, power_w: 90,
  memory_type: 'unified', assigned_services: [],
}

describe('RemoteNodeSection', () => {
  it('renders online node with GPU card and serving line', () => {
    const { container } = render(<RemoteNodeSection node={{
      name: 'sparky', display_name: 'DGX Spark GB10', platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [gpu], serving: { model: 'heretic', endpoint_ok: true }, error: null,
    }} />)
    expect(screen.getByText('DGX Spark GB10')).toBeInTheDocument()
    // GPUCard strips the "NVIDIA " brand prefix from the displayed name by design
    // (see GPUCard.jsx), so the rendered text is "GB10", not "NVIDIA GB10".
    expect(screen.getByText('GB10')).toBeInTheDocument()
    expect(screen.getByText(/serving/i)).toBeInTheDocument()
    expect(screen.getByText(/heretic/)).toBeInTheDocument()
    expect(screen.getByText(/online/i)).toBeInTheDocument()
    expect(container.firstChild.className).not.toMatch(/opacity-60/)
  })

  it('renders offline node greyed with last seen', () => {
    const { container } = render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'offline', last_seen: new Date(Date.now() - 240000).toISOString(),
      gpus: [], serving: null, error: null,
    }} />)
    expect(screen.getByText('sparky')).toBeInTheDocument()
    expect(screen.getByText(/offline/i)).toBeInTheDocument()
    expect(screen.getByText(/last seen/i)).toBeInTheDocument()
    expect(container.firstChild.className).toMatch(/opacity-60/)
  })

  it('renders error badge distinct from offline', () => {
    const { container } = render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'error', last_seen: null, gpus: [], serving: null,
      error: 'node returned HTTP 401',
    }} />)
    expect(screen.getByText(/error/i)).toBeInTheDocument()
    expect(screen.getByText(/HTTP 401/)).toBeInTheDocument()
    // An errored node is actively reporting a problem, so it must not be
    // dimmed to the same "visually dead" level as an offline node.
    expect(container.firstChild.className).not.toMatch(/opacity-60/)
  })

  it('renders an errored node that was previously seen', () => {
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'error', last_seen: new Date(Date.now() - 120000).toISOString(),
      gpus: [], serving: null, error: 'node returned HTTP 500',
    }} />)
    expect(screen.getByText(/error/i)).toBeInTheDocument()
    expect(screen.getByText(/last seen/i)).toBeInTheDocument()
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument()
  })

  it('explains an online node that reports zero GPUs', () => {
    // Design doc's "node up, collector failing" state: without a diagnostic
    // line the card body was simply empty and the state dead-ended.
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [], serving: null,
      error: 'GPU collector unavailable: nvidia-smi not found',
    }} />)
    expect(screen.getByText(/nvidia-smi not found/)).toBeInTheDocument()
    expect(screen.getByText(/no gpus reported/i)).toBeInTheDocument()
  })

  it('explains an online node with zero GPUs and no error message', () => {
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [], serving: null, error: null,
    }} />)
    expect(screen.getByText(/no gpus reported/i)).toBeInTheDocument()
  })

  it('reports an inference endpoint that is down', () => {
    // serving.model is null when the probe fails, so the old
    // `serving?.model &&` gate made "node up, inference endpoint down"
    // render as nothing at all.
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [gpu],
      serving: { model: null, endpoint_ok: false, container_status: 'exited' },
      error: null,
    }} />)
    expect(screen.getByText(/endpoint unreachable/i)).toBeInTheDocument()
  })

  it('reports a served model whose endpoint is unhealthy', () => {
    render(<RemoteNodeSection node={{
      name: 'sparky', display_name: null, platform: 'nvidia',
      status: 'online', last_seen: new Date().toISOString(),
      gpus: [gpu], serving: { model: 'heretic', endpoint_ok: false },
      error: null,
    }} />)
    expect(screen.getByText(/heretic/)).toBeInTheDocument()
    expect(screen.getByText(/endpoint unreachable/i)).toBeInTheDocument()
  })
})
