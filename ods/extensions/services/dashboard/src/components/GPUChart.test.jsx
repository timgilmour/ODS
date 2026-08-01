import { render } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { GPUChart } from './GPUChart'

const history = {
  timestamps: ['2026-08-01T00:00:00Z', '2026-08-01T00:00:05Z', '2026-08-01T00:00:10Z'],
  gpus: {
    '0': {
      utilization: [3, 3, 3],
      memory_percent: [50, 50, 50],
      temperature: [60, 61, 62],
      power_w: [100, 100, 100],
    },
  },
}

describe('GPUChart', () => {
  it('renders percent sparklines on a fixed 0-100 axis', () => {
    const { container } = render(<GPUChart history={history} gpuIndex={0} />)
    const polylines = container.querySelectorAll('polyline')
    // METRICS order: utilization first. On a fixed 0-100 axis a flat 3% line
    // sits near the bottom (y ≈ 33 of a 36-high viewBox); the auto-scale bug
    // pinned it to the top (y = 2).
    const firstPoint = polylines[0].getAttribute('points').split(' ')[0]
    const y = parseFloat(firstPoint.split(',')[1])
    expect(y).toBeGreaterThan(30)
  })

  it('still auto-scales metrics with no declared max', () => {
    const { container } = render(<GPUChart history={history} gpuIndex={0} />)
    const polylines = container.querySelectorAll('polyline')
    // temperature is METRICS[2]; flat series auto-scales to its own max → top of chart
    const firstPoint = polylines[2].getAttribute('points').split(' ')[0]
    const y = parseFloat(firstPoint.split(',')[1])
    expect(y).toBeLessThan(5)
  })
})
