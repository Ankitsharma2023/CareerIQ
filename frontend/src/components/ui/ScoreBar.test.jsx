/**
 * Score rendering. These are the numbers the whole product exists to show, and
 * the ways they go wrong are all arithmetic on untrusted input: a score that
 * is absent, out of range, or fractional.
 */
import React from 'react'
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ScoreBar from './ScoreBar'

const fill = (container) => container.querySelector('.ev-progress-fill')

describe('ScoreBar', () => {
  it('shows the rounded score', () => {
    render(<ScoreBar label="Semantic Match" value={71.6} />)
    expect(screen.getByText('72%')).toBeInTheDocument()
  })

  it('never draws past the end of the track', () => {
    // A score above 100 must not paint outside the component.
    const { container } = render(<ScoreBar label="Coverage" value={140} />)
    expect(fill(container).style.width).toBe('100%')
  })

  it('never draws a negative width', () => {
    const { container } = render(<ScoreBar label="Coverage" value={-20} />)
    expect(fill(container).style.width).toBe('0%')
  })

  it('renders zero as zero rather than as missing', () => {
    const { container } = render(<ScoreBar label="Coverage" value={0} />)
    expect(screen.getByText('0%')).toBeInTheDocument()
    expect(fill(container).style.width).toBe('0%')
  })

  it('colours by band', () => {
    const { container: good } = render(<ScoreBar label="a" value={80} />)
    const { container: mid } = render(<ScoreBar label="a" value={60} />)
    const { container: bad } = render(<ScoreBar label="a" value={20} />)
    expect(fill(good).style.background).toContain('34, 197, 94')   // green
    expect(fill(mid).style.background).toContain('245, 158, 11')   // amber
    expect(fill(bad).style.background).toContain('239, 68, 68')    // red
  })

  it('is a plain element when there is nothing to explain', () => {
    // A control that looks interactive and does nothing is worse than a label.
    render(<ScoreBar label="Semantic Match" value={50} />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('becomes a button when it can show its working', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(<ScoreBar label="Semantic Match" value={50} onSelect={onSelect} />)

    const button = screen.getByRole('button', { name: /what produced the Semantic Match score/i })
    await user.click(button)
    expect(onSelect).toHaveBeenCalledTimes(1)
  })

  it('reports its open state to assistive technology', () => {
    render(<ScoreBar label="Coverage" value={50} onSelect={() => {}} selected />)
    expect(screen.getByRole('button')).toHaveAttribute('aria-pressed', 'true')
  })
})
