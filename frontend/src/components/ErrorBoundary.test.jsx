/**
 * The boundary exists so one broken section cannot cost a completed report.
 *
 * The analysis is a file upload, a model load and up to thirty seconds of work,
 * and its whole result arrives in one event. Losing all of it to a formatting
 * bug in a tab the user may never open is the failure this prevents.
 */
import React from 'react'
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ErrorBoundary from './ErrorBoundary'

function Boom({ shouldThrow = true }) {
  if (shouldThrow) throw new Error('render exploded')
  return <p>section content</p>
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // React logs caught render errors itself; silencing keeps the output
    // readable without hiding the assertions.
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })
  afterEach(() => vi.restoreAllMocks())

  it('renders its children when nothing throws', () => {
    render(
      <ErrorBoundary label="Score & Fit">
        <Boom shouldThrow={false} />
      </ErrorBoundary>,
    )
    expect(screen.getByText('section content')).toBeInTheDocument()
  })

  it('shows a message instead of unmounting when a child throws', () => {
    render(
      <ErrorBoundary label="Score & Fit">
        <Boom />
      </ErrorBoundary>,
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText(/could not be displayed/i)).toBeInTheDocument()
  })

  it('names the section that failed', () => {
    render(
      <ErrorBoundary label="Gaps & ATS">
        <Boom />
      </ErrorBoundary>,
    )
    expect(screen.getByText(/Gaps & ATS/)).toBeInTheDocument()
  })

  it('tells the user the rest of the report survived', () => {
    render(
      <ErrorBoundary label="AI Coach">
        <Boom />
      </ErrorBoundary>,
    )
    expect(screen.getByText(/has not been lost/i)).toBeInTheDocument()
  })

  it('still logs the error rather than swallowing it', () => {
    render(
      <ErrorBoundary label="AI Coach">
        <Boom />
      </ErrorBoundary>,
    )
    const logged = console.error.mock.calls.flat().join(' ')
    expect(logged).toContain('Render error in')
  })

  it('lets the user retry the section', async () => {
    const user = userEvent.setup()
    let shouldThrow = true
    function Flaky() {
      if (shouldThrow) throw new Error('once')
      return <p>recovered</p>
    }

    render(
      <ErrorBoundary label="Score & Fit">
        <Flaky />
      </ErrorBoundary>,
    )
    shouldThrow = false
    await user.click(screen.getByRole('button', { name: /try this section again/i }))
    expect(screen.getByText('recovered')).toBeInTheDocument()
  })

  it('isolates the failure to its own boundary', () => {
    render(
      <div>
        <ErrorBoundary label="broken tab">
          <Boom />
        </ErrorBoundary>
        <ErrorBoundary label="working tab">
          <Boom shouldThrow={false} />
        </ErrorBoundary>
      </div>,
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('section content')).toBeInTheDocument()
  })
})
