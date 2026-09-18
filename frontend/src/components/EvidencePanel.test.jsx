/**
 * The panel that shows the resume line behind a number.
 *
 * Its whole job is to quote the candidate's own words, so the test that matters
 * is that it renders them unchanged.
 */
import React from 'react'
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import EvidencePanel from './EvidencePanel'

const mention = {
  section: 'experience',
  sentence: '- Built and operated Python services on Kubernetes, serving 40M requests a day',
  quantified: true,
}

describe('EvidencePanel', () => {
  it('renders nothing without a subject', () => {
    const { container } = render(<EvidencePanel mentions={[mention]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('quotes the resume line verbatim', () => {
    render(<EvidencePanel title="kubernetes" mentions={[mention]} />)
    expect(screen.getByText(mention.sentence)).toBeInTheDocument()
  })

  it('names the section the line came from', () => {
    render(<EvidencePanel title="kubernetes" mentions={[mention]} />)
    expect(screen.getByText('Experience')).toBeInTheDocument()
  })

  it('marks a quantified mention, because that is what earned the bonus', () => {
    render(<EvidencePanel title="kubernetes" mentions={[mention]} />)
    expect(screen.getByText(/quantified/i)).toBeInTheDocument()
  })

  it('does not mark an unquantified one', () => {
    render(
      <EvidencePanel
        title="redis"
        mentions={[{ ...mention, quantified: false, sentence: 'Python, Redis' }]}
      />,
    )
    expect(screen.queryByText(/quantified/i)).not.toBeInTheDocument()
  })

  it('explains an empty result instead of showing an empty box', () => {
    render(
      <EvidencePanel
        title="postgresql"
        mentions={[]}
        emptyMessage="This skill is claimed but never shown in context."
      />,
    )
    expect(screen.getByText(/never shown in context/i)).toBeInTheDocument()
  })

  it('announces itself to assistive technology', () => {
    render(<EvidencePanel title="kubernetes" mentions={[mention]} />)
    expect(screen.getByRole('region', { name: /evidence for kubernetes/i })).toBeInTheDocument()
  })
})
