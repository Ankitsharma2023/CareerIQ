import React from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

/**
 * Catches a render error in one part of the report instead of losing all of it.
 *
 * Why this exists: the analysis is one expensive request — a file upload, a
 * model load and up to thirty seconds of work — and its entire result arrives
 * in a single event. If one tab throws while rendering, React unmounts the
 * whole tree, and the user loses a completed report to a formatting bug in a
 * section they may not even have opened. They then re-upload and wait again.
 *
 * The tabs were made static imports for a related reason (see
 * ResultsDashboard): a failed dynamic import inside Suspense had no boundary
 * to catch it and rendered blank. Suspense handles a pending import; it does
 * not handle a throw.
 *
 * Deliberately not a global handler at the app root. Wrapping each tab means a
 * failure costs that tab and nothing else, and the rest of the report — the
 * score, the gaps, the evidence — stays on screen and usable.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Console rather than a reporting service: there is no error backend here,
    // and swallowing it silently would make this boundary a way to hide bugs
    // rather than contain them.
    console.error('Render error in', this.props.label || 'a section', error, info)
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <div
        className="ev-card"
        style={{ padding: 24, borderLeft: '4px solid #EF4444' }}
        role="alert"
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <AlertTriangle size={16} color="#EF4444" aria-hidden="true" />
          <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#1A1D2E' }}>
            This section could not be displayed
          </span>
        </div>
        <p style={{ fontSize: '0.8rem', color: '#6B7280', margin: '0 0 14px', lineHeight: 1.6 }}>
          Something went wrong rendering {this.props.label || 'this section'}. The rest of
          your report is unaffected — your analysis has not been lost, and the other tabs
          still work.
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            background: '#5147E5', color: '#fff', fontWeight: 600, fontSize: '0.8rem',
            border: 'none', borderRadius: 8, padding: '7px 14px', cursor: 'pointer',
          }}
        >
          <RefreshCw size={13} aria-hidden="true" /> Try this section again
        </button>
      </div>
    )
  }
}
