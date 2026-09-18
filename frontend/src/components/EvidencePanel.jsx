import React from 'react'
import { Quote, AlertCircle, Hash } from 'lucide-react'

const SECTION_LABELS = {
  experience: 'Experience',
  projects: 'Projects',
  skills: 'Skills list',
  education: 'Education',
  certifications: 'Certifications',
  summary: 'Summary',
  other: 'Elsewhere',
}

/**
 * The working behind a number, in the candidate's own words.
 *
 * Deliberately shows the sentence verbatim rather than a paraphrase: the point
 * is that a user can find the line in their own resume and check it. A
 * summarised version would be one more claim to take on trust, which is the
 * thing the evidence weighting exists to avoid.
 */
export default function EvidencePanel({ title, subtitle, mentions = [], emptyMessage }) {
  if (!title) return null

  return (
    <div
      style={{
        marginTop: 14, padding: '14px 16px', background: '#F8F9FC',
        border: '1px solid #E8EAF0', borderRadius: 12,
      }}
      className="animate-fade-in"
      role="region"
      aria-live="polite"
      aria-label={`Evidence for ${title}`}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Quote size={13} color="#5147E5" aria-hidden="true" />
        <span style={{ fontWeight: 700, fontSize: '0.8rem', color: '#1A1D2E' }}>{title}</span>
        {subtitle && (
          <span style={{ fontSize: '0.7rem', color: '#9CA3AF', marginLeft: 'auto' }}>{subtitle}</span>
        )}
      </div>

      {mentions.length === 0 ? (
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <AlertCircle size={13} color="#D97706" style={{ flexShrink: 0, marginTop: 2 }} aria-hidden="true" />
          <p style={{ fontSize: '0.78rem', color: '#6B7280', margin: 0, lineHeight: 1.55 }}>
            {emptyMessage || 'Nothing in the resume supports this yet.'}
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {mentions.map((mention, index) => (
            <div
              key={index}
              style={{
                padding: '9px 11px', background: '#fff',
                border: `1px solid ${mention.quantified ? '#BBF7D0' : '#F0F1F5'}`,
                borderLeft: `3px solid ${mention.quantified ? '#22C55E' : '#D8DCFC'}`,
                borderRadius: 8,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                <span style={{ fontSize: '0.62rem', fontWeight: 700, color: '#9CA3AF', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {SECTION_LABELS[mention.section] || mention.section}
                </span>
                {mention.quantified && (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '0.62rem', fontWeight: 700, color: '#16A34A' }}>
                    <Hash size={9} aria-hidden="true" /> quantified
                  </span>
                )}
              </div>
              <p style={{ fontSize: '0.78rem', color: '#374151', margin: 0, lineHeight: 1.55 }}>
                {mention.sentence}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
