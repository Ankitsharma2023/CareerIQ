import React from 'react'
import Badge from './ui/Badge'

const DOT_COLORS = {
  match:    '#22C55E',
  critical: '#EF4444',
  important:'#F59E0B',
  optional: '#5147E5',
  neutral:  '#9CA3AF',
}

export default function SkillBadge({ skill, variant = 'neutral', onSelect, selected = false }) {
  const dot = DOT_COLORS[variant] || DOT_COLORS.neutral
  const body = (
    <Badge variant={variant}>
      <span style={{ width:6, height:6, borderRadius:'50%', background:dot, flexShrink:0 }} />
      {skill}
    </Badge>
  )

  // Without a handler this stays a plain span — a badge that looks clickable
  // and does nothing is worse than one that does not look clickable.
  if (!onSelect) return body

  return (
    <button
      type="button"
      onClick={() => onSelect(skill)}
      aria-pressed={selected}
      aria-label={`Show the resume line behind ${skill}`}
      style={{
        background:'none', border:'none', padding:0, cursor:'pointer',
        borderRadius:99, outline: selected ? '2px solid #5147E5' : 'none',
        outlineOffset:2,
      }}
    >
      {body}
    </button>
  )
}
