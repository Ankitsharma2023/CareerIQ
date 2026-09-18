import React from 'react'

export default function ScoreBar({ label, value, color, gradient, height, showPercent = true, className = "", onSelect, selected = false }) {
  const barColor = color || (value >= 75 ? '#22C55E' : value >= 50 ? '#F59E0B' : '#EF4444')
  const background = gradient || barColor

  const body = (
    <div className={`mb-3 ${className}`}>
      <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
        <span style={{ fontSize:'0.75rem', color:'#6B7280', fontWeight:500 }}>{label}</span>
        {showPercent && (
          <span style={{ fontSize:'0.75rem', fontWeight:700, color:'#1A1D2E', fontVariantNumeric:'tabular-nums' }}>
            {Math.round(value)}%
          </span>
        )}
      </div>
      <div className="ev-progress-track" style={height ? { height } : undefined}>
        <div className="ev-progress-fill" style={{ width: `${Math.min(100, Math.max(0, value))}%`, background }} />
      </div>
    </div>
  )

  // A bar with nothing to explain stays inert rather than inviting a click
  // that does nothing.
  if (!onSelect) return body

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={`Show what produced the ${label} score`}
      style={{
        display:'block', width:'100%', textAlign:'left', background:'none',
        border:'none', padding:'2px 4px', margin:'0 -4px', cursor:'pointer',
        borderRadius:8, outline: selected ? '2px solid #5147E5' : 'none',
        outlineOffset:1,
      }}
    >
      {body}
    </button>
  )
}
