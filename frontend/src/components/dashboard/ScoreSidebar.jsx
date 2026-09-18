import React from 'react'
import { BookOpen } from 'lucide-react'
import AccordionSection from './AccordionSection'
import AnalysisHistory from '../AnalysisHistory'

const scoreColor = v => v >= 75 ? '#22C55E' : v >= 50 ? '#F59E0B' : '#EF4444'

export default function ScoreSidebar({ data, activeTab, onTabChange, onSelectAnalysis }) {
  const { semantic_match_score, ats_keyword_score, signal_noise, skill_gap_analysis, matching_skills, missing_skills } = data
  const fit = data.fit

  const clarity    = signal_noise?.clarity_score ?? 0
  // Server-computed (services/scoring.py); the blend is a fallback for
  // history entries saved before the fit field existed.
  const overall    = Math.round(
    data.fit?.overall ?? semantic_match_score * 0.4 + ats_keyword_score * 0.35 + clarity * 0.25,
  )
  const totalGaps  = (skill_gap_analysis?.critical?.length ?? 0) + (skill_gap_analysis?.important?.length ?? 0) + (skill_gap_analysis?.optional?.length ?? 0)

  const contentPct = Math.round(
    ((signal_noise?.quantified_sentences > 2 ? 25 : 10) +
     (signal_noise?.weak_phrases_found?.length === 0 ? 25 : 5) +
     (signal_noise?.strong_verbs_found?.length > 3 ? 25 : 10) +
     ((signal_noise?.passive_voice_count ?? 0) === 0 ? 25 : 10))
  )
  const skillsPct  = matching_skills.length + missing_skills.length > 0
    ? Math.round((matching_skills.length / (matching_skills.length + missing_skills.length)) * 100)
    : 0
  const atsPct     = Math.round(ats_keyword_score)

  // Section scores come from services/section_parser.py. The LLM used to
  // report its own set beside these; they disagreed, and only one of the two
  // could be checked.
  const sections   = data.section_scores ?? null
  const sectionRow = (label, key, threshold) => ({
    label,
    status: (sections?.[key] ?? 0) >= threshold ? 'ok' : 'warn',
    note:   sections ? `${Math.round(sections[key] ?? 0)}%` : 'N/A',
  })
  const sectionItems = [
    sectionRow('Experience', 'experience', 70),
    sectionRow('Skills',     'skills',     70),
    sectionRow('Education',  'education',  60),
    sectionRow('Projects',   'projects',   60),
    sectionRow('Summary',    'summary',    60),
  ]

  return (
    <aside className="score-sidebar">
      {/* Score number */}
      <div style={{ padding:'20px 20px 16px', textAlign:'center' }}>
        <p style={{ fontSize:'0.68rem', fontWeight:700, color:'#9CA3AF', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:4 }}>Your Score</p>
        <div>
          <span style={{ fontSize:'2.8rem', fontWeight:800, color:scoreColor(overall), fontVariantNumeric:'tabular-nums' }}>{overall}</span>
          <span style={{ fontSize:'1.2rem', color:'#9CA3AF', fontWeight:600 }}>/100</span>
        </div>
        <p style={{ fontSize:'0.72rem', color:'#9CA3AF', marginTop:2 }}>{totalGaps} gaps to close</p>
      </div>

      <div className="ev-divider" />

      {/* Accordion sections */}
      <AccordionSection label="Content" pct={contentPct} defaultOpen items={[
        { label:'ATS Parse Rate',    status:'ok',   note:'No issues' },
        { label:'Quantified Impact', status: signal_noise?.quantified_sentences > 2 ? 'ok' : 'warn', note:`${signal_noise?.quantified_sentences ?? 0} found` },
        { label:'Weak Phrases',      status: (signal_noise?.weak_phrases_found?.length ?? 0) === 0 ? 'ok' : 'issue', note: (signal_noise?.weak_phrases_found?.length ?? 0) === 0 ? 'None found' : `${signal_noise.weak_phrases_found.length} found` },
        { label:'Strong Verbs',      status: (signal_noise?.strong_verbs_found?.length ?? 0) > 3 ? 'ok' : 'warn', note:`${signal_noise?.strong_verbs_found?.length ?? 0} found` },
        { label:'Passive Voice',     status: (signal_noise?.passive_voice_count ?? 0) === 0 ? 'ok' : 'warn', note: (signal_noise?.passive_voice_count ?? 0) === 0 ? 'None found' : `${signal_noise.passive_voice_count} sentences` },
      ]} />
      <div className="ev-divider" />

      <AccordionSection label="Sections" pct={sections ? Math.round((sectionItems.filter(s=>s.status==='ok').length/sectionItems.length)*100) : 0} items={sectionItems} />
      <div className="ev-divider" />

      <AccordionSection label="ATS Essentials" pct={atsPct} items={[
        { label:'Keyword Coverage', status: ats_keyword_score >= 70 ? 'ok' : ats_keyword_score >= 40 ? 'warn' : 'issue', note:`${Math.round(ats_keyword_score)}%` },
        { label:'Semantic Match',   status: semantic_match_score >= 70 ? 'ok' : 'warn', note:`${Math.round(semantic_match_score)}%` },
        { label:'Contact Info',     status:'ok', note:'Detected' },
        { label:'ATS Compatibility',status: (data.ats_simulation?.overall_ats_score ?? 0) >= 70 ? 'ok' : 'warn', note: data.ats_simulation ? `${Math.round(data.ats_simulation.overall_ats_score)}%` : 'N/A' },
      ]} />
      <div className="ev-divider" />

      <AccordionSection label="Skill Gaps" pct={skillsPct} items={[
        { label:'Matching Skills',  status:'ok',   note:`${matching_skills.length} matched` },
        { label:'Critical Gaps',    status: (skill_gap_analysis?.critical?.length ?? 0) === 0 ? 'ok' : 'issue', note:`${skill_gap_analysis?.critical?.length ?? 0} gaps` },
        { label:'Important Gaps',   status: (skill_gap_analysis?.important?.length ?? 0) === 0 ? 'ok' : 'warn', note:`${skill_gap_analysis?.important?.length ?? 0} gaps` },
        { label:'Optional Gaps',    status:'warn', note:`${skill_gap_analysis?.optional?.length ?? 0} gaps` },
      ]} />
      <div className="ev-divider" />

      {/* Level fit — surfaced because it can move the score substantially */}
      {fit?.level?.note && (
        <>
          <div style={{ padding:'12px 20px' }}>
            <div style={{ background: fit.level.verdict === 'overqualified' ? '#FFFBEB' : '#EFF6FF',
                          border: `1px solid ${fit.level.verdict === 'overqualified' ? '#FDE68A' : '#BFDBFE'}`,
                          borderRadius:10, padding:'10px 12px' }}>
              <div style={{ fontSize:'0.68rem', fontWeight:700, textTransform:'uppercase',
                            letterSpacing:'0.06em', marginBottom:4,
                            color: fit.level.verdict === 'overqualified' ? '#B45309' : '#1D4ED8' }}>
                {fit.level.verdict === 'overqualified' ? 'Above this level' : 'Below this level'}
              </div>
              <p style={{ fontSize:'0.7rem', color:'#6B7280', margin:0, lineHeight:1.5 }}>
                {fit.level.note}
              </p>
            </div>
          </div>
          <div className="ev-divider" />
        </>
      )}

      {/* Unsupported skills — the lines a recruiter will probe */}
      {fit?.unsupported_skills?.length > 6 && (
        <>
          <div style={{ padding:'12px 20px' }}>
            <div style={{ background:'#FEF2F2', border:'1px solid #FECACA', borderRadius:10, padding:'10px 12px' }}>
              <div style={{ fontSize:'0.68rem', fontWeight:700, color:'#B91C1C', textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:4 }}>
                Listed but not shown
              </div>
              <p style={{ fontSize:'0.7rem', color:'#6B7280', margin:0, lineHeight:1.5 }}>
                {fit.unsupported_skills.length} skills appear only in a list. Move the
                important ones into an experience bullet with what you built.
              </p>
            </div>
          </div>
          <div className="ev-divider" />
        </>
      )}

      {/* Roadmap teaser */}
      <div style={{ padding:'14px 20px' }}>
        <div style={{ background:'#EEF0FE', border:'1px solid #D8DCFC', borderRadius:10, padding:'12px 14px', cursor:'pointer' }}
          onClick={() => onTabChange('ai_coach')}>
          <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:4 }}>
            <BookOpen size={14} color="#5147E5" />
            <span style={{ fontSize:'0.72rem', fontWeight:700, color:'#3D34C4' }}>30-Day Skill Roadmap</span>
          </div>
          <p style={{ fontSize:'0.68rem', color:'#6B7280' }}>A four-week plan for your {totalGaps} gaps</p>
        </div>
        <AnalysisHistory onSelectAnalysis={onSelectAnalysis} />
      </div>
    </aside>
  )
}
