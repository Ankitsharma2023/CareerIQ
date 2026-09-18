/**
 * API contract types.
 *
 * Mirrors the Pydantic models in backend/models/schemas.py. Regenerate the
 * machine version any time the backend changes:
 *
 *   1. python -c "import sys;sys.path.insert(0,'backend');from main import app;
 *      import json;json.dump(app.openapi(), open('frontend/openapi.json','w'))"
 *   2. npm run gen:types
 *
 * This hand-written file is the one components import, so the shape stays
 * readable; openapi.d.ts is the check that it has not drifted.
 */

export interface TopMatch {
  sentence: string
  score: number
}

export interface SkillGapAnalysis {
  critical: string[]
  important: string[]
  optional: string[]
}

export interface SignalNoiseResult {
  clarity_score: number
  weak_phrases_found: string[]
  strong_verbs_found: string[]
  quantified_sentences: number
  passive_voice_count: number
  buzzwords_found: string[]
  total_sentences_analyzed: number
}

export interface SectionFeedback {
  experience: string
  skills: string
  projects: string
  summary: string
}

export interface KeywordAnalysis {
  present: string[]
  missing_critical: string[]
  missing_recommended: string[]
}

export interface ExperienceInfo {
  detected_years: number
  required_years: number
  level: string
  meets_requirement: boolean
  gap_years: number
}

/**
 * The part of the model's judgement the user is shown.
 *
 * `overall_score` is not here on purpose: it is eval-only, excluded from the
 * API response, and the single score a user sees comes from `fit.overall`.
 * `ats_compatibility`, `readability_score`, `passive_voice_count`,
 * `section_scores` and `grammar_issues` were removed for the same reason —
 * each restated a number the deterministic pipeline already reports, and the
 * two never agreed. `salary_insight` and `competition_level` went with them:
 * nothing in the system knows either, so both were invented.
 */
export interface LLMEvaluation {
  experience_level: string
  years_detected: string
  keyword_analysis: KeywordAnalysis
  cliches_found: string[]
  quantified_achievements: number
  section_feedback: SectionFeedback
  top_improvements: string[]
  job_match_reasoning: string
  interview_questions: string[]
  resume_strengths: string[]
  fit_verdict: string
}

/** Points gained by moving a listed skill into an experience bullet.
 *  Computed by re-running the scoring function, so the gain is exact. */
export interface WhatIfSuggestion {
  skill: string
  current_weight: number
  projected_weight: number
  gain: number
  required_by_jd: boolean
}

/** One sentence in the resume that names a skill, as the candidate wrote it. */
export interface SkillMention {
  section: string
  sentence: string
  quantified: boolean
}

/** Why a skill scored what it did. `mentions` are verbatim so a user can find
 *  the line in their own resume and check it. */
export interface SkillEvidence {
  weight: number
  sections: string[]
  quantified: boolean
  demonstrated: boolean
  mentions: SkillMention[]
}

/** Text present in the file that a reader cannot see — white-on-white,
 *  sub-4pt or off-page. Removed before anything is scored. `inspected: false`
 *  means the document could not be checked, not that it is clean. */
export interface HiddenTextInfo {
  detected: boolean
  inspected: boolean
  hidden_char_count: number
  visible_char_count: number
  reasons: Record<string, number>
  samples: string[]
  injection_phrases: string[]
  hidden_text: string
}

export interface ATSCheck {
  score: number
  note: string
  evidence?: { examples?: string[] } & Record<string, unknown>
  [key: string]: unknown
}

export interface ATSSimulation {
  overall_ats_score: number
  checks: Record<string, ATSCheck>
  top_issues: string[]
  verdict: 'ATS-Ready' | 'Needs Improvement' | 'High ATS Risk' | string
}

export type LevelVerdict = 'appropriate' | 'overqualified' | 'underqualified'

export interface LevelFitInfo {
  detected_years: number
  required_years: number
  jd_seniority: 'junior' | 'senior' | 'unspecified' | string
  multiplier: number
  verdict: LevelVerdict
  note: string
}

/** The score the user sees, and what produced it. Computed server-side in
 *  services/scoring.py so the evaluation harness measures the same number. */
export interface FitBreakdown {
  overall: number
  semantic: number
  coverage: number
  clarity: number
  level: LevelFitInfo
  evidence_ratio: number
  unsupported_skills: string[]
  notes: string[]
}

export interface AnalysisResponse {
  semantic_match_score: number
  ats_keyword_score: number
  resume_skills: string[]
  jd_skills: string[]
  matching_skills: string[]
  missing_skills: string[]
  top_matches: TopMatch[]
  skill_gap_analysis: SkillGapAnalysis
  /** Gaps the posting states as requirements — what a recruiter screens on. */
  missing_must_haves?: string[]
  /** Each JD skill as 'required' or 'preferred'. */
  jd_skill_priorities?: Record<string, 'required' | 'preferred' | string>
  signal_noise: SignalNoiseResult
  total_skills_detected: number
  llm_evaluation?: LLMEvaluation | null
  experience_info?: ExperienceInfo | null
  /** Deterministic, from services/section_parser.py — the only section scores. */
  section_scores?: Record<string, number> | null
  ats_simulation?: ATSSimulation | null
  hidden_text?: HiddenTextInfo | null
  skill_evidence?: Record<string, SkillEvidence> | null
  what_if?: WhatIfSuggestion[]
  /** Optional because history entries saved before Phase 5 predate it. */
  fit?: FitBreakdown | null
  contact_info?: Record<string, string> | null
  word_count?: number | null
  processing_time_s?: number | null
  job_description?: string
  resume_text?: string
}

/* ── Streaming ─────────────────────────────────────────────────────────── */

export interface ProgressEvent {
  event: 'progress'
  progress: number
  message: string
}

export interface CompleteEvent {
  event: 'complete'
  progress: 100
  result: AnalysisResponse
}

export interface ErrorEvent {
  event: 'error'
  message: string
}

export type AnalysisStreamEvent = ProgressEvent | CompleteEvent | ErrorEvent

/* ── Coaching ──────────────────────────────────────────────────────────── */

export type CoachingMode = 'bullets' | 'roadmap'

export interface CoachingPayload {
  weak_phrases: string[]
  missing_skills: string[]
  job_description: string
  resume_text: string
}

export interface CoachingResponse {
  mode: CoachingMode
  label: string
  content: string
}


/** History entries are persisted, so they may predate any field above. */
export interface HistoryEntry {
  id: string
  createdAt: number
  jobTitle: string
  score: number
  response: AnalysisResponse
}
