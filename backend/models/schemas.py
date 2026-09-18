from typing import Any

from pydantic import BaseModel


class TopMatch(BaseModel):
    sentence: str
    score: float


class SkillGapAnalysis(BaseModel):
    critical:  list[str]
    important: list[str]
    optional:  list[str]


class SignalNoiseResult(BaseModel):
    clarity_score:            float
    weak_phrases_found:       list[str]
    strong_verbs_found:       list[str]
    quantified_sentences:     int
    passive_voice_count:      int = 0
    buzzwords_found:          list[str] = []
    total_sentences_analyzed: int = 0


class KeywordAnalysis(BaseModel):
    present:               list[str]
    missing_critical:      list[str]
    missing_recommended:   list[str]


class ExperienceInfo(BaseModel):
    detected_years:    int
    required_years:    int
    level:             str
    meets_requirement: bool
    gap_years:         int
    # Set when the resume had no recognisable Experience section, so the years
    # were counted across the whole document and may include education dates.
    # Defaults to False so existing callers constructing this model still work.
    low_confidence:    bool = False


class SectionFeedback(BaseModel):
    """Typed rather than Dict[str, str].

    Strict JSON-schema mode cannot express an open-ended map, and naming the
    sections also tells the model exactly which ones to comment on.
    """
    experience: str
    skills:     str
    projects:   str
    summary:    str


class LLMEvaluationPublic(BaseModel):
    """The part of the model's judgement the user is shown.

    Five fields were removed here because each restated a number the
    deterministic pipeline already produces, and the two never agreed:
    ``ats_compatibility`` against ``ATSSimulation.overall_ats_score``,
    ``readability_score`` and ``passive_voice_count`` against
    ``SignalNoiseResult``, ``section_scores`` against the response's own
    ``section_scores``, and ``grammar_issues`` against the weak-phrase pass.
    Two disagreeing numbers on one screen cannot both be right, and the LLM's
    was the one with nothing to validate it against.
    """
    experience_level:        str
    years_detected:          str
    keyword_analysis:        KeywordAnalysis
    cliches_found:           list[str]
    quantified_achievements: int
    section_feedback:        SectionFeedback
    top_improvements:        list[str]
    job_match_reasoning:     str
    interview_questions:     list[str]
    resume_strengths:        list[str]
    fit_verdict:             str


class LLMEvaluation(LLMEvaluationPublic):
    """What the provider is asked to return.

    ``overall_score`` is retained for the evaluation harness only: it is what
    ranks candidate models by Spearman against the human relevance grades
    (see ``evals/llm_score.py``). It is deliberately absent from
    :class:`AnalysisResponse` — the number a user sees comes from
    ``compute_fit`` and nothing else.
    """
    overall_score:           float

    def public(self) -> LLMEvaluationPublic:
        """Drop the eval-only score before the result leaves the process."""
        return LLMEvaluationPublic(**self.model_dump(exclude={"overall_score"}))


class SkillMentionInfo(BaseModel):
    """One sentence in the resume that names a skill, as the candidate wrote it."""
    section:    str
    sentence:   str
    quantified: bool = False


class SkillEvidenceInfo(BaseModel):
    """Why a skill scored what it did.

    The product's central claim is that evidence beats assertion, which is only
    checkable if the evidence is shown. `mentions` are the candidate's own
    sentences, unparaphrased, so the working can be compared against the resume.
    """
    weight:       float
    sections:     list[str] = []
    quantified:   bool = False
    demonstrated: bool = False
    mentions:     list[SkillMentionInfo] = []


class HiddenTextInfo(BaseModel):
    """Text present in the file that a reader cannot see.

    White-on-white, sub-4pt or off-page characters are dropped before anything
    is scored, so none of it reaches the skill extractor, the embeddings or the
    ATS simulation. This is the record of what was removed.
    """
    detected:           bool = False
    # False when the document could not be inspected character by character
    # (the pypdf fallback). No findings then means "not looked at", not "clean".
    inspected:          bool = True
    hidden_char_count:  int = 0
    visible_char_count: int = 0
    reasons:            dict[str, int] = {}
    samples:            list[str] = []
    injection_phrases:  list[str] = []
    hidden_text:        str = ""


class ATSCheckResult(BaseModel):
    score:    float
    note:     str
    evidence: dict[str, Any] | None = None


class ATSSimulation(BaseModel):
    overall_ats_score: float
    checks:            dict[str, Any]
    top_issues:        list[str]
    verdict:           str


class LevelFitInfo(BaseModel):
    detected_years:  int = 0
    required_years:  int = 0
    jd_seniority:    str = "unspecified"
    multiplier:      float = 1.0
    verdict:         str = "appropriate"
    note:            str = ""


class FitBreakdown(BaseModel):
    """The score the user sees, and what produced it."""
    overall:            float
    semantic:           float
    coverage:           float
    clarity:            float
    level:              LevelFitInfo
    evidence_ratio:     float = 1.0
    unsupported_skills: list[str] = []
    notes:              list[str] = []


class AnalysisResponse(BaseModel):
    # Core scores
    semantic_match_score:  float
    ats_keyword_score:     float
    # Skills
    resume_skills:   list[str]
    jd_skills:       list[str]
    matching_skills: list[str]
    missing_skills:  list[str]
    # Top sentence matches
    top_matches:     list[TopMatch]
    # Gap analysis
    skill_gap_analysis: SkillGapAnalysis
    # Writing quality
    signal_noise: SignalNoiseResult
    total_skills_detected: int
    # AI-powered
    # The eval-only overall_score is excluded; see LLMEvaluation.
    llm_evaluation:   LLMEvaluationPublic | None = None
    experience_info:  ExperienceInfo | None = None
    section_scores:   dict[str, float] | None = None
    # New: ATS simulation
    ats_simulation:   ATSSimulation | None = None
    # Unified fit score — the same function the evaluation harness scores
    fit:              FitBreakdown | None = None
    # The sentence behind each skill, for "why did I get this score?".
    skill_evidence:   dict[str, SkillEvidenceInfo] | None = None
    # Text a reader cannot see, removed before scoring.
    hidden_text:      HiddenTextInfo | None = None
    # New: metadata
    contact_info:     dict[str, str] | None = None
    word_count:       int | None = None
    processing_time_s: float | None = None
    job_description:   str | None = ""
    resume_text:       str | None = ""
