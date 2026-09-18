"""What a rewrite is actually worth, for this resume.

Item 18 of the v2 upgrade plan. The product tells a candidate that listing a
skill is worth less than demonstrating it. Until now nothing said how much
less, so the advice was a principle rather than a priority — with eleven
unsupported skills, which one is the evening worth spending on?

The projections are produced by re-running the scorer, not by estimating, so
the test that matters most is the one asserting the promised gain is the gain
actually delivered.
"""
from __future__ import annotations

import pytest
from services.evidence import (
    evidence_ratio,
    gather_evidence,
    unsupported_skills,
    weighted_coverage,
)
from services.experience_detector import detect_experience
from services.jd_requirements import classify_jd_skills
from services.scoring import compute_fit
from services.signal_noise_analyzer import analyze_signal_to_noise
from services.similarity import calculate_similarity
from services.skill_extractor import extract_skills_from_text, load_skills
from services.what_if import MIN_WORTHWHILE_GAIN, project_gains

RESUME = """Ana Silva
Experience
Backend Engineer, 2019 - 2024
- Built Python services, serving 20M requests a day
Skills
Python, Kubernetes, Docker, PostgreSQL, Redis, Terraform
"""

JD = """Requirements:
- Python, Kubernetes, Docker, PostgreSQL
- Terraform

Nice to have:
- Redis
"""


@pytest.fixture(scope="module")
def context():
    skills = load_skills()
    evidence = gather_evidence(RESUME, skills)
    jd_skills = extract_skills_from_text(JD, skills)
    priorities = classify_jd_skills(JD, skills)
    experience = detect_experience(RESUME, JD)
    inputs = {
        "semantic_score": calculate_similarity(RESUME, JD)["final_score"],
        "clarity_score": analyze_signal_to_noise(RESUME)["clarity_score"],
        "detected_years": experience["detected_years"],
        "required_years": experience["required_years"],
        "jd_text": JD,
        "unsupported": unsupported_skills(evidence),
    }
    return skills, evidence, jd_skills, priorities, inputs


def _score(evidence, jd_skills, priorities, inputs):
    return compute_fit(
        coverage_score=weighted_coverage(evidence, jd_skills, priorities),
        evidence_ratio_value=evidence_ratio(evidence),
        **inputs,
    ).overall


class TestThePromisedGainIsTheRealGain:
    def test_rewriting_the_resume_delivers_what_was_projected(self, context):
        """The claim the whole feature rests on. A projection produced by a
        different calculation than the score would drift, and the user would
        find out by rewriting their resume and not getting the number."""
        skills, evidence, jd_skills, priorities, inputs = context
        suggestion = project_gains(evidence, jd_skills, inputs, priorities)[0]

        rewritten = RESUME.replace(
            "- Built Python services, serving 20M requests a day",
            f"- Built Python services on {suggestion.skill}, serving 20M requests a day",
        )
        after = gather_evidence(rewritten, skills)

        before_score = _score(evidence, jd_skills, priorities, inputs)
        # Only coverage and the evidence ratio may move: the projection holds
        # semantic similarity and clarity fixed, so the comparison must too.
        after_score = _score(after, jd_skills, priorities, inputs)

        assert after_score - before_score == pytest.approx(suggestion.gain, abs=0.05)

    def test_the_projection_does_not_change_the_current_score(self, context):
        """Running the projection must not mutate the evidence it was given."""
        _, evidence, jd_skills, priorities, inputs = context
        before = _score(evidence, jd_skills, priorities, inputs)
        project_gains(evidence, jd_skills, inputs, priorities)
        assert _score(evidence, jd_skills, priorities, inputs) == before


class TestWhatIsSuggested:
    def test_only_skills_that_are_listed_but_not_shown(self, context):
        _, evidence, jd_skills, priorities, inputs = context
        suggested = {s.skill for s in project_gains(evidence, jd_skills, inputs, priorities)}
        assert "python" not in suggested, "python is already demonstrated"
        assert "kubernetes" in suggested

    def test_a_required_skill_is_flagged_and_a_preferred_one_is_not(self, context):
        _, evidence, jd_skills, priorities, inputs = context
        by_skill = {s.skill: s for s in project_gains(evidence, jd_skills, inputs, priorities)}
        assert by_skill["kubernetes"].required_by_jd is True
        assert by_skill["redis"].required_by_jd is False

    def test_a_preferred_skill_is_worth_less_than_a_required_one(self, context):
        """Follows from item 15's weighting, and is the reason the advice is
        ordered rather than alphabetical."""
        _, evidence, jd_skills, priorities, inputs = context
        by_skill = {s.skill: s for s in project_gains(evidence, jd_skills, inputs, priorities)}
        assert by_skill["redis"].gain < by_skill["kubernetes"].gain

    def test_suggestions_are_ordered_by_what_they_are_worth(self, context):
        _, evidence, jd_skills, priorities, inputs = context
        gains = [s.gain for s in project_gains(evidence, jd_skills, inputs, priorities)]
        assert gains == sorted(gains, reverse=True)

    def test_negligible_gains_are_not_offered(self, context):
        """Asking someone to rewrite a bullet for a tenth of a point teaches
        them the feature is decorative."""
        _, evidence, jd_skills, priorities, inputs = context
        for suggestion in project_gains(evidence, jd_skills, inputs, priorities):
            assert suggestion.gain >= MIN_WORTHWHILE_GAIN

    def test_a_resume_that_demonstrates_everything_gets_no_suggestions(self):
        skills = load_skills()
        thorough = """Ana Silva
Experience
Backend Engineer
- Built Python services on Kubernetes with Docker, serving 20M requests a day
- Ran PostgreSQL and Terraform in production
"""
        evidence = gather_evidence(thorough, skills)
        jd_skills = extract_skills_from_text(JD, skills)
        inputs = {
            "semantic_score": 70.0, "clarity_score": 70.0,
            "detected_years": 3, "required_years": 3, "jd_text": JD,
            "unsupported": unsupported_skills(evidence),
        }
        assert project_gains(evidence, jd_skills, inputs, classify_jd_skills(JD, skills)) == []


class TestTheResponseCarriesIt:
    def test_the_schema_exposes_it(self):
        from models.schemas import AnalysisResponse

        assert "what_if" in AnalysisResponse.model_fields

    def test_the_ui_warns_that_the_experience_must_be_real(self):
        """A tool reporting "+4.1 if you claim Kubernetes" to someone who has
        never used Kubernetes is telling them to lie profitably."""
        import pathlib

        tab = (
            pathlib.Path(__file__).resolve().parent.parent
            / "frontend/src/components/dashboard/SkillsTab.jsx"
        ).read_text(encoding="utf-8")
        assert "where the experience is real" in tab
