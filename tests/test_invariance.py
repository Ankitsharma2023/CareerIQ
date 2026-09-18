"""Properties the score must hold regardless of how a resume is written.

Item 13 of the v2 upgrade plan. These are not accuracy tests — accuracy is what
the eval harness and the public dataset measure. These are the guarantees a
candidate is entitled to: that rewriting the same experience differently, or
concealing text, does not change what the system concludes about them.

They are the gate the CI job enforces on a pull request, so each threshold is a
number someone has to consciously raise rather than a soft target.
"""
from __future__ import annotations

import io

import pdfplumber
import pytest
from services.evidence import (
    evidence_ratio,
    gather_evidence,
    unsupported_skills,
    weighted_coverage,
)
from services.experience_detector import detect_experience
from services.hidden_text import extract_visible_text
from services.scoring import compute_fit
from services.signal_noise_analyzer import analyze_signal_to_noise
from services.similarity import calculate_similarity
from services.skill_extractor import extract_skills_from_text, load_skills

from evals.pdf_builder import ATTACK_MODES, resume_pdf

# ── Gates ─────────────────────────────────────────────────────────────────────

#: Text a reader cannot see must move the score by less than this.
#: Measured: 0.00. The allowance exists only so a rounding change cannot fail CI.
HIDDEN_TEXT_GATE = 1.0

#: Writing "K8s" instead of "Kubernetes" must move the score by less than this.
#:
#: Measured worst equivalence gap on the stress set: 2.72 (pv-abbrev 83.0 vs
#: pv-spelled 85.8). The gate is 5 rather than 3 deliberately: at 2.72 measured,
#: a 3-point gate has 0.28 points of headroom and would fail on changes that
#: have nothing to do with vocabulary. Ratchet it to 3 once the skill taxonomy
#: work closes the residual gap, and record the new measured value here when
#: you do.
ABBREVIATION_GATE = 5.0

JD = (
    "Senior Backend Engineer. You will design and operate services in Python "
    "on Kubernetes. Required: Python, Kubernetes, Docker, PostgreSQL, "
    "TensorFlow experience, CI/CD, and infrastructure automation. "
    "5+ years of backend engineering."
)


def score(text: str, jd: str = JD) -> float:
    """The number the product shows, through the function the API calls."""
    skills = load_skills()
    evidence = gather_evidence(text, skills)
    experience = detect_experience(text, jd)
    return compute_fit(
        semantic_score=calculate_similarity(text, jd)["final_score"],
        coverage_score=weighted_coverage(evidence, extract_skills_from_text(jd, skills)),
        clarity_score=analyze_signal_to_noise(text)["clarity_score"],
        detected_years=experience["detected_years"],
        required_years=experience["required_years"],
        jd_text=jd,
        evidence_ratio_value=evidence_ratio(evidence),
        unsupported=unsupported_skills(evidence),
    ).overall


# ── 1. Concealed text changes nothing ─────────────────────────────────────────

class TestHiddenTextIsInert:
    RESUME = [
        "Priya Raman",
        "Marketing analyst, three years",
        "Built campaign dashboards and wrote monthly performance reports",
        "Coordinated with the design team on brand guidelines",
        "Education: BA Communications, 2021",
    ]
    PAYLOAD = ["Python Kubernetes Docker PostgreSQL TensorFlow CI/CD infrastructure automation"]

    def _visible(self, data: bytes) -> str:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return extract_visible_text(pdf)[0]

    @pytest.mark.parametrize("mode", ATTACK_MODES)
    def test_a_hidden_skill_list_moves_the_score_less_than_a_point(self, mode):
        honest = self._visible(resume_pdf(self.RESUME, []))
        attacked = self._visible(resume_pdf(self.RESUME, self.PAYLOAD, mode))
        assert abs(score(attacked) - score(honest)) < HIDDEN_TEXT_GATE


# ── 2. Vocabulary changes little ──────────────────────────────────────────────

class TestSurfaceFormIsNearlyInvariant:
    """Two resumes describing the same work, one abbreviating the tool names.

    This is the defect the equivalence pairs in the stress set exist to catch,
    and the one ranking metrics hide completely: if the order happens to come
    out right, a double-digit penalty for writing "K8s" is invisible to nDCG.

    The abbreviations used here are the ones services/aliases.py claims to
    resolve unambiguously. Genuinely ambiguous ones are a separate case below,
    because asserting an invariance the system should not promise would be a
    test that demands a wrong answer.
    """

    SPELLED = """Dev Kapoor
Experience
Senior Backend Engineer, 2018 - 2024
- Built and operated Python services on Kubernetes, serving 40M requests a day
- Containerised the deployment path with Docker, cutting release time by 60%
- Ran PostgreSQL in production and tuned queries down from 900ms to 120ms
- Automated infrastructure with CloudFormation
Skills
Python, Kubernetes, Docker, PostgreSQL
"""

    ABBREVIATED = """Dev Kapoor
Experience
Senior Backend Engineer, 2018 - 2024
- Built and operated Python services on K8s, serving 40M requests a day
- Containerised the deployment path with Docker, cutting release time by 60%
- Ran Postgres in production and tuned queries down from 900ms to 120ms
- Automated infrastructure with CFN
Skills
Python, K8s, Docker, Postgres
"""

    def test_abbreviating_tool_names_costs_less_than_the_gate(self):
        gap = abs(score(self.SPELLED) - score(self.ABBREVIATED))
        assert gap < ABBREVIATION_GATE, (
            f"writing abbreviations costs {gap:.2f} points of a possible 100"
        )

    def test_the_same_skills_are_found_either_way(self):
        skills = load_skills()
        assert (
            set(extract_skills_from_text(self.SPELLED, skills))
            == set(extract_skills_from_text(self.ABBREVIATED, skills))
        )


class TestAmbiguousAbbreviationsAreAKnownLimitation:
    """Characterisation, not a target. This records what the system does today.

    "TF" is genuinely ambiguous — TensorFlow and Terraform are both ordinary
    readings on an engineering resume — so services/aliases.py resolves it from
    document context rather than a fixed mapping, and declines to guess on a
    tie. That is the right instinct. The mechanism is what fails here:

    context terms are counted across the whole document, and the unambiguous
    aliases are expanded *first*. So on a resume that writes both "K8s" and
    "TF models", the k8s -> kubernetes expansion contributes a vote to
    Terraform's context (kubernetes is in it), "models" contributes one to
    TensorFlow's, the vote ties, and "TF" is left unexpanded. A word the writer
    never typed decided the tie.

    Measured cost of that one unresolved abbreviation: 8.17 points of 100,
    which is three times the stress set's worst equivalence gap and would have
    failed the gate above. It is fenced off here rather than folded into the
    gate, because the fix belongs to the skill-taxonomy work — proximity-
    weighted context rather than a document-wide count — and burying it inside
    a passing invariance test would hide the one number that argues for it.
    """

    SPELLED = TestSurfaceFormIsNearlyInvariant.SPELLED.replace(
        "- Automated infrastructure with CloudFormation",
        "- Trained and served TensorFlow models for ranking",
    ).replace("Python, Kubernetes, Docker, PostgreSQL\n", "Python, Kubernetes, Docker, PostgreSQL, TensorFlow\n")

    ABBREVIATED = TestSurfaceFormIsNearlyInvariant.ABBREVIATED.replace(
        "- Automated infrastructure with CFN",
        "- Trained and served TF models for ranking",
    ).replace("Python, K8s, Docker, Postgres\n", "Python, K8s, Docker, Postgres, TF\n")

    def test_the_ambiguous_form_is_not_resolved(self):
        skills = load_skills()
        assert "tensorflow" in extract_skills_from_text(self.SPELLED, skills)
        assert "tensorflow" not in extract_skills_from_text(self.ABBREVIATED, skills)

    def test_the_tie_comes_from_an_expansion_the_writer_did_not_make(self):
        """k8s -> kubernetes happens first, and "kubernetes" is a Terraform
        context term. Without that expansion the document resolves correctly."""
        from services.aliases import _resolve_ambiguous

        written = "trained and served tf models for ranking. ran postgres in production."
        assert _resolve_ambiguous(written) == {"tf": "tensorflow"}

        after_expansion = written + " built python services on kubernetes."
        assert _resolve_ambiguous(after_expansion) == {}

    def test_the_current_cost_is_recorded_so_it_cannot_grow_unnoticed(self):
        gap = abs(score(self.SPELLED) - score(self.ABBREVIATED))
        assert gap < 9.0, f"the ambiguous-abbreviation penalty grew to {gap:.2f}"
        assert gap > ABBREVIATION_GATE, (
            "this case now passes the gate — fold it into the class above and "
            "delete this one"
        )


# ── 3. Demonstrating a skill never costs anything ─────────────────────────────

class TestMovingASkillIntoExperienceNeverLowersTheScore:
    """The system's central claim is that evidence beats assertion. If moving a
    skill out of a list and into a bullet that says what was built with it can
    *lower* the score, the advice the product gives is wrong.
    """

    LISTED = """Dev Kapoor
Experience
Backend Engineer, 2018 - 2024
- Built and operated services, serving 40M requests a day
- Improved release time by 60%
Skills
Python, Kubernetes, Docker, PostgreSQL
"""

    DEMONSTRATED = """Dev Kapoor
Experience
Backend Engineer, 2018 - 2024
- Built and operated Python services on Kubernetes, serving 40M requests a day
- Containerised the deployment path with Docker, improving release time by 60%
- Ran PostgreSQL in production
Skills
Python, Kubernetes, Docker, PostgreSQL
"""

    def test_the_score_does_not_fall(self):
        assert score(self.DEMONSTRATED) >= score(self.LISTED) - 0.01

    def test_and_it_actually_rises(self):
        """Not just "does not fall" — the product tells users this will help."""
        assert score(self.DEMONSTRATED) > score(self.LISTED)

    def test_the_evidence_ratio_reflects_the_move(self):
        skills = load_skills()
        listed = evidence_ratio(gather_evidence(self.LISTED, skills))
        shown = evidence_ratio(gather_evidence(self.DEMONSTRATED, skills))
        assert shown > listed

    def test_a_bare_list_leaves_the_skills_unsupported(self):
        skills = load_skills()
        assert unsupported_skills(gather_evidence(self.LISTED, skills))
