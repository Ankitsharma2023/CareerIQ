"""Required versus preferred, parsed from the posting.

Item 15 of the v2 upgrade plan. Recruiters read these differently and so should
the score: missing a "nice to have" is not the same kind of gap as missing a
stated requirement.

The parse errs toward "required" on purpose. Telling a candidate that a real
requirement is optional is the expensive direction to be wrong in — they skip
the application — so an ambiguous line stays required.
"""
from __future__ import annotations

import pytest
from services.evidence import gather_evidence, weighted_coverage
from services.jd_requirements import (
    PREFERRED,
    PREFERRED_WEIGHT,
    REQUIRED,
    classify_jd_skills,
    split_requirements,
)
from services.skill_extractor import load_skills
from services.skill_gap_analyzer import classify_skill_gaps, missing_must_haves

SECTIONED_JD = """Senior Backend Engineer

Requirements:
- 5+ years of Python
- Kubernetes and Docker in production
- PostgreSQL

Nice to have:
- Go
- Kafka

Benefits:
- Remote friendly
"""


class TestSectionHeadings:
    def test_a_preferred_heading_opens_a_preferred_block(self):
        required, preferred = split_requirements(SECTIONED_JD)
        assert "Go" in preferred
        assert "Kafka" in preferred

    def test_the_requirements_stay_required(self):
        required, _ = split_requirements(SECTIONED_JD)
        assert "Python" in required
        assert "Kubernetes" in required

    def test_a_later_heading_closes_the_preferred_block(self):
        """"Benefits" is not a continuation of "Nice to have"."""
        _, preferred = split_requirements(SECTIONED_JD)
        assert "Remote friendly" not in preferred

    @pytest.mark.parametrize("heading", [
        "Nice to have:", "Nice-to-haves", "Preferred qualifications",
        "Bonus points", "Good to have", "Desirable",
    ])
    def test_the_usual_spellings_are_recognised(self, heading):
        jd = f"Requirements:\n- Python\n\n{heading}\n- Kubernetes\n"
        required, preferred = split_requirements(jd)
        assert "Kubernetes" in preferred
        assert "Python" in required


class TestInlineMarkers:
    def test_a_marker_binds_to_its_own_sentence_not_the_paragraph(self):
        """The defect this closes. Many postings are one unbroken paragraph, and
        applying the marker to the whole line filed genuine requirements as
        optional on the strength of a sentence about something else."""
        jd = (
            "We need strong experience in docker deployment. "
            "Docker containers are essential. "
            "Kubernetes experience is preferred."
        )
        required, preferred = split_requirements(jd)
        assert "docker deployment" in required
        assert "Kubernetes experience is preferred." in preferred
        assert "docker" not in preferred.lower()

    def test_a_marker_does_not_leak_into_the_following_lines(self):
        jd = "- Go is a plus\n- PostgreSQL in production\n"
        required, preferred = split_requirements(jd)
        assert "Go is a plus" in preferred
        assert "PostgreSQL" in required


class TestClassification:
    def test_skills_are_split_by_where_they_appear(self):
        priorities = classify_jd_skills(SECTIONED_JD, load_skills())
        assert priorities["python"] == REQUIRED
        assert priorities["kubernetes"] == REQUIRED
        assert priorities["go"] == PREFERRED
        assert priorities["kafka"] == PREFERRED

    def test_a_skill_in_both_halves_is_required(self):
        """The stronger claim wins: under-stating a requirement is the error
        that costs a candidate an application."""
        jd = "Requirements:\n- Kubernetes\n\nNice to have:\n- Kubernetes operators\n"
        assert classify_jd_skills(jd, load_skills())["kubernetes"] == REQUIRED

    def test_a_posting_with_no_preferences_marks_nothing_preferred(self):
        jd = "Requirements:\n- Python\n- Docker\n- PostgreSQL\n"
        priorities = classify_jd_skills(jd, load_skills())
        assert set(priorities.values()) == {REQUIRED}


class TestCoverageWeighting:
    RESUME = """Ana Silva
Experience
Backend Engineer, 2019 - 2024
- Built Python services, serving 20M requests a day
- Ran PostgreSQL in production
Skills
Python, PostgreSQL
"""

    def _coverage(self, jd):
        skills = load_skills()
        from services.skill_extractor import extract_skills_from_text

        evidence = gather_evidence(self.RESUME, skills)
        jd_skills = extract_skills_from_text(jd, skills)
        priorities = classify_jd_skills(jd, skills)
        return (
            weighted_coverage(evidence, jd_skills),
            weighted_coverage(evidence, jd_skills, priorities),
        )

    def test_missing_a_preference_costs_less_than_missing_a_requirement(self):
        preferred_gap = "Requirements:\n- Python\n- PostgreSQL\n\nNice to have:\n- Kafka\n"
        required_gap = "Requirements:\n- Python\n- PostgreSQL\n- Kafka\n"
        assert self._coverage(preferred_gap)[1] > self._coverage(required_gap)[1]

    def test_with_no_preferences_the_score_is_unchanged(self):
        """The change has to be inert on a posting that does not distinguish,
        or it is not a refinement, it is a different scorer."""
        plain = "Requirements:\n- Python\n- PostgreSQL\n- Kafka\n"
        before, after = self._coverage(plain)
        assert before == after

    def test_a_preference_still_counts_for_something(self):
        """Scoring it at zero would rank a candidate who has the preferred
        skill identically to one who does not."""
        assert 0 < PREFERRED_WEIGHT < 1


class TestTheGapsAreGradedByThePosting:
    def test_a_stated_preference_lands_in_optional(self):
        priorities = {"kafka": PREFERRED, "python": REQUIRED}
        result = classify_skill_gaps(
            ["kafka"], [], "Requirements: Python\nNice to have: Kafka", priorities
        )
        assert result["optional"] == ["kafka"]

    def test_a_repeated_requirement_is_critical(self):
        jd = "Requirements:\n- Docker deployment\n- Docker Compose for local work\n"
        result = classify_skill_gaps(["docker"], [], jd, {"docker": REQUIRED})
        assert result["critical"] == ["docker"]

    def test_a_skill_the_posting_never_names_is_optional(self):
        result = classify_skill_gaps(["cobol"], [], "We need Python.", {})
        assert result["optional"] == ["cobol"]

    def test_must_haves_exclude_preferences(self):
        priorities = {"kafka": PREFERRED, "kubernetes": REQUIRED, "go": PREFERRED}
        assert missing_must_haves(["kafka", "kubernetes", "go"], priorities) == ["kubernetes"]

    def test_must_haves_are_empty_when_nothing_required_is_missing(self):
        assert missing_must_haves(["go"], {"go": PREFERRED}) == []
