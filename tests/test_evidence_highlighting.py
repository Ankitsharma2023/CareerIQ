"""The score has to be able to show its working.

Item 14 of the v2 upgrade plan. The product's central claim is that evidence
beats assertion — that a skill shown in an experience bullet is worth more than
the same word in a comma-separated list. That claim is only checkable by a user
if the system can point at the line it used.
"""
from __future__ import annotations

import pathlib

from services.evidence import MAX_MENTIONS, evidence_payload, gather_evidence
from services.skill_extractor import load_skills

REPO = pathlib.Path(__file__).resolve().parent.parent

RESUME = """Dev Kapoor
Experience
Backend Engineer, 2018 - 2024
- Built and operated Python services on Kubernetes, serving 40M requests a day
- Containerised the deployment path with Docker, cutting release time by 60%
- Wrote the migration tooling in Python
Projects
- A Redis-backed rate limiter used across four services
Skills
Python, Kubernetes, Docker, PostgreSQL, Redis, Terraform
"""


def payload(skills=None):
    return evidence_payload(gather_evidence(RESUME, load_skills()), skills)


class TestTheSentenceIsTheCandidatesOwn:
    def test_a_demonstrated_skill_points_at_the_bullet(self):
        kubernetes = payload(["kubernetes"])["kubernetes"]
        sentences = [m["sentence"] for m in kubernetes["mentions"]]
        assert any("serving 40M requests a day" in s for s in sentences)

    def test_the_sentence_is_verbatim_not_paraphrased(self):
        """A paraphrase would be one more claim to take on trust. The user has
        to be able to find this line in their own resume."""
        for entry in payload().values():
            for mention in entry["mentions"]:
                assert mention["sentence"] in RESUME

    def test_a_quantified_mention_is_marked_and_comes_first(self):
        docker = payload(["docker"])["docker"]
        assert docker["mentions"][0]["quantified"] is True
        assert "60%" in docker["mentions"][0]["sentence"]

    def test_the_section_is_named_so_the_weight_is_explicable(self):
        kubernetes = payload(["kubernetes"])["kubernetes"]
        assert kubernetes["mentions"][0]["section"] == "experience"


class TestAListedSkillLooksDifferentFromADemonstratedOne:
    def test_a_bare_list_entry_is_reported_as_not_demonstrated(self):
        postgres = payload(["postgresql"])["postgresql"]
        assert postgres["demonstrated"] is False
        assert postgres["sections"] == ["skills"]
        assert postgres["weight"] < 0.5

    def test_and_its_only_evidence_is_the_list_itself(self):
        """Which is exactly what the user should see: the system is telling
        them the resume says nothing more than the word."""
        postgres = payload(["postgresql"])["postgresql"]
        assert len(postgres["mentions"]) == 1
        assert postgres["mentions"][0]["section"] == "skills"

    def test_a_demonstrated_skill_outscores_the_listed_one(self):
        result = payload(["kubernetes", "postgresql"])
        assert result["kubernetes"]["weight"] > result["postgresql"]["weight"]

    def test_a_projects_mention_counts_as_demonstrated(self):
        redis = payload(["redis"])["redis"]
        assert redis["demonstrated"] is True
        assert "projects" in redis["sections"]


class TestThePayloadStaysBounded:
    def test_mentions_are_capped(self):
        for entry in payload().values():
            assert len(entry["mentions"]) <= MAX_MENTIONS

    def test_duplicated_sentences_are_not_repeated(self):
        for entry in payload().values():
            sentences = [m["sentence"] for m in entry["mentions"]]
            assert len(sentences) == len(set(sentences))

    def test_it_can_be_restricted_to_the_skills_on_screen(self):
        assert set(payload(["docker", "redis"])) == {"docker", "redis"}

    def test_asking_for_nothing_returns_nothing(self):
        assert payload([]) == {}

    def test_a_very_long_line_is_not_offered_as_a_highlight(self):
        """Highlighting half a page explains nothing."""
        wall = "Skills\n" + ("Python and " * 200) + "Python\n"
        for entry in evidence_payload(gather_evidence(wall, load_skills())).values():
            for mention in entry["mentions"]:
                assert len(mention["sentence"]) <= 400


class TestTheResponseCarriesIt:
    def test_the_schema_exposes_skill_evidence(self):
        from models.schemas import AnalysisResponse

        assert "skill_evidence" in AnalysisResponse.model_fields

    def test_the_ui_reads_it(self):
        tab = (REPO / "frontend/src/components/dashboard/SkillsTab.jsx").read_text(
            encoding="utf-8"
        )
        assert "skill_evidence" in tab
        assert "EvidencePanel" in tab

    def test_a_badge_with_nothing_to_show_is_not_clickable(self):
        """A control that looks interactive and does nothing is worse than a
        plain label."""
        badge = (REPO / "frontend/src/components/SkillBadge.jsx").read_text(encoding="utf-8")
        assert "if (!onSelect) return body" in badge
