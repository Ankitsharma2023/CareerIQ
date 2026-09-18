"""Skills age. A 2016 mention is weaker evidence than last year's.

Item 20 of the v2 upgrade plan. A candidate who shipped Kubernetes last year
and one who touched it in 2016 do not have the same Kubernetes experience, and
the resume says which is which — the scorer was reading both as "appears in the
Experience section" and weighting them identically.

The tests pin a fixed ``current_year`` throughout. Passing the real one would
make every assertion here quietly change meaning each January.
"""
from __future__ import annotations

import pytest
from services.evidence import gather_evidence
from services.recency import (
    FLOOR_YEARS,
    FULL_WEIGHT_YEARS,
    OLDEST_MULTIPLIER,
    multiplier_for,
    role_blocks,
    skill_recency,
)
from services.skill_extractor import load_skills

NOW = 2026

RESUME = """Dev Kapoor
Experience
Senior Engineer, 2023 - Present
- Built Python services on Kubernetes, serving 40M requests a day
Engineer, 2014 - 2016
- Maintained Java systems with Oracle
Skills
Python, Kubernetes, Java, Oracle, Redis
"""


@pytest.fixture(scope="module")
def skills():
    return load_skills()


class TestTheCurve:
    def test_recent_work_is_not_discounted(self):
        assert multiplier_for(NOW, NOW) == 1.0
        assert multiplier_for(NOW - FULL_WEIGHT_YEARS, NOW) == 1.0

    def test_old_work_reaches_a_floor_and_stops(self):
        """A decay reaching zero would rank someone who ran Postgres for six
        years in 2016 below someone who has never opened it. That is false."""
        assert multiplier_for(NOW - FLOOR_YEARS, NOW) == OLDEST_MULTIPLIER
        assert multiplier_for(NOW - 20, NOW) == OLDEST_MULTIPLIER
        assert OLDEST_MULTIPLIER > 0.5

    def test_it_decays_smoothly_between_the_two(self):
        """No cliff a candidate falls off by leaving a job in December rather
        than January."""
        values = [multiplier_for(NOW - age, NOW) for age in range(0, FLOOR_YEARS + 2)]
        assert values == sorted(values, reverse=True)
        steps = [a - b for a, b in zip(values, values[1:], strict=False)]
        assert max(steps) < 0.1

    def test_an_undated_mention_is_treated_as_current(self):
        """Resumes routinely omit dates on the most recent role. Penalising
        that would be penalising formatting."""
        assert multiplier_for(None, NOW) == 1.0


class TestRoleBlocks:
    def test_the_section_is_cut_at_each_date_range(self):
        experience = "Senior Engineer, 2023 - Present\n- Python\nEngineer, 2014 - 2016\n- Java"
        blocks = role_blocks(experience, NOW)
        assert [year for _, year in blocks] == [NOW, NOW, 2016]

    def test_an_open_ended_range_counts_as_now(self):
        for word in ("Present", "current", "now"):
            blocks = role_blocks(f"Engineer, 2020 - {word}\n- Python", NOW)
            assert blocks[-1][1] == NOW

    def test_a_section_with_no_dates_is_not_aged(self):
        blocks = role_blocks("Engineer\n- Built Python services", NOW)
        assert blocks == [("Engineer\n- Built Python services", NOW)]


class TestSkillRecency:
    def test_each_skill_takes_the_year_of_the_role_it_sits_in(self, skills):
        found = skill_recency(RESUME, skills, NOW)
        assert found["kubernetes"].last_used == NOW
        assert found["java"].last_used == 2016

    def test_a_skill_used_in_two_roles_takes_the_more_recent(self, skills):
        """Averaging would penalise a long career for having had a first job."""
        resume = """Dev
Experience
Senior Engineer, 2024 - Present
- Built Python services
Engineer, 2012 - 2015
- Wrote Python scripts
"""
        assert skill_recency(resume, skills, NOW)["python"].last_used == NOW

    def test_skills_outside_experience_are_not_aged(self, skills):
        """A skills list carries no date to reason from."""
        assert "redis" not in skill_recency(RESUME, skills, NOW)

    def test_a_resume_with_no_experience_section_ages_nothing(self, skills):
        assert skill_recency("Just some prose about Python.", skills, NOW) == {}


class TestItFlowsIntoTheWeight:
    def test_an_old_skill_weighs_less_than_a_current_one(self, skills):
        evidence = gather_evidence(
            RESUME, skills, recency=skill_recency(RESUME, skills, NOW)
        )
        assert evidence["java"].weight < evidence["kubernetes"].weight

    def test_the_weight_records_why(self, skills):
        evidence = gather_evidence(
            RESUME, skills, recency=skill_recency(RESUME, skills, NOW)
        )
        assert evidence["java"].last_used == 2016
        assert evidence["java"].recency_multiplier == OLDEST_MULTIPLIER

    def test_ageing_never_pushes_experience_below_a_bare_list(self, skills):
        """The weight for a skills-list mention is 0.30. An aged experience
        mention must still beat it — the person did the work."""
        evidence = gather_evidence(
            RESUME, skills, recency=skill_recency(RESUME, skills, NOW)
        )
        assert evidence["java"].weight > evidence["redis"].weight

    def test_ageing_applies_per_section_not_to_the_final_weight(self, skills):
        """A skill in an old job and an undated project must keep the project's
        weight, not have it dragged down by the job's age."""
        resume = """Dev
Experience
Engineer, 2012 - 2014
- Maintained Java systems
Projects
- A Java library for parsing configuration files
"""
        evidence = gather_evidence(
            resume, skills, recency=skill_recency(resume, skills, NOW)
        )
        # The projects weight is 0.90 and carries no date, so it stands.
        assert evidence["java"].weight == pytest.approx(0.90, abs=0.01)

    def test_a_resume_with_only_current_roles_is_unchanged(self, skills):
        current = """Dev
Experience
Senior Engineer, 2024 - Present
- Built Python services on Kubernetes
"""
        aged = gather_evidence(current, skills, recency=skill_recency(current, skills, NOW))
        plain = gather_evidence(current, skills, recency={})
        assert {k: v.weight for k, v in aged.items()} == {k: v.weight for k, v in plain.items()}
