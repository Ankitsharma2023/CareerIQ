"""The rewriter must not teach users to lie.

Item 16 of the v2 upgrade plan. The bullet rewriter is the feature most able to
do direct harm: asked to make a resume line stronger, and "stronger" in this
genre means numbers, a model with no numbers to work from will supply them. A
candidate who pastes an invented metric into a resume is carrying it into an
interview where someone will ask about it.

The prompt already says not to. These test the part that checks.
"""
from __future__ import annotations

import pytest
from services import ai_coach
from services.fabrication_guard import describe, find_fabrications, is_faithful
from services.skill_extractor import load_skills

SOURCE = (
    "Worked on the backend API and helped with deployments. "
    "Built Python services, serving 40M requests a day, cutting p95 by 60%."
)


@pytest.fixture(scope="module")
def skills():
    return load_skills()


class TestInventedNumbers:
    def test_a_metric_that_is_not_in_the_resume_is_caught(self, skills):
        found = find_fabrications(SOURCE, "Reduced onboarding time by 35%", skills)
        assert [f.value for f in found if f.kind == "number"] == ["35%"]

    def test_a_number_the_resume_already_states_is_fine(self, skills):
        assert is_faithful(SOURCE, "Cut p95 latency by 60%", skills)

    def test_scale_words_are_normalised(self, skills):
        """"40M" and "40 million" are the same claim; flagging the rephrasing
        would make the guard fire on every faithful rewrite."""
        assert is_faithful(SOURCE, "Served 40 million requests daily", skills)

    def test_thousands_separators_are_normalised(self, skills):
        assert is_faithful(
            "Handled 40000000 requests", "Handled 40,000,000 requests", skills
        )

    def test_a_percentage_is_not_satisfied_by_a_bare_count(self, skills):
        """"60 engineers" must not license "60%" — different claims."""
        found = find_fabrications("Led 60 engineers", "Improved delivery by 60%", skills)
        assert any(f.kind == "number" for f in found)


class TestInventedTools:
    def test_a_technology_the_bullet_did_not_mention_is_caught(self, skills):
        found = find_fabrications(
            "Worked on the backend API", "Built the API on Kubernetes", skills
        )
        assert [f.value for f in found if f.kind == "tool"] == ["kubernetes"]

    def test_a_technology_already_in_the_excerpt_is_fine(self, skills):
        assert is_faithful(SOURCE, "Engineered Python services for the backend API", skills)

    def test_an_invented_tool_is_reported_once_not_twice(self, skills):
        """It is one thing invented, not a tool and a name."""
        found = find_fabrications("Worked on the API", "Built it on Kubernetes", skills)
        assert len(found) == 1


class TestInventedNames:
    def test_an_employer_that_appears_only_in_the_rewrite_is_caught(self, skills):
        found = find_fabrications("Led backend work", "Led backend work at Stripe", skills)
        assert [f.value for f in found if f.kind == "name"] == ["Stripe"]

    def test_ordinary_capitalised_words_are_not_names(self, skills):
        assert is_faithful("worked on the api", "Owned the api. The work shipped.", skills)


class TestItOnlyFlagsAdditions:
    """Dropping detail, rephrasing and reordering cannot make a resume claim
    something untrue, so none of them may be flagged."""

    @pytest.mark.parametrize("rewrite", [
        "Engineered and owned the backend API and its deployment path",
        "Built Python services",
        "Serving 40M requests a day, cut p95 by 60% across Python services",
        "Owned deployments; built Python services",
    ])
    def test_faithful_rewrites_pass(self, skills, rewrite):
        assert is_faithful(SOURCE, rewrite, skills), find_fabrications(SOURCE, rewrite, skills)


class TestTheCorrectionIsUsable:
    def test_it_names_what_was_invented(self, skills):
        found = find_fabrications(
            "Worked on the API", "Cut latency 35% on Kubernetes at Stripe", skills
        )
        message = describe(found)
        assert "35%" in message and "kubernetes" in message and "Stripe" in message


class TestTheRewriterRetriesThenRefuses:
    """The behaviour that matters: a fabricating model is retried once, and if
    it fabricates again the user is told rather than handed a plausible lie."""

    RESUME = "Experience\n- Worked on the backend API and helped with deployments\n"

    async def test_a_clean_first_answer_is_returned_unchanged(self, monkeypatch):
        async def _fake(prompt, *, max_tokens):
            return "• Engineered the backend API and owned its deployment path"

        monkeypatch.setattr(ai_coach, "_safe_text", _fake)
        result = await ai_coach.rewrite_bullets(["Worked on"], self.RESUME, "Backend role")
        assert "Engineered the backend API" in result

    async def test_a_fabricating_answer_is_retried_with_a_correction(self, monkeypatch):
        prompts = []
        replies = iter([
            "• Engineered the backend API, cutting latency 45%",
            "• Engineered the backend API and owned its deployment path",
        ])

        async def _fake(prompt, *, max_tokens):
            prompts.append(prompt)
            return next(replies)

        monkeypatch.setattr(ai_coach, "_safe_text", _fake)
        result = await ai_coach.rewrite_bullets(["Worked on"], self.RESUME, "Backend role")

        assert len(prompts) == 2, "it must retry"
        assert "45%" in prompts[1], "the retry must say what was rejected"
        assert "45%" not in result

    async def test_fabricating_twice_refuses_rather_than_returning_a_lie(self, monkeypatch):
        async def _fake(prompt, *, max_tokens):
            return "• Engineered the backend API, cutting latency 45% at Netflix"

        monkeypatch.setattr(ai_coach, "_safe_text", _fake)
        result = await ai_coach.rewrite_bullets(["Worked on"], self.RESUME, "Backend role")

        assert "45%" not in result
        assert "Netflix" not in result
        assert "without inventing" in result

    async def test_an_unavailable_model_is_not_reported_as_fabrication(self, monkeypatch):
        async def _fake(prompt, *, max_tokens):
            return ai_coach.UNAVAILABLE

        monkeypatch.setattr(ai_coach, "_safe_text", _fake)
        result = await ai_coach.rewrite_bullets(["Worked on"], self.RESUME, "Backend role")
        assert result == ai_coach.UNAVAILABLE
