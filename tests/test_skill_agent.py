"""The extraction agent must quote the resume, and the quote is checked.

Item 22 of the v2 upgrade plan. A model can read "built our own columnar store
on top of Arrow" and name a skill no tag list contains — which is both the case
the taxonomy cannot cover and the case where a model is most likely to invent
something. So it is not trusted to report skills; it is required to quote, and
the quote is verified in code.

Every test here stubs the transport. The suite must pass with no API key, and a
test that quietly needs one fails on a fork.
"""
from __future__ import annotations

import asyncio

import pytest
from services import skill_agent
from services.skill_agent import (
    AgentResult,
    ExtractedSkill,
    SkillExtraction,
    check,
    extract_with_agent,
    verify,
)

RESUME = """Ravi Menon — Senior Backend Engineer

Experience
Senior Backend Engineer, Paystack Systems  2021 - Present
- Built and operated Python services on Kubernetes, serving 40M requests a day
- Cut reconciliation lag from 20 minutes to 90 seconds using Redis streams
"""


class TestQuoteVerification:
    def test_a_quote_that_is_in_the_resume_stands(self):
        assert verify("kubernetes", "Built and operated Python services on Kubernetes", RESUME) is None

    def test_a_quote_that_is_not_in_the_resume_is_rejected(self):
        reason = verify("kafka", "Designed event pipelines with Kafka", RESUME)
        assert reason == "quote does not appear in the resume"

    def test_reflowed_whitespace_still_verifies(self):
        """A quote broken across lines is the same quote. Failing on that would
        reject honest answers and teach nothing."""
        assert verify(
            "kubernetes", "Built and operated Python\n   services on Kubernetes", RESUME
        ) is None

    def test_a_paraphrase_is_rejected(self):
        """The check has to fail here or it is not a check. Every extra
        normalisation makes a paraphrase likelier to slip through, which is why
        it stops at whitespace and case."""
        reason = verify("kubernetes", "Operated Python services using Kubernetes", RESUME)
        assert reason == "quote does not appear in the resume"

    def test_a_real_quote_for_an_unrelated_skill_is_rejected(self):
        """Otherwise a model satisfies the check by quoting any true sentence
        for any skill it likes."""
        reason = verify("kafka", "Cut reconciliation lag from 20 minutes to 90 seconds", RESUME)
        assert reason == "quote does not mention the skill"

    def test_an_empty_quote_is_rejected(self):
        assert verify("python", "", RESUME) == "no quote given"

    def test_a_whole_paragraph_is_not_evidence_of_one_skill(self):
        assert "paragraph" in verify("python", RESUME * 2, RESUME)

    def test_a_multi_word_skill_matches_on_its_own_terms(self):
        resume = "- Deployed models with machine learning pipelines in production"
        assert verify(
            "machine learning", "Deployed models with machine learning pipelines", resume
        ) is None


class TestResultAccounting:
    def _extraction(self, *pairs):
        return SkillExtraction(skills=[ExtractedSkill(skill=s, quote=q) for s, q in pairs])

    def test_verified_and_rejected_are_separated(self):
        result = check(
            self._extraction(
                ("kubernetes", "Built and operated Python services on Kubernetes"),
                ("kafka", "Designed event pipelines with Kafka"),
            ),
            RESUME,
        )
        assert set(result.verified) == {"kubernetes"}
        assert [r[0] for r in result.rejected] == ["kafka"]

    def test_the_fabrication_rate_is_the_share_that_failed(self):
        result = check(
            self._extraction(
                ("kubernetes", "Built and operated Python services on Kubernetes"),
                ("kafka", "Designed event pipelines with Kafka"),
                ("terraform", "Managed infrastructure with Terraform"),
            ),
            RESUME,
        )
        assert result.fabrication_rate == pytest.approx(2 / 3, abs=0.01)

    def test_a_clean_extraction_reports_zero(self):
        result = check(
            self._extraction(("redis", "using Redis streams")), RESUME
        )
        assert result.fabrication_rate == 0.0

    def test_an_uncalled_agent_reports_nothing_rather_than_zero(self):
        """Never called is not the same as called and clean; a dashboard
        showing 0% for a model that was never asked would be a lie."""
        result = AgentResult()
        assert result.called is False
        assert result.to_dict()["verified_count"] == 0


class TestTheCallItself:
    async def test_no_api_key_means_no_call(self, monkeypatch):
        monkeypatch.setattr(skill_agent.llm, "is_configured", lambda: False)
        result = await extract_with_agent(RESUME)
        assert result.called is False
        assert result.verified == {}

    async def test_a_timeout_degrades_rather_than_raising(self, monkeypatch):
        async def _hang(*_a, **_k):
            await asyncio.sleep(30)

        monkeypatch.setattr(skill_agent.llm, "is_configured", lambda: True)
        monkeypatch.setattr(skill_agent.llm, "complete_json", _hang)
        monkeypatch.setattr(skill_agent.settings, "llm_deadline_s", 0.2)

        result = await extract_with_agent(RESUME)
        assert result.failed is True
        assert result.verified == {}

    async def test_a_verified_claim_survives_the_round_trip(self, monkeypatch):
        async def _reply(*_a, **_k):
            return SkillExtraction(skills=[
                ExtractedSkill(skill="redis", quote="using Redis streams"),
                ExtractedSkill(skill="rust", quote="Wrote the hot path in Rust"),
            ])

        monkeypatch.setattr(skill_agent.llm, "is_configured", lambda: True)
        monkeypatch.setattr(skill_agent.llm, "complete_json", _reply)

        result = await extract_with_agent(RESUME)
        assert set(result.verified) == {"redis"}
        assert result.rejected[0][0] == "rust"

    async def test_the_resume_is_redacted_before_it_is_sent(self, monkeypatch):
        captured = {}

        async def _capture(prompt, *_a, **_k):
            captured["prompt"] = prompt
            return SkillExtraction(skills=[])

        monkeypatch.setattr(skill_agent.llm, "is_configured", lambda: True)
        monkeypatch.setattr(skill_agent.llm, "complete_json", _capture)

        await extract_with_agent(RESUME + "\nravi.menon@example.com\n")
        assert "ravi.menon@example.com" not in captured["prompt"]


class TestThePromptTellsTheModelTheRules:
    def test_it_says_the_quote_is_checked(self):
        prompt = skill_agent.build_prompt(RESUME)
        assert "word for word" in prompt
        assert "checked in code" in prompt

    def test_it_says_a_fabricated_quote_costs_the_skill(self):
        """A model told the check exists, and what failing costs, has a reason
        to omit rather than invent."""
        assert "costs you the skill" in skill_agent.build_prompt(RESUME)
