"""A slow provider must not hold a coaching endpoint past the client's patience.

Measured problem: core.llm gives each upstream request a 60s timeout, retries it
up to three times, and may walk three output-mode rungs — so one slow free-tier
provider could hold a coaching endpoint for minutes. The browser gives up at 45s
and (previously) retried, so the user watched a skeleton loader for 100 seconds
and then got an error card, having spent two LLM calls from a ~20/minute budget
to produce nothing.

These pin the ceiling: whatever happens underneath, the call returns inside
LLM_DEADLINE_S with a readable message rather than hanging.
"""
import asyncio
import time

import pytest
from core.config import settings
from services import ai_coach, llm_evaluator
from services.ai_coach import LLM_DEADLINE_S, UNAVAILABLE, generate_one


@pytest.fixture
def slow_provider(monkeypatch):
    """An upstream that never answers within the deadline."""
    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(LLM_DEADLINE_S * 10)
        raise AssertionError("should have been cancelled by the deadline")

    monkeypatch.setattr(ai_coach.llm, "is_configured", lambda: True)
    monkeypatch.setattr(ai_coach.llm, "complete_json", _hang)
    monkeypatch.setattr(ai_coach.llm, "complete_text", _hang)
    # Keep the test fast — the real 30s ceiling is asserted separately.
    monkeypatch.setattr(ai_coach, "LLM_DEADLINE_S", 0.2)


class TestTheDeadlineIsEnforced:
    @pytest.mark.parametrize("mode", ["bullets", "roadmap"])
    async def test_coaching_modes_give_up_rather_than_hang(self, slow_provider, mode):
        started = time.perf_counter()
        content = await generate_one(
            mode, weak_phrases=["Worked on backend"], missing_skills=["kubernetes"],
            job_description="Backend role", resume_text="Python engineer",
        )
        assert time.perf_counter() - started < 5
        assert content == UNAVAILABLE


class TestTheAnalysisPathIsBoundedToo:
    """/analyze awaits the evaluation and the browser sets no timeout on that
    stream, so an unbounded call there hangs the whole report."""

    async def test_master_evaluation_gives_up_rather_than_hanging(self, monkeypatch):
        async def _hang(*_a, **_k):
            await asyncio.sleep(60)
            raise AssertionError("should have been cancelled by the deadline")

        monkeypatch.setattr(llm_evaluator.llm, "is_configured", lambda: True)
        monkeypatch.setattr(llm_evaluator.llm, "complete_json", _hang)
        monkeypatch.setattr(settings, "llm_deadline_s", 0.2)

        started = time.perf_counter()
        result = await llm_evaluator.llm_master_evaluate("Python engineer", "Backend role", {})
        assert time.perf_counter() - started < 5, "the deadline did not fire"
        assert result is None, "a timed-out evaluation must degrade to None"


class TestTheCeilingFitsTheClient:
    def test_deadline_is_inside_the_browser_timeout(self):
        """The client allows 45s for this call; the server must answer first."""
        assert LLM_DEADLINE_S < 45, "a deadline above the client timeout cannot help"

    def test_deadline_allows_a_healthy_call(self):
        """A well-behaved free-tier response lands in roughly 9s."""
        assert LLM_DEADLINE_S >= 20


class TestFastPathsAreUnaffected:
    async def test_nothing_to_teach_skips_the_model(self, monkeypatch):
        """No missing skills means no roadmap to build, and no request spent."""
        def _boom(*_a, **_k):
            raise AssertionError("must not call the model when there is nothing to teach")
        monkeypatch.setattr(ai_coach.llm, "is_configured", lambda: True)
        monkeypatch.setattr(ai_coach.llm, "complete_text", _boom)

        started = time.perf_counter()
        content = await generate_one("roadmap", missing_skills=[], job_description="Backend role")
        assert time.perf_counter() - started < 1
        assert "already match" in content

    async def test_no_weak_phrases_skips_the_model(self, monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError("must not call the model when there is nothing to rewrite")
        monkeypatch.setattr(ai_coach.llm, "is_configured", lambda: True)
        monkeypatch.setattr(ai_coach.llm, "complete_text", _boom)
        assert "strong action language" in await generate_one("bullets", weak_phrases=[])
