"""Coaching inputs are bounded, and redaction stays off the event loop.

Item 7 of the v2 upgrade plan. /ai-coach/generate accepted a resume_text and a
job_description of any length, and rewrite_bullets ran five regex passes over
the whole resume inline in an async handler. That is the event-loop starvation
already fixed in /analyze, in the same process: while those passes ran, nothing
else in the service was served — including the platform's health check.
"""
import asyncio
import time

import httpx
import pytest
from core.config import settings
from httpx import ASGITransport
from main import app, limiter
from services import ai_coach

BLOCK_SECONDS = 1.0


def _reset_limiter():
    try:
        limiter._storage.reset()
    except Exception:
        try:
            limiter.reset()
        except Exception:
            pass


async def _post(body: dict) -> httpx.Response:
    _reset_limiter()
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post("/api/v1/ai-coach/generate", json=body, timeout=30.0)


class TestTheRequestIsBounded:
    async def test_an_oversized_resume_is_refused_at_the_edge(self):
        response = await _post({
            "mode": "bullets",
            "resume_text": "x" * (settings.max_resume_text_length + 1),
        })
        assert response.status_code == 422

    async def test_an_oversized_job_description_is_refused(self):
        response = await _post({
            "mode": "bullets",
            "job_description": "x" * (settings.max_jd_length + 1),
        })
        assert response.status_code == 422

    async def test_a_realistic_resume_still_fits(self, monkeypatch):
        """Two pages is ~4,500 characters; the limit must not be a product limit."""
        monkeypatch.setattr(ai_coach.llm, "is_configured", lambda: False)
        response = await _post({
            "mode": "bullets",
            "weak_phrases": [],
            "resume_text": "Engineered services. " * 500,
        })
        assert response.status_code == 200

    @pytest.mark.parametrize("field", ["weak_phrases", "missing_skills"])
    async def test_absurd_list_inputs_are_refused(self, field):
        response = await _post({"mode": "bullets", field: ["x"] * 500})
        assert response.status_code == 422


class TestRedactionDoesNotBlockTheLoop:
    async def test_the_loop_stays_responsive_while_redacting(self, monkeypatch):
        """A heartbeat ticks every 10ms; the largest gap between ticks is the
        longest the loop was unavailable. Against an inline redaction call this
        gap is the full duration of the stubbed pass."""
        def _slow_redact(text, contact_info=None):
            # Releases the GIL, like re over a large string does not — so this
            # blocks the loop only if it runs on the loop's own thread, which
            # is exactly the distinction under test.
            time.sleep(BLOCK_SECONDS)
            return text

        monkeypatch.setattr(ai_coach, "redact_for_prompt", _slow_redact)

        async def _fake_text(prompt, *, max_tokens):
            return "• A rewritten bullet"

        monkeypatch.setattr(ai_coach, "_safe_text", _fake_text)

        ticks: list[float] = []
        stop = asyncio.Event()

        async def _heartbeat():
            while not stop.is_set():
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(_heartbeat())
        await ai_coach.rewrite_bullets(["Worked on backend"], "resume text", "a job")
        stop.set()
        await beat

        assert len(ticks) >= 2, (
            "the heartbeat never ran — redaction held the loop for its whole duration"
        )
        worst_gap = max(b - a for a, b in zip(ticks, ticks[1:], strict=False))
        assert worst_gap < BLOCK_SECONDS / 2, (
            f"the loop was unavailable for {worst_gap:.2f}s — redaction is running on it"
        )

    async def test_the_text_is_redacted_before_it_is_truncated(self, monkeypatch):
        """Truncating first can cut an email in half and send the remainder."""
        captured = {}

        async def _capture(prompt, *, max_tokens):
            captured["prompt"] = prompt
            return ""

        monkeypatch.setattr(ai_coach, "_safe_text", _capture)

        resume = "A" * 880 + " contact me at ada.lovelace@example.com for details"
        await ai_coach.rewrite_bullets(["Worked on backend"], resume, "a job")

        assert "ada.lovelace@example.com" not in captured["prompt"]
        assert "example.com" not in captured["prompt"]
