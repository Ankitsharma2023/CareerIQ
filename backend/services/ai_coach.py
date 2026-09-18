"""Generative coaching features.

All calls go through :mod:`core.llm`, so they are async (no thread-per-socket),
share one client, and inherit model fallback.
"""
from __future__ import annotations

import asyncio
import logging

from core import llm
from core.config import settings
from core.redact import redact_for_prompt

logger = logging.getLogger(__name__)

UNAVAILABLE = "Could not generate this right now. Check the API key in backend/.env and try again."

# Ceiling on one coaching call. Defined in core.config so the analysis path and
# the coaching path cannot drift apart; see the comment there for why an
# unbounded LLM call is the defect this closes.
LLM_DEADLINE_S = settings.llm_deadline_s


async def _safe_text(prompt: str, *, max_tokens: int) -> str:
    """Run a text completion, degrading to a readable message on failure.

    Never surfaces the raw exception: the previous implementation returned
    "[Could not generate — {e}]" straight into the UI, leaking provider
    internals to the user.
    """
    try:
        return await asyncio.wait_for(
            llm.complete_text(prompt, max_tokens=max_tokens), timeout=LLM_DEADLINE_S
        )
    except asyncio.TimeoutError:
        logger.error("Coaching generation exceeded %.0fs — giving up", LLM_DEADLINE_S)
        return UNAVAILABLE
    except llm.LLMError as exc:
        logger.error("Coaching generation failed: %s", exc)
        return UNAVAILABLE


# ── Bullet rewriting ──────────────────────────────────────────────────────────

async def rewrite_bullets(weak_phrases: list[str], resume_text: str, job_description: str) -> str:
    if not weak_phrases:
        return "No weak phrases detected — your bullet points already use strong action language."

    # Five regex passes over the whole document, run on a worker thread. The
    # text is redacted before it is truncated, not after: slicing first can cut
    # an email address in half and send the surviving fragment upstream.
    # Running it inline held the event loop for the duration — the same defect
    # already fixed in /analyze, in a handler that shares the same process.
    redacted = await asyncio.to_thread(redact_for_prompt, resume_text)

    prompt = f"""You are an elite resume coach who has helped thousands of candidates land engineering roles.

WEAK PHRASES found in the resume: {", ".join(weak_phrases[:8])}

TARGET JOB DESCRIPTION:
{job_description[:600]}

RESUME EXCERPT (for context):
{redacted[:900]}

Rewrite exactly 4 improved bullet points. Rules:
- Start each with a strong past-tense action verb (Engineered, Architected, Spearheaded, Optimized)
- Include specific numbers, percentages or impact wherever the source supports it
- Do not invent metrics that are not implied by the resume excerpt
- Align language to the job description keywords
- Each bullet at most 20 words

Format:
• [Rewritten bullet 1]
• [Rewritten bullet 2]
• [Rewritten bullet 3]
• [Rewritten bullet 4]

Then one line: **Key change:** [what made these stronger]"""

    return await _safe_text(prompt, max_tokens=450)


# ── Cover letter ──────────────────────────────────────────────────────────────

async def generate_skill_roadmap(missing_skills: list[str], job_description: str) -> str:
    if not missing_skills:
        return (
            "You already match the critical skills for this role. Focus on deepening "
            "your expertise and building portfolio projects that demonstrate it."
        )

    prompt = f"""You are a senior engineering mentor building a focused learning plan.

Skills to acquire: {", ".join(missing_skills[:6])}
Target role context: {job_description[:500]}

Create a 30-day sprint roadmap in exactly this format:

**Week 1 — Foundation: [Skill]**
Goal: [one-line goal]
Resource: [type of resource and what to search for — do not invent URLs]
Daily action: [15-30 min daily task]

**Week 2 — Application: [Skill]**
Goal: [one-line goal]
Resource: [type of resource and what to search for]
Daily action: [task]

**Week 3 — Advanced: [Skill]**
Goal: [one-line goal]
Resource: [type of resource and what to search for]
Daily action: [task]

**Week 4 — Portfolio Project**
Build: [specific mini-project using all the skills above]
Outcome: [what to add to the resume afterwards]

Keep each section under 40 words. Name real tools. Never fabricate a URL."""

    return await _safe_text(prompt, max_tokens=500)


# ── Interview prep ────────────────────────────────────────────────────────────

# ── Dispatcher ────────────────────────────────────────────────────────────────

# One call per requested mode, generated on demand. Cover letters, LinkedIn
# summaries and interview questions were dropped: each cost a request from a
# hard daily budget to produce output any chat assistant writes just as well.
# Both remaining modes are anchored to the user's own resume text.
COACHING_MODES = ("bullets", "roadmap")

MODE_LABELS = {
    "bullets": "Improved bullet points",
    "roadmap": "30-day skill roadmap",
}


async def generate_one(
    mode: str,
    *,
    weak_phrases: list[str] | None = None,
    missing_skills: list[str] | None = None,
    job_description: str = "",
    resume_text: str = "",
) -> str:
    """Generate exactly one coaching artefact."""
    weak = weak_phrases or []
    missing = missing_skills or []

    if mode == "bullets":
        return await rewrite_bullets(weak, resume_text, job_description)
    if mode == "roadmap":
        return await generate_skill_roadmap(missing, job_description)

    raise ValueError(f"Unknown coaching mode {mode!r}; expected one of {COACHING_MODES}")
