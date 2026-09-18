"""Skill extraction by a model, with every claim checked against the resume.

The deterministic extractor can only find skills that are already in the
taxonomy. A model can read "built our own columnar store on top of Arrow" and
name a skill no tag list contains — which is exactly the case the taxonomy
cannot cover, and exactly the case where a model is most likely to invent
something.

So the model is not trusted to report skills. It is required to quote the
resume for each one, and the quote is checked in code:

* the quote must appear in the resume, contiguously, as written;
* the skill must be recognisable in its own quote.

A skill whose quote does not verify is dropped, and the drop is counted.
That count is the useful number — it is the model's fabrication rate on real
documents, measurable without a human reading anything, and
evals/bakeoff.py is where it gets compared across models.

The deterministic extractor stays. It is the fallback when there is no API key,
when the call times out, or when the response fails validation; it is the
baseline the agent is measured against; and its results are unioned with the
verified ones, because a model that misses a skill the taxonomy would have
found should not cost the candidate that skill.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from core import llm
from core.config import settings
from core.redact import redact_for_prompt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: How much of the resume the agent is shown. Matches the evaluator's budget.
MAX_RESUME_CHARS = 3500

#: A quote longer than this is a paragraph, not evidence of one skill.
MAX_QUOTE_CHARS = 300


class ExtractedSkill(BaseModel):
    """One skill, and the words in the resume that support it."""

    skill: str = Field(description="The skill, lowercase, as it would appear in a taxonomy")
    quote: str = Field(
        description=(
            "The exact sentence or phrase from the resume that shows this skill, "
            "copied word for word. Never paraphrase or summarise."
        )
    )


class SkillExtraction(BaseModel):
    skills: list[ExtractedSkill]


@dataclass
class AgentResult:
    """What survived verification, and what did not."""

    verified: dict[str, str] = field(default_factory=dict)   # skill -> quote
    rejected: list[tuple[str, str, str]] = field(default_factory=list)  # skill, quote, why
    called: bool = False
    failed: bool = False

    @property
    def fabrication_rate(self) -> float:
        """Share of returned skills whose quote could not be found in the resume."""
        total = len(self.verified) + len(self.rejected)
        return round(len(self.rejected) / total, 4) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "called": self.called,
            "failed": self.failed,
            "verified_count": len(self.verified),
            "rejected_count": len(self.rejected),
            "fabrication_rate": self.fabrication_rate,
            "rejected": [
                {"skill": skill, "quote": quote[:120], "reason": reason}
                for skill, quote, reason in self.rejected[:10]
            ],
        }


def _normalize(text: str) -> str:
    """Collapse whitespace and case so a quote survives reflowing, nothing more.

    Deliberately not stemming, stripping punctuation or removing stopwords: the
    check has to fail on a paraphrase, and every additional normalisation makes
    a paraphrase likelier to pass.
    """
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def verify(skill: str, quote: str, resume_text: str) -> str | None:
    """Return the reason this claim fails, or None if it stands."""
    if not quote or not quote.strip():
        return "no quote given"
    if len(quote) > MAX_QUOTE_CHARS:
        return "quote is a paragraph, not evidence of one skill"

    haystack = _normalize(resume_text)
    needle = _normalize(quote)
    if needle not in haystack:
        return "quote does not appear in the resume"

    # The quote has to be about the skill. Without this a model can satisfy the
    # check by quoting any real sentence for any skill it likes.
    if _normalize(skill) not in needle:
        tokens = [t for t in re.split(r"[\s/()-]+", _normalize(skill)) if len(t) > 2]
        if not tokens or not any(token in needle for token in tokens):
            return "quote does not mention the skill"

    return None


def check(extraction: SkillExtraction, resume_text: str) -> AgentResult:
    """Verify every claim the model made."""
    result = AgentResult(called=True)
    for item in extraction.skills:
        skill = (item.skill or "").strip().lower()
        if not skill:
            continue
        reason = verify(skill, item.quote, resume_text)
        if reason:
            result.rejected.append((skill, item.quote, reason))
        else:
            result.verified[skill] = item.quote.strip()
    return result


def build_prompt(resume_text: str, contact_info: dict | None = None) -> str:
    excerpt = redact_for_prompt(resume_text[:MAX_RESUME_CHARS], contact_info)
    return f"""You are extracting the technical skills a candidate demonstrably has.

Read the RESUME below. For every skill it shows, return the skill and the exact
words from the resume that show it.

Rules, all of which are checked in code after you answer:
- The quote must be copied from the resume word for word. Not paraphrased, not
  tidied, not summarised. If you cannot quote it, do not report the skill.
- The quote must mention the skill it is offered for.
- Report a skill once, with its strongest quote.
- Do not report a skill that only appears in a comma-separated list unless the
  list is all you have — prefer a quote showing the skill being used.
- Do not infer. "Built microservices" is not evidence of Kubernetes.

Any skill whose quote fails these checks is discarded, so a fabricated quote
costs you the skill rather than gaining you one.

<<<RESUME>>>
{excerpt}
<<<END RESUME>>>
"""


async def extract_with_agent(
    resume_text: str,
    contact_info: dict | None = None,
) -> AgentResult:
    """Ask the model for skills, keep only the ones it can quote for."""
    if not llm.is_configured():
        return AgentResult()

    try:
        extraction = await asyncio.wait_for(
            llm.complete_json(
                build_prompt(resume_text, contact_info),
                SkillExtraction,
                max_tokens=1500,
                temperature=0.0,
                schema_name="skill_extraction",
            ),
            timeout=settings.llm_deadline_s,
        )
    except asyncio.TimeoutError:
        logger.error("Skill agent exceeded %.0fs", settings.llm_deadline_s)
        return AgentResult(called=True, failed=True)
    except llm.LLMError as exc:
        logger.error("Skill agent failed: %s", exc)
        return AgentResult(called=True, failed=True)

    result = check(extraction, resume_text)
    if result.rejected:
        logger.info(
            "Skill agent: %d of %d claims unverified (%.0f%%)",
            len(result.rejected),
            len(result.rejected) + len(result.verified),
            result.fabrication_rate * 100,
        )
    return result
