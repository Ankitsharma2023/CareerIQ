import logging

from core.config import settings
from core.limiter import limiter
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from services.ai_coach import COACHING_MODES, MODE_LABELS, generate_one

logger = logging.getLogger(__name__)
router = APIRouter()


class CoachingRequest(BaseModel):
    """One coaching artefact, generated on demand.

    Replaces the five-call bundle, which fired every generator on every visit
    to the tab and cost five of roughly twenty free requests per minute for
    output the user may never scroll to.
    """

    mode: str = Field(description=f"One of: {', '.join(COACHING_MODES)}")
    weak_phrases: list[str] = Field(default=[], max_length=50)
    missing_skills: list[str] = Field(default=[], max_length=100)
    # Bounded because redaction runs five regex passes over resume_text before
    # the prompt is built. Unbounded, a single large paste made those passes
    # arbitrarily long — and an oversized body should be refused at the edge
    # with a 422 rather than absorbed and worked on.
    job_description: str = Field(default="", max_length=settings.max_jd_length)
    resume_text: str = Field(default="", max_length=settings.max_resume_text_length)


class CoachingResponse(BaseModel):
    mode: str
    label: str
    content: str


@router.get("/ai-coach/modes", tags=["AI Coach"])
async def list_modes():
    """What the client can ask for, so the UI is not hardcoded to the server."""
    return {"modes": [{"id": mode, "label": MODE_LABELS[mode]} for mode in COACHING_MODES]}


@router.post("/ai-coach/generate", response_model=CoachingResponse)
@limiter.limit(settings.rate_limit)
async def generate(request: Request, req: CoachingRequest):
    """Generate one coaching artefact."""
    if req.mode not in COACHING_MODES:
        raise HTTPException(
            400,
            detail=f"Unknown mode {req.mode!r}. Expected one of: {', '.join(COACHING_MODES)}",
        )

    try:
        content = await generate_one(
            req.mode,
            weak_phrases=req.weak_phrases,
            missing_skills=req.missing_skills,
            job_description=req.job_description,
            resume_text=req.resume_text,
        )
    except Exception as exc:
        logger.exception("Coaching generation failed for mode %s", req.mode)
        raise HTTPException(500, detail="Could not generate that right now.") from exc

    return CoachingResponse(mode=req.mode, label=MODE_LABELS[req.mode], content=content)
