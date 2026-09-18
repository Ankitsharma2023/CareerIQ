"""Weight a skill by how recently it was used.

A candidate who shipped Kubernetes last year and one who touched it in 2016 do
not have the same Kubernetes experience, and the resume says which is which.
Until now the scorer read both as a mention in the Experience section and gave
them identical weight.

The parse reuses the year ranges the experience detector already relies on. The
Experience section is cut into role blocks at each date range, and every skill
in a block inherits that block's end year; a skill appearing in several roles
takes the most recent, because that is the one a recruiter would ask about.

Three deliberate limits:

* **The multiplier floors, it does not decay to nothing.** Eight-year-old
  Postgres experience is weaker evidence than last year's, but it is not zero
  evidence — the person has still done it. A decay that reached zero would rank
  a career changer below someone who has never touched the tool at all.
* **An undated block is treated as current.** Resumes routinely omit dates on
  projects and on the most recent role. Penalising a missing date would
  penalise formatting, and this file is not a formatting critic.
* **Nothing outside Experience is aged.** A skills list has no date to reason
  from, and a project section rarely carries one.
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

from services.section_parser import parse_sections

# The same shape the experience detector matches, so the two cannot disagree
# about what a date range is.
_DATE_RANGE = re.compile(
    r"(20\d{2})\s*(?:[-–—]|to)\s*(20\d{2}|present|current|now)", re.I
)
_OPEN_ENDED = ("present", "current", "now")

#: Years after which a mention has aged as far as it will. Chosen against how
#: engineering resumes actually read: within about five years a tool is still
#: "what I work with", beyond eight it is "what I used to do".
FULL_WEIGHT_YEARS = 2
FLOOR_YEARS = 8

#: What an old mention retains. Not lower, because the alternative claim — that
#: someone who ran Postgres for six years in 2016 knows less about it than
#: someone who has never opened it — is false.
OLDEST_MULTIPLIER = 0.75


def _current_year() -> int:
    """Resolved at call time. Hardcoding it makes every resume age a year
    every January without anyone touching the code."""
    return datetime.date.today().year


def multiplier_for(end_year: int | None, current_year: int | None = None) -> float:
    """How much a mention from ``end_year`` is worth, between the floor and 1.0."""
    if end_year is None:
        return 1.0
    current_year = current_year or _current_year()

    age = current_year - end_year
    if age <= FULL_WEIGHT_YEARS:
        return 1.0
    if age >= FLOOR_YEARS:
        return OLDEST_MULTIPLIER

    # Linear between the two, so there is no cliff a candidate falls off by
    # having left a job in December rather than January.
    span = FLOOR_YEARS - FULL_WEIGHT_YEARS
    travelled = (age - FULL_WEIGHT_YEARS) / span
    return 1.0 - travelled * (1.0 - OLDEST_MULTIPLIER)


def role_blocks(experience_text: str, current_year: int | None = None) -> list[tuple[str, int]]:
    """Cut the Experience section into (text, end year) at each date range.

    Text before the first date range belongs to no dated role and is returned
    with the current year, so an undated opening block is not aged.
    """
    current_year = current_year or _current_year()
    matches = list(_DATE_RANGE.finditer(experience_text or ""))
    if not matches:
        return [(experience_text or "", current_year)] if experience_text else []

    blocks: list[tuple[str, int]] = []
    if matches[0].start() > 0:
        head = experience_text[: matches[0].start()].strip()
        if head:
            blocks.append((head, current_year))

    for index, match in enumerate(matches):
        end_raw = match.group(2).lower()
        end_year = current_year if end_raw in _OPEN_ENDED else int(end_raw)
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(experience_text)
        blocks.append((experience_text[start:stop], end_year))

    return blocks


@dataclass
class SkillRecency:
    """When a skill was last used, and what that is worth."""

    last_used: int
    multiplier: float

    def to_dict(self) -> dict:
        return {"last_used": self.last_used, "multiplier": round(self.multiplier, 3)}


def skill_recency(
    resume_text: str, skills_list, current_year: int | None = None
) -> dict[str, SkillRecency]:
    """Map each skill mentioned in Experience to its recency multiplier.

    A skill named in several roles takes its most recent one — that is the
    experience a recruiter would ask about, and averaging would penalise a long
    career for having had a first job.
    """
    from services.skill_extractor import extract_skills_from_text

    current_year = current_year or _current_year()
    experience = (parse_sections(resume_text).get("experience") or "").strip()
    if not experience:
        return {}

    best: dict[str, SkillRecency] = {}
    for text, end_year in role_blocks(experience, current_year):
        factor = multiplier_for(end_year, current_year)
        for skill in extract_skills_from_text(text, skills_list):
            current = best.get(skill)
            if current is None or factor > current.multiplier:
                best[skill] = SkillRecency(last_used=end_year, multiplier=factor)

    return best
