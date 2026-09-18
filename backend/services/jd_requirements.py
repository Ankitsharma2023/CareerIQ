"""Split a job description into what is required and what is merely preferred.

Recruiters read these differently and so should the score. Missing a "nice to
have" is not the same kind of gap as missing a stated requirement, and until
now the system could not tell them apart: ``classify_skill_gaps`` graded a gap
by how many times the word appeared in the posting, so a requirement mentioned
once ranked below a preferred tool mentioned twice.

The parse is deliberately conservative. Job descriptions are written by people,
not to a schema, and a wrong split is worse than no split — telling a candidate
that a genuine requirement is optional is the expensive error. So:

* a section is only "preferred" when the posting says so in one of the usual
  ways, and everything else defaults to required;
* an inline marker on a single line ("Go is a plus") applies to that line only;
* if nothing in the posting marks anything as preferred, every skill comes back
  required and the scoring is exactly what it was before this existed.
"""
from __future__ import annotations

import re

REQUIRED = "required"
PREFERRED = "preferred"

#: How much a preferred skill counts, relative to a required one, when scoring
#: coverage. Not zero: a posting listing Go under "nice to have" still prefers
#: a candidate who knows Go, and scoring it at nothing would rank two otherwise
#: identical candidates the same.
PREFERRED_WEIGHT = 0.4

# Headings that open a preferred block. Ordered longest-first where one
# contains another, so "preferred qualifications" is not matched as "preferred".
_PREFERRED_HEADINGS = [
    r"nice[\s-]*to[\s-]*haves?",
    r"good[\s-]*to[\s-]*have",
    r"preferred qualifications",
    r"preferred skills",
    r"preferred",
    r"bonus points",
    r"bonus",
    r"desirable",
    r"pluses",
    r"we'?d love",
    r"would be a plus",
    r"not required",
]

# Headings that open (or re-open) a required block.
_REQUIRED_HEADINGS = [
    r"requirements?",
    r"required qualifications",
    r"required skills",
    r"minimum qualifications",
    r"basic qualifications",
    r"must[\s-]*haves?",
    r"what you'?ll need",
    r"what we'?re looking for",
    r"you will need",
    r"qualifications",
    r"essential",
    r"responsibilities",
    r"about the role",
]

_PREFERRED_HEADING_RE = re.compile(
    rf"^\s*[-*•]?\s*(?:{'|'.join(_PREFERRED_HEADINGS)})\b", re.I
)
_REQUIRED_HEADING_RE = re.compile(
    rf"^\s*[-*•]?\s*(?:{'|'.join(_REQUIRED_HEADINGS)})\b", re.I
)

# Markers that make one line preferred without opening a section.
_INLINE_PREFERRED_RE = re.compile(
    r"\b(?:is a plus|a plus\b|nice to have|good to have|bonus|preferred|"
    r"desirable|not required|optional)\b",
    re.I,
)

#: A "preferred" heading with this much text after it is probably not a heading
#: at all but a sentence that happens to contain the word.
_MAX_HEADING_LENGTH = 60

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Any short, unbulleted line ending in a colon is a new section. Without this,
# a "Nice to have" block ran on through whatever followed it — "Benefits:",
# "About us:", "Compensation:" — and anything named there was read as a
# preference. The bullet exclusion matters: "- Kafka" is an item, not a heading.
_SECTION_BREAK_RE = re.compile(r"^(?![-*\u2022])\s*\S[^\n]{0,58}:\s*$")


def split_requirements(jd_text: str) -> tuple[str, str]:
    """Return (required text, preferred text).

    Lines default to required. A preferred heading switches the mode until a
    required heading switches it back, which handles the common shape where
    "Nice to have" is followed by "Benefits" or "About us".
    """
    required: list[str] = []
    preferred: list[str] = []
    mode = REQUIRED

    for line in (jd_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        is_short = len(stripped) <= _MAX_HEADING_LENGTH
        if is_short and _PREFERRED_HEADING_RE.match(stripped):
            mode = PREFERRED
            continue
        if is_short and _REQUIRED_HEADING_RE.match(stripped):
            mode = REQUIRED
            continue
        if _SECTION_BREAK_RE.match(line):
            # An unrecognised section heading. It is not a preference block, so
            # stop treating what follows as one.
            mode = REQUIRED
            continue

        # An inline marker binds to its own *sentence*, not the whole line and
        # not the rest of the posting. Many descriptions are one unbroken
        # paragraph: "...docker deployment. Docker containers are essential.
        # Kubernetes experience is preferred." Applying the marker to the line
        # would file docker as optional on the strength of a sentence about
        # Kubernetes — and telling a candidate a real requirement is optional
        # is the expensive direction to be wrong in.
        for sentence in _SENTENCE_SPLIT_RE.split(stripped):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _INLINE_PREFERRED_RE.search(sentence):
                preferred.append(sentence)
            else:
                (preferred if mode == PREFERRED else required).append(sentence)

    return "\n".join(required), "\n".join(preferred)


def classify_jd_skills(jd_text: str, skills_list) -> dict[str, str]:
    """Map each skill the posting names to ``required`` or ``preferred``.

    A skill appearing in both halves is required — the stronger claim wins,
    because under-stating a requirement is the error that costs a candidate an
    application.
    """
    from services.skill_extractor import extract_skills_from_text

    required_text, preferred_text = split_requirements(jd_text)

    priorities = {
        skill: PREFERRED
        for skill in extract_skills_from_text(preferred_text, skills_list)
    }
    for skill in extract_skills_from_text(required_text, skills_list):
        priorities[skill] = REQUIRED

    return priorities


def skill_weight(priority: str) -> float:
    return PREFERRED_WEIGHT if priority == PREFERRED else 1.0
