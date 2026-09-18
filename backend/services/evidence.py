"""Evidence-weighted skill detection.

Measured problem this solves: a resume listing 48 technologies under "Technical
Skills" with no evidence of having used any of them ranked *first* — above the
engineer who had actually built the pipelines. Presence-based matching cannot
tell a claim from a demonstration, so the cheapest possible resume wins.

The fix is to weight a skill by *where* it appears. A skill named in an
experience bullet alongside what was built with it is worth far more than the
same word sitting in a comma-separated list, and a skill attached to a
quantified outcome is worth more still.

This is deliberately not a keyword count. Listing a skill is not evidence of
it; that is the entire point.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from services.section_parser import parse_sections
from services.skill_extractor import extract_skills_from_text

# How much a mention is worth, by the section it appears in.
# "skills" is low on purpose: a list is a claim, not a demonstration.
SECTION_WEIGHT: dict[str, float] = {
    "experience": 1.00,
    "projects": 0.90,
    "certifications": 0.65,
    "summary": 0.50,
    "other": 0.50,
    "education": 0.40,
    "skills": 0.30,
}

# A mention sitting next to a number ("cut p95 by 60%", "serving 40M requests")
# is corroborated in a way a bare mention is not.
QUANTIFIED_BONUS = 0.25
MAX_WEIGHT = 1.0

_QUANTIFIER_RE = re.compile(
    r"\d+\s*%|\$\s*\d+|\d+\s*[kmb]\b|\d+x\b|\d{3,}|"
    r"(?:increased|decreased|reduced|improved|grew|saved|cut|serving|handling)\s+[^.]*\d+",
    re.I,
)


#: How many supporting sentences to keep per skill. Three is enough to justify
#: a score to a user; keeping all of them makes the response grow with the
#: length of the resume for no added explanatory value.
MAX_MENTIONS = 3

#: A sentence longer than this is almost certainly a paragraph the splitter
#: failed on, and highlighting half a page explains nothing.
MAX_MENTION_CHARS = 400


@dataclass
class SkillMention:
    """One place a skill was actually written, and what it was written next to."""

    section: str
    sentence: str
    quantified: bool = False

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "sentence": self.sentence,
            "quantified": self.quantified,
        }


@dataclass
class SkillEvidence:
    """Where a skill was found and how much that is worth."""

    skill: str
    sections: list[str] = field(default_factory=list)
    weight: float = 0.0
    quantified: bool = False
    #: The sentences behind the weight, so the score can be shown its working.
    mentions: list[SkillMention] = field(default_factory=list)

    @property
    def is_demonstrated(self) -> bool:
        """Appears somewhere other than a bare skills list."""
        return any(section not in ("skills",) for section in self.sections)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+|\n", text) if s.strip()]


def _mentions_of(text: str, skill: str, section: str) -> list[SkillMention]:
    """Every sentence in this section that names the skill.

    Matching is done on the sentence as written, so what comes back is the
    candidate's own words — which is the point. A paraphrase would not let a
    user check the claim against their resume.
    """
    found = []
    for sentence in _sentences(text):
        if skill not in sentence.lower():
            continue
        if len(sentence) > MAX_MENTION_CHARS:
            continue
        found.append(
            SkillMention(
                section=section,
                sentence=sentence,
                quantified=bool(_QUANTIFIER_RE.search(sentence)),
            )
        )
    return found


def _has_quantifier_near(text: str, skill: str) -> bool:
    """Is the skill mentioned in a sentence that also carries a number?"""
    return any(mention.quantified for mention in _mentions_of(text, skill, ""))


def gather_evidence(
    resume_text: str,
    skills_list: Sequence[str],
    sections: dict[str, str] | None = None,
) -> dict[str, SkillEvidence]:
    """Map each detected skill to the strength of its supporting evidence."""
    parsed = sections if sections is not None else parse_sections(resume_text)
    evidence: dict[str, SkillEvidence] = {}

    for section_name, section_text in parsed.items():
        if not section_text or not section_text.strip():
            continue

        weight = SECTION_WEIGHT.get(section_name, 0.5)
        for skill in extract_skills_from_text(section_text, skills_list):
            entry = evidence.setdefault(skill, SkillEvidence(skill=skill))
            entry.sections.append(section_name)

            mentions = _mentions_of(section_text, skill, section_name)
            score = weight
            if any(mention.quantified for mention in mentions):
                entry.quantified = True
                score = min(MAX_WEIGHT, score + QUANTIFIED_BONUS)

            entry.weight = max(entry.weight, score)

            # Quantified mentions first: those are the ones that earned the
            # bonus, so they are the ones worth showing when the user asks why.
            for mention in sorted(mentions, key=lambda m: not m.quantified):
                if len(entry.mentions) >= MAX_MENTIONS:
                    break
                if all(mention.sentence != kept.sentence for kept in entry.mentions):
                    entry.mentions.append(mention)

    # A skill the section parser missed entirely still counts, at the weight of
    # an unattributed mention — section detection is heuristic and should not
    # silently drop a real skill.
    for skill in extract_skills_from_text(resume_text, skills_list):
        if skill not in evidence:
            evidence[skill] = SkillEvidence(
                skill=skill, sections=["other"], weight=SECTION_WEIGHT["other"]
            )

    return evidence


def weighted_coverage(
    evidence: dict[str, SkillEvidence],
    jd_skills: Sequence[str],
) -> float:
    """Percentage of the job's skills that are actually evidenced.

    A resume claiming every skill in a list scores roughly a third of one that
    demonstrates the same skills in its experience.
    """
    required = [s for s in {s.strip().lower() for s in jd_skills} if s]
    if not required:
        return 0.0

    total = sum(evidence[skill].weight for skill in required if skill in evidence)
    return round((total / len(required)) * 100, 2)


def evidence_ratio(evidence: dict[str, SkillEvidence]) -> float:
    """Share of claimed skills that appear outside a bare skills list.

    Near zero is the signature of a padded skills section.
    """
    if not evidence:
        return 0.0
    demonstrated = sum(1 for entry in evidence.values() if entry.is_demonstrated)
    return round(demonstrated / len(evidence), 3)


def unsupported_skills(evidence: dict[str, SkillEvidence]) -> list[str]:
    """Skills claimed in a list but never shown in context.

    Useful to the candidate directly: these are the lines a recruiter will
    probe and the resume cannot answer.
    """
    return sorted(
        entry.skill for entry in evidence.values() if not entry.is_demonstrated
    )


def evidence_payload(
    evidence: dict[str, SkillEvidence],
    skills: Sequence[str] | None = None,
) -> dict[str, dict]:
    """The per-skill working, for the UI to show when a user asks why.

    Restricted to ``skills`` when given — normally the ones actually on screen,
    so the response does not carry the working for skills nobody asked about.
    """
    wanted = {s.strip().lower() for s in skills} if skills is not None else None
    return {
        entry.skill: {
            "weight": round(entry.weight, 3),
            "sections": sorted(set(entry.sections)),
            "quantified": entry.quantified,
            "demonstrated": entry.is_demonstrated,
            "mentions": [m.to_dict() for m in entry.mentions],
        }
        for entry in evidence.values()
        if wanted is None or entry.skill in wanted
    }
