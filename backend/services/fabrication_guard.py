"""Check that a rewritten bullet says nothing the original did not.

The bullet rewriter is the feature most likely to do direct harm. It is asked
to make a resume line stronger, and "stronger" in this genre means numbers —
so a model with no numbers to work from will supply them. A user who pastes
"Improved API response time by 40%" into a resume because a tool wrote it is
now carrying a fabricated metric into an interview where someone will ask about
it.

The prompt already says not to. This is the part that checks, because a prompt
instruction is a request and not a guarantee.

Three things are compared against the source text:

* **Numbers.** Any quantity in the rewrite that is not in the original.
  Normalised first, so "40M", "40 million" and "40,000,000" are one number and
  a faithful rephrasing is not flagged.
* **Tools.** Any technology named in the rewrite and not in the original.
  Adding "using Kubernetes" to a bullet that never mentioned it is inventing
  experience, even when the résumé lists Kubernetes elsewhere: the claim being
  made is that this work used it.
* **Named things.** Capitalised names — employers, products, certifications,
  standards — that appear only in the rewrite.

Conservative on purpose: it only ever flags something *added*. Dropping detail,
rephrasing, reordering and restructuring all pass, because none of them can
make the resume claim something untrue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Quantities: percentages, money, multipliers, counts, and scale suffixes.
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"\s*(%|percent|k\b|m\b|b\b|bn\b|thousand|million|billion|x\b)?",
    re.I,
)

_SCALE = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

_PROPER_RE = re.compile(r"\b([A-Z][A-Za-z0-9+#.\-]{1,})\b")

# Where a new segment starts: a line, a bullet, or a sentence end. The first
# word of each is capitalised by convention rather than because it is a name,
# and resume bullets conventionally open with a verb — "Served 40 million
# requests" is not a claim about a company called Served.
#
# A capitalised first word is therefore skipped. This is better than the
# alternative of listing every past-tense verb a model might reach for: that
# list is unbounded, and each word missing from it is a false positive that
# rejects a faithful rewrite and spends a retry.
_SEGMENT_SPLIT_RE = re.compile(r"(?:[.!?;:]\s+|\n+|^)\s*(?:[-*\u2022]\s*)?")

# Words that are capitalised for reasons other than being a name.
_NOT_NAMES = {
    "I", "A", "An", "The", "This", "That", "These", "Those", "My", "Our", "We",
    "It", "Its", "In", "On", "At", "For", "With", "By", "To", "From", "Of",
    "And", "Or", "But", "As", "Led", "Built", "Designed", "Developed", "Managed",
    "Created", "Improved", "Increased", "Reduced", "Delivered", "Implemented",
    "Engineered", "Architected", "Spearheaded", "Optimized", "Optimised",
    "Drove", "Owned", "Shipped", "Launched", "Migrated", "Automated", "Scaled",
    "Key", "Change", "Rewritten", "Bullet", "Team", "Company", "Project",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}


@dataclass
class Fabrication:
    """Something the rewrite asserts that the source did not."""

    kind: str   # number | tool | name
    value: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.value}"


def _numbers(text: str) -> set[float]:
    """Every quantity in the text, normalised so scale words do not matter."""
    found: set[float] = set()
    for digits, suffix in _NUMBER_RE.findall(text or ""):
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        suffix = (suffix or "").lower().strip()
        if suffix in _SCALE:
            value *= _SCALE[suffix]
        # A percentage and a bare count are different claims, so they are kept
        # apart: "40%" must not be satisfied by a source that said "40 users".
        if suffix in ("%", "percent"):
            found.add(-value)
        else:
            found.add(value)
    return found


def _tools(text: str, skills_list) -> set[str]:
    from services.skill_extractor import extract_skills_from_text

    return set(extract_skills_from_text(text or "", skills_list))


def _names(text: str) -> set[str]:
    found: set[str] = set()
    for segment in _SEGMENT_SPLIT_RE.split(text or ""):
        segment = segment.strip()
        if not segment:
            continue
        # Drop the leading word, whatever it is, then take the rest.
        remainder = segment.split(" ", 1)
        if len(remainder) < 2:
            continue
        found.update(
            token for token in _PROPER_RE.findall(remainder[1])
            if token not in _NOT_NAMES
        )
    return found


def find_fabrications(source: str, rewritten: str, skills_list) -> list[Fabrication]:
    """Everything the rewrite claims that the source does not support."""
    source_lower = (source or "").lower()

    fabrications = [
        Fabrication("number", _format(value))
        for value in sorted(_numbers(rewritten) - _numbers(source))
    ]
    fabrications += [
        Fabrication("tool", tool)
        for tool in sorted(_tools(rewritten, skills_list) - _tools(source, skills_list))
    ]
    # A technology already reported as an invented tool is not reported again
    # as an invented name; one finding per thing invented.
    reported_tools = {f.value for f in fabrications if f.kind == "tool"}
    fabrications += [
        Fabrication("name", name)
        for name in sorted(_names(rewritten) - _names(source))
        if name.lower() not in source_lower and name.lower() not in reported_tools
    ]
    return fabrications


def _format(value: float) -> str:
    if value < 0:
        return f"{-value:g}%"
    return f"{value:g}"


def is_faithful(source: str, rewritten: str, skills_list) -> bool:
    return not find_fabrications(source, rewritten, skills_list)


def describe(fabrications: list[Fabrication]) -> str:
    """A correction to hand back to the model on a retry."""
    numbers = [f.value for f in fabrications if f.kind == "number"]
    tools = [f.value for f in fabrications if f.kind == "tool"]
    names = [f.value for f in fabrications if f.kind == "name"]

    parts = []
    if numbers:
        parts.append(f"figures that do not appear in the resume: {', '.join(numbers)}")
    if tools:
        parts.append(f"technologies the original bullet did not mention: {', '.join(tools)}")
    if names:
        parts.append(f"names the original did not contain: {', '.join(names)}")
    return "; ".join(parts)
