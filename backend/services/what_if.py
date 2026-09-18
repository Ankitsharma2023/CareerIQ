"""What the score would be if a listed skill were actually shown.

The product tells a candidate that listing a skill is worth less than
demonstrating it. The obvious next question — "worth how much less, for *my*
resume?" — had no answer, so the advice was a principle rather than a
priority: nothing said which of eleven unsupported skills to spend the evening
rewriting.

This answers it by re-running the real scorer. For each candidate skill, the
evidence map is copied with that one skill promoted to the weight an experience
mention carries, and ``compute_fit`` is called again with every other input
unchanged. The difference is the projection.

Two properties follow from doing it this way rather than with a model or a
formula:

* It cannot disagree with the score. The number quoted is produced by the same
  function that produced the number on screen, so "+3.2" means the score will
  read 3.2 higher, not approximately.
* It costs nothing. No LLM call, no network, and the whole set is a few
  hundred microseconds of arithmetic — so it can be computed for every skill on
  every analysis rather than offered as a paid extra.

What it is *not*: advice to write a bullet the candidate cannot support. The
projection assumes the experience is real and merely unstated. That is the
common case — people list what they have used and describe only their last
job — but the copy around it has to say so, because a tool that reports "+4.1
if you claim Kubernetes" to someone who has never used Kubernetes is telling
them to lie profitably.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from services.evidence import SECTION_WEIGHT, SkillEvidence, evidence_ratio, weighted_coverage
from services.scoring import compute_fit

#: The weight a skill reaches once it appears in an experience bullet. Not the
#: quantified maximum: this projects "you mentioned it in context", not "you
#: mentioned it in context with a number attached", which would overstate what
#: the suggested edit actually earns.
DEMONSTRATED_WEIGHT = SECTION_WEIGHT["experience"]

#: Below this the suggestion is noise. A user asked to rewrite a bullet for a
#: tenth of a point will conclude the whole feature is decorative.
MIN_WORTHWHILE_GAIN = 0.1

MAX_SUGGESTIONS = 8


@dataclass
class WhatIf:
    skill: str
    current_weight: float
    projected_weight: float
    gain: float
    #: True when the posting states this skill as a requirement rather than a
    #: preference — which is what makes the gain worth acting on first, over an
    #: equal gain on something the posting calls nice to have.
    required_by_jd: bool

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "current_weight": round(self.current_weight, 3),
            "projected_weight": round(self.projected_weight, 3),
            "gain": round(self.gain, 2),
            "required_by_jd": self.required_by_jd,
        }


def project_gains(
    evidence: dict[str, SkillEvidence],
    jd_skills: list[str],
    fit_inputs: dict[str, Any],
    priorities: dict[str, str] | None = None,
    candidates: list[str] | None = None,
) -> list[WhatIf]:
    """Points gained by moving each listed-only skill into an experience bullet.

    ``fit_inputs`` is everything ``compute_fit`` needs except the two values
    this recomputes — coverage and the evidence ratio — so the projection holds
    semantic similarity, clarity and level fit fixed. Those do not change by
    moving a word into a different section, and letting them float would
    attribute unrelated movement to the suggested edit.
    """
    from services.jd_requirements import REQUIRED

    priorities = priorities or {}
    # In the posting *as a requirement* — a preferred skill is in jd_skills too,
    # and flagging it as required would mis-sort the advice.
    required = {
        s.strip().lower() for s in jd_skills
        if priorities.get(s.strip().lower(), REQUIRED) == REQUIRED
    }
    base_coverage = weighted_coverage(evidence, jd_skills, priorities)
    base = compute_fit(
        coverage_score=base_coverage,
        evidence_ratio_value=evidence_ratio(evidence),
        **fit_inputs,
    )

    if candidates is None:
        candidates = [
            entry.skill for entry in evidence.values() if not entry.is_demonstrated
        ]

    projections: list[WhatIf] = []
    for skill in candidates:
        entry = evidence.get(skill)
        if entry is None or entry.weight >= DEMONSTRATED_WEIGHT:
            continue

        # One skill promoted, everything else held exactly as it is.
        hypothetical = copy.deepcopy(evidence)
        promoted = hypothetical[skill]
        promoted.weight = DEMONSTRATED_WEIGHT
        promoted.sections = [*promoted.sections, "experience"]

        projected = compute_fit(
            coverage_score=weighted_coverage(hypothetical, jd_skills, priorities),
            evidence_ratio_value=evidence_ratio(hypothetical),
            **fit_inputs,
        )
        gain = projected.overall - base.overall
        if gain < MIN_WORTHWHILE_GAIN:
            continue

        projections.append(
            WhatIf(
                skill=skill,
                current_weight=entry.weight,
                projected_weight=DEMONSTRATED_WEIGHT,
                gain=gain,
                required_by_jd=skill in required,
            )
        )

    # Biggest gain first, and a skill the posting asks for ahead of an equal
    # gain it does not.
    projections.sort(key=lambda p: (-p.gain, not p.required_by_jd, p.skill))
    return projections[:MAX_SUGGESTIONS]
