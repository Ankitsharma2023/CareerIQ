"""Sort the skills a resume is missing by how much the posting wants them."""
from __future__ import annotations

from services.jd_requirements import PREFERRED, REQUIRED, classify_jd_skills


def classify_skill_gaps(
    missing_skills,
    jd_skills,
    jd_text,
    priorities: dict[str, str] | None = None,
):
    """Bucket the missing skills into critical, important and optional.

    This used to grade a gap by how many times the skill appeared in the
    posting: twice or more was critical, once was important, never was
    optional. Two things were wrong with that. A requirement stated once ranked
    below a preferred tool that happened to be repeated, and the "optional"
    bucket could only ever hold skills that were not in the posting at all —
    so nothing a posting actually called optional ever landed there.

    The posting's own structure decides it now. Within the required skills,
    repetition still separates critical from important: a requirement a posting
    returns to is one it cares about.
    """
    if priorities is None:
        from services.skill_extractor import load_skills

        priorities = classify_jd_skills(jd_text, load_skills())

    classification = {"critical": [], "important": [], "optional": []}
    lowered = (jd_text or "").lower()

    for skill in missing_skills:
        mentions = lowered.count(skill.lower())
        priority = priorities.get(skill.lower())

        if mentions == 0 or priority == PREFERRED:
            # Either the posting never names it, or it names it as a
            # preference. Neither is a requirement of this job.
            classification["optional"].append(skill)
        elif mentions >= 2:
            classification["critical"].append(skill)
        else:
            classification["important"].append(skill)

    return classification


def missing_must_haves(missing_skills, priorities: dict[str, str]) -> list[str]:
    """The gaps that are stated requirements, called out on their own.

    A recruiter screens on these before reading anything else, so a candidate
    needs them separated from the long tail of nice-to-haves rather than mixed
    into one gap count.
    """
    return sorted(
        skill for skill in missing_skills
        if priorities.get(skill.lower()) == REQUIRED
    )
