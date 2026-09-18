def get_missing_skills(resume_skills: list, jd_skills: list) -> list:
    """Return skills present in JD but missing in resume"""
    return list(set(jd_skills) - set(resume_skills))


def get_matching_skills(resume_skills: list, jd_skills: list) -> list:
    """Return common skills"""
    return list(set(resume_skills).intersection(set(jd_skills)))


# generate_feedback lived here. It formatted the match score and the missing
# skill list into a paragraph — both already on the screen above it, as a ring,
# a bar and a badge list — and ended with the same generic advice every time.
