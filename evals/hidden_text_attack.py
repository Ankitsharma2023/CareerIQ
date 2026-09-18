#!/usr/bin/env python3
"""How much does invisible text move each scorer?

The attack: take a candidate who does not fit the job, and paste the job's own
skill list into the PDF where a reader cannot see it — white on white, half a
point tall, or below the page edge. The resume looks unchanged. Every text
extractor reads the skills.

The measurement is score inflation: the attacked resume's score minus the same
resume's score without the hidden block. Zero is the only good answer. It is
reported for four scorers on the same documents:

    tfidf            TF-IDF cosine, the standard non-neural baseline
    embedding        the sentence embedding similarity this project uses
    careeriq_raw     the full pipeline reading all extractable text
    careeriq         the full pipeline reading only what a reader can see

    python evals/hidden_text_attack.py
    python evals/hidden_text_attack.py --json > attack.json
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pdfplumber  # noqa: E402

from evals.dataset import load_cases  # noqa: E402
from evals.pdf_builder import ATTACK_MODES, resume_pdf  # noqa: E402
from evals.provenance import provenance  # noqa: E402
from evals.tfidf import TfidfSpace  # noqa: E402

from services.evidence import (  # noqa: E402  isort:skip
    evidence_ratio,
    gather_evidence,
    unsupported_skills,
    weighted_coverage,
)
from services.experience_detector import detect_experience  # noqa: E402  isort:skip
from services.jd_requirements import classify_jd_skills  # noqa: E402  isort:skip
from services.hidden_text import extract_visible_text  # noqa: E402  isort:skip
from services.scoring import compute_fit  # noqa: E402  isort:skip
from services.signal_noise_analyzer import analyze_signal_to_noise  # noqa: E402  isort:skip
from services.similarity import calculate_similarity  # noqa: E402  isort:skip
from services.skill_extractor import extract_skills_from_text, load_skills  # noqa: E402  isort:skip

# Candidates this weak are where an inflated score does real harm: it puts
# someone unqualified above someone qualified.
WEAK_ENOUGH_TO_ATTACK = 1


def careeriq_score(resume_text: str, jd_text: str, skills: list[str]) -> float:
    """The number the product shows, via the same function the API calls."""
    evidence = gather_evidence(resume_text, skills)
    jd_skills = extract_skills_from_text(jd_text, skills)
    priorities = classify_jd_skills(jd_text, skills)
    experience = detect_experience(resume_text, jd_text)
    fit = compute_fit(
        semantic_score=calculate_similarity(resume_text, jd_text)["final_score"],
        coverage_score=weighted_coverage(evidence, jd_skills, priorities),
        clarity_score=analyze_signal_to_noise(resume_text)["clarity_score"],
        detected_years=experience["detected_years"],
        required_years=experience["required_years"],
        jd_text=jd_text,
        evidence_ratio_value=evidence_ratio(evidence),
        unsupported=unsupported_skills(evidence),
    )
    return fit.overall


def read_pdf(data: bytes) -> tuple[str, str]:
    """Return (everything extractable, only what a reader can see)."""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        raw = "\n".join(
            page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf.pages
        )
        # Off-page characters can fall outside the layout grid, so they are
        # appended directly — otherwise the "no detector" arm would look
        # stronger than it is against that one variant.
        loose = "".join(c["text"] for page in pdf.pages for c in page.chars)
        if len(loose) > len(raw.replace("\n", "")):
            raw = raw + "\n" + loose
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        visible, _ = extract_visible_text(pdf)
    return raw, visible


def hidden_block(jd_text: str, skills: list[str]) -> list[str]:
    """The JD's own skill list, which is what an attacker actually pastes."""
    jd_skills = extract_skills_from_text(jd_text, skills)
    return [" ".join(jd_skills)] if jd_skills else []


def run(limit: int | None = None) -> dict:
    cases = load_cases()
    skills = load_skills()

    corpus = [c.jd_text for c in cases] + [
        cand.resume_text for c in cases for cand in c.candidates
    ]
    tfidf = TfidfSpace(corpus)

    targets = [
        (case, candidate)
        for case in cases
        for candidate in case.candidates
        if candidate.relevance <= WEAK_ENOUGH_TO_ATTACK
    ]
    if limit:
        targets = targets[:limit]

    rows: list[dict] = []
    for case, candidate in targets:
        payload = hidden_block(case.jd_text, skills)
        if not payload:
            continue

        lines = [ln for ln in candidate.resume_text.splitlines() if ln.strip()]
        honest_pdf = resume_pdf(lines, [])
        honest_raw, honest_visible = read_pdf(honest_pdf)

        honest = {
            "tfidf": tfidf.similarity(honest_raw, case.jd_text),
            "embedding": calculate_similarity(honest_raw, case.jd_text)["final_score"],
            "careeriq_raw": careeriq_score(honest_raw, case.jd_text, skills),
            "careeriq": careeriq_score(honest_visible, case.jd_text, skills),
        }

        for mode in ATTACK_MODES:
            raw, visible = read_pdf(resume_pdf(lines, payload, mode))
            attacked = {
                "tfidf": tfidf.similarity(raw, case.jd_text),
                "embedding": calculate_similarity(raw, case.jd_text)["final_score"],
                "careeriq_raw": careeriq_score(raw, case.jd_text, skills),
                "careeriq": careeriq_score(visible, case.jd_text, skills),
            }
            # A leak is a word the hidden block introduced that was not
            # already in the honest visible text — comparing against the
            # payload alone would count the candidate's own real skills.
            introduced = set(payload[0].lower().split()) - set(honest_visible.lower().split())
            leaked = sorted(introduced & set(visible.lower().split()))

            rows.append({
                "case": case.id,
                "candidate": candidate.id,
                "relevance": candidate.relevance,
                "mode": mode,
                "honest": {k: round(v, 2) for k, v in honest.items()},
                "attacked": {k: round(v, 2) for k, v in attacked.items()},
                "inflation": {k: round(attacked[k] - honest[k], 2) for k in honest},
                "leaked_terms": leaked,
                "hidden_recovered": bool(leaked),
            })

    scorers = ["tfidf", "embedding", "careeriq_raw", "careeriq"]
    summary = {
        "documents": len(rows),
        "by_scorer": {
            name: {
                "mean_inflation": round(statistics.mean(r["inflation"][name] for r in rows), 2),
                "max_inflation": round(max(r["inflation"][name] for r in rows), 2),
                "documents_moved": sum(1 for r in rows if abs(r["inflation"][name]) > 0.5),
            }
            for name in scorers
        } if rows else {},
        "by_mode": {
            mode: {
                name: round(
                    statistics.mean(
                        r["inflation"][name] for r in rows if r["mode"] == mode
                    ), 2
                )
                for name in scorers
            }
            for mode in ATTACK_MODES
        } if rows else {},
        "leaked_documents": sum(1 for r in rows if r["hidden_recovered"]),
    }
    return {"provenance": provenance(), "summary": summary, "rows": rows}


def render(report: dict) -> str:
    summary = report["summary"]
    lines = ["", "Hidden-text attack: score inflation", "=" * 68, ""]
    lines.append(f"  {summary['documents']} attacked documents, 3 modes each")
    lines.append("")
    lines.append(f"  {'SCORER':<16}{'MEAN':>10}{'WORST':>10}{'MOVED':>10}")
    lines.append("  " + "-" * 44)
    for name, stats in summary["by_scorer"].items():
        lines.append(
            f"  {name:<16}{stats['mean_inflation']:>+10.2f}{stats['max_inflation']:>+10.2f}"
            f"{stats['documents_moved']:>10}"
        )
    lines += ["", "  BY ATTACK MODE (mean inflation)", ""]
    header = f"  {'MODE':<14}" + "".join(f"{n:>15}" for n in summary["by_scorer"])
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for mode, scores in summary["by_mode"].items():
        lines.append(f"  {mode:<14}" + "".join(f"{v:>+15.2f}" for v in scores.values()))
    lines += [
        "",
        "  MEAN   — average points gained by hiding the JD's skill list in the PDF.",
        "  WORST  — the largest single gain.",
        "  MOVED  — documents whose score changed by more than half a point.",
        "",
        f"  Hidden text reached the scored input in {summary['leaked_documents']} of "
        f"{summary['documents']} documents after filtering.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="attack only the first N candidates")
    args = parser.parse_args()

    report = run(limit=args.limit)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
