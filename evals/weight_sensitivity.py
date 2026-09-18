#!/usr/bin/env python3
"""Does the answer depend on the weights, or on the pipeline?

compute_fit blends three components at 0.45 semantic / 0.40 coverage / 0.15
clarity. Those numbers were chosen by judgement. A reviewer's first question
about any such blend is whether the reported result is a property of the system
or a property of a lucky triple, and the only answer is to sweep them.

The sweep is exact, not approximate: the components are computed once per
candidate, then compute_fit itself is called for every grid point with the
module constants rebound. No copy of the blending formula lives here, so this
cannot drift from the product.

Two datasets, because they answer different questions:

  stress   nDCG@5 and the worst equivalence gap, on the hand-labelled set
  public   Spearman and within-job pairwise accuracy, on 1,759 third-party
           labels, reusing the scored checkpoint so nothing is re-embedded

    python evals/weight_sensitivity.py --checkpoint run.jsonl
    python evals/weight_sensitivity.py --checkpoint run.jsonl --json > sweep.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from evals import metrics, stats  # noqa: E402
from evals.dataset import load_cases  # noqa: E402
from evals.public_dataset import LABEL_ORDER  # noqa: E402

from services import scoring  # noqa: E402  isort:skip
from services.evidence import (  # noqa: E402  isort:skip
    evidence_ratio,
    gather_evidence,
    unsupported_skills,
    weighted_coverage,
)
from services.experience_detector import detect_experience  # noqa: E402  isort:skip
from services.jd_requirements import classify_jd_skills  # noqa: E402  isort:skip
from services.signal_noise_analyzer import analyze_signal_to_noise  # noqa: E402  isort:skip
from services.similarity import calculate_similarity  # noqa: E402  isort:skip
from services.skill_extractor import extract_skills_from_text, load_skills  # noqa: E402  isort:skip

STEP = 0.05


def grid() -> list[tuple[float, float, float]]:
    """Every (semantic, coverage, clarity) triple on a 0.05 simplex."""
    points = []
    steps = int(round(1.0 / STEP))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            semantic = round(i * STEP, 2)
            coverage = round(j * STEP, 2)
            points.append((semantic, coverage, round(1.0 - semantic - coverage, 2)))
    return points


def blend(components: list[dict], weights: tuple[float, float, float]) -> np.ndarray:
    """Re-score every cached candidate under one weight triple.

    The constants are rebound around the call rather than passed, because
    compute_fit reads them as module globals — which means this exercises the
    real function, clarity damping, level multiplier and all.
    """
    semantic, coverage, clarity = weights
    previous = (scoring.WEIGHT_SEMANTIC, scoring.WEIGHT_COVERAGE, scoring.WEIGHT_CLARITY)
    scoring.WEIGHT_SEMANTIC, scoring.WEIGHT_COVERAGE, scoring.WEIGHT_CLARITY = (
        semantic, coverage, clarity,
    )
    try:
        return np.array([scoring.compute_fit(**c).overall for c in components])
    finally:
        (
            scoring.WEIGHT_SEMANTIC,
            scoring.WEIGHT_COVERAGE,
            scoring.WEIGHT_CLARITY,
        ) = previous


# ── Component caches ──────────────────────────────────────────────────────────

def stress_components() -> tuple[list[dict], list[dict]]:
    """Components per candidate, plus the case structure the metrics need."""
    skills = load_skills()
    components: list[dict] = []
    cases: list[dict] = []

    for case in load_cases():
        jd_skills = extract_skills_from_text(case.jd_text, skills)
        index_range = []
        for candidate in case.candidates:
            evidence = gather_evidence(candidate.resume_text, skills)
            experience = detect_experience(candidate.resume_text, case.jd_text)
            index_range.append(len(components))
            components.append({
                "semantic_score": calculate_similarity(
                    candidate.resume_text, case.jd_text
                )["final_score"],
                "coverage_score": weighted_coverage(
                evidence, jd_skills, classify_jd_skills(case.jd_text, skills)
            ),
                "clarity_score": analyze_signal_to_noise(
                    candidate.resume_text
                )["clarity_score"],
                "detected_years": experience["detected_years"],
                "required_years": experience["required_years"],
                "jd_text": case.jd_text,
                "evidence_ratio_value": evidence_ratio(evidence),
                "unsupported": unsupported_skills(evidence),
            })
        cases.append({
            "id": case.id,
            "indices": index_range,
            "relevance": [c.relevance for c in case.candidates],
            "ids": [c.id for c in case.candidates],
            "ideal": case.ideal_order(),
            "pairs": [p for p in case.equivalent_pairs if len(p) == 2],
        })

    return components, cases


def public_components(
    checkpoint: Path, cache: Path | None = None
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Rebuild compute_fit's inputs for the public split without re-embedding.

    The checkpoint already holds the two expensive components — the embedding
    similarity and the weighted coverage. Clarity, level and evidence are
    regex and counting work, about two minutes for the whole split, so they
    are recomputed rather than stored.
    """
    if cache and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        return stored["components"], np.array(stored["labels"]), np.array(stored["groups"])

    from huggingface_hub import hf_hub_download

    scored: dict[int, dict] = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scored[record["index"]] = record

    path = hf_hub_download("cnamuangtoun/resume-job-description-fit", "test.csv",
                           repo_type="dataset")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    skills = load_skills()
    components, labels, groups = [], [], []
    for index in sorted(scored):
        row = rows[index]
        resume, jd = row["resume_text"], row["job_description_text"]
        evidence = gather_evidence(resume, skills)
        experience = detect_experience(resume, jd)
        components.append({
            "semantic_score": scored[index]["scores"]["embedding"],
            "coverage_score": scored[index]["scores"]["coverage"],
            "clarity_score": analyze_signal_to_noise(resume)["clarity_score"],
            "detected_years": experience["detected_years"],
            "required_years": experience["required_years"],
            "jd_text": jd,
            "evidence_ratio_value": evidence_ratio(evidence),
            "unsupported": unsupported_skills(evidence),
        })
        labels.append(LABEL_ORDER[row["label"].strip()])
        groups.append(jd[:120])

    if cache:
        cache.write_text(
            json.dumps({"components": components, "labels": labels, "groups": groups}),
            encoding="utf-8",
        )
    return components, np.array(labels), np.array(groups)


# ── Metrics per grid point ────────────────────────────────────────────────────

def stress_metrics(scores: np.ndarray, cases: list[dict]) -> dict:
    ndcgs, gaps = [], []
    for case in cases:
        case_scores = {
            case["ids"][i]: scores[index] for i, index in enumerate(case["indices"])
        }
        ranked = sorted(
            range(len(case["indices"])),
            key=lambda i: -scores[case["indices"][i]],
        )
        ndcgs.append(metrics.ndcg([case["relevance"][i] for i in ranked], k=5))
        for left, right in case["pairs"]:
            if left in case_scores and right in case_scores:
                gaps.append(abs(case_scores[left] - case_scores[right]))
    return {
        "ndcg@5": round(metrics.mean(ndcgs), 4),
        "worst_equivalence_gap": round(max(gaps, default=0.0), 2),
    }


def run(checkpoint: Path | None, cache: Path | None = None) -> dict:
    points = grid()
    components, cases = stress_components()

    stress_rows = []
    for weights in points:
        scores = blend(components, weights)
        stress_rows.append({"weights": list(weights), **stress_metrics(scores, cases)})

    public_rows = []
    if checkpoint and checkpoint.exists():
        pub_components, labels, groups = public_components(checkpoint, cache)
        for weights in points:
            scores = blend(pub_components, weights)
            accuracy, _ = stats.pairwise_accuracy(scores, labels, groups)
            public_rows.append({
                "weights": list(weights),
                "spearman": round(stats.spearman(scores, labels), 4),
                "pairwise": round(accuracy, 4),
            })

    shipped = (scoring.WEIGHT_SEMANTIC, scoring.WEIGHT_COVERAGE, scoring.WEIGHT_CLARITY)
    return {
        "shipped_weights": list(shipped),
        "grid_step": STEP,
        "points": len(points),
        "stress": stress_rows,
        "public": public_rows,
    }


# ── Rendering ─────────────────────────────────────────────────────────────────

def heatmap(rows: list[dict], key: str, title: str, shipped: list[float]) -> list[str]:
    """Semantic weight down the page, coverage across; clarity is the remainder."""
    by_point = {(r["weights"][0], r["weights"][1]): r[key] for r in rows}
    coverages = sorted({w for _, w in by_point})
    semantics = sorted({s for s, _ in by_point})

    values = list(by_point.values())
    best, worst = max(values), min(values)

    lines = ["", f"  {title}", ""]
    lines.append("       coverage " + "".join(f"{c:>6.2f}" for c in coverages))
    lines.append("  semantic     " + "-" * (6 * len(coverages)))
    for semantic in semantics:
        cells = []
        for coverage in coverages:
            value = by_point.get((semantic, coverage))
            if value is None:
                cells.append("     .")
                continue
            marker = "*" if [semantic, coverage, round(1 - semantic - coverage, 2)] == shipped else " "
            cells.append(f"{value:>5.2f}{marker}" if value >= 10 else f"{value:>5.3f}{marker}")
        lines.append(f"    {semantic:>5.2f} |" + "".join(cells))
    lines.append("")
    lines.append(f"    best {best:.4f}   worst {worst:.4f}   spread {best - worst:.4f}"
                 "   (* = shipped weights)")
    return lines


def render(report: dict) -> str:
    shipped = report["shipped_weights"]
    lines = ["", "Component weight sensitivity", "=" * 76, ""]
    lines.append(
        f"  shipped: semantic {shipped[0]}, coverage {shipped[1]}, clarity {shipped[2]}"
    )
    lines.append(f"  {report['points']} weight triples on a {report['grid_step']} simplex")

    lines += heatmap(report["stress"], "ndcg@5", "STRESS SET — nDCG@5", shipped)
    lines += heatmap(
        report["stress"], "worst_equivalence_gap",
        "STRESS SET — worst equivalence gap (lower is better)", shipped,
    )
    if report["public"]:
        lines += heatmap(report["public"], "spearman",
                         "PUBLIC DATASET — Spearman vs human label", shipped)
        lines += heatmap(report["public"], "pairwise",
                         "PUBLIC DATASET — within-job pairwise accuracy", shipped)

    for name, key, rows in (
        ("stress nDCG@5", "ndcg@5", report["stress"]),
        ("public Spearman", "spearman", report["public"]),
    ):
        if not rows:
            continue
        best = max(rows, key=lambda r: r[key])
        current = next(r for r in rows if r["weights"] == shipped)
        lines += [
            "",
            f"  {name}: shipped {current[key]:.4f} at {shipped}, "
            f"best {best[key]:.4f} at {best['weights']}",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a scored public-dataset checkpoint, to sweep that too")
    parser.add_argument("--component-cache", type=Path, default=None,
                        help="reuse the recomputed clarity/level/evidence components")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args.checkpoint, args.component_cache)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
