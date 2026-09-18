#!/usr/bin/env python3
"""Score a public, third-party-labelled dataset.

Everything the stress set can tell us is bounded by the fact that its labels
were written by the person who tuned the system. This is the same measurement
against labels nobody here produced.

The metric definitions are fixed in evals/PREREGISTRATION.md, which was
committed before this file was run. Read that first; this is its implementation
and deliberately contains no choices it does not already state.

    python evals/public_dataset.py --limit 100      # smoke test
    python evals/public_dataset.py                  # the full split
    python evals/public_dataset.py --json > public.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from evals import stats  # noqa: E402
from evals.tfidf import TfidfSpace  # noqa: E402

from services.evidence import (  # noqa: E402  isort:skip
    evidence_ratio,
    gather_evidence,
    unsupported_skills,
    weighted_coverage,
)
from services.experience_detector import detect_experience  # noqa: E402  isort:skip
from services.scoring import compute_fit  # noqa: E402  isort:skip
from services.signal_noise_analyzer import analyze_signal_to_noise  # noqa: E402  isort:skip
from services.similarity import calculate_similarity  # noqa: E402  isort:skip
from services.skill_extractor import extract_skills_from_text, load_skills  # noqa: E402  isort:skip

class _MemoizedEmbedding:
    """Embed each distinct string once.

    The split has 1,759 rows but only 477 distinct resumes and 71 distinct job
    descriptions, and a resume's sentence embeddings do not depend on the job
    it is being compared against. Embedding per row therefore repeats the same
    work 3.7 times over.

    This changes no number. It is a cache in front of the same model, keyed on
    the exact string, returning the same vectors the uncached path returns; it
    exists because the alternative is 40 minutes of recomputing identical
    embeddings. Verified by running a subsample both ways.
    """

    def __init__(self, inner):
        self._inner = inner
        self._cache: dict[str, object] = {}

    def embed(self, texts, **kwargs):
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing, **kwargs), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


DATASET = "cnamuangtoun/resume-job-description-fit"
SPLIT = "test.csv"

# Ordinal, per the pre-registration. The three labels describe degrees of one
# quantity and the system emits one continuous score.
LABEL_ORDER = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}

SCORERS = ("tfidf", "embedding", "coverage", "careeriq")
SEED = 20260918


def load_rows() -> tuple[list[dict], str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, SPLIT, repo_type="dataset")
    # The snapshot directory is the revision hash, so the exact bytes these
    # numbers came from are recoverable.
    revision = Path(path).parent.name
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, revision


def score_row(resume: str, jd: str, skills: list[str], tfidf: TfidfSpace) -> dict:
    """All four scorers, sharing the one expensive call between them."""
    similarity = calculate_similarity(resume, jd)["final_score"]
    evidence = gather_evidence(resume, skills)
    jd_skills = extract_skills_from_text(jd, skills)
    coverage = weighted_coverage(evidence, jd_skills)
    experience = detect_experience(resume, jd)

    fit = compute_fit(
        semantic_score=similarity,
        coverage_score=coverage,
        clarity_score=analyze_signal_to_noise(resume)["clarity_score"],
        detected_years=experience["detected_years"],
        required_years=experience["required_years"],
        jd_text=jd,
        evidence_ratio_value=evidence_ratio(evidence),
        unsupported=unsupported_skills(evidence),
    )
    return {
        "tfidf": tfidf.similarity(resume, jd),
        "embedding": similarity,
        "coverage": coverage,
        "careeriq": fit.overall,
    }


def run(
    limit: int | None,
    checkpoint: Path | None,
    progress_every: int = 25,
    cache_embeddings: bool = True,
    budget_seconds: float | None = None,
) -> dict:
    rows, revision = load_rows()
    subsampled = False

    if limit and limit < len(rows):
        # Per the pre-registration: a uniform random draw at a fixed seed, and
        # the report says it was a subsample. Not the first N, which would
        # inherit whatever order the file happens to be in.
        picks = np.random.default_rng(SEED).choice(len(rows), size=limit, replace=False)
        rows = [rows[i] for i in sorted(picks)]
        subsampled = True

    if cache_embeddings:
        from services import similarity as _similarity

        _similarity._model = _MemoizedEmbedding(_similarity.get_model())

    skills = load_skills()
    tfidf = TfidfSpace(
        [r["resume_text"] for r in rows] + [r["job_description_text"] for r in rows]
    )

    done: dict[int, dict] = {}
    if checkpoint and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves one truncated line. Losing that
                # row costs a second on the next pass; refusing to start costs
                # the whole checkpoint.
                continue
            done[record["index"]] = record
        print(f"resuming from {len(done)} scored rows", file=sys.stderr)

    handle = checkpoint.open("a", encoding="utf-8") if checkpoint else None
    started = time.perf_counter()
    try:
        for index, row in enumerate(rows):
            if index in done:
                continue
            record = {
                "index": index,
                "label": LABEL_ORDER[row["label"].strip()],
                "job": row["job_description_text"][:120],
                "scores": score_row(
                    row["resume_text"], row["job_description_text"], skills, tfidf
                ),
            }
            done[index] = record
            if handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            if progress_every and (len(done) % progress_every == 0):
                elapsed = time.perf_counter() - started
                print(
                    f"{len(done)}/{len(rows)} rows, {elapsed / 60:.1f} min elapsed",
                    file=sys.stderr,
                    flush=True,
                )
            # The shell this runs in has a wall-clock limit, so the job is
            # designed to stop cleanly and be re-entered against the same
            # checkpoint rather than be killed mid-write.
            if budget_seconds and (time.perf_counter() - started) > budget_seconds:
                print(
                    f"budget reached at {len(done)}/{len(rows)} rows; rerun to continue",
                    file=sys.stderr,
                    flush=True,
                )
                break
    finally:
        if handle:
            handle.close()

    records = [done[i] for i in sorted(done)]
    if len(records) < len(rows):
        return {"incomplete": True, "scored": len(records), "total": len(rows)}
    return summarize(records, revision, subsampled)


def summarize(records: list[dict], revision: str, subsampled: bool) -> dict:
    labels = np.array([r["label"] for r in records])
    groups = np.array([r["job"] for r in records])

    results = {}
    for name in SCORERS:
        scores = np.array([r["scores"][name] for r in records], dtype=np.float64)
        accuracy, pairs = stats.pairwise_accuracy(scores, labels, groups)
        rho = stats.bootstrap_ci(stats.spearman, scores, labels, resamples=10_000, seed=SEED)
        macro, per_class = stats.macro_ovr_auc(scores, labels)
        results[name] = {
            "pairwise_accuracy_within_job": round(accuracy, 4),
            "pairs_compared": pairs,
            "spearman": rho.to_dict(),
            "roc_auc_macro_ovr": round(macro, 4),
            "roc_auc_per_class": {
                str(k): round(v, 4) for k, v in per_class.items()
            },
            "roc_auc_good_fit_vs_rest": round(stats.roc_auc(scores, labels == 2), 4),
        }

    # Is the product's number better than the baseline it has to beat?
    comparisons = {}
    careeriq = np.array([r["scores"]["careeriq"] for r in records], dtype=np.float64)
    for name in ("tfidf", "embedding", "coverage"):
        other = np.array([r["scores"][name] for r in records], dtype=np.float64)
        comparisons[f"careeriq_vs_{name}"] = stats.paired_permutation_test(
            stats.spearman, careeriq, other, labels, resamples=10_000, seed=SEED
        )

    # ── Exploratory, added after the pre-registered result was in hand ──────
    #
    # Flagged as post-hoc because it is. The pre-registered pairwise accuracy
    # pools every pair, so a job with 89 rows contributes 3,916 pairs and a job
    # with 5 contributes 10 — the pooled figure is really a weighted average
    # dominated by a handful of large jobs. Both are legitimate; they answer
    # different questions ("over all comparisons" vs "over all jobs"), and they
    # disagree here, which is the interesting part. The primary metric remains
    # the pre-registered one.
    exploratory = {}
    for name in SCORERS:
        scores = np.array([r["scores"][name] for r in records], dtype=np.float64)
        per_job = []
        for group in set(groups.tolist()):
            mask = groups == group
            accuracy, pairs = stats.pairwise_accuracy(scores[mask], labels[mask], groups[mask])
            if pairs >= 10:
                per_job.append(accuracy)
        keep = labels != 1
        extremes, extreme_pairs = stats.pairwise_accuracy(
            scores[keep], labels[keep], groups[keep]
        )
        exploratory[name] = {
            "per_job_median_pairwise": round(float(np.median(per_job)), 4),
            "per_job_mean_pairwise": round(float(np.mean(per_job)), 4),
            "jobs_measured": len(per_job),
            "jobs_above_0.6": int(sum(1 for a in per_job if a > 0.6)),
            "jobs_below_0.4": int(sum(1 for a in per_job if a < 0.4)),
            "good_vs_no_fit_only_pairwise": round(extremes, 4),
            "good_vs_no_fit_pairs": extreme_pairs,
            "mean_score_by_label": {
                str(int(value)): round(float(scores[labels == value].mean()), 2)
                for value in sorted(set(labels.tolist()))
            },
        }

    return {
        "dataset": {
            "name": DATASET,
            "split": SPLIT,
            "revision": revision,
            "rows": len(records),
            "subsampled": subsampled,
            "jobs": int(len(set(groups.tolist()))),
            "label_counts": {
                str(value): int((labels == value).sum()) for value in sorted(set(labels.tolist()))
            },
        },
        "scorers": results,
        "comparisons": comparisons,
        "exploratory_post_hoc": exploratory,
    }


def render(report: dict) -> str:
    data = report["dataset"]
    lines = ["", "Public dataset evaluation", "=" * 76, ""]
    lines.append(f"  {data['name']} [{data['split']}]  revision {data['revision'][:12]}")
    lines.append(
        f"  {data['rows']} rows across {data['jobs']} job descriptions"
        + ("  (SUBSAMPLE)" if data["subsampled"] else "")
    )
    lines.append(f"  labels (0=No Fit, 1=Potential, 2=Good): {data['label_counts']}")
    lines.append("")
    lines.append("  PRIMARY")
    lines.append(f"  {'SCORER':<12}{'PAIRWISE':>10}{'SPEARMAN':>11}{'95% CI':>22}")
    lines.append("  " + "-" * 55)
    for name, result in report["scorers"].items():
        rho = result["spearman"]
        interval = f"[{rho['ci_low']:.4f}, {rho['ci_high']:.4f}]"
        lines.append(
            f"  {name:<12}{result['pairwise_accuracy_within_job']:>10.4f}"
            f"{rho['point']:>11.4f}"
            + f"{interval:>22}"
        )
    lines += ["", "  SECONDARY (ROC-AUC)", ""]
    lines.append(f"  {'SCORER':<12}{'MACRO OVR':>12}{'GOOD FIT':>12}")
    lines.append("  " + "-" * 36)
    for name, result in report["scorers"].items():
        lines.append(
            f"  {name:<12}{result['roc_auc_macro_ovr']:>12.4f}"
            f"{result['roc_auc_good_fit_vs_rest']:>12.4f}"
        )
    lines += ["", "  PAIRED PERMUTATION TESTS on Spearman (10,000 resamples)", ""]
    for name, test in report["comparisons"].items():
        verdict = "significant" if test["p_value"] < 0.05 else "not significant"
        lines.append(
            f"    {name:<26} diff {test['difference']:>+8.4f}   p = {test['p_value']:<8} {verdict}"
        )
    lines.append("")
    lines.append(f"  Pairs compared within job: {report['scorers']['careeriq']['pairs_compared']}")

    exploratory = report.get("exploratory_post_hoc")
    if exploratory:
        lines += ["", "  EXPLORATORY (post-hoc, not pre-registered)", ""]
        lines.append(
            f"  {'SCORER':<12}{'PER-JOB MED':>13}{'GOOD vs NO FIT':>16}{'MEAN 0/1/2':>22}"
        )
        lines.append("  " + "-" * 63)
        for name, values in exploratory.items():
            by_label = values["mean_score_by_label"]
            spread = " / ".join(f"{by_label[k]:.1f}" for k in sorted(by_label))
            lines.append(
                f"  {name:<12}{values['per_job_median_pairwise']:>13.4f}"
                f"{values['good_vs_no_fit_only_pairwise']:>16.4f}{spread:>22}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="uniform random subsample size")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="append each scored row here, and resume from it if interrupted",
    )
    parser.add_argument(
        "--budget-seconds", type=float, default=None,
        help="stop cleanly after this long; rerun with the same checkpoint to continue",
    )
    parser.add_argument(
        "--no-cache-embeddings", action="store_true",
        help="embed every row from scratch (same numbers, ~4x slower)",
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="report from an existing checkpoint without scoring anything",
    )
    args = parser.parse_args()

    if args.summarize_only:
        if not args.checkpoint or not args.checkpoint.exists():
            print("--summarize-only needs an existing --checkpoint", file=sys.stderr)
            return 1
        records = []
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        records.sort(key=lambda r: r["index"])
        report = summarize(records, load_rows()[1], subsampled=bool(args.limit))
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0

    report = run(
        args.limit,
        args.checkpoint,
        cache_embeddings=not args.no_cache_embeddings,
        budget_seconds=args.budget_seconds,
    )
    if report.get("incomplete"):
        print(json.dumps(report), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
