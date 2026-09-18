#!/usr/bin/env python3
"""The human grading study: sampling, blind packets, and scoring.

Item 11 of the v2 upgrade plan is the one piece of this work that cannot be
written. It needs three people to read resumes. Everything around that is here,
so the only remaining step is finding them.

    # once, to build the packets
    python evals/grading_study.py prepare --checkpoint run.jsonl --out study/

    # after the graders return their CSVs
    python evals/grading_study.py score --dir study/

Why this exists when the public dataset already has labels: those labels come
with no rubric, no record of who produced them and no measure of how much the
labellers agreed. A correlation against labels of unknown reliability has an
unknown ceiling — if the labels themselves are noisy, a low correlation may be
the noise rather than the system. Grading a sample locally, with a stated
rubric and a measured agreement figure, gives that ceiling a number.

Design decisions worth stating because they are what make the result usable:

* **Graders never see a CareerIQ score.** The packets contain the resume, the
  job description and an empty grade column. Nothing else.
* **Each grader gets the pairs in a different order**, from a different seed,
  so a drift in strictness over a session does not correlate across graders and
  inflate agreement.
* **The sample is stratified by CareerIQ score decile**, so the study covers the
  range rather than clustering wherever the system happens to put most resumes.
  Stratifying on the system's own output is a deliberate choice: the question
  is whether the score tracks human judgement across its range, which a sample
  concentrated in one decile could not answer.
* **A held-out human is scored the same way as the system.** Each grader is
  correlated against the mean of the other two. That is the ceiling: if three
  people agree with each other at 0.6, a system scoring 0.5 is doing well, and
  one scoring 0.5 against a ceiling of 0.95 is not. Reporting the system's
  correlation without that number is the most common way this measurement gets
  overstated.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from evals import stats  # noqa: E402
from evals.agreement import krippendorff_alpha, percent_agreement  # noqa: E402
from evals.public_dataset import LABEL_ORDER  # noqa: E402

SEED = 20260918
DEFAULT_PAIRS = 50
GRADERS = ("a", "b", "c")
GRADES = [0, 1, 2, 3]

RUBRIC = """# Grading guide

You are deciding, for each pair, how well this resume fits this job.

Grade on 0-3. The question is **"would I move this candidate to a first
interview for this specific role?"** — not whether the resume is well written,
and not whether the person seems capable in general.

| Grade | Meaning |
|---|---|
| **3** | Strong fit. Clearly meets the core requirements, with evidence. I would interview. |
| **2** | Good fit. Meets most core requirements; the gaps are learnable on the job. Probably interview. |
| **1** | Stretch. Some relevant ground, but missing something central. Interview only in a thin pool. |
| **0** | Not a fit. Wrong field, wrong level, or the core requirements are absent. |

Rules that keep the grades comparable across graders:

1. **Judge against the job description in front of you**, not against the other
   resumes in the packet. Two resumes can both be 3.
2. **Ignore formatting, typos and writing quality** unless they make the resume
   impossible to understand. A badly written resume from someone who has done
   the work is still a 3.
3. **Do not assume unstated skills.** If the role requires Kubernetes and the
   resume never mentions it, that is a gap, even if the candidate clearly could
   learn it.
4. **Seniority counts.** A principal engineer applying to a junior role is a
   mismatch, not a 3. So is the reverse.
5. **If you are torn between two grades, pick the lower one** and say why in the
   notes column. Consistency matters more than precision here.
6. **Grade every pair, in the order given.** If you genuinely cannot decide,
   leave it blank rather than guessing — a blank is handled, a coin flip is not.

Do not discuss pairs with the other graders until everyone has finished. The
measurement being taken is how much independent readers agree; comparing notes
first destroys it.

Fill in the `grade` column of your CSV and send it back unchanged otherwise.
"""


def _rows_with_scores(checkpoint: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download

    scored: dict[int, dict] = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scored[record["index"]] = record

    path = hf_hub_download(
        "cnamuangtoun/resume-job-description-fit", "test.csv", repo_type="dataset"
    )
    with open(path, encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))

    return [
        {
            "index": index,
            "resume": source[index]["resume_text"],
            "jd": source[index]["job_description_text"],
            "dataset_label": LABEL_ORDER[source[index]["label"].strip()],
            "careeriq": scored[index]["scores"]["careeriq"],
        }
        for index in sorted(scored)
    ]


def prepare(checkpoint: Path, out: Path, count: int) -> None:
    rows = _rows_with_scores(checkpoint)
    rng = random.Random(SEED)

    # Stratify by CareerIQ decile so the study spans the score range.
    by_score = sorted(rows, key=lambda r: r["careeriq"])
    buckets = [by_score[i::10] for i in range(10)]
    sample: list[dict] = []
    per_bucket = max(1, count // 10)
    for bucket in buckets:
        sample.extend(rng.sample(bucket, min(per_bucket, len(bucket))))
    sample = sample[:count]

    out.mkdir(parents=True, exist_ok=True)
    (out / "GRADING_GUIDE.md").write_text(RUBRIC, encoding="utf-8")

    # The key never goes to a grader.
    with (out / "_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "dataset_index", "careeriq_score", "dataset_label"])
        for position, row in enumerate(sample):
            writer.writerow([f"p{position:03d}", row["index"], row["careeriq"], row["dataset_label"]])

    for grader in GRADERS:
        ordered = list(enumerate(sample))
        # A different order per grader, so drift in strictness over a session
        # does not line up across graders and look like agreement.
        random.Random(SEED + ord(grader)).shuffle(ordered)

        with (out / f"grades_{grader}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_id", "job_description", "resume", "grade", "notes"])
            for position, row in ordered:
                writer.writerow([f"p{position:03d}", row["jd"], row["resume"], "", ""])

        (out / f"packet_{grader}.html").write_text(
            _packet_html(grader, ordered), encoding="utf-8"
        )

    print(f"{len(sample)} pairs, {len(GRADERS)} packets written to {out}/")
    print("  grades_*.csv    fill in the grade column (0-3)")
    print("  packet_*.html   the same pairs, laid out for reading")
    print("  _key.csv        CareerIQ scores and dataset labels — do not share")


def _packet_html(grader: str, ordered: list[tuple[int, dict]]) -> str:
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Grading packet {grader.upper()}</title>",
        "<style>body{font:15px/1.6 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;color:#1a1d2e}h2{margin-top:3rem;border-top:2px solid #e8eaf0;"
        "padding-top:1.5rem}.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}"
        "pre{white-space:pre-wrap;font:13px/1.55 ui-monospace,monospace;background:#f8f9fc;"
        "border:1px solid #f0f1f5;border-radius:8px;padding:1rem;max-height:30rem;"
        "overflow:auto}h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;"
        "color:#9ca3af}</style>",
        f"<h1>Grading packet {grader.upper()}</h1>",
        "<p>Read the guide first. Record grades in "
        f"<code>grades_{grader}.csv</code> — this page is for reading only.</p>",
    ]
    for position, row in ordered:
        parts.append(f"<h2>p{position:03d}</h2><div class='cols'>")
        parts.append(f"<div><h3>Job description</h3><pre>{html.escape(row['jd'])}</pre></div>")
        parts.append(f"<div><h3>Resume</h3><pre>{html.escape(row['resume'])}</pre></div>")
        parts.append("</div>")
    return "\n".join(parts)


def score(directory: Path) -> dict:
    key: dict[str, dict] = {}
    with (directory / "_key.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key[row["pair_id"]] = {
                "careeriq": float(row["careeriq_score"]),
                "dataset_label": int(row["dataset_label"]),
            }

    grades: dict[str, dict[str, float]] = {}
    for grader in GRADERS:
        path = directory / f"grades_{grader}.csv"
        if not path.exists():
            continue
        collected: dict[str, float] = {}
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("grade") or "").strip()
                if raw:
                    collected[row["pair_id"]] = float(raw)
        if collected:
            grades[grader] = collected

    if len(grades) < 2:
        return {"error": f"need at least two returned packets, found {len(grades)}"}

    pair_ids = sorted(key)
    present = [g for g in GRADERS if g in grades]
    matrix = np.array(
        [[grades[g].get(pid, float("nan")) for g in present] for pid in pair_ids]
    )

    graded = ~np.all(np.isnan(matrix), axis=1)
    mean_grade = np.nanmean(matrix[graded], axis=1)
    careeriq = np.array([key[pid]["careeriq"] for pid in pair_ids])[graded]
    dataset_label = np.array([key[pid]["dataset_label"] for pid in pair_ids])[graded]

    # The ceiling: how well does one human predict the other two?
    ceiling = {}
    for position, grader in enumerate(present):
        others = [p for p in range(len(present)) if p != position]
        if not others:
            continue
        mine = matrix[graded][:, position]
        theirs = np.nanmean(matrix[graded][:, others], axis=1)
        usable = ~np.isnan(mine) & ~np.isnan(theirs)
        if usable.sum() > 2:
            ceiling[grader] = round(stats.spearman(mine[usable], theirs[usable]), 4)

    system = stats.bootstrap_ci(
        stats.spearman, careeriq, mean_grade, resamples=10_000, seed=SEED
    )

    return {
        "pairs_graded": int(graded.sum()),
        "graders": present,
        "agreement": {
            "krippendorff_alpha_ordinal": round(krippendorff_alpha(matrix, GRADES), 4),
            "exact_agreement": round(percent_agreement(matrix), 4),
            "within_one_grade": round(percent_agreement(matrix, tolerance=1), 4),
        },
        "careeriq_vs_mean_human": system.to_dict(),
        "human_ceiling_spearman": ceiling,
        "mean_human_ceiling": (
            round(float(np.mean(list(ceiling.values()))), 4) if ceiling else None
        ),
        "careeriq_vs_dataset_label": round(stats.spearman(careeriq, dataset_label), 4),
        "mean_human_vs_dataset_label": round(stats.spearman(mean_grade, dataset_label), 4),
    }


def render(report: dict) -> str:
    if "error" in report:
        return f"\n  {report['error']}\n"

    agreement = report["agreement"]
    alpha = agreement["krippendorff_alpha_ordinal"]
    verdict = (
        "reliable" if alpha >= 0.80
        else "tentative conclusions only" if alpha >= 0.667
        else "too noisy to conclude from"
    )
    system = report["careeriq_vs_mean_human"]
    ceiling = report["mean_human_ceiling"]

    lines = [
        "", "Human grading study", "=" * 66, "",
        f"  {report['pairs_graded']} pairs, graders: {', '.join(report['graders'])}",
        "",
        "  AGREEMENT BETWEEN GRADERS",
        f"    Krippendorff alpha (ordinal)   {alpha:.4f}   {verdict}",
        f"    exact agreement                {agreement['exact_agreement']:.1%}",
        f"    within one grade               {agreement['within_one_grade']:.1%}",
        "",
        "  CORRELATION WITH THE MEAN HUMAN GRADE",
        f"    CareerIQ                       {system['point']:.4f}"
        f"   [{system['ci_low']:.4f}, {system['ci_high']:.4f}]",
    ]
    for grader, value in report["human_ceiling_spearman"].items():
        lines.append(f"    grader {grader} vs the other two    {value:.4f}")
    if ceiling is not None:
        lines.append(f"    human ceiling (mean)           {ceiling:.4f}")
        if ceiling > 0:
            lines += [
                "",
                f"    CareerIQ reaches {system['point'] / ceiling:.0%} of the ceiling set",
                "    by how well one grader predicts the other two.",
            ]
    lines += [
        "",
        "  CROSS-CHECK AGAINST THE PUBLIC DATASET LABELS",
        f"    mean human grade vs dataset    {report['mean_human_vs_dataset_label']:.4f}",
        f"    CareerIQ vs dataset            {report['careeriq_vs_dataset_label']:.4f}",
        "",
        "    If the graders disagree with the dataset's own labels about as much",
        "    as CareerIQ does, the dataset's labels are the noisy part and the",
        "    public-split result understates the system. If the graders agree with",
        "    the dataset and CareerIQ does not, they do not.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("prepare", help="sample pairs and write the blind packets")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--out", type=Path, default=Path("study"))
    build.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)

    report = sub.add_parser("score", help="score the returned packets")
    report.add_argument("--dir", type=Path, default=Path("study"))
    report.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.checkpoint, args.out, args.pairs)
        return 0

    result = score(args.dir)
    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
