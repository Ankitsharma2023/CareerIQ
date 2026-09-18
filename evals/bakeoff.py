#!/usr/bin/env python3
"""Compare candidate models on the evaluation set.

This is the tool that replaces "switch models until one returns parseable
output". It reports the metric that actually decides the choice — how often the
model returns schema-conformant JSON — alongside how well its scores track
human relevance grades.

    python evals/bakeoff.py --models deepseek/deepseek-chat google/gemini-flash-1.5
    python evals/bakeoff.py --models MODEL_A MODEL_B --limit 6 --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import metrics  # noqa: E402
from evals.dataset import EvalCase, load_cases  # noqa: E402
from evals.llm_score import eval_score  # noqa: E402


async def run_extraction(model: str, cases: list[EvalCase], limit: int) -> dict:
    """Compare a model's skill extraction against the deterministic baseline.

    The number this exists for is ``fabrication_rate``: the share of skills the
    model claimed whose supporting quote could not be found in the resume.
    Unlike agreement with human grades, it needs no labels and no annotator —
    the resume is the ground truth, and a quote either appears in it or does
    not. That makes it the cheapest honest signal for choosing a model, and the
    one a paper can report without a caveat about label quality.

    ``novel`` is the other half: skills the model verified that the taxonomy
    does not contain. That is what a model is for here, since the deterministic
    extractor cannot exceed its own vocabulary by construction.
    """
    from core import llm
    from services.skill_agent import SkillExtraction, build_prompt, check
    from services.skill_extractor import extract_skills_from_text, load_skills

    skills_list = load_skills()
    pairs = [
        candidate
        for case in cases
        for candidate in case.candidates
    ][:limit]

    verified_total = rejected_total = 0
    novel: set[str] = set()
    missed: set[str] = set()
    failures: list[str] = []

    for candidate in pairs:
        try:
            extraction = await llm.complete_json(
                build_prompt(candidate.resume_text),
                SkillExtraction,
                models=[model],
                max_tokens=1500,
                temperature=0.0,
                schema_name="skill_extraction",
            )
        except Exception as exc:
            failures.append(f"{candidate.id}: {type(exc).__name__}: {exc}"[:160])
            continue

        result = check(extraction, candidate.resume_text)
        verified_total += len(result.verified)
        rejected_total += len(result.rejected)

        deterministic = set(extract_skills_from_text(candidate.resume_text, skills_list))
        novel |= set(result.verified) - deterministic
        missed |= deterministic - set(result.verified)

    claimed = verified_total + rejected_total
    return {
        "model": model,
        "resumes": len(pairs),
        "claimed": claimed,
        "verified": verified_total,
        "rejected": rejected_total,
        "fabrication_rate": round(rejected_total / claimed, 4) if claimed else 0.0,
        "novel_skills": sorted(novel)[:25],
        "novel_count": len(novel),
        "missed_by_agent": sorted(missed)[:25],
        "missed_count": len(missed),
        "failures": failures[:3],
    }


def render_extraction(results: list[dict]) -> str:
    lines = ["", "Extraction bake-off: regex baseline vs agent", "=" * 78, ""]
    lines.append(
        f"{'MODEL':<34}{'CLAIMED':>9}{'VERIFIED':>10}{'FABRICATED':>12}{'NOVEL':>8}{'MISSED':>8}"
    )
    lines.append("-" * 78)
    for row in sorted(results, key=lambda r: r["fabrication_rate"]):
        lines.append(
            f"{row['model']:<34}{row['claimed']:>9}{row['verified']:>10}"
            f"{row['fabrication_rate']:>11.1%}{row['novel_count']:>8}{row['missed_count']:>8}"
        )
    lines += [
        "",
        "  FABRICATED — share of claimed skills whose quote is not in the resume.",
        "               Needs no labels: the resume is the ground truth.",
        "  NOVEL      — verified skills the taxonomy does not contain. This is what",
        "               a model is for; the regex baseline cannot exceed its own",
        "               vocabulary by construction.",
        "  MISSED     — skills the taxonomy found and the agent did not. The reason",
        "               the two are unioned rather than swapped.",
        "",
    ]
    for row in results:
        if row["novel_skills"]:
            lines.append(f"  {row['model']} found beyond the taxonomy:")
            lines.append(f"    {', '.join(row['novel_skills'])}")
            lines.append("")
    return "\n".join(lines)


async def run_model(model: str, cases: list[EvalCase], limit: int) -> dict:
    from core import llm
    from models.schemas import LLMEvaluation
    from services.llm_evaluator import build_prompt

    conformant = 0
    failures: list[str] = []
    latencies: list[float] = []
    scores: list[float] = []
    grades: list[int] = []

    pairs = [
        (case, candidate)
        for case in cases
        for candidate in case.candidates
    ][:limit]

    for case, candidate in pairs:
        started = time.perf_counter()
        try:
            result = await llm.complete_json(
                build_prompt(candidate.resume_text, case.jd_text),
                LLMEvaluation,
                models=[model],
                max_tokens=2000,
                schema_name="resume_evaluation",
            )
            latencies.append(time.perf_counter() - started)
            conformant += 1
            scores.append(eval_score(result))
            grades.append(candidate.relevance)
        except Exception as exc:
            failures.append(f"{case.id}/{candidate.id}: {type(exc).__name__}: {exc}"[:160])

    total = len(pairs)
    return {
        "model": model,
        "calls": total,
        "schema_conformance_rate": round(conformant / total, 4) if total else 0.0,
        "failures": len(failures),
        "failure_samples": failures[:3],
        "median_latency_s": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else None,
        "spearman_vs_human": round(metrics.spearman(scores, grades), 4),
        "mean_score": round(metrics.mean(scores), 2) if scores else None,
    }


def render(results: list[dict]) -> str:
    lines = ["", "Model bake-off", "=" * 78, ""]
    header = f"{'MODEL':<38} {'CONFORM':>8} {'FAILS':>6} {'LAT s':>7} {'SPEARMAN':>9}"
    lines.append(header)
    lines.append("-" * 78)

    for row in sorted(results, key=lambda r: (-r["schema_conformance_rate"], -r["spearman_vs_human"])):
        latency = f"{row['median_latency_s']:.2f}" if row["median_latency_s"] else "  —"
        lines.append(
            f"{row['model']:<38} "
            f"{row['schema_conformance_rate']:>7.1%} "
            f"{row['failures']:>6} "
            f"{latency:>7} "
            f"{row['spearman_vs_human']:>9.3f}"
        )

    lines += [
        "",
        "  CONFORM  — share of calls returning schema-valid JSON. Anything below",
        "             100% means the model is unreliable for this workload, no",
        "             matter how good its reasoning benchmarks look.",
        "  SPEARMAN — how well the model's own eval-only score tracks the human",
        "             relevance grades. Negative means it disagrees with you.",
        "",
    ]

    for row in results:
        if row["failure_samples"]:
            lines.append(f"  {row['model']} failures:")
            lines += [f"    {sample}" for sample in row["failure_samples"]]
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="OpenRouter model IDs")
    parser.add_argument("--limit", type=int, default=8, help="candidates per model")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--extraction", action="store_true",
        help="compare skill extraction and quote fidelity instead of fit scoring",
    )
    args = parser.parse_args()

    from core import llm

    if not llm.is_configured():
        print("OPENROUTER_API_KEY is not set — cannot run a bake-off.", file=sys.stderr)
        return 1

    cases = load_cases()
    if args.extraction:
        results = [
            asyncio.run(run_extraction(model, cases, args.limit)) for model in args.models
        ]
        print(json.dumps(results, indent=2) if args.json else render_extraction(results))
        return 0

    results = [asyncio.run(run_model(model, cases, args.limit)) for model in args.models]

    print(json.dumps(results, indent=2) if args.json else render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
