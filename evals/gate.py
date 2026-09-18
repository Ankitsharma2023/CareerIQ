#!/usr/bin/env python3
"""Fail a pull request that makes the measured numbers worse.

The evaluation harness only helps if someone runs it. This runs it on every
change and compares against the committed baseline, so a regression is caught
by the pull request rather than noticed three commits later — or not at all.

    python evals/gate.py                      # compare against evals/baseline.json
    python evals/gate.py --update             # accept the current numbers as the baseline

Deliberately not gating on everything the harness reports. A gate that fires on
noise gets disabled within a week, and a disabled gate is worse than none
because it looks like coverage. Each metric below is one a human chose, with a
tolerance wide enough to absorb float and ordering noise and narrow enough to
catch a real drop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.dataset import load_cases  # noqa: E402
from evals.run import evaluate  # noqa: E402

BASELINE = Path(__file__).parent / "baseline.json"

#: metric path -> (direction, tolerance)
#:
#: "higher" means a drop of more than the tolerance fails; "lower" means a rise
#: does. Tolerances are absolute, in the metric's own units.
GATED: dict[str, tuple[str, float]] = {
    "ranking.ndcg@5": ("higher", 0.005),
    "ranking.spearman": ("higher", 0.010),
    "extraction.skill_f1": ("higher", 0.010),
    "extraction.missing_skill_f1": ("higher", 0.010),
    "parsing.overall_recall": ("higher", 0.010),
    # Lower is better here: this is the penalty for writing "K8s" instead of
    # "Kubernetes". It is also gated absolutely below.
    "equivalence.max_gap": ("lower", 0.50),
}

#: Absolute ceilings, independent of the baseline. These stop a slow drift that
#: never trips the per-change tolerance from arriving somewhere unacceptable.
CEILINGS: dict[str, float] = {
    # Matches ABBREVIATION_GATE in tests/test_invariance.py. Ratchet both
    # together, and record the new measured value in that file when you do.
    "equivalence.max_gap": 5.0,
}

FLOORS: dict[str, float] = {
    "ranking.ndcg@5": 0.95,
    "extraction.skill_f1": 0.85,
}


def reach(report: dict, path: str) -> float:
    value = report
    for part in path.split("."):
        value = value[part]
    return float(value)


def compare(current: dict, baseline: dict) -> list[str]:
    """Return one line per failure; empty means the gate passes."""
    failures: list[str] = []

    for path, (direction, tolerance) in GATED.items():
        try:
            now, before = reach(current, path), reach(baseline, path)
        except (KeyError, TypeError):
            failures.append(f"{path}: missing from the baseline or the current run")
            continue

        if direction == "higher" and now < before - tolerance:
            failures.append(
                f"{path}: {now:.4f} is below the baseline {before:.4f} "
                f"by more than {tolerance}"
            )
        elif direction == "lower" and now > before + tolerance:
            failures.append(
                f"{path}: {now:.4f} is above the baseline {before:.4f} "
                f"by more than {tolerance}"
            )

    for path, ceiling in CEILINGS.items():
        now = reach(current, path)
        if now > ceiling:
            failures.append(f"{path}: {now:.4f} exceeds the absolute ceiling {ceiling}")

    for path, floor in FLOORS.items():
        now = reach(current, path)
        if now < floor:
            failures.append(f"{path}: {now:.4f} is below the absolute floor {floor}")

    return failures


def render(current: dict, baseline: dict, failures: list[str]) -> str:
    lines = ["", "Evaluation gate", "=" * 62, ""]
    lines.append(f"  {'METRIC':<30}{'BASELINE':>10}{'NOW':>10}{'DELTA':>10}")
    lines.append("  " + "-" * 60)
    for path, (direction, _) in GATED.items():
        try:
            now, before = reach(current, path), reach(baseline, path)
        except (KeyError, TypeError):
            lines.append(f"  {path:<30}{'—':>10}{'—':>10}{'—':>10}")
            continue
        delta = now - before
        arrow = "" if abs(delta) < 1e-9 else (
            "  ok" if (delta > 0) == (direction == "higher") else "  worse"
        )
        lines.append(f"  {path:<30}{before:>10.4f}{now:>10.4f}{delta:>+10.4f}{arrow}")

    lines.append("")
    if failures:
        lines.append("  FAILED")
        lines += [f"    - {failure}" for failure in failures]
        lines += [
            "",
            "  If the change is intended and the new numbers are correct, re-cut",
            "  the baseline with `python evals/gate.py --update` and say in the",
            "  commit message what moved and why.",
        ]
    else:
        lines.append("  passed")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the baseline")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()

    current = evaluate(load_cases())

    if args.update:
        args.baseline.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"no baseline at {args.baseline}; run with --update", file=sys.stderr)
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    failures = compare(current, baseline)
    print(render(current, baseline, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
