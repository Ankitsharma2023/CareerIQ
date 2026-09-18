#!/usr/bin/env python3
"""Turn the score into a statement about outcomes, where one is warranted.

"You scored 62" means nothing on its own. "Resumes scoring 60-69 were rated a
good fit by human labellers 25% of the time" means something — but only if the
25% is distinguishable from the rate across all resumes, and only if the band
holds enough documents to say so.

Both conditions fail more often than they hold for this scorer, which is why
this module reports whether a band is informative rather than assuming it is.
Printing "25% were a good fit" beside a base rate of 26%, with no interval,
would dress up noise as a finding — and a user reading it would reasonably
conclude the score predicts their outcome when the data says it barely does.

    python evals/calibration.py --checkpoint run.jsonl --out backend/data/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from evals.provenance import provenance  # noqa: E402
from evals.stats import wilson_interval  # noqa: E402

BAND_WIDTH = 10
GOOD_FIT = 2

#: Below this a band cannot support a claim about anything, whatever its rate.
MIN_BAND_SIZE = 30


def load_scores(checkpoint: Path, scorer: str = "careeriq") -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        scores.append(record["scores"][scorer])
        labels.append(record["label"])
    return np.array(scores, dtype=float), np.array(labels, dtype=int)


def build(scores: np.ndarray, labels: np.ndarray) -> dict:
    base_rate = float((labels == GOOD_FIT).mean())
    bands = []

    for low in range(0, 100, BAND_WIDTH):
        high = low + BAND_WIDTH
        in_band = (scores >= low) & (scores < high) if high < 100 else (scores >= low)
        total = int(in_band.sum())
        if total == 0:
            continue

        good = int((labels[in_band] == GOOD_FIT).sum())
        rate = good / total
        ci_low, ci_high = wilson_interval(good, total)

        # A band says something only when its interval excludes the rate across
        # every resume. Otherwise the honest report is that this band is not
        # distinguishable from the population, which is a fact about the
        # scorer, not a gap in the data.
        informative = total >= MIN_BAND_SIZE and not (ci_low <= base_rate <= ci_high)

        bands.append({
            "low": low,
            "high": high if high < 100 else 100,
            "n": total,
            "good_fit_rate": round(rate, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "informative": informative,
            "direction": "above" if rate > base_rate else "below",
        })

    return {
        "provenance": provenance(),
        "base_rate": round(base_rate, 4),
        "documents": int(len(scores)),
        "observed_range": [round(float(scores.min()), 1), round(float(scores.max()), 1)],
        "band_width": BAND_WIDTH,
        "min_band_size": MIN_BAND_SIZE,
        "informative_bands": sum(1 for b in bands if b["informative"]),
        "bands": bands,
    }


def render(table: dict) -> str:
    lines = [
        "", "Score calibration", "=" * 64, "",
        f"  {table['documents']} labelled documents, "
        f"{table['base_rate']:.1%} rated a good fit overall",
        f"  observed score range {table['observed_range'][0]} to {table['observed_range'][1]}",
        "",
        f"  {'BAND':<10}{'N':>6}{'GOOD FIT':>10}{'95% CI':>18}   SAYS SOMETHING?",
        "  " + "-" * 62,
    ]
    for band in table["bands"]:
        interval = f"[{band['ci_low']:.1%}, {band['ci_high']:.1%}]"
        verdict = "yes" if band["informative"] else "no — overlaps the base rate"
        lines.append(
            f"  {band['low']}-{band['high'] - 1:<6}{band['n']:>6}"
            f"{band['good_fit_rate']:>10.1%}{interval:>18}   {verdict}"
        )
    lines += [
        "",
        f"  {table['informative_bands']} of {len(table['bands'])} bands are "
        "distinguishable from the base rate.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="write the table as JSON")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    scores, labels = load_scores(args.checkpoint)
    table = build(scores, labels)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        print(f"calibration written to {args.out}", file=sys.stderr)

    print(json.dumps(table, indent=2) if args.json else render(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
