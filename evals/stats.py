"""Confidence intervals, significance tests and ranking statistics.

Written on numpy rather than scipy/sklearn, for the reason the rest of this
repo gives: those are large dependencies for a handful of formulas, and a
formula you can read is worth more in a paper than one behind an estimator with
defaults you have to look up.

Every function that resamples takes an explicit seed. An unseeded confidence
interval is not reproducible, and a reviewer who cannot regenerate the interval
has no reason to believe it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    low: float
    high: float
    resamples: int
    seed: int

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def to_dict(self) -> dict:
        return {
            "point": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "resamples": self.resamples,
            "seed": self.seed,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not silently bias the correlation."""
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    ranked[order] = np.arange(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranked[order[start:index]] = np.mean(np.arange(start, index))
            start = index
    return ranked


def spearman(scores, labels) -> float:
    """Spearman's rho, tie-corrected (Pearson correlation of average ranks)."""
    x, y = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denominator) if denominator else 0.0


def roc_auc(scores, positive) -> float:
    """Area under the ROC curve, as the Mann-Whitney U statistic.

    AUC is the probability that a randomly chosen positive outranks a randomly
    chosen negative, which is exactly U / (n_pos * n_neg) on average ranks —
    no curve or threshold sweep needed, and ties are handled correctly by the
    average-rank treatment.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(scores) + 1.0
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_ovr_auc(scores, labels) -> tuple[float, dict[int, float]]:
    """One-vs-rest AUC per label, and the unweighted mean of them."""
    labels = np.asarray(labels)
    per_class = {
        int(value): roc_auc(scores, labels == value) for value in sorted(set(labels.tolist()))
    }
    usable = [v for v in per_class.values() if not np.isnan(v)]
    return (float(np.mean(usable)) if usable else float("nan")), per_class


def pairwise_accuracy(scores, labels, groups=None) -> tuple[float, int]:
    """Fraction of label-differing pairs the scorer orders correctly.

    Ties in score count as half, which is what a coin flip would earn — scoring
    them as wins would reward a scorer that returns a constant.

    With ``groups``, only pairs inside the same group are compared. That is the
    decision a resume ranker actually makes: candidates for one opening, not
    candidates across different openings.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(groups) if groups is not None else np.zeros(len(scores))

    correct = 0.0
    total = 0
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        if len(index) < 2:
            continue
        s, y = scores[index], labels[index]
        label_diff = y[:, None] - y[None, :]
        score_diff = s[:, None] - s[None, :]
        upper = np.triu(np.ones_like(label_diff, dtype=bool), k=1)
        comparable = upper & (label_diff != 0)
        if not comparable.any():
            continue
        # A tie has sign 0 against a non-zero label difference, so it can
        # never land in `agree`; it is counted separately at half.
        agree = np.sign(label_diff[comparable]) == np.sign(score_diff[comparable])
        ties = score_diff[comparable] == 0
        correct += float(agree.sum()) + 0.5 * float(ties.sum())
        total += int(comparable.sum())

    return (correct / total if total else float("nan")), total


# ── Resampling ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    statistic,
    *arrays,
    resamples: int = 10_000,
    seed: int = 20260918,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap over rows, resampling every array together.

    Rows are resampled jointly so a pair stays a pair — resampling scores and
    labels independently would destroy the very association being estimated.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)

    point = float(statistic(*arrays))
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picks = rng.integers(0, n, size=n)
        draws[index] = statistic(*(a[picks] for a in arrays))

    tail = (1.0 - confidence) / 2.0
    low, high = np.nanpercentile(draws, [tail * 100, (1 - tail) * 100])
    return Interval(point, float(low), float(high), resamples, seed)


def paired_permutation_test(
    statistic,
    first,
    second,
    *shared,
    resamples: int = 10_000,
    seed: int = 20260918,
) -> dict:
    """Two-sided paired permutation test on the difference of two scorers.

    The pairing matters: both scorers saw the same documents, so the null is
    "which scorer produced which number for this document was arbitrary", and
    the permutation swaps the two scorers within a row rather than shuffling
    rows. Treating the samples as independent would overstate the variance and
    hide real differences.

    The p-value uses the (hits + 1) / (resamples + 1) form, which cannot return
    zero — a permutation test can only ever say "smaller than 1/(n+1)".
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    shared = [np.asarray(a) for a in shared]
    observed = float(statistic(first, *shared) - statistic(second, *shared))

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(resamples):
        swap = rng.random(len(first)) < 0.5
        a = np.where(swap, second, first)
        b = np.where(swap, first, second)
        if abs(float(statistic(a, *shared) - statistic(b, *shared))) >= abs(observed):
            hits += 1

    return {
        "difference": round(observed, 4),
        "p_value": round((hits + 1) / (resamples + 1), 5),
        "resamples": resamples,
        "seed": seed,
    }
