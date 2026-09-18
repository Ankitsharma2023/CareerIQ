"""Inter-annotator agreement for ordinal grades.

Krippendorff's alpha rather than Fleiss' kappa, for two reasons that matter
here. It handles an ordinal scale, so a 0 graded against a 3 counts as worse
disagreement than a 0 against a 1 — which is the whole point of a 0-3 fit
scale. And it tolerates missing values, so an annotator who skips a pair does
not invalidate that pair for everyone else.

Definition used (Krippendorff 2004), stated so it can be checked:

    alpha = 1 - D_o / D_e

    D_o = (1/n) * sum_ck  o_ck * delta(c,k)^2
    D_e = (1/(n(n-1))) * sum_ck  n_c * n_k * delta(c,k)^2

where ``o`` is the coincidence matrix — each unit with m ratings contributes
1/(m-1) to every ordered pair of distinct raters — ``n_c`` are its marginals,
``n`` their total, and for the ordinal metric

    delta(c,k)^2 = ( sum_{g=c..k} n_g  -  (n_c + n_k)/2 )^2

Interpretation, by the usual convention: 0.80 and above is reliable, 0.667 to
0.80 supports tentative conclusions only, below that the labels are too noisy
to draw conclusions from at all. Reported as measured either way.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def coincidence_matrix(ratings: np.ndarray, values: list[int]) -> np.ndarray:
    """``ratings`` is units x raters, with NaN for a missing grade."""
    index = {value: position for position, value in enumerate(values)}
    matrix = np.zeros((len(values), len(values)), dtype=np.float64)

    for unit in ratings:
        present = [v for v in unit if not (isinstance(v, float) and math.isnan(v))]
        if len(present) < 2:
            # A unit graded once tells us nothing about agreement; it is
            # excluded rather than counted as perfect agreement with itself.
            continue
        weight = 1.0 / (len(present) - 1)
        for first, second in itertools.permutations(present, 2):
            matrix[index[int(first)], index[int(second)]] += weight

    return matrix


def _ordinal_delta_squared(values: list[int], marginals: np.ndarray) -> np.ndarray:
    """Distance between two grades, scaled by how the scale is actually used."""
    size = len(values)
    delta = np.zeros((size, size), dtype=np.float64)
    for c in range(size):
        for k in range(size):
            low, high = min(c, k), max(c, k)
            total = marginals[low:high + 1].sum()
            delta[c, k] = (total - (marginals[c] + marginals[k]) / 2.0) ** 2
    return delta


def krippendorff_alpha(ratings: np.ndarray, values: list[int] | None = None) -> float:
    """Ordinal alpha. 1.0 is perfect agreement, 0.0 is chance, negative is worse."""
    ratings = np.asarray(ratings, dtype=np.float64)
    if values is None:
        observed = ratings[~np.isnan(ratings)]
        values = sorted({int(v) for v in observed})
    if len(values) < 2:
        # Everyone used one grade for everything. There is no disagreement and
        # also no information; alpha is undefined rather than perfect.
        return float("nan")

    matrix = coincidence_matrix(ratings, values)
    marginals = matrix.sum(axis=1)
    total = marginals.sum()
    if total < 2:
        return float("nan")

    delta = _ordinal_delta_squared(values, marginals)

    observed_disagreement = (matrix * delta).sum() / total
    expected_disagreement = (
        np.outer(marginals, marginals) - np.diag(marginals)
    )
    expected_disagreement = (expected_disagreement * delta).sum() / (total * (total - 1))

    if expected_disagreement == 0:
        return float("nan")
    return float(1.0 - observed_disagreement / expected_disagreement)


def percent_agreement(ratings: np.ndarray, tolerance: int = 0) -> float:
    """Share of rater pairs within ``tolerance`` grades of each other.

    Reported alongside alpha because it is the number a non-statistician can
    check by eye, and because "they agreed exactly 41% of the time but were
    within one grade 92% of the time" says something alpha alone does not.
    """
    ratings = np.asarray(ratings, dtype=np.float64)
    hits = considered = 0
    for unit in ratings:
        present = [v for v in unit if not math.isnan(v)]
        for first, second in itertools.combinations(present, 2):
            considered += 1
            if abs(first - second) <= tolerance:
                hits += 1
    return hits / considered if considered else float("nan")
