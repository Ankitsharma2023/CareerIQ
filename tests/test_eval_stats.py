"""The statistics the paper's claims rest on.

Each function is checked against a case whose answer is known by construction,
because a confidence interval or a p-value that is quietly wrong is worse than
none: it is a wrong claim with an interval attached.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evals.stats import (
    bootstrap_ci,
    macro_ovr_auc,
    paired_permutation_test,
    pairwise_accuracy,
    roc_auc,
    spearman,
)


class TestSpearman:
    def test_perfect_agreement_is_one(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0

    def test_perfect_disagreement_is_minus_one(self):
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0

    def test_it_measures_rank_not_value(self):
        """The point of Spearman: a monotone transform must not change it."""
        scores = [1, 2, 3, 4, 5]
        labels = [0, 1, 1, 2, 2]
        assert spearman(scores, labels) == spearman([math.exp(s) for s in scores], labels)

    def test_ties_are_averaged_not_broken_by_order(self):
        assert spearman([1, 1, 2, 2], [0, 0, 1, 1]) == 1.0
        assert spearman([1, 1, 2, 2], [0, 1, 0, 1]) == 0.0

    def test_a_constant_scorer_correlates_with_nothing(self):
        assert spearman([5, 5, 5, 5], [0, 1, 2, 3]) == 0.0


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc([1, 2, 3, 4], [False, False, True, True]) == 1.0

    def test_inverted_separation(self):
        assert roc_auc([4, 3, 2, 1], [False, False, True, True]) == 0.0

    def test_all_tied_is_chance(self):
        assert roc_auc([1, 1, 1, 1], [True, False, True, False]) == 0.5

    def test_it_equals_the_probability_a_positive_outranks_a_negative(self):
        rng = np.random.default_rng(7)
        scores = rng.normal(size=400)
        positive = rng.random(400) < 0.4
        wins = sum(
            1.0 if p > n else 0.5 if p == n else 0.0
            for p in scores[positive]
            for n in scores[~positive]
        )
        expected = wins / (positive.sum() * (~positive).sum())
        assert roc_auc(scores, positive) == pytest.approx(expected, abs=1e-9)

    def test_one_class_only_is_undefined_rather_than_zero(self):
        assert math.isnan(roc_auc([1, 2, 3], [True, True, True]))

    def test_macro_averages_each_label_against_the_rest(self):
        macro, per_class = macro_ovr_auc([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])
        assert per_class == {0: 0.0, 1: 0.5, 2: 1.0}
        assert macro == pytest.approx(0.5)


class TestPairwiseAccuracy:
    def test_correct_ordering_scores_one(self):
        accuracy, pairs = pairwise_accuracy([10, 20, 30], [0, 1, 2])
        assert (accuracy, pairs) == (1.0, 3)

    def test_reversed_ordering_scores_zero(self):
        assert pairwise_accuracy([30, 20, 10], [0, 1, 2])[0] == 0.0

    def test_a_constant_scorer_gets_exactly_chance(self):
        """Ties must count as half. Counting them as wins would make a scorer
        that returns 50 for everyone look perfect."""
        assert pairwise_accuracy([50, 50, 50], [0, 1, 2])[0] == 0.5

    def test_equal_labels_are_not_compared(self):
        _, pairs = pairwise_accuracy([10, 20, 30], [1, 1, 1])
        assert pairs == 0

    def test_only_pairs_within_a_group_are_compared(self):
        """Across two jobs the scorer is right within each and wrong between —
        and between is a comparison the product never makes."""
        scores = [1, 2, 90, 80]
        labels = [0, 1, 0, 1]
        groups = ["job-a", "job-a", "job-b", "job-b"]
        within, pairs = pairwise_accuracy(scores, labels, groups)
        assert pairs == 2
        assert within == 0.5
        assert pairwise_accuracy(scores, labels)[1] == 4, "ungrouped compares everything"


class TestBootstrap:
    def test_the_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        x = np.arange(200.0)
        y = x + rng.normal(0, 30, 200)
        interval = bootstrap_ci(spearman, x, y, resamples=400, seed=11)
        assert interval.low <= interval.point <= interval.high

    def test_it_is_reproducible_from_the_seed(self):
        x, y = np.arange(60.0), np.arange(60.0) ** 1.5
        first = bootstrap_ci(spearman, x, y, resamples=200, seed=42)
        second = bootstrap_ci(spearman, x, y, resamples=200, seed=42)
        assert (first.low, first.high) == (second.low, second.high)

    def test_pairs_are_resampled_together(self):
        """Resampling the two arrays independently would destroy the very
        association being estimated, and the interval would straddle zero."""
        x = np.arange(150.0)
        interval = bootstrap_ci(spearman, x, x.copy(), resamples=300, seed=5)
        assert interval.low > 0.99

    def test_no_association_gives_an_interval_containing_zero(self):
        rng = np.random.default_rng(9)
        interval = bootstrap_ci(
            spearman, rng.normal(size=300), rng.normal(size=300), resamples=400, seed=2
        )
        assert not interval.excludes_zero()


class TestPairedPermutation:
    def test_a_real_difference_is_significant(self):
        rng = np.random.default_rng(13)
        labels = rng.integers(0, 3, 300)
        good = labels + rng.normal(0, 0.3, 300)
        bad = rng.normal(0, 1, 300)
        result = paired_permutation_test(spearman, good, bad, labels, resamples=400, seed=4)
        assert result["p_value"] < 0.01
        assert result["difference"] > 0

    def test_two_scorers_of_equal_quality_are_not(self):
        rng = np.random.default_rng(17)
        labels = rng.integers(0, 3, 300)
        first = labels + rng.normal(0, 1.0, 300)
        second = labels + rng.normal(0, 1.0, 300)
        result = paired_permutation_test(spearman, first, second, labels, resamples=400, seed=6)
        assert result["p_value"] > 0.05

    def test_the_p_value_can_never_be_zero(self):
        """A permutation test can only say "below 1/(n+1)"; reporting 0 would
        claim an impossibility."""
        labels = np.arange(100.0)
        result = paired_permutation_test(
            spearman, labels, -labels, labels, resamples=50, seed=1
        )
        assert result["p_value"] > 0
        assert result["p_value"] == pytest.approx(1 / 51, abs=1e-4)
