"""The human grading study's statistics and scoring path.

Item 11 needs three people, which is the one thing that cannot be written. What
can be written is everything around them — and it has to be right before anyone
spends hours grading, because a bug found afterwards means asking them again.

So the scoring path is exercised end to end here against simulated graders
whose agreement is known by construction.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from evals.agreement import krippendorff_alpha, percent_agreement
from evals.grading_study import GRADERS, GRADES, score

NAN = float("nan")


class TestKrippendorffAlpha:
    def test_total_agreement_is_one(self):
        ratings = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
        assert krippendorff_alpha(ratings, GRADES) == 1.0

    def test_random_grading_is_about_zero(self):
        rng = np.random.default_rng(3)
        ratings = rng.integers(0, 4, (300, 3)).astype(float)
        assert abs(krippendorff_alpha(ratings, GRADES)) < 0.1

    def test_systematic_disagreement_is_negative(self):
        """Two graders using the scale in opposite directions is worse than
        chance, and alpha has to say so rather than bottoming out at zero."""
        ratings = np.array([[0, 3], [3, 0], [0, 3], [3, 0]], dtype=float)
        assert krippendorff_alpha(ratings, GRADES) < 0

    def test_the_ordinal_metric_weights_distance(self):
        """A 0 against a 3 must cost more than a 0 against a 1. This is the
        reason for choosing alpha over Fleiss' kappa, so it is worth pinning."""
        near = np.array([[0, 1], [1, 2], [2, 3], [3, 3], [0, 0], [1, 1]], dtype=float)
        far = np.array([[0, 3], [1, 2], [2, 3], [3, 0], [0, 0], [1, 1]], dtype=float)
        assert krippendorff_alpha(near, GRADES) > krippendorff_alpha(far, GRADES)

    def test_a_missing_grade_does_not_invalidate_the_pair(self):
        ratings = np.array([[0, 0, NAN], [1, 1, 1], [2, NAN, 2], [3, 3, 3]], dtype=float)
        assert krippendorff_alpha(ratings, GRADES) == 1.0

    def test_a_pair_graded_once_is_excluded_not_counted_as_agreement(self):
        """One rating cannot agree with itself; counting it would inflate alpha
        exactly when graders are dropping out."""
        ratings = np.array([[0, NAN, NAN], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
        assert krippendorff_alpha(ratings, GRADES) == 1.0

    def test_a_single_grade_used_throughout_is_undefined(self):
        """Everyone said 2 for everything. No disagreement, no information."""
        assert np.isnan(krippendorff_alpha(np.full((10, 3), 2.0), GRADES))


class TestPercentAgreement:
    def test_exact_and_within_one_differ(self):
        ratings = np.array([[0, 0, 1], [1, 1, 1]], dtype=float)
        assert percent_agreement(ratings) == pytest.approx(4 / 6)
        assert percent_agreement(ratings, tolerance=1) == 1.0


def _write_study(tmp_path, grades_by_grader, careeriq_scores, dataset_labels):
    pair_ids = [f"p{i:03d}" for i in range(len(careeriq_scores))]
    with (tmp_path / "_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "dataset_index", "careeriq_score", "dataset_label"])
        for index, pair in enumerate(pair_ids):
            writer.writerow([pair, index, careeriq_scores[index], dataset_labels[index]])

    for grader, grades in grades_by_grader.items():
        with (tmp_path / f"grades_{grader}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_id", "job_description", "resume", "grade", "notes"])
            for pair, grade in zip(pair_ids, grades, strict=True):
                writer.writerow([pair, "jd text", "resume text", "" if grade is None else grade, ""])
    return pair_ids


class TestScoringReturnedPackets:
    def _simulate(self, n=40, seed=5, noise=0.0, system_follows=True):
        rng = np.random.default_rng(seed)
        truth = rng.integers(0, 4, n)
        graders = {}
        for name in GRADERS:
            jitter = rng.normal(0, noise, n) if noise else np.zeros(n)
            graders[name] = np.clip(np.round(truth + jitter), 0, 3).astype(int).tolist()
        scores = (
            (truth * 20 + rng.normal(0, 3, n)) if system_follows else rng.normal(50, 15, n)
        )
        return graders, scores.tolist(), truth.tolist()

    def test_perfect_graders_and_a_tracking_system(self, tmp_path):
        graders, scores, truth = self._simulate()
        _write_study(tmp_path, graders, scores, truth)
        report = score(tmp_path)

        assert report["pairs_graded"] == 40
        assert report["agreement"]["krippendorff_alpha_ordinal"] == 1.0
        assert report["careeriq_vs_mean_human"]["point"] > 0.9
        assert report["mean_human_ceiling"] == 1.0

    def test_a_system_that_tracks_nothing_is_reported_as_such(self, tmp_path):
        graders, scores, truth = self._simulate(system_follows=False)
        _write_study(tmp_path, graders, scores, truth)
        report = score(tmp_path)

        assert abs(report["careeriq_vs_mean_human"]["point"]) < 0.4
        assert not report["careeriq_vs_mean_human"]["ci_low"] > 0.4

    def test_noisy_graders_lower_the_ceiling_not_just_the_system(self, tmp_path):
        """The point of measuring the ceiling. If the graders only agree with
        each other at 0.6, a system at 0.5 is not failing by the same margin
        that 0.5 against a ceiling of 1.0 would be."""
        graders, scores, truth = self._simulate(noise=1.2, seed=11)
        _write_study(tmp_path, graders, scores, truth)
        report = score(tmp_path)

        assert report["agreement"]["krippendorff_alpha_ordinal"] < 0.9
        assert report["mean_human_ceiling"] < 1.0

    def test_blank_grades_are_skipped_rather_than_read_as_zero(self, tmp_path):
        """A blank means "could not decide". Reading it as 0 would turn every
        abstention into a strong negative judgement."""
        graders, scores, truth = self._simulate(n=30, seed=7)
        graders["a"] = [None] + graders["a"][1:]
        _write_study(tmp_path, graders, scores, truth)
        report = score(tmp_path)

        assert report["pairs_graded"] == 30
        assert report["agreement"]["krippendorff_alpha_ordinal"] == 1.0

    def test_two_returned_packets_are_enough(self, tmp_path):
        graders, scores, truth = self._simulate(n=25, seed=9)
        del graders["c"]
        _write_study(tmp_path, graders, scores, truth)
        report = score(tmp_path)

        assert report["graders"] == ["a", "b"]
        assert "error" not in report

    def test_one_returned_packet_is_refused(self, tmp_path):
        """Agreement needs at least two people; reporting alpha for one grader
        would be a number with no meaning."""
        graders, scores, truth = self._simulate(n=25, seed=13)
        graders = {"a": graders["a"]}
        _write_study(tmp_path, graders, scores, truth)
        assert "error" in score(tmp_path)


class TestThePacketsAreBlind:
    def test_no_score_or_label_reaches_a_grader(self, tmp_path):
        """The single thing that would invalidate the whole study."""
        from evals.grading_study import _packet_html

        rows = [(0, {"jd": "Backend engineer", "resume": "Python", "careeriq": 91.5,
                     "dataset_label": 2, "index": 7})]
        packet = _packet_html("a", rows)
        assert "91.5" not in packet
        assert "careeriq" not in packet.lower()
        assert "dataset_label" not in packet

    def test_each_grader_sees_a_different_order(self, tmp_path):
        """Otherwise drift in strictness over a session lines up across graders
        and is measured as agreement."""
        import random

        from evals.grading_study import SEED

        orders = []
        for grader in GRADERS:
            items = list(range(50))
            random.Random(SEED + ord(grader)).shuffle(items)
            orders.append(items)
        assert orders[0] != orders[1] != orders[2]
