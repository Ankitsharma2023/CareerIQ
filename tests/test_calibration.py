"""The score, translated into a statement about outcomes — where one is warranted.

Item 19 of the v2 upgrade plan. "You scored 62" is not information. What
happened to other resumes that scored 62 is information, and it is answerable
because the public split carries 1,759 independent human labels.

The answer is mostly "nothing distinguishable", and the tests that matter here
are the ones asserting the system says so instead of dressing the base rate up
as a finding.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from evals.calibration import MIN_BAND_SIZE, build
from evals.stats import wilson_interval

REPO = pathlib.Path(__file__).resolve().parent.parent


class TestWilsonInterval:
    def test_it_brackets_the_observed_rate(self):
        low, high = wilson_interval(30, 100)
        assert low < 0.30 < high

    def test_it_stays_inside_zero_and_one(self):
        """The normal approximation runs below 0 here; a calibration band with
        a negative lower bound is not a thing that can be shown to a user."""
        low, high = wilson_interval(0, 12)
        assert low == 0.0
        assert 0 < high < 1

        low, high = wilson_interval(12, 12)
        assert high == 1.0
        assert 0 < low < 1

    def test_a_bigger_sample_gives_a_tighter_interval(self):
        small = wilson_interval(30, 100)
        large = wilson_interval(300, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_an_empty_band_admits_nothing(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestBandsAreOnlyCalledInformativeWhenTheyAre:
    def _table(self, scores, labels):
        return build(np.array(scores, dtype=float), np.array(labels, dtype=int))

    def test_a_band_matching_the_base_rate_is_not_informative(self):
        """The failure this exists to prevent: quoting 26% beside a base rate
        of 26% reads as a finding and is not one."""
        scores = [45.0] * 200 + [15.0] * 200
        labels = ([2] * 52 + [0] * 148) * 2
        table = self._table(scores, labels)
        assert all(not band["informative"] for band in table["bands"])

    def test_a_band_that_really_differs_is_informative(self):
        scores = [85.0] * 200 + [15.0] * 200
        labels = [2] * 180 + [0] * 20 + [0] * 200
        table = self._table(scores, labels)
        top = next(b for b in table["bands"] if b["low"] == 80)
        assert top["informative"] is True
        assert top["direction"] == "above"

    def test_a_thin_band_is_never_informative_however_extreme(self):
        """Five documents cannot support a claim about anything."""
        scores = [95.0] * 5 + [15.0] * 400
        labels = [2] * 5 + [0] * 400
        table = self._table(scores, labels)
        thin = next(b for b in table["bands"] if b["low"] == 90)
        assert thin["n"] < MIN_BAND_SIZE
        assert thin["informative"] is False

    def test_the_base_rate_is_the_rate_across_everything(self):
        table = self._table([10.0] * 100 + [90.0] * 100, [0] * 100 + [2] * 100)
        assert table["base_rate"] == pytest.approx(0.5)


class TestTheShippedTable:
    def test_it_exists_and_records_its_provenance(self):
        table = json.loads(
            (REPO / "backend" / "data" / "calibration.json").read_text(encoding="utf-8")
        )
        assert table["documents"] > 1000
        assert table["provenance"]["git"]["commit"]

    def test_the_measured_result_is_that_almost_nothing_is_informative(self):
        """Recorded as a test so a later change that appears to fix this has to
        acknowledge it. One band of seven, and not the highest one."""
        table = json.loads(
            (REPO / "backend" / "data" / "calibration.json").read_text(encoding="utf-8")
        )
        assert table["informative_bands"] <= 2, (
            "if calibration genuinely improved, update this test and say so"
        )


class TestLookup:
    def test_a_score_inside_a_band_finds_it(self):
        from services.calibration import describe

        result = describe(45)
        assert result is not None
        assert result["band_low"] == 40

    def test_a_score_the_data_never_saw_returns_nothing(self):
        """Silence beats applying the nearest band to a score outside the
        observed range — the product shows /100 but never produces above 75."""
        from services.calibration import describe

        assert describe(95) is None

    def test_a_missing_table_degrades_rather_than_failing(self, monkeypatch, tmp_path):
        from services import calibration

        calibration._table.cache_clear()
        monkeypatch.setattr(calibration, "TABLE_PATH", tmp_path / "absent.json")
        try:
            assert calibration.describe(45) is None
        finally:
            calibration._table.cache_clear()

    def test_a_malformed_table_degrades_rather_than_failing(self, monkeypatch, tmp_path):
        from services import calibration

        broken = tmp_path / "calibration.json"
        broken.write_text("{not json", encoding="utf-8")
        calibration._table.cache_clear()
        monkeypatch.setattr(calibration, "TABLE_PATH", broken)
        try:
            assert calibration.describe(45) is None
        finally:
            calibration._table.cache_clear()


class TestTheUiDoesNotOverstateIt:
    def test_a_non_informative_band_is_labelled_as_such(self):
        sidebar = (
            REPO / "frontend/src/components/dashboard/ScoreSidebar.jsx"
        ).read_text(encoding="utf-8")
        assert "not" in sidebar and "distinguishable" in sidebar
        assert "not a prediction" in sidebar
