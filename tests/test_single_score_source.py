"""One number, one source.

Item 2 of the v2 upgrade plan. The LLM used to report an overall score, an ATS
compatibility score, a readability score, its own section scores, a passive
voice count and a grammar issue list — every one of them a second opinion on a
number the deterministic pipeline already produces. The two disagreed, and the
LLM's had nothing to validate it against.

These tests fail if any of them comes back.
"""
from __future__ import annotations

import pathlib

import pytest
from models.schemas import AnalysisResponse, LLMEvaluation, LLMEvaluationPublic

REPO = pathlib.Path(__file__).resolve().parent.parent

# Item 3: fields the model was asked to invent, with no data source behind them.
GUESSED = ["salary_insight", "competition_level"]

DUPLICATED = [
    "overall_score",
    "ats_compatibility",
    "readability_score",
    "section_scores",
    "passive_voice_count",
    "grammar_issues",
]


@pytest.mark.parametrize("field", DUPLICATED)
def test_field_is_not_in_the_public_llm_view(field):
    assert field not in LLMEvaluationPublic.model_fields


@pytest.mark.parametrize("field", GUESSED)
def test_guessed_fields_are_gone_everywhere(field):
    """No pay range, no count of rival applicants — the system knows neither."""
    assert field not in LLMEvaluation.model_fields
    assert field not in LLMEvaluationPublic.model_fields

    prompt = (REPO / "backend/services/llm_evaluator.py").read_text(encoding="utf-8")
    assert field not in prompt

    types = (REPO / "frontend/src/types/api.ts").read_text(encoding="utf-8")
    assert f"{field}: " not in types

    prep = (REPO / "frontend/src/components/InterviewPrep.jsx").read_text(encoding="utf-8")
    assert f"llm.{field}" not in prep


def test_the_eval_only_score_survives_on_the_provider_schema():
    """Item 1 depends on it: deleting it would disable model selection."""
    assert "overall_score" in LLMEvaluation.model_fields


def test_public_view_drops_the_eval_only_score():
    from test_llm_cache import make_evaluation

    public = make_evaluation(85).public()
    assert isinstance(public, LLMEvaluationPublic)
    assert "overall_score" not in public.model_dump()


def test_response_carries_the_public_view_only():
    annotation = str(AnalysisResponse.model_fields["llm_evaluation"].annotation)
    assert "LLMEvaluationPublic" in annotation


def test_section_scores_have_one_source():
    """The response keeps the deterministic section scores and only those."""
    assert "section_scores" in AnalysisResponse.model_fields
    assert "section_scores" not in LLMEvaluationPublic.model_fields


def test_the_ui_does_not_read_the_removed_fields():
    sidebar = (REPO / "frontend/src/components/dashboard/ScoreSidebar.jsx").read_text(encoding="utf-8")
    for field in DUPLICATED:
        assert f"llm.{field}" not in sidebar and f"llm?.{field}" not in sidebar
