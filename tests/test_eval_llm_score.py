"""The evaluation harness must not depend on the product's response shape.

Item 1 of the v2 upgrade plan: the harness reads a model-produced score through
``evals.llm_score`` so that removing that number from ``AnalysisResponse``
(item 2) cannot silently disable model selection.
"""
from __future__ import annotations

import pathlib

import pytest

from evals.llm_score import EVAL_SCORE_FIELD, eval_score

REPO = pathlib.Path(__file__).resolve().parent.parent


class _Stub:
    overall_score = 73.5


def test_eval_score_reads_the_eval_only_field():
    assert eval_score(_Stub()) == pytest.approx(73.5)


def test_eval_score_raises_rather_than_defaulting():
    """A silent 0.0 would look like a model that scores everything badly."""
    with pytest.raises(AttributeError):
        eval_score(object())


def test_field_is_named_on_the_llm_schema():
    from models.schemas import LLMEvaluation

    assert EVAL_SCORE_FIELD in LLMEvaluation.model_fields


@pytest.mark.parametrize("module", ["evals/run.py", "evals/bakeoff.py"])
def test_harness_does_not_read_the_score_attribute_directly(module):
    source = (REPO / module).read_text(encoding="utf-8")
    assert ".overall_score" not in source, (
        f"{module} reads the eval-only score directly; go through evals.llm_score"
    )
