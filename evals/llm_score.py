"""The harness's own read of a model-produced score.

``LLMEvaluation.overall_score`` is an *eval-only* field. It is excluded from
``AnalysisResponse``, so no part of the product reads it: the single number a
user sees comes from ``compute_fit``. Two conflicting scores on one screen was
the defect that made the LLM number impossible to validate.

The harness still needs a model-produced number — it is what ranks candidate
models by Spearman against the human relevance grades, and what the two-agent
extraction work will be measured against. So every such read goes through this
module instead of reaching into the product schema directly. If the field is
ever renamed or moved, this is the only place that changes, and the product
stays unaffected either way.
"""
from __future__ import annotations

from typing import Any

#: The attribute on the LLM's structured output that carries its own 0-100
#: judgement of fit. Eval-only: see the module docstring.
EVAL_SCORE_FIELD = "overall_score"


def eval_score(evaluation: Any) -> float:
    """Return the model's own fit score, for measurement purposes only.

    Raises ``AttributeError`` rather than returning a default, because a silent
    0.0 here would look exactly like a model that scores everything badly and
    would quietly corrupt a bake-off.
    """
    return float(getattr(evaluation, EVAL_SCORE_FIELD))
