"""A TF-IDF cosine baseline, hand-rolled.

The published resume-matching literature reports TF-IDF cosine as the standard
non-neural baseline, so the comparison needs one. Written here in thirty lines
rather than pulled in from scikit-learn: the project has been deliberate about
dependencies, and a baseline whose exact behaviour is visible is worth more in
a paper than one behind a configurable estimator.

Standard choices, stated so they can be checked: lowercase alphanumeric tokens
of three characters or more, sublinear term frequency (1 + log tf), smoothed
inverse document frequency, L2-normalised vectors, cosine similarity.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"\b[a-z][a-z0-9+#.]{2,}\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfSpace:
    """Document frequencies fitted on a corpus, reused for every comparison."""

    def __init__(self, corpus: list[str]):
        self.n_documents = max(len(corpus), 1)
        self.document_frequency: Counter[str] = Counter()
        for document in corpus:
            self.document_frequency.update(set(tokenize(document)))

    def _idf(self, term: str) -> float:
        # Smoothed, so an unseen term gets a finite weight rather than infinity.
        return math.log((1 + self.n_documents) / (1 + self.document_frequency[term])) + 1.0

    def vector(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        if not counts:
            return {}
        weights = {
            term: (1.0 + math.log(count)) * self._idf(term) for term, count in counts.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()} if norm else {}

    def similarity(self, first: str, second: str) -> float:
        left, right = self.vector(first), self.vector(second)
        if not left or not right:
            return 0.0
        smaller, larger = (left, right) if len(left) < len(right) else (right, left)
        cosine = sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
        return round(max(0.0, min(1.0, cosine)) * 100, 2)
