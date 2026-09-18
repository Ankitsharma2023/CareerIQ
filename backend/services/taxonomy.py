"""Skill vocabulary, alias resolution and implied skills.

Replaces the hand-maintained 178-term list with a taxonomy built from Stack
Overflow's tags (see scripts/fetch_taxonomy.py for why not ESCO, with the
measurement that decided it).

Three pieces, in the order they matter:

* **Exact linking, by Aho-Corasick.** Every canonical term and every alias goes
  into one automaton, and the text is scanned once regardless of how many
  thousand terms there are. The previous extractor ran a regex per skill, which
  is why the list had to stay small to stay fast.
* **Alias resolution from two tables, merged.** Stack Overflow's 1,278
  synonyms, curated by the people who use the terms — js to javascript, nodejs
  to node.js, rails to ruby-on-rails — plus this project's own table, which
  carries entries Stack Overflow has no synonym for: iac, gha, cfn, and the
  context-disambiguated tf. Replacing one with the other loses real coverage in
  both directions, which a first version of this file did: it dropped
  terraform, prometheus, github actions and infrastructure as code from six
  stress resumes before the merge was added.

  Hyphen and space spellings are registered together, because Stack Overflow
  tags "machine-learning" and a resume writes "machine learning".
* **Adjacency, from tag co-occurrence.** Docker co-occurs with containers,
  docker-compose — and also with java and linux. That is co-occurrence, not
  implication, and the distinction matters: this is deliberately *not* used to
  credit a candidate with skills they did not write. Item 16 exists to stop the
  system inventing claims on a user's behalf, and inferring "you have Docker so
  you must have Kubernetes" would be the same mistake in the scorer instead of
  the rewriter. It is used only to tell a candidate that a gap sits next to
  something they already do.

Matching is boundary-aware: "go" must not fire inside "going", and "c" must not
fire inside "client". Terms carrying their own punctuation (c++, node.js, c#,
.net) are matched as written.
"""
from __future__ import annotations

import json
import logging
import re
from collections import deque
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"

# A URL or an email address is contact information, not a skill claim. Without
# this, "github.com/ravimenon" in a resume header credits the candidate with
# GitHub and "linkedin.com/in/..." with the LinkedIn API — the second on 4 of
# 42 stress resumes. Masked rather than removed so every character offset the
# matcher reports still lines up with the original text.
_URL_OR_EMAIL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b"
    r"|\b(?:github|gitlab|bitbucket|linkedin|twitter|x)\.com/\S*",
    re.I,
)

#: Characters that continue a term rather than ending it. A match must not be
#: flanked by these, so "go" does not fire inside "golang" or "going", while
#: "node.js" and "c++" match as written.
_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789")
#: Characters that genuinely change which term was written. "c" followed by
#: "+" or "#" is a different language; "node" followed by ".js" is a different
#: technology. A hyphen is not in this set: "Redis-backed rate limiter" and
#: "cloud-native" are compound English, and rejecting them cost redis on a
#: resume that plainly says Redis.
_TERM_EXTENDING = set("+#_")


class AhoCorasick:
    """Multi-pattern search: one pass over the text, whatever the term count.

    Written out rather than pulled in from pyahocorasick because it is eighty
    lines, the dependency is a C extension that complicates the deployment, and
    a reader of this file can check the matching rules against the boundary
    behaviour the tests describe.
    """

    __slots__ = ("_goto", "_fail", "_output")

    def __init__(self, terms: list[str]):
        self._goto: list[dict[str, int]] = [{}]
        self._output: list[list[str]] = [[]]
        self._fail: list[int] = [0]

        for term in terms:
            node = 0
            for character in term:
                node = self._goto[node].setdefault(character, self._new_node())
            self._output[node].append(term)

        self._build_failures()

    def _new_node(self) -> int:
        self._goto.append({})
        self._output.append([])
        self._fail.append(0)
        return len(self._goto) - 1

    def _build_failures(self) -> None:
        queue: deque[int] = deque()
        for node in self._goto[0].values():
            self._fail[node] = 0
            queue.append(node)

        while queue:
            current = queue.popleft()
            for character, target in self._goto[current].items():
                queue.append(target)
                state = self._fail[current]
                while state and character not in self._goto[state]:
                    state = self._fail[state]
                self._fail[target] = self._goto[state].get(character, 0)
                if self._fail[target] == target:
                    self._fail[target] = 0
                self._output[target].extend(self._output[self._fail[target]])

    def find(self, text: str):
        """Yield ``(start, end, term)`` for every occurrence, overlaps included."""
        state = 0
        for index, character in enumerate(text):
            while state and character not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(character, 0)
            for term in self._output[state]:
                yield index - len(term) + 1, index + 1, term


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """Is this match a whole term rather than a fragment of a longer one?"""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    following = text[end + 1] if end + 1 < len(text) else ""

    if before and (before in _WORD_CHARS or before in _TERM_EXTENDING or before == "."):
        return False
    if after and after in _WORD_CHARS:
        return False
    if after and after in _TERM_EXTENDING:
        # "c" in "c++" and "c#" is not the C language being named twice.
        return False
    if after == "." and following and following in _WORD_CHARS:
        # "node" in "node.js", "asp" in "asp.net". A full stop ending a
        # sentence is fine.
        return False
    return True


@lru_cache(maxsize=1)
def _taxonomy() -> dict:
    try:
        return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("No taxonomy at %s; falling back to the curated list", TAXONOMY_PATH)
        return {"skills": {}, "aliases": {}, "implied": {}}
    except (OSError, json.JSONDecodeError):
        logger.exception("Taxonomy at %s could not be read", TAXONOMY_PATH)
        return {"skills": {}, "aliases": {}, "implied": {}}


def _spelling_variants(term: str, canonical_names: set[str] | None = None) -> set[str]:
    """Hyphen and space spellings of the same name.

    Stack Overflow tags "machine-learning"; a resume writes "machine learning".
    Both must reach the same canonical term or the taxonomy is worse than the
    list it replaces.

    The space variant is withheld when the first word is itself a skill.
    "machine learning" is a unit, but "python requests" in prose is the word
    python followed by an ordinary noun — and generating that variant credited
    three stress resumes with the python-requests library for the sentence
    "used the python requests library".
    """
    variants = {term}
    if "-" in term:
        spaced = term.replace("-", " ")
        head = term.split("-", 1)[0]
        if not canonical_names or head not in canonical_names:
            variants.add(spaced)
    if " " in term:
        variants.add(term.replace(" ", "-"))
    return variants


@lru_cache(maxsize=1)
def _automaton() -> tuple[AhoCorasick, dict[str, str]]:
    """The automaton, plus the map from every matchable term to its canonical."""
    taxonomy = _taxonomy()

    # Where both vocabularies name the same skill, the curated spelling wins.
    # Stack Overflow tags "machine-learning"; this project, its evaluation
    # labels and its UI all say "machine learning", and silently renaming the
    # canonical form would collapse skill_f1 against hand-written labels that
    # use the other spelling.
    try:
        from services.skill_extractor import load_skills

        curated = {s.strip().lower() for s in load_skills()}
    except Exception:  # pragma: no cover
        curated = set()

    names = set(taxonomy.get("skills", {})) | curated

    preferred: dict[str, str] = {}
    for name in taxonomy.get("skills", {}):
        chosen = name
        for variant in _spelling_variants(name, names):
            if variant in curated:
                chosen = variant
                break
        preferred[name] = chosen

    canonical: dict[str, str] = {}
    for name, chosen in preferred.items():
        for variant in _spelling_variants(name, names):
            canonical[variant] = chosen
    for name in curated:
        for variant in _spelling_variants(name, names):
            canonical.setdefault(variant, name)

    def add_alias(alias: str, target: str) -> None:
        for variant in _spelling_variants(alias, names):
            # An alias never overrides a canonical term of the same spelling:
            # "react" is a tag in its own right and must not be rewritten away.
            if variant not in taxonomy.get("skills", {}):
                canonical.setdefault(variant, target)

    for alias, target in taxonomy.get("aliases", {}).items():
        add_alias(alias, preferred.get(target, target))

    # This project's own table, kept because it has entries Stack Overflow has
    # no synonym for.
    try:
        from services.aliases import ALIASES

        for alias, target in ALIASES.items():
            add_alias(alias, target)
            # The target may be a term Stack Overflow does not tag at all
            # ("infrastructure as code"), in which case it becomes canonical.
            for variant in _spelling_variants(target, names):
                canonical.setdefault(variant, target)
    except Exception:  # pragma: no cover - the taxonomy still works without it
        logger.exception("Curated aliases could not be merged")

    return AhoCorasick(sorted(canonical)), canonical


def available() -> bool:
    return bool(_taxonomy().get("skills"))


def extract(text: str) -> set[str]:
    """Canonical skills named in the text, aliases resolved.

    Context-dependent abbreviations are expanded first. A fixed alias table
    cannot decide "TF": it is TensorFlow on a resume about models and Terraform
    on one about clusters, and services/aliases.py resolves that from the
    document before the automaton sees it. Without this step the taxonomy loses
    terraform on exactly the resume item 13's invariance tests were written
    around.
    """
    if not text:
        return set()

    lowered = text.lower()
    # Blank out contact details first: a profile URL is not a claim of skill.
    lowered = _URL_OR_EMAIL_RE.sub(lambda m: " " * len(m.group(0)), lowered)
    try:
        from services.aliases import expand

        lowered = expand(lowered)
    except Exception:  # pragma: no cover
        logger.exception("Alias expansion failed; matching the raw text")

    automaton, canonical = _automaton()

    # Overlapping matches are all kept. A first version suppressed anything
    # contained in a longer match — "github" inside "github actions" — which
    # looked tidier and broke symmetry: a resume saying "CI/CD pipelines" and a
    # job description saying "CI/CD" would then extract different skills and the
    # JD's requirement would be reported as a gap the resume plainly fills.
    # Crediting both is what makes the two sides comparable, and it is the
    # behaviour tests/test_skill_extraction.py has pinned since before this
    # taxonomy existed.
    return {
        canonical[term]
        for start, end, term in automaton.find(lowered)
        if _boundary_ok(lowered, start, end)
    }


def adjacent_to(skills) -> set[str]:
    """Skills that commonly co-occur with the ones given.

    Never added to a candidate's skill set. Co-occurrence is not implication —
    docker co-occurs with java as readily as with containers — and crediting
    someone with a skill they did not write would be the scorer making the
    fabrication the rewriter is forbidden to make. Used to say "this gap is
    adjacent to what you already do", which is a claim about the gap.
    """
    taxonomy = _taxonomy().get("implied", {})
    named = {s.strip().lower() for s in skills}
    out: set[str] = set()
    for skill in named:
        out.update(taxonomy.get(skill, []))
    return out - named


def adjacency_for(gap: str, held) -> list[str]:
    """Which of the candidate's skills sit next to a gap they do not have."""
    taxonomy = _taxonomy().get("implied", {})
    neighbours = set(taxonomy.get(gap.strip().lower(), []))
    # Symmetric: co-occurrence has no direction, so a gap counts as adjacent to
    # a held skill if either lists the other.
    for skill in held:
        if gap.strip().lower() in taxonomy.get(skill.strip().lower(), []):
            neighbours.add(skill.strip().lower())
    return sorted(neighbours & {s.strip().lower() for s in held})


def canonical_for(term: str) -> str | None:
    """The canonical name for a term or alias, or None if unknown."""
    _, canonical = _automaton()
    return canonical.get(term.strip().lower())


def size() -> dict[str, int]:
    taxonomy = _taxonomy()
    return {
        "skills": len(taxonomy.get("skills", {})),
        "aliases": len(taxonomy.get("aliases", {})),
        "implied": len(taxonomy.get("implied", {})),
    }
