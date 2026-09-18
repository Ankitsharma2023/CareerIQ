#!/usr/bin/env python3
"""Build the skill taxonomy from Stack Overflow's tag data.

Why here and not ESCO. ESCO is the obvious choice for an occupational taxonomy
and it was measured first: of this project's 178 skills, 26 (15%) have an exact
ESCO match, 77 (43%) are absent entirely — Docker, Kubernetes, Terraform, AWS,
Django, FastAPI, Kafka — and 75 (42%) return search hits with no exact match.
That last group is the dangerous one. ESCO's best hit for "react" is "react to
emergency situations in a live performance"; for "infrastructure as code",
"assign codes to product items"; for "distributed systems", "change power
distribution systems". A linker over ESCO labels would map a frontend
developer's React experience onto emergency response. ESCO describes the whole
European labour market at the level of "use software design patterns", not the
tool names that decide an engineering hire.

Stack Overflow's tags are the vocabulary engineers actually write, with three
properties this needs:

* **popularity counts**, which give a principled cut-off instead of a
  hand-picked list;
* **tag synonyms** — a curated alias table maintained by the people who use the
  terms: js -> javascript, nodejs -> node.js, rails -> ruby-on-rails. This is
  the thing that fixes abbreviation sensitivity;
* **related tags**, which give co-occurrence strong enough to seed implied
  skills (kubernetes implies containers).

    python scripts/fetch_taxonomy.py --out backend/data/taxonomy.json

The API allows 300 requests a day unauthenticated. The script caches every page
and is safe to re-run: it resumes rather than refetching, and stops cleanly
when the quota runs low rather than burning the rest on partial data.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
PAGE_SIZE = 100

#: Tags below this many questions are too rare to be worth a taxonomy entry and
#: are where the noise lives — typos, one-off library names, abandoned tools.
#:
#: In practice the API's own limit binds first: page 25 is the last one an
#: unauthenticated caller may read, which reaches tags with about 3,000
#: questions. So this threshold is a floor for a future authenticated run
#: rather than the cut that is actually applied today, and the built file
#: records the real minimum it reached.
MIN_TAG_COUNT = 400

#: Unauthenticated callers cannot read beyond this page.
MAX_UNAUTHENTICATED_PAGE = 25

#: A list of tags is not a list of skills. Stack Overflow tags include the
#: ordinary nouns of programming — arrays, string, list, function, loops,
#: image, forms, performance, algorithm — and matching those against resume
#: prose would invent skills out of sentences like "improved performance".
#:
#: So a tag that is among the commonest English words is dropped, unless it
#: carries a technology marker (a digit, dot, plus or hash: c++, node.js, c#)
#: or already appears in this project's curated list, which is the trusted
#: ground truth for exactly this question. That rescues python, java, css,
#: html, sql, go, ruby, swift, spring and less from the filter while removing
#: 326 tags that would have produced false skills.
#:
#: The cut-off is a judgement, and 10,000 is chosen because a false skill is
#: worse than a missing one: the product reports gaps, and a JD saying "on call
#: rotation" must not produce a missing skill called "rotation". At 5,000 it
#: did — "rotation" is English rank 7,984 and a Stack Overflow tag with 12,022
#: questions. The cost is losing "dns" (rank 6,660), which was not in the
#: curated list either, so nothing regresses.
#:
#: No threshold separates the two categories cleanly. "for-loop" has 71,930
#: questions and is not a resume skill at any cut-off, which is why a short
#: explicit blocklist sits below this: popularity cannot express "this is a
#: language construct, not a competence".
COMMON_WORDS_CUTOFF = 10_000

#: Tags that survive every automatic filter and are still not skills. Language
#: constructs and artefacts of how questions get tagged, not competences
#: anyone claims on a resume. Kept short and explicit — if it grows past a
#: page, the filter above is the thing to fix.
NOT_SKILLS = {
    "for-loop", "while-loop", "if-statement", "foreach", "nested-loops",
    "return", "casting", "syntax", "variables", "methods", "inheritance",
    "types", "null", "exception", "error-handling", "logging", "parsing",
    "sorting", "filter", "printing", "rounding", "concatenation",
    "user-interface", "dashboard", "backend", "frontend", "web", "mobile",
    "desktop", "server", "client", "browser", "terminal", "console",
    "installation", "deployment", "configuration", "migration", "versioning",
    "authentication", "authorization", "encryption", "validation",
    # Academic and profile furniture that appears in every resume header or
    # education line and is never a claim of competence.
    "computer-science", "linkedin-api", "new-operator", "operator-keyword",
    "resources", "documentation", "naming-conventions", "code-formatting",
}
WORDLIST_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/"
    "master/google-10000-english-usa.txt"
)

#: Stop when fewer than this remain, so a re-run still has room to make progress.
QUOTA_FLOOR = 20

CACHE = Path(__file__).resolve().parent.parent / ".taxonomy-cache"


def _get(path: str, cache_key: str, **params) -> dict:
    cached = CACHE / f"{cache_key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    params.setdefault("site", SITE)
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    payload = json.loads(raw)

    CACHE.mkdir(exist_ok=True)
    cached.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.4)  # the API asks for this; being a good citizen is cheap
    return payload


def fetch_tags(max_pages: int) -> list[dict]:
    tags: list[dict] = []
    for page in range(1, min(max_pages, MAX_UNAUTHENTICATED_PAGE) + 1):
        try:
            payload = _get(
                "tags", f"tags-{page}",
                page=page, pagesize=PAGE_SIZE, order="desc", sort="popular",
            )
        except urllib.error.HTTPError as exc:
            # 403 past page 25, or a throttle. Either way, what has been
            # fetched is usable and refetching it would waste quota.
            print(f"  tags page {page}: HTTP {exc.code}, stopping", file=sys.stderr)
            break
        items = payload.get("items", [])
        tags.extend(items)
        print(f"  tags page {page}: {len(items)} (quota {payload.get('quota_remaining')})",
              file=sys.stderr)
        if not payload.get("has_more") or payload.get("quota_remaining", 0) < QUOTA_FLOOR:
            break
        if items and items[-1]["count"] < MIN_TAG_COUNT:
            break
    return tags


def fetch_synonyms(max_pages: int) -> list[dict]:
    synonyms: list[dict] = []
    for page in range(1, min(max_pages, MAX_UNAUTHENTICATED_PAGE) + 1):
        try:
            payload = _get(
                "tags/synonyms", f"synonyms-{page}",
                page=page, pagesize=PAGE_SIZE, order="desc", sort="applied",
            )
        except urllib.error.HTTPError as exc:
            print(f"  synonyms page {page}: HTTP {exc.code}, stopping", file=sys.stderr)
            break
        items = payload.get("items", [])
        synonyms.extend(items)
        print(f"  synonyms page {page}: {len(items)} (quota {payload.get('quota_remaining')})",
              file=sys.stderr)
        if not payload.get("has_more") or payload.get("quota_remaining", 0) < QUOTA_FLOOR:
            break
    return synonyms


def fetch_related(tags: list[str], limit: int) -> dict[str, list[str]]:
    """Co-occurrence for the most popular tags, to seed implied skills.

    One request per tag, so this is capped deliberately: the implication graph
    is most useful at the top of the distribution, where a tag like kubernetes
    reliably co-occurs with docker and containers.
    """
    related: dict[str, list[str]] = {}
    for index, tag in enumerate(tags[:limit]):
        try:
            payload = _get(
                f"tags/{urllib.parse.quote(tag)}/related", f"related-{tag.replace('/', '_')}",
                pagesize=12, order="desc", sort="popular",
            )
        except urllib.error.HTTPError as exc:
            print(f"  related {tag}: HTTP {exc.code}", file=sys.stderr)
            break
        items = [t["name"] for t in payload.get("items", []) if t["name"] != tag]
        related[tag] = items[:8]
        if payload.get("quota_remaining", 0) < QUOTA_FLOOR:
            print(f"  stopping at {index + 1} related lookups; quota low", file=sys.stderr)
            break
    return related


def common_words(cutoff: int) -> set[str]:
    cached = CACHE / "common-words.txt"
    if not cached.exists():
        CACHE.mkdir(exist_ok=True)
        with urllib.request.urlopen(WORDLIST_URL, timeout=30) as response:
            cached.write_bytes(response.read())
    words = cached.read_text(encoding="utf-8").split()
    return {w.strip().lower() for w in words[:cutoff]}


def _is_technology(tag: str) -> bool:
    """Tags carrying a version, dot, plus or hash are never ordinary words."""
    return any(character in tag for character in ".+#0123456789")


def build(tags: list[dict], synonyms: list[dict], related: dict, cutoff: int) -> dict:
    from services.skill_extractor import load_skills

    curated = {s.strip().lower() for s in load_skills()}
    ordinary = common_words(cutoff)

    kept: dict[str, int] = {}
    dropped: list[str] = []
    for tag in tags:
        name = tag["name"]
        if tag["count"] < MIN_TAG_COUNT:
            continue
        if name in NOT_SKILLS and name not in curated:
            dropped.append(name)
            continue
        if name in ordinary and not _is_technology(name) and name not in curated:
            dropped.append(name)
            continue
        kept[name] = tag["count"]

    # The curated list is authoritative: anything in it belongs in the taxonomy
    # whether or not Stack Overflow ranks it highly.
    for skill in curated:
        kept.setdefault(skill, 0)

    # Only synonyms whose target survived the popularity cut, and never one
    # that collides with a canonical name — "react" is itself a tag, so it must
    # not be rewritten to "reactjs" and lose its own identity.
    #
    # The ordinary-word filter applies here too, and it matters more than it
    # does for tags. Stack Overflow has the synonym "requests" -> the
    # python-requests library, and "requests" appears in almost every backend
    # resume: "serving 40M requests a day" was crediting three stress
    # candidates with a HTTP client library they never mentioned.
    aliases: dict[str, str] = {}
    for entry in synonyms:
        source, target = entry["from_tag"], entry["to_tag"]
        if target not in kept or source in kept:
            continue
        if source in NOT_SKILLS:
            dropped.append(source)
            continue
        if source in ordinary and not _is_technology(source) and source not in curated:
            dropped.append(source)
            continue
        aliases[source] = target

    implied = {
        tag: [t for t in others if t in kept]
        for tag, others in related.items()
        if tag in kept
    }

    return {
        "source": "Stack Overflow tags, tag synonyms and related tags",
        "common_words_cutoff": cutoff,
        "dropped_as_ordinary_words": len(dropped),
        "dropped_sample": sorted(dropped)[:20],
        "min_count_reached": min((c for c in kept.values() if c), default=None),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_tag_count": MIN_TAG_COUNT,
        "skills": kept,
        "aliases": aliases,
        "implied": {k: v for k, v in implied.items() if v},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag-pages", type=int, default=30)
    parser.add_argument("--synonym-pages", type=int, default=25)
    parser.add_argument("--related", type=int, default=60)
    parser.add_argument("--cutoff", type=int, default=COMMON_WORDS_CUTOFF,
                        help="how many of the commonest English words to treat as not-skills")
    args = parser.parse_args()

    print("fetching tags...", file=sys.stderr)
    tags = fetch_tags(args.tag_pages)
    print("fetching synonyms...", file=sys.stderr)
    synonyms = fetch_synonyms(args.synonym_pages)
    print("fetching related tags...", file=sys.stderr)
    ranked = [t["name"] for t in tags if t["count"] >= MIN_TAG_COUNT]
    related = fetch_related(ranked, args.related)

    taxonomy = build(tags, synonyms, related, args.cutoff)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(taxonomy, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{len(taxonomy['skills'])} skills, {len(taxonomy['aliases'])} aliases, "
        f"{len(taxonomy['implied'])} with implications -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
