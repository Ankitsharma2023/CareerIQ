"""The skill taxonomy: matching, aliases and what is deliberately not a skill.

Item 21 of the v2 upgrade plan, with one substitution. The plan specified ESCO.
ESCO was measured first and rejected: of this project's 178 skills, 26 (15%)
have an exact ESCO match, 77 (43%) are absent entirely — Docker, Kubernetes,
Terraform, AWS, Django, FastAPI, Kafka — and 75 (42%) return search hits with
no exact match. ESCO's best hit for "react" is "react to emergency situations
in a live performance". A linker over ESCO labels would map a frontend
developer onto emergency response.

Stack Overflow's tags are the vocabulary engineers write, and its tag-synonym
table is a curated alias set maintained by the people who use the terms.
"""
from __future__ import annotations

import pytest
from services import taxonomy
from services.skill_extractor import extract_skills_from_text, load_skills


@pytest.fixture(scope="module")
def curated():
    return load_skills()


class TestTheTaxonomyIsBuilt:
    def test_it_is_present_and_substantial(self):
        assert taxonomy.available()
        sizes = taxonomy.size()
        assert sizes["skills"] > 1500
        assert sizes["aliases"] > 1000

    def test_it_covers_the_tools_esco_does_not(self):
        """The 43% ESCO is missing entirely, which is the reason for the swap."""
        for skill in ("docker", "kubernetes", "terraform", "fastapi", "kafka", "django"):
            assert taxonomy.canonical_for(skill) == skill, skill


class TestMatching:
    @pytest.mark.parametrize("text,expected", [
        ("Wrote C++ and C# daily", {"c++", "c#"}),
        ("Built with node.js", {"node.js"}),
        ("Shipped asp.net core services", {"asp.net"}),
        ("A Redis-backed rate limiter", {"redis"}),
        ("cloud-native java services", {"java"}),
    ])
    def test_terms_with_punctuation_match_as_written(self, text, expected):
        assert expected <= taxonomy.extract(text)

    @pytest.mark.parametrize("text,absent", [
        ("Going to the shop", "go"),
        ("A client of the firm", "c"),
        ("Improved the process", "r"),
    ])
    def test_a_term_does_not_fire_inside_a_longer_word(self, text, absent):
        assert absent not in taxonomy.extract(text)

    def test_c_does_not_match_inside_c_plus_plus(self):
        """"C++" is not the C language named twice."""
        found = taxonomy.extract("Wrote C++ for eight years")
        assert "c++" in found
        assert "c" not in found

    def test_overlapping_matches_are_all_kept(self):
        """A resume saying "CI/CD pipelines" and a JD saying "CI/CD" must
        extract compatibly, or the JD's requirement is reported as a gap the
        resume plainly fills. Suppressing the contained match broke this."""
        found = taxonomy.extract("Owned CI/CD pipelines end to end")
        assert "ci/cd" in found


class TestAliases:
    @pytest.mark.parametrize("alias,canonical", [
        ("k8s", "kubernetes"),
        ("js", "javascript"),
        ("nodejs", "node.js"),
        ("iac", "infrastructure as code"),
    ])
    def test_short_forms_resolve(self, alias, canonical):
        assert taxonomy.canonical_for(alias) == canonical

    def test_both_alias_tables_are_merged(self):
        """Stack Overflow has no synonym for iac, gha or cfn; this project's
        table does. Replacing one with the other loses coverage in both
        directions — a first version dropped terraform, prometheus and github
        actions from six stress resumes."""
        assert taxonomy.canonical_for("iac") == "infrastructure as code"
        assert taxonomy.canonical_for("js") == "javascript"

    def test_hyphen_and_space_spellings_reach_the_same_skill(self):
        """Stack Overflow tags machine-learning; a resume writes machine
        learning."""
        assert "machine learning" in taxonomy.extract("worked on machine learning models")

    def test_the_curated_spelling_wins_where_both_exist(self):
        """The evaluation labels and the UI say "machine learning"; renaming the
        canonical form to the tag spelling would collapse skill_f1 against
        labels written the other way."""
        assert taxonomy.canonical_for("machine-learning") == "machine learning"

    def test_an_alias_never_overrides_a_real_term(self):
        """"react" is a tag in its own right and must not be rewritten away."""
        assert taxonomy.canonical_for("react") == "react"

    def test_context_decides_an_ambiguous_abbreviation(self):
        assert "tensorflow" in taxonomy.extract("trained and served tf models for ranking")
        assert "terraform" in taxonomy.extract("provisioned clusters with tf")


class TestWhatIsDeliberatelyNotASkill:
    def test_contact_urls_are_not_skill_claims(self):
        """github.com/name in a header credited the candidate with GitHub, and
        linkedin.com/in/name with the LinkedIn API, on 4 of 42 stress resumes."""
        found = taxonomy.extract("linkedin.com/in/ravimenon | github.com/ravimenon")
        assert "github" not in found
        assert "linkedin-api" not in found

    def test_an_email_address_is_not_a_skill_claim(self):
        assert taxonomy.extract("ravi.menon@example.com") == set()

    def test_a_degree_is_not_a_skill(self):
        assert "computer-science" not in taxonomy.extract(
            "Bachelor of Science in Computer Science, 2019"
        )

    def test_ordinary_programming_nouns_are_excluded(self):
        """Stack Overflow tags the nouns of programming. Matching them against
        resume prose invents skills out of "improved performance"."""
        found = taxonomy.extract("Improved performance and refactored the architecture")
        assert "performance" not in found
        assert "architecture" not in found

    def test_a_compound_tag_does_not_match_a_prose_phrase(self):
        """"used the python requests library" is python plus a noun, not the
        python-requests package."""
        assert "python-requests" not in taxonomy.extract("used the python requests library")


class TestAdjacencyIsNotEvidence:
    def test_co_occurring_skills_are_never_added_to_a_candidate(self):
        """Crediting someone with Kubernetes because they wrote Docker would be
        the scorer making exactly the fabrication item 16 forbids the rewriter
        from making."""
        found = taxonomy.extract("Built and shipped with Docker")
        assert "kubernetes" not in found

    def test_adjacency_reports_what_the_candidate_already_has(self):
        adjacent = taxonomy.adjacency_for("kubernetes", ["docker", "python"])
        assert "docker" in adjacent


class TestIntegration:
    def test_the_default_vocabulary_goes_through_the_taxonomy(self, curated):
        found = extract_skills_from_text("Deployed on K8s with Docker", curated)
        assert "kubernetes" in found

    def test_a_caller_supplied_list_keeps_the_original_matcher(self):
        """"Find exactly these skills" is a different question from "find every
        skill you know about", and several tests ask the first one."""
        found = extract_skills_from_text("Python and Kubernetes and Rust", ["python", "rust"])
        assert set(found) == {"python", "rust"}

    def test_nothing_the_curated_list_found_is_lost(self, curated):
        """The swap must not regress coverage of the vocabulary it replaces."""
        text = (
            "Built Python services on Kubernetes with Docker and Terraform. "
            "Used Prometheus, GitHub Actions, FastAPI, PostgreSQL and Redis."
        )
        taxonomy_found = taxonomy.extract(text)
        for skill in ("python", "kubernetes", "docker", "terraform", "prometheus",
                      "github actions", "fastapi", "postgresql", "redis"):
            assert skill in taxonomy_found, skill
