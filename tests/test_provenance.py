"""Every reported number carries what produced it.

Item 17 of the v2 upgrade plan. "nDCG@5 was 0.9984" is not a reproducible
claim. "nDCG@5 was 0.9984 at commit e45ed19, on bge-small-en-v1.5, against
stress_a.jsonl with digest 27eb7246" is — including for someone who is not us
and does not have this machine.
"""
from __future__ import annotations

import json

from evals.provenance import dataset_digests, provenance, render


class TestTheBlockIdentifiesTheRun:
    def test_it_records_the_commit(self):
        assert len(provenance()["git"]["commit"]) >= 7

    def test_it_says_when_the_tree_was_dirty(self):
        """A dirty tree means the commit does not describe the code that ran,
        which is worth knowing before a number goes into a paper."""
        assert isinstance(provenance()["git"]["dirty"], bool)

    def test_it_records_the_embedding_model(self):
        from services.similarity import EMBEDDING_MODEL

        assert provenance()["models"]["embedding"] == EMBEDDING_MODEL

    def test_it_records_the_llm_chain_not_just_that_one_exists(self):
        from core.config import settings

        assert provenance()["models"]["llm_chain"] == list(settings.model_chain)

    def test_it_records_the_versions_that_can_move_a_score(self):
        """A pandas bump cannot change nDCG. An onnxruntime bump can."""
        packages = provenance()["packages"]
        assert "fastembed" in packages
        assert "numpy" in packages

    def test_it_is_json_serialisable(self):
        json.dumps(provenance())


class TestDatasetDigests:
    def test_every_data_file_is_digested(self):
        digests = dataset_digests()
        assert "stress_a.jsonl" in digests
        assert "stress_b.jsonl" in digests

    def test_a_digest_is_stable_across_calls(self):
        assert dataset_digests() == dataset_digests()

    def test_a_relabelled_case_changes_the_digest(self, tmp_path, monkeypatch):
        """The labels are the ground truth the whole harness rests on. Editing
        one silently would invalidate every number measured before it."""
        from evals import provenance as module

        data = tmp_path / "data"
        data.mkdir()
        target = data / "stress_a.jsonl"
        target.write_text('{"id": "a", "relevance": 3}\n', encoding="utf-8")
        monkeypatch.setattr(module, "DATA_DIR", data)
        before = module.dataset_digests()

        target.write_text('{"id": "a", "relevance": 2}\n', encoding="utf-8")
        assert module.dataset_digests() != before


class TestTheBaselineCarriesIt:
    def test_the_committed_baseline_records_its_own_provenance(self):
        import pathlib

        baseline = json.loads(
            (pathlib.Path(__file__).resolve().parent.parent / "evals" / "baseline.json")
            .read_text(encoding="utf-8")
        )
        assert baseline["provenance"]["git"]["commit"]
        assert baseline["provenance"]["datasets"]


class TestRendering:
    def test_a_dirty_tree_is_called_out_in_the_text_output(self):
        block = provenance()
        block["git"]["dirty"] = True
        assert "DIRTY" in "\n".join(render(block))

    def test_a_clean_tree_is_not(self):
        block = provenance()
        block["git"]["dirty"] = False
        assert "DIRTY" not in "\n".join(render(block))
