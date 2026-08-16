"""
EntityProcessor.process_matches — entity parsing and context assembly.

This module had no test coverage. It parses chunk metadata from whichever
Pinecone corpus the deployment points at, and the platform is used against
corpora that follow two different entity conventions:

    typed    "disease: psoriasis"   — e.g. the `enervera` index
    untyped  "psoriasis"            — e.g. `enervara-specialists`

The parser originally skipped anything without a colon, which silently
discarded every entity in an untyped corpus. These tests pin both conventions
so the behaviour cannot regress.
"""

from __future__ import annotations

import pytest

from graphrag.processors.entity_processor import EntityProcessor


def _match(entities, summary="A summary.", mid="c1"):
    return {"id": mid, "metadata": {"entities": entities, "summary": summary}}


# ---------------------------------------------------------------------------
# Untyped corpus (dermatology: enervara-specialists)
# ---------------------------------------------------------------------------

class TestUntypedEntities:
    """Bare entity names, no `type:` prefix."""

    def test_untyped_entities_are_extracted(self):
        matches = [_match(["psoriasis", "papule", "erythema", "acitretin"])]
        _, entities, _ = EntityProcessor.process_matches(matches)
        assert set(entities) == {"psoriasis", "papule", "erythema", "acitretin"}

    def test_untyped_entities_survive_a_priority_filter(self):
        """
        priority_entity_types cannot match an untyped corpus, but that must
        degrade to "no priority ordering", never to "no entities".
        """
        matches = [_match(["psoriasis", "papule"])]
        _, entities, _ = EntityProcessor.process_matches(
            matches, priority_entity_types=["disease", "drug"]
        )
        assert set(entities) == {"psoriasis", "papule"}

    def test_names_are_lowercased(self):
        matches = [_match(["Psoriasis", "  Papule  "])]
        _, entities, _ = EntityProcessor.process_matches(matches)
        assert set(entities) == {"psoriasis", "papule"}


# ---------------------------------------------------------------------------
# Typed corpus (gastroenterology: enervera)
# ---------------------------------------------------------------------------

class TestTypedEntities:
    """`type: name` entities — the convention the parser was written for."""

    def test_typed_entities_are_extracted_without_their_prefix(self):
        matches = [_match(["disease: gastric ulcers", "procedure: endoscopy"])]
        _, entities, _ = EntityProcessor.process_matches(matches)
        assert set(entities) == {"gastric ulcers", "endoscopy"}

    def test_priority_types_are_surfaced_first(self):
        matches = [_match([
            "disease: crohn's",
            "procedure: colonoscopy",
            "drug: mesalamine",
        ])]
        _, entities, _ = EntityProcessor.process_matches(
            matches, priority_entity_types=["disease"]
        )
        assert entities[0] == "crohn's", "priority entity must lead the list"
        assert set(entities) == {"crohn's", "colonoscopy", "mesalamine"}

    def test_unmatched_priority_type_is_harmless(self):
        """A configured type absent from the corpus must not drop entities."""
        matches = [_match(["disease: crohn's"])]
        _, entities, _ = EntityProcessor.process_matches(
            matches, priority_entity_types=["lab_value", "threshold"]
        )
        assert entities == ["crohn's"]


# ---------------------------------------------------------------------------
# Mixed and malformed input
# ---------------------------------------------------------------------------

class TestMixedAndMalformed:
    def test_both_conventions_in_one_result_set(self):
        matches = [
            _match(["disease: psoriasis"], mid="typed"),
            _match(["papule"], mid="untyped"),
        ]
        _, entities, _ = EntityProcessor.process_matches(matches)
        assert set(entities) == {"psoriasis", "papule"}

    @pytest.mark.parametrize("bad", ["", "   ", ":", "disease:", "disease:   "])
    def test_empty_names_are_dropped_not_stored_as_blanks(self, bad):
        _, entities, _ = EntityProcessor.process_matches([_match([bad, "psoriasis"])])
        assert entities == ["psoriasis"]

    def test_missing_entities_key_does_not_raise(self):
        matches = [{"id": "c1", "metadata": {"summary": "S"}}]
        ctx, entities, summaries = EntityProcessor.process_matches(matches)
        assert entities == []
        assert summaries == ["S"]

    def test_no_matches_returns_placeholder(self):
        ctx, entities, summaries = EntityProcessor.process_matches([])
        assert entities == [] and summaries == []
        assert isinstance(ctx, str) and ctx


# ---------------------------------------------------------------------------
# Context assembly — this is what actually reaches the LLM
# ---------------------------------------------------------------------------

class TestContextAssembly:
    def test_summaries_are_returned_and_rendered(self):
        matches = [
            _match(["psoriasis"], summary="Psoriasis is a chronic plaque disease.", mid="a"),
            _match(["eczema"],    summary="Eczema presents with itch.",             mid="b"),
        ]
        ctx, _, summaries = EntityProcessor.process_matches(matches)
        assert summaries == [
            "Psoriasis is a chronic plaque disease.",
            "Eczema presents with itch.",
        ]
        for s in summaries:
            assert s in ctx

    def test_context_is_independent_of_entity_convention(self):
        """
        Grounding comes from summaries, so an untyped corpus must produce the
        same context as a typed one. This is why the service still answered
        correctly before the parser was fixed.
        """
        typed   = [_match(["disease: psoriasis"], summary="Same summary.")]
        untyped = [_match(["psoriasis"],          summary="Same summary.")]
        ctx_typed, _, _ = EntityProcessor.process_matches(typed)
        ctx_untyped, _, _ = EntityProcessor.process_matches(untyped)
        assert ctx_typed == ctx_untyped

    def test_entity_list_is_capped(self):
        many = [f"entity-{i}" for i in range(200)]
        _, entities, _ = EntityProcessor.process_matches([_match(many)])
        assert len(entities) <= 30, "entity list must stay bounded for prompt budget"
