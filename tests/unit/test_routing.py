"""Unit tests for the routing decision function.

These tests cover the table from the plan: trivial input, gatekeeper-driven
short-circuits, gatekeeper-failure safety net, intent → QueryType mapping,
and the vague-query MEMORY_FIRST fallback.

Pure-function tests — no Pinecone, Neo4j, Redis, or LLM is touched.
"""

from __future__ import annotations

from typing import Any

import pytest

from Memory_Layer.session_memory.models import StructuredState
from Memory_Layer.session_memory.retriever import WorkingMemory
from graphrag.query_understanding.query_types import QueryType
from graphrag.query_understanding.routing import (
    RoutingMode,
    decide_routing,
    is_trivial_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_wm(
    *,
    turn_count: int = 0,
    symptoms: list[str] | None = None,
    drugs: list[str] | None = None,
    conditions: list[str] | None = None,
) -> WorkingMemory:
    """Build a WorkingMemory with just the fields routing cares about."""
    state = StructuredState(
        symptoms=symptoms or [],
        drugs=drugs or [],
        conditions=conditions or [],
    )
    return WorkingMemory(
        session_id="test",
        summary="",
        recent_turns=[],
        state=state,
        active_task=state.active_task,
        risk_level=str(state.risk_level),
        discussed_entities=list(state.discussed_entities),
        preferences=dict(state.preferences),
        turn_count=turn_count,
        has_summary=False,
    )


def analysis(intent: str, **extra: Any) -> dict[str, Any]:
    """Build a minimal gatekeeper-shaped analysis dict."""
    payload: dict[str, Any] = {"intent": intent}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# is_trivial_input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["hi", "Hello", "hey!", "thanks", "thank you", "ok.", "okay",
     "got it", "cool", "nice", "great", "noted", "  hi  ", "Hi!"],
)
def test_is_trivial_input_matches_acknowledgments(raw: str) -> None:
    assert is_trivial_input(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "hi there how are you",          # not a bare greeting
        "hello, my chest hurts",         # carries clinical content
        "thanks for the info, but...",   # acknowledgment + more
        "what is myocarditis?",
        "fever 3 days",
        "is it serious?",                # follow-up; not trivial
        "",                              # empty isn't trivial
    ],
)
def test_is_trivial_input_rejects_substantive_text(raw: str) -> None:
    assert not is_trivial_input(raw)


@pytest.mark.parametrize("raw", ["yes", "no", "no.", "Yes", "NO", "sure", " yes "])
def test_closed_answers_are_not_trivial(raw: str) -> None:
    """
    A bare yes/no answers a clarifying question THIS service asked, so it is
    clinical content, not an acknowledgment.

    Regression: classifying "no." as trivial skipped the gatekeeper, which set
    vector_top_k = 0 (concluding answer generated with no corpus behind it) and
    query_type = "unknown", which in turn suppressed the exposure-history layer
    so the model was handed [LOCAL CONDITIONS] with no instruction to use them.
    """
    assert not is_trivial_input(raw)


def test_closed_answer_still_reaches_retrieval_in_session() -> None:
    """The turn that concludes a consultation must not be the ungrounded one."""
    mode, _ = decide_routing(
        analysis=analysis("symptom_query", medical_entities={"symptoms": ["rash"]}),
        wm=make_wm(turn_count=3),
        raw_query="no.",
    )
    assert mode is not RoutingMode.NO_RETRIEVAL


# ---------------------------------------------------------------------------
# decide_routing — trivial pre-gate
# ---------------------------------------------------------------------------


def test_trivial_input_in_established_session_skips_retrieval() -> None:
    mode, qt = decide_routing(
        analysis=analysis("symptom_query", medical_entities={"symptoms": ["fever"]}),
        wm=make_wm(turn_count=3),
        raw_query="hi",
    )
    assert mode == RoutingMode.NO_RETRIEVAL
    assert qt == QueryType.UNKNOWN


def test_trivial_input_on_first_turn_still_routes_via_gatekeeper() -> None:
    # First-turn greetings should NOT take the pre-gate shortcut — the
    # gatekeeper still gets to decide so the assistant can introduce itself.
    mode, qt = decide_routing(
        analysis=analysis("greeting"),
        wm=make_wm(turn_count=0),
        raw_query="hi",
    )
    # Falls through to gatekeeper's "greeting" → NO_RETRIEVAL anyway.
    assert mode == RoutingMode.NO_RETRIEVAL
    assert qt == QueryType.UNKNOWN


def test_acknowledgment_in_session_skips_retrieval() -> None:
    mode, qt = decide_routing(
        analysis=analysis("unknown"),
        wm=make_wm(turn_count=2),
        raw_query="thanks!",
    )
    assert mode == RoutingMode.NO_RETRIEVAL
    assert qt == QueryType.UNKNOWN


# ---------------------------------------------------------------------------
# decide_routing — gatekeeper-driven short-circuits
# ---------------------------------------------------------------------------


def test_followup_query_with_clinical_state_keeps_retrieval() -> None:
    """
    CHANGED DELIBERATELY. This previously asserted NO_RETRIEVAL/UNKNOWN.

    "is it serious?" about a known fever is where a consultation concludes, and
    concluding with no corpus — and with the exposure layer suppressed by
    QueryType.UNKNOWN — was the defect. With clinical state in the session a
    continuation now keeps a cheap retrieval budget and a usable query_type.
    """
    mode, qt = decide_routing(
        analysis=analysis("followup_query", final_action="route_to_followup"),
        wm=make_wm(turn_count=2, symptoms=["fever"]),
        raw_query="is it serious?",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.SYMPTOM_QUERY


def test_greeting_intent_skips_retrieval_even_without_pregate_match() -> None:
    mode, qt = decide_routing(
        analysis=analysis("greeting"),
        wm=make_wm(turn_count=1),
        raw_query="good morning doctor",  # not in TRIVIAL_INPUT regex
    )
    assert mode == RoutingMode.NO_RETRIEVAL
    assert qt == QueryType.UNKNOWN


def test_route_to_followup_action_alone_short_circuits() -> None:
    mode, qt = decide_routing(
        analysis=analysis("symptom_query", final_action="route_to_followup"),
        wm=make_wm(turn_count=2),
        raw_query="and now?",
    )
    assert mode == RoutingMode.NO_RETRIEVAL


# ---------------------------------------------------------------------------
# decide_routing — gatekeeper failure safety net
# ---------------------------------------------------------------------------


def test_empty_analysis_degrades_to_memory_first_not_full_retrieval() -> None:
    mode, qt = decide_routing(
        analysis={},
        wm=make_wm(turn_count=1),
        raw_query="any medical question",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.UNKNOWN


def test_none_analysis_degrades_to_memory_first() -> None:
    mode, qt = decide_routing(
        analysis=None,
        wm=make_wm(turn_count=0),
        raw_query="any medical question",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.UNKNOWN


def test_analysis_missing_intent_degrades_to_memory_first() -> None:
    mode, qt = decide_routing(
        analysis={"final_action": "retrieve"},
        wm=make_wm(),
        raw_query="anything",
    )
    assert mode == RoutingMode.MEMORY_FIRST


# ---------------------------------------------------------------------------
# decide_routing — full retrieval for real medical intents with grounding
# ---------------------------------------------------------------------------


def test_diagnosis_query_with_entity_full_retrieval() -> None:
    mode, qt = decide_routing(
        analysis=analysis(
            "diagnosis_query",
            medical_entities={"conditions": ["myocarditis"]},
        ),
        wm=make_wm(),
        raw_query="what is myocarditis?",
    )
    assert mode == RoutingMode.HYBRID_RAG
    assert qt == QueryType.DIAGNOSIS


def test_medication_query_maps_to_drug_interaction() -> None:
    mode, qt = decide_routing(
        analysis=analysis(
            "medication_query",
            medical_entities={"drugs": ["ibuprofen", "aspirin"]},
        ),
        wm=make_wm(),
        raw_query="can I take ibuprofen with aspirin?",
    )
    assert mode == RoutingMode.HYBRID_RAG
    assert qt == QueryType.DRUG_INTERACTION


def test_symptom_query_with_entity_full_retrieval() -> None:
    mode, qt = decide_routing(
        analysis=analysis(
            "symptom_query",
            medical_entities={"symptoms": ["fever"]},
        ),
        wm=make_wm(),
        raw_query="fever 3 days",
    )
    assert mode == RoutingMode.HYBRID_RAG
    assert qt == QueryType.SYMPTOM_QUERY


def test_treatment_query_maps_to_guideline() -> None:
    mode, qt = decide_routing(
        analysis=analysis(
            "treatment_query",
            medical_entities={"conditions": ["hypertension"]},
        ),
        wm=make_wm(),
        raw_query="how is hypertension managed?",
    )
    assert mode == RoutingMode.HYBRID_RAG
    assert qt == QueryType.GUIDELINE


# ---------------------------------------------------------------------------
# decide_routing — vague-query MEMORY_FIRST fallback
# ---------------------------------------------------------------------------


def test_vague_symptom_query_without_entities_or_memory_uses_memory_first() -> None:
    mode, qt = decide_routing(
        analysis=analysis("symptom_query", medical_entities={}),
        wm=make_wm(),  # empty state
        raw_query="my back hurts a bit",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    # QueryType still flows through so MEMORY_FIRST callers could use it later.
    assert qt == QueryType.SYMPTOM_QUERY


def test_vague_query_with_memory_context_uses_full_retrieval() -> None:
    mode, qt = decide_routing(
        analysis=analysis("symptom_query", medical_entities={}),
        wm=make_wm(turn_count=2, symptoms=["chest pain"]),
        raw_query="my back hurts a bit",
    )
    assert mode == RoutingMode.HYBRID_RAG
    assert qt == QueryType.SYMPTOM_QUERY


def test_vague_treatment_query_uses_memory_first() -> None:
    mode, qt = decide_routing(
        analysis=analysis("treatment_query", medical_entities={}),
        wm=make_wm(turn_count=4),
        raw_query="what should I do?",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.GUIDELINE


# ---------------------------------------------------------------------------
# decide_routing — unrecognized intent
# ---------------------------------------------------------------------------


def test_intent_unknown_without_short_circuit_uses_memory_first() -> None:
    # "unknown" is not in GATEKEEPER_INTENT_TO_QUERYTYPE and isn't a
    # conversational short-circuit either → safest default is MEMORY_FIRST.
    mode, qt = decide_routing(
        analysis=analysis("unknown", medical_entities={"symptoms": ["headache"]}),
        wm=make_wm(),
        raw_query="something feels off",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.UNKNOWN


def test_intent_not_in_mapping_uses_memory_first() -> None:
    # An intent the gatekeeper might invent that we don't know about.
    mode, qt = decide_routing(
        analysis=analysis("lifestyle_query", medical_entities={"symptoms": ["fatigue"]}),
        wm=make_wm(),
        raw_query="why am I tired?",
    )
    assert mode == RoutingMode.MEMORY_FIRST
    assert qt == QueryType.UNKNOWN


# ---------------------------------------------------------------------------
# Clinical continuations must keep their corpus AND their query_type
# ---------------------------------------------------------------------------


def test_clinical_followup_keeps_retrieval_and_query_type() -> None:
    """
    Regression: "so what do you think is causing it?" arrives as
    followup_query. Routing it to NO_RETRIEVAL/UNKNOWN meant the concluding
    turn of a consultation had no corpus behind it, and QueryType.UNKNOWN
    stripped the exposure layer that tells the model to use [LOCAL CONDITIONS].
    """
    mode, qt = decide_routing(
        analysis=analysis("followup_query"),
        wm=make_wm(turn_count=4, symptoms=["cracked knuckles"]),
        raw_query="so what do you think is causing it?",
    )
    assert mode is RoutingMode.MEMORY_FIRST
    assert qt is QueryType.SYMPTOM_QUERY


def test_route_to_followup_action_behaves_the_same() -> None:
    mode, qt = decide_routing(
        analysis={"intent": "other", "final_action": "route_to_followup"},
        wm=make_wm(turn_count=4, conditions=["eczema"]),
        raw_query="and what about at night?",
    )
    assert mode is RoutingMode.MEMORY_FIRST
    assert qt is QueryType.SYMPTOM_QUERY


def test_social_followup_without_clinical_state_stays_cheap() -> None:
    """No clinical content in the session — nothing to retrieve, keep it free."""
    mode, qt = decide_routing(
        analysis=analysis("followup_query"),
        wm=make_wm(turn_count=2),
        raw_query="what did you say before?",
    )
    assert mode is RoutingMode.NO_RETRIEVAL
    assert qt is QueryType.UNKNOWN


def test_greeting_is_still_free() -> None:
    mode, qt = decide_routing(
        analysis=analysis("greeting"),
        wm=make_wm(turn_count=3, symptoms=["rash"]),
        raw_query="hello again",
    )
    assert mode is RoutingMode.NO_RETRIEVAL
    assert qt is QueryType.UNKNOWN
