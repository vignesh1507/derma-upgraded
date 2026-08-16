"""
Routing decisions for the GraphRAG pipeline.

A single pure function (`decide_routing`) maps the gatekeeper's analysis +
the current session memory + the raw user query to one of three modes:

    NO_RETRIEVAL  — skip Pinecone, entity extraction, and Neo4j entirely.
                    The LLM answers from session memory only.
    MEMORY_FIRST  — minimal vector backfill (3 chunks, no graph) for vague
                    queries that benefit from a tiny bit of clinical grounding.
    HYBRID_RAG    — full pipeline as configured per QueryType.

Why this exists: the gatekeeper returns intents (`diagnosis_query`,
`medication_query`, `treatment_query`, ...) that do not match the QueryType
enum exactly. The previous inline routing block fell through to
`QueryType.UNKNOWN` — whose config is `vector_top_k=15, graph_hops=1` —
so any unrecognized intent triggered full retrieval. This module fixes
that by giving `decide_routing` the final say on retrieval cost.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from Memory_Layer.session_memory.retriever import WorkingMemory
from graphrag.query_understanding.query_types import QueryType


class RoutingMode(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    MEMORY_FIRST = "memory_first"
    HYBRID_RAG = "hybrid_rag"


# Map gatekeeper intent strings to QueryType. Intents intentionally omitted
# (greeting, followup_query, emergency, unknown) are short-circuited or routed
# to NO_RETRIEVAL/MEMORY_FIRST before QueryType is consulted, so the absence
# of a mapping is the correct signal.
GATEKEEPER_INTENT_TO_QUERYTYPE: dict[str, QueryType] = {
    "symptom_query": QueryType.SYMPTOM_QUERY,
    "diagnosis_query": QueryType.DIAGNOSIS,
    "medication_query": QueryType.DRUG_INTERACTION,
    "treatment_query": QueryType.GUIDELINE,
}


# Trivial inputs that need no external knowledge AND no gatekeeper LLM call
# once a session is established. Match-anchored so "hi there how are you" does
# not slip through.
#
# "yes" / "no" / "sure" are DELIBERATELY ABSENT. This service asks closed
# clarifying questions, so a bare "no." is the discriminating clinical answer,
# not an acknowledgment. Treating it as trivial skipped the gatekeeper, which
# dropped vector_top_k to 0 AND set query_type to "unknown" — and "unknown" is
# outside the exposure-relevant intents, so the layer telling the model to use
# [LOCAL CONDITIONS] was stripped too. Observed live: a Leh consultation
# concluded "irritant contact dermatitis, likely harsh soaps" on a turn where
# the prompt carried "dew point 4C — air is very dry", with no corpus retrieval
# behind the treatment plan. One extra analyzer call per yes/no turn is the
# correct trade.
TRIVIAL_INPUT = re.compile(
    r"^\s*(hi|hello|hey|thanks?|thank\s*you|ok|okay|got\s*it|cool|nice|great|noted)[!.\s]*$",
    re.IGNORECASE,
)


def is_trivial_input(raw_query: str) -> bool:
    """True if the user's message is a short acknowledgment or greeting."""
    return bool(TRIVIAL_INPUT.match(raw_query or ""))


def _has_extracted_entities(analysis: dict[str, Any]) -> bool:
    entities = analysis.get("medical_entities") or {}
    return any(entities.get(k) for k in ("symptoms", "drugs", "conditions"))


def _has_memory_clinical_context(wm: WorkingMemory) -> bool:
    state = wm.state
    return bool(state.symptoms or state.drugs or state.conditions)


def decide_routing(
    analysis: dict[str, Any] | None,
    wm: WorkingMemory,
    raw_query: str,
) -> tuple[RoutingMode, QueryType]:
    """
    Decide retrieval mode + downstream QueryType from gatekeeper analysis,
    current working memory, and the raw user query.

    Pure function: no I/O, no logging side effects. Caller logs the decision.
    """
    # 1) Trivial acknowledgment in an established session → no retrieval, no LLM gatekeeper.
    if is_trivial_input(raw_query) and wm.turn_count > 0:
        return RoutingMode.NO_RETRIEVAL, QueryType.UNKNOWN

    # 2) Gatekeeper said this is conversational continuation.
    intent = (analysis or {}).get("intent") or ""
    action = (analysis or {}).get("final_action") or ""
    is_continuation = intent == "followup_query" or action == "route_to_followup"

    if intent == "greeting":
        return RoutingMode.NO_RETRIEVAL, QueryType.UNKNOWN

    if is_continuation:
        # A continuation of a CLINICAL thread is usually where the consultation
        # actually concludes — "so what do you think is causing it?" arrives as
        # followup_query. Sending that to NO_RETRIEVAL/UNKNOWN cost two things
        # at once: the concluding answer had no corpus behind it, and
        # QueryType.UNKNOWN suppressed the exposure layer, so the model was
        # handed [LOCAL CONDITIONS] with no instruction to use them. Observed
        # live in a Leh consultation that concluded on ambient dryness data it
        # had been told to ignore.
        #
        # Gated on real clinical state, so a purely social continuation still
        # costs nothing. SYMPTOM_QUERY is the carry-over type: it is the
        # general clinical-conversation type this platform routes on, and the
        # one the exposure and OTC layers key off.
        if _has_memory_clinical_context(wm):
            return RoutingMode.MEMORY_FIRST, QueryType.SYMPTOM_QUERY
        return RoutingMode.NO_RETRIEVAL, QueryType.UNKNOWN

    # 3) Gatekeeper failed (empty / no intent) → degrade DOWN to the cheap path,
    #    never up to full retrieval.
    if not analysis or not analysis.get("intent"):
        return RoutingMode.MEMORY_FIRST, QueryType.UNKNOWN

    # 4) Recognized intent but not one we map to a QueryType (e.g. "unknown"
    #    that wasn't short-circuited as non-medical) → minimal backfill.
    query_type = GATEKEEPER_INTENT_TO_QUERYTYPE.get(intent)
    if query_type is None:
        return RoutingMode.MEMORY_FIRST, QueryType.UNKNOWN

    # 5) Recognized medical intent, but the query is vague AND we have no
    #    clinical context in memory yet → small backfill rather than full pull.
    if not _has_extracted_entities(analysis) and not _has_memory_clinical_context(wm):
        return RoutingMode.MEMORY_FIRST, query_type

    # 6) Full retrieval — known medical intent with either extracted entities
    #    or supporting clinical state in memory to ground the retrieval.
    return RoutingMode.HYBRID_RAG, query_type


__all__ = [
    "RoutingMode",
    "GATEKEEPER_INTENT_TO_QUERYTYPE",
    "TRIVIAL_INPUT",
    "is_trivial_input",
    "decide_routing",
]
