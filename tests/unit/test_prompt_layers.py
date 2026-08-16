"""
Unit tests for the layered system-prompt composer.

The contract locked in here matches the experienced-clinician prompt
style: minimal high-signal questioning, probabilistic ranked
differentials with plain-English mechanisms, gated "consult a doctor"
phrasing (only when red flags or genuine uncertainty warrant it),
silent memory reuse with no re-asking, RAG integrated naturally
without meta-leak or fabrication, and escalation only for severe /
high-risk signs.

Assertions use ``.lower()`` substring matching where exact casing is
not the contract (case-sensitive only when capitalisation carries
weight, like section headers and ALL-CAPS rule emphasis). The
composed prompt fits a ~1100-token budget; tests target rule presence,
not exact wording.
"""

from __future__ import annotations

import pytest

from app.services.orchestration.prompt_layers import (
    layer_exposure_history,
    _INTENT_BLOCK_PLANS,
    compose_system_prompt,
    layer_block_plan,
    layer_core_identity,
    layer_formatting_constraints,
    layer_output_contract,
    layer_retrieval_grounding,
    layer_runtime_modifiers,
    layer_safety_policy,
    layer_session_state_instructions,
    layer_tool_instructions,
)
from graphrag.schemas.blocks import BLOCK_TYPES


# ---------------------------------------------------------------------------
# Layer 1 — Behaviour rules (experienced clinician)
# ---------------------------------------------------------------------------


def test_core_identity_experienced_clinician_persona():
    out = layer_core_identity()
    lo = out.lower()
    # Dermatology specialist, explicitly scoped to the skin.
    assert "dermatologist" in lo
    assert "general (internal) medicine" not in lo
    # Scope is stated in terms of skin structures.
    assert "skin" in lo
    assert "hair" in lo and "nails" in lo
    # Must hand off rather than reason past its specialty.
    assert "specialist" in lo
    assert "thoughtful" in lo or "doctor in clinic" in lo
    # Probabilistic clinical reasoning chain is the ethos.
    assert "probabilistic" in lo
    assert "differential" in lo
    assert "mechanism" in lo
    # Anti-defensive / anti-chatbot stance.
    assert "never defensive" in lo
    assert "robotic" in lo
    # Heard → thought about → helped (warmth + reasoning).
    assert "patient's concern" in lo or "patient" in lo


# ---------------------------------------------------------------------------
# Layer 2 — Safety & evidence constraints
# ---------------------------------------------------------------------------


def test_safety_probabilistic_and_evidence_grounded():
    out = layer_safety_policy().lower()
    assert "probabilistic language" in out
    assert "definitive diagnosis" in out
    assert "evidence-grounded" in out
    assert "established medical knowledge" in out
    # Anti-hallucination — explicit list of things never to invent.
    assert "never invent" in out
    for forbidden in ("symptoms", "mechanisms", "doses", "studies", "guidelines"):
        assert forbidden in out


def test_safety_gates_consult_a_doctor_phrase():
    out = layer_safety_policy().lower()
    # The phrase is no longer chanted — it's explicitly gated.
    assert "do not chant" in out
    assert "\"consult a doctor\"" in out
    # The disclaimer phrase remains available when warranted.
    assert "only a doctor can properly examine and confirm this" in out
    # Conditions for using it.
    assert "red flags" in out
    assert "genuine uncertainty" in out
    # Pairing requirements.
    assert "specific trigger" in out
    assert "timeframe" in out
    # Reject the bolt-on style.
    assert "mechanical bolt-on" in out or "bolt-on" in out


# ---------------------------------------------------------------------------
# Layer 3 — Runtime modifiers (risk + personalisation)
# ---------------------------------------------------------------------------


def test_runtime_personalisation_with_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=True)
    assert "Hey Aarav" in out
    assert "sparingly" in out
    assert "PERSONALISATION" in out


def test_runtime_personalisation_no_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=False).lower()
    assert "no name is known" in out
    assert "never invent" in out
    assert '"patient"' in out and '"user"' in out
    assert "hey aarav" not in out


def test_runtime_risk_critical_surfaces_warning():
    out = layer_runtime_modifiers(risk_level="critical", has_name=False)
    assert "⚠️ CRITICAL" in out
    # Critical block tells the LLM to skip the interview and escalate.
    assert "SKIP the interview" in out or "skip the interview" in out.lower()
    # Personalisation block still follows the risk header.
    assert "PERSONALISATION" in out


def test_runtime_risk_none_omits_header():
    out = layer_runtime_modifiers(risk_level="none", has_name=False)
    assert "⚠️" not in out
    assert "CRITICAL" not in out
    assert "Elevated risk" not in out


# ---------------------------------------------------------------------------
# Layer 4 — Memory & context reuse
# ---------------------------------------------------------------------------


def test_memory_reuse_silent_and_no_reasking():
    out = layer_session_state_instructions()
    lo = out.lower()
    assert "MEMORY & CONTEXT REUSE" in out
    # Memory is used silently, treated as already known.
    assert "silently" in lo
    assert "already known" in lo
    # The hard "never re-ask" rule, with named examples.
    assert "never re-ask" in lo
    for known_field in ("age", "sex", "name", "duration", "history", "meds"):
        assert known_field in lo
    # No restart, no echoing their own words.
    assert "never restart" in lo
    assert "echo" in lo or "summarise" in lo
    # New question takes priority.
    assert "current question is the priority" in lo
    assert "never redirect" in lo


# ---------------------------------------------------------------------------
# Layer 5 — RAG grounding policy
# ---------------------------------------------------------------------------


def test_retrieval_grounding_natural_integration():
    out = layer_retrieval_grounding()
    lo = out.lower()
    assert "CLINICAL KNOWLEDGE GROUNDING" in out
    assert "integrate it" in lo or "integrate it naturally" in lo
    # Paraphrase, never quote chunks verbatim.
    assert "paraphrase" in lo
    assert "never quote" in lo or "never quote chunks" in lo


def test_retrieval_grounding_no_meta_leak():
    out = layer_retrieval_grounding()
    assert "Never reference retrieval" in out
    # Each meta-leak term forbidden by name.
    for term in ("retrieval", "vectors", "summaries", "chunks", "graph", "memory"):
        assert term in out


def test_retrieval_grounding_no_fabrication():
    out = layer_retrieval_grounding().lower()
    assert "never fabricate" in out
    for forbidden in ("study", "dose", "brand", "guideline"):
        assert forbidden in out


# ---------------------------------------------------------------------------
# Layer 6 — Consultation flow + questioning strategy (converge, don't interrogate)
# ---------------------------------------------------------------------------


def test_consultation_flow_holds_and_updates_a_working_assessment():
    out = layer_tool_instructions()
    lo = out.lower()
    assert "consultation flow" in lo
    # A running working assessment that visibly updates (issues 6, 10).
    assert "working assessment" in lo
    assert "differential" in lo
    # Must not just restate the patient's symptoms.
    assert "never just restate" in lo or "restate" in lo


def test_consultation_flow_converges_and_completes():
    lo = layer_tool_instructions().lower()
    # Explicit convergence + completion strategy (issues 1, 7, 11).
    assert "converge" in lo
    assert "completion" in lo
    assert "red flags" in lo and "monitor" in lo
    # Stop at high confidence.
    assert "80%" in lo or "stop asking" in lo


def test_consultation_flow_summary_is_a_checkpoint_not_narration():
    lo = layer_tool_instructions().lower()
    # Summaries are consolidation checkpoints, not per-turn narration.
    assert "checkpoint" in lo
    assert "per-turn narration" in lo or "not per-turn" in lo
    # Named suppression cases (greeting / single-fact answers).
    assert "greeting" in lo
    assert "5 days" in lo or "single-fact" in lo
    # Questions still chosen by information gain.
    assert "information-gain" in lo or "information gain" in lo


def test_consultation_flow_educational_answers_first():
    lo = layer_tool_instructions().lower()
    # Educational/explanatory intents answer fully first, not history-taking (issue 5).
    assert "educational" in lo
    assert "answer first" in lo or "fully answer" in lo
    # Follow-ups on educational only if they change the recommendation / user asks.
    assert "change the recommendation" in lo or "personalised advice" in lo


def test_summary_synthesises_not_restates_last_message():
    # Obs 4/10: the summary must synthesise accumulated findings + working
    # assessment, not paraphrase the patient's latest message. The block plan
    # names it explicitly; the shared consultation-flow layer enforces it for
    # both prose and block modes (so the composed prompt always carries it).
    block = layer_block_plan(query_type="symptom_query", consolidate=True).lower()
    assert "working assessment" in block
    assert "not a restatement" in block
    composed = compose_system_prompt(query_type="symptom_query").lower()
    assert "working assessment" in composed
    assert "restate" in composed


def test_completion_closes_gracefully_without_appending_followup():
    # Obs 6: recognise the natural stopping point; don't tack on a follow-up.
    lo = layer_tool_instructions().lower()
    assert "natural end" in lo or "stopping point" in lo
    assert "do not append another follow-up" in lo
    assert "invite" in lo


def test_working_assessment_explains_why_leading_beats_alternatives():
    # Obs 7: brief reasoning why the leading cause beats the alternatives.
    lo = layer_tool_instructions().lower()
    assert "discriminating feature" in lo or "beats the alternatives" in lo or "fits better" in lo


def test_questioning_strategy_hard_caps():
    out = layer_tool_instructions().lower()
    # Hard cap: 1 per turn.
    assert "one follow-up question" in out.lower() or "one question" in out.lower()
    assert "at most" in out.lower()
    # Never re-ask answered facts (issue 3).
    assert "never re-ask" in out.lower()


def test_questioning_strategy_every_question_explains_why():
    out = layer_tool_instructions().lower()
    assert "clinical reasoning" in out
    # Worked example anchors the cadence.
    assert "chest pain" in out
    # Anti-padding rules.
    assert "never vague" in out
    assert "never multiple" in out
    assert "fill space" in out


# ---------------------------------------------------------------------------
# Layer 7a — PROSE response format (untouched /chat + /chat/stream paths)
# ---------------------------------------------------------------------------


# Every clinical intent the gatekeeper analyzer can emit — the prompt must treat
# all of them as substantive (regression against the taxonomy-mismatch bug where
# educational/decision intents fell through to the 1–2 sentence branch).
ANALYZER_SUBSTANTIVE_INTENTS = [
    "symptom_query", "diagnosis_query", "medication_query", "treatment_query",
    "condition_explanation", "lab_interpretation", "prognosis_query",
    "prevention_query", "lifestyle_query", "procedure_query",
    "comparison_query", "risk_assessment", "followup_query", "unknown",
]


@pytest.mark.parametrize("query_type", ANALYZER_SUBSTANTIVE_INTENTS)
def test_prose_substantive_uses_response_format(query_type: str):
    out = layer_formatting_constraints(query_type=query_type)
    assert "RESPONSE FORMAT" in out
    assert "substantive clinical" in out.lower()
    assert "flowing natural prose" in out.lower()
    # Must NOT be the short brush-off branch.
    assert "1–2 sentences" not in out


@pytest.mark.parametrize("query_type", [
    "condition_explanation", "prognosis_query", "prevention_query",
    "lifestyle_query", "procedure_query", "comparison_query", "risk_assessment",
])
def test_educational_intents_get_dedicated_block_plans(query_type: str):
    # These used to fall to the generic default plan (or worse, non-substantive).
    out = layer_block_plan(query_type=query_type)
    assert "substantive clinical reply" in out.lower()
    assert query_type in _INTENT_BLOCK_PLANS


def test_prose_substantive_includes_escalation_policy():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "ESCALATION POLICY" in out
    assert "112" in out and "102" in out and "108" in out
    assert "SKIP the interview" in out


def test_prose_does_not_leak_query_header():
    out = layer_formatting_constraints(query_type="symptom_query")
    # The "(query:" parenthetical that used to bleed into answers is gone.
    assert "(query:" not in out
    assert "Never repeat it" in out


def test_prose_non_substantive_is_short():
    out = layer_formatting_constraints(query_type="greeting")
    assert "non-substantive" in out.lower()
    assert "1–2 sentences" in out
    assert "ESCALATION POLICY" not in out


# ---------------------------------------------------------------------------
# Layer 7b — BLOCK plan (NDJSON /chat/blocks path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_block_plan_substantive(query_type: str):
    # consolidate=True → the assessment/consolidation plan (summary + steps).
    out = layer_block_plan(query_type=query_type, consolidate=True)
    assert "BLOCK PLAN" in out
    assert "substantive clinical reply" in out.lower()
    assert "summary" in out
    assert "next_steps" in out


def test_block_plan_gathering_turn_is_question_only():
    # A triage turn that still needs info: just the one question, no summary/narration.
    out = layer_block_plan(query_type="symptom_query", allow_followups=True, terminal=False)
    lo = out.lower()
    assert "information-gathering" in lo
    assert "follow_up_questions" in out
    assert "do not emit a summary" in lo
    # Educational intents are NOT gathering — they answer directly even if a
    # follow-up is allowed.
    edu = layer_block_plan(query_type="condition_explanation", allow_followups=True).lower()
    assert "information-gathering" not in edu


def test_block_plan_assessment_turn_leads_with_empathy_and_stopping():
    """On the consolidation turn, emit the summary/assessment and stop questioning."""
    out = layer_block_plan(query_type="symptom_query", consolidate=True, allow_followups=False)
    lo = out.lower()
    assert "empathy" in lo or "empathetic" in lo
    assert "confidence" in lo or "high confidence" in lo
    assert "80" in out or "80%" in lo or "high" in lo
    assert "condition_list" not in out or "only emit condition_list" in lo
    assert "follow_up_questions" in out


def test_lab_tests_on_concluded_symptom_turn():
    # Recommended tests are offered at diagnosis (concluding turn), before OTC.
    out = layer_block_plan(query_type="symptom_query", consolidate=True, allow_followups=False)
    assert "lab_tests" in out
    # Ordered before otc_medications (tests belong to the plan).
    assert out.index("lab_tests") < out.index("otc_medications")


def test_lab_tests_absent_while_gathering():
    out = layer_block_plan(query_type="symptom_query", allow_followups=True, terminal=False)
    assert "lab_tests" not in out


def test_lab_tests_absent_for_educational_intent():
    # Educational conclusions don't recommend investigations.
    out = layer_block_plan(query_type="condition_explanation", consolidate=True)
    assert "lab_tests" not in out


def test_lab_tests_on_decision_turn():
    out = layer_block_plan(query_type="symptom_query", response_mode="binary_decision")
    assert "lab_tests" in out


def test_block_plan_followups_gated_by_allow_flag():
    # Use a non-triage intent so the standard plan (with the {followups} line) is used.
    on = layer_block_plan(query_type="medication_query", allow_followups=True)
    assert "at most ONE high-signal" in on  # follow-up requested
    off = layer_block_plan(query_type="medication_query", allow_followups=False)
    assert "at most ONE high-signal" not in off  # not requested
    # ...and explicitly forbidden, so the model won't volunteer one.
    assert "do not emit a follow_up_questions" in off.lower()


def test_block_plan_forbids_questions_outside_followup_block():
    # Live runs showed the model smuggling questions into next_steps; forbid it
    # (on the assessment turn, which carries the response tail).
    lo = layer_block_plan(query_type="symptom_query", consolidate=True, allow_followups=False).lower()
    assert "questions only in a follow_up_questions block" in lo
    assert "never a filler or question-only turn" in lo


def test_block_plan_terminal_drops_followups_and_notes_closing_turn():
    out = layer_block_plan(query_type="symptom_query", terminal=True, allow_followups=True)
    assert "at most ONE high-signal" not in out
    assert "closing/assessment turn" in out
    assert "do not emit a follow_up_questions" in out.lower()


def test_block_plan_critical_risk_structure():
    out = layer_block_plan(query_type="symptom_query", risk_level="critical")
    assert "CRITICAL RISK" in out
    assert 'severity "critical"' in out
    assert "summary" in out
    assert "condition_list" in out
    assert "next_steps" in out
    assert "Do NOT emit follow_up_questions" in out


def test_block_plan_greeting_single_summary():
    out = layer_block_plan(query_type="greeting")
    assert "BLOCK PLAN" in out and "greeting" in out.lower()
    assert "one `summary`" in out
    assert "condition_list" not in out


def test_block_plan_non_substantive_single_summary():
    # `followup_query` is now substantive (a real question deserves a real
    # answer); the non-substantive fallback covers unmapped/meta tokens.
    out = layer_block_plan(query_type="smalltalk")
    assert "non-substantive" in out.lower()
    assert "one `summary`" in out
    assert "No condition_list" in out
    assert "no follow_up_questions" in out


# ---------------------------------------------------------------------------
# Layer 8 — Output contract (NDJSON)
# ---------------------------------------------------------------------------


def test_output_contract_is_ndjson_and_lists_all_block_types():
    from graphrag.schemas.blocks import CONTROL_BLOCK_TYPES, MODEL_BLOCK_TYPES

    out = layer_output_contract()
    assert "OUTPUT CONTRACT" in out
    assert "NDJSON" in out
    # The contract advertises every MODEL-emittable block type…
    for bt in MODEL_BLOCK_TYPES:
        assert bt in out
    # …but NEVER server-only control blocks (else the model imitates them).
    for bt in CONTROL_BLOCK_TYPES:
        assert bt not in out
    # Forbids array/wrapping/markdown and shows the two-line example.
    lo = out.lower()
    assert "one json block object per line" in lo
    assert "the entire reply must be json only" in lo
    assert "no surrounding array" in lo or "no array" in lo
    assert "no prose outside the json" in lo
    assert '{"type":"summary"' in out
    assert '"steps"' in out
    assert '"conditions"' in out
    assert '"description"' in out
    assert 'do not rename fields' in lo or 'required fields' in lo


# ---------------------------------------------------------------------------
# Composer — joins, skips, idempotent, budget
# ---------------------------------------------------------------------------


def test_compose_joins_all_layers_for_substantive_with_name_critical():
    # Default (prose) mode — the untouched /chat path.
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="critical",
        has_name=True,
    )
    # Markers from every non-empty layer must appear in the composed prompt.
    assert "experienced consultant dermatologist" in out          # L1
    assert "SAFETY & EVIDENCE" in out                              # L2
    assert "⚠️ CRITICAL" in out and "Hey Aarav" in out             # L3
    assert "MEMORY & CONTEXT REUSE" in out                         # L4
    assert "CLINICAL KNOWLEDGE GROUNDING" in out                   # L5
    assert "CONSULTATION FLOW" in out                              # L6
    assert "RESPONSE FORMAT" in out and "ESCALATION POLICY" in out  # L7 prose
    # Prose mode does NOT carry the NDJSON contract.
    assert "OUTPUT CONTRACT" not in out


def test_compose_prose_is_default_no_block_contract():
    out = compose_system_prompt(query_type="symptom_query")
    assert "RESPONSE FORMAT" in out
    assert "OUTPUT CONTRACT" not in out
    assert "BLOCK PLAN" not in out


def test_compose_blocks_mode_appends_output_contract_last():
    out = compose_system_prompt(query_type="symptom_query", output_format="blocks")
    assert "BLOCK PLAN" in out
    # The NDJSON contract is always the final layer in block mode.
    assert out.index("OUTPUT CONTRACT") > out.index("BLOCK PLAN")
    assert out.rstrip().endswith("}")  # ends on the example's closing brace
    # Block mode replaces the prose format layer.
    assert "RESPONSE FORMAT" not in out


def test_compose_skips_empty_layers_for_low_risk_no_name_greeting():
    out = compose_system_prompt(
        query_type="greeting",
        risk_level="none",
        has_name=False,
    )
    # Risk header is suppressed when risk_level is none.
    assert "⚠️ CRITICAL" not in out
    assert "Elevated risk" not in out
    # Personalisation still present but in the no-name variant.
    assert "no name is known" in out.lower()
    assert "Hey Aarav" not in out
    # Greeting → short prose non-substantive branch (default mode).
    assert "1–2 sentences" in out
    # No clinical escalation scaffolding for a greeting.
    assert "ESCALATION POLICY" not in out


def test_compose_idempotent_pure_function():
    args = dict(query_type="symptom_query", risk_level="low", has_name=True)
    a = compose_system_prompt(**args)
    b = compose_system_prompt(**args)
    assert a == b


def test_compose_no_blank_line_runs():
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    # Layers are joined with "\n\n"; no triple-newline runs should appear.
    assert "\n\n\n" not in out


def test_compose_defaults_safe():
    # No kwargs other than query_type — defaults risk=none, has_name=False, prose.
    out = compose_system_prompt(query_type="symptom_query")
    assert "experienced consultant dermatologist" in out
    assert "RESPONSE FORMAT" in out
    assert "⚠️" not in out
    assert "Hey Aarav" not in out
    assert "no name is known" in out.lower()


# ---------------------------------------------------------------------------
# Budget check — soft cap aligned with the bumped SYSTEM_PROMPT_MAX_TOKENS
# ---------------------------------------------------------------------------


def test_compose_typical_path_fits_token_budget():
    """
    The composed prompt for the substantive-no-name-no-risk path should fit
    within roughly 1300 tokens (~5200 chars, conservative 4-chars/token).
    The cap was raised 4600 → 5200 (NDJSON OUTPUT CONTRACT), → 5700 (general-
    medicine breadth + multi-system escalation), → 6800 when Layer 6 became a
    full consultation-flow spec (working assessment, convergence/completion,
    info-gain questioning, educational answer-first). If this keeps climbing,
    de-duplicate Layers 6 and 7 rather than lifting the cap again.

    The dermatology re-skin deliberately held this line: the derma persona was
    written to roughly the same length as the general-medicine one it replaced,
    so a specialty swap costs no extra budget.

    Raised 7000 -> 8400 for the EXPOSURE HISTORY layer (~1170 chars). That is
    new clinical capability, not bloat: dermatology is the specialty where the
    exposure IS often the diagnosis (contact dermatitis, photodermatoses,
    occupational hand eczema), and without the layer the model generalised
    about weather instead of using the readings it was given. Gated to
    history-taking intents, so education and drug turns pay nothing.
    """
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    chars = len(out)
    assert chars <= 8400, (
        f"Composed prompt is {chars} chars (~{chars // 4} tokens); "
        f"tighten layer text or re-evaluate budget."
    )


# ---------------------------------------------------------------------------
# Exposure history layer (dermatology)
# ---------------------------------------------------------------------------

class TestExposureHistoryLayer:
    @pytest.mark.parametrize("intent", [
        "symptom_query", "diagnosis_query", "followup_query",
        "risk_assessment", "prevention_query",
    ])
    def test_emitted_for_history_taking_intents(self, intent):
        assert "EXPOSURE HISTORY" in layer_exposure_history(query_type=intent)

    @pytest.mark.parametrize("intent", [
        "condition_explanation", "medication_query", "lab_interpretation",
        "comparison_query", "greeting", "",
    ])
    def test_omitted_where_a_review_is_noise(self, intent):
        assert layer_exposure_history(query_type=intent) == ""

    def test_covers_skin_specific_exposures(self):
        """Contact allergens and occupation are often the diagnosis in derma."""
        text = layer_exposure_history(query_type="symptom_query").lower()
        for factor in ("cosmetics", "detergent", "hair dye", "jewellery",
                       "occupational", "travel", "photosensitivity", "contacts"):
            assert factor in text, f"{factor} missing from exposure review"

    def test_forbids_settling_on_a_single_cause(self):
        text = layer_exposure_history(query_type="symptom_query")
        assert "COMBINE" in text and "do NOT stop there" in text

    def test_requires_citing_real_figures(self):
        text = layer_exposure_history(query_type="symptom_query")
        assert "cite the actual figures" in text and "LOCAL CONDITIONS" in text

    def test_uv_is_worth_raising_even_when_unrelated(self):
        """Sun protection is advice a dermatologist gives anyway."""
        text = layer_exposure_history(query_type="symptom_query")
        assert "High UV is worth raising even when the complaint is" in text

    def test_relevance_is_mechanistic_not_regional_deviation(self):
        """
        Two failure modes pull in opposite directions and the layer must hold
        both.

        Over-reading: Shillong sits near 98% humidity permanently, so raising
        it against a mole check is noise.

        Under-reading: the earlier wording ("normal for the region is NOT
        evidence") told the model to discount Leh's dew point of 4C precisely
        BECAUSE Leh is always dry — and it duly blamed harsh soaps for
        cracked knuckles in Ladakh.

        The resolution is that relevance follows the MECHANISM linking a
        reading to the presenting complaint, never the deviation from local
        normal — which is also what models.py states: skin responds to
        absolute conditions.
        """
        text = layer_exposure_history(query_type="symptom_query")
        # irrelevant to the complaint → still dropped
        assert "irrelevant to the complaint, ignore" in text
        # but regional normality alone is no longer grounds to discount
        assert "do NOT discount a reading merely because it is normal" in text
        assert "ABSOLUTE conditions" in text

    def test_reaches_the_composed_prompt(self):
        assert "EXPOSURE HISTORY" in compose_system_prompt(
            query_type="symptom_query", risk_level="none")

    def test_absent_from_composed_prompt_for_education(self):
        assert "EXPOSURE HISTORY" not in compose_system_prompt(
            query_type="condition_explanation", risk_level="none")


def test_regional_normality_does_not_discount_a_reading() -> None:
    """
    The layer previously said ambient humidity "normal for the region is NOT
    evidence about this patient" — which contradicted the service's own
    physiology (models.py: skin responds to ABSOLUTE conditions) and told the
    model to discount Leh's dew point of 4C precisely because Leh is always
    dry. That is what produced a Leh consultation blaming harsh soaps.
    """
    text = layer_exposure_history(query_type="symptom_query").lower()
    assert "absolute conditions" in text
    assert "normal for that region" in text
    assert "not evidence about this patient" not in text
