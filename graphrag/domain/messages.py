"""
Canned (non-LLM) answers as typed blocks, plus the terminal-turn rule.

The refuse / out-of-scope / emergency short-circuits never call the LLM, but the
transport is uniform NDJSON — so they must emit Blocks too, not raw strings.
These builders are the single source for that canned content; both the async
orchestrator and the legacy sync pipeline render them onto the same wire.

`is_terminal_turn` centralises the terminal definition. Stopping is
confidence-based: the gatekeeper estimates a 0–100 confidence in the leading
diagnosis each turn (`diagnostic_confidence`), and once it reaches
`settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD` the interview ends and the model
delivers its assessment. The diagnostic turn cap (`settings.MAX_DIAGNOSTIC_TURNS`)
is a hard backstop so a low-confidence conversation can't loop forever, and an
emergency redirect is always terminal. A terminal turn must not ask further
follow-up questions (enforced again in the validator).
"""

from __future__ import annotations

from typing import Any

from graphrag.config.settings import settings
from graphrag.schemas.blocks import (
    Block,
    NextStepsBlock,
    NextStepsData,
    SummaryBlock,
    SummaryData,
    WarningBlock,
    WarningData,
)

_REFUSAL_TEXT = (
    "I'm designed to assist only with healthcare-related questions. Please ask a "
    "medical or health-related question so I can help."
)

_EMERGENCY_TEXT = (
    "Your symptoms may indicate a serious or life-threatening condition. Do not "
    "wait — seek emergency care now."
)


def refusal_blocks() -> list[Block]:
    """Non-medical request refused. -> [summary]"""
    return [SummaryBlock(type="summary", data=SummaryData(text=_REFUSAL_TEXT))]


def out_of_scope_blocks() -> list[Block]:
    """Out-of-scope (non-health) request. -> [summary]"""
    return [SummaryBlock(type="summary", data=SummaryData(text=_REFUSAL_TEXT))]


def emergency_blocks() -> list[Block]:
    """Emergency redirect. -> [warning(critical), next_steps]"""
    return [
        WarningBlock(
            type="warning",
            data=WarningData(text=_EMERGENCY_TEXT, severity="critical"),
        ),
        NextStepsBlock(
            type="next_steps",
            data=NextStepsData(
                steps=[
                    "Call emergency services now — 112, or 108 / 102 for an ambulance.",
                    "Go to the nearest emergency room or hospital immediately.",
                    "If symptoms worsen while waiting, call back and report the change.",
                ]
            ),
        ),
    ]


_MH_VALIDATION_TEXT = (
    "I'm really glad you told me this, and I'm sorry you're carrying so much "
    "right now. What you're feeling is real, it matters, and you don't have to "
    "face it alone — reaching out took courage, and support is available."
)

_MH_IMMEDIATE_DANGER_TEXT = (
    "If you might act on these thoughts or you feel unsafe right now, please "
    "treat this as an emergency and get immediate help."
)


def mental_health_crisis_blocks() -> list[Block]:
    """
    Psychological crisis (suicidal ideation, self-harm, acute distress).

    Distinct from the physical emergency path: leads with empathy and
    validation, points to crisis-specific support lines, and only then covers
    imminent-danger escalation. -> [summary, warning(critical), next_steps]
    """
    return [
        SummaryBlock(type="summary", data=SummaryData(text=_MH_VALIDATION_TEXT)),
        WarningBlock(
            type="warning",
            data=WarningData(text=_MH_IMMEDIATE_DANGER_TEXT, severity="critical"),
        ),
        NextStepsBlock(
            type="next_steps",
            data=NextStepsData(
                steps=[
                    "If you're in immediate danger, call your local emergency "
                    "number now (112) or go to the nearest emergency room.",
                    "Talk to someone right now — Tele-MANAS, India's free 24/7 "
                    "mental-health line: dial 14416 or 1-800-891-4416.",
                    "KIRAN mental-health helpline: 1800-599-0019 (24/7, "
                    "multi-language). Outside India, contact your local crisis line.",
                    "If you can, reach out to someone you trust and stay with "
                    "them, and move away from anything you might use to harm yourself.",
                    "These feelings can ease with support — you deserve help, and "
                    "asking for it is a strong first step, not a weakness.",
                ]
            ),
        ),
    ]


def canned_blocks_for(final_action: str) -> list[Block]:
    """Map a gatekeeper short-circuit action to its canned block list."""
    if final_action == "emergency_redirect":
        return emergency_blocks()
    if final_action == "mental_health_crisis":
        return mental_health_crisis_blocks()
    # "refuse" (and any other non-LLM short-circuit) -> refusal/out-of-scope.
    return refusal_blocks()


def parse_diagnostic_confidence(raw: object) -> float | None:
    """
    Best-effort parse of the gatekeeper's ``diagnostic_confidence`` into a 0–100
    score, or ``None`` when it is missing/unparseable.

    Accepts ints, floats, and numeric strings ("85", "85%"). A fractional value
    in (0, 1) is read as a probability and scaled to 0–100 (0.85 -> 85). Booleans
    are rejected so a stray ``True`` never reads as 100.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        raw = raw.strip().rstrip("%").strip()
        try:
            raw = float(raw)
        except ValueError:
            return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        if 0.0 < val < 1.0:  # probability form, e.g. 0.85
            val *= 100.0
        return max(0.0, min(100.0, val))
    return None


def is_terminal_turn(
    *,
    turn_count: int,
    analysis: dict[str, Any] | None,
    confidence_threshold: int | None = None,
) -> bool:
    """
    Whether this is a closing turn that must not emit follow-up questions.

    Confidence-based stopping is the primary rule. Terminal when, in priority
    order:
        1. the gatekeeper flagged an emergency redirect (always closes), or
        2. the estimated ``diagnostic_confidence`` for the leading diagnosis has
           reached ``confidence_threshold`` (defaults to
           ``settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD``) — enough has been
           gathered, so stop asking and present the assessment, or
        3. the conversation has hit the diagnostic turn cap (a hard backstop so a
           low-confidence case can't interview forever).
    """
    analysis = analysis or {}

    # Canned crisis short-circuits are always closing turns — no follow-ups.
    if analysis.get("final_action") in {"emergency_redirect", "mental_health_crisis"}:
        return True

    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD
    )
    confidence = parse_diagnostic_confidence(analysis.get("diagnostic_confidence"))
    if confidence is not None and confidence >= threshold:
        return True

    if turn_count >= settings.MAX_DIAGNOSTIC_TURNS:
        return True

    return False
