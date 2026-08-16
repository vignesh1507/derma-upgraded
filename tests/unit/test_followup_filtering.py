"""
Follow-up questions must not ask for data we already measured.

The analyser proposes follow-ups at stage -1, before retrieval and before the
environmental lookup, so it cannot know that [LOCAL CONDITIONS] were attached.
Without a filter a turn can state the UV index and humidity in the answer and
then ask the patient what the UV and humidity are.

MOST OF THIS FILE GUARDS THE OTHER DIRECTION. In dermatology the personal sun
history — hours outdoors, sunscreen, covering — is the highest-value question
the service asks, and no UV reading can answer it: UV 11 over the city says
nothing about whether THIS patient stood under it. Silently suppressing "do you
use sunscreen?" would be a far worse defect than the redundancy being removed,
so the personal list wins every tie and is tested far harder than the ambient
one.
"""

from __future__ import annotations

import pytest

from app.services.orchestration.pipeline import AsyncOrchestrator

BLOCK = (
    "[LOCAL CONDITIONS]\n"
    "UV index today (max): 11 — Extreme\n"
    "Weather: 38C (feels 43C), humidity 22%, dew point 4C, sunny"
)

_drop = AsyncOrchestrator._drop_answered_followups


class TestSunBehaviourIsNeverSuppressed:
    """The regression that would matter. Every one of these must survive."""

    @pytest.mark.parametrize("q", [
        "How much sun exposure do you get during the day?",
        "Do you use sunscreen, and what SPF?",
        "How long are you outdoors each day?",
        "Do you wear a hat or covering when you go outside?",
        "Are you outside during the midday hours?",
        "Do you apply sunblock before going out?",
        "Do you seek shade when working outside?",
        "How much time do you spend in the sun?",
        "Is your face covered when you are outdoors?",
        "On high-UV days, do you still go out without sunscreen?",
        "Even when the UV index is high, do you spend time outside?",
    ])
    def test_personal_sun_history_survives(self, q):
        assert _drop([q], BLOCK) == [q], "suppressed a sun-behaviour question"


class TestOtherPersonalExposuresSurvive:
    @pytest.mark.parametrize("q", [
        "Have you started any new cream, cosmetic or skincare product?",
        "What soap or detergent do you use?",
        "Do you take very hot showers?",
        "Do you wear gloves when handling chemicals at work?",
        "Any new jewellery or nickel contact?",
        "Have you been swimming in chlorinated water?",
        "Do you use a room heater or air conditioning at home?",
        "Are you taking any medication that could cause photosensitivity?",
        "Is there a family history of eczema or psoriasis?",
        "Do you have any known allergies?",
        "Have you travelled anywhere recently?",
    ])
    def test_kept(self, q):
        assert _drop([q], BLOCK) == [q]


class TestAmbientQuestionsAreDropped:
    @pytest.mark.parametrize("q", [
        "What is the UV index where you are?",
        "What is the air quality like in your city?",
        "How humid is it where you live?",
        "Has the weather changed recently?",
        "What is the climate like there?",
        "What specific environmental factors are present in your surroundings?",
        "What is the current AQI?",
    ])
    def test_dropped_when_conditions_are_known(self, q):
        assert _drop([q], BLOCK) == []

    def test_kept_when_nothing_was_measured(self):
        """Fail-open: with no reading attached we still need to ask."""
        q = ["What is the UV index where you are?"]
        assert _drop(q, "") == q


class TestClinicalQuestionsAreUntouched:
    @pytest.mark.parametrize("q", [
        "Can you describe what you mean by 'bad skin'?",
        "Are the patches scaly, weeping, or blistered?",
        "How long have you had these lesions?",
        "Is the rash itchy or painful?",
        "Has it spread to other parts of your body?",
    ])
    def test_morphology_questions_pass_through(self, q):
        """Derma's analyser mostly asks these — the filter must not see them."""
        assert _drop([q], BLOCK) == [q]


class TestEdges:
    def test_empty_questions(self):
        assert _drop([], BLOCK) == []

    def test_no_block_no_questions(self):
        assert _drop([], "") == []

    def test_order_preserved(self):
        qs = ["Do you use sunscreen?", "How long has this been going on?"]
        assert _drop(qs, BLOCK) == qs

    def test_case_insensitive(self):
        assert _drop(["WHAT IS THE AIR QUALITY THERE?"], BLOCK) == []

    def test_case_insensitive_personal_still_wins(self):
        q = ["DO YOU USE SUNSCREEN ON HIGH UV INDEX DAYS?"]
        assert _drop(q, BLOCK) == q
