"""
state_extractor.py
──────────────────
Updated state extractor to handle chronic conditions, allergies, and persistence
of previous concerns for the Enervera memory layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import Message, RiskLevel, Role, SessionMemory, StructuredState


# ============================================================================
# Heuristic Pattern Registry
# ============================================================================

# NOTE ON SCOPE (dermatology deployment)
# The LLM analyzer (graphrag/query_understanding/analyzer.py) does the primary
# entity extraction and merges via merge_analysis_entities, so this regex layer
# is defence-in-depth: it keeps state populated on no-LLM / analyzer-failure
# paths. The skin terms below were added for that reason; the general-medicine
# terms are retained because patients volunteer them as comorbidity context
# ("I'm diabetic and I have a foot ulcer").

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    # ── Dermatological ──────────────────────────────────────────────────────
    "itching":            [r"\bitch(y|ing|iness)?\b", r"\bkhujli\b", r"\bpruritus\b",
                           r"\bscratch(ing)?\b"],
    "rash":               [r"\brash(es)?\b", r"\bred patch(es)?\b", r"\beruption\b"],
    "scaling":            [r"\bscal(y|ing|es)\b", r"\bflak(y|ing|es)\b", r"\bpeeling\b"],
    "dryness":            [r"\bdry skin\b", r"\bxerosis\b", r"\brough skin\b"],
    "papules":            [r"\bpapule(s)?\b", r"\bbump(s)?\b", r"\bpimple(s)?\b",
                           r"\bzit(s)?\b", r"\bacne\b"],
    "pustules":           [r"\bpustule(s)?\b", r"\bboil(s)?\b", r"\bpus\b",
                           r"\bwhitehead(s)?\b"],
    "blisters":           [r"\bblister(s|ing)?\b", r"\bvesicle(s)?\b", r"\bbulla(e)?\b"],
    "wheals":             [r"\bwheal(s)?\b", r"\bhives\b", r"\burticaria\b",
                           r"\braised welts?\b"],
    "plaques":            [r"\bplaque(s)?\b", r"\bthick(ened)? patch(es)?\b"],
    "oozing":             [r"\booz(e|ing)\b", r"\bweeping\b", r"\bdischarge\b",
                           r"\bcrust(ing|ed)?\b"],
    "skin_ulcer":         [r"\bskin ulcer\b", r"\bnon-?healing (sore|wound)\b",
                           r"\bopen sore\b"],
    "hyperpigmentation":  [r"\bdark (spot|patch)(es)?\b", r"\bpigmentation\b",
                           r"\bhyperpigment\w*\b", r"\bmelasma\b", r"\btanning\b"],
    "hypopigmentation":   [r"\bwhite patch(es)?\b", r"\bwhite spot(s)?\b",
                           r"\bdepigment\w*\b", r"\bloss of colou?r\b"],
    "hair_loss":          [r"\bhair ?(fall|loss)\b", r"\blosing hair\b",
                           r"\bbald(ing|ness)?\b", r"\bbald patch(es)?\b",
                           r"\bthinning hair\b", r"\balopecia\b"],
    "dandruff":           [r"\bdandruff\b", r"\bflaky scalp\b", r"\bitchy scalp\b"],
    "nail_changes":       [r"\bnail (pitting|discolou?ration|thickening)\b",
                           r"\bbrittle nails?\b", r"\bnail fungus\b",
                           r"\bdiscolou?red nails?\b"],
    "burning_skin":       [r"\bburning (sensation|skin)\b", r"\bsting(ing)?\b"],
    "swelling_skin":      [r"\bswollen\b", r"\bswelling\b", r"\bangioedema\b",
                           r"\bpuffy\b"],
    "photosensitivity":   [r"\bphotosensitiv\w*\b", r"\bsun sensitiv\w*\b",
                           r"\bworse in (the )?sun\b"],
    "ring_lesion":        [r"\bring ?-?(shaped|like)\b", r"\bring ?worm\b",
                           r"\bcircular (patch|lesion)\b"],

    # ── General / systemic ──────────────────────────────────────────────────
    "fever":              [r"\bfever\b", r"\bhigh temperature\b", r"\btemperature\b"],
    "chills":             [r"\bchill(s|ing)?\b", r"\bshiver(ing)?\b"],
    "sore_throat":        [r"\bsore throat\b", r"\bthroat pain\b", r"\bthroat ache\b"],
    "cough":              [r"\bcough(ing)?\b"],
    "shortness_of_breath":[r"\bshortness of breath\b", r"\bbreathing difficult\b", r"\bcan'?t breathe\b"],
    "chest_pain":         [r"\bchest pain\b", r"\bchest tightness\b"],
    "headache":           [r"\bheadache\b", r"\bhead pain\b"],
    "fatigue":            [r"\bfatigue\b", r"\btired\b", r"\bexhausted\b"],
    "nausea":             [r"\bnausea\b", r"\bfeeling sick\b"],
    "dizziness":          [r"\bdizzy\b", r"\bdizziness\b"],
}

# Body sites — morphology plus site is what actually drives a dermatological
# differential ("scaly plaques on the elbows" vs "on the flexures").
BODY_SITE_PATTERNS: dict[str, list[str]] = {
    "scalp":      [r"\bscalp\b", r"\bhead skin\b"],
    "face":       [r"\bface\b", r"\bfacial\b", r"\bcheek(s)?\b", r"\bforehead\b"],
    "trunk":      [r"\btrunk\b", r"\bchest\b", r"\bback\b", r"\babdomen\b", r"\bstomach\b"],
    "arms":       [r"\barm(s)?\b", r"\belbow(s)?\b", r"\bforearm(s)?\b"],
    "hands":      [r"\bhand(s)?\b", r"\bpalm(s)?\b", r"\bfinger(s)?\b"],
    "legs":       [r"\bleg(s)?\b", r"\bknee(s)?\b", r"\bshin(s)?\b", r"\bthigh(s)?\b"],
    "feet":       [r"\bfoot\b", r"\bfeet\b", r"\bsole(s)?\b", r"\bbetween (the )?toes\b",
                   r"\btoe(s)?\b"],
    "groin":      [r"\bgroin\b", r"\binner thigh\b", r"\bprivate area\b"],
    "flexures":   [r"\bflexur\w*\b", r"\bskin fold(s)?\b", r"\bunderarm(s)?\b",
                   r"\barmpit(s)?\b", r"\bbehind the knee(s)?\b"],
    "nails":      [r"\bnail(s)?\b", r"\bfingernail(s)?\b", r"\btoenail(s)?\b"],
}

# Distinguishing between acute conditions and chronic conditions
CHRONIC_PATTERNS: dict[str, list[str]] = {
    # ── Chronic skin disease ────────────────────────────────────────────────
    "psoriasis":          [r"\bpsoriasis\b", r"\bpsoriatic\b"],
    "atopic_dermatitis":  [r"\batopic dermatitis\b", r"\beczema\b", r"\batopy\b"],
    "vitiligo":           [r"\bvitiligo\b", r"\bleucoderma\b", r"\bleukoderma\b"],
    "rosacea":            [r"\brosacea\b"],
    "chronic_urticaria":  [r"\bchronic (urticaria|hives)\b"],
    "acne_vulgaris":      [r"\bacne vulgaris\b", r"\bchronic acne\b"],
    "lichen_planus":      [r"\blichen planus\b"],
    "hidradenitis":       [r"\bhidradenitis\b"],
    "androgenetic_alopecia": [r"\bandrogenetic alopecia\b", r"\bmale ?pattern\b",
                              r"\bfemale ?pattern\b"],

    # ── General comorbidities ───────────────────────────────────────────────
    "diabetes":           [r"\bdiabetes\b", r"\bdiabetic\b"],
    "hypertension":       [r"\bhypertension\b", r"\bhigh blood pressure\b", r"\bhigh bp\b"],
    "asthma":             [r"\basthma\b", r"\basthmatic\b"],
    "heart_disease":      [r"\bheart disease\b", r"\bcardiac issue\b"],
    "thyroid":            [r"\bthyroid\b"],
    "arthritis":          [r"\barthritis\b", r"\bjoint pain\b"],
}

CONDITION_PATTERNS: dict[str, list[str]] = {
    # ── Acute / named skin conditions ───────────────────────────────────────
    "tinea":              [r"\btinea\b", r"\bring ?worm\b", r"\bdhobi ?itch\b",
                           r"\bjock itch\b", r"\bathlete'?s foot\b"],
    "candidiasis":        [r"\bcandid(a|iasis)\b", r"\byeast infection\b"],
    "scabies":            [r"\bscabies\b", r"\bkhujli mite\b", r"\bmites?\b"],
    "impetigo":           [r"\bimpetigo\b"],
    "cellulitis":         [r"\bcellulitis\b", r"\berysipelas\b"],
    "folliculitis":       [r"\bfolliculitis\b"],
    "herpes":             [r"\bherpes\b", r"\bcold sore(s)?\b", r"\bshingles\b",
                           r"\bzoster\b"],
    "warts":              [r"\bwart(s)?\b", r"\bverruca\b", r"\bmolluscum\b"],
    "contact_dermatitis": [r"\bcontact dermatitis\b", r"\ballergic dermatitis\b"],
    "seborrhoeic_derm":   [r"\bseborrh?oeic\b", r"\bseborrheic\b"],
    "drug_eruption":      [r"\bdrug (rash|eruption|reaction)\b",
                           r"\bstevens[- ]?johnson\b", r"\bsjs\b"],
    "leprosy":            [r"\bleprosy\b", r"\bhansen'?s\b"],
    "melanoma":           [r"\bmelanoma\b"],
    "skin_cancer":        [r"\bskin cancer\b", r"\bbasal cell\b",
                           r"\bsquamous cell\b", r"\bbcc\b", r"\bscc\b"],
    "keloid":             [r"\bkeloid\b", r"\bhypertrophic scar\b"],

    # ── General ─────────────────────────────────────────────────────────────
    "flu":                [r"\bflu\b", r"\binfluenza\b"],
    "strep_throat":       [r"\bstrep throat\b"],
    "covid":              [r"\bcovid\b", r"\bcoronavirus\b"],
    "infection":          [r"\binfection\b", r"\binfected\b"],
    "migraine":           [r"\bmigraine\b"],
}

ALLERGY_PATTERNS: dict[str, list[str]] = {
    "penicillin":    [r"\ballergic to penicillin\b", r"\bpenicillin allergy\b"],
    "pollen":        [r"\bpollen\b", r"\bhay fever\b"],
    "dust":          [r"\bdust allergy\b", r"\ballergic to dust\b"],
    "peanuts":       [r"\bpeanut allergy\b", r"\ballergic to peanuts\b"],
    "shellfish":     [r"\bshellfish allergy\b"],
    # ── Contact allergens — the derma-specific ones ─────────────────────────
    "nickel":        [r"\bnickel\b", r"\ballergic to (metal|jewell?ery)\b"],
    "fragrance":     [r"\bfragrance allergy\b", r"\ballergic to (perfume|fragrance)\b"],
    "hair_dye":      [r"\bhair ?dye allergy\b", r"\ballergic to hair ?dye\b", r"\bppd\b"],
    "latex":         [r"\blatex allergy\b", r"\ballergic to latex\b"],
    "cosmetics":     [r"\ballergic to (cosmetics|makeup)\b", r"\bcosmetic allergy\b"],
    "sulfa":         [r"\bsulfa (allergy|drugs?)\b", r"\ballergic to sulfa\b"],
}

DRUG_PATTERNS: dict[str, list[str]] = {
    # ── Topical steroids ────────────────────────────────────────────────────
    "hydrocortisone":[r"\bhydrocortisone\b"],
    "betamethasone": [r"\bbetamethasone\b", r"\bbetnovate\b"],
    "clobetasol":    [r"\bclobetasol\b", r"\btenovate\b", r"\bdermovate\b"],
    "mometasone":    [r"\bmometasone\b", r"\belocon\b"],
    # ── Antifungals ─────────────────────────────────────────────────────────
    "clotrimazole":  [r"\bclotrimazole\b", r"\bcandid\b"],
    "ketoconazole":  [r"\bketoconazole\b", r"\bnizoral\b"],
    "terbinafine":   [r"\bterbinafine\b", r"\blamisil\b"],
    "itraconazole":  [r"\bitraconazole\b", r"\bsporanox\b"],
    "fluconazole":   [r"\bfluconazole\b"],
    # ── Acne ────────────────────────────────────────────────────────────────
    "benzoyl_peroxide":[r"\bbenzoyl peroxide\b", r"\bbenzac\b"],
    "adapalene":     [r"\badapalene\b", r"\bdifferin\b"],
    "tretinoin":     [r"\btretinoin\b", r"\bretino\b"],
    "isotretinoin":  [r"\bisotretinoin\b", r"\baccutane\b", r"\bsotret\b"],
    "clindamycin":   [r"\bclindamycin\b", r"\bclindac\b"],
    # ── Antihistamines ──────────────────────────────────────────────────────
    "cetirizine":    [r"\bcetirizine\b", r"\bcetzine\b"],
    "levocetirizine":[r"\blevocetirizine\b", r"\blevocet\b"],
    "fexofenadine":  [r"\bfexofenadine\b", r"\ballegra\b"],
    "hydroxyzine":   [r"\bhydroxyzine\b", r"\batarax\b"],
    # ── Antibacterials / scabies / psoriasis / hair ─────────────────────────
    "mupirocin":     [r"\bmupirocin\b", r"\bbactroban\b", r"\bt-?bact\b"],
    "fusidic_acid":  [r"\bfusidic acid\b", r"\bfucidin\b"],
    "doxycycline":   [r"\bdoxycycline\b"],
    "permethrin":    [r"\bpermethrin\b"],
    "ivermectin":    [r"\bivermectin\b"],
    "calcipotriol":  [r"\bcalcipotriol\b", r"\bdaivonex\b"],
    "methotrexate":  [r"\bmethotrexate\b"],
    "tacrolimus":    [r"\btacrolimus\b", r"\bprotopic\b"],
    "minoxidil":     [r"\bminoxidil\b", r"\bmintop\b", r"\brogaine\b"],
    "finasteride":   [r"\bfinasteride\b"],
    "salicylic_acid":[r"\bsalicylic acid\b"],
    "emollient":     [r"\bemollient(s)?\b", r"\bmoisturis\w*\b", r"\bmoisturiz\w*\b",
                      r"\bvaseline\b", r"\bpetroleum jelly\b"],
    "sunscreen":     [r"\bsunscreen\b", r"\bsunblock\b", r"\bspf\b"],
    # ── General ─────────────────────────────────────────────────────────────
    "paracetamol":   [r"\bparacetamol\b", r"\bacetaminophen\b", r"\btylenol\b"],
    "ibuprofen":     [r"\bibuprofen\b", r"\badvil\b", r"\bnurofen\b"],
    "aspirin":       [r"\baspirin\b"],
    "insulin":       [r"\binsulin\b"],
    "amoxicillin":   [r"\bamoxicillin\b"],
}

SEVERITY_PATTERNS: dict[str, list[str]] = {
    "mild":     [r"\bmild\b", r"\bslight\b"],
    "moderate": [r"\bmoderate\b", r"\bmedium\b"],
    "severe":   [r"\bsevere\b", r"\bextreme\b", r"\bintense\b", r"\bvery bad\b"],
}

DURATION_RE = re.compile(
    r"(?:for|since|over|past|last)\s+"
    r"(\d+\s+(?:second|minute|hour|day|week|month|year)s?|yesterday|this morning)",
    re.IGNORECASE,
)

AGE_RE  = re.compile(r"\b(\d{1,3})\s*(?:year(?:s)?\s*old|y\.?o\.?)\b", re.IGNORECASE)
SEX_RE  = re.compile(r"\b(male|female|man|woman)\b", re.IGNORECASE)

SEX_NORMALISE = {"man": "male", "woman": "female"}

# ── Name extraction ────────────────────────────────────────────────────────
# We pull a first name when the patient explicitly introduces themselves. The
# answer prompt uses it to address the user naturally instead of as "patient".
#
# High-confidence patterns first — these are explicit declarations and almost
# never produce false positives.
_NAME_EXPLICIT_RES: list[re.Pattern[str]] = [
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthe name'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bname'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Z][a-zA-Z'\-]{1,30})(?:\s+speaking|\s+here|\s*[,.])"),
    re.compile(r"\bi go by ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
]

# Lower-confidence: "I am X" / "I'm X". Only trust if X looks like a name —
# capitalised in the original text AND not a common adjective / state word.
_NAME_SOFT_RE = re.compile(r"\b[Ii]\s*'?\s*[am]{1,2}\s+([A-Z][a-zA-Z'\-]{1,30})\b")

# Common words that can follow "I'm" / "I am" but are NOT names. Lowercased.
_NAME_STOPWORDS: frozenset[str] = frozenset({
    # states / feelings
    "sick", "tired", "fine", "ok", "okay", "good", "bad", "well", "great",
    "happy", "sad", "worried", "scared", "confused", "anxious", "depressed",
    "stressed", "exhausted", "hungry", "thirsty", "dizzy", "nauseous", "dying",
    "fasting", "bleeding", "burning", "shaking", "freezing",
    # statuses
    "married", "single", "pregnant", "diabetic", "allergic", "asthmatic",
    "hypertensive", "vegetarian", "vegan", "lost", "ready", "back", "done",
    "late", "early", "here", "there", "home", "outside", "indoors",
    # progressive verbs after "I'm"
    "having", "feeling", "going", "trying", "looking", "doing", "taking",
    "thinking", "wondering", "asking", "calling", "writing", "experiencing",
    "suffering", "noticing", "starting", "ending", "drinking", "eating",
    # other common
    "afraid", "unsure", "unable", "old", "young", "new", "sorry", "sure",
    "really", "always", "never", "still", "just", "also", "very",
})


def _extract_name(text: str) -> str | None:
    """
    Return the first sensible name found in `text`, or None.

    Prefers explicit declarations ("my name is X", "call me X") before the
    softer "I'm X" pattern. Soft matches are filtered through a stopword list
    so phrases like "I'm sick" / "I'm Diabetic" never produce a "name".
    """
    for pat in _NAME_EXPLICIT_RES:
        m = pat.search(text)
        if m:
            return _format_name(m.group(1))

    m = _NAME_SOFT_RE.search(text)
    if m:
        candidate = m.group(1)
        if candidate.lower() not in _NAME_STOPWORDS:
            return _format_name(candidate)
    return None


def _format_name(raw: str) -> str:
    """Normalise to Title-Case, preserving internal apostrophes and hyphens."""
    return "-".join(part[:1].upper() + part[1:].lower() for part in raw.split("-"))

# ============================================================================
# Helpers
# ============================================================================

def _match_patterns(text: str, pattern_dict: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    for name, patterns in pattern_dict.items():
        for pat in patterns:
            if re.search(pat, lower):
                found.append(name)
                break
    return found

def _extract_demographics(text: str) -> dict[str, Any]:
    demo: dict[str, Any] = {}
    age_m = AGE_RE.search(text)
    if age_m: demo["age"] = int(age_m.group(1))
    sex_m = SEX_RE.search(text)
    if sex_m:
        val = sex_m.group(1).lower()
        demo["sex"] = SEX_NORMALISE.get(val, val)
    name = _extract_name(text)
    if name:
        demo["name"] = name
    return demo

def _deduplicate(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# ============================================================================
# State Extraction Logic
# ============================================================================

@dataclass
class RawEntities:
    symptoms:           list[str] = field(default_factory=list)
    conditions:         list[str] = field(default_factory=list)
    chronic_conditions: list[str] = field(default_factory=list)
    allergies:          list[str] = field(default_factory=list)
    drugs:              list[str] = field(default_factory=list)
    severity:           list[str] = field(default_factory=list)
    duration:           list[str] = field(default_factory=list)
    demographics:       dict[str, Any] = field(default_factory=dict)
    preferences:        dict[str, Any] = field(default_factory=dict)
    risk_level:         RiskLevel = RiskLevel.NONE

    def all_named_entities(self) -> list[str]:
        """Flat list of all recognised medical terms for discussed_entities."""
        return self.symptoms + self.conditions + self.drugs + self.chronic_conditions + self.allergies

def extract_entities(text: str, message: Message | None = None) -> RawEntities:
    symptoms   = _match_patterns(text, SYMPTOM_PATTERNS)
    conditions = _match_patterns(text, CONDITION_PATTERNS)
    chronic    = _match_patterns(text, CHRONIC_PATTERNS)
    allergies  = _match_patterns(text, ALLERGY_PATTERNS)
    drugs      = _match_patterns(text, DRUG_PATTERNS)
    severity   = _match_patterns(text, SEVERITY_PATTERNS)
    
    duration = [m.group(0).strip() for m in DURATION_RE.finditer(text)]
    demo = _extract_demographics(text)

    risk = RiskLevel.NONE
    if message and message.risk_level:
        risk = message.risk_level
    elif set(symptoms) & {"chest_pain", "shortness_of_breath"}:
        risk = RiskLevel.CRITICAL

    return RawEntities(
        symptoms=symptoms,
        conditions=conditions,
        chronic_conditions=chronic,
        allergies=allergies,
        drugs=drugs,
        severity=severity,
        duration=duration,
        demographics=demo,
        risk_level=risk
    )

def update_preferences(state: StructuredState, patch: RawEntities) -> dict[str, Any]:
    merged = dict(state.preferences or {})
    for key, val in patch.preferences.items():
        merged[key] = val
    return merged

def merge_state(existing: StructuredState, patch: RawEntities) -> StructuredState:
    data = existing.model_copy(deep=True)

    data.symptoms = _deduplicate(data.symptoms + patch.symptoms)
    data.conditions = _deduplicate(data.conditions + patch.conditions)
    data.chronic_conditions = _deduplicate(data.chronic_conditions + patch.chronic_conditions)
    data.allergies = _deduplicate(data.allergies + patch.allergies)
    data.drugs = _deduplicate(data.drugs + patch.drugs)
    data.severity = _deduplicate(data.severity + patch.severity)
    data.duration = _deduplicate(data.duration + patch.duration)

    for k, v in patch.demographics.items():
        data.demographics[k] = v

    # Preferences merge
    data.preferences = update_preferences(data, patch)

    # Maintain a history of concerns for context-aware RAG
    if patch.symptoms:
        data.previous_concerns = _deduplicate(data.previous_concerns + patch.symptoms)

    # Risk only escalates
    risk_order = ["none", "low", "medium", "high", "critical"]
    patch_risk = patch.risk_level.value if hasattr(patch.risk_level, "value") else str(patch.risk_level)
    existing_risk = data.risk_level.value if hasattr(data.risk_level, "value") else str(data.risk_level)

    if risk_order.index(patch_risk.lower()) > risk_order.index(existing_risk.lower()):
        data.risk_level = patch.risk_level

    data.discussed_entities = _deduplicate(
        data.discussed_entities + patch.all_named_entities()
    )

    return data


def extract_state(session: SessionMemory, message: Message) -> StructuredState:
    if message.role != Role.USER:
        return session.state

    raw = extract_entities(message.content, message)
    updated = merge_state(session.state, raw)

    if message.query_type:
        updated.active_task = message.query_type
        updated.last_intent = message.query_type

    return updated


def _clean_entity_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip().lower() for v in values if str(v).strip()]


def _drop_negated(items: list[str], negated: list[str]) -> list[str]:
    """Remove any item that contains a negated finding (e.g. negated 'fever'
    drops 'fever' and 'high fever')."""
    return [x for x in items if not any(n in str(x).strip().lower() for n in negated)]


def merge_analysis_entities(
    state: StructuredState, analysis: dict[str, Any] | None
) -> StructuredState:
    """
    Fold the gatekeeper analyzer's LLM-extracted ``medical_entities`` into the
    session state — the reliable extraction path.

    The regex extractor under-captures (it misses free-text symptoms like
    "watery eyes", most durations, and measured severities), so the analyzer's
    per-turn LLM extraction is the trustworthy signal for "what has the patient
    already told us". This keeps conversation state accurate so the answer model
    never re-asks and can consolidate on time. Also applies NEGATIONS: findings
    the patient explicitly denied ("no fever", "no swelling") are removed from
    state so they aren't carried as active symptoms. Idempotent + deduplicated;
    a no-op when there's nothing to add.
    """
    entities = (analysis or {}).get("medical_entities") or {}

    patch = RawEntities(
        symptoms=_clean_entity_list(entities.get("symptoms")),
        drugs=_clean_entity_list(entities.get("drugs")),
        conditions=_clean_entity_list(entities.get("conditions")),
        allergies=_clean_entity_list(entities.get("allergies")),
        duration=_clean_entity_list(entities.get("duration")),
        severity=_clean_entity_list(entities.get("severity")),
    )
    negated = _clean_entity_list(entities.get("negated"))

    if not (patch.all_named_entities() or patch.duration or patch.severity or negated):
        return state

    merged = merge_state(state, patch)
    if negated:
        merged = merged.model_copy(deep=True)
        merged.symptoms = _drop_negated(merged.symptoms, negated)
        merged.conditions = _drop_negated(merged.conditions, negated)
        merged.previous_concerns = _drop_negated(merged.previous_concerns, negated)
    return merged
