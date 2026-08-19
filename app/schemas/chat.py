"""Chat request/response/stream schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field


class IdentityEnvelope(BaseModel):
    """
    New identity contract the Backend MAY send alongside (or instead of) the
    legacy top-level ``user_id``/``session_id``. Fully optional and additive —
    absent → the legacy fields are used, so existing callers are unaffected.
    """

    # ``patient_id`` is the authenticated Mongo User._id; ``user_id`` is accepted
    # as an alias so either naming works during the Backend rollout.
    patient_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    consumer_id: str | None = None   # which upstream consumer/service initiated


class ChatRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        validation_alias=AliasChoices("query", "message"),
    )
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    # When provided, the orchestrator loads the user's episodic memory before
    # the LLM call and ingests the turn after the answer. When omitted, the
    # episodic stage is skipped (parity with the CLI's --user-id flag).
    user_id: str | None = None
    # New identity contract (optional). Preferred over the legacy fields above
    # when ENABLE_IDENTITY_V1 is on; ignored/absent keeps legacy behaviour.
    identity: IdentityEnvelope | None = None
    # Optional patient location for environmental context (UV, moisture, AQI).
    # "City, Country" or "lat,lon". Ignored unless ENVIRONMENTAL_CONTEXT_ENABLED.
    # Note: Indian PIN codes are NOT supported by the upstream provider.
    location: str | None = Field(default=None, max_length=120)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    request_id: str
    analysis: dict[str, Any] | None = None
    timing_ms: dict[str, int] = Field(default_factory=dict)
    routing: dict[str, Any] = Field(default_factory=dict)
    followup_questions: list[str] = Field(default_factory=list)
    # True once the consultation has reached a concluded answer — the client may
    # then offer "Show this to your doctor" (the SOAP note at POST /chat/soap).
    show_doctor_summary: bool = False


class ChatStreamEvent(BaseModel):
    type: Literal["chunk", "done", "error", "meta"]
    data: str | None = None
    timing_ms: dict[str, int] | None = None
    error: dict[str, str] | None = None


class MediaInfo(BaseModel):
    """Metadata-only view of a processed upload (never carries raw bytes)."""

    category: str
    route: str
    mime_type: str
    size_bytes: int
    filename: str | None = None
    storage_uri: str | None = None
    caption: str | None = None
    extracted_facts: list[str] = Field(default_factory=list)


class ImageChatResponse(ChatResponse):
    """A `/chat/image` answer: a normal chat response plus the upload metadata."""

    media: MediaInfo


class SoapRequest(BaseModel):
    """Trigger a fresh doctor-facing SOAP note for an existing session."""

    session_id: str = Field(min_length=1)
    user_id: str | None = None
    identity: IdentityEnvelope | None = None


class SoapNote(BaseModel):
    """
    Doctor-facing SOAP note, generated on demand from the latest conversation.

    Grounded strictly in the conversation — never fabricated. Each section is
    plain prose; `unavailable` explicitly names clinically relevant information
    the conversation did not provide (e.g. "no vital signs recorded").
    """

    subjective: str
    objective: str
    assessment: str
    plan: str
    unavailable: list[str] = Field(default_factory=list)
    session_id: str
    request_id: str
    generated_at: str  # ISO-8601 UTC, stamped by the route
