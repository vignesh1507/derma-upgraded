"""
Chat routes.

POST /chat                — single-shot answer (JSON in, JSON out, prose)
POST /chat/stream         — SSE stream of prose tokens
POST /chat/blocks         — NDJSON stream of typed UI blocks
POST /chat/stream/blocks  — SSE stream of the same typed UI blocks
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep
from app.identity import IdentityContext
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ImageChatResponse,
    MediaInfo,
    SoapNote,
    SoapRequest,
)
from app.services.media import MediaArtifact, MediaValidationError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _resolve_identity(
    *,
    ctx: ContainerDep,
    request_id: str,
    session_id: str,
    user_id: str | None,
    envelope=None,
) -> IdentityContext:
    """
    Build the canonical IdentityContext from either the new ``identity`` envelope
    or the legacy ``user_id``/``session_id`` fields. Parsing always accepts both;
    ``ENABLE_IDENTITY_V1`` only controls whether the envelope is preferred.
    """
    return IdentityContext.resolve(
        request_id=request_id,
        legacy_session_id=session_id,
        legacy_user_id=user_id,
        envelope_session_id=(envelope.session_id if envelope else None),
        envelope_patient_id=((envelope.patient_id or envelope.user_id) if envelope else None),
        envelope_consumer_id=(envelope.consumer_id if envelope else None),
        identity_v1_enabled=ctx.settings.ENABLE_IDENTITY_V1,
    )


def _media_info(artifact: MediaArtifact) -> MediaInfo:
    """Project the internal artifact onto the public, bytes-free response shape."""
    return MediaInfo(
        category=artifact.category.value,
        route=artifact.route.value,
        mime_type=artifact.mime_type,
        size_bytes=artifact.size_bytes,
        filename=artifact.filename,
        storage_uri=artifact.storage_uri,
        caption=artifact.caption,
        extracted_facts=artifact.extracted_facts,
    )


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(req: ChatRequest, request: Request, ctx: ContainerDep) -> ChatResponse:
    """
    Run one full pipeline turn and return the answer + per-stage timing.

    The orchestrator pulls session memory, runs analyzer + retrieval + KG +
    optional episodic context, then asks Gemini for a non-streaming answer.
    """
    request_id = _request_id(request)
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=req.session_id,
        user_id=req.user_id, envelope=req.identity,
    )
    result = await ctx.orchestrator.run(
        query=req.query, identity=identity, location=req.location
    )
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        request_id=result.request_id,
        # The gatekeeper `analysis` block is internal; only surface it when
        # diagnostics are explicitly enabled (default off in production).
        analysis=result.analysis if ctx.settings.EXPOSE_DIAGNOSTICS else None,
        timing_ms=result.timing_ms,
        routing=result.routing,
        followup_questions=result.followup_questions,
        show_doctor_summary=result.show_doctor_summary,
    )


@router.post(
    "/chat/image",
    response_model=ImageChatResponse,
    response_model_exclude_none=True,
)
async def chat_image(
    request: Request,
    ctx: ContainerDep,
    image: UploadFile = File(..., description="image/png or image/jpeg"),
    query: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    user_id: str | None = Form(default=None),
) -> ImageChatResponse:
    """
    Upload an image alongside an optional question (multipart/form-data).

    The media pipeline validates, classifies, and routes the upload: clinical/
    general photos go to the multimodal LLM; lab/radiology/other reports are sent
    through document extraction (no visual interpretation). The result is folded
    into a normal pipeline turn — same session memory, retrieval, and answer
    path as `/chat`. Only metadata (type, caption, extracted facts, storage URI)
    is persisted; raw bytes never enter memory.
    """
    if not ctx.settings.MEDIA_UPLOAD_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image upload is disabled.",
        )

    request_id = _request_id(request)
    data = await image.read()

    try:
        media_result = await ctx.media_pipeline.process(
            data=data,
            mime_type=image.content_type,
            filename=image.filename,
            query=query,
        )
    except MediaValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    sid = session_id or uuid.uuid4().hex
    # Multipart image upload carries the legacy Form fields (no JSON envelope).
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=sid, user_id=user_id, envelope=None
    )
    result = await ctx.orchestrator.run(
        query=media_result.effective_query,
        identity=identity,
        media=media_result.attachment,
        location=getattr(req, "location", None),
    )
    return ImageChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        request_id=result.request_id,
        analysis=result.analysis if ctx.settings.EXPOSE_DIAGNOSTICS else None,
        timing_ms=result.timing_ms,
        routing=result.routing,
        followup_questions=result.followup_questions,
        show_doctor_summary=result.show_doctor_summary,
        media=_media_info(media_result.attachment.artifact),
    )


@router.post("/chat/soap", response_model=SoapNote)
async def chat_soap(req: SoapRequest, request: Request, ctx: ContainerDep) -> SoapNote:
    """
    Generate a fresh doctor-facing SOAP note for an existing session.

    Triggered on demand ("Show this to your doctor"). The note is regenerated
    every call from the latest full conversation context, so any new patient
    information is reflected. Strictly grounded in the conversation — missing
    clinical information is named in `unavailable`, never fabricated.
    """
    from datetime import datetime, timezone

    from app.services.memory.session import load_session
    from app.services.soap import generate_soap_async

    request_id = _request_id(request)
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=req.session_id,
        user_id=req.user_id, envelope=req.identity,
    )
    bundle = await load_session(
        ctx.session_manager, identity.session_id, user_id=identity.user_id
    )
    if bundle.session.total_messages == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No conversation found for this session_id.",
        )

    sections = await generate_soap_async(
        bundle.session, model=ctx.settings.ANSWER_MODEL
    )
    return SoapNote(
        **sections,
        session_id=identity.session_id,
        request_id=request_id,
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@router.post("/chat/blocks")
async def chat_blocks(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    NDJSON stream of typed UI blocks (`application/x-ndjson`).

    One JSON block object per line, emitted as generated. The client reads the
    body stream, splits on `\\n`, JSON.parses each complete line, and renders by
    `.type` as it arrives — no text parsing, no waiting for the full response.

    Block types: summary, key_points, bullet_list, follow_up_questions, warning,
    next_steps, condition_list. Each line is `{"type": ..., "data": {...}}`.
    """
    request_id = _request_id(request)
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=req.session_id,
        user_id=req.user_id, envelope=req.identity,
    )
    ndjson = _to_ndjson(
        ctx.orchestrator.stream_blocks(
            query=req.query, identity=identity, location=req.location
        )
    )
    return StreamingResponse(
        ndjson,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


@router.post("/chat/stream/blocks")
async def chat_stream_blocks(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    Server-Sent Events stream of typed UI blocks (`text/event-stream`).

    Same validated blocks as `/chat/blocks`, but SSE-framed for `EventSource`
    and proxy-friendly clients: each block is one `data: <json>\\n\\n` event,
    terminated by a final `data: [DONE]\\n\\n`. The JSON payload is identical
    to an NDJSON line: `{"type": ..., "data": {...}}`.

    Reuses the same `stream_blocks` pipeline as `/chat/blocks` — only the
    transport framing differs.
    """
    request_id = _request_id(request)
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=req.session_id,
        user_id=req.user_id, envelope=req.identity,
    )
    sse_blocks = _blocks_to_sse(
        ctx.orchestrator.stream_blocks(
            query=req.query, identity=identity, location=req.location
        )
    )
    return StreamingResponse(
        sse_blocks,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    Server-Sent Events stream.

    Each event is `data: <json>\\n\\n`. Event payload types:
        {"type":"meta","data":{...}}     pipeline metadata before tokens
        {"type":"chunk","data":"..."}    one token / piece of answer text
        {"type":"done","timing_ms":...}  final event; client should close
        {"type":"error","error":{...}}   terminal error
    """
    request_id = _request_id(request)
    identity = _resolve_identity(
        ctx=ctx, request_id=request_id, session_id=req.session_id,
        user_id=req.user_id, envelope=req.identity,
    )
    sse_stream = _to_sse(
        ctx.orchestrator.stream(
            query=req.query, identity=identity, location=req.location
        )
    )
    return StreamingResponse(
        sse_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


async def _to_sse(events: AsyncIterator[dict]) -> AsyncIterator[bytes]:
    """Encode dict events as SSE `data: <json>\\n\\n` lines."""
    async for ev in events:
        payload = json.dumps(ev, ensure_ascii=False, default=str)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def _to_ndjson(blocks: "AsyncIterator") -> AsyncIterator[bytes]:
    """Encode validated Blocks as NDJSON — one `{...}\\n` line per block."""
    from graphrag.validators.answer_validator import block_to_line

    async for block in blocks:
        yield block_to_line(block).encode("utf-8")


async def _blocks_to_sse(blocks: "AsyncIterator") -> AsyncIterator[bytes]:
    """Encode validated Blocks as SSE — `data: <json>\\n\\n` per block, then [DONE]."""
    async for block in blocks:
        payload = json.dumps(block.model_dump(mode="json"), ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _request_id(request: Request) -> str:
    """Read X-Request-ID from headers if the client sent one; mint otherwise."""
    rid = request.headers.get("x-request-id")
    if rid:
        return rid
    rid = uuid.uuid4().hex
    return rid
