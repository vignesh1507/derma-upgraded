"""
AsyncOrchestrator — async-native port of GraphRAGPipeline.run().

Stage-by-stage parity with the existing sync pipeline:

    -2  Session memory load (async — SessionManager directly)
    -1  Medical query analyzer (async — Gemini JSON mode)
     0  Routing decision (pure)
     1  Pinecone retrieval (sync client → asyncio.to_thread)
     2  Entity extraction (pure / CPU)
     3  Neo4j traversal (sync driver → asyncio.to_thread)
   3.5  Episodic memory context (async-native)
     4  Gemini answer (async non-streaming; streaming path in stream())
     5  Episodic ingest (async-native, fire-and-forget)
    5b  Session save (async)

The orchestrator never holds Pinecone/Neo4j connections itself — it borrows
them from AppContainer. State is request-scoped (request_id, session, ...).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from app.services.memory.session import (
    assemble_memory_payload,
    build_retrieval_query,
    load_session,
    save_after_turn,
)
from graphrag.query_understanding import (
    QueryType,
    RoutingMode,
    decide_routing,
    get_config,
    is_trivial_input,
)

if TYPE_CHECKING:
    from app.container import AppContainer
    from app.identity import IdentityContext
    from app.services.media.types import MediaAttachment
    from graphrag.schemas.blocks import Block

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    answer: str
    session_id: str
    request_id: str
    analysis: dict[str, Any] | None = None
    timing_ms: dict[str, int] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    followup_questions: list[str] = field(default_factory=list)
    show_doctor_summary: bool = False


class AsyncOrchestrator:
    def __init__(self, container: "AppContainer") -> None:
        self._c = container

    # ------------------------------------------------------------------
    # Environmental context (UV, moisture, air quality)
    # ------------------------------------------------------------------

    def _environmental_block(self, location: str | None) -> str:
        """
        Local conditions for `location`, or "" — never raises, never blocks.

        Returns "" whenever the feature is off, no location is known, the
        provider is unavailable, or the reading is not clinically noteworthy.
        """
        if not location:
            return ""
        provider = getattr(self._c, "environmental_provider", None)
        if provider is None:
            return ""
        try:
            ctx = provider.fetch(location)
        except Exception:                       # noqa: BLE001 - fail open
            logger.warning("Environmental lookup raised; continuing without it")
            return ""
        if ctx is None or not ctx.is_clinically_relevant:
            return ""
        return ctx.to_prompt_block()

    # ------------------------------------------------------------------
    # Public — non-streaming
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        query: str,
        identity: "IdentityContext",
        media: "MediaAttachment | None" = None,
        location: str | None = None,
    ) -> ChatResult:
        """
        Run one full pipeline turn.

        Identity is carried by ``identity`` (patient + session + request id); the
        raw id strings are extracted here at the boundary for the existing stores.

        ``media`` (optional) folds an uploaded image into the turn: its raw parts
        are sent to the answer LLM (photo route only), its extracted text is
        injected as answer context, and a metadata-only note is recorded in
        memory. When ``media`` is None this is the unchanged text-only flow.
        """
        session_id = identity.session_id
        user_id = identity.user_id
        request_id = identity.request_id

        timing: dict[str, int] = {}
        t0 = time.monotonic()

        # Stage -2: Session memory
        with _Stage("session_load", timing):
            bundle = await load_session(self._c.session_manager, session_id, user_id=user_id)

        session = bundle.session
        wm = bundle.working_memory
        memory_query_text = build_retrieval_query(query, wm)
        analyzer_input = memory_query_text if (wm.turn_count or wm.has_summary) else query

        # Stage -1: Gatekeeper analyzer
        trivial_skip = is_trivial_input(query) and wm.turn_count > 0
        if trivial_skip:
            analysis: dict[str, Any] = {}
        else:
            with _Stage("analyze", timing):
                analysis = await self._c.analyzer.aanalyze(analyzer_input)

        # Short-circuit: refuse / emergency_redirect
        final_action = (analysis or {}).get("final_action")
        if analysis and "error" not in analysis and final_action in {"refuse", "emergency_redirect", "mental_health_crisis"}:
            msg = _canned_message(final_action)
            await save_after_turn(
                self._c.session_manager,
                session=session,
                user_query=query,
                assistant_answer=msg,
                analysis=analysis,
                query_type="emergency" if final_action in {"emergency_redirect", "mental_health_crisis"} else "unknown",
                user_id=user_id,
            )
            timing["total"] = int((time.monotonic() - t0) * 1000)
            return ChatResult(
                answer=msg,
                session_id=session_id,
                request_id=request_id,
                analysis=analysis,
                timing_ms=timing,
                routing={"mode": "short_circuit", "intent": final_action},
            )

        followup_questions = self._extract_followups(analysis)

        # A concluded turn (gathering done / confident) trips the sticky
        # doctor-summary flag so /chat clients can offer the SOAP export.
        from graphrag.domain.messages import is_terminal_turn

        elapsed = session.total_messages
        if is_terminal_turn(turn_count=elapsed, analysis=analysis) or _should_consolidate(
            wm, analysis, self._c.settings, total_messages=elapsed
        ):
            session.doctor_summary_ready = True

        # Rewritten query (if analyzer suggested one)
        rewritten = (analysis or {}).get("rewritten_query")
        active_query = (
            rewritten.strip() if rewritten and rewritten.strip() and rewritten != query else query
        )

        # Stage 0: Routing
        routing_mode, query_type = decide_routing(
            analysis=analysis, wm=wm, raw_query=query
        )
        cfg = get_config(query_type)
        intent_str = (analysis or {}).get("intent") or "unknown"
        vector_top_k, reranker_top_k, graph_hops = _route_budget(routing_mode, cfg)

        # Stage 1: Pinecone (sync client → thread)
        retrieval_query_text = build_retrieval_query(active_query, wm)
        if vector_top_k > 0:
            with _Stage("vector_retrieve", timing):
                matches = await asyncio.to_thread(
                    self._c.vector_retriever.retrieve,
                    retrieval_query_text,
                    vector_top_k,
                    reranker_top_k,
                )
        else:
            matches = []

        # Stage 2: Entity extraction (pure)
        from graphrag.processors.entity_processor import EntityProcessor  # local: keep import light
        processor = EntityProcessor()
        vector_context_str, extracted_entities, _ = processor.process_matches(
            matches,
            priority_entity_types=cfg.priority_entity_types,
            boost_drug_pairs=cfg.boost_drug_pairs,
            query=retrieval_query_text,
        )

        # Stage 3: Neo4j (sync driver → thread)
        if self._c.settings.GRAPH_RETRIEVAL_ENABLED and graph_hops > 0 and extracted_entities:
            with _Stage("graph_retrieve", timing):
                graph_lines = await asyncio.to_thread(
                    self._c.kg_retriever.retrieve_relations,
                    extracted_entities,
                    graph_hops,
                    20,
                )
            graph_context_str = (
                "\n".join(f"- {g}" for g in graph_lines) if graph_lines else "No relevant relations found."
            )
        else:
            graph_context_str = ""

        # Stage 3.5: Episodic memory context (async-native)
        episodic_context_str = ""
        if user_id and self._c.episodic is not None:
            with _Stage("episodic_context", timing):
                episodic_context_str = await self._load_episodic_context(
                    user_id=user_id, query_text=retrieval_query_text
                )

        # Authoritative demographics (MongoDB) — loaded ONCE. The object drives
        # both the injected block and suppression of conflicting conversational
        # age/sex in the session-state block (Mongo is the source of truth).
        demo = await self._load_demographics(user_id)
        authoritative = _authoritative_demographic_fields(demo)

        # Stage 4: LLM answer
        memory_payload = assemble_memory_payload(
            wm=wm,
            user_query=query,
            query_type=intent_str,
            goal=cfg.goal,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
            authoritative_demographics=authoritative,
        )
        combined_memory = memory_payload.memory_context
        if episodic_context_str:
            combined_memory = episodic_context_str.strip() + "\n\n" + combined_memory
        environmental_block = self._environmental_block(location)
        followup_questions = self._drop_answered_followups(
            followup_questions, environmental_block
        )

        from app.services.demographics import render_demographic_block

        demographic_context = render_demographic_block(demo, analysis, query)

        with _Stage("llm", timing):
            answer = await self._answer_async(
                query=query,
                memory_context=combined_memory,
                conversation_history=memory_payload.conversation_context,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                query_type=intent_str,
                goal=cfg.goal,
                risk_level=str((analysis or {}).get("risk_level") or "none"),
                media_context=media.context_text if media else "",
                media=media.parts if media else None,
                demographic_context=demographic_context,
                environmental_context=environmental_block,
            )

        if followup_questions and answer:
            followup_block = (
                "\n\n---\n💬 **To help me give you a more precise answer next time, "
                "could you also share:**\n"
                + "\n".join(f"- {q}" for q in followup_questions)
            )
            answer = answer + followup_block

        # Memory records the upload as a metadata-only note (caption/findings/
        # type), NEVER the raw bytes — see app/services/media.
        stored_query = f"{query}\n\n{media.memory_note}" if media else query

        # Stage 5: Episodic ingest (fire-and-forget; never blocks response)
        if user_id and self._c.episodic is not None:
            asyncio.create_task(self._ingest_episodic_safe(identity=identity, utterance=stored_query))

        # Stage 5b: Session save
        with _Stage("session_save", timing):
            await save_after_turn(
                self._c.session_manager,
                session=session,
                user_query=stored_query,
                assistant_answer=answer or "",
                analysis=analysis or {},
                query_type=query_type.value,
                user_id=user_id,
            )

        timing["total"] = int((time.monotonic() - t0) * 1000)

        return ChatResult(
            answer=answer or "",
            session_id=session_id,
            request_id=request_id,
            analysis=analysis or None,
            timing_ms=timing,
            routing={
                "mode": routing_mode.value,
                "intent": intent_str,
                "query_type": query_type.value,
                "vector_top_k": vector_top_k,
                "graph_hops": graph_hops,
            },
            followup_questions=followup_questions,
            show_doctor_summary=session.doctor_summary_ready,
        )

    # ------------------------------------------------------------------
    # Public — streaming (filled in by phase 3)
    # ------------------------------------------------------------------

    async def stream(
        self,
        *,
        query: str,
        identity: "IdentityContext",
        location: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
        """
        Yield SSE-shaped events as the pipeline progresses.

        Event types:
            meta   — pipeline metadata (routing, timing of pre-LLM stages)
            chunk  — one piece of streamed model output
            done   — final event with total timing + assistant_answer
            error  — terminal error event (also ends the stream)

        The pre-LLM stages run exactly as in run(); only Stage 4 changes
        from a single await to an async iterator. After the stream ends we
        run session_save + fire-and-forget episodic ingest just like run().
        """
        from app.services.llm.streaming import stream_gemini_tokens
        from graphrag.config.settings import settings as cfg
        from graphrag.llm.gemini_client import DEFAULT_MODEL

        session_id = identity.session_id
        user_id = identity.user_id
        request_id = identity.request_id

        timing: dict[str, int] = {}
        t0 = time.monotonic()

        # ------------------------------------------------------------------
        # Pre-LLM stages (identical to run())
        # ------------------------------------------------------------------
        try:
            with _Stage("session_load", timing):
                bundle = await load_session(self._c.session_manager, session_id, user_id=user_id)
            session = bundle.session
            wm = bundle.working_memory
            memory_query_text = build_retrieval_query(query, wm)
            analyzer_input = memory_query_text if (wm.turn_count or wm.has_summary) else query

            trivial_skip = is_trivial_input(query) and wm.turn_count > 0
            if trivial_skip:
                analysis: dict[str, Any] = {}
            else:
                with _Stage("analyze", timing):
                    analysis = await self._c.analyzer.aanalyze(analyzer_input)

            final_action = (analysis or {}).get("final_action")
            if analysis and "error" not in analysis and final_action in {"refuse", "emergency_redirect", "mental_health_crisis"}:
                msg = _canned_message(final_action)
                await save_after_turn(
                    self._c.session_manager,
                    session=session,
                    user_query=query,
                    assistant_answer=msg,
                    analysis=analysis,
                    query_type="emergency" if final_action in {"emergency_redirect", "mental_health_crisis"} else "unknown",
                    user_id=user_id,
                )
                yield {"type": "chunk", "data": msg}
                timing["total"] = int((time.monotonic() - t0) * 1000)
                yield {"type": "done", "timing_ms": timing}
                return

            followup_questions = self._extract_followups(analysis)
            rewritten = (analysis or {}).get("rewritten_query")
            active_query = (
                rewritten.strip() if rewritten and rewritten.strip() and rewritten != query else query
            )

            routing_mode, query_type = decide_routing(
                analysis=analysis, wm=wm, raw_query=query
            )
            route_cfg = get_config(query_type)
            intent_str = (analysis or {}).get("intent") or "unknown"
            vector_top_k, reranker_top_k, graph_hops = _route_budget(routing_mode, route_cfg)

            retrieval_query_text = build_retrieval_query(active_query, wm)
            if vector_top_k > 0:
                with _Stage("vector_retrieve", timing):
                    matches = await asyncio.to_thread(
                        self._c.vector_retriever.retrieve,
                        retrieval_query_text,
                        vector_top_k,
                        reranker_top_k,
                    )
            else:
                matches = []

            from graphrag.processors.entity_processor import EntityProcessor
            processor = EntityProcessor()
            vector_context_str, extracted_entities, _ = processor.process_matches(
                matches,
                priority_entity_types=route_cfg.priority_entity_types,
                boost_drug_pairs=route_cfg.boost_drug_pairs,
                query=retrieval_query_text,
            )

            if self._c.settings.GRAPH_RETRIEVAL_ENABLED and graph_hops > 0 and extracted_entities:
                with _Stage("graph_retrieve", timing):
                    graph_lines = await asyncio.to_thread(
                        self._c.kg_retriever.retrieve_relations,
                        extracted_entities, graph_hops, 20,
                    )
                graph_context_str = (
                    "\n".join(f"- {g}" for g in graph_lines)
                    if graph_lines else "No relevant relations found."
                )
            else:
                graph_context_str = ""

            episodic_context_str = ""
            if user_id and self._c.episodic is not None:
                with _Stage("episodic_context", timing):
                    episodic_context_str = await self._load_episodic_context(
                        user_id=user_id, query_text=retrieval_query_text
                    )

            demo = await self._load_demographics(user_id)
            authoritative = _authoritative_demographic_fields(demo)
            memory_payload = assemble_memory_payload(
                wm=wm,
                user_query=query,
                query_type=intent_str,
                goal=route_cfg.goal,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                authoritative_demographics=authoritative,
            )
            combined_memory = memory_payload.memory_context
            if episodic_context_str:
                combined_memory = episodic_context_str.strip() + "\n\n" + combined_memory
            environmental_block = self._environmental_block(location)
            followup_questions = self._drop_answered_followups(
                followup_questions, environmental_block
            )

            # Tell the client what's about to happen so a UI can show status.
            yield {
                "type": "meta",
                "data": {
                    "routing": {
                        "mode": routing_mode.value,
                        "intent": intent_str,
                        "query_type": query_type.value,
                    },
                    "timing_ms": dict(timing),
                },
            }

            # ------------------------------------------------------------------
            # Stage 4: streaming LLM answer
            # ------------------------------------------------------------------
            from app.services.demographics import render_demographic_block

            demographic_context = render_demographic_block(demo, analysis, query)
            system_prompt, user_prompt = _compose_answer_prompts(
                query=query,
                memory_context=combined_memory,
                conversation_history=memory_payload.conversation_context,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                query_type=intent_str,
                risk_level=str((analysis or {}).get("risk_level") or "none"),
                demographic_context=demographic_context,
                environmental_context=environmental_block,
            )

            llm_t0 = time.monotonic()
            answer_chunks: list[str] = []
            async for piece in stream_gemini_tokens(
                model=cfg.ANSWER_MODEL or DEFAULT_MODEL,
                system_instruction=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            ):
                answer_chunks.append(piece)
                yield {"type": "chunk", "data": piece}
            timing["llm"] = int((time.monotonic() - llm_t0) * 1000)

            answer = "".join(answer_chunks)
            if followup_questions and answer:
                followup_block = (
                    "\n\n---\n💬 **To help me give you a more precise answer next time, "
                    "could you also share:**\n"
                    + "\n".join(f"- {q}" for q in followup_questions)
                )
                yield {"type": "chunk", "data": followup_block}
                answer = answer + followup_block

            # ------------------------------------------------------------------
            # Post-stream: ingest + session save (don't block the done event)
            # ------------------------------------------------------------------
            if user_id and self._c.episodic is not None:
                asyncio.create_task(
                    self._ingest_episodic_safe(identity=identity, utterance=query)
                )

            with _Stage("session_save", timing):
                await save_after_turn(
                    self._c.session_manager,
                    session=session,
                    user_query=query,
                    assistant_answer=answer,
                    analysis=analysis or {},
                    query_type=query_type.value,
                    user_id=user_id,
                )

            timing["total"] = int((time.monotonic() - t0) * 1000)
            yield {"type": "done", "timing_ms": timing}

        except Exception as exc:
            logger.exception("Streaming pipeline failed: %s", exc)
            yield {"type": "error", "error": {"code": "PIPELINE_ERROR", "message": str(exc)}}

    # ------------------------------------------------------------------
    # Public — streaming typed UI blocks (NDJSON transport)
    # ------------------------------------------------------------------

    async def stream_blocks(
        self,
        *,
        query: str,
        identity: "IdentityContext",
        location: str | None = None,
) -> "AsyncIterator[Block]":
        """
        STAGE-4 answer as a stream of validated UI blocks.

        Same pre-LLM stages as run()/stream(); STAGE 4 streams Gemini tokens
        through the per-line block validator (partial recovery) and yields
        Block objects as they validate — the first block reaches the client
        without buffering the whole answer. refuse / emergency short-circuits
        yield canned builder blocks instead of calling the LLM. The route
        encodes each Block as one NDJSON line.
        """
        from app.services.llm.streaming import stream_gemini_tokens
        from graphrag.config.settings import settings as cfg
        from graphrag.domain.messages import canned_blocks_for, is_terminal_turn
        from graphrag.llm.gemini_client import DEFAULT_MODEL
        from graphrag.processors.entity_processor import EntityProcessor
        from graphrag.validators.answer_validator import aiter_blocks, render_blocks_text

        session_id = identity.session_id
        user_id = identity.user_id
        request_id = identity.request_id

        emitted: list["Block"] = []
        try:
            bundle = await load_session(self._c.session_manager, session_id, user_id=user_id)
            session = bundle.session
            wm = bundle.working_memory
            memory_query_text = build_retrieval_query(query, wm)
            analyzer_input = memory_query_text if (wm.turn_count or wm.has_summary) else query

            trivial_skip = is_trivial_input(query) and wm.turn_count > 0
            if trivial_skip:
                analysis: dict[str, Any] = {}
            else:
                analysis = await self._c.analyzer.aanalyze(analyzer_input)

            # Canned short-circuit — refuse / emergency. NDJSON blocks, no LLM.
            final_action = (analysis or {}).get("final_action")
            if analysis and "error" not in analysis and final_action in {"refuse", "emergency_redirect", "mental_health_crisis"}:
                for block in canned_blocks_for(final_action):
                    emitted.append(block)
                    yield block
                await save_after_turn(
                    self._c.session_manager,
                    session=session,
                    user_query=query,
                    assistant_answer=render_blocks_text(emitted),
                    analysis=analysis,
                    query_type="emergency" if final_action in {"emergency_redirect", "mental_health_crisis"} else "unknown",
                    user_id=user_id,
                )
                return

            # Use the monotonic message counter (survives the window cap) so the
            # turn cap actually fires and the interview can terminate.
            elapsed = session.total_messages
            terminal = is_terminal_turn(turn_count=elapsed, analysis=analysis)
            # Consolidate (summarise instead of asking again) once enough facts
            # have accumulated, enough exchanges have passed, or we're confident.
            consolidate = _should_consolidate(wm, analysis, self._c.settings, total_messages=elapsed)
            needs_followup = bool((analysis or {}).get("needs_followup"))
            # On a consolidate/closing turn, deliver the assessment — don't tack
            # on another question.
            allow_followups = needs_followup and not terminal and not consolidate
            response_mode = str((analysis or {}).get("response_mode") or "generative_answer")

            rewritten = (analysis or {}).get("rewritten_query")
            active_query = (
                rewritten.strip() if rewritten and rewritten.strip() and rewritten != query else query
            )

            routing_mode, query_type = decide_routing(analysis=analysis, wm=wm, raw_query=query)
            route_cfg = get_config(query_type)
            intent_str = (analysis or {}).get("intent") or "unknown"
            vector_top_k, reranker_top_k, graph_hops = _route_budget(routing_mode, route_cfg)

            retrieval_query_text = build_retrieval_query(active_query, wm)
            if vector_top_k > 0:
                matches = await asyncio.to_thread(
                    self._c.vector_retriever.retrieve,
                    retrieval_query_text, vector_top_k, reranker_top_k,
                )
            else:
                matches = []

            processor = EntityProcessor()
            vector_context_str, extracted_entities, _ = processor.process_matches(
                matches,
                priority_entity_types=route_cfg.priority_entity_types,
                boost_drug_pairs=route_cfg.boost_drug_pairs,
                query=retrieval_query_text,
            )

            if self._c.settings.GRAPH_RETRIEVAL_ENABLED and graph_hops > 0 and extracted_entities:
                graph_lines = await asyncio.to_thread(
                    self._c.kg_retriever.retrieve_relations, extracted_entities, graph_hops, 20,
                )
                graph_context_str = (
                    "\n".join(f"- {g}" for g in graph_lines)
                    if graph_lines else "No relevant relations found."
                )
            else:
                graph_context_str = ""

            episodic_context_str = ""
            if user_id and self._c.episodic is not None:
                episodic_context_str = await self._load_episodic_context(
                    user_id=user_id, query_text=retrieval_query_text
                )

            demo = await self._load_demographics(user_id)
            authoritative = _authoritative_demographic_fields(demo)
            memory_payload = assemble_memory_payload(
                wm=wm,
                user_query=query,
                query_type=intent_str,
                goal=route_cfg.goal,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                authoritative_demographics=authoritative,
            )
            combined_memory = memory_payload.memory_context
            if episodic_context_str:
                combined_memory = episodic_context_str.strip() + "\n\n" + combined_memory
            environmental_block = self._environmental_block(location)

            from app.services.demographics import render_demographic_block

            demographic_context = render_demographic_block(demo, analysis, query)
            system_prompt, user_prompt = _compose_answer_prompts(
                query=query,
                memory_context=combined_memory,
                conversation_history=memory_payload.conversation_context,
                vector_context=vector_context_str,
                graph_context=graph_context_str,
                query_type=intent_str,
                risk_level=str((analysis or {}).get("risk_level") or "none"),
                # A consolidate turn closes the interview like terminal: the
                # validator drops any stray follow_up so the model must deliver
                # the summary/assessment instead of asking again.
                terminal=terminal or consolidate,
                allow_followups=allow_followups,
                consolidate=consolidate,
                response_mode=response_mode,
                demographic_context=demographic_context,
                environmental_context=environmental_block,
                output_format="blocks",
            )

            token_stream = stream_gemini_tokens(
                model=cfg.ANSWER_MODEL or DEFAULT_MODEL,
                system_instruction=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            )
            async for block in aiter_blocks(token_stream, terminal=terminal or consolidate):
                emitted.append(block)
                yield block

            # A concluded answer (gathering done + final assessment delivered)
            # flips the sticky doctor-summary flag. Emit a trailing control block
            # so the client can reveal the "Show this to your doctor" affordance.
            # It is NOT appended to `emitted` — control state never enters memory.
            if terminal or consolidate:
                session.doctor_summary_ready = True
            yield _answer_state_block(session.doctor_summary_ready)

            if user_id and self._c.episodic is not None:
                asyncio.create_task(self._ingest_episodic_safe(identity=identity, utterance=query))

            await save_after_turn(
                self._c.session_manager,
                session=session,
                user_query=query,
                assistant_answer=render_blocks_text(emitted),
                analysis=analysis or {},
                query_type=query_type.value,
                user_id=user_id,
            )

        except Exception as exc:
            logger.exception("Block stream pipeline failed: %s", exc)
            # Surface a minimal block so the client isn't left hanging mid-stream.
            from graphrag.schemas.blocks import SummaryBlock, SummaryData

            yield SummaryBlock(
                type="summary",
                data=SummaryData(
                    text="Sorry — something went wrong while generating the answer. "
                    "Please try again."
                ),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _answer_async(
        self,
        *,
        query: str,
        memory_context: str,
        conversation_history: str,
        vector_context: str,
        graph_context: str,
        query_type: str,
        goal: str,
        risk_level: str = "none",
        media_context: str = "",
        media: "list | None" = None,
        demographic_context: str = "",
        environmental_context: str = "",
    ) -> str:
        """
        Non-streaming Gemini answer. Reuses GeminiLLM's prompt assembly but
        bypasses the sync stdout-printing path. When ``media`` parts are given
        the answer call is multimodal; ``media_context`` adds extracted text.
        """
        from graphrag.llm.gemini_client import DEFAULT_MODEL, generate_text_async
        from graphrag.config.settings import settings as cfg

        system_prompt, user_prompt = _compose_answer_prompts(
            query=query,
            memory_context=memory_context,
            conversation_history=conversation_history,
            vector_context=vector_context,
            graph_context=graph_context,
            query_type=query_type,
            risk_level=risk_level,
            media_context=media_context,
            demographic_context=demographic_context,
            environmental_context=environmental_context,
        )
        # Vision-capable model when an image is attached; text model otherwise.
        model = (cfg.VISION_MODEL if media else cfg.ANSWER_MODEL) or DEFAULT_MODEL
        try:
            return await generate_text_async(
                user_prompt,
                model=model,
                system_instruction=system_prompt,
                temperature=0.2,
                media=media or None,
            )
        except Exception as exc:
            logger.exception("LLM answer failed: %s", exc)
            return ""

    async def _load_demographics(self, user_id: str | None):
        """
        Load the AI-safe DemographicContextV1 for this user (or None). Fail-open:
        any error/missing data returns None so a turn never breaks. This is the
        single Mongo read per turn — the same object drives both the injected
        demographics block and the authoritative-field suppression in session
        state (so the two demographic sources can't conflict).
        """
        svc = getattr(self._c, "demographics", None)
        if svc is None or not user_id:
            return None
        try:
            return await svc.load(user_id)
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("Demographics load failed: %s", exc)
            return None

    async def _load_episodic_context(self, *, user_id: str, query_text: str) -> str:
        """Best-effort episodic context block; empty string on any failure."""
        try:
            from episodic.schemas.retrieval import RetrievalRequest
            block = await self._c.episodic.context_pipeline.build(
                RetrievalRequest(user_id=user_id, query_text=query_text)
            )
            return block.rendered_prompt or ""
        except Exception as exc:
            logger.warning("Episodic context load failed: %s", exc)
            return ""

    async def _ingest_episodic_safe(
        self, *, identity: "IdentityContext", utterance: str
    ) -> None:
        """
        Ingest the turn into episodic memory (existing behaviour) AND, once the
        clinical extractor has produced an episode, hand that episode to the PMS
        producer. This is THE integration point after clinical extraction — the
        producer's wired client is NullPMSClient today, so nothing is sent yet.
        """
        try:
            result = await self._c.episodic.ingest_pipeline.run(
                user_id=identity.user_id, utterance=utterance
            )
        except Exception as exc:
            logger.warning("Episodic ingest failed: %s", exc)
            return

        # ── PMS producer: emit ClinicalMemoryEventV1 AFTER extraction ─────────
        # `result.stored` is the freshly-extracted, persisted Episode. Emitting
        # is fail-open and (today) a no-op sink, so the turn is never affected.
        episode = getattr(result, "stored", None)
        if episode is not None:
            from app.services.pms import ClinicalMemoryProducer

            await ClinicalMemoryProducer(self._c.pms).emit_from_episode(
                identity=identity, episode=episode
            )

    @staticmethod
    def _extract_followups(analysis: dict[str, Any] | None) -> list[str]:
        if not analysis or not analysis.get("needs_followup"):
            return []
        raw = analysis.get("followup_questions") or []
        # Hard cap: ≤1 question per turn (project contract).
        return [q for q in raw[:1] if q]

    @staticmethod
    def _drop_answered_followups(
        questions: list[str], environmental_block: str
    ) -> list[str]:
        """
        Drop follow-ups asking for AMBIENT conditions we have already measured.

        The analyser proposes follow-ups at stage -1, before retrieval and
        before the environmental lookup, so it cannot know that [LOCAL
        CONDITIONS] were attached. Left alone this produces a turn that states
        the UV index and the humidity in the answer and then asks the patient
        what the UV and humidity are.

        THE ASYMMETRY MATTERS MORE HERE THAN ANYWHERE ELSE IN THE PLATFORM.
        Personal sun behaviour — hours outdoors, sunscreen, covering — is the
        highest-value history a dermatology service takes, and a UV reading
        cannot answer any of it: UV 11 over the city says nothing about whether
        this patient was under it. Suppressing "do you use sunscreen?" would be
        a far worse defect than the redundancy being fixed. So the personal
        list wins every tie, and the ambient list stays deliberately narrow —
        only phrasings our own block genuinely answers.
        """
        if not environmental_block or not questions:
            return questions
        kept: list[str] = []
        for q in questions:
            personal = bool(_PERSONAL_RE.search(q))
            ambient = bool(_AMBIENT_RE.search(q))
            if ambient and not personal:
                logger.info(
                    "Dropping follow-up already answered by local conditions: %r", q
                )
                continue
            kept.append(q)
        return kept


# Ambient conditions the environmental provider already measures. Kept NARROW
# on purpose: a question only qualifies if our own block genuinely answers it.
# Bare "sun" is absent by design — "how much sun do you get?" is behaviour, not
# a reading.
_AMBIENT_TERMS: tuple[str, ...] = (
    "uv index", "uv level", "uv radiation",
    "air quality", "air pollution", "pollution", "aqi", "pm2.5", "pm10",
    "particulate", "smog", "haze",
    "humid", "dew point", "weather", "climate", "season",
    "environmental factor", "environmental condition", "surroundings",
)

# Personal exposures and behaviours we do NOT know and must keep asking about.
# Any question touching one of these survives even if it also names an ambient
# term — "on high-UV days do you wear sunscreen?" is a behaviour question.
_PERSONAL_EXPOSURE_TERMS: tuple[str, ...] = (
    # sun behaviour — the highest-value dermatology history there is
    "sun exposure", "in the sun", "sunlight", "sunscreen", "spf", "sunblock",
    "outdoors", "outside", "shade", "hat", "cover", "clothing",
    "how long are you", "time do you spend", "midday",
    # topicals, irritants and contactants
    "product", "cosmetic", "cream", "moistur", "soap", "detergent", "shampoo",
    "fragrance", "perfume", "dye", "nickel", "jewellery", "jewelry", "glove",
    "chemical", "chlorine", "swim", "shower", "bath", "hot water",
    # occupational and domestic
    "work", "job", "occupation", "factory", "field",
    "home", "house", "indoor", "heater", "air conditioning",
    # systemic and other
    "medication", "drug", "tablet", "travel", "contact", "family", "history",
    "allerg",
)


def _term_matcher(terms: tuple[str, ...]) -> "re.Pattern[str]":
    """
    Word-START anchored matcher.

    Substring matching is wrong here and was caught in test: "hat" matches
    inside "W-hat", so every question opening "What is the..." looked like a
    question about sun-protective clothing. Anchoring the LEFT edge only keeps
    stems working — "humid" still matches "humidity", "allerg" matches
    "allergies" — while "\bhat" no longer fires inside "what".
    """
    return re.compile("|".join(r"\b" + re.escape(t) for t in terms), re.I)


_AMBIENT_RE = _term_matcher(_AMBIENT_TERMS)
_PERSONAL_RE = _term_matcher(_PERSONAL_EXPOSURE_TERMS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Stage:
    """Record `stage`: duration_ms in the timing dict, exception-safe."""

    def __init__(self, name: str, sink: dict[str, int]) -> None:
        self._name = name
        self._sink = sink
        self._t0 = 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._sink[self._name] = int((time.monotonic() - self._t0) * 1000)


def _route_budget(mode: RoutingMode, cfg) -> tuple[int, int, int]:
    if mode == RoutingMode.NO_RETRIEVAL:
        return (0, 0, 0)
    if mode == RoutingMode.MEMORY_FIRST:
        return (3, 3, 0)
    return (cfg.vector_top_k, cfg.reranker_top_k, cfg.graph_hops)


def _authoritative_demographic_fields(demo) -> frozenset[str]:
    """
    Which demographic fields the canonical MongoDB profile authoritatively owns
    this turn, among the ones the conversational session state can also carry
    (age, sex). Used to suppress conflicting conversational values from the
    session-state block. Height/weight/BMI/location are Mongo-only (the session
    state never extracts them), so they need no suppression.
    """
    if demo is None:
        return frozenset()
    fields: set[str] = set()
    if getattr(demo, "age", None) is not None:
        fields.add("age")
    if getattr(demo, "sex", None):
        fields.add("sex")
    return frozenset(fields)


def _answer_state_block(show_doctor_summary: bool) -> "Block":
    """Build the trailing control block carrying per-turn answer state."""
    from graphrag.schemas.blocks import AnswerStateBlock, AnswerStateData

    return AnswerStateBlock(
        type="answer_state",
        data=AnswerStateData(show_doctor_summary=show_doctor_summary),
    )


def _canned_message(final_action: str) -> str:
    if final_action == "refuse":
        return (
            "I’m designed to assist only with healthcare-related questions. "
            "Please ask a medical or health-related question so I can help."
        )
    if final_action == "mental_health_crisis":
        return (
            "I’m really glad you told me this, and I’m sorry you’re carrying so "
            "much right now. What you’re feeling is real and you don’t have to "
            "face it alone. If you might act on these thoughts or feel unsafe, "
            "please call 112 or go to the nearest emergency room now. To talk to "
            "someone right away, India’s free 24/7 Tele-MANAS line is 14416 "
            "(or 1-800-891-4416), and KIRAN is 1800-599-0019; outside India, "
            "contact your local crisis line. If you can, reach out to someone "
            "you trust and stay with them. Asking for help is a strong first step."
        )
    return (
        "🚨 Medical Emergency: Your symptoms may indicate a serious or "
        "life-threatening condition. Please call 112 immediately or go to the "
        "nearest emergency room or hospital as soon as possible."
    )


def _should_consolidate(
    wm, analysis: dict[str, Any] | None, settings, total_messages: int = 0
) -> bool:
    """
    Whether this triage turn should stop gathering and consolidate a summary.

    True once enough distinct clinical facts have accumulated
    (``CONSOLIDATE_MIN_FACTS``), the consultation has run enough exchanges
    (``CONSOLIDATE_AFTER_TURNS`` — a backstop for cases the weak extractor
    under-counts), or the gatekeeper is already confident. Keeps summaries as
    checkpoints, and guarantees the interview can't collect info forever.
    """
    from Memory_Layer.session_memory import count_clinical_facts
    from graphrag.domain.messages import parse_diagnostic_confidence

    if count_clinical_facts(wm.state) >= settings.CONSOLIDATE_MIN_FACTS:
        return True
    if total_messages // 2 >= settings.CONSOLIDATE_AFTER_TURNS:
        return True
    confidence = parse_diagnostic_confidence((analysis or {}).get("diagnostic_confidence"))
    return confidence is not None and confidence >= settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD


def _compose_answer_prompts(
    *,
    query: str,
    memory_context: str,
    conversation_history: str,
    vector_context: str,
    graph_context: str,
    query_type: str,
    risk_level: str = "none",
    terminal: bool = False,
    allow_followups: bool = True,
    output_format: str = "prose",
    consolidate: bool = False,
    response_mode: str = "generative_answer",
    media_context: str = "",
    demographic_context: str = "",
    environmental_context: str = "",
) -> tuple[str, str]:
    """
    Compose the (system, user) prompt pair for the answer LLM.

    System prompt is built via the layered composer in
    [app.services.orchestration.prompt_layers]; CLI and FastAPI now share
    this single source of truth. The `has_name` flag is inferred from the
    rendered memory block — `_state_lines` writes a `Patient name:` line
    when `state.demographics["name"]` is populated. `output_format` selects
    prose (default — /chat + /chat/stream) vs NDJSON blocks (/chat/blocks).
    ``media_context`` (optional) carries a caption / extracted document text for
    an uploaded image and is injected as its own block when present.
    """
    from app.services.orchestration.prompt_layers import compose_system_prompt

    has_name = "Patient name:" in memory_context
    system_prompt = compose_system_prompt(
        query_type=query_type,
        risk_level=risk_level,
        has_name=has_name,
        terminal=terminal,
        allow_followups=allow_followups,
        consolidate=consolidate,
        response_mode=response_mode,
        output_format=output_format,
    )

    media_block = f"\n=== UPLOADED FILE ===\n{media_context}\n" if media_context else ""
    # Authoritative current patient facts (from MongoDB) — kept SEPARATE from
    # session/episodic memory and only present when relevant to this query.
    demo_block = f"\n{demographic_context}\n" if demographic_context else ""
    # Measured local conditions get their OWN section rather than being folded
    # into clinical memory. Buried in memory the model treats them as "things
    # the patient told me" and generalises instead of citing the readings.
    env_block = (
        f"\n=== LOCAL ENVIRONMENT (measured, not patient-reported) ===\n"
        f"{environmental_context}\n"
        if environmental_context else ""
    )

    user_prompt = f"""
USER QUESTION: {query}
{media_block}{demo_block}{env_block}
=== STRUCTURED CLINICAL MEMORY ===
{memory_context}

=== RECENT CONVERSATION ===
{conversation_history}

=== RETRIEVED MEDICAL CONTEXT ===
{vector_context}

=== GRAPH RELATIONS ===
{graph_context}
"""
    return system_prompt, user_prompt
