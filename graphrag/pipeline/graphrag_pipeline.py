import asyncio
import json

from graphrag.config.settings import settings
from graphrag.retrievers.pinecone_retriever import PineconeRetriever
from graphrag.retrievers.neo4j_retriever import Neo4jRetriever
from graphrag.processors.entity_processor import EntityProcessor
from graphrag.llm.gemini_llm import GeminiLLM
from graphrag.memory import SessionMemoryAdapter
from graphrag.query_understanding import (
    QueryType,
    RoutingMode,
    decide_routing,
    get_config,
    is_trivial_input,
)
from graphrag.query_understanding.analyzer import MedicalQueryAnalyzer
from graphrag.utils.logger import get_logger

logger = get_logger(__name__)

_SEPARATOR = "─" * 72


class GraphRAGPipeline:
    def __init__(self, redis_url: str | None = None):
        try:
            self.pinecone_retriever = PineconeRetriever()
            self.neo4j_retriever    = Neo4jRetriever()
            self.llm                = GeminiLLM()
            self.entity_processor   = EntityProcessor()
            self.query_analyzer     = MedicalQueryAnalyzer()
            self.memory_adapter     = SessionMemoryAdapter(redis_url=redis_url)

            self._episodic = None
            # Persistent event loop for the episodic async calls. Created lazily
            # on first use so we don't pay startup cost when --user-id is unused.
            # Using one loop for the whole pipeline lifetime (vs asyncio.run per
            # call) prevents the 'Event loop is closed' error from the genai
            # SDK's cached AsyncClient binding to a now-dead loop.
            self._loop = None
            if settings.EPISODIC_MEMORY_ENABLED:
                try:
                    from episodic.api.dependencies import build_container
                    self._episodic = build_container()
                    logger.info("📚 Episodic memory layer ACTIVE (pass --user-id to use it)")
                except Exception as exc:
                    logger.warning(
                        "Episodic memory layer disabled — failed to initialize: %s", exc
                    )

            logger.info("\n" + "★" * 80)
            logger.info("★  GRAPH-RAG ENGINE  ·  QUERY UNDERSTANDING LAYER ACTIVE")
            logger.info("★  Stack: Classify → Vector → Rerank → Graph → LLM")
            logger.info("★" * 80 + "\n")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    # ------------------------------------------------------------------

    def run(self, query_text: str, session_id: str = "default", user_id: str | None = None):
        original_query_text = query_text
        logger.info(f"\n{'═' * 72}")
        logger.info(f"📝 Original Query: {query_text}")
        logger.info(f"{'═' * 72}")

        logger.info(f"\n{_SEPARATOR}")
        logger.info("STAGE -2 -> Session Memory Load")
        logger.info(_SEPARATOR)

        memory_bundle = self.memory_adapter.load(session_id)
        session = memory_bundle.session
        working_memory = memory_bundle.working_memory
        memory_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text,
            wm=working_memory,
        )
        analyzer_query_text = (
            memory_query_text
            if working_memory.turn_count or working_memory.has_summary
            else query_text
        )

        # ── Stage -1: Medical Gatekeeper / Query Analyzer ─────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("🛡️   STAGE -1 → Medical Gatekeeper & Analyzer")
        logger.info(_SEPARATOR)

        trivial_skip = is_trivial_input(original_query_text) and working_memory.turn_count > 0
        if trivial_skip:
            logger.info("⏭️  Trivial acknowledgment in established session — skipping gatekeeper LLM.")
            analysis = {}
        else:
            analysis = self.query_analyzer.analyze(analyzer_query_text)

        followup_questions = []
        if analysis and "error" not in analysis and analysis.get("final_action"):
            logger.info(f"🧠 Analysis Results:\n{json.dumps(analysis, indent=2)}")

            final_action = analysis.get("final_action")
            if final_action == "refuse":
                msg = "❌ I can only answer healthcare-related questions. Please ask a medical question."
                print(f"\n{msg}\n")
                self.memory_adapter.update_after_interaction(
                    session=session,
                    user_query=original_query_text,
                    assistant_answer=msg,
                    analysis=analysis,
                    query_type="unknown",
                )
                return msg
            elif final_action == "emergency_redirect":
                msg = "🚨 EMERGENCY: Your symptoms sound like a serious emergency. Please call emergency services immediately — 112, or 108 / 102 for an ambulance — or go to the nearest hospital."
                print(f"\n{msg}\n")
                self.memory_adapter.update_after_interaction(
                    session=session,
                    user_query=original_query_text,
                    assistant_answer=msg,
                    analysis=analysis,
                    query_type="emergency",
                )
                return msg

            if analysis.get("needs_followup"):
                # Hard cap: at most ONE follow-up question. Multiple questions in
                # one turn create friction and the answer LLM already ends with
                # its own at-most-one clarifier.
                raw_followups = analysis.get("followup_questions") or []
                followup_questions = [q for q in raw_followups[:1] if q]
                if followup_questions:
                    logger.info("💬 One follow-up question will be appended to answer.")

            rewritten = analysis.get("rewritten_query")
            if rewritten and rewritten.strip() and rewritten != query_text:
                logger.info(f"🔄 Query optimized: '{rewritten}'")
                query_text = rewritten
        elif not trivial_skip:
            logger.warning("⚠️ Query Analyzer returned no valid result. Proceeding with original query.")
            analysis = {}

        # ── Stage 0: Query Understanding & Routing ────────────────────────
        routing_mode, query_type = decide_routing(
            analysis=analysis,
            wm=working_memory,
            raw_query=original_query_text,
        )
        config = get_config(query_type)
        intent_str = (analysis or {}).get("intent") or "unknown"

        if routing_mode == RoutingMode.NO_RETRIEVAL:
            logger.info("⏭️  ROUTING: NO_RETRIEVAL (memory-only response)")
            vector_top_k = 0
            reranker_top_k = 0
            graph_hops = 0
        elif routing_mode == RoutingMode.MEMORY_FIRST:
            logger.info("🧠 ROUTING: MEMORY_FIRST (small clinical backfill, no graph)")
            vector_top_k = 3
            reranker_top_k = 3
            graph_hops = 0
        else:  # HYBRID_RAG
            logger.info("🔍 ROUTING: HYBRID_RAG (full retrieval active)")
            vector_top_k = config.vector_top_k
            reranker_top_k = config.reranker_top_k
            graph_hops = config.graph_hops

        retrieval_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text,
            wm=working_memory,
        )

        logger.info(f"   Intent  : {intent_str.upper()}")
        logger.info(f"   Mode    : {routing_mode.value.upper()}")
        logger.info(f"   top_k   : {vector_top_k}")
        logger.info(f"   Graph   : {'enabled' if graph_hops > 0 else 'GATED/DISABLED'}")

        # ── Stage 1: Vector Retrieval + Reranking ─────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 1 → Vector Retrieval + Reranking")
        logger.info(_SEPARATOR)

        if vector_top_k > 0:
            matches = self.pinecone_retriever.retrieve(
                retrieval_query_text,
                vector_top_k=vector_top_k,
                reranker_top_k=reranker_top_k,
            )
        else:
            logger.info("❌ Vector retrieval skipped.")
            matches = []

        # ── Stage 2: Entity Extraction ────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 2 → Entity Extraction")
        logger.info(_SEPARATOR)

        vector_context_str, extracted_entities, _ = self.entity_processor.process_matches(
            matches,
            priority_entity_types = config.priority_entity_types,
            boost_drug_pairs      = config.boost_drug_pairs,
            query                 = retrieval_query_text,
        )

        # ── Stage 3: Graph Retrieval ──────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 3 → Knowledge Graph Traversal")
        logger.info(_SEPARATOR)

        if settings.GRAPH_RETRIEVAL_ENABLED and graph_hops > 0 and extracted_entities:
            graph_context_list = self.neo4j_retriever.retrieve_relations(
                extracted_entities,
                hops  = graph_hops,
                limit = 20,
            )
            graph_context_str = "\n".join([f"- {g}" for g in graph_context_list]) if graph_context_list else "No relevant relations found."
        else:
            logger.info(f"⏭️   Graph skipped (Gated or no entities).")
            graph_context_str = ""

        # ── Stage 3.5: Episodic Memory Retrieval ─────────────────────────
        # Only when a --user-id is supplied AND the episodic container
        # initialized cleanly. Best-effort: a failure here degrades to no
        # episodic context, never breaks the pipeline.
        episodic_context_str = ""
        if user_id and self._episodic is not None:
            logger.info(f"\n{_SEPARATOR}")
            logger.info("🧠  STAGE 3.5 → Episodic Memory Retrieval")
            logger.info(_SEPARATOR)
            episodic_context_str = self._load_episodic_context(
                user_id=user_id, query_text=retrieval_query_text
            )
            if episodic_context_str:
                logger.info(f"   Episodic context: {len(episodic_context_str)} chars")
            else:
                logger.info("   Episodic context: empty")

        # ── Stage 4: LLM ─────────────────────────────────────────────────
        logger.info(f"\n{_SEPARATOR}")
        logger.info("⚙️   STAGE 4 → LLM Response Generation")
        logger.info(_SEPARATOR)

        # Assemble the full conversational context
        memory_payload = self.memory_adapter.assemble_payload(
            wm=working_memory,
            user_query=original_query_text,
            query_type=intent_str,
            goal=config.goal,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
        )

        # Concatenate episodic memory in front of the Redis session block so
        # the answer LLM sees long-term patient history before short-term turns.
        combined_memory_context = memory_payload.memory_context
        if episodic_context_str:
            combined_memory_context = (
                episodic_context_str.strip() + "\n\n" + combined_memory_context
            )

        # Pass the rich memory context and history to the LLM
        answer = self.llm.generate_response(
            query_text      = original_query_text,
            vector_context  = vector_context_str,
            graph_context   = graph_context_str,
            memory_context  = combined_memory_context,
            conversation_history = memory_payload.conversation_context,
            query_type      = intent_str,
            goal            = config.goal
        )

        # Append follow-up questions at the end if any
        if followup_questions and answer:
            followup_block = "\n\n---\n💬 **To help me give you a more precise answer next time, could you also share:**\n" + "\n".join([f"- {q}" for q in followup_questions])
            print(followup_block)
            answer += followup_block

        self.memory_adapter.update_after_interaction(
            session=session,
            user_query=original_query_text,
            assistant_answer=answer or "",
            analysis=analysis,
            query_type=query_type.value,
        )

        # ── Stage 5: Episodic Memory Ingest (post-answer, best-effort) ───
        if user_id and self._episodic is not None:
            self._ingest_episodic_turn(user_id=user_id, utterance=original_query_text)

        return answer

    # ------------------------------------------------------------------

    def run_stream(self, query_text: str, session_id: str = "default", user_id: str | None = None):
        """
        STAGE-4 answer as a stream of validated UI blocks (NDJSON path).

        Mirrors ``run()`` stage-for-stage, but STAGE 4 streams Gemini tokens
        through the per-line block validator and yields ``Block`` objects as
        they validate. refuse / emergency short-circuits yield canned builder
        blocks instead of calling the LLM. The CLI prints each block as one
        NDJSON line; transport stays uniform.
        """
        from graphrag.domain.messages import canned_blocks_for, is_terminal_turn
        from graphrag.validators.answer_validator import render_blocks_text

        original_query_text = query_text
        emitted = []

        memory_bundle = self.memory_adapter.load(session_id)
        session = memory_bundle.session
        working_memory = memory_bundle.working_memory
        memory_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text, wm=working_memory,
        )
        analyzer_query_text = (
            memory_query_text
            if working_memory.turn_count or working_memory.has_summary
            else query_text
        )

        trivial_skip = is_trivial_input(original_query_text) and working_memory.turn_count > 0
        analysis = {} if trivial_skip else self.query_analyzer.analyze(analyzer_query_text)

        # Canned short-circuit — refuse / emergency / mental-health crisis. No LLM.
        final_action = (analysis or {}).get("final_action")
        _crisis = {"emergency_redirect", "mental_health_crisis"}
        if analysis and "error" not in analysis and final_action in {"refuse", *_crisis}:
            for block in canned_blocks_for(final_action):
                emitted.append(block)
                yield block
            self.memory_adapter.update_after_interaction(
                session=session,
                user_query=original_query_text,
                assistant_answer=render_blocks_text(emitted),
                analysis=analysis,
                query_type="emergency" if final_action in _crisis else "unknown",
            )
            return

        # Monotonic message counter (survives the recent-turns window cap) so the
        # turn cap actually fires and consolidation can't loop forever.
        elapsed = session.total_messages
        terminal = is_terminal_turn(turn_count=elapsed, analysis=analysis)
        # Consolidate once enough facts are gathered, enough exchanges have
        # passed, or the gatekeeper is confident.
        from Memory_Layer.session_memory import count_clinical_facts
        from graphrag.domain.messages import parse_diagnostic_confidence

        _facts = count_clinical_facts(working_memory.state)
        _conf = parse_diagnostic_confidence((analysis or {}).get("diagnostic_confidence"))
        consolidate = (
            _facts >= settings.CONSOLIDATE_MIN_FACTS
            or elapsed // 2 >= settings.CONSOLIDATE_AFTER_TURNS
            or (_conf is not None and _conf >= settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD)
        )
        needs_followup = bool((analysis or {}).get("needs_followup"))
        # On a consolidate/closing turn, deliver the assessment — don't ask again.
        allow_followups = needs_followup and not terminal and not consolidate
        response_mode = str((analysis or {}).get("response_mode") or "generative_answer")

        rewritten = (analysis or {}).get("rewritten_query")
        if rewritten and rewritten.strip() and rewritten != query_text:
            query_text = rewritten

        routing_mode, query_type = decide_routing(
            analysis=analysis, wm=working_memory, raw_query=original_query_text,
        )
        config = get_config(query_type)
        intent_str = (analysis or {}).get("intent") or "unknown"

        if routing_mode == RoutingMode.NO_RETRIEVAL:
            vector_top_k = reranker_top_k = graph_hops = 0
        elif routing_mode == RoutingMode.MEMORY_FIRST:
            vector_top_k = reranker_top_k = 3
            graph_hops = 0
        else:
            vector_top_k = config.vector_top_k
            reranker_top_k = config.reranker_top_k
            graph_hops = config.graph_hops

        retrieval_query_text = self.memory_adapter.build_retrieval_query(
            query_text=query_text, wm=working_memory,
        )

        if vector_top_k > 0:
            matches = self.pinecone_retriever.retrieve(
                retrieval_query_text, vector_top_k=vector_top_k, reranker_top_k=reranker_top_k,
            )
        else:
            matches = []

        vector_context_str, extracted_entities, _ = self.entity_processor.process_matches(
            matches,
            priority_entity_types=config.priority_entity_types,
            boost_drug_pairs=config.boost_drug_pairs,
            query=retrieval_query_text,
        )

        if settings.GRAPH_RETRIEVAL_ENABLED and graph_hops > 0 and extracted_entities:
            graph_context_list = self.neo4j_retriever.retrieve_relations(
                extracted_entities, hops=graph_hops, limit=20,
            )
            graph_context_str = (
                "\n".join([f"- {g}" for g in graph_context_list])
                if graph_context_list else "No relevant relations found."
            )
        else:
            graph_context_str = ""

        episodic_context_str = ""
        if user_id and self._episodic is not None:
            episodic_context_str = self._load_episodic_context(
                user_id=user_id, query_text=retrieval_query_text
            )

        memory_payload = self.memory_adapter.assemble_payload(
            wm=working_memory,
            user_query=original_query_text,
            query_type=intent_str,
            goal=config.goal,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
        )
        combined_memory_context = memory_payload.memory_context
        if episodic_context_str:
            combined_memory_context = episodic_context_str.strip() + "\n\n" + combined_memory_context

        block_stream = self.llm.generate_blocks(
            query_text=original_query_text,
            vector_context=vector_context_str,
            graph_context=graph_context_str,
            memory_context=combined_memory_context,
            conversation_history=memory_payload.conversation_context,
            query_type=intent_str,
            goal=config.goal,
            risk_level=str((analysis or {}).get("risk_level") or "none"),
            # A consolidate turn CLOSES the interview like a terminal turn: the
            # validator drops any stray follow_up the model volunteers, forcing
            # it to deliver the summary/assessment instead of asking again.
            terminal=terminal or consolidate,
            allow_followups=allow_followups,
            consolidate=consolidate,
            response_mode=response_mode,
        )
        for block in block_stream:
            emitted.append(block)
            yield block

        # A concluded answer flips the sticky doctor-summary flag; emit a trailing
        # control block so the client can offer "Show this to your doctor". Not
        # appended to `emitted` — control state never enters memory.
        if terminal or consolidate:
            session.doctor_summary_ready = True
        from graphrag.schemas.blocks import AnswerStateBlock, AnswerStateData
        yield AnswerStateBlock(
            type="answer_state",
            data=AnswerStateData(show_doctor_summary=session.doctor_summary_ready),
        )

        self.memory_adapter.update_after_interaction(
            session=session,
            user_query=original_query_text,
            assistant_answer=render_blocks_text(emitted),
            analysis=analysis,
            query_type=query_type.value,
        )

        if user_id and self._episodic is not None:
            self._ingest_episodic_turn(user_id=user_id, utterance=original_query_text)

    # ------------------------------------------------------------------
    # Episodic memory helpers (sync wrappers around the async services)
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """
        Run a coroutine on the pipeline's persistent event loop.

        Pinned to one loop for the pipeline's lifetime so the genai SDK's
        cached AsyncClient (and any other async http session it holds onto)
        stays bound to a live loop across multiple turns.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _load_episodic_context(self, *, user_id: str, query_text: str) -> str:
        """Return a prompt-ready episodic context block. Returns '' on any failure."""
        try:
            from episodic.schemas.retrieval import RetrievalRequest
            req = RetrievalRequest(user_id=user_id, query_text=query_text)
            block = self._run_async(self._episodic.context_pipeline.build(req))
            return block.rendered_prompt or ""
        except Exception as exc:
            logger.warning("Episodic context load failed: %s", exc)
            return ""

    def _ingest_episodic_turn(self, *, user_id: str, utterance: str) -> None:
        """
        Best-effort ingest of the user's turn into episodic memory.

        Runs extract → contradiction check → clarification triage → store. If
        any LLM call hits a rate limit or the network is flaky, we log and
        move on — the user's main answer has already streamed.
        """
        try:
            result = self._run_async(
                self._episodic.ingest_pipeline.run(
                    user_id=user_id, utterance=utterance
                )
            )
        except Exception as exc:
            logger.warning("Episodic ingest failed: %s", exc)
            return

        if result.stored is not None:
            logger.info(
                "📥 Episodic memory ingested: episode_id=%s category=%s priority=%s",
                result.stored.episode_id,
                result.stored.category.value,
                result.stored.clinical_priority.value,
            )
        elif result.clarification.needs_clarification:
            qs = "; ".join(q.question for q in result.clarification.questions)
            logger.info("📝 Episodic ingest deferred — clarification needed: %s", qs)
        else:
            logger.info("📭 Episodic ingest skipped (no clinical content extracted).")

        if result.contradictions.has_contradictions:
            logger.info(
                "⚠️  Episodic contradiction signal: %d item(s), penalty=%.2f, triggers_clarification=%s",
                len(result.contradictions.contradictions),
                result.contradictions.confidence_penalty,
                result.contradictions.triggers_clarification,
            )

    # ------------------------------------------------------------------

    def close(self):
        if hasattr(self, "neo4j_retriever"):
            self.neo4j_retriever.close()
        loop = getattr(self, "_loop", None)
        if loop is not None and not loop.is_closed():
            try:
                loop.close()
            except Exception as exc:
                logger.debug("Pipeline loop close raised: %s", exc)
