"""
Typed configuration for the Enervera Medical GraphRAG system.

All environment variables are declared in `Settings` (a `pydantic-settings`
model). Required vars default to `None` so importing this module never fails;
callers must invoke `settings.validate_required(mode)` at startup to fail fast
with a clear error listing every missing variable.

Backward compatibility:
    The legacy `Config` class is preserved as a thin facade so existing code
    paths (`from graphrag.config.settings import Config`) keep working. New
    code should `from graphrag.config.settings import settings`.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load `.env` once at import so legacy modules that read `os.getenv` directly
# (e.g. clean_chunks.py, check_api.py) still see the values. pydantic-settings
# also reads the file; the double load is a no-op for existing env vars.
load_dotenv()


Mode = Literal["api", "cli", "ingest"]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing for the requested mode."""


class Settings(BaseSettings):
    """Typed environment configuration loaded from `.env` or the process env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ----- HTTP service (FastAPI on Render) -----
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # Optional API key for the /chat and /episodic routes. If unset, those
    # routes are open. If set, every request must send X-API-Key matching.
    API_KEY: Optional[str] = None
    # Comma-separated list of allowed CORS origins for the frontend. Use "*"
    # to allow any origin (fine while the service is gated by API_KEY, but
    # tighten this for production once the real frontend URL is known).
    # Example: "https://app.enervera.com,https://staging.enervera.com"
    CORS_ORIGINS: str = "*"
    # When true, /chat includes the internal gatekeeper `analysis` block
    # (intent, risk_level, medical_entities, rewritten_query, final_action) in
    # the JSON response. Default false so production never exposes the query
    # analyser's internals; enable only for local debugging.
    EXPOSE_DIAGNOSTICS: bool = False

    # ----- Environmental context (UV, moisture, air quality) -----
    # Supplementary local conditions injected alongside memory. Off by default:
    # it adds an outbound call. Fails open in every error path.
    ENVIRONMENTAL_CONTEXT_ENABLED: bool = False
    # WeatherAPI.com key. Env var is WEATHER_API (matches the deploy secret).
    WEATHER_API: Optional[str] = None
    # Bare city names resolve against a global gazetteer, so "Delhi" alone
    # returns Delhi, Ontario. Unqualified locations get this country appended.
    ENVIRONMENTAL_DEFAULT_COUNTRY: str = "India"
    ENVIRONMENTAL_TIMEOUT_S: float = 2.5
    ENVIRONMENTAL_CACHE_TTL_S: float = 3600.0

    # ----- Pinecone (vector index) -----
    PINECONE_API_KEY: Optional[str] = None
    # Dermatology knowledge corpus. The persona alone does not make answers
    # specialty-correct — this index is what grounds them.
    #
    # Contract any replacement must satisfy: embedding model llama-text-embed-v2,
    # dimension 1024, metric cosine, chunk metadata carrying at least
    # `summary` and `entities` (the entity processor reads both).
    PINECONE_INDEX_NAME: str = "enervara-specialists"

    # Namespace within PINECONE_INDEX_NAME to retrieve from.
    #
    # `enervara-specialists` is one index partitioned per specialty
    # (dermatology, cardiology, ent, ophthalmology, orthopaedics,
    # pulmonology_v1) and holds NOTHING in the default namespace. A query
    # without a namespace therefore matches zero vectors and the service
    # answers ungrounded with no error — so this must stay set for any
    # deployment pointed at that index.
    #
    # Empty string preserves the previous behaviour (query the default
    # namespace), which is what single-namespace indexes like `enervera` need.
    PINECONE_NAMESPACE: str = "dermatology"

    # ----- Neo4j (knowledge graph) -----
    NEO4J_URI: str = "bolt://127.0.0.1:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None
    # Master switch for the Stage-3 graph-traversal step. Currently OFF: the
    # Neo4j Aura instance is unreachable and graph relations add latency + noise
    # without value. Flip to true (or set env GRAPH_RETRIEVAL_ENABLED=true) to
    # re-enable once the graph is back.
    GRAPH_RETRIEVAL_ENABLED: bool = False

    # ----- Google Gemini (LLM provider: answer, classifier, analyzer, extraction, cleaning) -----
    GEMINI_API_KEY: Optional[str] = None

    # Most LLM roles use gemini-2.5-flash-lite. The final answer uses the heavier
    # gemini-2.5-flash for better clinical synthesis. Override any role via env.
    ANSWER_MODEL: str = "gemini-2.5-flash"
    EXTRACTION_MODEL: str = "gemini-2.5-flash-lite"
    CLEANING_MODEL: str = "gemini-2.5-flash-lite"
    CLASSIFIER_MODEL: str = "gemini-2.5-flash-lite"
    QUERY_ANALYZER_MODEL: str = "gemini-2.5-flash-lite"
    SUMMARIZATION_MODEL: str = "gemini-2.5-flash-lite"

    # ----- Media / image upload (multimodal input) -----
    # Master switch for the /chat/image endpoint. When false the route returns
    # 503 so the text-only service is unaffected.
    MEDIA_UPLOAD_ENABLED: bool = True
    # Hard ceiling on a single uploaded file (bytes). Default 10 MB.
    MEDIA_MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    # Comma-separated allow-list of accepted MIME types. PNG + JPEG to start;
    # add "application/pdf" etc. here as processors gain support.
    MEDIA_ALLOWED_MIME_TYPES: str = "image/png,image/jpeg"
    # Where uploaded bytes are persisted by the default LocalMediaStorage.
    # Swap the storage backend (S3/GCS) without touching this path.
    MEDIA_STORAGE_DIR: str = "data/uploads"
    # Multimodal-capable model for image classification + understanding.
    # Defaults to the fuller flash model for stronger vision; override per env.
    VISION_MODEL: str = "gemini-2.5-flash"

    # Legacy — kept readable for backward compat but no longer required.
    OPEN_ROUTER_KEY: Optional[str] = None

    # Confidence-based stopping. The gatekeeper estimates a 0–100 confidence in
    # the leading diagnosis each turn (``diagnostic_confidence``). Once it reaches
    # this threshold the turn is treated as "terminal": the model stops asking
    # follow-up questions, delivers its assessment, and the validator drops any
    # follow_up_questions block. This replaces the old fixed-turn questioning
    # pattern as the PRIMARY stop signal.
    DIAGNOSTIC_CONFIDENCE_THRESHOLD: int = 80

    # Diagnostic turn cap. A hard backstop so a low-confidence conversation can't
    # interview forever: once a session reaches this many turns the answer is
    # forced "terminal" even if confidence never crossed the threshold.
    MAX_DIAGNOSTIC_TURNS: int = 8

    # Consolidation trigger. During a triage interview each turn is just the next
    # question (no per-turn summary). Once this many distinct clinical facts have
    # accumulated (symptom / duration / severity / medication / condition), the
    # turn "consolidates": the model emits a synthesised summary + assessment
    # instead of another bare question. Keeps summaries as checkpoints, not
    # turn-by-turn narration.
    #
    # Now that extraction is LLM-based (the analyzer reliably captures symptom +
    # duration + severity + medication as distinct slots), 3 distinct facts is a
    # solid "enough to consolidate" bar. Confidence and the turn backstop below
    # still trigger consolidation for single-slot cases (e.g. a wound).
    CONSOLIDATE_MIN_FACTS: int = 3

    # Turn-based consolidation backstop (in completed exchanges). The fact count
    # above under-fires when everything lands in one slot (e.g. a wound = all
    # "symptoms"), so after this many exchanges the assistant consolidates a
    # summary regardless — it must never keep asking indefinitely.
    CONSOLIDATE_AFTER_TURNS: int = 3

    # ----- Redis (session memory; optional — falls back to in-memory) -----
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL_SEC: int = 7200

    # ----- MongoDB (read-only authoritative patient demographics) -----
    # The canonical demographic record lives in `enervara.users`. It is loaded
    # per-turn (keyed on the request's user_id), projected to AI-safe fields
    # only, and injected into the answer prompt when clinically relevant. It is
    # NEVER written to, never embedded into Pinecone, and never mixed into Redis
    # or episodic memory. All of it fails open: any Mongo error or missing field
    # degrades to "no demographic context" without breaking /chat.
    MONGO_URI: Optional[str] = None
    MONGO_DB: str = "enervara"
    MONGO_USERS_COLLECTION: str = "users"
    # Master switch. When false (or MONGO_URI unset), demographics are skipped
    # entirely and the pipeline behaves exactly as before.
    DEMOGRAPHICS_ENABLED: bool = True
    # Short timeouts so a slow/unreachable Mongo never stalls a chat turn.
    MONGO_TIMEOUT_MS: int = 3000

    # ----- PMS (Patient Memory Service) — shadow producer, OFF by default -----
    # Master rollout flag. FALSE (production default) → NullPMSClient: no HTTP,
    # no latency, no behaviour change. TRUE → HttpPMSClient in fire-and-forget
    # SHADOW mode (chat never waits for PMS; PMS failures never affect users).
    ENABLE_PMS_SHADOW: bool = False
    # Identity-v1 rollout signal. Inbound parsing ALWAYS accepts both the legacy
    # (user_id/session_id) and new (identity envelope) formats regardless; this
    # flag only controls whether the new envelope is preferred at resolution.
    ENABLE_IDENTITY_V1: bool = True
    # PMS connection (only used when ENABLE_PMS_SHADOW=true; unset → NullPMSClient).
    # PMS_BASE_URL is the internal ALB URL, read from config — never hardcoded.
    PMS_BASE_URL: Optional[str] = None
    PMS_INGEST_PATH: str = "/v1/clinical-memory/events"
    # Service JWT for `Authorization: Bearer`. Provisioned out-of-band by the
    # configured auth component (secret manager / env) — never hardcoded. Without
    # it the client refuses to send (no anonymous PMS requests).
    PMS_SERVICE_JWT: Optional[str] = None
    # Separate connect + read timeouts (short — this is a background call).
    PMS_CONNECT_TIMEOUT_MS: int = 1000
    PMS_READ_TIMEOUT_MS: int = 1500
    PMS_MAX_RETRIES: int = 1          # bounded retry on transient (timeout/5xx/429) only
    PMS_CONSUMER_ID: str = "general-medicine"   # GM's identity toward PMS
    # Legacy single-timeout knob — kept for back-compat; connect/read above win.
    PMS_TIMEOUT_MS: int = 1500

    # ----- PostgreSQL (longitudinal memory; required for new memory/ subsystem) -----
    DATABASE_URL: Optional[str] = None
    MEMORY_CACHE_TTL_SEC: int = 300

    # ----- Episodic memory (Pinecone-backed, isolated from longitudinal) -----
    # Master switch for wiring the episodic layer into the main pipeline.
    # When false, the layer + its FastAPI app still work standalone, but the
    # CLI pipeline does not call it. Activated automatically when --user-id
    # is passed on the CLI.
    EPISODIC_MEMORY_ENABLED: bool = True
    PINECONE_EPISODIC_INDEX_NAME: str = "episodicmemory"
    EPISODIC_EXTRACTION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_CLARIFICATION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_CONTRADICTION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_COMPRESSION_MODEL: str = "gemini-2.5-flash-lite"
    EPISODIC_DEFAULT_TOP_K: int = 20      # how many to pull from Pinecone before rerank
    EPISODIC_DEFAULT_RETURN_K: int = 5    # how many to return after rerank + compression
    EPISODIC_DECAY_HALF_LIFE_DAYS: int = 14
    EPISODIC_MAX_CLARIFICATIONS_PER_TURN: int = 1  # contract — never exceed
    # Off-line ranking evaluation: when true, the retriever logs every query +
    # its ranked results to data/eval/retrieval_log.jsonl. Used by
    # `python -m episodic.eval.cli` for labeling and precision@k.
    EPISODIC_EVAL_LOGGING_ENABLED: bool = False
    EPISODIC_EVAL_LOG_PATH: str = "data/eval/retrieval_log.jsonl"
    EPISODIC_EVAL_LABELS_PATH: str = "data/eval/labels.jsonl"

    # Per-mode required-field sets. Add new modes here as needed.
    _REQUIRED_BY_MODE: ClassVar[dict[Mode, tuple[str, ...]]] = {
        "cli": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
        "api": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
        "ingest": ("PINECONE_API_KEY", "NEO4J_PASSWORD", "GEMINI_API_KEY"),
    }

    def validate_required(self, mode: Mode) -> None:
        """Raise ConfigError if any env var required for `mode` is unset."""
        required = self._REQUIRED_BY_MODE.get(mode)
        if required is None:
            raise ConfigError(
                f"Unknown mode '{mode}'. Expected one of: "
                f"{sorted(self._REQUIRED_BY_MODE)}"
            )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ConfigError(
                f"Missing required environment variables for mode '{mode}': "
                + ", ".join(missing)
                + ". Set them in your .env (see .env.example) or process environment."
            )


# Module-level singleton — safe to import even when env is empty.
settings = Settings()


class Config:
    """
    Legacy facade preserved for backward compatibility.

    Prefer `from graphrag.config.settings import settings` in new code.
    Attribute names are intentionally kept to match the original Config class
    (note: NEO4J_USER aliases NEO4J_USERNAME, NEO4J_PWD aliases NEO4J_PASSWORD).
    """

    PINECONE_API_KEY = settings.PINECONE_API_KEY
    PINECONE_INDEX_NAME = settings.PINECONE_INDEX_NAME
    PINECONE_NAMESPACE = settings.PINECONE_NAMESPACE
    NEO4J_URI = settings.NEO4J_URI
    NEO4J_USER = settings.NEO4J_USERNAME
    NEO4J_PWD = settings.NEO4J_PASSWORD
    GEMINI_API_KEY = settings.GEMINI_API_KEY
    OPEN_ROUTER_KEY = settings.OPEN_ROUTER_KEY  # legacy


__all__ = ["Settings", "Config", "ConfigError", "Mode", "settings"]
