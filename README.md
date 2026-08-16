# Enervera Dermatology GraphRAG

A dermatology question-answering service — covering skin, hair, nails and mucosa — that fuses dense vector search (Pinecone), a structured medical knowledge graph (Neo4j), Redis-backed short-term session memory, and a Pinecone-backed long-term episodic memory — exposed as an async **FastAPI** service that streams Gemini answers behind a multi-stage safety, routing, and retrieval pipeline.



---

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Environment Variables](#environment-variables)
8. [Data Pipeline](#data-pipeline)
9. [Running the Service](#running-the-service)
10. [HTTP API](#http-api)
11. [CLI (legacy)](#cli-legacy)
12. [Query Lifecycle](#query-lifecycle)
13. [Configuration Reference](#configuration-reference)
14. [Deployment (Docker + Render)](#deployment-docker--render)
15. [Observability](#observability)
16. [Testing](#testing)
17. [Known Limitations](#known-limitations)

---

## Architecture

```
                ┌──────────────────────────────────────┐
                │  HTTP client (curl, web app, agent)  │
                └──────────────────┬───────────────────┘
                                   │  POST /chat, /chat/stream, /episodic/*
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  FastAPI service (app/)                                       │
   │   • RequestID + Timing + optional X-API-Key middleware        │
   │   • Lifespan-built AppContainer (singletons, async)           │
   │   • JSON structlog with request_id / session_id binding       │
   └────────────────────────────────┬──────────────────────────────┘
                                    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  AsyncOrchestrator (app/services/orchestration/pipeline.py)   │
   └────────────────────────────────┬──────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼

  Stage -2: Memory Load        Stage -1: Gatekeeper       Stage 0: Routing
  SessionManager (async)       MedicalQueryAnalyzer       decide_routing()
  Redis-backed; builds         Gemini (gemini-2.5-flash-lite)  MEMORY_FIRST vs
  augmented retrieval query    intent / risk / action     HYBRID_RAG

       │                            │                            │
       └────────────┬───────────────┴────────────────────────────┘
                    ▼
            Stage 1: Vector Search
            PineconeRetriever (sync, wrapped in asyncio.to_thread)
            llama-text-embed-v2 + bge-reranker-v2-m3
                    │
                    ▼
            Stage 2: Entity Extraction
            EntityProcessor — flatten, prioritize, drug-pair boost
                    │
                    ▼
            Stage 3: Graph Traversal
            Neo4jRetriever (sync, wrapped in asyncio.to_thread)
            1-hop default, 2-hop for drug interactions
                    │
                    ▼
            Stage 3.5: Episodic Context  (optional, when user_id is set)
            episodic/ — Pinecone-backed long-term patient memory
                    │
                    ▼
            Stage 4: LLM Synthesis
            Gemini async (non-stream) OR sync-iterator → async-generator
            bridge for SSE streaming (/chat/stream)
                    │
                    ▼
            Stage 5: Episodic Ingest    (fire-and-forget asyncio.create_task)
            Stage 5b: Session Save      (SessionManager.save_session async)
```

All I/O on the request path is awaitable. The only sync clients in the project (Pinecone + Neo4j) are wrapped with `asyncio.to_thread`, so the event loop never blocks. There is **no `asyncio.run()` or `loop.run_until_complete()` in the FastAPI request path.**

---

## Features

- **Async-native FastAPI service** — `/chat`, `/chat/stream` (SSE), `/health`, `/healthz/ready`, `/metrics`, plus the full episodic memory API under `/episodic/*`.
- **Token streaming over SSE** — Gemini's sync stream is bridged into an `asyncio.Queue`-driven async generator so the first token reaches the client within ~1–2 s.
- **Two-tier query understanding** — LLM gatekeeper (Gemini Flash) for safety + a fast rule-first router for retrieval shaping.
- **Adaptive routing** — `MEMORY_FIRST` skips expensive retrieval for greetings and follow-ups; `HYBRID_RAG` engages full vector + graph for new medical topics.
- **Drug-interaction aware** — 2-hop Neo4j traversal and chunk-level drug-pair boosting for multi-drug queries.
- **Short-term session memory** — Redis-backed conversation state with rolling summarization and graceful in-memory fallback.
- **Long-term episodic memory** — per-user Pinecone namespace (`episodicmemory` index) with compression, gist generation, and recency/importance-weighted retrieval. Engaged automatically when `user_id` is supplied.
- **Optional X-API-Key auth** — middleware enforces a shared header on `/chat/*` and `/episodic/*` when `API_KEY` is set; `/health*` and `/metrics` always bypass.
- **Observability built in** — structlog JSON logs with request-scoped `request_id` / `session_id`; `/metrics` returns an in-memory snapshot (counters + p50/p95 latency).
- **Single-image deploy** — multi-stage Dockerfile + `render.yaml` Blueprint.

---

## Tech Stack

| Layer | Technology |
|---|---|
| HTTP framework | FastAPI + Uvicorn (`uvicorn[standard]`) |
| Vector store | Pinecone (`llama-text-embed-v2` embeddings, `bge-reranker-v2-m3` reranker) |
| Knowledge graph | Neo4j Aura (TLS, `neo4j+s://`) — same driver works against a local `bolt://` instance for dev |
| Short-term memory | Redis 5 (`redis.asyncio`, `orjson` serialization, 2-hour TTL) |
| Long-term memory | Pinecone (`episodicmemory` index, per-user namespace) |
| LLM | Google Gemini via `google-genai` — final **answer** on `gemini-2.5-flash`; gatekeeper / classifier / memory roles on `gemini-2.5-flash-lite` |
| Structured logging | structlog (JSON) bridged into stdlib `logging` |
| PDF parsing | PyMuPDF (`fitz`) |
| Data validation | Pydantic v2 + pydantic-settings |
| Container | python:3.11-slim multi-stage Dockerfile |
| Deploy target | Render (Web Service, Docker runtime) |
| Language | Python 3.10+ |

The final **answer** (`ANSWER_MODEL`) runs on `gemini-2.5-flash` for stronger clinical synthesis; every other role — gatekeeper, classifier, episodic compression, gist generation, summarizer — uses `gemini-2.5-flash-lite`. Override any role via the matching `*_MODEL` env var.

---

## Project Structure

```
RAG/
├── app/                                  # FastAPI service (production entry point)
│   ├── main.py                           # create_app() factory + module-level `app`
│   ├── container.py                      # AppContainer + build_container() lifespan wiring
│   ├── api/
│   │   ├── deps.py                       # get_container / ContainerDep
│   │   ├── middleware.py                 # RequestID, Timing, optional APIKey
│   │   └── routes/
│   │       ├── chat.py                   # POST /chat, POST /chat/stream
│   │       ├── health.py                 # GET /health, GET /healthz/ready
│   │       └── metrics.py                # GET /metrics + thread-safe registry
│   ├── core/
│   │   ├── config.py                     # facade over graphrag settings (+ API_KEY, PORT)
│   │   ├── logging.py                    # structlog JSON + stdlib bridge
│   │   ├── lifespan.py                   # FastAPI startup/shutdown
│   │   └── exceptions.py                 # AppError hierarchy + handlers
│   ├── services/
│   │   ├── orchestration/pipeline.py     # AsyncOrchestrator (run + stream)
│   │   ├── memory/session.py             # async drop-in for SessionMemoryAdapter
│   │   └── llm/streaming.py              # sync-iter → async-gen bridge
│   └── schemas/
│       ├── chat.py                       # ChatRequest, ChatResponse, ChatStreamEvent
│       └── common.py                     # ErrorResponse, HealthStatus, MetricsSnapshot
│
├── graphrag/                             # Core retrieval/synthesis engine (sync-safe)
│   ├── pipeline/graphrag_pipeline.py     # legacy sync pipeline (CLI only)
│   ├── retrievers/{pinecone,neo4j}_retriever.py
│   ├── processors/entity_processor.py
│   ├── query_understanding/{analyzer,query_config,query_types,routing}.py
│   ├── memory/session_adapter.py         # sync facade — used only by the legacy CLI
│   ├── llm/gemini_client.py              # generate_text, generate_text_async, generate_stream
│   ├── config/settings.py
│   └── utils/{logger,rate_limit}.py
│
├── Memory_Layer/session_memory/          # Async Redis short-term memory
│   ├── models.py                         # Pydantic models (Message, StructuredState, ...)
│   ├── session_manager.py                # async Redis CRUD + in-memory fallback
│   ├── retriever.py                      # WorkingMemory assembly
│   ├── state_extractor.py                # clinical-state extraction
│   ├── summarizer.py                     # rolling summary builder
│   └── context_builder.py                # token-budgeted prompt assembly
│
├── episodic/                             # Long-term episodic memory subsystem
│   ├── api/{app.py,routes.py,dependencies.py}
│   ├── retrieval/, ingestion/, compression/, repository/, schemas/, ...
│
├── chunking/                             # Document → MicroChunk pipeline
│   └── (loaders, cleaners, extractors, normalizers, validators, schemas, storage)
│
├── scripts/                              # Ingestion + maintenance entry points
│   ├── chunker.py, clean_chunks.py
│   ├── ingest_pinecone.py, ingest_neo4j.py
│   ├── progress.py, check_api.py
│
├── memory/                               # Postgres/pgvector longitudinal layer (built but unwired)
├── documents/                            # Source PDFs
├── data/                                 # Final processed chunks per category
│
├── main.py                               # legacy interactive CLI (sync, kept for local debugging)
├── Dockerfile                            # multi-stage build for Render
├── render.yaml                           # Render Blueprint
├── pyproject.toml                        # dependencies + optional extras (api, memory, test, dev)
├── .env.example                          # template
└── tests/
    ├── unit/                             # routing + analyzer unit tests
    ├── integration/                      # episodic integration + FastAPI integration tests
    └── StressTest/                       # Excel-driven batch runner
```

---

## Prerequisites

- **Python 3.10+** (Docker image uses 3.11-slim)
- **Pinecone** account with two indexes:
  - the chunk index (default name `enervera`)
  - the episodic memory index (default name `episodicmemory`)
- **Neo4j Aura** instance (recommended — copy the `neo4j+s://` URI from the Aura console). A local `bolt://127.0.0.1:7687` instance works the same way for offline dev.
- **Redis** instance — the service starts without it but loses short-term session persistence
- **Google Gemini** API key

---

## Installation

```powershell
git clone <repo-url>
cd RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate              # macOS/Linux

# Editable install with the FastAPI extras (and memory + test if you want them):
pip install -e ".[api]"
# Full developer set:
# pip install -e ".[dev]"
```

`pyproject.toml` is the source of truth for dependencies. `requirements.txt` is kept as a convenience snapshot only.

---

## Environment Variables

Copy `.env.example` to `.env` and fill the secrets in. Service-relevant variables:

```env
# --- LLM ---
GEMINI_API_KEY=...                       # required; powers every LLM role

# --- Pinecone ---
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=enervera             # chunk index
PINECONE_EPISODIC_INDEX_NAME=episodicmemory

# --- Neo4j (Aura) ---
NEO4J_URI=neo4j+s://<your-aura-dbid>.databases.neo4j.io   # or bolt://127.0.0.1:7687 for local
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

# --- Redis ---
REDIS_URL=redis://localhost:6379/0       # optional; falls back to in-memory store

# --- Episodic memory ---
EPISODIC_MEMORY_ENABLED=true

# --- FastAPI service ---
PORT=8000
LOG_LEVEL=INFO
API_KEY=                                 # leave blank to disable auth; set a token to enforce X-API-Key
```

Run `python scripts/check_api.py` to verify the LLM + Pinecone + Neo4j + Redis credentials before booting the service.

---

## Data Pipeline

The ingestion flow runs in four stages. Each writes a versioned artifact so steps can be re-run independently. All entry points now live under `scripts/`.

### 1. Chunk source PDFs

```powershell
python scripts/chunker.py
```

Walks `documents/`, runs `DocumentProcessingPipeline.process_pdf` per file, and writes raw chunks to `chunking/output/v1/<category>/<book>/`. Uses `ThreadPoolExecutor(max_workers=20)` and marks processed blocks with `.done` files so re-runs skip completed work. Failed blocks land in `logs/failed_blocks/`.

### 2. Clean and normalize

```powershell
python scripts/clean_chunks.py
```

A second LLM pass (Gemini Flash) normalizes entity types and relation types against a fixed clinical vocabulary, removes duplicates, and re-validates against `MicroChunk`. Output: `chunking/output/v2_cleaned/`.

### 3. Ingest to Pinecone

```powershell
python scripts/ingest_pinecone.py
```

Batches of 50 chunks are embedded with `llama-text-embed-v2` (Pinecone hosted inference) and upserted with metadata: `chunk_id`, `entities` (flat `"type: name"` strings), `book`, `topic`, and a 300-char `summary`.

### 4. Ingest to Neo4j

```powershell
python scripts/ingest_neo4j.py
```

Each `MicroChunk` becomes:

- **Nodes** — `:Entity` with a dynamic type label (`:Disease`, `:Drug`, `:Symptom`, …) merged on `name`.
- **Edges** — typed relations (`CAUSES`, `TREATS`, `INDICATES`, `INCREASES_RISK_OF`, …) merged with `ON CREATE SET created_at = timestamp()`.

A 20-relations-per-source cap prevents fan-out explosions.

### Monitoring progress

```powershell
python scripts/progress.py
```

---

## Running the Service

### Local — uvicorn

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Local — Docker

```powershell
docker build -t enervera-api .
docker run --rm -p 8000:8000 --env-file .env enervera-api
```

The container drops to a non-root user (`enervera`, uid 10001) and exposes port 8000. `PORT` is honored if set.

Open `http://localhost:8000/docs` for the live OpenAPI explorer.

---

## HTTP API

All endpoints accept and return JSON unless noted. **For frontend integration (CORS, SSE parsing, auth, error envelope, JS/TS examples) read [docs/FRONTEND.md](docs/FRONTEND.md) — that's the canonical contract for any client outside this repo.**

### `GET /health`

Cheap liveness probe. No upstream I/O. Returns `{"status":"ok"}` typically in <50 ms. Used as Render's `healthCheckPath`.

### `GET /healthz/ready`

Pings Redis, Pinecone, and Neo4j with short timeouts. Returns each subsystem's status. Use for orchestrator readiness gates.

### `GET /metrics`

In-memory snapshot — no Prometheus dependency. Fields:

```json
{
  "requests_total": 0,
  "requests_inflight": 0,
  "errors_total": 0,
  "uptime_seconds": 0,
  "latency_ms_p50": 0,
  "latency_ms_p95": 0,
  "pinecone_calls_total": 0,
  "neo4j_calls_total": 0,
  "llm_tokens_total": 0
}
```

### `POST /chat`

```json
{
  "query": "What are the common symptoms of myocarditis?",
  "session_id": "patient_42",
  "user_id": "cardio_test"
}
```

`session_id` is optional (auto-generated UUID hex when omitted). `user_id` is optional — when present, the long-term episodic memory pipeline is engaged.

Response:

```json
{
  "answer": "...",
  "session_id": "patient_42",
  "request_id": "01J...",
  "analysis": { "intent": "symptom_query", "risk_level": "low", "final_action": "retrieve" },
  "routing": { "mode": "HYBRID_RAG", "vector_top_k": 15, "graph_hops": 1 },
  "timing_ms": { "memory_load": 12, "gatekeeper": 280, "vector": 410, "graph": 95, "llm": 1840, "total": 2700 },
  "followup_questions": ["How long have the symptoms been present?"]
}
```

### `POST /chat/stream`

Same request body. Returns `text/event-stream`:

```
data: {"type":"chunk","data":"Hyper"}

data: {"type":"chunk","data":"tension is..."}

data: {"type":"done","timing_ms":{"total":3120}}

data: [DONE]
```

Test from a shell:

```powershell
curl.exe -N -X POST http://localhost:8000/chat/stream `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"What is hypertension?\",\"session_id\":\"demo-1\"}'
```

### Episodic memory — `/episodic/*`

The episodic FastAPI router is mounted into the same app. Key endpoints (full list under `/docs`):

- `POST /episodic/context` — retrieve compressed episodic context for a query
- `POST /episodic/ingest` — ingest a turn into long-term memory
- `GET /episodic/user/{user_id}` — list episodes
- `DELETE /episodic/user/{user_id}` — purge a user's namespace

```powershell
curl.exe -X POST http://localhost:8000/episodic/context `
  -H "Content-Type: application/json" `
  -d '{\"user_id\":\"cardio_test\",\"query_text\":\"chest pain\"}'
```

### Authentication

When `API_KEY` is set in the environment, every request to `/chat/*` and `/episodic/*` must carry a matching `X-API-Key` header. `/health*` and `/metrics` always bypass.

```powershell
curl.exe -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -H "X-API-Key: $env:API_KEY" `
  -d '{\"query\":\"I have a mild headache\",\"session_id\":\"demo-1\"}'
```

### Error envelope

All handled errors return a uniform shape:

```json
{
  "code": "UPSTREAM_UNAVAILABLE",
  "message": "Pinecone request failed",
  "request_id": "01J...",
  "details": null
}
```

Codes: `INVALID_INPUT` (400), `UNAUTHORIZED` (401), `RATE_LIMITED` (429), `UPSTREAM_UNAVAILABLE` (502/503), `INTERNAL_ERROR` (500 catch-all).

---

## CLI (legacy)

The original sync REPL still works for local debugging — it bypasses FastAPI entirely.

```powershell
python main.py                            # interactive
python main.py "What is myocarditis?"     # one-shot
python main.py --session-id patient_42    # persistent session
python main.py --new-session              # random UUID session
```

REPL meta-commands: `:memory`, `:session`, `:help`, `quit`.

This path uses `asyncio.run()` internally, which is fine for a CLI but **never** reachable from the FastAPI service.

---

## Query Lifecycle

For a query like **"Can I take ibuprofen with my fever?"** in an existing session (`user_id=cardio_test`):

1. **Memory load (Stage -2)** — `SessionManager.load_session` returns the existing `SessionMemory` from Redis. `build_retrieval_query` composes the augmented retrieval query (current question + structured state + recent turns).
2. **Gatekeeper (Stage -1)** — `MedicalQueryAnalyzer.aanalyze` returns `{intent: "medication_query", risk_level: "low", final_action: "retrieve"}`. Emergency or non-medical actions short-circuit here.
3. **Routing (Stage 0)** — `decide_routing` recognizes the drug-interaction pattern, escalates to `HYBRID_RAG`, and pulls the `DRUG_INTERACTION` config (`vector_top_k=15`, `graph_hops=2`, `boost_drug_pairs=True`).
4. **Vector retrieval (Stage 1)** — `await asyncio.to_thread(pinecone.retrieve, ...)` returns 15 chunks reranked to the top 5.
5. **Entity extraction (Stage 2)** — `ibuprofen`, `fever`, and related entities are flattened. Chunks containing multiple query drugs are promoted.
6. **Graph traversal (Stage 3)** — `await asyncio.to_thread(neo4j.retrieve_relations, ...)` returns 2-hop paths like `ibuprofen -[INCREASES_RISK_OF]→ gi_bleeding -[COMPLICATES]→ heart_disease`.
7. **Episodic context (Stage 3.5)** — `await episodic.context_pipeline.build(...)` returns compressed prior episodes from this user's Pinecone namespace.
8. **LLM synthesis (Stage 4)** — `assemble_memory_payload` builds the token-budgeted prompt; Gemini streams tokens over SSE (or is awaited end-to-end for `/chat`).
9. **Episodic ingest (Stage 5)** — fire-and-forget `asyncio.create_task(episodic.ingest_pipeline.run(...))`; client response is not blocked.
10. **Session save (Stage 5b)** — `extract_state` parses the user turn for new clinical entities, both turns appended, `maybe_summarize` compresses older turns past the 8-turn threshold, persisted back to Redis.

---

## Configuration Reference

### Per-query retrieval config — `graphrag/query_understanding/query_config.py`

| Query type | `vector_top_k` | `reranker_top_k` | `graph_hops` | Priority entities |
|---|---|---|---|---|
| `symptom_query` | 15 | 5 | 1 | disease, symptom, syndrome |
| `drug_interaction` | 15 | 5 | 2 | drug, drug_class, mechanism, side_effect |
| `diagnosis` | 15 | 5 | 1 (off) | disease, syndrome, condition |
| `guideline` | 20 | 7 | 1 | procedure, drug, treatment, protocol |
| `lab_interpretation` | 15 | 5 | 1 | test, lab_value, biomarker |
| `prognosis` | 15 | 5 | 1 | outcome, risk_factor, survival |
| `out_of_context` | 0 | 0 | 0 | — |
| `unknown` | 15 | 5 | 1 | — |

When routing flips to `MEMORY_FIRST`, the pipeline overrides these to `top_k=2`, `reranker=2`, `graph_hops=0`.

### Short-term memory tuning — `Memory_Layer/session_memory/`

| Constant | Default | File |
|---|---|---|
| `MAX_RECENT_TURNS` | 10 | `models.py` |
| `SESSION_TTL_SEC` | 7200 (2h) | `session_manager.py` |
| `SUMMARIZE_THRESHOLD` | 8 turns | `summarizer.py` |
| `KEEP_LAST_N` | 4 turns | `summarizer.py` |
| `DEFAULT_TOKEN_BUDGET` | 3500 | `context_builder.py` |

### Chunk schema constraints — `chunking/schemas/models.py`

- `len(entities) >= 5`
- `len(relations) >= len(entities) / 2`
- `approx_tokens <= 400`

---

## Deployment (Docker + Render)

### Dockerfile

The multi-stage `Dockerfile` builds the wheel + installs the `api` extra in a builder stage, then copies site-packages into a slim runtime stage. The runtime drops to a non-root user (`enervera`, uid 10001) and runs:

```
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips=*
```

### Render Blueprint

`render.yaml` defines a single Docker Web Service on the Starter plan:

```yaml
services:
  - type: web
    name: enervera-api
    runtime: docker
    plan: starter
    region: oregon
    healthCheckPath: /health
    envVars:
      - key: GEMINI_API_KEY        # secret — set in Render dashboard
        sync: false
      - key: PINECONE_API_KEY
        sync: false
      - key: NEO4J_URI
        sync: false
      - key: NEO4J_USERNAME
        sync: false
      - key: NEO4J_PASSWORD
        sync: false
      - key: REDIS_URL
        sync: false
      - key: API_KEY
        sync: false
      - key: PINECONE_INDEX_NAME
        value: enervera
      - key: PINECONE_EPISODIC_INDEX_NAME
        value: episodicmemory
      - key: EPISODIC_MEMORY_ENABLED
        value: "true"
      - key: LOG_LEVEL
        value: INFO
```

### Deploy steps

1. **Push** the repo to GitHub (Render reads from a git remote).
2. **New → Blueprint** in the Render dashboard; point it at the repo. Render reads `render.yaml`.
3. **Set secrets** (`sync: false` keys) in the dashboard — these are never committed.
4. **Deploy.** First build is ~2–3 min; first cold start ~10–15 s (Pinecone + Neo4j connection warm-up).
5. **Verify:**

   ```powershell
   curl.exe https://<service>.onrender.com/health
   curl.exe -X POST https://<service>.onrender.com/chat `
     -H "Content-Type: application/json" `
     -H "X-API-Key: $env:API_KEY" `
     -d '{\"query\":\"What is hypertension?\",\"session_id\":\"prod-smoke\"}'
   ```

Render's Starter plan idles inactive containers — first request after a sleep pays the boot cost again. Use a paid plan or a 5-minute keep-warm ping if cold start matters.

---

## Observability

- **Structured logs** — every log line is a single JSON object with `event`, `level`, `timestamp`, `logger`, and (when middleware bound them) `request_id`, `session_id`, `stage`, `duration_ms`. Both `structlog.get_logger()` and stdlib `logging.getLogger(__name__)` flow through the same JSON renderer.
- **Request IDs** — `X-Request-ID` is echoed back on every response (minted if the client didn't supply one) and bound onto structlog contextvars for the lifetime of the request.
- **Metrics snapshot** — `GET /metrics` returns counters and a rolling p50/p95 latency window (1024 samples). No external metrics dependency; if you scale beyond one instance, switch to Prometheus.
- **`/healthz/ready`** — distinguishes "process is up" (`/health`) from "all upstreams reachable" (`/healthz/ready`).

---

## Testing

```powershell
# Unit tests — pure logic, no network
pytest tests/unit/ -v

# Integration tests — hit real Pinecone + Neo4j + Redis + Gemini
pytest -m episodic_integration -v

# FastAPI end-to-end (httpx + ASGITransport, no port bind)
pytest tests/integration/test_api_chat.py -v
```

The FastAPI integration suite (`tests/integration/test_api_chat.py`) covers:

- `/health` returns 200 fast
- `/metrics` exposes the expected counters
- `/chat` returns a non-empty answer with per-stage `timing_ms`
- `/chat/stream` yields at least one `chunk` event followed by a terminal `done` + `[DONE]`
- `API_KEY` enforcement (401 without header, 200 with matching header)
- `/health` bypasses auth even when `API_KEY` is set

### Stress test (Excel-driven batch runner)

```powershell
python tests/StressTest/run_stress_test.py `
    --excel "tests/StressTest/RAG STRESS TEST 2.xlsx" `
    --delay 2
```

Reads questions from a `questions` column, writes answers back to `model_answer`, and saves after every row.

---

## Known Limitations

- **Single-process throughput.** One uvicorn worker handles ~50 concurrent SSE streams comfortably. Beyond that, run with `--workers 2+` (sessions are in Redis, so horizontal scaling is safe).
- **Render cold start ~10–15 s** because Pinecone lazy-builds its index client and the Neo4j driver opens a TLS connection to Aura. Lifespan pre-warms both, but Starter-plan idle suspend still pays the boot cost on the first request after sleep.
- **Pinecone + Neo4j drivers are sync.** They're wrapped with `asyncio.to_thread` rather than rewritten; this is fine for current load. Revisit if QPS justifies a true async driver.
- **In-memory metrics.** `/metrics` is per-process. Aggregate via Prometheus / OTel when running more than one instance.
- **No rate limiting / WAF in-app.** Render's frontproxy handles abuse; slowapi can be layered in later if needed.
- **Graph traversal (Stage 3) is currently disabled** (`GRAPH_RETRIEVAL_ENABLED=false`) — the Neo4j Aura instance is unreachable, so the pipeline skips the graph step to avoid latency + errors. Answers run on vector retrieval + memory only. Set `GRAPH_RETRIEVAL_ENABLED=true` to re-enable once Neo4j is back.
- **Longitudinal Postgres memory is built but unwired.** The `memory/` subsystem is intentionally not part of the request path right now.
- **Token estimation is approximate.** A flat 4-chars-per-token heuristic drives context-window budgeting. Swap in a real tokenizer for tight budgets.
- **Neo4j drops edges silently** when a source entity already has 20 relations.

---

## License

Internal project. All rights reserved.
