# Frontend integration guide

The Enervera Dermatology GraphRAG service is **API-only**. Any UI — web app, mobile app, embedded widget — runs in its own repository / deployment and talks to this service over HTTPS. This document is the contract between that frontend and the service.

If you're reading this from a frontend repo: everything you need is here. The backend is documented elsewhere; you should never need to read backend code to integrate.

---

## Table of contents

0. [Build guide — start here (for the assistant wiring this up)](#build-guide--start-here)
1. [Quick start](#quick-start)
2. [Base URL and environments](#base-url-and-environments)
3. [Authentication](#authentication)
4. [CORS](#cors)
5. [Endpoints](#endpoints)
   - [`GET /health`](#get-health)
   - [`GET /healthz/ready`](#get-healthzready)
   - [`GET /metrics`](#get-metrics)
   - [`POST /chat`](#post-chat) — non-streaming JSON answer
   - [`POST /chat/stream`](#post-chatstream) — SSE prose stream
   - [`POST /chat/blocks`](#post-chatblocks) — NDJSON typed UI blocks
   - [`POST /chat/stream/blocks`](#post-chatstreamblocks) — SSE typed UI blocks
   - [`POST /chat/image`](#post-chatimage) — multipart image upload
   - [`/episodic/*`](#episodic)
6. [Error envelope](#error-envelope)
7. [Session and user_id semantics](#session-and-user_id-semantics)
8. [SSE streaming — full protocol](#sse-streaming--full-protocol)
9. [Reference clients](#reference-clients)
   - [Fetch / vanilla JS](#fetch--vanilla-js)
   - [TypeScript types](#typescript-types)
   - [React hook](#react-hook)
   - [curl](#curl)
10. [Operational notes](#operational-notes)
11. [Versioning and compatibility](#versioning-and-compatibility)

---

## Build guide — start here

> **If you are an AI assistant (or engineer) building the chat UI: implement from this section.** It is self-contained — you do not need to read backend code. The rest of the document is reference depth you can consult as needed.

### The whole model in one paragraph

Send a chat turn to **`POST /chat/blocks`**. The response streams **typed "blocks"** as **NDJSON** — one JSON object per line, each `{ "type": ..., "data": {...} }`. You render each block by its `type` as it arrives. A chat turn = an ordered list of rendered blocks (a summary bubble, maybe a warning banner, condition cards, next-steps, and at most one follow-up chip). Persist a `session_id` across turns so the assistant remembers the conversation. That's the entire integration.

**Pick this endpoint:** use `/chat/blocks` for a structured chat UI. (`/chat` returns one plain-text `answer` if you only want a text bubble; `/chat/stream/blocks` is the same blocks over SSE if your infra needs `EventSource`. See the [endpoint chooser](#which-chat-endpoint-should-i-use).)

### What you build

1. A **chat client** that POSTs a turn and yields blocks (drop-in code below).
2. A **block renderer** that maps each `block.type` to a UI component (spec below — implement all seven types).
3. **Session state**: one `session_id` per conversation, persisted and reused.
4. **Safety rendering**: emergency / mental-health-crisis turns arrive as blocks — render prominently, make phone numbers tappable, keep the crisis one calm (not a red error).
5. Optional: **image upload** via `/chat/image` (multipart) — returns the same blocks + `media` metadata.

### Drop-in TypeScript client (`/chat/blocks`)

```ts
export type Block =
  | { type: "summary";             data: { text: string } }
  | { type: "key_points";          data: { points: string[] } }
  | { type: "bullet_list";         data: { title: string | null; items: string[] } }
  | { type: "follow_up_questions"; data: { questions: string[] } }   // ≤ 1 question
  | { type: "warning";             data: { text: string; severity: "info" | "caution" | "critical" } }
  | { type: "next_steps";          data: { steps: string[] } }
  | { type: "condition_list";      data: { conditions: { name: string; likelihood: string | null; description: string | null }[] } }
  | { type: "decision";            data: { verdict: "yes" | "no" | "possibly" | "seek_urgent_care" | "insufficient_information"; rationale: string } }
  | { type: "otc_medications";     data: { medications: { name: string; purpose: string; dosage: string | null; caution: string | null }[] } }
  | { type: "lab_tests";           data: { tests: { name: string; reason: string; urgency: "routine" | "soon" | "urgent" | null }[] } }
  | { type: "answer_state";        data: { show_doctor_summary: boolean } };   // control block, always last

export interface ChatTurn { query: string; session_id?: string; user_id?: string }

/** Stream typed blocks for one chat turn. Each yielded value is a fully-valid Block. */
export async function* streamChatBlocks(
  base: string,
  body: ChatTurn,
  init: { apiKey?: string; signal?: AbortSignal } = {},
): AsyncGenerator<Block, void, void> {
  const res = await fetch(`${base}/chat/blocks`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(init.apiKey && { "X-API-Key": init.apiKey }) },
    body: JSON.stringify(body),
    signal: init.signal,
  });
  if (!res.ok || !res.body) throw new Error(`chat failed: HTTP ${res.status} ${await res.text().catch(() => "")}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder("utf-8");
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line) as Block;   // guaranteed valid JSON block
    }
  }
  if (buf.trim()) yield JSON.parse(buf) as Block;    // final line may lack a trailing \n
}
```

Usage (framework-agnostic — swap `renderBlock` for your components):

```ts
async function sendTurn(query: string) {
  const blocks: Block[] = [];
  for await (const block of streamChatBlocks(BASE, { query, session_id: sessionId, user_id }, { apiKey })) {
    blocks.push(block);
    renderBlock(block);          // append/replace in the current assistant message
  }
  // `blocks` is the full assistant turn once the loop ends.
}
```

### Block → UI rendering spec (implement all nine content blocks + 1 control block)

Render blocks **in the order received** (don't reorder by type). `data` shapes are exact — no other fields appear.

| `type` | `data` | Render as |
|---|---|---|
| `summary` | `{ text }` | The main assistant message text (a chat bubble / paragraph). Usually the first block. |
| `decision` | `{ verdict: "yes"\|"no"\|"possibly"\|"seek_urgent_care"\|"insufficient_information", rationale }` | A **verdict headline** — a colored pill/banner with the display label (map `"seek_urgent_care"` → "SEEK URGENT CARE", `"insufficient_information"` → "NEED MORE INFO", else uppercase the word) followed by the `rationale` text. Leads decision-type turns instead of a `summary`. Suggested colors: `yes` green, `no` red, `possibly` amber, `seek_urgent_care` prominent red, `insufficient_information` neutral. |
| `key_points` | `{ points: string[] }` | A bulleted list. |
| `bullet_list` | `{ title: string\|null, items: string[] }` | Optional heading (`title`) + bulleted `items`. |
| `condition_list` | `{ conditions: [{ name, likelihood: string\|null, description: string\|null }] }` | "Possible causes" — one card/row per condition: **name**, a `likelihood` badge (`"most likely"`/`"possible"`/`"less likely"` — may be null), and `description` (may be null). |
| `warning` | `{ text, severity: "info"\|"caution"\|"critical" }` | An alert. **`severity` drives styling:** `info` = subtle, `caution` = amber, `critical` = prominent. Always show `critical` front-and-center. |
| `next_steps` | `{ steps: string[] }` | A "What to do" list (numbered/ordered). Linkify any phone numbers as `tel:`. |
| `follow_up_questions` | `{ questions: string[] }` (≤1 item) | A single suggested-reply **chip**. Tapping it sends its text as the next turn's `query`. |
| `otc_medications` | `{ medications: [{ name, purpose, dosage: string\|null, caution: string\|null }] }` | A "Over-the-counter options" card list, shown **last**. One row per medicine: **name** (bold), `purpose` (what it helps), `dosage` badge if present, and `caution` as a subtle warning line if present. These are self-care OTC suggestions only — a good place for a small "not a prescription — check with a pharmacist" disclaimer. |
| `lab_tests` | `{ tests: [{ name, reason, urgency: "routine"\|"soon"\|"urgent"\|null }] }` | A "Recommended tests / investigations" card list, shown near the end (before `otc_medications`). One row per test: **name** (bold), `reason` (why it's suggested), and an `urgency` badge if present (`urgent` red, `soon` amber, `routine`/null neutral). These are suggested investigations to discuss with a doctor/lab, **not orders** — good spot for a "confirm with your doctor" note and a future "book a test" CTA. |
| `answer_state` | `{ show_doctor_summary: boolean }` | **Control block, not a message — don't render it.** It's always the **very last** block of every turn. Read `show_doctor_summary`: when `true`, reveal a **"Show this to your doctor"** button that calls `POST /chat/soap` (see below). Once true it stays true for the session. |

### Guarantees you can build on (don't code defensively around these)

- **Every line is a valid block** — `JSON.parse` + switch on `type`; you never get partial objects, wrong-typed fields, or non-JSON.
- **Always ≥ 1 block** — if generation fails you still get a fallback `summary`; the stream is never empty.
- **≤ 1 `follow_up_questions` block, ≤ 1 question inside** — render one chip, never a list of questions. On a closing/assessment turn there's no follow-up at all.
- **Conversation converges** — the assistant asks a focused question while gathering, then delivers a `summary` (+ `condition_list`/`next_steps`) once it has enough. You don't manage this; just render what streams.

### Safety responses (render these prominently)

Some turns short-circuit to a fixed, safety-reviewed block set. Detect by block shape + `warning.severity` and style accordingly:

| Situation | Blocks (in order) | Render |
|---|---|---|
| **Physical emergency** | `warning`(`critical`) → `next_steps` | Prominent critical banner; make emergency numbers (112 / 108) tappable `tel:` links. |
| **Mental-health crisis** | `summary`(empathetic) → `warning`(`critical`) → `next_steps`(helplines: Tele-MANAS 14416, KIRAN 1800-599-0019) | **Calm, supportive — NOT a red error.** Lead with the summary; keep helpline numbers tappable. |
| **Refusal / out-of-scope** | a single `summary` | Plain message. |

Rule of thumb: **any `warning` with `severity:"critical"` = show it front-and-center and make the numbers in the following `next_steps` tappable.**

### Session + user_id (get this right or memory breaks)

- **`session_id`**: one per conversation. First turn — send your own or omit it; the response's `session_id` is authoritative, **persist it** (e.g. `sessionStorage`) and send it on every subsequent turn. Same id ⇒ the assistant remembers earlier turns (symptoms, answers) and won't re-ask.
- **`user_id`**: optional, stable per real person (not per device/session). Enables long-term memory across conversations. Omit if you don't have accounts.

### Image upload (`/chat/image`)

`multipart/form-data`, **not** JSON — let the browser set the boundary (don't set `Content-Type`). Returns a normal chat response (JSON, not a stream) plus a `media` object.

```ts
const fd = new FormData();
fd.append("image", file);                 // File from <input type="file">, image/png or image/jpeg, ≤10MB
fd.append("query", "is this rash serious?"); // optional
fd.append("session_id", sessionId);          // optional
const res = await fetch(`${BASE}/chat/image`, {
  method: "POST",
  headers: { ...(apiKey && { "X-API-Key": apiKey }) },
  body: fd,
}).then(r => r.json());   // { answer, media: { category, route, caption, extracted_facts, ... }, ... }
```

### Do / Don't

- **Do** render blocks in arrival order; **do** persist `session_id`; **do** send `X-API-Key` when configured; **do** linkify phone numbers.
- **Don't** set `Content-Type` on the image upload; **don't** re-order or dedupe blocks; **don't** build defensive JSON repair (the server guarantees validity); **don't** render a lone `warning(critical)` quietly — it's urgent.

### Build checklist

- [ ] Config: `BASE` URL + optional `apiKey` (env, never hard-coded in a public bundle — see [Authentication](#authentication)).
- [ ] `streamChatBlocks` client wired to `/chat/blocks`.
- [ ] `renderBlock` handles all 7 types (table above).
- [ ] `session_id` persisted from the first response and reused.
- [ ] Follow-up chip: tapping sends its text as the next turn.
- [ ] Safety rendering: `warning(critical)` prominent; crisis calm; `tel:` links.
- [ ] Error handling: non-2xx → parse the [error envelope](#error-envelope); retry `UPSTREAM_UNAVAILABLE`/`RATE_LIMITED` with backoff.
- [ ] (Optional) `/chat/image` upload + `media` display; `user_id` for long-term memory.

Everything below is reference depth (auth, CORS, SSE prose protocol, per-endpoint detail, ops). Consult it when you need it.

---

## Quick start

```ts
const BASE = "https://enervera-api.onrender.com";

// Health check — no auth, no body
await fetch(`${BASE}/health`).then(r => r.json());

// Send a chat message
const r = await fetch(`${BASE}/chat`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": import.meta.env.VITE_ENERVERA_API_KEY,
  },
  body: JSON.stringify({
    query: "I have a mild headache and a slight fever since this morning.",
    session_id: "session-abc-123",
    user_id: "patient-42",          // optional; enables long-term episodic memory
  }),
});
const { answer, timing_ms, request_id } = await r.json();
```

That's the minimum. The streaming endpoint, error handling, and richer integration patterns are below.

---

## Base URL and environments

The service exposes everything on a single origin. There are no path prefixes for environments.

| Environment | URL (example) |
|---|---|
| Local dev | `http://localhost:8000` |
| Render staging | `https://enervera-api-staging.onrender.com` |
| Render production | `https://enervera-api.onrender.com` |

Decide the URL per-environment in your frontend's own config — the API itself is identical across environments. Render's Starter plan idles after ~15 min of inactivity, so the first request after sleep takes 10–15 s. Show a loading state and don't bail early.

---

## Authentication

Authentication is an **optional shared header**. The backend's behaviour is:

- If the backend's `API_KEY` env var is **unset**, no authentication is required. Any caller can hit `/chat` and `/episodic`.
- If `API_KEY` is **set**, every request to `/chat/*` and `/episodic/*` must carry `X-API-Key: <that-value>`. Missing or wrong → `401 UNAUTHORIZED`.
- `/health`, `/healthz/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json` are **always public** and never need the header.

The frontend SHOULD send the header unconditionally on protected routes — it's harmless when the backend doesn't require it.

```ts
const headers: Record<string, string> = { "Content-Type": "application/json" };
if (apiKey) headers["X-API-Key"] = apiKey;
```

> **Never embed the API key in client-side bundles in a public deployment.** It will leak to anyone with DevTools. For a public web app, route requests through a tiny BFF (Backend-For-Frontend) that holds the key server-side, OR run the backend without an API key behind a network boundary. This header is fit for internal tools, partner integrations, and server-to-server calls.

---

## CORS

The backend ships `CORSMiddleware`. Configure allowed origins via the `CORS_ORIGINS` env var on the backend:

| `CORS_ORIGINS` value | Behaviour |
|---|---|
| `*` (default) | Any origin allowed. `credentials` disabled (browser enforces this with `*`). Fine for early testing. |
| `https://app.enervera.com,https://staging.enervera.com` | Exact-match origins only. `credentials` enabled — but this service uses header auth, not cookies, so it makes no functional difference. |

**Pin `CORS_ORIGINS` to your real frontend URL(s) before going to production.** A wildcard combined with an API key is a footgun: anyone with the key can call from any origin.

The middleware accepts `GET`, `POST`, `OPTIONS` and the headers `Content-Type`, `X-API-Key`, `X-Request-ID`, `Accept`. Preflight cache: 10 minutes.

`X-Request-ID` is **exposed back to JS** so the frontend can log it next to errors and quote it in support tickets.

---

## Endpoints

All bodies are JSON unless otherwise noted. Responses are JSON except the streaming ones.

### Which chat endpoint should I use?

There are five ways to send a chat turn. **They all share the same request fields** (`query`, `session_id?`, `user_id?`) — pick by how you want to render the answer:

| Endpoint | Transport | Answer shape | Use when |
|---|---|---|---|
| `POST /chat` | JSON in/out | one text `answer` | simplest; you just want the final text |
| `POST /chat/stream` | SSE | text streamed token-by-token | live "typing" UX, free-text bubble |
| `POST /chat/blocks` | NDJSON | typed UI blocks (cards, chips, warnings) | you render structured UI and can read a `fetch` body stream |
| `POST /chat/stream/blocks` | SSE | same typed blocks, SSE-framed | structured UI but you prefer `EventSource`/SSE infra |
| `POST /chat/image` | multipart in, JSON out | one text `answer` + `media` metadata | the user is uploading an image/photo/report |

**Recommendation for a rich chat UI:** use `/chat/blocks` (or `/chat/stream/blocks`) for text turns and `/chat/image` for uploads. Everything below documents each in detail; the **block guarantees + safety-response handling** under `/chat/blocks` are the most important part to read.

### `GET /health`

Cheap liveness probe. No upstream I/O, no auth.

```json
{ "status": "ok", "checks": {} }
```

Use this for: connectivity smoke tests, status pages, dashboard pings. Don't depend on it to mean "all upstreams are healthy" — for that, use `/healthz/ready`.

### `GET /healthz/ready`

Readiness probe. Verifies the backend has built its DI container and that Pinecone, Neo4j, and Redis respond. Slower (≤8 s worst case).

```json
{
  "status": "ok",            // or "degraded" / "starting"
  "checks": {
    "redis":   "ok",         // or "fallback" (in-memory) or "fail: <ExceptionClass>"
    "pinecone": "ok",
    "neo4j":   "ok"
  }
}
```

Surface this only in admin-style UIs. End users don't need to see Pinecone status.

### `GET /metrics`

In-process metrics snapshot. No auth.

```json
{
  "requests_total": 1280,
  "requests_inflight": 2,
  "errors_total": 4,
  "uptime_seconds": 5400,
  "latency_ms_p50": 1820,
  "latency_ms_p95": 4100,
  "pinecone_calls_total": 1240,
  "neo4j_calls_total": 980,
  "llm_tokens_total": 5_400_000
}
```

Useful for a status widget. For production observability use a proper APM.

### `POST /chat`

The non-streaming chat endpoint. Returns the full answer after the pipeline completes (typically 2–6 s for substantive clinical queries).

**Request**

```jsonc
{
  "query":      "I have chest pain that started 2 hours ago — what could be causing it?",
  "session_id": "patient-42-session-2025-05-19",   // optional; auto-uuid if omitted
  "user_id":    "patient-42"                       // optional; enables long-term episodic memory
}
```

Field rules:
- `query` — required, 1–4000 chars.
- `session_id` — optional. If omitted, the backend mints a UUID hex and returns it in the response. **Keep the same `session_id` across turns** for short-term memory (last ~10 turns, 2-hour TTL).
- `user_id` — optional. When present, the backend writes to and reads from a long-term episodic memory namespace dedicated to that user. Use a stable ID per real person (not per device, not per session).

**Response (200)**

```jsonc
{
  "answer": "Hey Aarav — that kind of chest tightness, especially given your history…",
  "session_id": "patient-42-session-2025-05-19",
  "request_id": "01J0X7K8R2M3Z4ABCDEF1234XY",       // also echoed in the X-Request-ID header
  // gatekeeper LLM output — ONLY present when the backend runs with
  // EXPOSE_DIAGNOSTICS=true (off in production). Treat as optional/absent.
  "analysis": {
    "intent": "symptom_query",
    "risk_level": "low",
    "final_action": "retrieve"
  },
  "routing": {
    "mode": "HYBRID_RAG",                             // or MEMORY_FIRST, NO_RETRIEVAL
    "intent": "symptom_query",
    "query_type": "symptom_query",
    "vector_top_k": 15,
    "graph_hops": 1
  },
  "timing_ms": {
    "session_load": 12,
    "analyze": 280,
    "vector_retrieve": 410,
    "graph_retrieve": 95,
    "episodic_context": 220,
    "llm": 1840,
    "session_save": 18,
    "total": 2875
  },
  "followup_questions": ["How long have you had the chronic asthma?"],
  "show_doctor_summary": false
}
```

The `followup_questions` array contains **at most one** item — there's a hard cap. Render it as a chip / suggested-reply button if you want; or ignore it.

`show_doctor_summary` is `true` once the consultation has reached a concluded answer (and stays `true` after) — reveal the "Show this to your doctor" button that calls [`POST /chat/soap`](#post-chatsoap--doctor-facing-soap-note). On the block endpoints this same signal arrives as the trailing `answer_state` block instead.

**Safety short-circuits (prose paths).** For a physical emergency or a mental-health crisis, `answer` is a fixed, safety-reviewed message (not a generated one) — an emergency instruction, or an empathetic mental-health message with crisis helplines (Tele-MANAS 14416, KIRAN 1800-599-0019, 112). It's still plain text, so nothing special is required, but you SHOULD linkify phone numbers (`tel:`) and present a crisis message calmly, not as an error. If you want these as structured blocks instead (to style the crisis differently), use `/chat/blocks` — see its "Special canned responses" table.

### `POST /chat/stream`

Same request body as `/chat`, but the response is `text/event-stream` (Server-Sent Events). First token typically lands in **1–2 s**; total wall time is similar to `/chat`. Use this for live "typing" UX.

See [SSE streaming — full protocol](#sse-streaming--full-protocol) for the wire format and a robust parser.

### `POST /chat/blocks`

Same request body as `/chat`, but the response is **NDJSON** (`application/x-ndjson`): a stream of typed UI blocks, **one JSON object per line**, emitted as generated. Use this when you want to render structured UI (cards, condition lists, warnings, follow-up chips) incrementally instead of free-text. `/chat` (JSON) and `/chat/stream` (SSE) are unchanged — this is an additive endpoint.

Read the body stream, split on `\n`, `JSON.parse` each complete line, and render by `.type` as it arrives — no text parsing, no waiting for the full response. There is **no** `data:` prefix and **no** `[DONE]` sentinel; the stream ends when the body closes.

Each line is `{"type": ..., "data": {...}}`. Block types:

| `type` | `data` shape |
|---|---|
| `summary` | `{ text }` |
| `key_points` | `{ points: string[] }` |
| `bullet_list` | `{ title: string\|null, items: string[] }` |
| `follow_up_questions` | `{ questions: string[] }` |
| `warning` | `{ text, severity: "info"\|"caution"\|"critical" }` |
| `next_steps` | `{ steps: string[] }` |
| `condition_list` | `{ conditions: [{ name, likelihood: string\|null, description: string\|null }] }` |
| `decision` | `{ verdict: "yes"\|"no"\|"possibly"\|"seek_urgent_care"\|"insufficient_information", rationale }` |
| `otc_medications` | `{ medications: [{ name, purpose, dosage: string\|null, caution: string\|null }] }` |
| `lab_tests` | `{ tests: [{ name, reason, urgency: "routine"\|"soon"\|"urgent"\|null }] }` |
| `answer_state` | `{ show_doctor_summary: boolean }` (control block — always last, don't render) |

Field notes:
- `warning.severity` is one of `"info" | "caution" | "critical"` — use it to pick styling (subtle / amber / prominent). No other values occur.
- `condition_list[].likelihood`, when present, is human text like `"most likely" | "possible" | "less likely"` — render as a label/badge; it may be `null`.
- `decision.verdict` is one of exactly five values — render as a colored verdict pill/banner. On a decision-type turn this block leads (no `summary`); the rest of the turn (`key_points` / `warning` / `next_steps`) supports it. A `seek_urgent_care` verdict always comes with a `critical` `warning`.
- `otc_medications` appears **only on a concluded answer** (the final assessment or a decision turn), always as the **last** block, and only when safe self-care applies — you won't get it mid-consultation or on educational/emergency turns. `name` + `purpose` are always present; `dosage` and `caution` may be `null`. These are OTC (over-the-counter) suggestions only, never prescriptions.
- `lab_tests` appears **only on a concluded answer** (final assessment or a decision turn), when investigations would confirm/narrow the diagnosis — not mid-consultation, not on educational/emergency turns. It sits near the end, just before `otc_medications`. `name` + `reason` are always present; `urgency` may be `null`. These are suggested investigations to discuss with a doctor/lab, **not orders**.
- `answer_state` is a **control block, not content** — it's emitted as the final block of every turn and carries `show_doctor_summary`. Don't render it; use it to toggle the "Show this to your doctor" affordance. The flag is **sticky** (once `true`, stays `true` for the session).
- `follow_up_questions.questions` never has more than one item (see guarantees below).
- Blocks arrive in a sensible reading order (e.g. `summary` first, `next_steps`/`follow_up_questions` last). Render them in the order received; don't reorder by type.

Example stream (two lines):

```
{"type":"summary","data":{"text":"Night-time cough may have several causes."}}
{"type":"follow_up_questions","data":{"questions":["Do you experience wheezing?","Do you have heartburn?"]}}
```

Minimal browser parser:

```js
const r = await fetch("/chat/blocks", { method: "POST", headers, body: JSON.stringify({ query, session_id }) });
const reader = r.body.getReader();
const dec = new TextDecoder();
let buf = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  let nl;
  while ((nl = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (line) renderBlock(JSON.parse(line));   // switch on block.type
  }
}
if (buf.trim()) renderBlock(JSON.parse(buf));   // flush trailing line (no final \n)
```

**Guarantees you can rely on (do NOT re-implement defensively):**

- **Every line is a schema-valid block.** Malformed model output is repaired or dropped server-side. You never receive a partial object, a wrong-typed field, or a raw string — just `JSON.parse` each line and switch on `.type`.
- **You always get at least one block.** If generation fails entirely, the server emits a fallback `summary` — the stream is never empty.
- **At most one `follow_up_questions` block per turn, with at most one question inside.** Render it as a single suggested-reply chip; you'll never get a wall of questions.
- **On a closing/assessment turn, no `follow_up_questions` is sent** — the assistant has converged to an answer, so show the assessment/next-steps and let the user type freely.

**Special "canned" responses (safety short-circuits).** Some turns bypass the model and stream a fixed, safety-reviewed block set. You detect them by block shape + `warning.severity`, and you MUST render them prominently:

| Situation | Blocks (in order) | How to render |
|---|---|---|
| **Physical emergency** | `warning` (`severity:"critical"`) → `next_steps` | Prominent critical banner; make emergency numbers (112 / 108) tappable `tel:` links. |
| **Mental-health crisis** (self-harm / suicidal ideation) | `summary` (empathetic) → `warning` (`severity:"critical"`) → `next_steps` (crisis helplines: Tele-MANAS **14416**, KIRAN **1800-599-0019**, plus **112**) | Render **calmly and supportively — NOT an alarming red error**. Lead with the summary, keep helpline numbers tappable. |
| **Refusal / out-of-scope** | single `summary` | Plain message. |

None of these contain `follow_up_questions`. Rule of thumb: **any `warning` with `severity:"critical"` means show it front-and-center and make the numbers in the following `next_steps` tappable.** Treat `severity` as your styling signal — `info` (subtle), `caution` (amber), `critical` (prominent).

### `POST /chat/stream/blocks`

The **same typed blocks as `/chat/blocks`**, but **SSE-framed** (`text/event-stream`) for `EventSource`/proxy-friendly clients that can't easily read a raw `fetch` body. Each block is one SSE event whose `data:` payload is the identical block JSON (`{"type": ..., "data": {...}}`), terminated by a final `data: [DONE]` event. Pick this over `/chat/blocks` when you want to consume blocks with `EventSource` or behind infra that prefers SSE.

```
data: {"type":"summary","data":{"text":"Night-time cough may have several causes."}}

data: {"type":"follow_up_questions","data":{"questions":["Do you experience wheezing?"]}}

data: [DONE]
```

Parse exactly like the prose SSE stream below (split on `\n\n`, strip `data: `, stop on `[DONE]`), except each payload is a block object — switch on `block.type` and render. Same validation/terminal/canned guarantees as `/chat/blocks`.

### `POST /chat/soap` — doctor-facing SOAP note

Generates a fresh **SOAP note** (Subjective / Objective / Assessment / Plan) for an existing session — the "Show this to your doctor" export. Wire this button to appear when the latest `answer_state` block had `show_doctor_summary: true` (or the `/chat` JSON response had `show_doctor_summary: true`).

- **Regenerated on demand**, every call, from the **latest full conversation** — so tapping it again after more chat gives an updated note.
- **Grounded strictly in the conversation.** Nothing is fabricated; missing clinically-relevant info is listed in `unavailable` (e.g. "No vital signs recorded"). `objective` will often say measurements weren't reported — that's expected, not an error.

Request (JSON):
```json
{ "session_id": "patient_42", "user_id": "optional" }
```

Response (JSON):
```json
{
  "subjective": "Patient reports a fever for 5 days, currently 102F. Paracetamol gives partial relief. Denies cough or sore throat.",
  "objective": "Temperature 102F (patient-reported). No other objective measurements or examination findings were reported.",
  "assessment": "Persistent fever for 5 days without localizing symptoms; warrants prompt in-person evaluation. No confirmed diagnosis.",
  "plan": "In-person review within 24–48h; continue paracetamol per label; hydrate; monitor for red flags (severe headache, rash, breathlessness, confusion).",
  "unavailable": ["No physical examination performed", "No allergies stated", "No past medical history stated"],
  "session_id": "patient_42",
  "request_id": "01J...",
  "generated_at": "2026-07-16T09:12:44.512000+00:00"
}
```

Render as a clean, printable/exportable document — one heading per section (S/O/A/P). Show `unavailable` as a small "Not documented in this conversation" list so the doctor knows what's missing. Add a "Regenerate" action that re-calls the endpoint. `404` means there's no conversation for that `session_id` yet.

### `POST /chat/image`

Upload an image **with** an optional question and get a normal chat answer back, plus metadata about the upload. Unlike every other chat endpoint this one is **`multipart/form-data`**, not JSON — so the browser sets the `Content-Type` boundary for you (don't set it by hand).

Use it for: a patient photographing a rash/wound, or a photo/scan of a lab or radiology report. The backend decides what the image *is* and handles it accordingly (see "how routing works" below).

**Request — `multipart/form-data`**

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | ✅ | `image/png` or `image/jpeg`. Max 10 MB. Content is sniffed — a `.png` that isn't really a PNG is rejected. |
| `query` | text | optional | The question about the image. If omitted, the backend uses a sensible default ("What can you tell me about it?" / "Please review this document"). |
| `session_id` | text | optional | Same semantics as `/chat`. Auto-uuid if omitted; returned in the response. |
| `user_id` | text | optional | Same semantics as `/chat`. Enables long-term episodic memory. |

How text + image travel together: the `query` text and the `image` bytes are sent in the **same multipart request**. Server-side they're folded into **one** pipeline turn — the image (for photos) and the question reach the model together, so the answer is about *both*. You do **not** call `/chat` separately; this endpoint replaces it for the turn that carries an image.

**How routing works (why you don't need to tell us the image type)**

The backend classifies the upload first, then routes:

- **Clinical or general photo** → sent to the multimodal model alongside your question. The answer reasons over the actual pixels.
- **Lab report / radiology report / prescription / other document** → sent through **document extraction** (structured text/values are pulled out), *not* visual interpretation. The answer is grounded on the extracted findings.

You get the decision back in `media.category` and `media.route` so the UI can label it ("📷 Photo" vs "📄 Lab report"). Raw image bytes are **never** stored in memory — only metadata (type, caption, extracted findings, a storage URI).

**Response (200)** — a normal `ChatResponse` plus a `media` object:

```jsonc
{
  "answer": "Thanks for the photo — the area on your forearm looks like…",
  "session_id": "patient-42-session",
  "request_id": "01J0X7…",
  "routing": { "mode": "HYBRID_RAG", "intent": "symptom_query", "query_type": "symptom_query", "vector_top_k": 15, "graph_hops": 1 },
  "timing_ms": { "total": 4120 },
  "followup_questions": ["How long has the rash been there?"],
  "media": {
    "category": "clinical_photo",        // clinical_photo | general_photo | lab_report | radiology_report | document | other_medical_document | unknown
    "route": "multimodal_llm",           // multimodal_llm | document_extraction
    "mime_type": "image/png",
    "size_bytes": 482113,
    "filename": "rash.png",
    "storage_uri": "file:///srv/uploads/3f9a….png",   // reference only; not a public URL
    "caption": "Red, raised patch on the left forearm…",  // photo: a description; document: a one-line summary
    "extracted_facts": []                // document route: ["Hemoglobin 9.1 g/dL (low)", …]; photo route: []
  }
}
```

**Errors:** `400 INVALID_INPUT` for a bad upload (wrong MIME, > 10 MB, empty, or content that doesn't match the declared type). `503` if image upload is disabled on the backend (`MEDIA_UPLOAD_ENABLED=false`).

**Browser example**

```ts
const fd = new FormData();
fd.append("image", fileInput.files[0]);        // a File from <input type="file">
fd.append("query", "is this rash something to worry about?");
fd.append("session_id", sessionId);

const r = await fetch(`${BASE}/chat/image`, {
  method: "POST",
  headers: { ...(apiKey && { "X-API-Key": apiKey }) },  // do NOT set Content-Type — the browser adds the boundary
  body: fd,
});
const res = await r.json();   // { answer, media, … }
```

> This is a **non-streaming** endpoint — you get the full answer once the pipeline (classification + extraction + answer) completes, typically 3–7 s. The streaming endpoints (`/chat/stream`, `/chat/blocks`) are text-only today and do not accept images.

### `/episodic/*`

The long-term episodic memory subsystem. Mounted on the same service with the same auth and CORS. Use this if you need to read/write episodes outside of a chat turn (e.g. an "edit memory" admin UI).

| Method + path | Purpose |
|---|---|
| `POST /episodic/extract` | Extract candidate episodes from an utterance without storing them. |
| `POST /episodic/store` | Persist a candidate to Pinecone. |
| `POST /episodic/retrieve` | Query a user's episodic memory by similarity. |
| `POST /episodic/context` | Get compressed episodic context for a query (same call used inside `/chat`). |
| `POST /episodic/clarify` | Generate a single clarifying question grounded in past episodes. |
| `POST /episodic/contradictions` | Find contradictions between a new episode and existing ones. |

Full schemas + payloads live in [docs/EPISODIC_API.md](EPISODIC_API.md). The chat endpoints exercise these implicitly — you usually don't need direct calls unless you're building memory-management UI.

---

## Error envelope

Every handled error uses one shape:

```jsonc
{
  "code":       "UPSTREAM_UNAVAILABLE",   // stable machine-readable code (see table)
  "message":    "Pinecone request failed",
  "request_id": "01J0X7K8R2M3Z4ABCDEF1234XY",
  "details":    null                       // optional, free-form
}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `INVALID_INPUT` | Body failed Pydantic validation, query too long/short. |
| 401 | `UNAUTHORIZED` | `API_KEY` is set on the backend and the header is missing/wrong. |
| 429 | `RATE_LIMITED` | Upstream (Gemini quota) or in-app limit hit. Retry after a back-off. |
| 502/503 | `UPSTREAM_UNAVAILABLE` | Pinecone, Neo4j, Redis, or Gemini failed. The `request_id` lets you correlate to backend logs. |
| 500 | `INTERNAL_ERROR` | Catch-all. Bug. Quote the `request_id` when reporting. |

Frontend recipe:

```ts
async function chat(body: ChatRequest, apiKey?: string): Promise<ChatResponse> {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(apiKey && { "X-API-Key": apiKey }) },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err: ErrorResponse = await r.json().catch(() => ({
      code: "INTERNAL_ERROR", message: r.statusText, request_id: r.headers.get("x-request-id") ?? "",
    }));
    throw new EnerveraError(err.code, err.message, err.request_id, r.status);
  }
  return r.json();
}
```

For retries: `UPSTREAM_UNAVAILABLE` and `RATE_LIMITED` are safe to retry with exponential back-off (start at 1 s, cap at 8 s, give up after 3 tries). `INVALID_INPUT`, `UNAUTHORIZED`, and `INTERNAL_ERROR` should not auto-retry — they indicate a bug or a user-action problem.

---

## Session and user_id semantics

These two IDs do different things — don't conflate them.

| | `session_id` | `user_id` |
|---|---|---|
| Required? | No (auto-uuid if missing) | No |
| Lifetime | 2 hours (Redis TTL) | Permanent (Pinecone namespace) |
| Stores | Last ~10 turns + rolling summary + extracted clinical state | Compressed long-term clinical episodes |
| Resets when | Idle for 2 h, or you change the ID | Never, unless you delete the namespace |
| Shareable across devices? | No — one per active conversation | Yes — pin to the real human |

Practical mapping for a typical app:

- **Anonymous chat widget** → leave both unset on first message; persist the returned `session_id` in `sessionStorage` and reuse it for the rest of the visit.
- **Logged-in patient app** → `session_id` = a per-conversation UUID; `user_id` = your account ID. Use the same `user_id` forever for that patient so episodic memory accumulates.
- **Multi-user kiosk** → `session_id` per session; do not send `user_id` unless the kiosk authenticates the patient. Sending the wrong `user_id` writes to the wrong long-term memory.

If you send `user_id`, the backend asynchronously ingests the user's turn into long-term memory after the response (fire-and-forget — does not block the response). Reads happen synchronously inside the chat pipeline and add ~150–400 ms.

---

## SSE streaming — full protocol

`POST /chat/stream` returns `text/event-stream`. The frame format follows the SSE spec: each event is a block of lines, frames are separated by a blank line (`\n\n`), and each line in a frame is prefixed `data: ` followed by JSON (or the literal `[DONE]` terminator).

### Event types

The `data:` payload (when not `[DONE]`) is a JSON object with a `type` discriminator:

| `type` | Payload fields | When |
|---|---|---|
| `meta` | `data.routing`, `data.timing_ms` | Once, **before** any chunks — tells the UI how the pipeline routed. |
| `chunk` | `data: string` | The next piece of the answer. Concatenate `chunk.data` values in order to rebuild the full answer. |
| `done` | `timing_ms`, optionally `routing`, `analysis`, `followup_questions`, `session_id` | Always emitted as the second-to-last event. Contains the final pipeline timing. |
| `error` | `error: { code, message }` | Terminal failure. The stream ends after this. |

After the `done` (or `error`) event, the server sends one more frame: `data: [DONE]\n\n` — that's the cue to stop reading.

### Worked example wire frames

```
data: {"type":"meta","data":{"routing":{"mode":"HYBRID_RAG","intent":"symptom_query","query_type":"symptom_query"},"timing_ms":{"session_load":10,"analyze":260,"vector_retrieve":410}}}

data: {"type":"chunk","data":"Hey "}

data: {"type":"chunk","data":"Aarav — "}

data: {"type":"chunk","data":"that kind of chest tightness…"}

data: {"type":"done","timing_ms":{"session_load":10,"analyze":260,"vector_retrieve":410,"llm":1840,"total":2860},"followup_questions":["How long?"],"session_id":"patient-42-session"}

data: [DONE]
```

### Robust JS parser

Browsers don't have a `POST`-capable `EventSource`. Use `fetch` + `ReadableStream` + manual frame buffering. The parser below handles partial chunks (a single SSE frame can be split across multiple `read()` calls).

```ts
type StreamEvent =
  | { type: "meta";  data: { routing?: any; timing_ms?: Record<string, number> } }
  | { type: "chunk"; data: string }
  | { type: "done";  timing_ms?: Record<string, number>; routing?: any; analysis?: any; followup_questions?: string[]; session_id?: string }
  | { type: "error"; error: { code: string; message: string } };

export async function* streamChat(
  base: string,
  body: { query: string; session_id?: string; user_id?: string },
  init: { apiKey?: string; signal?: AbortSignal } = {},
): AsyncGenerator<StreamEvent, void, void> {
  const r = await fetch(`${base}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(init.apiKey && { "X-API-Key": init.apiKey }),
    },
    body: JSON.stringify(body),
    signal: init.signal,
  });
  if (!r.ok || !r.body) {
    const txt = await r.text().catch(() => "");
    throw new Error(`stream failed: HTTP ${r.status} ${txt}`);
  }

  const reader = r.body.getReader();
  const dec = new TextDecoder("utf-8");
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });

    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);

      const payload = frame
        .split("\n")
        .filter(l => l.startsWith("data:"))
        .map(l => l.slice(5).trimStart())
        .join("\n");
      if (!payload) continue;
      if (payload === "[DONE]") return;

      try {
        yield JSON.parse(payload) as StreamEvent;
      } catch {
        // ignore malformed frames — server-side guarantees JSON, but a corp
        // proxy could inject a comment line. Defensive parsing.
      }
    }
  }
}
```

Usage:

```ts
let answer = "";
for await (const ev of streamChat(BASE, { query: "…", session_id }, { apiKey })) {
  switch (ev.type) {
    case "chunk":  answer += ev.data; updateUI(answer); break;
    case "meta":   showRoutingBadge(ev.data.routing); break;
    case "done":   showTiming(ev.timing_ms); break;
    case "error":  throw new Error(`${ev.error.code}: ${ev.error.message}`);
  }
}
```

### Cancellation

Pass an `AbortController.signal`. Aborting closes the underlying connection; the backend cancels its in-flight Gemini stream within ~500 ms.

```ts
const ac = new AbortController();
cancelButton.onclick = () => ac.abort();
for await (const ev of streamChat(BASE, body, { signal: ac.signal })) { /* … */ }
```

### Buffering and proxies

The backend sets `Cache-Control: no-cache` and `X-Accel-Buffering: no` on the stream response. If you put a CDN or reverse proxy in front of the API, ensure it does **not** buffer `text/event-stream`. Render's frontproxy is already correctly configured.

---

## Reference clients

### Fetch / vanilla JS

```js
async function chat(query, sessionId, userId, apiKey, base) {
  const r = await fetch(`${base}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(apiKey && { "X-API-Key": apiKey }) },
    body: JSON.stringify({ query, session_id: sessionId, ...(userId && { user_id: userId }) }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}
```

### TypeScript types

```ts
export interface ChatRequest {
  query: string;             // 1..4000 chars
  session_id?: string;       // auto-uuid if omitted
  user_id?: string;          // enables long-term episodic memory
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  request_id: string;
  analysis?: {
    intent?: string;
    risk_level?: "none" | "low" | "medium" | "high" | "critical";
    final_action?: "retrieve" | "refuse" | "emergency_redirect" | string;
    [k: string]: unknown;
  };
  routing: {
    mode: "HYBRID_RAG" | "MEMORY_FIRST" | "NO_RETRIEVAL";
    intent: string;
    query_type: string;
    vector_top_k: number;
    graph_hops: number;
  };
  timing_ms: Record<string, number>;
  followup_questions: string[]; // length 0 or 1
}

// POST /chat/image — multipart in (FormData), JSON out.
export type ImageCategory =
  | "clinical_photo" | "general_photo" | "lab_report"
  | "radiology_report" | "document" | "other_medical_document" | "unknown";

export interface MediaInfo {
  category: ImageCategory;
  route: "multimodal_llm" | "document_extraction";
  mime_type: string;
  size_bytes: number;
  filename?: string | null;
  storage_uri?: string | null;   // backend reference, not a public URL
  caption?: string | null;
  extracted_facts: string[];     // populated on the document route
}

export interface ImageChatResponse extends ChatResponse {
  media: MediaInfo;
}

export interface ErrorResponse {
  code: "INVALID_INPUT" | "UNAUTHORIZED" | "RATE_LIMITED" | "UPSTREAM_UNAVAILABLE" | "INTERNAL_ERROR" | string;
  message: string;
  request_id: string;
  details?: unknown;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "starting";
  checks?: Record<string, string>;
}

export interface MetricsSnapshot {
  requests_total: number;
  requests_inflight: number;
  errors_total: number;
  uptime_seconds: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  pinecone_calls_total: number;
  neo4j_calls_total: number;
  llm_tokens_total: number;
}
```

### React hook

```tsx
import { useCallback, useRef, useState } from "react";

export function useEnerveraStream(base: string, apiKey?: string) {
  const [answer, setAnswer] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const send = useCallback(async (query: string, sessionId: string, userId?: string) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setAnswer(""); setError(null); setIsStreaming(true);
    try {
      let acc = "";
      for await (const ev of streamChat(base, { query, session_id: sessionId, user_id: userId }, { apiKey, signal: ac.signal })) {
        if (ev.type === "chunk") { acc += ev.data; setAnswer(acc); }
        else if (ev.type === "error") throw new Error(`${ev.error.code}: ${ev.error.message}`);
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      setIsStreaming(false);
    }
  }, [base, apiKey]);

  const cancel = useCallback(() => abortRef.current?.abort(), []);
  return { answer, isStreaming, error, send, cancel };
}
```

### curl

```bash
# Non-streaming
curl -X POST https://enervera-api.onrender.com/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ENERVERA_API_KEY" \
  -d '{"query":"What is hypertension?","session_id":"demo-1"}'

# Streaming
curl -N -X POST https://enervera-api.onrender.com/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ENERVERA_API_KEY" \
  -d '{"query":"What is hypertension?","session_id":"demo-1"}'

# Image upload (multipart — let curl set the Content-Type/boundary via -F)
curl -X POST https://enervera-api.onrender.com/chat/image \
  -H "X-API-Key: $ENERVERA_API_KEY" \
  -F "image=@rash.png;type=image/png" \
  -F "query=is this rash serious?" \
  -F "session_id=demo-1"

# Episodic context
curl -X POST https://enervera-api.onrender.com/episodic/context \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ENERVERA_API_KEY" \
  -d '{"user_id":"patient-42","query_text":"chest pain"}'
```

---

## Operational notes

### Cold starts

Render's Starter plan suspends idle containers. The first request after sleep pays the boot cost: typically **10–15 s** because Pinecone lazy-creates its index client and Neo4j opens a Bolt connection. The lifespan pre-warms both, so subsequent requests are normal-speed.

Mitigations:
- Show a `"Waking up the assistant…"` state if the first request takes > 3 s.
- Optional 5-minute keep-warm ping from your frontend's monitoring or a cron job (`GET /health` is the right target).
- Upgrade the backend to a paid plan in production.

### Timeouts

- Non-stream `/chat`: target ≤ 8 s for a typical query, P95 ≤ 15 s. Use a 30 s fetch timeout.
- Stream `/chat/stream`: first token within 2 s on a warm container. Total wall time similar to `/chat`. Don't time out the *fetch* — time out individual reads (no chunk in 20 s → abort).

### Rate limiting

There's no in-app rate limiter today. The bottleneck is the Gemini API quota — when it's exhausted you'll see `RATE_LIMITED`. Frontends should:
- Disable the send button while a turn is in flight.
- Back off (1 s → 2 s → 4 s) on `RATE_LIMITED`, max 3 tries, then surface "service busy, try in a minute".

### Logs and correlation

Every response carries `X-Request-ID` (also in the JSON `request_id` field). Quote it in bug reports — backend logs are indexed on this ID.

### What NOT to call

- `/docs`, `/redoc`, `/openapi.json` are for humans / Swagger tools, not for programmatic consumption.
- `/metrics` is OK to poll but treat the numbers as advisory. Single-process, in-memory, restarts to zero on redeploy.

---

## Versioning and compatibility

- The current API is **`0.x`**. Breaking changes are possible until `1.0`. Pin to a deployed commit SHA or watch the changelog if stability matters.
- The response **schema** is documented above. Additive changes (new fields, new event types in the SSE stream) are non-breaking — your parser should ignore unknown event `type` values and unknown JSON keys.
- The `request_id`, `session_id`, and error `code` strings are **stable** — those will not change shape across minor versions.

For changes that affect this contract, open an issue on the backend repo and we'll bump the version.
