# Aura — Clinical Intelligence Assistant

Ask a clinical question, get an answer assembled only from indexed guidelines and
protocols, with every claim traceable to the page it came from. If nothing in the
corpus covers the question, Aura says so instead of guessing.

Runs entirely on free tiers. Portfolio/demo build — see [Honest status](#honest-status).

---

## Architecture at a glance

```
                    ┌──────────────────────────────────────┐
                    │              Clinician               │
                    │        (browser, desktop/phone)      │
                    └───────────────────┬──────────────────┘
                                        │ HTTPS
                                        ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  FRONTEND — Next.js 16 (App Router, client-rendered)         │
      │  Vercel Hobby  ·  static shell + fetch/SSE to the API        │
      └───────────────────────────┬─────────────────────────────────┘
                                  │  JSON  +  text/event-stream
                                  ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  BACKEND — FastAPI (single uvicorn process)                  │
      │  Render Free  ·  retrieval, generation, ingest, auth         │
      └───────┬──────────────────────────────────┬──────────────────┘
              │ SQL + pgvector                   │ HTTPS
              ▼                                  ▼
   ┌────────────────────────┐       ┌──────────────────────────────┐
   │ Postgres 16 + pgvector │       │  LLM / embedding provider    │
   │ Supabase Free (500MB)  │       │  Gemini · Groq · mock        │
   │ threads · messages ·   │       │  embeddings are Gemini-only  │
   │ chunks(embedding)      │       └──────────────────────────────┘
   └────────────────────────┘
```

One question, end to end:

```
 POST /api/v1/chat/stream
   → validate  → resolve thread  → expand short follow-ups from history
   → retrieve  (pgvector cosine, else TF-IDF over the corpus)
   → build_context()   one numbering pass → prompt markers AND citation list
   ◀ event: meta       citations flushed BEFORE generation starts
   → generate          streamed off the event loop
   ◀ event: token …
   → validate_citations()   every [n] must resolve to a real source
   ◀ event: done       {full_text, citations, citation_check}
```

Full component breakdown and the reasoning behind each decision:
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Run it locally

Two terminals. No database or API key required — the mock provider replays
retrieved source text verbatim with real citations, which exercises the whole
interface.

```bash
# Terminal 1 — backend on :8000
cd backend
uv sync                                    # first time only
LLM_PROVIDER=mock uv run uvicorn app.main:app --port 8000
```

```bash
# Terminal 2 — frontend on :3000
cd frontend
bun install                                # first time only
bun run dev
```

Open **http://localhost:3000**.

<details>
<summary>Using a real LLM and database</summary>

```bash
cp .env.example .env      # then fill in the values below
```

| Setting | Effect |
|---|---|
| `LLM_PROVIDER` | `mock` (no key needed), `gemini`, or `groq` |
| `GEMINI_API_KEY` | Required for **embeddings**, even when `LLM_PROVIDER=groq` |
| `GROQ_API_KEY` | Generation only |
| `DATABASE_URL` | Postgres with the `vector` extension enabled |

A real environment variable **overrides `.env`**. If you export `GEMINI_API_KEY`
in your shell, editing `.env` will appear to do nothing — `unset` it first.

With Docker installed, `docker compose up --build` starts all three (Next.js,
FastAPI, Postgres+pgvector) in one command.
</details>

### Check it is actually working

```bash
curl -s http://localhost:8000/health | jq
```

| Field | Meaning |
|---|---|
| `status` | `ok` = connected to Postgres. `degraded` = **running from memory, nothing persists** |
| `storage_mode` | `postgres` or `in-memory (ephemeral)` |
| `retrieval_mode` | `pgvector` = semantic search. `tfidf` = keyword matching (no embeddings available) |
| `threshold` | The score floor actually in force, chosen by mode |
| `db_error` | Why the database is unreachable, when it is |

Running with no database is expected locally and the app will work — it just
tells you so rather than pretending. Interactive API docs: http://localhost:8000/docs

### Things to try

- *"First-line therapy for hypertension with CKD?"* — grounded answer; hover a
  `[1]` marker and its evidence card highlights with it
- *"what do you know about hair problems"* — refuses, with no citations
- Upload a PDF from the rail, then ask about its contents
- Narrow the window below ~768px — the rail becomes a drawer you can drag and flick

---

## Deploy

Free tier, three accounts, no card: Supabase → Render → Vercel, in that order.
Step-by-step with the gotchas: **[DEPLOY.md](DEPLOY.md)**.

The one check that matters after deploying is `/health`: if `storage_mode` says
`in-memory (ephemeral)`, the database is not connected and nothing is being saved
— the API will keep serving in that state, so nothing else will tell you.

---

## Endpoints

| | |
|---|---|
| `POST /api/v1/chat/stream` | SSE: `meta` → `token`… → `done` → `heartbeat` |
| `POST /api/v1/documents/upload` | PDF ingest; returns a `job_id` to poll |
| `GET /api/v1/documents/jobs/{id}` | `completed` \| `partial` \| `failed`, with real counts |
| `GET /api/v1/documents` | Indexed documents |
| `GET /api/v1/chunks/{id}` | One source chunk |
| `POST /api/v1/threads` · `GET /api/v1/threads/{id}/messages` | Conversations |
| `GET /health` · `GET /ready` | State and readiness (`/ready` returns 503 when not) |

---

## Tests

294 across both stacks — unit, integration, E2E, real-time SSE, UI, browser and
performance. Layout and the non-obvious traps: **[TESTING.md](TESTING.md)**.

```bash
cd backend  && uv run pytest -q       # 196
cd frontend && bun run test           # 61  (unit + component)
cd frontend && bun run test:e2e       # 37  (real browser; bunx playwright install chromium)
k6 run backend/tests/load_sse_k6.js   # load; http_req_waiting is the TTFT proxy
```

### Toolchain

| | | |
|---|---|---|
| Runtime + package manager | Bun | `bun install`, `bun run`, `bunx` — no npm/npx anywhere |
| Framework | Next.js 16 + React 18 | Turbopack is the default bundler |
| Typecheck | TypeScript 7 (`bun run typecheck`) | The native compiler — what shipped as `tsgo` |
| Lint + format | Biome (`bun run lint`) | Replaces ESLint and Prettier |

Measured on this repo: build 5.8s → 1.5s, typecheck 0.75s → 0.24s, lint 30ms,
cold install 12.5s for 317 packages.

`bun.lock` is the only lockfile — `package-lock.json` was removed. Two lockfiles
in one tree is also what made Turbopack unable to infer the project root.

CI runs the backend suite twice — against real Postgres+pgvector, and with no
database at all — plus a Playwright project that drives the **real** backend with
no stubs, so a frontend/backend contract mismatch fails there rather than in
production.

---

## Honest status

A portfolio build, not a production clinical system.

- **Auth is off by default.** Every visitor is anonymous and shares one dataset.
  Set `ENABLE_AUTH=true` with `API_KEYS=...`, plus `NEXT_PUBLIC_API_KEY` on the
  frontend. That key ships in the public bundle, so it gates a demo, not secrets.
- **Retrieval falls back to keyword matching** unless a Gemini key with quota is
  configured *and* documents were ingested with embeddings. `/health` reports
  which mode is live — trust it over anything written here.
- **No migrations.** Schema comes from `create_all`; changing a model on an
  existing database needs manual work.
- **Single instance only.** Rate limiting, ingest jobs and the embedding cache
  live in process memory.
- **No test exercises a real LLM or embedding call.** Every provider is mocked, so
  those two integration points are the least proven in the system.
- **Grounding is checked, not guaranteed.** Every citation marker is validated
  against the retrieved sources, and untrusted document text is delimited in the
  prompt — but a model can still be talked around.

Free-tier behaviour: Render sleeps after 15 minutes idle, so the first request
takes ~30s (the UI says so rather than appearing hung); Supabase pauses after
about a week of inactivity and wakes on the next request.

---

## Not medical advice

A reference tool for exploring indexed documents. Not a diagnosis, and not a
substitute for clinical judgement.
