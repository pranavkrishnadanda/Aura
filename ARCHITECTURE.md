# Architecture (L1)

A level-1 view: the components, what flows between them, and the decisions worth
knowing about. For deployment steps see [DEPLOY.md](DEPLOY.md); for the test
layout see [TESTING.md](TESTING.md).

---

## 1. System context

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

**Why the split.** Static hosting cannot run Python or hold a database, so the UI
and the API are deployed separately and talk over HTTPS. That is also why CORS is
a real concern here rather than an afterthought.

---

## 2. Request path — asking a question

```
 Browser                    FastAPI                        Stores
 ───────                    ───────                        ──────
 POST /api/v1/chat/stream
   {message, thread_id}
        │
        ├──────────────────▶ validate length, bounds
        │                    resolve thread + ownership ──▶ threads
        │                            │
        │                    expand_followup()  ──────────▶ messages
        │                    (short back-reference only)
        │                            │
        │                    retrieve_async()   ──────────▶ chunks
        │                      ├ pgvector cosine  (needs embeddings)
        │                      └ TF-IDF fallback  (whole corpus, in RAM)
        │                            │
        │                    build_context()
        │                      one numbering pass produces BOTH the
        │                      markers the model sees and the citation
        │                      list the client renders
        │                            │
        ◀── event: meta ─────────────┤  citations flushed before generation
        │                            │
        │                    generate_answer() ───────────▶ LLM provider
        ◀── event: token ────────────┤  (streamed, off the event loop)
        ◀── event: token ────────────┤
        │                            │
        │                    validate_citations()
        │                      every [n] must resolve to a real source
        │                            │
        │                    persist assistant turn ──────▶ messages
        ◀── event: done ─────────────┤  {full_text, citations, citation_check}
        ◀── event: heartbeat ────────┘
```

**meta arrives first, before a single token.** The evidence panel renders while
the answer is still streaming, which is what makes time-to-first-byte the number
that matters rather than total response time.

**The numbering is produced once.** `build_context()` returns the prompt text and
the citation list together. They were computed independently and drifted whenever
a mid-ranked chunk fell below threshold: the model was told to cite `[3]` while
the client only held `[1]` and `[2]`, so the marker rendered as dead text.

**Retrieval failure fails before the stream opens** and returns 503. Once headers
are sent you can only truncate, and answering from a partial corpus would present
an outage as a normal result.

---

## 3. Ingest path — adding a document

```
 POST /api/v1/documents/upload  (multipart)
        │
        ├─ extension + %PDF- magic bytes
        ├─ read in 1MB chunks, abort past MAX_PDF_MB
        │     (a full read before the size check does not bound memory)
        │
        ├─ create_job()  ──▶ _jobs (in-process dict)
        └─ BackgroundTasks ──▶ ingest_pdf_sync()
                                   │
                                   ├─ PyMuPDF page text
                                   ├─ chunk_text()   ~600 words, 100 overlap
                                   ├─ _embed_texts() batches of 100 → Gemini
                                   │     a failed batch degrades only itself
                                   └─ upsert → chunks
                                         │
     GET /api/v1/documents/jobs/{id} ◀───┘
        completed | partial | failed   ← reports what was actually stored
```

The client polls the job; the upload response only confirms the file was
accepted. Reporting "completed" when no chunk persisted is what told users a
document was indexed when retrieval could never see it.

---

## 4. Components

### Backend (`backend/app/`)

| Module | Responsibility |
|---|---|
| `main.py` | HTTP surface: endpoints, CORS, rate limits, SSE assembly, health/readiness |
| `rag.py` | Retrieval, prompt construction, generation, citation validation, refusal |
| `db.py` | Storage: engine lifecycle, thread/message/chunk access, in-memory fallback |
| `ingest.py` | PDF → text → chunks → embeddings → storage, with job tracking |
| `auth.py` | API-key identity; fails closed when enabled without keys |
| `models.py` | SQLAlchemy schema, including the `Vector(EMBED_DIM)` column and ivfflat index |
| `config.py` | Settings, all overridable by environment |
| `schemas.py` | Request validation |

### Frontend (`frontend/`)

| Module | Responsibility |
|---|---|
| `components/Chat.tsx` | Conversation state, streaming lifecycle, layout |
| `components/AnswerProse.tsx` | Renders prose, links markers to evidence, provenance line |
| `components/CitationPanel.tsx` | One source, verbatim |
| `components/AdminUpload.tsx` | Upload plus job polling |
| `components/Drawer.tsx` | Gesture-driven mobile navigation |
| `lib/api.ts` | SSE parsing and the single-settle contract |
| `lib/motion.ts` | Momentum projection, boundary resistance, interruptible spring |

---

## 5. Decisions worth knowing

**Two retrieval modes, two thresholds.** Dense cosine and sparse TF-IDF scores are
not on the same scale, so one number cannot gate both. `RETRIEVAL_THRESHOLD`
(0.85) applies in pgvector mode, `TFIDF_THRESHOLD` (0.10) in the fallback.
`/health` reports `retrieval_mode` and the floor actually in force.

**pgvector requires three things at once:** a live database, a Gemini key with
quota, and at least one embedded chunk. Missing any one means TF-IDF, which
refits a vectoriser over the whole corpus per query — linear, and fine to roughly
5,000 pages.

**The database is optional at runtime, deliberately.** If Postgres is unreachable
the API keeps serving from in-memory dicts so a demo still runs. That is also a
trap: it means the app looks healthy with no database at all, which is exactly
what happened when the shipped driver did not match the URL scheme. So the
fallback announces itself — `/health` reports `degraded` with the cause, `/ready`
returns 503, and CI asserts a real connection before running the suite.

**Grounding is verified, not requested.** The system prompt asks for a citation on
every claim; `validate_citations()` then checks the output and reports the result
in the `done` event. A marker past the end of the source list is a fabricated
reference — worse than an uncited sentence, because it looks verifiable.

**Untrusted document text is delimited.** Retrieved chunks are wrapped in
`<context>` and the system prompt states their contents are reference material,
never instructions. This is mitigation, not a guarantee: instruction-following
models can still be talked around.

**Blocking work runs off the event loop.** Retrieval and the synchronous provider
SDKs are dispatched to threads, so concurrent SSE streams overlap instead of
serialising behind each other.

**State that lives in the process,** and therefore assumes a single instance:
the rate limiter, the ingest job table, and the query embedding cache. Fine on
one Render instance; all three need shared storage to scale horizontally.

---

## 6. Limits

| | |
|---|---|
| Auth | Off by default — every visitor is anonymous and shares one dataset |
| Migrations | None; schema comes from `create_all` |
| Horizontal scale | Single instance only (in-process state above) |
| Retrieval ceiling | TF-IDF is linear to ~5,000 pages; pgvector removes it |
| Prompt caching | Not implemented — the stable prefix is ~135 tokens against a ~1024 minimum |
| Provider coverage | No test exercises a real LLM or embedding call; every provider is mocked |
