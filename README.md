# Aura — Clinical Intelligence Streaming Assistant ($0 Stack)

Zero-cost RAG chat with streaming, citations, and hard refusal.

### Stack (Free Tier)
| Layer | Service | Why |
|---|---|---|
| Frontend (Next.js 14) | **Vercel Hobby** | Created Next.js; git-push auto-deploy |
| Backend (FastAPI + SSE) | **Render Free Web Service** | Free Linux container, GitHub auto-deploy. Sleeps after 15m → first request ~30s wake |
| DB (Postgres + pgvector) | **Supabase Free (500MB)** | Managed Postgres, `pgvector` pre-installed, encrypted at rest |
| LLM (streaming) | **Groq (Llama 3) or Gemini 1.5 Flash** | Groq ~500 tok/s, generous free tier. Gemini 15 req/min free. No credit card |

> GitHub Pages cannot run Python/Databases — this split is required.

### Deployment

Step-by-step free-tier deploy (Supabase + Render + Vercel), including the
verification commands and the gotchas: **[DEPLOY.md](DEPLOY.md)**.

### Deployment Workflow
1. **Local dev:** `docker compose up` — runs Next.js, FastAPI, local Postgres+pgvector
2. **Push to GitHub:** single repo `aura/` with `frontend/` and `backend/` folders
3. **Vercel:** Import repo → Root Directory `frontend` → env `NEXT_PUBLIC_API_URL=https://<your-render>.onrender.com`
4. **Render:** New Web Service → Root Directory `backend` → Build `pip install -r requirements.txt` → Start `uvicorn app.main:app --host 0.0.0.0 --port 10000`
5. **Supabase:** Create project → Enable `vector` extension → copy `DATABASE_URL` to Render env + local `.env`
6. **LLM key:** Add `GROQ_API_KEY` or `GEMINI_API_KEY` + `LLM_PROVIDER=groq|gemini|mock` to Render env

### Local Quickstart
```bash
cp .env.example .env
# fill GROQ_API_KEY or GEMINI_API_KEY, or leave LLM_PROVIDER=mock for offline demo
docker compose up --build
# frontend http://localhost:3000  backend http://localhost:8000/docs
```

### Key Endpoints
- `POST /api/v1/chat/stream` — SSE stream with citations. The score floor is
  `RETRIEVAL_THRESHOLD` (0.85) in pgvector mode and `TFIDF_THRESHOLD` (0.10) in
  the TF-IDF fallback; `/health` reports which is active.
- `POST /api/v1/documents/upload` — PDF ingest (chunk → embed → pgvector)
- `GET  /api/v1/documents` / `GET /api/v1/chunks/{id}`
- `POST /api/v1/threads` / `GET /api/v1/threads/{id}/messages`

### Status — read before demoing

This is a portfolio/demo build, not a production clinical system. Known limits:

- **Auth is off by default.** With `ENABLE_AUTH=false` every caller is anonymous
  and shares one dataset. Set `ENABLE_AUTH=true` and `API_KEYS=...` to enforce
  per-user isolation — note the frontend does not yet send a key.
- **Retrieval falls back to TF-IDF** unless `GEMINI_API_KEY` is set *and* has
  quota *and* documents have been ingested with embeddings. `GET /health` reports
  `retrieval_mode` (`pgvector` | `tfidf`) and the threshold actually in force —
  trust that over any number written here.
- **No migrations.** Schema comes from `create_all`; changing a model on an
  existing database requires manual intervention.
- **If Postgres is unreachable the API keeps serving from memory.** `/health`
  reports `status: degraded` with `storage_mode: in-memory (ephemeral)` and
  `/ready` returns 503. Nothing is persisted in that state.

### Compliance Notes
- Supabase encrypts at rest (AES-256) + TLS in transit — this applies only when
  the app is actually connected to Postgres; see `storage_mode` in `/health`.
- `LOG_QUERIES=false` by default — query text is not logged.
- Render's free tier sleeps after 15 minutes idle, so the first request takes
  ~30s. That is incompatible with an uptime guarantee; use a paid tier for one.

### Tests

Unit, integration, E2E, real-time SSE, UI and performance suites across both
stacks. See [TESTING.md](TESTING.md).

```bash
cd backend && uv run pytest -q      # backend, all layers
cd frontend && npm test            # unit + component
cd frontend && npm run test:e2e    # real browser (needs: npx playwright install chromium)
```

### Load Test
```bash
k6 run backend/tests/load_sse_k6.js
# override concurrency / target:
VUS=500 API_URL=https://<your-render>.onrender.com/api/v1/chat/stream \
  k6 run backend/tests/load_sse_k6.js
```
Defaults to 200 VUs. `http_req_waiting` (time to first byte) is the TTFT proxy;
`http_req_duration` covers the whole stream and is not a TTFT measurement.
