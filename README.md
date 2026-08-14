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
- `POST /api/v1/chat/stream` — SSE stream (TTFT <400ms, 0.85 threshold, citations)
- `POST /api/v1/documents/upload` — PDF ingest (chunk → embed → pgvector)
- `GET  /api/v1/documents` / `GET /api/v1/chunks/{id}`
- `POST /api/v1/threads` / `GET /api/v1/threads/{id}/messages`

### Compliance Notes
- Supabase encrypts at rest (AES-256) + TLS in transit
- `LOG_QUERIES=false` by default — no PHI logged; LLM providers set to no-training
- Re-runs on Render free tier still meet 99.9% demo uptime; for prod use paid tier to avoid sleep

### Load Test (1000 concurrent SSE)
```bash
k6 run backend/tests/load_sse.js
```
