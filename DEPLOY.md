# Deploying Aura on $0

Three free accounts, no card required: Supabase (database), Render (API), Vercel
(frontend). Do them in that order — each step needs a value from the one before.

Total time: ~20 minutes.

---

## 1. Database — Supabase

1. [supabase.com](https://supabase.com) → **New project**. Pick a region near you
   and save the database password it generates.
2. **SQL Editor** → run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
   The app tries this itself but Supabase may refuse it to non-superusers, and
   without it the `chunks.embedding` column cannot be created.
3. **Project Settings → Database → Connection string → URI**. Copy it.

   Prefer the **Session pooler** URI — host `aws-0-<region>.pooler.supabase.com`,
   port **`5432`**. It survives Render's small connection allowance better than a
   direct connection.

   The **transaction** pooler (port `6543`) also works: the app disables prepared
   statements so pgbouncer transaction mode does not fail with
   *"prepared statement already exists"*. Session mode is still the better default.

   Replace `[YOUR-PASSWORD]` with the real password. Keep the `postgresql://`
   prefix — the app rewrites it to the psycopg3 driver on its own.

**Free tier reality:** the project pauses after ~1 week with no queries. One
request wakes it. Everything you ingested survives.

---

## 2. API — Render

1. [render.com](https://render.com) → **New → Web Service** → connect this repo.
2. Settings:
   - **Root Directory:** `backend`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/ready`
   - **Instance Type:** Free
3. **Environment** variables:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Supabase pooler URI from step 1 |
   | `LLM_PROVIDER` | `gemini` (or `groq`, or `mock` for no key at all) |
   | `GEMINI_API_KEY` | your key — **required for embeddings even if `LLM_PROVIDER=groq`** |
   | `CORS_ORIGINS` | your Vercel URL, added in step 3 |
   | `LOG_QUERIES` | `false` |
   | `ALLOW_ANONYMOUS_UPLOAD` | `false` for anything real — see below |

4. Deploy, then **verify it is really on Postgres**:
   ```bash
   curl https://<your-service>.onrender.com/health
   ```
   You want `"status": "ok"` and `"storage_mode": "postgres"`.

   If you see `"status": "degraded"` and `"storage_mode": "in-memory (ephemeral)"`,
   the database is NOT connected — read `db_error` in the same response, it names
   the cause. The API keeps serving in that state, so nothing else will tell you.

   Also check `"retrieval_mode"`. `pgvector` means semantic search is live;
   `tfidf` means embeddings are unavailable (usually a missing key or exhausted
   quota) and it fell back to keyword matching.

**Free tier reality:** sleeps after 15 minutes idle; the next request takes ~30s
to wake it. The UI shows a "waking the server" notice rather than appearing hung.

---

## 3. Frontend — Vercel

1. [vercel.com](https://vercel.com) → **Add New → Project** → import this repo.
2. **Root Directory:** `frontend` (Vercel autodetects Next.js).
3. **Environment Variables:**

   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<your-service>.onrender.com` |

   No trailing slash. `NEXT_PUBLIC_*` is baked in at build time, so changing it
   later needs a redeploy, not just a restart.
4. Deploy, copy the resulting URL.
5. **Go back to Render** and set `CORS_ORIGINS` to that exact Vercel URL
   (`https://your-app.vercel.app`, no trailing slash), then redeploy. Until you
   do, the browser blocks every request and the UI looks broken with only a
   console error.

---

## Verifying it actually works

```bash
API=https://<your-service>.onrender.com

curl -s $API/health | jq        # status, storage_mode, retrieval_mode
curl -s -o /dev/null -w '%{http_code}\n' $API/ready   # 200 = ready, 503 = not

curl -sN -X POST $API/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is first-line therapy for hypertension with CKD?","thread_id":"smoke"}'
```

The stream should emit `meta` (with citations), then `token` frames, then `done`.

Then open the Vercel URL and ask the same question in the UI.

---

## When something looks broken

| Symptom | Cause |
|---|---|
| UI loads, every request fails | `CORS_ORIGINS` on Render does not exactly match the Vercel origin |
| `/health` says `degraded` | DB unreachable. `db_error` in the response names it. Usually a wrong password or the direct URI instead of the pooler |
| `retrieval_mode: tfidf` | No usable `GEMINI_API_KEY`, or its quota is exhausted. Answers still work, but by keyword matching, not embeddings |
| Answers cite nothing | Corpus is empty, or every score is under the threshold. `GET /api/v1/documents` shows what is indexed |
| First request takes 30s | Render free tier cold start. Expected |
| Uploads say "partial" | Some chunks failed to persist. The job's `error` field says how many |

## The corpus is a security boundary

Uploaded document text is interpolated into the prompt for **other users'**
questions. A poisoned PDF was demonstrated making a live model obey it rather than
its own instructions. The prompt is hardened against this and it stops most
attempts, but the defence proved model-dependent — the same reframing that one
model refused, another obeyed.

If this deployment is reachable by anyone you do not trust, set
`ALLOW_ANONYMOUS_UPLOAD=false` and add documents with an API key. That is the only
measure that removes the attack rather than reducing its odds.

## Enabling auth

Off by default — every visitor is anonymous and shares one dataset, which is fine
for a public demo. To turn it on, set `ENABLE_AUTH=true` and `API_KEYS=key1,key2`
on Render, and `NEXT_PUBLIC_API_KEY` on Vercel. Note the frontend key ships inside
the public JS bundle, so it is a demo gate, not a real secret.
