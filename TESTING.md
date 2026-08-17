# Testing

```bash
# Backend — everything
cd backend && uv run pytest -q

# Backend — by layer
uv run pytest tests/unit -q            # pure logic, no I/O
uv run pytest tests/integration -q     # API, DB layer, ingest pipeline
uv run pytest tests/e2e -q             # full flows, providers mocked
uv run pytest tests/realtime -q        # SSE framing, disconnects, failures
uv run pytest tests/performance -q     # event-loop and latency budgets
uv run pytest -m "not slow" -q         # skip the slow ones
uv run pytest --cov=app --cov-report=term-missing

# Frontend — unit + component
cd frontend && npm test
npm run test:watch
npm run test:coverage

# Frontend — real browser
npx playwright install chromium        # once
npm run test:e2e
npx playwright test --ui               # interactive

# Load / performance
k6 run backend/tests/load_sse_k6.js
VUS=500 API_URL=https://<host>/api/v1/chat/stream k6 run backend/tests/load_sse_k6.js
```

## Layout

| Path | Layer | What it covers |
|---|---|---|
| `backend/tests/unit/` | Unit | Chunking, DB-URL normalisation, auth primitives, threshold selection, cache pruning, prompt construction |
| `backend/tests/integration/` | Integration | Every endpoint, the DB layer against the in-memory fallback and simulated failures, the ingest pipeline, CORS and rate limiting |
| `backend/tests/e2e/` | End-to-end | Chat flow, upload→retrieve, provider behaviour with Groq/Gemini mocked |
| `backend/tests/realtime/` | Real-time | SSE frame validity, event ordering, client disconnect, mid-stream failure |
| `backend/tests/performance/` | Performance | Event-loop responsiveness, TTFT, chunking throughput, cache and job-table bounds |
| `frontend/tests/unit/` | Unit | `streamChat` SSE parsing: split frames, UTF-8 boundaries, multi-line `data:`, abort, single-settle |
| `frontend/tests/components/` | UI | Chat, AdminUpload, CitationPanel via Testing Library |
| `frontend/tests/unit/motion.test.ts` | Unit | Momentum projection, boundary resistance, spring convergence/interruption |
| `frontend/e2e/` | Browser E2E | Playwright against a real Next.js build, backend stubbed at the network layer |
| `frontend/e2e/integrated.spec.ts` | Contract | **No stubs** — real browser against the real FastAPI process |
| `frontend/e2e/drawer.spec.ts` | Gesture | Real pointer drags: 1:1 tracking, resistance, flick, catch mid-flight |
| `backend/tests/load_sse_k6.js` | Load | Concurrent SSE; `http_req_waiting` is the TTFT proxy |

## Things worth knowing

**The integrated project is the only place the two halves meet.** Every other spec
stubs the API, which verifies the UI behaves given well-formed responses — not
that the frontend and backend agree on field names, event shapes or citation
numbering. A stub encodes whatever the frontend already expects, so a contract
mismatch is invisible to it.

**No test makes a real LLM or embedding call.** Providers are mocked and Playwright stubs the
API, so the suite needs no database, no LLM key, and no quota. Set `E2E_LIVE_API=1`
to point Playwright at a running backend instead.

**Streaming tests must reset sse-starlette between requests.** `EventSourceResponse`
caches `AppStatus.should_exit_event` process-globally while `TestClient` creates a
fresh event loop per request, so a second streaming call in the same process dies
with *"is bound to a different event loop"*. `tests/conftest.py` resets it per test;
a test issuing multiple streams must reset it per request. This is a test-harness
artifact only — production runs one loop for the process lifetime.

**Performance thresholds are deliberately loose.** They exist to catch
order-of-magnitude regressions — an accidental O(n²), an unbounded cache, a
per-chunk DB commit — not to police machine variance. `test_event_loop.py` includes
a control test asserting the *blocking* path does starve the loop; without it the
non-blocking assertion would prove nothing.

**CI runs the backend suite twice**: once against real Postgres with pgvector, once
with no database at all. The first asserts the DB is genuinely connected before
testing, because the app's fallback is designed to keep serving without one — so
every other test would still pass if the driver silently broke. The second asserts
that degraded mode reports itself honestly.
