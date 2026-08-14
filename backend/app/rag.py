"""
Production RAG:
- Retrieval: pgvector cosine (Gemini embeddings, 0.85) with TF-IDF fallback (0.10)
- Generation: Gemini/Groq streaming, greeting + boundary AI (never hard deterministic)
- Embeddings cached in-memory for demo; prod uses pgvector + Redis
"""
import asyncio, re, time, logging
from typing import List, Tuple
from app.config import settings
from app.db import list_chunks, is_db_available
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("aura.rag")

HARD_REFUSAL = "I cannot find verified clinical guidelines to answer this query."  # kept for analytics, not user-facing

# Cache the "are there any embedded chunks?" probe; it is consulted per request and
# per streamed token, and the answer only changes when a document is ingested.
_embedded_probe: dict = {"value": False, "checked_at": 0.0}
_EMBEDDED_PROBE_TTL = 30.0


def _has_embedded_chunks() -> bool:
    now = time.time()
    if now - _embedded_probe["checked_at"] < _EMBEDDED_PROBE_TTL:
        return _embedded_probe["value"]
    value = False
    try:
        from app.db import _SessionLocal
        from sqlalchemy import text as sql_text
        with _SessionLocal() as s:
            value = s.execute(sql_text("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1")).first() is not None
    except Exception as e:
        logger.debug("embedded-chunk probe failed, assuming none: %s", e)
        value = False
    _embedded_probe.update({"value": value, "checked_at": now})
    return value

def retrieval_mode() -> str:
    """Which retrieval path a query will actually take right now.

    'pgvector' requires a live DB, an embedding key, and at least one embedded
    chunk. Anything less means we are really running sparse TF-IDF, and the
    caller (and /health) should be told that rather than shown a vector number.
    """
    if is_db_available() and settings.GEMINI_API_KEY and _has_embedded_chunks():
        return "pgvector"
    return "tfidf" if HAS_SKLEARN else "token-overlap"


def effective_threshold() -> float:
    """Score floor for the retrieval mode actually in use.

    Previously this returned 0.10 on every branch while /health and render.yaml
    advertised RETRIEVAL_THRESHOLD=0.85, so the documented grounding gate was
    never the one enforced. Dense and sparse cosine scores live on different
    scales, so each mode gets its own configured floor.
    """
    return settings.RETRIEVAL_THRESHOLD if retrieval_mode() == "pgvector" else settings.TFIDF_THRESHOLD

GREETING_PATTERNS = {"hi", "hello", "hey", "hiya", "help", "hi there", "hello there"}
def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in GREETING_PATTERNS or (len(t.split()) <= 2 and t in GREETING_PATTERNS)

# Simple in-memory embedding cache, keyed by query text. Entries were previously
# never removed -- expired ones were skipped on read but still retained, so the dict
# grew without bound and kept raw clinical queries resident for the process
# lifetime. Bounded and pruned below.
_embed_cache: dict = {}
_EMBED_CACHE_MAX = 500


def _prune_embed_cache() -> None:
    now = time.time()
    for k in [k for k, (_, ts) in _embed_cache.items() if now - ts >= settings.EMBED_CACHE_TTL]:
        _embed_cache.pop(k, None)
    if len(_embed_cache) > _EMBED_CACHE_MAX:
        for k in sorted(_embed_cache, key=lambda k: _embed_cache[k][1])[: len(_embed_cache) - _EMBED_CACHE_MAX]:
            _embed_cache.pop(k, None)

def _embed_query_gemini(query: str) -> List[float] | None:
    if not settings.GEMINI_API_KEY:
        return None
    if query in _embed_cache and time.time() - _embed_cache[query][1] < settings.EMBED_CACHE_TTL:
        return _embed_cache[query][0]
    try:
        import google.generativeai as genai
        from app.ingest import _embed_call
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Must match the ingest-side call exactly: a query embedded at a different
        # width than the stored chunks cannot be compared against them.
        res = _embed_call(genai, query)
        vec = res["embedding"]
        if vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]
        _embed_cache[query] = (vec, time.time())
        _prune_embed_cache()
        return vec
    except Exception as e:
        logger.warning("query embedding failed, falling back to TF-IDF: %s", e)
        return None

def retrieve(query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
    # Try pgvector cosine if DB available and we have query embedding
    if is_db_available():
        vec = _embed_query_gemini(query)
        if vec is not None:
            try:
                from app.db import _SessionLocal
                from app.models import Chunk
                from sqlalchemy import text as sql_text
                # Use pgvector cosine distance: 1 - cosine
                with _SessionLocal() as s:
                    # Raw SQL for cosine: embedding <=> query
                    rows = s.execute(sql_text(
                        "SELECT id, doc_id, doc_title, page, text, 1 - (embedding <=> CAST(:vec AS vector)) as score "
                        "FROM chunks WHERE embedding IS NOT NULL ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k"
                    ), {"vec": str(vec), "k": top_k}).fetchall()
                    if rows:
                        result = []
                        for r in rows:
                            result.append(({"id": r.id, "doc_id": r.doc_id, "doc_title": r.doc_title, "page": r.page, "text": r.text}, float(r.score)))
                        return result
            except Exception as e:
                print(f"[rag] pgvector search failed, fallback TF-IDF: {e}")
    # TF-IDF fallback
    chunks = list_chunks()
    if not chunks:
        return []
    if HAS_SKLEARN:
        corpus = [c["text"] for c in chunks] + [query]
        vec = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        q_vec = vec[-1]
        c_vecs = vec[:-1]
        sims = cosine_similarity(q_vec, c_vecs)[0]
    else:
        q_tokens = set(re.findall(r"\w+", query.lower()))
        sims = []
        for c in chunks:
            c_tokens = set(re.findall(r"\w+", c["text"].lower()))
            sims.append(len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens)))
        sims = np.array(sims)
    ranked = sorted(zip(chunks, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

async def stream_groq(prompt: str, context: str):
    from groq import Groq
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")
    client = Groq(api_key=settings.GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are Aura, a clinical intelligence assistant. Answer ONLY from the provided context. Every claim must include an inline citation like [1] or [2]. If context is insufficient, say you cannot find verified guidelines. Never hallucinate."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer with citations:"},
        ],
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta
            await asyncio.sleep(0.01)

async def stream_gemini(prompt: str, context: str):
    import google.generativeai as genai
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL,
        system_instruction="You are Aura, a clinical assistant. You MUST answer ONLY from the provided Context. Every factual sentence MUST end with a citation like [1] or [2] referencing the Context number. If Context is empty or not relevant, say exactly: 'I cannot find verified clinical guidelines to answer this query.' Never invent citations or facts. Be concise, clinical, and helpful.")
    resp = model.generate_content(f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer concisely with inline citations [n]:", stream=True)
    for chunk in resp:
        if getattr(chunk, "text", None):
            yield chunk.text
            await asyncio.sleep(0.005)

async def stream_mock(answer: str):
    for token in answer.split(" "):
        yield token + " "
        await asyncio.sleep(0.03)

async def generate_answer(query: str, retrieved: List[Tuple[dict, float]]):
    """Yield tokens — greeting → boundary-aware AI, not deterministic echo."""
    if is_greeting(query):
        greeting = "Hi — I'm Aura, your clinical intelligence assistant. Ask me about treatment guidelines, drug interactions, or trial protocols — every answer will be grounded in verified sources with citations. Try: “First-line therapy for hypertension with CKD?”"
        async for tok in stream_mock(greeting):
            yield tok
        return
    thresh = effective_threshold()
    if not retrieved or retrieved[0][1] < thresh:
        boundary = (
            f"That's outside my current clinical intelligence scope — I’m built to answer only from verified clinical guidelines, trial protocols, and pharmacological data with citations [like FDA guidance]. "
            f"Your question about “{query[:120]}” doesn’t match my approved knowledge base, so I can’t cite a source for it. "
            f"I can help with: treatment guidelines, drug interactions, contraindications, dosing, or trial eligibility — try asking “What is first-line therapy for hypertension with CKD?” or upload a protocol PDF."
        )
        provider = settings.LLM_PROVIDER
        try:
            if provider == "gemini" and settings.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(settings.GEMINI_MODEL,
                    system_instruction="You are Aura. For out-of-scope questions, politely explain you only answer from verified clinical guidelines with citations, and redirect to clinical topics. Never invent medical facts. Be concise and helpful.")
                resp = model.generate_content(boundary, stream=True)
                for chunk in resp:
                    if getattr(chunk, "text", None):
                        yield chunk.text
                        await asyncio.sleep(0.005)
                return
            elif provider == "groq" and settings.GROQ_API_KEY:
                from groq import Groq
                client = Groq(api_key=settings.GROQ_API_KEY)
                stream = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[{"role": "system", "content": "You are Aura. Explain out-of-scope politely, redirect to clinical intelligence."},
                              {"role": "user", "content": boundary}],
                    temperature=0.2, stream=True)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield delta
                        await asyncio.sleep(0.01)
                return
        except Exception as e:
            # Falling back to the canned boundary text is correct here, but silence
            # made a persistently broken provider indistinguishable from normal
            # out-of-scope handling.
            logger.warning("boundary rephrase via %s failed, using canned text: %s", provider, e)
        async for tok in stream_mock(boundary):
            yield tok
        return

    context_parts = []
    citations = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        if score < thresh:
            continue
        context_parts.append(f"[{i}] {chunk['text']} (Source: {chunk['doc_title']}, p.{chunk['page']}, score={score:.2f})")
        citations.append(chunk)
    context = "\n".join(context_parts)

    if not context_parts:
        boundary = f"That's outside my current clinical intelligence scope — I can only cite verified guidelines. Your question about “{query[:80]}” has no matching source. Try rephrasing as a clinical question."
        async for tok in stream_mock(boundary):
            yield tok
        return

    provider = settings.LLM_PROVIDER
    try:
        if provider == "groq":
            async for tok in stream_groq(query, context):
                yield tok
        elif provider == "gemini":
            async for tok in stream_gemini(query, context):
                yield tok
        else:
            # Offline mock. It may only replay retrieved text verbatim, each span
            # tagged with the citation it actually came from.
            #
            # This previously appended fixed clinical claims -- "contraindications
            # including angioedema and renal artery stenosis [2]" for any query with
            # more than one citation, and lisinopril dosing advice whenever that
            # string appeared -- regardless of the question or what the sources said,
            # under a citation marker pointing at an unrelated chunk. Inventing
            # medical guidance and attributing it to a source is the exact failure
            # this product claims to prevent.
            mock_answer = " ".join(
                f"{chunk['text']} [{i}]" for i, chunk in enumerate(citations[:2], 1)
            )
            async for tok in stream_mock(mock_answer):
                yield tok
    except Exception as e:
        # Do not dress a provider outage up as an answer. The old fallback emitted
        # raw chunk text with a fabricated [1], so a clinician saw something that
        # looked like a grounded response when generation had actually failed.
        logger.error("generation failed via provider=%s: %s", provider, e)
        sources = ", ".join("{} p.{}".format(c["doc_title"], c["page"]) for c in citations[:3])
        notice = (
            "I couldn't complete that answer — the language model is unavailable right now. "
            "Please retry rather than relying on a partial response. "
            f"The sources I retrieved for this query were: {sources}."
        )
        async for tok in stream_mock(notice):
            yield tok
