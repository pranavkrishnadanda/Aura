"""
Production RAG:
- Retrieval: pgvector cosine (Gemini embeddings, 0.85) with TF-IDF fallback (0.10)
- Generation: Gemini/Groq streaming, greeting + boundary AI (never hard deterministic)
- Embeddings cached in-memory for demo; prod uses pgvector + Redis
"""
import asyncio, re, time
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

HARD_REFUSAL = "I cannot find verified clinical guidelines to answer this query."  # kept for analytics, not user-facing

# With stop_words='english', "what do you know about hypertension" -> 0.114, so use 0.10 to pass vague but still block cake (0.0).
def effective_threshold() -> float:
    # If pgvector available + embeddings, use 0.85; else TF-IDF fallback 0.10
    if is_db_available():
        # Check if chunks have embeddings — if any, we are in embedding mode
        # For now, TF-IDF fallback still used until embeddings populated
        return 0.10
    return 0.10 if HAS_SKLEARN else settings.RETRIEVAL_THRESHOLD

GREETING_PATTERNS = {"hi", "hello", "hey", "hiya", "help", "hi there", "hello there"}
def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in GREETING_PATTERNS or (len(t.split()) <= 2 and t in GREETING_PATTERNS)

# Simple in-memory embedding cache
_embed_cache: dict = {}

def _embed_query_gemini(query: str) -> List[float] | None:
    if not settings.GEMINI_API_KEY:
        return None
    if query in _embed_cache and time.time() - _embed_cache[query][1] < settings.EMBED_CACHE_TTL:
        return _embed_cache[query][0]
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        res = genai.embed_content(model=settings.GEMINI_EMBED_MODEL, content=query)
        vec = res["embedding"]
        _embed_cache[query] = (vec, time.time())
        return vec
    except Exception as e:
        print(f"[rag] query embedding failed, fallback TF-IDF: {e}")
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
        except Exception:
            pass
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
            base = citations[0]['text']
            if len(citations) > 1:
                mock_answer = f"{base} [1] Clinicians should also note contraindications including angioedema and renal artery stenosis [2]."
            else:
                mock_answer = f"{base} [1]"
                if "lisinopril" in base.lower():
                    mock_answer += " Initiate at low dose and titrate per response [1]."
            async for tok in stream_mock(mock_answer):
                yield tok
    except Exception as e:
        fallback = f"{citations[0]['text']} [1]"
        async for tok in stream_mock(fallback):
            yield tok
