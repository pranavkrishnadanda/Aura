from pathlib import Path

from pydantic_settings import BaseSettings

# app/config.py -> app/ -> backend/ -> repo root
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/aura"
    LLM_PROVIDER: str = "mock"  # groq | gemini | mock
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"
    # gemini-1.5-flash has been retired and is absent from list_models, so every
    # generateContent call 404s. The "-latest" alias is used so this does not rot
    # again the next time a specific version is withdrawn.
    GEMINI_MODEL: str = "gemini-flash-latest"
    # text-embedding-004 has been retired from the v1beta endpoint and now 404s on
    # embedContent; gemini-embedding-001 is its replacement.
    GEMINI_EMBED_MODEL: str = "models/gemini-embedding-001"
    # Must equal the Vector(...) width in models.py, which reads this value. The
    # newer embedding models default to 3072 dims, so we request 768 explicitly --
    # a mismatch makes every pgvector insert fail and silently degrade to TF-IDF.
    EMBED_DIM: int = 768
    # Similarity floor for pgvector cosine retrieval (embedding mode).
    RETRIEVAL_THRESHOLD: float = 0.85
    # Separate, much lower floor for the TF-IDF fallback: sparse cosine scores are
    # not comparable to dense-embedding cosine scores, so one number cannot serve
    # both modes. Vague-but-valid clinical queries land around 0.11 here.
    TFIDF_THRESHOLD: float = 0.10
    TOP_K: int = 5
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    CORS_ORIGINS: str = "http://localhost:3000"
    LOG_QUERIES: bool = False
    # production hardening
    MAX_MESSAGE_LENGTH: int = 4000
    # Bound generation. Unbounded output can run past the platform's request
    # timeout mid-stream, leaving the reader with a truncated clinical answer.
    MAX_OUTPUT_TOKENS: int = 1024
    RATE_LIMIT_ANON: str = "60/minute"
    RATE_LIMIT_AUTH: str = "300/minute"
    MAX_PDF_MB: int = 50
    ENABLE_AUTH: bool = False  # set True in prod to require X-API-Key
    # Comma-separated API keys accepted when ENABLE_AUTH is true. Enabling auth
    # without setting these fails closed (every request is rejected) rather than
    # silently admitting everyone.
    API_KEYS: str = ""
    EMBED_CACHE_TTL: int = 3600

    class Config:
        # Absolute, not "./.env". A bare relative path resolves against the working
        # directory, so the file was only picked up when the process happened to be
        # started from the repo root -- and the documented command is
        # `cd backend && uvicorn ...`, which silently ignored it. Keys looked unset
        # while the file plainly contained them.
        #
        # Both locations are accepted: backend/.env for a backend-only checkout,
        # and the repo root .env that the frontend also reads.
        env_file = (
            str(_BACKEND_DIR / ".env"),
            str(_REPO_ROOT / ".env"),
        )
        extra = "ignore"

settings = Settings()
