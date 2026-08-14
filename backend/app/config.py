from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/aura"
    LLM_PROVIDER: str = "mock"  # groq | gemini | mock
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"
    GEMINI_MODEL: str = "gemini-1.5-flash"
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
        env_file = ".env"
        extra = "ignore"

settings = Settings()
