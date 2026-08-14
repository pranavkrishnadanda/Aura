from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/aura"
    LLM_PROVIDER: str = "mock"  # groq | gemini | mock
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBED_MODEL: str = "models/text-embedding-004"
    RETRIEVAL_THRESHOLD: float = 0.85
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
    EMBED_CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
