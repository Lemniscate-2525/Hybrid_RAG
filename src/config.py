"""
Centralized Configuration Module using Pydantic Settings.
Validates all environment variables at startup and enforces security defaults.
"""
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application settings
    APP_NAME: str = "Production Hybrid RAG"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development", description="development, testing, or production")
    DEBUG: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Security & Auth
    SECRET_KEY: str = Field(default="change-this-in-production-use-a-strong-random-key-64-bytes-minimum", min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    BCRYPT_ROUNDS: int = 12

    # Database & Storage
    DATABASE_URL: str = Field(default=f"sqlite:///{BASE_DIR / 'data' / 'app.db'}")
    UPLOAD_DIR: Path = Field(default=BASE_DIR / "data" / "uploads")

    # Qdrant Vector DB (support both standard and legacy casing)
    QDRANT_URL: Optional[str] = Field(default=None, validation_alias="Qdrant_URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, validation_alias="Qdrant_API_KEY")
    COLLECTION_NAME: str = Field(default="hybrid_rag_documents", validation_alias="Collection_Name")

    # LLM Provider Configuration (Supports OpenRouter, Mistral, and OpenAI-compatible APIs)
    LLM_API_KEY: Optional[str] = Field(default=None, validation_alias="LLM_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    MISTRAL_API_KEY: Optional[str] = Field(default=None, validation_alias="MistralAI")
    LLM_BASE_URL: str = Field(default="https://openrouter.ai/api/v1/chat/completions", validation_alias="LLM_BASE_URL")
    LLM_MODEL: str = Field(default="meta-llama/llama-3.2-3b-instruct:free", validation_alias="LLM_MODEL")
    MISTRAL_MODEL: str = "open-mixtral-8x7b"

    # Embedding & Reranker Models
    DENSE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL_NAME: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Redis (Rate Limiting & Caching)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_RATE_LIMIT_ENABLED: bool = True
    REDIS_CACHE_ENABLED: bool = True
    REDIS_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    REDIS_TIMEOUT_MS: int = 1000

    # Rate Limits (requests per window)
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10
    RATE_LIMIT_QUERY_PER_MINUTE: int = 30
    RATE_LIMIT_UPLOAD_PER_HOUR: int = 20

    # Ingestion & Validation Guardrails
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    MAX_DOCUMENT_PAGES: int = 150
    MAX_QUERY_LENGTH: int = 500
    MIN_QUERY_LENGTH: int = 2
    MAX_RETRIEVAL_LIMIT: int = 20
    DEFAULT_RETRIEVAL_LIMIT: int = 5
    TOP_K_RERANK: int = 2

    # LLM Reliability
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 3
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RESET_TIMEOUT: float = 30.0

    # CORS & Trusted Proxies
    CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8000"]
    TRUSTED_PROXIES: List[str] = ["127.0.0.1", "::1"]

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()

