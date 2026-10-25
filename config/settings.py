"""Typed application settings, read from environment variables and the .env file.

Usage:
    from config.settings import get_settings
    settings = get_settings()
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunable values in one place. Field names map to UPPER_CASE env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_api_key: SecretStr = SecretStr("")  # SecretStr keeps the key out of logs/reprs
    openai_base_url: str = "https://api.openai.com/v1"
    llm_max_tokens: int = Field(default=800, gt=0)        # cap on answer length
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # Cap on retrieved text sent to the LLM (controls cost and context size).
    max_context_chars: int = Field(default=8000, gt=0)

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking (characters) ---
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    # --- Retrieval ---
    top_k: int = Field(default=5, gt=0)
    similarity_threshold: float = 0.30
    reranking_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Raw cross-encoder logit; chunks scoring lower are dropped (use e.g. -100 to keep all).
    rerank_min_score: float | None = -6.0
    # How many first-stage candidates the cross-encoder re-scores.
    rerank_candidates: int = Field(default=20, gt=0)
    # With reranking on, this replaces SIMILARITY_THRESHOLD for the first stage. It is
    # lower so that relevant passages with modest similarity still reach the reranker,
    # which does the real filtering (see evaluation/reports/retrieval_experiments.md).
    rerank_first_stage_min_score: float = 0.15
    # Hybrid search (with reranking only): add the best keyword (BM25) matches to the
    # reranker's candidates, so exact names, codes and e-mail addresses are found even
    # when their embedding similarity is near zero.
    keyword_search_enabled: bool = True
    keyword_candidates: int = Field(default=10, ge=0)

    # --- Storage ---
    documents_dir: Path = Path("data/documents")
    processed_dir: Path = Path("data/processed")
    database_url: str = "sqlite:///data/processed/app.db"

    # --- Upload limits ---
    max_upload_size_mb: int = Field(default=20, gt=0)
    # Kept as a comma-separated string so it reads naturally from .env
    allowed_extensions: str = ".pdf,.txt,.md"

    @field_validator("documents_dir", "processed_dir")
    @classmethod
    def _resolve_relative_to_project(cls, value: Path) -> Path:
        """Relative paths are anchored at the project root, not the current directory."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def index_dir(self) -> Path:
        """Where the FAISS index and its chunk metadata are saved."""
        return self.processed_dir / "index"

    @property
    def allowed_extension_set(self) -> frozenset[str]:
        """Normalised set of extensions, e.g. {'.pdf', '.txt', '.md'}."""
        exts = (e.strip().lower() for e in self.allowed_extensions.split(","))
        return frozenset(e if e.startswith(".") else f".{e}" for e in exts if e)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()
