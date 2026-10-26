"""Request/response schemas for the REST API."""

from pydantic import BaseModel, Field

from models.answer import Answer
from models.search import RetrievedChunk


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    sources: list[str] | None = None  # restrict to these file names


class ConversationTurn(BaseModel):
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    sources: list[str] | None = None
    # Earlier turns of the chat, oldest first, so follow-ups can be understood.
    history: list[ConversationTurn] = Field(default_factory=list, max_length=10)


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class AskResponse(Answer):
    pass


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunks: int
    pages: int


class SyncResponse(BaseModel):
    added: list[str]
    removed: list[str]
    unchanged: list[str]
    empty: list[str]
    failed: dict[str, str]
    chunks_added: int
    chunks_removed: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    embedding_model: str
    reranking_enabled: bool
    llm_provider: str
    llm_model: str
    llm_configured: bool


class ErrorResponse(BaseModel):
    detail: str
