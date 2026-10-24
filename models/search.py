"""Data models for search results."""

from pydantic import BaseModel

from models.chunk import Chunk


class RetrievedChunk(BaseModel):
    """A chunk returned by a search, with its similarity score."""

    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more relevant
    rerank_score: float | None = None  # cross-encoder logit, set only when reranking ran
