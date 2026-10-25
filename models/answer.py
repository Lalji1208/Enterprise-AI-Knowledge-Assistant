"""Data models for generated answers and their citations."""

from typing import Literal

from pydantic import BaseModel, Field

AnswerStatus = Literal["answered", "not_found"]


class Citation(BaseModel):
    """One source the answer actually cites, as [number] in the answer text."""

    number: int = Field(ge=1)
    source: str
    page_number: int
    chunk_id: str
    text: str                      # the full chunk text, for display / verification
    score: float
    rerank_score: float | None = None


class Answer(BaseModel):
    question: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    # Grounding checks (Phase 9). Warnings are shown to users, never hidden.
    warnings: list[str] = Field(default_factory=list)
    # "retrieval": passages found by search; "overview": the opening of one document,
    # used for questions like "what is this paper about?"
    mode: Literal["retrieval", "overview"] = "retrieval"
    model: str | None = None
    retrieved_count: int = 0       # chunks retrieved before context selection
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def is_grounded(self) -> bool:
        """An answer counts as grounded when it cites at least one valid source."""
        return self.status == "answered" and bool(self.citations)
