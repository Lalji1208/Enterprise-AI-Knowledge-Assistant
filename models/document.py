"""Data models for loaded documents."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Page(BaseModel):
    """Raw text of one page. Plain-text files are treated as a single page."""

    page_number: int = Field(ge=1)  # 1-based, matches what users see in a PDF viewer
    text: str


class Document(BaseModel):
    """A source file after parsing, before cleaning/chunking."""

    doc_id: str                  # content hash: identical files get the same id
    source: str                  # file name, used later in citations
    file_type: str               # extension without the dot: "pdf", "txt", "md"
    pages: list[Page]
    size_bytes: int
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Full document text with pages separated by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def is_empty(self) -> bool:
        """True when no text could be extracted (e.g. a scanned PDF with no text layer)."""
        return not any(page.text.strip() for page in self.pages)
