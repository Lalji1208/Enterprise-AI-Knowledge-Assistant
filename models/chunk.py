"""Data model for text chunks - the unit that gets embedded and retrieved."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A slice of one page of one document.

    Chunks never span pages, so (source, page_number) is always an exact citation.
    `start_char`/`end_char` index into that page's cleaned text.
    """

    chunk_id: str                 # "<doc_id>-p<page>-c<index>", stable across runs
    doc_id: str
    source: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)  # position within the whole document
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @property
    def char_count(self) -> int:
        return len(self.text)
