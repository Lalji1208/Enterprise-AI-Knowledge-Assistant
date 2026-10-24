"""Phase 3 - Chunking: split cleaned documents into overlapping, citable chunks.

Strategy (per page):
  1. Segment the text into units at paragraph/line breaks and sentence ends.
     Units longer than a chunk are split at word boundaries (hard cut only for a
     single "word" longer than a chunk, e.g. a URL or base64 blob).
  2. Greedily pack whole units into chunks of at most `chunk_size` characters.
  3. Start the next chunk inside the last `chunk_overlap` characters - at a
     sentence start if one falls there, otherwise at a word start.

Sentence detection is a simple regex, so abbreviations like "e.g." may split a
sentence; the overlap keeps such splits harmless for retrieval.

Usage:
    from rag.chunker import chunk_document
    chunks = chunk_document(cleaned_doc, chunk_size=500, chunk_overlap=50)
"""

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable

from models.chunk import Chunk
from models.document import Document

_UNIT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\S+")

Span = tuple[int, int]


def validate_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
        )


def _split_long_span(text: str, start: int, end: int, max_len: int) -> list[Span]:
    """Split one over-long unit at word boundaries; hard-cut over-long words."""
    pieces: list[Span] = []
    cur_start = cur_end = None
    for m in _WORD.finditer(text, start, end):
        w_start, w_end = m.span()
        # A single word longer than max_len: flush, then cut it into slices.
        while w_end - w_start > max_len:
            if cur_start is not None:
                pieces.append((cur_start, cur_end))
                cur_start = None
            pieces.append((w_start, w_start + max_len))
            w_start += max_len
        if cur_start is None:
            cur_start, cur_end = w_start, w_end
        elif w_end - cur_start <= max_len:
            cur_end = w_end
        else:
            pieces.append((cur_start, cur_end))
            cur_start, cur_end = w_start, w_end
    if cur_start is not None:
        pieces.append((cur_start, cur_end))
    return pieces


def segment(text: str, max_len: int) -> list[Span]:
    """Sentence/line units as (start, end) spans, trimmed, each <= max_len chars."""
    units: list[Span] = []
    pos = 0
    for boundary in [*_UNIT_BOUNDARY.finditer(text), None]:
        end = boundary.start() if boundary else len(text)
        raw = text[pos:end]
        stripped = raw.strip()
        if stripped:
            s = pos + (len(raw) - len(raw.lstrip()))
            e = s + len(stripped)
            units.extend([(s, e)] if e - s <= max_len else _split_long_span(text, s, e, max_len))
        if boundary:
            pos = boundary.end()
    return units


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Span]:
    """Return chunk spans over `text`. Pure function - easy to test and reuse."""
    validate_params(chunk_size, chunk_overlap)
    units = segment(text, chunk_size)
    if not units:
        return []

    starts = [s for s, _ in units]
    ends = [e for _, e in units]
    word_starts = [m.start() for m in _WORD.finditer(text)] if chunk_overlap else []

    spans: list[Span] = []
    s = starts[0]
    while True:
        # Furthest unit end that keeps the chunk within chunk_size.
        e = ends[bisect_right(ends, s + chunk_size) - 1]
        if spans and e <= spans[-1][1]:
            # The overlap left no room for the next (long) unit, so this chunk would
            # add nothing new. Drop the overlap and start at the next unit instead.
            s = starts[bisect_left(starts, spans[-1][1])]
            continue
        spans.append((s, e))
        if e == ends[-1]:
            return spans

        next_s = None
        if chunk_overlap:
            lo = max(e - chunk_overlap, s + 1)
            for candidates in (starts, word_starts):
                i = bisect_left(candidates, lo)
                if i < len(candidates) and candidates[i] < e:
                    next_s = candidates[i]
                    break
        if next_s is None:  # no overlap possible: start at the next unit
            next_s = starts[bisect_left(starts, e)]
        s = next_s


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk every page of a (cleaned) document. Empty pages yield no chunks."""
    chunks: list[Chunk] = []
    for page in document.pages:
        for s, e in chunk_text(page.text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-p{page.page_number}-c{index}",
                    doc_id=document.doc_id,
                    source=document.source,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=page.text[s:e],
                    start_char=s,
                    end_char=e,
                )
            )
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    return [c for doc in documents for c in chunk_document(doc, chunk_size, chunk_overlap)]
