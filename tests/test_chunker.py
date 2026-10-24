"""Tests for Phase 3: chunking (rag/chunker.py)."""

import re
from itertools import pairwise

import pytest

from models.document import Document, Page
from rag.chunker import chunk_document, chunk_text, segment

SENTENCES = " ".join(f"Sentence number {i} talks about leave policy." for i in range(40))


def make_doc(pages: list[str]) -> Document:
    return Document(
        doc_id="d1",
        source="policy.pdf",
        file_type="pdf",
        pages=[Page(page_number=i, text=t) for i, t in enumerate(pages, start=1)],
        size_bytes=1,
    )


def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def assert_valid(text: str, spans: list[tuple[int, int]], size: int) -> None:
    """Shared invariants: size limit, word-aligned edges, full coverage, progress."""
    words = word_spans(text)
    word_starts = {s for s, _ in words}
    word_ends = {e for _, e in words}
    covered = set()
    for s, e in spans:
        assert 0 < e - s <= size
        # Starts on a word, unless inside a word too long to fit in any chunk.
        assert s in word_starts or any(ws < s < we and we - ws > size for ws, we in words)
        covered.update(range(s, e))
    for s, e in words:
        assert set(range(s, e)) <= covered, f"word at {s} not covered"
    assert [s for s, _ in spans] == sorted({s for s, _ in spans})  # strictly increasing
    assert [e for _, e in spans] == sorted({e for _, e in spans})  # every chunk adds text
    assert all(e in word_ends or e - s == size for s, e in spans)


# ---------- chunk_text ----------

def test_short_text_is_one_chunk() -> None:
    assert chunk_text("  Hello world.  ", 500, 50) == [(2, 14)]


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("   \n  ", 500, 50) == []


@pytest.mark.parametrize(("size", "overlap"), [(100, 0), (100, 30), (500, 50), (60, 59)])
def test_invariants_hold(size: int, overlap: int) -> None:
    assert_valid(SENTENCES, chunk_text(SENTENCES, size, overlap), size)


def test_chunks_end_at_sentence_boundaries() -> None:
    spans = chunk_text(SENTENCES, 200, 50)
    assert len(spans) > 1
    assert all(SENTENCES[s:e].endswith(".") for s, e in spans)


def test_consecutive_chunks_overlap() -> None:
    spans = chunk_text(SENTENCES, 200, 50)
    for (_, prev_end), (next_start, _) in pairwise(spans):
        assert prev_end - 50 <= next_start < prev_end


def test_zero_overlap_means_disjoint() -> None:
    spans = chunk_text(SENTENCES, 200, 0)
    for (_, prev_end), (next_start, _) in pairwise(spans):
        assert next_start >= prev_end


def test_long_sentence_splits_at_words() -> None:
    text = " ".join(["word"] * 300)  # no punctuation at all
    spans = chunk_text(text, 100, 20)
    assert_valid(text, spans, 100)
    assert all(text[s:e].split() == ["word"] * len(text[s:e].split()) for s, e in spans)


def test_giant_token_is_hard_cut() -> None:
    text = "start " + "x" * 250 + " end"
    spans = chunk_text(text, 100, 10)
    assert all(e - s <= 100 for s, e in spans)
    joined = "".join(text[s:e] for s, e in chunk_text(text, 100, 0))
    assert joined.replace(" ", "") == text.replace(" ", "")


def test_overlap_before_long_unit_does_not_duplicate_chunk() -> None:
    """Regression (found by fuzzing): an overlap start right before a unit too long
    to fit used to emit a chunk ending where the previous one ended."""
    text = "x. " + "w " * 29 + "w. " + "c" * 95 + "."
    spans = chunk_text(text, 100, 50)
    assert_valid(text, spans, 100)
    assert text[spans[-1][0]:spans[-1][1]] == "c" * 95 + "."


def test_paragraphs_and_lines_are_units() -> None:
    text = "Heading\n- item one\n- item two\n\nNext paragraph"
    assert [text[s:e] for s, e in segment(text, 500)] == [
        "Heading", "- item one", "- item two", "Next paragraph"
    ]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_params_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


# ---------- chunk_document ----------

def test_chunk_document_metadata_and_offsets() -> None:
    doc = make_doc([SENTENCES, "", "Short last page."])
    chunks = chunk_document(doc, 300, 50)

    assert {c.page_number for c in chunks} == {1, 3}  # empty page 2 skipped
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        page_text = doc.pages[c.page_number - 1].text
        assert c.text == page_text[c.start_char:c.end_char]
        assert c.source == "policy.pdf" and c.doc_id == "d1"
    assert chunks[-1].text == "Short last page."


def test_chunk_ids_are_deterministic() -> None:
    doc = make_doc([SENTENCES])
    assert [c.chunk_id for c in chunk_document(doc, 200, 50)] == [
        c.chunk_id for c in chunk_document(doc, 200, 50)
    ]
