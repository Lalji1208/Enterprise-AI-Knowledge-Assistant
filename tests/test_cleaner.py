"""Tests for Phase 2: text cleaning (rag/cleaner.py)."""

from models.document import Document, Page
from rag.cleaner import (
    CleaningConfig,
    clean_document,
    clean_text,
    fix_hyphenation,
    normalize_unicode,
    unwrap_lines,
)


def make_doc(pages: list[str], file_type: str = "pdf") -> Document:
    return Document(
        doc_id="abc",
        source=f"x.{file_type}",
        file_type=file_type,
        pages=[Page(page_number=i, text=t) for i, t in enumerate(pages, start=1)],
        size_bytes=1,
    )


# ---------- individual steps ----------

def test_normalize_unicode_ligatures_and_invisibles() -> None:
    raw = "e\ufb03cient\u200b work\u00a0here\r\nnext\x0cline\tend"
    assert normalize_unicode(raw) == "efficient work here\nnextline end"


def test_fix_hyphenation_joins_only_known_words() -> None:
    # Joined form occurs elsewhere in the text -> it was hyphenation: join.
    assert fix_hyphenation("See information below. More infor-\nmation here.") == (
        "See information below. More information here."
    )
    # Unknown joined form -> a real compound broken at its hyphen: keep the hyphen.
    # (Found by the evaluation corpus: "Part-\ntime" used to become "Parttime".)
    assert fix_hyphenation("Part-\ntime employees") == "Part-time employees"
    assert fix_hyphenation("infor-\nmation") == "infor-mation"
    assert fix_hyphenation("well-known") == "well-known"
    assert fix_hyphenation("COVID-\n19") == "COVID-\n19"  # not lowercase -> keep


def test_hyphenation_vocabulary_spans_pages() -> None:
    pages = ["The information desk.", "Ask for infor-\nmation."]
    cleaned = clean_document(make_doc(pages))
    assert cleaned.pages[1].text == "Ask for information."


def test_unwrap_lines_joins_paragraph_but_keeps_lists() -> None:
    raw = (
        "This sentence was\nhard wrapped by the PDF.\n\n"
        "Rules:\n- first item\n  continues here\n- second item\n1. numbered"
    )
    assert unwrap_lines(raw) == (
        "This sentence was hard wrapped by the PDF.\n\n"
        "Rules:\n- first item continues here\n- second item\n1. numbered"
    )


def test_unwrap_lines_keeps_headings_separate() -> None:
    assert unwrap_lines("# Title\nBody line one\nline two") == "# Title\nBody line one line two"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  a    b \n\n\n\n c  ") == "a b\n\nc"


# ---------- page-level ----------

def test_repeated_headers_footers_and_page_numbers_removed() -> None:
    pages = [
        f"ACME Corp - Confidential\nBody text of page {i}.\nPage {i} of 4"
        for i in range(1, 5)
    ]
    cleaned = clean_document(make_doc(pages))
    assert [p.text for p in cleaned.pages] == [f"Body text of page {i}." for i in range(1, 5)]


def test_headers_kept_when_too_few_pages() -> None:
    pages = ["Header\nBody one", "Header\nBody two"]
    cleaned = clean_document(make_doc(pages))
    assert cleaned.pages[0].text == "Header Body one"


def test_bare_number_mid_page_is_kept() -> None:
    page = "Quarterly totals\nRevenue\n\n42\n\nCosts\nmore text\n7"
    cleaned = clean_text(page)
    assert "42" in cleaned       # table value mid-page stays
    assert not cleaned.endswith("7")  # trailing page number goes


def test_clean_document_preserves_identity_and_page_numbers() -> None:
    doc = make_doc(["one", "two"])
    cleaned = clean_document(doc)
    assert cleaned.doc_id == doc.doc_id
    assert cleaned.source == doc.source
    assert [p.page_number for p in cleaned.pages] == [1, 2]
    assert doc.pages[0].text == "one"  # original not mutated


def test_markdown_is_not_reflowed() -> None:
    doc = make_doc(["Line one\nLine two"], file_type="md")
    assert clean_document(doc).pages[0].text == "Line one\nLine two"


def test_steps_can_be_disabled() -> None:
    config = CleaningConfig(fix_hyphenation=False, unwrap_lines=False)
    assert clean_text("infor-\nmation", config) == "infor-\nmation"
