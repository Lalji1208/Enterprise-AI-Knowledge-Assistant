"""Tests for Phase 1: document ingestion (rag/loader.py)."""

from pathlib import Path

import pytest
from conftest import make_pdf

from rag.loader import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    load_directory,
    load_document,
)

# ---------- single files ----------

def test_load_txt(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("Leave policy: 20 days per year.", encoding="utf-8")
    doc = load_document(f)
    assert doc.source == "notes.txt"
    assert doc.file_type == "txt"
    assert doc.page_count == 1
    assert doc.text == "Leave policy: 20 days per year."
    assert not doc.is_empty


def test_load_markdown_with_bom(tmp_path: Path) -> None:
    f = tmp_path / "guide.md"
    f.write_bytes("﻿# Title\nBody".encode())
    doc = load_document(f)
    assert doc.text == "# Title\nBody"  # BOM stripped


def test_non_utf8_text_falls_back(tmp_path: Path) -> None:
    f = tmp_path / "legacy.txt"
    f.write_bytes("caf\xe9".encode("cp1252"))
    assert load_document(f).text == "café"


def test_load_pdf_keeps_pages(tmp_path: Path) -> None:
    f = make_pdf(tmp_path / "report.pdf", ["First page text", "Second page text"])
    doc = load_document(f)
    assert doc.file_type == "pdf"
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "First page text" in doc.pages[0].text
    assert "Second page text" in doc.pages[1].text


def test_pdf_without_text_is_flagged_empty(tmp_path: Path) -> None:
    f = make_pdf(tmp_path / "scan.pdf", [""])
    assert load_document(f).is_empty


def test_corrupt_pdf_raises(tmp_path: Path) -> None:
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(f)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "sheet.xlsx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(f)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_document(tmp_path / "nope.txt")


def test_doc_id_is_content_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    assert load_document(a).doc_id == load_document(b).doc_id


# ---------- directories ----------

def test_load_directory_collects_failures_and_skips_duplicates(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "copy.txt").write_text("alpha", encoding="utf-8")  # duplicate content
    (tmp_path / "two.md").write_text("beta", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"garbage")
    (tmp_path / "ignored.docx").write_bytes(b"x")  # unsupported -> not even attempted

    docs, failures = load_directory(tmp_path)

    assert sorted(d.text for d in docs) == ["alpha", "beta"]
    assert list(failures) == ["bad.pdf"]


def test_load_directory_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("deep", encoding="utf-8")
    assert load_directory(tmp_path)[0] == []
    assert [d.source for d in load_directory(tmp_path, recursive=True)[0]] == ["deep.txt"]


def test_load_directory_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_directory(tmp_path / "missing")


# ---------- settings ----------

def test_settings_defaults_and_extensions() -> None:
    from config.settings import PROJECT_ROOT, Settings

    s = Settings(allowed_extensions="PDF, .txt,md", _env_file=None)
    assert s.allowed_extension_set == {".pdf", ".txt", ".md"}
    assert s.documents_dir == PROJECT_ROOT / "data" / "documents"
    assert "sk-" not in repr(s.openai_api_key)  # SecretStr never shows the value


def test_settings_reject_overlap_not_smaller_than_size() -> None:
    from pydantic import ValidationError

    from config.settings import Settings

    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100, _env_file=None)
