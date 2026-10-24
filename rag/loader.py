"""Phase 1 - Document ingestion: read PDF / TXT / MD files into Document objects.

Usage:
    from rag.loader import load_document, load_directory
    doc = load_document("data/documents/policy.pdf")
    docs, failures = load_directory("data/documents")
"""

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from models.document import Document, Page

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = frozenset({".pdf", ".txt", ".md"})


class DocumentLoadError(Exception):
    """Raised when a file cannot be turned into a Document."""


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised for extensions the loader does not handle."""


def _content_hash(data: bytes) -> str:
    """Short, stable id derived from file contents."""
    return hashlib.sha256(data).hexdigest()[:16]


def _decode_text(data: bytes, path: Path) -> str:
    """Decode text files, tolerating a BOM and falling back from UTF-8 to cp1252."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not valid UTF-8; decoding as cp1252", path.name)
        return data.decode("cp1252", errors="replace")


def _read_pdf_pages(path: Path) -> tuple[list[Page], dict[str, str]]:
    """Extract text page by page, plus title/author metadata when present."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentLoadError(f"{path.name} is password-protected")
        pages = [
            Page(page_number=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    except PdfReadError as exc:
        raise DocumentLoadError(f"{path.name} is not a readable PDF: {exc}") from exc

    metadata: dict[str, str] = {}
    info = reader.metadata
    if info is not None:
        if info.title:
            metadata["title"] = str(info.title)
        if info.author:
            metadata["author"] = str(info.author)
    return pages, metadata


def load_document(path: str | Path, allowed_extensions: Iterable[str] | None = None) -> Document:
    """Load a single file. Raises DocumentLoadError (or a subclass) on failure."""
    path = Path(path)
    allowed = frozenset(allowed_extensions) if allowed_extensions else DEFAULT_EXTENSIONS
    suffix = path.suffix.lower()

    if suffix not in allowed:
        raise UnsupportedFileTypeError(f"{path.name}: unsupported file type '{suffix}'")
    if not path.is_file():
        raise DocumentLoadError(f"{path} does not exist or is not a file")

    data = path.read_bytes()
    metadata: dict[str, str] = {}

    if suffix == ".pdf":
        pages, metadata = _read_pdf_pages(path)
    else:
        pages = [Page(page_number=1, text=_decode_text(data, path))]

    document = Document(
        doc_id=_content_hash(data),
        source=path.name,
        file_type=suffix.lstrip("."),
        pages=pages,
        size_bytes=len(data),
        metadata=metadata,
    )
    if document.is_empty:
        # Not an error: the file is valid, but it may need OCR later.
        logger.warning("%s: no extractable text", path.name)
    return document


def load_directory(
    directory: str | Path,
    allowed_extensions: Iterable[str] | None = None,
    recursive: bool = False,
) -> tuple[list[Document], dict[str, str]]:
    """Load every supported file in a directory.

    One bad file never stops the batch: failures are returned as {file name: reason}.
    Duplicate files (same content hash) are loaded only once.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DocumentLoadError(f"{directory} is not a directory")

    allowed = frozenset(allowed_extensions) if allowed_extensions else DEFAULT_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    candidates = sorted(
        p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in allowed
    )

    documents: list[Document] = []
    failures: dict[str, str] = {}
    seen_ids: set[str] = set()

    for path in candidates:
        try:
            doc = load_document(path, allowed)
        except DocumentLoadError as exc:
            failures[path.name] = str(exc)
            logger.error("Failed to load %s: %s", path.name, exc)
            continue
        if doc.doc_id in seen_ids:
            logger.info("Skipping duplicate %s", path.name)
            continue
        seen_ids.add(doc.doc_id)
        documents.append(doc)

    return documents, failures
