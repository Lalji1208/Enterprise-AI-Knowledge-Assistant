"""Phase 2 - Text cleaning: normalise raw extracted text before chunking.

Cleaning is page-aware so page numbers (needed for citations) survive.

Usage:
    from rag.cleaner import clean_document
    cleaned = clean_document(doc)
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from models.document import Document, Page


@dataclass(frozen=True)
class CleaningConfig:
    """Toggle individual steps, mainly so experiments can measure their effect."""

    normalize_unicode: bool = True
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    fix_hyphenation: bool = True
    unwrap_lines: bool = True
    # A line counts as a repeated header/footer if it appears on at least this
    # share of pages, and the document has at least `min_pages_for_headers` pages.
    header_footer_threshold: float = 0.6
    min_pages_for_headers: int = 3
    # Only the first/last N lines of each page are considered header/footer candidates.
    header_footer_scan_lines: int = 2


# Zero-width and other invisible characters that PDFs often contain.
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?\s*(?:(?:of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
# Lines that must keep their own line: bullets, numbered items, markdown headings,
# table rows, block quotes.
_STRUCTURAL_LINE = re.compile(r"^\s*(?:[-*•▪◦+]\s|\d{1,3}[.)]\s|#{1,6}\s|\||>)")
_HYPHEN_BREAK = re.compile(r"(\w+)-\n[ \t]*([a-z]\w*)")
_WORD_TOKEN = re.compile(r"\w+")


def normalize_unicode(text: str) -> str:
    """NFKC (splits ligatures like 'ﬁ', unifies full-width chars), drop invisibles,
    map exotic spaces to ' ', and unify line endings."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL.sub("", text)
    # NFKC already maps most odd spaces; this catches the rest.
    return "".join(" " if ch.isspace() and ch != "\n" else ch for ch in text)


def _line_signature(line: str) -> str:
    """Compare header/footer lines ignoring digits, so 'Page 3' == 'Page 4'."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_repeated_lines(pages: list[str], config: CleaningConfig) -> set[str]:
    """Signatures of lines that recur at the top/bottom of many pages."""
    if len(pages) < config.min_pages_for_headers:
        return set()
    n = config.header_footer_scan_lines
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        # On short pages only the very first/last line can be a header/footer;
        # otherwise body lines would become candidates too.
        edge = lines[:n] + lines[-n:] if len(lines) > 2 * n else lines[:1] + lines[-1:]
        counts.update({_line_signature(ln) for ln in edge})  # count once per page
    min_pages = config.header_footer_threshold * len(pages)
    return {sig for sig, c in counts.items() if c >= min_pages and sig}


def remove_lines(text: str, signatures: set[str], drop_page_numbers: bool, edge_lines: int) -> str:
    """Drop repeated header/footer lines anywhere, and page-number lines only at the
    top/bottom of the page (a bare number mid-page is likely table data)."""
    lines = text.split("\n")
    content_idx = [i for i, ln in enumerate(lines) if ln.strip()]
    edges = set(content_idx[:edge_lines] + content_idx[-edge_lines:])
    kept = []
    for i, line in enumerate(lines):
        if signatures and _line_signature(line) in signatures:
            continue
        if drop_page_numbers and i in edges and _PAGE_NUMBER.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def fix_hyphenation(text: str, vocabulary: set[str] | None = None) -> str:
    """Repair words split across lines at a hyphen.

    A line-end hyphen is either inserted by hyphenation ('infor-' + 'mation') or
    part of the word ('part-' + 'time'). Business documents mostly contain the
    second kind, so the halves are joined only when the joined word also occurs
    elsewhere in the document (`vocabulary`, lower-cased words); otherwise the
    hyphen is kept and only the line break removed. This never invents a word.
    Lines continuing with an uppercase letter or digit ('COVID-' + '19') are untouched.
    """
    if vocabulary is None:
        vocabulary = {w.lower() for w in _WORD_TOKEN.findall(text)}

    def join(match: re.Match) -> str:
        head, tail = match.group(1), match.group(2)
        joined = head + tail
        return joined if joined.lower() in vocabulary else f"{head}-{tail}"

    return _HYPHEN_BREAK.sub(join, text)


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph; keep blank lines and list items."""
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for para in paragraphs:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = [lines[0]]
        for line in lines[1:]:
            # New line if this one is structural, or the previous one is a heading /
            # table row. A plain line after a bullet continues that bullet.
            if _STRUCTURAL_LINE.match(line) or merged[-1].startswith(("#", "|")):
                merged.append(line)
            else:
                merged[-1] = f"{merged[-1]} {line}"
        out.append("\n".join(merged))
    return "\n\n".join(out)


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str, config: CleaningConfig | None = None) -> str:
    """Clean a single string (no cross-page header detection)."""
    config = config or CleaningConfig()
    return _clean_pages([text], config)[0]


def _clean_pages(pages: list[str], config: CleaningConfig) -> list[str]:
    if config.normalize_unicode:
        pages = [normalize_unicode(p) for p in pages]
    else:
        pages = [p.replace("\r\n", "\n") for p in pages]

    repeated = find_repeated_lines(pages, config) if config.remove_headers_footers else set()
    if repeated or config.remove_page_numbers:
        pages = [
            remove_lines(p, repeated, config.remove_page_numbers, config.header_footer_scan_lines)
            for p in pages
        ]

    if config.fix_hyphenation:
        # Vocabulary from the whole document, so a word seen on page 1 helps page 5.
        vocabulary = {w.lower() for p in pages for w in _WORD_TOKEN.findall(p)}
        pages = [fix_hyphenation(p, vocabulary) for p in pages]
    if config.unwrap_lines:
        pages = [unwrap_lines(p) for p in pages]
    return [collapse_whitespace(p) for p in pages]


def clean_document(document: Document, config: CleaningConfig | None = None) -> Document:
    """Return a cleaned copy. doc_id, source and page numbers are unchanged."""
    config = config or CleaningConfig()
    # Markdown line breaks carry structure; never re-flow them.
    if document.file_type == "md" and config.unwrap_lines:
        config = CleaningConfig(**{**config.__dict__, "unwrap_lines": False})

    cleaned = _clean_pages([p.text for p in document.pages], config)
    pages = [Page(page_number=p.page_number, text=t) for p, t in zip(document.pages, cleaned)]
    return document.model_copy(update={"pages": pages})
