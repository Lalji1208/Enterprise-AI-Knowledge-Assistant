"""Build the evaluation corpus into a folder the normal ingestion pipeline can read.

`evaluation/corpus/` holds the source texts:
  *.md / *.txt     copied as they are
  *.pdf.txt        rendered to real PDFs, so evaluation exercises PDF parsing and
                   header/footer cleaning. Format: first line "HEADER: <text>",
                   pages separated by a line "=== PAGE ===".
"""

import shutil
import textwrap
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
PAGE_BREAK = "=== PAGE ==="
WRAP_CHARS = 90


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[str], header: str = "") -> Path:
    """Write a simple text PDF: optional header line, wrapped body, 'Page i of n' footer."""
    n = len(pages)
    font_id = 3 + 2 * n
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{3 + 2 * i} 0 R" for i in range(n)), n
        ),
    ]
    for i, page in enumerate(pages):
        lines: list[str] = []
        for paragraph in page.strip().split("\n"):
            lines.extend(textwrap.wrap(paragraph, WRAP_CHARS) or [""])
        body = " ".join(f"({_pdf_escape(line)}) Tj T*" for line in lines)
        ops = (
            f"BT /F1 9 Tf 72 760 Td ({_pdf_escape(header)}) Tj ET "
            f"BT /F1 10 Tf 14 TL 72 730 Td {body} ET "
            f"BT /F1 9 Tf 280 40 Td (Page {i + 1} of {n}) Tj ET"
        )
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>"
        )
        data = ops.encode("latin-1")
        objects.append(f"<< /Length {len(data)} >>\nstream\n{ops}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def build_corpus(target: Path, source: Path = CORPUS_DIR) -> list[Path]:
    """(Re)create `target` with the corpus documents. Returns the written files."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    written = []
    for src in sorted(source.iterdir()):
        if src.name.endswith(".pdf.txt"):
            text = src.read_text(encoding="utf-8")
            first, _, rest = text.partition("\n")
            header = first.removeprefix("HEADER:").strip() if first.startswith("HEADER:") else ""
            body = rest if header else text
            pages = [p for p in body.split(PAGE_BREAK) if p.strip()]
            written.append(write_pdf(target / src.name.removesuffix(".txt"), pages, header))
        elif src.suffix in {".md", ".txt"}:
            written.append(Path(shutil.copy2(src, target / src.name)))
    return written
