"""Load, clean and chunk every document in DOCUMENTS_DIR and print a summary.

Run from the project root:
    python scripts/ingest.py
    python scripts/ingest.py --dir path/to/folder --recursive
    python scripts/ingest.py --show report.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from rag.chunker import chunk_document
from rag.cleaner import clean_document
from rag.loader import load_directory


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=settings.documents_dir)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--show", metavar="FILE", help="print cleaned text of one file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # our own error line is enough

    documents, failures = load_directory(
        args.dir, settings.allowed_extension_set, recursive=args.recursive
    )

    print(f"Loaded {len(documents)} document(s) from {args.dir}\n")
    total_chunks = 0
    for doc in documents:
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned, settings.chunk_size, settings.chunk_overlap)
        total_chunks += len(chunks)
        flag = "  (no text - needs OCR?)" if doc.is_empty else ""
        print(f"  {doc.source:<40} {doc.file_type:<4} pages={doc.page_count:<4} "
              f"chars={len(doc.text)}->{len(cleaned.text):<8} chunks={len(chunks):<5} "
              f"id={doc.doc_id}{flag}")
        if args.show == doc.source:
            print(f"\n{cleaned.text}\n")
    if total_chunks:
        print(f"\nTotal chunks: {total_chunks} (size={settings.chunk_size}, "
              f"overlap={settings.chunk_overlap})")
    if failures:
        print(f"\n{len(failures)} file(s) failed:")
        for name, reason in failures.items():
            print(f"  {name}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
