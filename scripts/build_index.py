"""Sync the FAISS index with DOCUMENTS_DIR (embeds only new or changed files).

Run from the project root:
    python scripts/build_index.py
    python scripts/build_index.py --rebuild          # re-embed everything
    python scripts/build_index.py --query "How many leave days do I get?"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from rag.embedder import get_embedder
from rag.vector_store import VectorStoreError
from services.indexing import sync_directory


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=settings.documents_dir)
    parser.add_argument("--rebuild", action="store_true", help="ignore the saved index")
    parser.add_argument("--query", help="search the index after syncing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    embedder = get_embedder()
    start = time.perf_counter()
    try:
        store, report = sync_directory(settings, embedder, rebuild=args.rebuild,
                                       directory=args.dir)
    except VectorStoreError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"Index: {settings.index_dir}  (model: {embedder.model_name})")
    for label, names in [("added", report.added), ("removed", report.removed),
                         ("unchanged", report.unchanged), ("no text", report.empty)]:
        if names:
            print(f"  {label:<10} {len(names):>3}: {', '.join(sorted(names))}")
    for name, reason in report.failed.items():
        print(f"  failed     {name}: {reason}")
    print(f"Chunks: +{report.chunks_added} -{report.chunks_removed} -> {report.total_chunks} "
          f"total  ({time.perf_counter() - start:.1f}s)")

    if args.query:
        results = store.search(embedder.embed_query(args.query), top_k=settings.top_k)
        print(f"\nTop {len(results)} for: {args.query}")
        for r in results:
            c = r.chunk
            print(f"  {r.score:.3f}  {c.source} p.{c.page_number}: {c.text[:100]}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
