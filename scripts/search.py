"""Search the saved index from the command line (build it first with build_index.py).

Run from the project root:
    python scripts/search.py "How many leave days do I get?"
    python scripts/search.py "receipts" --top-k 3 --min-score 0 --source handbook.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import RetrievalError, Retriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, help="default: TOP_K")
    parser.add_argument("--min-score", type=float, help="default: SIMILARITY_THRESHOLD")
    parser.add_argument("--source", action="append", help="limit to a file (repeatable)")
    parser.add_argument("--max-per-doc", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    try:
        retriever = Retriever.from_settings()
        results = retriever.retrieve(args.query, top_k=args.top_k, min_score=args.min_score,
                                     sources=args.source, max_per_document=args.max_per_doc)
    except RetrievalError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not results:
        hint = "--min-score 0" + (" and RERANKING_ENABLED=false" if retriever.reranker else "")
        print(f"No relevant passages found (to see the closest matches anyway: {hint}).")
        return 0
    for rank, r in enumerate(results, start=1):
        c = r.chunk
        rerank = f", rerank {r.rerank_score:.2f}" if r.rerank_score is not None else ""
        print(f"[{rank}] {c.source}, page {c.page_number}  (score {r.score:.3f}{rerank})")
        print(f"    {c.text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
