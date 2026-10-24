"""Ask a question and get a cited answer from the indexed documents.

Run from the project root (build the index first with build_index.py):
    python scripts/ask.py "How many leave days do I get?"
    python scripts/ask.py "What is the expense deadline?" --source expenses.pdf --show-sources
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.generator import AnswerGenerator
from rag.llm import LLMError
from rag.retriever import RetrievalError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--source", action="append", help="limit to a file (repeatable)")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--show-sources", action="store_true", help="print cited passages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    try:
        generator = AnswerGenerator.from_settings()
        answer = generator.answer(args.question, sources=args.source, top_k=args.top_k)
    except (RetrievalError, LLMError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"\n{answer.answer}\n")
    for warning in answer.warnings:
        print(f"[WARNING] {warning}")
    if answer.citations:
        print("Sources:")
        for c in answer.citations:
            print(f"  [{c.number}] {c.source}, page {c.page_number}")
            if args.show_sources:
                print(f"      \"{c.text}\"")
    if answer.prompt_tokens is not None:
        print(f"\n(model: {answer.model}, tokens: {answer.prompt_tokens} in / "
              f"{answer.completion_tokens} out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
