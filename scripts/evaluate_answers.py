"""Evaluate answer quality and hallucinations with the configured LLM (Phase 14).

Run from the project root. Calls the LLM once per question (56 questions with the
default dataset; with gpt-4o-mini this costs roughly 1-2 US cents in total):
    python scripts/evaluate_answers.py               # all questions
    python scripts/evaluate_answers.py --split test  # held-out questions only
Writes evaluation/reports/answer_eval.md and .json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from evaluation.answer_eval import results_to_dicts, run_answer_eval, summarize
from evaluation.corpus_builder import build_corpus
from evaluation.dataset import load_dataset
from evaluation.experiments import REPORTS_DIR
from rag.embedder import get_embedder
from rag.generator import AnswerGenerator
from rag.llm import LLMError, build_llm
from rag.reranker import get_reranker
from rag.retriever import Retriever
from services.indexing import sync_directory


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    base = get_settings()
    try:
        llm = build_llm(base)
    except LLMError as exc:
        print(f"[ERROR] {exc}")
        return 1

    workspace = base.processed_dir / "eval" / "answers"
    corpus = workspace / "corpus"
    build_corpus(corpus)
    settings = base.model_copy(update={"documents_dir": corpus, "processed_dir": workspace})
    embedder = get_embedder()
    store, _ = sync_directory(settings, embedder, rebuild=True)
    reranker = get_reranker() if settings.reranking_enabled else None
    generator = AnswerGenerator(Retriever.for_store(store, embedder, settings, reranker), llm,
                                settings.max_context_chars)

    items = load_dataset(split=args.split)
    print(f"Asking {len(items)} questions with {llm.model} ...")
    try:
        results = run_answer_eval(generator, items)
    except LLMError as exc:
        print(f"[ERROR] {exc}")
        return 1
    summary = summarize(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "answer_eval.json").write_text(
        json.dumps({"model": llm.model, "split": args.split or "all", "summary": summary,
                    "results": results_to_dicts(results)}, indent=1), encoding="utf-8")
    lines = [
        "# Answer evaluation", "",
        (f"Model `{llm.model}`, split: {args.split or 'all'}, {summary['questions']} questions, "
        f"reranking {'on' if reranker else 'off'}."), "",
        "| metric | value |", "|---|---|",
        f"| status accuracy (answer vs. 'not found') | {pct(summary['status_accuracy'])} |",
        f"| answer accuracy (answerable, facts correct) | {pct(summary['answer_accuracy'])} |",
        f"| false refusal rate | {pct(summary['false_refusal_rate'])} |",
        (f"| **hallucination rate** (unanswerable answered) | "
        f"**{pct(summary['hallucination_rate'])}** |"),
        f"| citation precision | {pct(summary['citation_precision'])} |",
        f"| uncited answers | {pct(summary['uncited_answer_rate'])} |",
        f"| prompt-injection resistance | {pct(summary['injection_pass_rate'])} |",
        (f"| tokens (prompt / completion) | {summary['prompt_tokens']} / "
        f"{summary['completion_tokens']} |"),
        "", "## Failures", "",
    ]
    by_id = {r.item_id: r for r in results}
    for item_id in summary["failures"]:
        r = by_id[item_id]
        lines.append(f"- **{item_id}** ({r.status}): {r.answer[:200]}")
    if not summary["failures"]:
        lines.append("None.")
    (REPORTS_DIR / "answer_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines[4:14]:  # the metrics table
        print(line)
    print(f"\nReport: {REPORTS_DIR / 'answer_eval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
