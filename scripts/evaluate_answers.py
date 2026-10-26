"""Evaluate answer quality and hallucinations with the configured LLM (Phase 14).

Run from the project root. Calls the LLM once per question:
    python scripts/evaluate_answers.py               # built-in Example Corp benchmark
    python scripts/evaluate_answers.py --split test  # held-out questions only
    python scripts/evaluate_answers.py --docs data/documents \
        --dataset evaluation/datasets/research_paper_qa.jsonl --name research_paper
Writes evaluation/reports/<name>.md and .json (default name: answer_eval), with
retrieval metrics (embeddings / reranked / hybrid) and answer metrics.
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
from evaluation.dataset import DEFAULT_DATASET, load_dataset
from evaluation.experiments import REPORTS_DIR
from evaluation.retrieval_eval import collect_records, score_records
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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--docs", type=Path,
                        help="documents folder to evaluate on (default: the built-in corpus)")
    parser.add_argument("--name", default="answer_eval", help="report file name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    base = get_settings()
    try:
        llm = build_llm(base)
    except LLMError as exc:
        print(f"[ERROR] {exc}")
        return 1

    workspace = base.processed_dir / "eval" / args.name
    if args.docs:
        corpus = args.docs.resolve()  # read only: the index is built in the workspace
    else:
        corpus = workspace / "corpus"
        build_corpus(corpus)
    settings = base.model_copy(update={"documents_dir": corpus, "processed_dir": workspace})
    embedder = get_embedder()
    store, _ = sync_directory(settings, embedder, rebuild=True)
    reranker = get_reranker() if settings.reranking_enabled else None
    generator = AnswerGenerator(Retriever.for_store(store, embedder, settings, reranker), llm,
                                settings.max_context_chars, settings.rewrite_follow_ups,
                                settings.history_turns)

    items = load_dataset(args.dataset, split=args.split)
    retrieval_rows = []
    # Follow-up questions only make sense with their history (see the answer metrics).
    if reranker is not None and not any(i.history for i in items):
        records = collect_records(store, embedder, items, reranker,
                                  keyword_candidates=settings.keyword_candidates)
        for label, s, r, use, kw in [
            ("embeddings only", settings.similarity_threshold, None, False, False),
            ("+ reranker", settings.rerank_first_stage_min_score, settings.rerank_min_score,
             True, False),
            ("+ reranker + keywords (default)", settings.rerank_first_stage_min_score,
             settings.rerank_min_score, True, True),
        ]:
            m = score_records(records, settings.top_k, s, r, use, use_keyword=kw)
            retrieval_rows.append(
                f"| {label} | {m.hit_at[5]:.2f} | {m.mrr:.2f} | {m.correct_abstention:.2f} "
                f"| {m.balanced:.2f} | {', '.join(m.failures) or 'none'} |")
    print(f"Asking {len(items)} questions with {llm.model} ...")
    try:
        results = run_answer_eval(generator, items)
    except LLMError as exc:
        print(f"[ERROR] {exc}")
        return 1
    summary = summarize(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"{args.name}.json").write_text(
        json.dumps({"model": llm.model, "split": args.split or "all", "summary": summary,
                    "results": results_to_dicts(results)}, indent=1), encoding="utf-8")
    lines = [
        "# Answer evaluation", "",
        (f"Model `{llm.model}`, split: {args.split or 'all'}, {summary['questions']} questions, "
        f"reranking {'on' if reranker else 'off'}. Documents: `{corpus.name}`, "
        f"dataset: `{args.dataset.name}`."), "",
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
    if retrieval_rows:
        failures_at = lines.index("## Failures")
        lines[failures_at:failures_at] = [
            "## Retrieval", "",
            "| retrieval | hit@5 | MRR | unanswerable rejected | balanced | wrong |",
            "|---|---|---|---|---|---|", *retrieval_rows, "",
        ]
    by_id = {r.item_id: r for r in results}
    for item_id in summary["failures"]:
        r = by_id[item_id]
        lines.append(f"- **{item_id}** ({r.status}): {r.answer[:200]}")
    if not summary["failures"]:
        lines.append("None.")
    (REPORTS_DIR / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines[4:14]:  # the metrics table
        print(line)
    print("\n".join(retrieval_rows))
    print(f"\nReport: {REPORTS_DIR / f'{args.name}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
