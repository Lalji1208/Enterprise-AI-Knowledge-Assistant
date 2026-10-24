"""Phase 15 - Retrieval experiments: chunking configurations and threshold tuning.

Protocol (to avoid reporting numbers tuned on the same questions):
  - thresholds are tuned on the `dev` split only;
  - all headline numbers are reported on the held-out `test` split.

Usage: python scripts/run_experiments.py   (writes evaluation/reports/)
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from config.settings import Settings, get_settings
from evaluation.corpus_builder import build_corpus
from evaluation.dataset import EvalItem, load_dataset
from evaluation.retrieval_eval import (
    QuestionRecord,
    collect_records,
    score_records,
    sweep,
)
from rag.embedder import Embedder
from rag.reranker import Reranker
from services.indexing import sync_directory

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
CHUNK_CONFIGS = [(300, 30), (500, 0), (500, 50), (500, 100), (800, 80)]
TUNE_CONFIG = (500, 50)  # the production default
BI_THRESHOLDS = [None, *(round(0.05 * i, 2) for i in range(13))]          # 0.00 .. 0.60
FIRST_STAGE_THRESHOLDS = [None, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
RERANK_THRESHOLDS = [None, *(round(-10 + 0.5 * i, 1) for i in range(25))]  # -10 .. 2
POOL_SIZE = 20
TOP_K = 5


def _fmt(v: float | None) -> str:
    return "off" if v is None else f"{v:g}"


def run(
    workspace: Path,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> dict:
    """Run all experiments. Models default to the configured ones (tests inject fakes)."""
    base = settings or get_settings()
    items = load_dataset()
    dev = [i for i in items if i.split == "dev"]
    test = [i for i in items if i.split == "test"]
    corpus = workspace / "corpus"
    build_corpus(corpus)

    embedder = embedder or Embedder(base.embedding_model)
    reranker = reranker or Reranker(base.reranker_model)

    records: dict[tuple[int, int], list[QuestionRecord]] = {}
    build_seconds: dict[tuple[int, int], float] = {}
    for size, overlap in CHUNK_CONFIGS:
        cfg = base.model_copy(update={
            "documents_dir": corpus, "processed_dir": workspace / f"cs{size}_ov{overlap}",
            "chunk_size": size, "chunk_overlap": overlap,
        })
        start = time.perf_counter()
        store, report = sync_directory(cfg, embedder, rebuild=True)
        build_seconds[(size, overlap)] = time.perf_counter() - start
        logger.info("chunks %s/%s: %d chunks", size, overlap, report.total_chunks)
        records[(size, overlap)] = collect_records(store, embedder, items, reranker, POOL_SIZE)

    def split_of(recs: list[QuestionRecord], part: list[EvalItem]) -> list[QuestionRecord]:
        ids = {i.id for i in part}
        return [r for r in recs if r.item_id in ids]

    defaults = {"bi": base.similarity_threshold, "rerank": base.rerank_min_score,
                "rerank_first_stage": base.rerank_first_stage_min_score}

    # --- 1. chunking comparison (test split, current default thresholds) ---
    chunking = []
    for key, recs in records.items():
        t = split_of(recs, test)
        chunking.append({
            "chunk_size": key[0], "chunk_overlap": key[1],
            "chunks": len({c.chunk_id for r in recs for c in r.candidates}),
            "index_build_s": round(build_seconds[key], 1),
            "no_thresholds": score_records(t, TOP_K).as_dict(),
            "bi_default": score_records(t, TOP_K, defaults["bi"]).as_dict(),
            "rerank_default": score_records(t, TOP_K, defaults["rerank_first_stage"],
                                            defaults["rerank"], use_rerank=True).as_dict(),
        })

    # --- 2. threshold tuning on dev, evaluated on test ---
    recs = records[TUNE_CONFIG]
    dev_recs, test_recs = split_of(recs, dev), split_of(recs, test)
    bi_best_s, _, _bi_dev = sweep(dev_recs, BI_THRESHOLDS, top_k=TOP_K)[0]
    rr_best_s, rr_best_r, _rr_dev = sweep(dev_recs, FIRST_STAGE_THRESHOLDS, RERANK_THRESHOLDS,
                                         use_rerank=True, top_k=TOP_K)[0]
    # First-stage cutoff tuned on dev with the rerank cutoff held at its default.
    fs_best_s, _, _fs_dev = sweep(dev_recs, FIRST_STAGE_THRESHOLDS, [defaults["rerank"]],
                                  use_rerank=True, top_k=TOP_K)[0]
    variants = {
        "bi_default": (defaults["bi"], None, False),
        "bi_tuned": (bi_best_s, None, False),
        "rerank_same_first_stage": (defaults["bi"], defaults["rerank"], True),
        "rerank_default": (defaults["rerank_first_stage"], defaults["rerank"], True),
        "rerank_tuned_first_stage": (fs_best_s, defaults["rerank"], True),
        "rerank_tuned_both": (rr_best_s, rr_best_r, True),
    }
    tuning = {
        name: {
            "min_score": s, "rerank_min_score": r, "use_rerank": use,
            "dev": score_records(dev_recs, TOP_K, s, r, use).as_dict(),
            "test": score_records(test_recs, TOP_K, s, r, use).as_dict(),
            "test_failures": score_records(test_recs, TOP_K, s, r, use).failures,
        }
        for name, (s, r, use) in variants.items()
    }

    result = {
        "dataset": {"items": len(items), "dev": len(dev), "test": len(test),
                    "answerable": sum(i.answerable for i in items)},
        "models": {"embedding": base.embedding_model, "reranker": base.reranker_model},
        "defaults": defaults, "top_k": TOP_K, "pool_size": POOL_SIZE,
        "chunking": chunking, "tuning": tuning,
        "records": {f"{k[0]}_{k[1]}": [asdict(r) for r in v] for k, v in records.items()},
    }
    return result


def write_reports(result: dict, out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "retrieval_experiments.json"
    summary = {k: v for k, v in result.items() if k != "records"}
    json_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    # Per-question candidate lists (~1 MB) are for diagnosis only; git-ignored.
    (out_dir / "retrieval_experiments_records.json").write_text(
        json.dumps(result["records"]), encoding="utf-8")

    d, t = result["dataset"], result["tuning"]
    lines = [
        "# Retrieval experiments",
        "",
        (f"Dataset: {d['items']} questions ({d['answerable']} answerable), "
        f"dev {d['dev']} / test {d['test']}. Embeddings `{result['models']['embedding']}`, "
        f"reranker `{result['models']['reranker']}`, top_k={result['top_k']}."),
        "",
        ("Metrics: **hit@k** = share of answerable questions with a relevant chunk in the top k; "
        "**MRR** = mean reciprocal rank of the first relevant chunk; **abstain** = share of "
        "unanswerable questions for which nothing is returned; **balanced** = mean of hit@5 "
        "and abstain. Thresholds were tuned on *dev*; all tables show *test* unless noted."),
        "",
        "## 1. Chunking (test split)",
        "",
        ("| chunk / overlap | chunks | hit@1 | hit@5 | MRR | hit@5 bi-default | abstain bi-default "
        "| hit@5 rerank-default | abstain rerank-default | balanced rerank-default |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["chunking"]:
        n, b, r = row["no_thresholds"], row["bi_default"], row["rerank_default"]
        lines.append(
            f"| {row['chunk_size']} / {row['chunk_overlap']} | {row['chunks']} | {n['hit@1']:.2f} "
            f"| {n['hit@5']:.2f} | {n['mrr']:.2f} | {b['hit@5']:.2f} | {b['correct_abstention']:.2f} "
            f"| {r['hit@5']:.2f} | {r['correct_abstention']:.2f} | {r['balanced']:.2f} |"
        )
    lines += [
        "",
        "*no thresholds* columns (hit@1, hit@5, MRR) measure pure ranking quality.",
        "",
        f"## 2. Thresholds (chunk {TUNE_CONFIG[0]} / overlap {TUNE_CONFIG[1]})",
        "",
        ("| variant | min_score | rerank_min_score | dev balanced | test hit@5 | test false abstain "
        "| test abstain | **test balanced** | test MRR | latency ms |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, v in t.items():
        lines.append(
            f"| {name} | {_fmt(v['min_score'])} | {_fmt(v['rerank_min_score'])} "
            f"| {v['dev']['balanced']:.2f} | {v['test']['hit@5']:.2f} "
            f"| {v['test']['false_abstention']:.2f} | {v['test']['correct_abstention']:.2f} "
            f"| **{v['test']['balanced']:.2f}** | {v['test']['mrr']:.2f} "
            f"| {v['test']['mean_latency_ms']:.0f} |"
        )
    lines += ["", "Test questions still handled wrongly:", ""]
    lines += [f"- {name}: {', '.join(v['test_failures']) or 'none'}" for name, v in t.items()]
    lines += ["", ("Latency includes the reranker for every row (it was recorded once per "
              "question); bi-encoder-only retrieval alone is far faster."), ""]
    md_path = out_dir / "retrieval_experiments.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
