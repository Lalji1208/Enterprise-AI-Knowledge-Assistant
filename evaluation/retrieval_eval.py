"""Phase 13 - Retrieval evaluation.

Two steps, so threshold sweeps are cheap:
  1. collect_records(): for every question, run the expensive part once - embed,
     search, rerank - and record the candidate pool with both scores.
  2. score_records(): replay the production filtering (thresholds, top_k) on the
     recorded pools for any threshold combination, in milliseconds.

The replay matches Retriever exactly: the first stage keeps the best-scoring
candidates above `min_score` (a prefix of the score-sorted pool); with a
reranker, those are re-sorted by rerank score, filtered by `rerank_min_score`,
and cut to `top_k`.

Metrics:
  answerable questions   hit@k (a relevant chunk in the top k), MRR,
                         false abstention (nothing returned)
  unanswerable questions correct abstention (nothing returned)
  balanced score         mean of answerable hit@top_k and correct abstention, i.e.
                         equal weight to "finds answers" and "knows when to stay silent"
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from evaluation.dataset import EvalItem
from rag.embedder import Embedder
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.vector_store import VectorStore


@dataclass
class Candidate:
    chunk_id: str
    source: str
    page_number: int
    score: float
    rerank_score: float | None
    relevant: bool


@dataclass
class QuestionRecord:
    item_id: str
    split: str
    category: str
    answerable: bool
    candidates: list[Candidate]  # sorted by bi-encoder score, best first
    latency_ms: float


@dataclass
class RetrievalMetrics:
    n_answerable: int = 0
    n_unanswerable: int = 0
    hit_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    false_abstention: float = 0.0
    correct_abstention: float = 0.0
    balanced: float = 0.0
    mean_latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)  # ids answered wrongly

    def as_dict(self) -> dict:
        return {
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            **{f"hit@{k}": round(v, 4) for k, v in sorted(self.hit_at.items())},
            "mrr": round(self.mrr, 4),
            "false_abstention": round(self.false_abstention, 4),
            "correct_abstention": round(self.correct_abstention, 4),
            "balanced": round(self.balanced, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
        }


def collect_records(
    store: VectorStore,
    embedder: Embedder,
    items: Sequence[EvalItem],
    reranker: Reranker | None = None,
    pool_size: int = 20,
) -> list[QuestionRecord]:
    """Run retrieval with all thresholds disabled and record every candidate."""
    first_stage = Retriever(store, embedder, top_k=pool_size, min_score=None)
    records = []
    for item in items:
        start = time.perf_counter()
        pool = first_stage.retrieve(item.question)
        if reranker is not None:
            pool = reranker.rerank(item.question, pool)
            pool.sort(key=lambda r: r.score, reverse=True)  # keep bi-encoder order
        latency = (time.perf_counter() - start) * 1000
        records.append(QuestionRecord(
            item_id=item.id, split=item.split, category=item.category,
            answerable=item.answerable, latency_ms=latency,
            candidates=[Candidate(r.chunk.chunk_id, r.chunk.source, r.chunk.page_number,
                                  r.score, r.rerank_score, item.is_relevant(r.chunk))
                        for r in pool],
        ))
    return records


def replay(
    record: QuestionRecord,
    top_k: int,
    min_score: float | None,
    rerank_min_score: float | None = None,
    use_rerank: bool = False,
) -> list[Candidate]:
    """What the production retriever would return for this question."""
    kept = [c for c in record.candidates if min_score is None or c.score >= min_score]
    if use_rerank:
        kept = sorted(kept, key=lambda c: c.rerank_score, reverse=True)  # stable
        kept = [c for c in kept if rerank_min_score is None or c.rerank_score >= rerank_min_score]
    return kept[:top_k]


def score_records(
    records: Sequence[QuestionRecord],
    top_k: int = 5,
    min_score: float | None = None,
    rerank_min_score: float | None = None,
    use_rerank: bool = False,
    ks: Sequence[int] = (1, 3, 5),
) -> RetrievalMetrics:
    m = RetrievalMetrics()
    hits = {k: 0 for k in ks}
    reciprocal_ranks, false_abstain, correct_abstain = 0.0, 0, 0
    for rec in records:
        returned = replay(rec, top_k, min_score, rerank_min_score, use_rerank)
        if rec.answerable:
            m.n_answerable += 1
            rank = next((i for i, c in enumerate(returned, start=1) if c.relevant), None)
            for k in ks:
                hits[k] += rank is not None and rank <= k
            reciprocal_ranks += 1 / rank if rank else 0.0
            false_abstain += not returned
            if rank is None:
                m.failures.append(rec.item_id)
        else:
            m.n_unanswerable += 1
            correct_abstain += not returned
            if returned:
                m.failures.append(rec.item_id)
    if m.n_answerable:
        m.hit_at = {k: hits[k] / m.n_answerable for k in ks}
        m.mrr = reciprocal_ranks / m.n_answerable
        m.false_abstention = false_abstain / m.n_answerable
    if m.n_unanswerable:
        m.correct_abstention = correct_abstain / m.n_unanswerable
    found = m.hit_at.get(top_k, 0.0) if m.n_answerable else 1.0
    silent = m.correct_abstention if m.n_unanswerable else 1.0
    m.balanced = (found + silent) / 2
    m.mean_latency_ms = sum(r.latency_ms for r in records) / len(records) if records else 0.0
    return m


def sweep(
    records: Sequence[QuestionRecord],
    min_scores: Sequence[float | None],
    rerank_min_scores: Sequence[float | None] = (None,),
    use_rerank: bool = False,
    top_k: int = 5,
) -> list[tuple[float | None, float | None, RetrievalMetrics]]:
    """Score every threshold combination; best balanced score first.
    Ties prefer more lenient thresholds (fewer false abstentions)."""
    rows = [
        (s, r, score_records(records, top_k, s, r, use_rerank))
        for s in min_scores
        for r in (rerank_min_scores if use_rerank else (None,))
    ]

    def leniency(v: float | None) -> float:
        return float("-inf") if v is None else v

    rows.sort(key=lambda row: (-row[2].balanced, row[2].false_abstention,
                               leniency(row[0]), leniency(row[1])))
    return rows
