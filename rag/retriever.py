"""Phase 6 - Retrieval: question in, the most relevant distinct chunks out.

Pipeline:
  1. Embed the query.
  2. Fetch more candidates than needed from the vector store (over-fetching), so
     filtering and de-duplication can still fill `top_k` slots.
  3. Drop results below `min_score`, outside the requested sources, or that
     near-duplicate a better-scoring result; optionally cap chunks per document.
  4. If too few survive, fetch a larger candidate set and repeat.
  5. Optional (Phase 7): re-score ~20 candidates with a cross-encoder, drop those
     below `rerank_min_score`, and keep the best `top_k`.

Usage:
    from rag.retriever import Retriever
    retriever = Retriever.from_settings()
    results = retriever.retrieve("How many leave days do I get?")
"""

import re
from collections import Counter
from collections.abc import Collection

import numpy as np

from config.settings import Settings, get_settings
from models.search import RetrievedChunk
from rag.embedder import Embedder, get_embedder
from rag.reranker import Reranker, get_reranker
from rag.vector_store import VectorStore, VectorStoreError

OVERFETCH_FACTOR = 4
MAX_QUERY_CHARS = 2000
# Two chunks of the same page whose character ranges overlap by more than this
# share of the shorter one are treated as the same passage.
NEAR_DUPLICATE_OVERLAP = 0.5


class RetrievalError(Exception):
    """Raised for invalid queries or a missing index."""


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _is_near_duplicate(candidate: RetrievedChunk, kept: list[RetrievedChunk]) -> bool:
    c = candidate.chunk
    c_text = _normalized(c.text)
    for other in kept:
        o = other.chunk
        if _normalized(o.text) == c_text:  # identical boilerplate, any document
            return True
        if o.doc_id == c.doc_id and o.page_number == c.page_number:
            overlap = min(c.end_char, o.end_char) - max(c.start_char, o.start_char)
            shorter = min(c.end_char - c.start_char, o.end_char - o.start_char)
            if overlap > NEAR_DUPLICATE_OVERLAP * shorter:
                return True
    return False


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        min_score: float | None = None,
        max_per_document: int | None = None,
        reranker: Reranker | None = None,
        rerank_min_score: float | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        if store.model_name != embedder.model_name:
            raise RetrievalError(
                f"index built with '{store.model_name}' cannot be queried with "
                f"'{embedder.model_name}'"
            )
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.max_per_document = max_per_document
        self.reranker = reranker
        self.rerank_min_score = rerank_min_score
        self.rerank_candidates = rerank_candidates

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, embedder: Embedder | None = None
    ) -> "Retriever":
        """Load the saved index; TOP_K, SIMILARITY_THRESHOLD and RERANKING_* apply."""
        settings = settings or get_settings()
        embedder = embedder or get_embedder()
        try:
            store = VectorStore.load(settings.index_dir, expected_model=embedder.model_name)
        except VectorStoreError as exc:
            raise RetrievalError(
                f"{exc}. Build the index first: python scripts/build_index.py"
            ) from exc
        reranker = get_reranker() if settings.reranking_enabled else None
        return cls.for_store(store, embedder, settings, reranker)

    @classmethod
    def for_store(
        cls, store: VectorStore, embedder: Embedder, settings: Settings,
        reranker: Reranker | None = None,
    ) -> "Retriever":
        """Configure a retriever from settings around an already-loaded store."""
        first_stage = (settings.rerank_first_stage_min_score if reranker is not None
                       else settings.similarity_threshold)
        return cls(
            store,
            embedder,
            top_k=settings.top_k,
            min_score=first_stage,
            reranker=reranker,
            rerank_min_score=settings.rerank_min_score,
            rerank_candidates=settings.rerank_candidates,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        sources: Collection[str] | None = None,
        max_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Best matches first. Arguments left as None fall back to the instance defaults.

        `sources` restricts results to those file names (e.g. {"handbook.pdf"}).
        Returns [] when nothing clears `min_score` - callers should then answer
        "not found in the documents" rather than guess.
        """
        query = query.strip()
        if not query:
            raise RetrievalError("query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise RetrievalError(f"query is longer than {MAX_QUERY_CHARS} characters")
        top_k = top_k or self.top_k
        if top_k <= 0:
            raise RetrievalError(f"top_k must be > 0, got {top_k}")
        min_score = self.min_score if min_score is None else min_score
        per_doc = max_per_document or self.max_per_document
        allowed = set(sources) if sources else None

        if not len(self.store):
            return []
        query_vector = self.embedder.embed_query(query)

        if self.reranker is None:
            return self._first_stage(query_vector, top_k, min_score, allowed, per_doc)

        # The per-document cap is applied after reranking, on the final order.
        pool_size = max(top_k, self.rerank_candidates)
        candidates = self._first_stage(query_vector, pool_size, min_score, allowed, None)
        reranked = self.reranker.rerank(query, candidates, min_score=self.rerank_min_score)
        return _take(reranked, top_k, per_doc)

    def _first_stage(
        self,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float | None,
        allowed: set[str] | None,
        per_doc: int | None,
    ) -> list[RetrievedChunk]:
        """Bi-encoder search with filtering, de-duplication and adaptive over-fetching."""
        fetch_k = top_k * OVERFETCH_FACTOR
        while True:
            candidates = self.store.search(query_vector, top_k=fetch_k)
            kept: list[RetrievedChunk] = []
            per_doc_counts: Counter[str] = Counter()
            below_threshold = False
            for cand in candidates:
                if min_score is not None and cand.score < min_score:
                    below_threshold = True
                    break  # candidates are sorted, nothing better follows
                if allowed is not None and cand.chunk.source not in allowed:
                    continue
                if per_doc and per_doc_counts[cand.chunk.doc_id] >= per_doc:
                    continue
                if _is_near_duplicate(cand, kept):
                    continue
                kept.append(cand)
                per_doc_counts[cand.chunk.doc_id] += 1
                if len(kept) == top_k:
                    return kept
            # Not enough yet: stop if fetching more cannot help, else widen the net.
            if below_threshold or fetch_k >= len(self.store):
                return kept
            fetch_k *= OVERFETCH_FACTOR


def _take(results: list[RetrievedChunk], top_k: int, per_doc: int | None) -> list[RetrievedChunk]:
    """First `top_k` results, respecting an optional per-document cap."""
    if not per_doc:
        return results[:top_k]
    counts: Counter[str] = Counter()
    out = []
    for r in results:
        if counts[r.chunk.doc_id] < per_doc:
            out.append(r)
            counts[r.chunk.doc_id] += 1
            if len(out) == top_k:
                break
    return out
