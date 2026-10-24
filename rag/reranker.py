"""Phase 7 - Reranking: re-score retrieved chunks with a cross-encoder.

The bi-encoder (rag/embedder.py) embeds query and chunk separately, which is fast
but judges only topical similarity. A cross-encoder reads the query and the chunk
together, so it can tell "about the same topic" from "answers the question".
It is too slow to run over the whole index, so it re-scores the first-stage
candidates only.

Scores are the model's raw logits (unbounded; higher = more relevant). For
cross-encoder/ms-marco-MiniLM-L-6-v2, answering passages scored about -3..+3 and
unanswerable/off-topic ones below -7 in our probes, hence RERANK_MIN_SCORE=-6.0.

Usage:
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranked = reranker.rerank("How many leave days?", candidates, min_score=-6.0)
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from models.search import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
        model: Any = None,  # inject a fake model in tests
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Return candidates sorted by cross-encoder score, with `rerank_score` set.

        The original bi-encoder `score` is kept. Ties keep first-stage order.
        """
        if not candidates:
            return []
        scores = self.model.predict(
            [(query, c.chunk.text) for c in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        rescored = [
            c.model_copy(update={"rerank_score": float(s)}) for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda r: r.rerank_score, reverse=True)  # stable sort
        if min_score is not None:
            rescored = [r for r in rescored if r.rerank_score >= min_score]
        return rescored


@lru_cache
def get_reranker() -> Reranker:
    from config.settings import get_settings

    return Reranker(get_settings().reranker_model)
