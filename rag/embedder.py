"""Phase 4 - Embeddings: turn chunks and queries into dense vectors.

Vectors are L2-normalised float32, so an inner product equals cosine similarity
(this is what the FAISS index in Phase 5 relies on).

Queries and documents go through `encode_query` / `encode_document`, which apply
model-specific prompts (e.g. "query: " / "passage: " for E5 models) when the
model defines them, and do nothing extra for models like all-MiniLM-L6-v2.

Usage:
    from rag.embedder import get_embedder
    embedder = get_embedder()
    vectors = embedder.embed_chunks(chunks)      # shape (n, dim)
    query_vec = embedder.embed_query("How many leave days?")  # shape (dim,)
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from models.chunk import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around a sentence-transformers model, loaded on first use."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: str | None = None,
        model: Any = None,  # inject a pre-built/fake model (used by tests)
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            # Imported here so importing this module stays fast (torch is heavy).
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        dim = self.model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"{self.model_name} does not report an embedding dimension")
        return int(dim)

    def _encode(self, method: str, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = getattr(self.model, method)(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed passages. Returns an (n, dim) float32 array of unit vectors."""
        return self._encode("encode_document", texts, show_progress)

    def embed_chunks(self, chunks: Sequence[Chunk], show_progress: bool = False) -> np.ndarray:
        """Embed chunks; row i corresponds to chunks[i]."""
        return self.embed_documents([c.text for c in chunks], show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query. Returns a (dim,) float32 unit vector."""
        if not query.strip():
            raise ValueError("query must not be empty")
        return self._encode("encode_query", [query], show_progress=False)[0]


@lru_cache
def get_embedder() -> Embedder:
    """Shared Embedder built from settings (model is loaded once per process)."""
    from config.settings import get_settings

    return Embedder(get_settings().embedding_model)
