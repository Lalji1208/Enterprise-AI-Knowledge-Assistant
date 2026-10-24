"""Phase 5 - Vector database: a persistent FAISS index plus chunk metadata.

FAISS stores only vectors and int64 ids; this class keeps the matching Chunk for
every id and saves both side by side:

    <index_dir>/index.faiss   FAISS index (exact inner-product search)
    <index_dir>/store.json    chunks, embedding model name, dimension

Vectors must be L2-normalised (rag.embedder does this), so scores are cosine
similarities.

Usage:
    store = VectorStore(dimension=384, model_name="sentence-transformers/all-MiniLM-L6-v2")
    store.add(chunks, vectors)
    results = store.search(query_vector, top_k=5)
    store.save(index_dir);  store = VectorStore.load(index_dir)
"""

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from models.chunk import Chunk
from models.search import RetrievedChunk

INDEX_FILE = "index.faiss"
META_FILE = "store.json"
FORMAT_VERSION = 1


class VectorStoreError(Exception):
    """Raised for invalid input, missing files, or an index/model mismatch."""


class VectorStore:
    def __init__(self, dimension: int, model_name: str) -> None:
        if dimension <= 0:
            raise VectorStoreError(f"dimension must be > 0, got {dimension}")
        self.dimension = dimension
        self.model_name = model_name
        # IDMap2 lets us delete a document's vectors without rebuilding everything.
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._chunks: dict[int, Chunk] = {}
        self._ids_by_chunk_id: dict[str, int] = {}
        self._next_id = 0
        # FAISS is not safe for concurrent writes; the API (Phase 10) may call us from threads.
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._chunks)

    # ---------- write ----------

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Add chunks with their vectors (row i <-> chunks[i]).

        Chunks whose chunk_id is already stored are skipped. Returns the number added.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(chunks), self.dimension):
            raise VectorStoreError(
                f"expected vectors of shape ({len(chunks)}, {self.dimension}), got {vectors.shape}"
            )
        with self._lock:
            keep = [i for i, c in enumerate(chunks) if c.chunk_id not in self._ids_by_chunk_id]
            if not keep:
                return 0
            ids = np.arange(self._next_id, self._next_id + len(keep), dtype=np.int64)
            self._index.add_with_ids(np.ascontiguousarray(vectors[keep]), ids)
            for int_id, i in zip(ids.tolist(), keep):
                self._chunks[int_id] = chunks[i]
                self._ids_by_chunk_id[chunks[i].chunk_id] = int_id
            self._next_id += len(keep)
            return len(keep)

    def remove_document(self, doc_id: str) -> int:
        """Delete every chunk of one document. Returns the number removed."""
        with self._lock:
            ids = [i for i, c in self._chunks.items() if c.doc_id == doc_id]
            if ids:
                self._index.remove_ids(np.array(ids, dtype=np.int64))
                for i in ids:
                    del self._ids_by_chunk_id[self._chunks.pop(i).chunk_id]
            return len(ids)

    # ---------- read ----------

    def documents(self) -> dict[str, str]:
        """{doc_id: source file name} for every indexed document."""
        with self._lock:
            return {c.doc_id: c.source for c in self._chunks.values()}

    def document_stats(self) -> list[dict]:
        """Per document: doc_id, source, chunk count and number of pages with text."""
        with self._lock:
            stats: dict[str, dict] = {}
            for c in self._chunks.values():
                entry = stats.setdefault(
                    c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0, "pages": set()}
                )
                entry["chunks"] += 1
                entry["pages"].add(c.page_number)
        return sorted(
            ({**e, "pages": len(e["pages"])} for e in stats.values()),
            key=lambda e: e["source"].lower(),
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, min_score: float | None = None
    ) -> list[RetrievedChunk]:
        """Most similar chunks first. `min_score` drops weak matches."""
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be > 0, got {top_k}")
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError(
                f"query has dimension {query.shape[1]}, index expects {self.dimension}"
            )
        with self._lock:
            if not self._chunks:
                return []
            scores, ids = self._index.search(query, min(top_k, len(self._chunks)))
            results = [
                RetrievedChunk(chunk=self._chunks[int(i)], score=float(s))
                for s, i in zip(scores[0], ids[0])
                if i != -1  # FAISS pads missing results with -1
            ]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    # ---------- persistence ----------

    def save(self, directory: str | Path) -> None:
        """Write index + metadata. Each file is replaced atomically, so a crash
        mid-save never leaves a half-written file behind."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # serialize_index + Python I/O avoids FAISS's trouble with non-ASCII Windows paths.
            index_bytes = faiss.serialize_index(self._index).tobytes()
            meta = {
                "format_version": FORMAT_VERSION,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "next_id": self._next_id,
                "chunks": [
                    {"id": i, **c.model_dump(mode="json")} for i, c in self._chunks.items()
                ],
            }
        _atomic_write(directory / INDEX_FILE, index_bytes)
        _atomic_write(directory / META_FILE, json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    @classmethod
    def load(cls, directory: str | Path, expected_model: str | None = None) -> "VectorStore":
        """Load a saved store. If `expected_model` is given and differs from the model
        that built the index, raise - vectors from different models are not comparable."""
        directory = Path(directory)
        index_path, meta_path = directory / INDEX_FILE, directory / META_FILE
        if not (index_path.is_file() and meta_path.is_file()):
            raise VectorStoreError(f"no saved index in {directory}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise VectorStoreError(f"unsupported index format: {meta.get('format_version')}")
        if expected_model and meta["model_name"] != expected_model:
            raise VectorStoreError(
                f"index was built with '{meta['model_name']}' but EMBEDDING_MODEL is "
                f"'{expected_model}'; rebuild it with: python scripts/build_index.py --rebuild"
            )

        store = cls(meta["dimension"], meta["model_name"])
        raw = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        store._index = faiss.deserialize_index(raw)
        for record in meta["chunks"]:
            int_id = record.pop("id")
            chunk = Chunk.model_validate(record)
            store._chunks[int_id] = chunk
            store._ids_by_chunk_id[chunk.chunk_id] = int_id
        store._next_id = meta["next_id"]

        if store._index.ntotal != len(store._chunks) or store._index.d != store.dimension:
            raise VectorStoreError(
                f"index and metadata in {directory} are out of sync; rebuild with --rebuild"
            )
        return store


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
