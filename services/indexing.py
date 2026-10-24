"""Keep the vector index in sync with the documents folder.

A document is identified by its content hash (doc_id), so:
  - new file / edited file  -> new doc_id  -> cleaned, chunked, embedded, added
  - unchanged file          -> same doc_id -> skipped (no re-embedding)
  - deleted file / old version of an edited file -> its chunks are removed
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings
from models.document import Document
from rag.chunker import chunk_document
from rag.cleaner import clean_document
from rag.embedder import Embedder
from rag.loader import load_directory
from rag.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)      # source names
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)      # no extractable text
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0
    chunks_removed: int = 0
    total_chunks: int = 0


def load_or_create_store(settings: Settings, embedder: Embedder, rebuild: bool = False) -> VectorStore:
    """Load the saved index, or start an empty one if none exists (or `rebuild`)."""
    path = settings.index_dir
    if not rebuild:
        try:
            return VectorStore.load(path, expected_model=embedder.model_name)
        except VectorStoreError as exc:
            if (path / "store.json").exists():
                raise  # an index exists but is unusable: surface it, don't silently wipe it
            logger.info("No existing index (%s); starting a new one", exc)
    return VectorStore(embedder.dimension, embedder.model_name)


def sync_documents(
    store: VectorStore,
    documents: list[Document],
    embedder: Embedder,
    settings: Settings,
    protected_sources: frozenset[str] = frozenset(),
) -> SyncReport:
    """Make `store` match `documents`. Does not save; the caller decides when.

    Indexed documents whose source is in `protected_sources` (files that exist but
    failed to load this time) are kept rather than treated as deleted.
    """
    report = SyncReport()
    indexed = store.documents()  # doc_id -> source
    current_ids = {d.doc_id for d in documents}

    for doc_id, source in indexed.items():
        if doc_id not in current_ids and source not in protected_sources:
            report.chunks_removed += store.remove_document(doc_id)
            report.removed.append(source)

    for doc in documents:
        if doc.doc_id in indexed:
            report.unchanged.append(doc.source)
            continue
        if doc.is_empty:
            report.empty.append(doc.source)
            continue
        chunks = chunk_document(clean_document(doc), settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            report.empty.append(doc.source)
            continue
        vectors = embedder.embed_chunks(chunks, show_progress=len(chunks) > 64)
        report.chunks_added += store.add(chunks, vectors)
        report.added.append(doc.source)

    report.total_chunks = len(store)
    return report


def sync_store(
    store: VectorStore, settings: Settings, embedder: Embedder, directory: Path | None = None
) -> SyncReport:
    """Sync an already-loaded store with the documents folder and save it."""
    documents, failures = load_directory(directory or settings.documents_dir,
                                         settings.allowed_extension_set)
    report = sync_documents(store, documents, embedder, settings,
                            protected_sources=frozenset(failures))
    report.failed = failures
    store.save(settings.index_dir)
    return report


def sync_directory(
    settings: Settings, embedder: Embedder, rebuild: bool = False, directory: Path | None = None
) -> tuple[VectorStore, SyncReport]:
    """Load (or create) the saved index, sync it with the documents folder, and save."""
    store = load_or_create_store(settings, embedder, rebuild=rebuild)
    return store, sync_store(store, settings, embedder, directory)
