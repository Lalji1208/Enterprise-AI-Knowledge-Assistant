"""Application service: the one object the API talks to.

It owns the long-lived pieces (embedder, in-memory vector store, retriever) and
serialises everything that changes the documents folder or the index, so two
uploads can never sync at the same time. Searches run concurrently.

The LLM client is created on the first question only, so search, upload and the
UI keep working when no API key/credit is configured.
"""

import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Collection
from dataclasses import asdict
from pathlib import Path

from config.settings import Settings
from models.answer import Answer
from models.search import RetrievedChunk
from rag.embedder import Embedder
from rag.generator import AnswerGenerator
from rag.llm import LLMClient, build_llm
from rag.loader import DocumentLoadError, load_document
from rag.reranker import Reranker
from rag.retriever import Retriever
from services.indexing import SyncReport, load_or_create_store, sync_store

logger = logging.getLogger(__name__)

# Letters/digits from any language, space, dot, dash, underscore, parentheses.
_SAFE_NAME = re.compile(r"^[\w\-. ()]+$")
_MAX_NAME_LENGTH = 150


class UploadError(Exception):
    """Rejected upload. `status_code` is the HTTP status the API should return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class DocumentNotFoundError(Exception):
    pass


_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str) -> str:
    """Validate an uploaded/requested file name. It is never trusted as a path:
    anything with a directory part is rejected rather than "cleaned"."""
    name = (name or "").strip()
    if not name:
        raise UploadError("File name is empty.")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise UploadError("File name must not contain a path.")
    if (
        name.startswith(".")
        or name.endswith((".", " "))  # Windows silently strips these
        or not _SAFE_NAME.match(name)
        or len(name) > _MAX_NAME_LENGTH
        or Path(name).stem.upper() in _WINDOWS_RESERVED
    ):
        raise UploadError(
            "File name may only contain letters, digits, spaces and . - _ ( ), must not "
            f"start with a dot, and must be at most {_MAX_NAME_LENGTH} characters."
        )
    return name


class AssistantService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        reranker: Reranker | None = None,
        llm_factory: Callable[[], LLMClient] | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = load_or_create_store(settings, embedder)
        self.retriever = Retriever.for_store(self.store, embedder, settings, reranker)
        self._llm_factory = llm_factory or (lambda: build_llm(settings))
        self._generator: AnswerGenerator | None = None
        self._write_lock = threading.Lock()
        settings.documents_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> "AssistantService":
        from rag.embedder import get_embedder
        from rag.reranker import get_reranker

        return cls(settings, get_embedder(), get_reranker() if settings.reranking_enabled else None)

    def warm_up(self) -> None:
        """Load the ML models now, so the first request is not slow."""
        self.embedder.embed_query("warm up")
        if self.retriever.reranker is not None:
            self.retriever.reranker.model  # noqa: B018 - property access loads the model

    # ---------- status ----------

    def health(self) -> dict:
        s = self.settings
        return {
            "status": "ok",
            "documents": len(self.store.documents()),
            "chunks": len(self.store),
            "embedding_model": self.embedder.model_name,
            "reranking_enabled": self.retriever.reranker is not None,
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model,
            "llm_configured": s.llm_provider.lower() != "openai"
            or bool(s.openai_api_key.get_secret_value().strip()),
        }

    def list_documents(self) -> list[dict]:
        return self.store.document_stats()

    # ---------- changes to documents / index ----------

    def sync(self) -> SyncReport:
        with self._write_lock:
            return sync_store(self.store, self.settings, self.embedder)

    def upload(self, filename: str, data: bytes, replace: bool = False) -> SyncReport:
        """Validate, store and index one file. The documents folder is only touched
        after the file has been parsed successfully."""
        name = safe_filename(filename)
        suffix = Path(name).suffix.lower()
        if suffix not in self.settings.allowed_extension_set:
            allowed = ", ".join(sorted(self.settings.allowed_extension_set))
            raise UploadError(f"Unsupported file type '{suffix}'. Allowed: {allowed}.", 415)
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise UploadError(f"File is larger than {self.settings.max_upload_size_mb} MB.", 413)
        if not data:
            raise UploadError("File is empty.")

        target = self.settings.documents_dir / name
        with self._write_lock:
            if target.exists() and not replace:
                raise UploadError(f"'{name}' already exists. Upload with replace=true.", 409)
            tmp_dir = self.settings.processed_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
            try:
                tmp.write_bytes(data)
                try:
                    load_document(tmp, self.settings.allowed_extension_set)
                except DocumentLoadError as exc:
                    reason = str(exc).replace(tmp.name, name)
                    raise UploadError(f"Could not read '{name}': {reason}", 422) from exc
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            return sync_store(self.store, self.settings, self.embedder)

    def delete(self, source: str) -> SyncReport:
        name = safe_filename(source)
        with self._write_lock:
            path = self.settings.documents_dir / name
            if not path.is_file():
                raise DocumentNotFoundError(name)
            path.unlink()
            return sync_store(self.store, self.settings, self.embedder)

    # ---------- queries ----------

    def search(
        self, query: str, top_k: int | None = None, sources: Collection[str] | None = None
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=top_k, sources=sources)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        sources: Collection[str] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Answer:
        if self._generator is None:
            # Raises LLMError (e.g. missing key) without breaking the rest of the app.
            self._generator = AnswerGenerator(
                self.retriever, self._llm_factory(), self.settings.max_context_chars,
                self.settings.rewrite_follow_ups, self.settings.history_turns,
            )
        return self._generator.answer(question, history=history, top_k=top_k, sources=sources)


def report_to_dict(report: SyncReport) -> dict:
    return asdict(report)
