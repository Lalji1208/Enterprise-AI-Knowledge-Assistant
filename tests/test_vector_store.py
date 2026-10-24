"""Tests for Phase 5: vector store (rag/vector_store.py) and index sync (services/indexing.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import DIM, FakeModel, make_pdf

from config.settings import Settings
from models.chunk import Chunk
from rag.embedder import Embedder
from rag.vector_store import VectorStore, VectorStoreError
from services.indexing import sync_directory


def make_chunk(doc_id: str, i: int, text: str = "text", source: str | None = None) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}-p1-c{i}", doc_id=doc_id, source=source or f"{doc_id}.txt",
                 page_number=1, chunk_index=i, text=text, start_char=0, end_char=len(text))


def unit(*values: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


@pytest.fixture
def store() -> VectorStore:
    s = VectorStore(DIM, "fake-model")
    s.add(
        [make_chunk("a", 0), make_chunk("a", 1), make_chunk("b", 0)],
        np.stack([unit(1, 0), unit(0, 1), unit(1, 1)]),
    )
    return s


# ---------- VectorStore ----------

def test_search_orders_by_similarity(store: VectorStore) -> None:
    results = store.search(unit(1, 0.1), top_k=3)
    assert [r.chunk.chunk_id for r in results] == ["a-p1-c0", "b-p1-c0", "a-p1-c1"]
    assert results[0].score == pytest.approx(0.995, abs=1e-3)
    assert results[0].score > results[1].score > results[2].score


def test_search_top_k_larger_than_store_and_min_score(store: VectorStore) -> None:
    assert len(store.search(unit(1, 0), top_k=50)) == 3  # no -1 padding leaks out
    assert [r.chunk.chunk_id for r in store.search(unit(1, 0), top_k=3, min_score=0.5)] == [
        "a-p1-c0", "b-p1-c0"
    ]


def test_search_empty_store_returns_nothing() -> None:
    assert VectorStore(DIM, "m").search(unit(1), top_k=5) == []


def test_duplicate_chunk_ids_are_skipped(store: VectorStore) -> None:
    assert store.add([make_chunk("a", 0)], unit(1).reshape(1, -1)) == 0
    assert len(store) == 3


def test_remove_document(store: VectorStore) -> None:
    assert store.remove_document("a") == 2
    assert store.documents() == {"b": "b.txt"}
    assert [r.chunk.doc_id for r in store.search(unit(1, 0), top_k=5)] == ["b"]
    assert store.remove_document("missing") == 0
    # Removed chunk ids can be added again (e.g. a file restored).
    assert store.add([make_chunk("a", 0)], unit(1).reshape(1, -1)) == 1


@pytest.mark.parametrize("shape", [(2, DIM), (3, DIM + 1), (DIM,)])
def test_add_rejects_bad_shapes(store: VectorStore, shape: tuple[int, ...]) -> None:
    chunks = [make_chunk("c", i) for i in range(3)]
    with pytest.raises(VectorStoreError):
        store.add(chunks, np.zeros(shape, dtype=np.float32))


def test_search_rejects_bad_input(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(DIM + 1, dtype=np.float32))
    with pytest.raises(VectorStoreError):
        store.search(unit(1), top_k=0)


def test_save_and_load_roundtrip(store: VectorStore, tmp_path: Path) -> None:
    store.remove_document("b")  # make sure removals persist too
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, expected_model="fake-model")

    assert len(loaded) == 2
    assert loaded.documents() == {"a": "a.txt"}
    before = [(r.chunk, r.score) for r in store.search(unit(1, 0.3), top_k=5)]
    after = [(r.chunk, r.score) for r in loaded.search(unit(1, 0.3), top_k=5)]
    assert before == after
    # New ids continue after the old ones instead of colliding.
    loaded.add([make_chunk("z", 0)], unit(0, 0, 1).reshape(1, -1))
    assert len(loaded) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_load_unicode_path(store: VectorStore, tmp_path: Path) -> None:
    target = tmp_path / "índice-数据"
    store.save(target)
    assert len(VectorStore.load(target)) == 3


def test_load_rejects_model_mismatch(store: VectorStore, tmp_path: Path) -> None:
    store.save(tmp_path)
    with pytest.raises(VectorStoreError, match="rebuild"):
        VectorStore.load(tmp_path, expected_model="other-model")


def test_load_detects_out_of_sync_files(store: VectorStore, tmp_path: Path) -> None:
    store.save(tmp_path)
    meta = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
    meta["chunks"].pop()
    (tmp_path / "store.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorStoreError, match="out of sync"):
        VectorStore.load(tmp_path)


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(VectorStoreError):
        VectorStore.load(tmp_path / "nothing")


# ---------- sync_directory ----------

@pytest.fixture
def env(tmp_path: Path) -> tuple[Settings, Embedder, FakeModel, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    settings = Settings(documents_dir=docs, processed_dir=tmp_path / "processed",
                        chunk_size=200, chunk_overlap=20, _env_file=None)
    fake = FakeModel()
    return settings, Embedder("fake-model", model=fake), fake, docs


def embedded_texts(fake: FakeModel) -> list[str]:
    return [t for kind, texts, _ in fake.calls if kind == "document" for t in texts]


def test_sync_adds_skips_updates_and_removes(env) -> None:
    settings, embedder, fake, docs = env
    (docs / "leave.txt").write_text("Employees get twenty days of leave.", encoding="utf-8")
    (docs / "expenses.md").write_text("Submit receipts within 30 days.", encoding="utf-8")

    store, report = sync_directory(settings, embedder)
    assert sorted(report.added) == ["expenses.md", "leave.txt"]
    assert report.total_chunks == 2
    assert (settings.index_dir / "index.faiss").is_file()

    # Second run: nothing changed -> nothing re-embedded.
    fake.calls.clear()
    _, report = sync_directory(settings, embedder)
    assert sorted(report.unchanged) == ["expenses.md", "leave.txt"]
    assert fake.calls == []

    # Edit one file, delete the other.
    (docs / "leave.txt").write_text("Employees get twenty-five days of leave.", encoding="utf-8")
    (docs / "expenses.md").unlink()
    store, report = sync_directory(settings, embedder)
    assert report.added == ["leave.txt"]
    assert sorted(report.removed) == ["expenses.md", "leave.txt"]  # old leave.txt version too
    assert embedded_texts(fake) == ["Employees get twenty-five days of leave."]
    assert [c.chunk.text for c in store.search(embedder.embed_query("leave"), top_k=5)] == [
        "Employees get twenty-five days of leave."
    ]


def test_sync_keeps_chunks_of_files_that_fail_to_load(env) -> None:
    """A file that is still present but temporarily unreadable must not lose its
    chunks; a file that is really gone must."""
    settings, embedder, _, docs = env
    make_pdf(docs / "handbook.pdf", ["Holiday rules for all staff"])
    (docs / "memo.txt").write_text("Office closed on Friday.", encoding="utf-8")
    sync_directory(settings, embedder)

    (docs / "handbook.pdf").write_bytes(b"corrupted during copy")
    (docs / "memo.txt").unlink()
    store, report = sync_directory(settings, embedder)

    assert list(report.failed) == ["handbook.pdf"]
    assert report.removed == ["memo.txt"]
    assert set(store.documents().values()) == {"handbook.pdf"}


def test_rebuild_ignores_existing_index(env) -> None:
    settings, embedder, fake, docs = env
    (docs / "a.txt").write_text("Alpha.", encoding="utf-8")
    sync_directory(settings, embedder)
    fake.calls.clear()
    _, report = sync_directory(settings, embedder, rebuild=True)
    assert report.added == ["a.txt"]
    assert embedded_texts(fake) == ["Alpha."]


def test_existing_index_from_other_model_is_not_silently_wiped(env) -> None:
    settings, embedder, fake, docs = env
    (docs / "a.txt").write_text("Alpha.", encoding="utf-8")
    sync_directory(settings, embedder)
    with pytest.raises(VectorStoreError, match="rebuild"):
        sync_directory(settings, Embedder("another-model", model=fake))
