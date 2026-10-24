"""Tests for Phase 4: embeddings (rag/embedder.py).

Unit tests use a fake model (no download). The `integration` test loads the real
model from EMBEDDING_MODEL; run it with:  pytest -m integration
"""

import numpy as np
import pytest
from conftest import DIM, FakeModel

from models.chunk import Chunk
from rag.embedder import Embedder


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d-p1-c{i}", doc_id="d", source="s.txt", page_number=1,
                 chunk_index=i, text=text, start_char=0, end_char=len(text))


@pytest.fixture
def fake() -> FakeModel:
    return FakeModel()


@pytest.fixture
def embedder(fake: FakeModel) -> Embedder:
    return Embedder("fake-model", batch_size=4, model=fake)


def test_embed_documents_shape_dtype_and_norm(embedder: Embedder) -> None:
    vecs = embedder.embed_documents(["annual leave policy", "expense claims"])
    assert vecs.shape == (2, DIM)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)


def test_embed_chunks_preserves_order(embedder: Embedder, fake: FakeModel) -> None:
    chunks = [make_chunk(0, "first text"), make_chunk(1, "second text")]
    embedder.embed_chunks(chunks)
    assert fake.calls[-1][1] == ["first text", "second text"]


def test_query_and_documents_use_their_own_encoders(embedder: Embedder, fake: FakeModel) -> None:
    embedder.embed_documents(["doc"])
    vec = embedder.embed_query("question")
    assert [c[0] for c in fake.calls] == ["document", "query"]
    assert vec.shape == (DIM,)
    assert fake.calls[-1][2]["batch_size"] == 4
    assert fake.calls[-1][2]["normalize_embeddings"] is True


def test_empty_inputs(embedder: Embedder, fake: FakeModel) -> None:
    assert embedder.embed_documents([]).shape == (0, DIM)
    assert fake.calls == []  # model not called for nothing
    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_model_is_not_loaded_until_needed() -> None:
    e = Embedder("does-not-exist/never-downloaded")
    assert e._model is None  # constructing must not touch the network


@pytest.mark.integration
def test_real_model_ranks_relevant_passage_first() -> None:
    from config.settings import get_settings

    embedder = Embedder(get_settings().embedding_model)
    passages = [
        "Employees receive twenty days of paid annual leave each year.",
        "Expense reports must include original receipts.",
        "The cafeteria opens at 8 a.m. on weekdays.",
    ]
    doc_vecs = embedder.embed_documents(passages)
    query_vec = embedder.embed_query("How many vacation days do I get?")

    assert doc_vecs.shape == (3, embedder.dimension)
    np.testing.assert_allclose(np.linalg.norm(doc_vecs, axis=1), 1.0, rtol=1e-4)
    scores = doc_vecs @ query_vec  # cosine similarity because vectors are unit length
    assert int(np.argmax(scores)) == 0
