"""Tests for Phase 6: retrieval (rag/retriever.py)."""

from pathlib import Path

import numpy as np
import pytest
from conftest import DIM, FakeModel

from config.settings import Settings
from models.chunk import Chunk
from models.search import RetrievedChunk
from rag.embedder import Embedder
from rag.retriever import RetrievalError, Retriever
from rag.vector_store import VectorStore
from services.indexing import sync_directory


class StubEmbedder:
    """Returns a fixed query vector, so tests control scores exactly."""

    model_name = "stub"

    def __init__(self) -> None:
        self.query_vector = np.zeros(DIM, dtype=np.float32)
        self.calls = 0

    def embed_query(self, query: str) -> np.ndarray:
        self.calls += 1
        return self.query_vector


def vec(score: float) -> np.ndarray:
    """Unit vector whose cosine with the query e0 is exactly `score`."""
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[1] = score, np.sqrt(1 - score**2)
    return v


def chunk(doc: str, i: int, text: str, page: int = 1, span: tuple[int, int] | None = None,
          source: str | None = None) -> Chunk:
    start, end = span or (i * 1000, i * 1000 + len(text))
    return Chunk(chunk_id=f"{doc}-p{page}-c{i}", doc_id=doc, source=source or f"{doc}.pdf",
                 page_number=page, chunk_index=i, text=text, start_char=start, end_char=end)


def build(entries: list[tuple[Chunk, float]], **kwargs) -> tuple[Retriever, StubEmbedder]:
    store = VectorStore(DIM, "stub")
    store.add([c for c, _ in entries], np.stack([vec(s) for _, s in entries]))
    embedder = StubEmbedder()
    embedder.query_vector[0] = 1.0
    return Retriever(store, embedder, **kwargs), embedder


def ids(results) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def test_returns_top_k_in_score_order() -> None:
    r, _ = build([(chunk("a", i, f"text {i}"), s) for i, s in enumerate([0.2, 0.9, 0.5, 0.7])],
                 top_k=3)
    results = r.retrieve("q")
    assert ids(results) == ["a-p1-c1", "a-p1-c3", "a-p1-c2"]
    assert [round(x.score, 3) for x in results] == [0.9, 0.7, 0.5]


def test_min_score_filters_and_can_return_nothing() -> None:
    r, _ = build([(chunk("a", 0, "x"), 0.8), (chunk("a", 1, "y"), 0.25)], min_score=0.3)
    assert ids(r.retrieve("q")) == ["a-p1-c0"]
    assert r.retrieve("q", min_score=0.95) == []


def test_identical_text_across_documents_is_deduplicated() -> None:
    disclaimer = "This document is confidential. Do not distribute."
    r, _ = build([(chunk("a", 0, disclaimer), 0.9), (chunk("b", 0, disclaimer.upper()), 0.85),
                  (chunk("c", 0, "Real answer"), 0.6)], top_k=3)
    assert ids(r.retrieve("q")) == ["a-p1-c0", "c-p1-c0"]


def test_overlapping_spans_on_same_page_are_deduplicated() -> None:
    r, _ = build([
        (chunk("a", 0, "first passage", span=(0, 100)), 0.9),
        (chunk("a", 1, "mostly the same passage", span=(30, 130)), 0.88),  # 70% overlap
        (chunk("a", 2, "next passage", span=(90, 190)), 0.8),   # 10% overlap: keep
        (chunk("a", 3, "other page", page=2, span=(0, 100)), 0.7),  # same span, other page
    ], top_k=4)
    assert ids(r.retrieve("q")) == ["a-p1-c0", "a-p1-c2", "a-p2-c3"]


def test_source_filter_overfetches_until_enough() -> None:
    # 30 strong hits in big.pdf would hide the weaker hits in small.pdf
    # without widening the candidate set.
    entries = [(chunk("big", i, f"big {i}"), 0.9 - i * 0.001) for i in range(30)]
    entries += [(chunk("small", i, f"small {i}", source="small.pdf"), 0.5) for i in range(3)]
    r, embedder = build(entries, top_k=2)
    results = r.retrieve("q", sources={"small.pdf"})
    assert [x.chunk.source for x in results] == ["small.pdf", "small.pdf"]
    assert embedder.calls == 1  # widening re-searches, but never re-embeds


def test_max_per_document() -> None:
    entries = [(chunk("a", i, f"a {i}"), 0.9 - i * 0.01) for i in range(5)]
    entries += [(chunk("b", 0, "b 0"), 0.5)]
    r, _ = build(entries, top_k=3)
    assert [x.chunk.doc_id for x in r.retrieve("q", max_per_document=2)] == ["a", "a", "b"]


def test_fewer_results_than_top_k_when_store_is_small() -> None:
    r, _ = build([(chunk("a", 0, "only"), 0.9)], top_k=5)
    assert ids(r.retrieve("q")) == ["a-p1-c0"]


def test_empty_store_returns_nothing_without_embedding() -> None:
    embedder = StubEmbedder()
    r = Retriever(VectorStore(DIM, "stub"), embedder)
    assert r.retrieve("anything") == []
    assert embedder.calls == 0


@pytest.mark.parametrize("bad", ["", "   ", "x" * 2001])
def test_invalid_queries_raise(bad: str) -> None:
    r, _ = build([(chunk("a", 0, "x"), 0.9)])
    with pytest.raises(RetrievalError):
        r.retrieve(bad)


def test_model_mismatch_is_rejected() -> None:
    with pytest.raises(RetrievalError):
        Retriever(VectorStore(DIM, "model-a"), StubEmbedder())


def test_from_settings_without_index_gives_helpful_error(tmp_path: Path) -> None:
    settings = Settings(processed_dir=tmp_path, _env_file=None)
    with pytest.raises(RetrievalError, match="build_index"):
        Retriever.from_settings(settings, Embedder("fake-model", model=FakeModel()))


def test_from_settings_end_to_end_with_fake_model(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.txt").write_text("annual leave days policy", encoding="utf-8")
    (docs / "food.txt").write_text("cafeteria lunch menu", encoding="utf-8")
    settings = Settings(documents_dir=docs, processed_dir=tmp_path / "p", top_k=1,
                        similarity_threshold=0.0, _env_file=None)
    embedder = Embedder("fake-model", model=FakeModel())
    sync_directory(settings, embedder)

    retriever = Retriever.from_settings(settings, embedder)
    assert [x.chunk.source for x in retriever.retrieve("annual leave policy")] == ["leave.txt"]


@pytest.mark.integration
def test_real_model_retrieval_quality(tmp_path: Path) -> None:
    from config.settings import get_settings

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "leave.txt").write_text(
        "Employees are entitled to twenty days of paid annual leave per year.", encoding="utf-8")
    (docs / "expenses.txt").write_text(
        "Expense claims must include receipts and be filed within 30 days.", encoding="utf-8")
    (docs / "security.txt").write_text(
        "Passwords must be at least 12 characters and rotated every 90 days.", encoding="utf-8")
    settings = Settings(documents_dir=docs, processed_dir=tmp_path / "p",
                        embedding_model=get_settings().embedding_model, _env_file=None)
    embedder = Embedder(settings.embedding_model)
    sync_directory(settings, embedder)
    retriever = Retriever.from_settings(settings, embedder)

    for question, expected in [
        ("How much vacation time do I get?", "leave.txt"),
        ("When do I have to hand in receipts?", "expenses.txt"),
        ("What are the password requirements?", "security.txt"),
    ]:
        assert retriever.retrieve(question)[0].chunk.source == expected, question
    # Off-topic question: nothing should clear the default threshold.
    assert retriever.retrieve("Who won the football world cup in 1998?") == []


# ---------- Phase 7: reranking ----------

class FakeCrossEncoder:
    """Scores a (query, passage) pair by shared words, minus 5 (so 'no overlap' < 0)."""

    def __init__(self) -> None:
        self.pairs_seen: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs) -> np.ndarray:
        self.pairs_seen.extend(pairs)
        return np.array([len(set(q.lower().split()) & set(p.lower().split())) - 5.0
                         for q, p in pairs])


def test_reranker_reorders_and_keeps_both_scores() -> None:
    from rag.reranker import Reranker

    fake = FakeCrossEncoder()
    entries = [
        (chunk("a", 0, "leave policy overview for all staff"), 0.9),   # topical, not an answer
        (chunk("a", 1, "how many leave days do employees get each year"), 0.6),
    ]
    r, _ = build(entries, top_k=2, reranker=Reranker("fake", model=fake))
    results = r.retrieve("how many leave days do employees get")
    assert ids(results) == ["a-p1-c1", "a-p1-c0"]
    assert results[0].score == pytest.approx(0.6, abs=1e-6)  # first-stage score kept
    assert results[0].rerank_score == 2.0  # 7 shared words - 5


def test_rerank_min_score_drops_non_answers() -> None:
    from rag.reranker import Reranker

    entries = [(chunk("a", 0, "cafeteria menu"), 0.8), (chunk("a", 1, "leave days per year"), 0.7)]
    r, _ = build(entries, reranker=Reranker("fake", model=FakeCrossEncoder()),
                 rerank_min_score=-4.5)
    assert ids(r.retrieve("leave days")) == ["a-p1-c1"]
    assert r.retrieve("parking rules") == []  # nothing actually answers it


def test_rerank_pool_is_larger_than_top_k_and_cap_applies_after() -> None:
    from rag.reranker import Reranker

    fake = FakeCrossEncoder()
    # The best answer is ranked 10th by the bi-encoder: only a wide pool finds it.
    entries = [(chunk("a", i, f"filler text {i}"), 0.9 - i * 0.01) for i in range(9)]
    entries += [(chunk("b", 0, "exact answer words here"), 0.8)]
    entries += [(chunk("b", 1, "exact answer words again"), 0.79)]
    r, _ = build(entries, top_k=2, reranker=Reranker("fake", model=fake), rerank_candidates=20)
    results = r.retrieve("exact answer words", max_per_document=1)
    assert len(fake.pairs_seen) == 11  # whole pool re-scored
    assert [x.chunk.doc_id for x in results] == ["b", "a"]  # cap: one per document


def test_reranker_not_called_when_nothing_retrieved() -> None:
    from rag.reranker import Reranker

    fake = FakeCrossEncoder()
    r, _ = build([(chunk("a", 0, "x"), 0.1)], min_score=0.5, reranker=Reranker("f", model=fake))
    assert r.retrieve("q") == []
    assert fake.pairs_seen == []


@pytest.mark.integration
def test_real_reranker_separates_related_from_answering() -> None:
    """The case the bi-encoder got wrong: 'parental leave' looked closer to the
    annual-leave passage than the real 'vacation' question did."""
    from config.settings import get_settings
    from rag.reranker import Reranker

    settings = get_settings()
    reranker = Reranker(settings.reranker_model)
    passage = chunk("h", 0, "Employees are entitled to twenty days of paid annual leave per year.")
    candidates = [RetrievedChunk(chunk=passage, score=0.5)]
    threshold = settings.rerank_min_score

    answering = reranker.rerank("How many days of annual leave do I get?", candidates)[0]
    unanswerable = reranker.rerank("What is the parental leave policy?", candidates)[0]
    assert answering.rerank_score > threshold > unanswerable.rerank_score


def test_first_stage_threshold_depends_on_reranking() -> None:
    """With a reranker, the lower RERANK_FIRST_STAGE_MIN_SCORE applies so relevant
    passages with modest similarity still reach the reranker (evaluation finding)."""
    from rag.reranker import Reranker

    settings = Settings(similarity_threshold=0.30, rerank_first_stage_min_score=0.15,
                        _env_file=None)
    store, embedder = VectorStore(DIM, "stub"), StubEmbedder()
    assert Retriever.for_store(store, embedder, settings).min_score == 0.30
    with_rerank = Retriever.for_store(store, embedder, settings, Reranker("f", model=object()))
    assert with_rerank.min_score == 0.15
    assert with_rerank.rerank_min_score == settings.rerank_min_score
