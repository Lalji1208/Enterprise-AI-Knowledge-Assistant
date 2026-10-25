"""Tests for hybrid search: the BM25 keyword index and its use in the retriever."""

import numpy as np
from conftest import DIM, FakeModel

from models.chunk import Chunk
from rag.embedder import Embedder
from rag.keyword_index import CachedKeywordIndex, KeywordIndex, tokenize
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.vector_store import VectorStore


def chunk(i: int, text: str, source: str = "paper.pdf") -> Chunk:
    return Chunk(chunk_id=f"d{source}-p1-c{i}", doc_id=f"d{source}", source=source,
                 page_number=1, chunk_index=i, text=text, start_char=i * 1000,
                 end_char=i * 1000 + len(text))


CHUNKS = [
    chunk(0, "Resume screening ranks candidates with cosine similarity."),
    chunk(1, "REFERENCES [1] Arguello, J., & Jaime, A. (2013). Retrieving diverse images."),
    chunk(2, "Report incidents to security@example.com within one hour."),
    chunk(3, "Call the IT service desk at extension 4400."),
    chunk(4, "Arguello appears here too.", source="other.pdf"),
]


def test_tokenize_keeps_exact_identifiers_and_drops_question_words() -> None:
    assert tokenize("Find Arguello and Jaime, please!") == ["arguello", "jaime"]
    assert tokenize("Who is security@example.com? Call 4400.") == [
        "security@example.com", "call", "4400"]
    assert tokenize("part-time pro-rata") == ["part-time", "pro-rata"]


def test_bm25_finds_rare_exact_terms() -> None:
    index = KeywordIndex(CHUNKS)
    assert index.search("find Arguello and Jaime")[0][0].chunk_index == 1  # both names
    assert index.search("what is extension 4400")[0][0].chunk_index == 3
    assert index.search("security@example.com")[0][0].chunk_index == 2
    assert index.search("what is it about") == []  # only stopwords -> nothing


def test_bm25_respects_source_filter_and_top_k() -> None:
    index = KeywordIndex(CHUNKS)
    assert {c.source for c, _ in index.search("Arguello", sources={"other.pdf"})} == {"other.pdf"}
    assert len(index.search("Arguello", top_k=1)) == 1


def store_with(chunks: list[Chunk]) -> VectorStore:
    store = VectorStore(DIM, "fake-model")
    store.add(chunks, Embedder("fake-model", model=FakeModel()).embed_chunks(chunks))
    return store


def test_cached_index_rebuilds_after_store_changes() -> None:
    store = store_with(CHUNKS[:2])
    cached = CachedKeywordIndex(store)
    assert cached.get().search("4400") == []
    store.add([CHUNKS[3]], Embedder("fake-model", model=FakeModel()).embed_chunks([CHUNKS[3]]))
    assert cached.get().search("4400")[0][0].chunk_index == 3


def test_similarities_match_faiss_scores() -> None:
    store = store_with(CHUNKS)
    q = Embedder("fake-model", model=FakeModel()).embed_query("extension 4400")
    from_faiss = {r.chunk.chunk_id: r.score for r in store.search(q, top_k=5)}
    direct = store.similarities(q, list(from_faiss))
    for cid, score in from_faiss.items():
        assert abs(direct[cid] - score) < 1e-5


class KeywordLovingCrossEncoder:
    """Scores a pair by shared rare words, so keyword-found passages can pass."""

    def predict(self, pairs, **kwargs):
        return np.array([2.0 if "arguello" in p.lower() and "arguello" in q.lower() else -9.0
                         for q, p in pairs])


def test_hybrid_retrieval_finds_name_that_embeddings_miss() -> None:
    store = store_with(CHUNKS)
    embedder = Embedder("fake-model", model=FakeModel())
    reranker = Reranker("fake", model=KeywordLovingCrossEncoder())
    # A first-stage cutoff no chunk can pass: only keyword search can find anything.
    dense_only = Retriever(store, embedder, min_score=0.99, reranker=reranker,
                           rerank_min_score=-6.0)
    hybrid = Retriever(store, embedder, min_score=0.99, reranker=reranker,
                       rerank_min_score=-6.0, keyword_candidates=10)
    assert dense_only.retrieve("find Arguello and Jaime") == []
    results = hybrid.retrieve("find Arguello and Jaime", sources={"paper.pdf"})
    assert [r.chunk.chunk_index for r in results] == [1]
    assert results[0].rerank_score == 2.0


def test_keyword_hits_still_need_the_reranker_to_agree() -> None:
    store = store_with(CHUNKS)
    hybrid = Retriever(store, Embedder("fake-model", model=FakeModel()), min_score=0.99,
                       reranker=Reranker("fake", model=KeywordLovingCrossEncoder()),
                       rerank_min_score=-6.0, keyword_candidates=10)
    assert hybrid.retrieve("extension 4400") == []  # keyword match, but reranker says -9


def test_keyword_search_is_off_without_a_reranker() -> None:
    r = Retriever(store_with(CHUNKS), Embedder("fake-model", model=FakeModel()),
                  keyword_candidates=10)
    assert r.keyword_candidates == 0
