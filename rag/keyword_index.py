"""Keyword (BM25) search over the indexed chunks - the lexical half of hybrid search.

Dense embeddings capture meaning but barely register rare exact strings: names
("Arguello"), codes, e-mail addresses, extensions. Such a query can score ~0.0 in
cosine similarity against the very passage that contains it. BM25 scores exact
term overlap weighted by rarity, so it finds those passages; the retriever adds
its hits to the candidates and lets the cross-encoder decide.

The index is built in memory from the vector store's chunks and rebuilt when the
store changes (cheap: tens of thousands of short chunks take well under a second).
"""

import math
import re
import threading
from collections import Counter
from collections.abc import Collection

from models.chunk import Chunk

_TOKEN = re.compile(r"[\w@.\-]*\w", re.UNICODE)
# Question and instruction words that would otherwise match everywhere.
_STOPWORDS = frozenset({
    "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "by", "can",
    "could", "did", "do", "does", "find", "for", "from", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "mention", "mentioned", "my", "of",
    "on", "or", "please", "say", "says", "show", "tell", "that", "the", "their",
    "them", "there", "these", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you",
    "your",
})


def tokenize(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text))
            if t not in _STOPWORDS]


class KeywordIndex:
    def __init__(self, chunks: Collection[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks = list(chunks)
        self._tf = [Counter(tokenize(c.text)) for c in self._chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._chunks)
        # BM25 idf (the "+1" form, always positive)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(
        self, query: str, top_k: int = 10, sources: Collection[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Best-matching chunks with their BM25 score (> 0 only), best first."""
        terms = set(tokenize(query)) & self._idf.keys()
        if not terms or not self._chunks:
            return []
        scored = []
        for chunk, tf, length in zip(self._chunks, self._tf, self._len):
            if sources is not None and chunk.source not in sources:
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class CachedKeywordIndex:
    """Keeps a KeywordIndex in step with a VectorStore, rebuilding after changes."""

    def __init__(self, store) -> None:
        self._store = store
        self._index: KeywordIndex | None = None
        self._version = -1
        self._lock = threading.Lock()

    def get(self) -> KeywordIndex:
        with self._lock:
            if self._index is None or self._version != self._store.version:
                self._version = self._store.version
                self._index = KeywordIndex(self._store.all_chunks())
            return self._index
