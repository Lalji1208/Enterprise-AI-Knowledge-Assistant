"""Automatic citations for answers the model wrote without [n] markers.

Small local models often state correct facts but forget the citation format
(11-24% of answers in our evaluation). Such an answer cannot be verified by the
reader, so each of its sentences is matched to the passages the model was given:
a passage supports a sentence when it contains most of the sentence's content
words AND every number in the sentence. Supported sentences get that passage's
[n]; a sentence no passage supports gets nothing, so a citation is never guessed.
No LLM call is involved.
"""

import re
from collections.abc import Sequence

from models.search import RetrievedChunk
from rag.keyword_index import tokenize

MIN_WORD_OVERLAP = 0.5   # share of the sentence's content words found in the passage
MIN_CONTENT_WORDS = 2    # shorter sentences ("Yes.") carry no checkable claim

# Words that describe the answer rather than carry its content.
_FILLER = frozenset({
    "according", "source", "sources", "provided", "document", "documents", "states",
    "stated", "mentions", "mentioned", "based", "information", "following", "also",
    "however", "therefore", "specifically", "used", "uses", "use", "using",
})
# A sentence ends at . ! ? followed by whitespace or the end - so "62.97%",
# "kaggle.com" and "e.g" stay inside their sentence - or at a line break (list items).
_SENTENCE = re.compile(r"(?:[^\n.!?]|[.!?](?=\S))+[.!?]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


STEM_CHARS = 6  # "tokenize"/"tokenizer", "library"/"libraries" compare as the same word


def _stem(word: str) -> str:
    """Crude prefix stem for words; numbers and identifiers stay exact."""
    return word[:STEM_CHARS] if word.isalpha() and len(word) > STEM_CHARS else word


def _content_words(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if t not in _FILLER and len(t) > 1}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def best_supporting_passage(sentence: str, passages: Sequence[str]) -> int | None:
    """Index of the passage that best supports the sentence, or None."""
    words = _content_words(sentence)
    if len(words) < MIN_CONTENT_WORDS:
        return None
    numbers = _numbers(sentence)
    best, best_overlap = None, 0.0
    for i, passage in enumerate(passages):
        if numbers and not numbers <= _numbers(passage):
            continue  # a number the passage does not contain: not its source
        overlap = len(words & _content_words(passage)) / len(words)
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    return best if best_overlap >= MIN_WORD_OVERLAP else None


def add_citations(answer: str, context: Sequence[RetrievedChunk]) -> tuple[str, list[int]]:
    """Insert [n] after each supported sentence. Returns the new text and the
    citation numbers used (1-based, in order of first use)."""
    passages = [c.chunk.text for c in context]
    used: list[int] = []

    def cite(match: re.Match) -> str:
        sentence = match.group(0)
        i = best_supporting_passage(sentence, passages)
        if i is None:
            return sentence
        n = i + 1
        if n not in used:
            used.append(n)
        body = sentence.rstrip()
        trailing = sentence[len(body):]
        stripped = body.rstrip(".!?")
        return f"{stripped} [{n}]{body[len(stripped):]}{trailing}"

    return _SENTENCE.sub(cite, answer), used
