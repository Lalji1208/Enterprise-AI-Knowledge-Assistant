"""Tests for automatic citations (rag/citations.py) and their use in the generator."""

from models.chunk import Chunk
from models.search import RetrievedChunk
from rag.citations import add_citations, best_supporting_passage
from rag.generator import AnswerGenerator
from rag.llm import LLMResponse


def passage(i: int, text: str, page: int = 1) -> RetrievedChunk:
    c = Chunk(chunk_id=f"d-p{page}-c{i}", doc_id="d", source="paper.pdf", page_number=page,
              chunk_index=i, text=text, start_char=0, end_char=len(text))
    return RetrievedChunk(chunk=c, score=0.5, rerank_score=1.0)


CONTEXT = [
    passage(0, "The Random Forest classifier produced accuracy of 59.88%.", page=4),
    passage(1, "The LinearSVM classifier produced accuracy of 62.97%.", page=4),
    passage(2, "The dataset was collected from kaggle.com and contains 2400+ resumes.", page=2),
]
TEXTS = [c.chunk.text for c in CONTEXT]


def test_numbers_must_match() -> None:
    # Both accuracy passages share the words; only the number decides.
    assert best_supporting_passage("The LinearSVM classifier reached 62.97% accuracy.", TEXTS) == 1
    assert best_supporting_passage("Random Forest reached 59.88% accuracy.", TEXTS) == 0
    assert best_supporting_passage("The SVM reached 75% accuracy.", TEXTS) is None


def test_unsupported_and_trivial_sentences_get_no_citation() -> None:
    assert best_supporting_passage("The system was deployed at three companies.", TEXTS) is None
    assert best_supporting_passage("Yes.", TEXTS) is None


def test_filler_words_do_not_count_as_support() -> None:
    assert best_supporting_passage("According to the provided sources, it was used.", TEXTS) is None


def test_add_citations_marks_each_supported_sentence() -> None:
    text, used = add_citations(
        "The data comes from kaggle.com with 2400+ resumes. The LinearSVM classifier "
        "was best with 62.97% accuracy. It was deployed widely.", CONTEXT)
    assert text == ("The data comes from kaggle.com with 2400+ resumes [3]. The LinearSVM "
                    "classifier was best with 62.97% accuracy [2]. It was deployed widely.")
    assert used == [3, 2]


def test_add_citations_handles_lists_and_missing_punctuation() -> None:
    text, used = add_citations("Results:\n- LinearSVM accuracy 62.97%\n- Random Forest 59.88%",
                               CONTEXT)
    assert "62.97% [2]" in text and "59.88% [1]" in text and used == [2, 1]


class OneReplyLLM:
    model = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, messages):
        return LLMResponse(text=self.reply, model=self.model)


class FixedRetriever:
    store = None

    def retrieve(self, query, **kwargs):
        return CONTEXT


def test_generator_auto_cites_uncited_answers() -> None:
    ans = AnswerGenerator(FixedRetriever(), OneReplyLLM(
        "The LinearSVM classifier produced the best accuracy, 62.97%.")).answer("Which was best?")
    assert ans.auto_cited and ans.warnings == []
    assert [(c.number, c.page_number) for c in ans.citations] == [(2, 4)]
    assert ans.answer.endswith("62.97% [2].")


def test_generator_keeps_warning_when_nothing_matches() -> None:
    ans = AnswerGenerator(FixedRetriever(), OneReplyLLM(
        "It was deployed at three companies.")).answer("Was it deployed?")
    assert not ans.auto_cited and ans.citations == []
    assert any("does not cite" in w for w in ans.warnings)


def test_model_citations_are_left_alone() -> None:
    ans = AnswerGenerator(FixedRetriever(), OneReplyLLM(
        "LinearSVM reached 62.97% [2].")).answer("Which was best?")
    assert not ans.auto_cited and [c.number for c in ans.citations] == [2]


def test_word_forms_match_so_the_method_passage_beats_the_reference_list() -> None:
    """Regression (found on a real paper): "tokenize" did not match "tokenizer", so the
    sentence was cited to the bibliography entry instead of the method section."""
    passages = [
        ("[17] Loper, E. and Bird, S. (2002). NLTK: The Natural Language Toolkit. "
        "In Proceedings of the ACL-02 Workshop on Teaching Natural Language Processing."),
        "Now using tokenizer (NLTK tokenizer) the cleaned data is divided into small tokens.",
    ]
    assert best_supporting_passage("The library used to tokenize the text is NLTK.",
                                   passages) == 1
