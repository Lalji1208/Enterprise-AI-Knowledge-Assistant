"""Tests for chat follow-up questions (rag/conversation.py and its use in the generator)."""

import pytest

from models.chunk import Chunk
from models.search import RetrievedChunk
from rag.conversation import build_rewrite_messages, needs_rewrite, rewrite_question
from rag.generator import AnswerGenerator
from rag.llm import LLMResponse

HISTORY = [{"question": "Which classifiers does the paper use?",
            "answer": "It uses Random Forest and Linear SVM [1]."}]


class ScriptedLLM:
    """First call: the rewrite (if any). Later calls: the answer."""

    model = "fake"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.messages: list[list[dict]] = []

    def complete(self, messages):
        self.messages.append(messages)
        return LLMResponse(text=self.replies.pop(0), model=self.model)


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.store = None

    def retrieve(self, query: str, **kwargs):
        self.queries.append(query)
        c = Chunk(chunk_id="d-p3-c1", doc_id="d", source="paper.pdf", page_number=3,
                  chunk_index=1, text="LinearSVM produced accuracy of 62.97%.",
                  start_char=0, end_char=38)
        return [RetrievedChunk(chunk=c, score=0.5, rerank_score=1.0)]


@pytest.mark.parametrize(("question", "expected"), [
    ("Which of them was more accurate?", True),     # pronoun
    ("And the SVM?", True),                         # opener + short
    ("What about its training time?", True),
    ("Accuracy?", True),                            # very short
    ("What is the minimum RAM required?", False),   # standalone
    ("Which dataset is used for training the classifiers?", False),
])
def test_needs_rewrite(question: str, expected: bool) -> None:
    assert needs_rewrite(question, HISTORY) is expected


def test_no_history_never_rewrites() -> None:
    assert not needs_rewrite("And the SVM?", [])
    assert not needs_rewrite("And the SVM?", None)


def test_rewrite_prompt_contains_the_conversation() -> None:
    msgs = build_rewrite_messages("Which was better?", HISTORY)
    user = msgs[1]["content"]
    assert "User: Which classifiers does the paper use?" in user
    assert "Assistant: It uses Random Forest and Linear SVM [1]." in user
    assert user.endswith("Follow-up question: Which was better?\nStandalone question:")


@pytest.mark.parametrize(("reply", "expected"), [
    ('"Which classifier was more accurate, Random Forest or Linear SVM?"',
     "Which classifier was more accurate, Random Forest or Linear SVM?"),
    ("Standalone question: Which classifier was more accurate?",
     "Which classifier was more accurate?"),
    ("", "Which was better?"),                                 # empty -> original
    ("Line one\nLine two", "Which was better?"),               # rambling -> original
    ("x" * 600, "Which was better?"),                          # too long -> original
])
def test_rewrite_cleaning_and_fallback(reply: str, expected: str) -> None:
    assert rewrite_question(ScriptedLLM(reply), "Which was better?", HISTORY) == expected


def test_follow_up_is_searched_with_the_rewritten_question() -> None:
    retriever = RecordingRetriever()
    llm = ScriptedLLM("Which classifier was more accurate, Random Forest or Linear SVM?",
                      "Linear SVM was more accurate [1].")
    ans = AnswerGenerator(retriever, llm).answer("Which of them was more accurate?",
                                                 history=HISTORY)
    assert retriever.queries == ["Which classifier was more accurate, Random Forest or Linear SVM?"]
    assert ans.searched_question == retriever.queries[0]
    assert ans.question == "Which of them was more accurate?"  # what the user typed
    assert ans.status == "answered" and ans.citations[0].page_number == 3
    assert "Random Forest or Linear SVM" in llm.messages[1][1]["content"]  # answer prompt too


def test_standalone_question_costs_no_extra_call() -> None:
    retriever = RecordingRetriever()
    llm = ScriptedLLM("It needs 4 GB [1].")
    ans = AnswerGenerator(retriever, llm).answer("What is the minimum RAM required?",
                                                 history=HISTORY)
    assert retriever.queries == ["What is the minimum RAM required?"]
    assert len(llm.messages) == 1 and ans.searched_question is None


def test_rewriting_can_be_disabled_and_history_is_capped() -> None:
    retriever = RecordingRetriever()
    gen = AnswerGenerator(retriever, ScriptedLLM("Answer [1]."), rewrite_follow_ups=False)
    gen.answer("And the SVM?", history=HISTORY)
    assert retriever.queries == ["And the SVM?"]

    llm = ScriptedLLM("Standalone?", "Answer [1].")
    long_history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
    AnswerGenerator(RecordingRetriever(), llm, history_turns=2).answer("And it?",
                                                                     history=long_history)
    rewrite_prompt = llm.messages[0][1]["content"]
    assert "q4" in rewrite_prompt and "q5" in rewrite_prompt and "q3" not in rewrite_prompt
