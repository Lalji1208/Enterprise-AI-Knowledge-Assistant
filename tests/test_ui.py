"""Tests for Phase 11: Streamlit UI (ui/streamlit_app.py) and its API client."""

from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from ui.api_client import ApiClient, ApiError

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")

HEALTH = {"status": "ok", "documents": 1, "chunks": 3, "embedding_model": "mini",
          "reranking_enabled": True, "llm_provider": "openai", "llm_model": "gpt",
          "llm_configured": True}
DOCS = [{"doc_id": "d1", "source": "handbook.pdf", "chunks": 3, "pages": 2}]
CHUNK = {"chunk_id": "d1-p2-c1", "doc_id": "d1", "source": "handbook.pdf", "page_number": 2,
         "chunk_index": 1, "text": "Employees get twenty days of annual leave.",
         "start_char": 0, "end_char": 42}
ANSWER = {"question": "q", "answer": "You get twenty days [1].", "status": "answered",
          "warnings": ["The answer was cut off at the length limit (LLM_MAX_TOKENS)."],
          "citations": [{"number": 1, "source": "handbook.pdf", "page_number": 2,
                         "chunk_id": "d1-p2-c1", "text": CHUNK["text"], "score": 0.61,
                         "rerank_score": 1.5}]}


# ---------- ApiClient ----------

class FakeResponse:
    def __init__(self, status: int, body) -> None:
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response, self.error, self.calls = response, error, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_client_builds_requests() -> None:
    session = FakeSession(FakeResponse(200, {"results": []}))
    client = ApiClient("http://api:8000/", session=session)
    client.search("leave", top_k=3, sources=[])
    method, url, kwargs = session.calls[-1]
    assert (method, url) == ("POST", "http://api:8000/api/search")
    assert kwargs["json"] == {"query": "leave", "top_k": 3, "sources": None}

    client.delete("Q3 report (final).pdf")
    assert session.calls[-1][1] == "http://api:8000/api/documents/Q3%20report%20%28final%29.pdf"

    client.upload("a.txt", b"x", replace=True)
    assert session.calls[-1][2]["params"] == {"replace": "true"}


@pytest.mark.parametrize(("response", "status", "message"), [
    (FakeResponse(503, {"detail": "no credits"}), 503, "no credits"),
    (FakeResponse(422, {"detail": [{"msg": "too short"}, {"msg": "bad"}]}), 422, "too short; bad"),
    (FakeResponse(500, ValueError("not json")), 500, "not json"),
])
def test_client_error_details(response, status: int, message: str) -> None:
    with pytest.raises(ApiError) as info:
        ApiClient("http://x", session=FakeSession(response)).health()
    assert info.value.status == status and message in info.value.message


def test_client_unreachable() -> None:
    session = FakeSession(error=requests.ConnectionError("refused"))
    with pytest.raises(ApiError) as info:
        ApiClient("http://x", session=session).health()
    assert info.value.status == 0 and "Cannot reach" in info.value.message


# ---------- Streamlit app ----------

@pytest.fixture
def fake_api(monkeypatch):
    """Patch the client's methods; returns a dict to tweak responses per test."""
    state = {"health": dict(HEALTH), "docs": list(DOCS), "ask": ANSWER, "search": None,
             "ask_error": None}

    monkeypatch.setattr(ApiClient, "health", lambda self: state["health"])
    monkeypatch.setattr(ApiClient, "documents", lambda self: state["docs"])

    def ask(self, question, top_k=None, sources=None, history=None):
        state["last_ask"] = (question, top_k, sources)
        state["last_history"] = history
        if state["ask_error"]:
            raise state["ask_error"]
        return state["ask"]

    def search(self, query, top_k=None, sources=None):
        return state["search"] or {"query": query, "results": [{"chunk": CHUNK, "score": 0.6,
                                                                "rerank_score": None}]}

    monkeypatch.setattr(ApiClient, "ask", ask)
    monkeypatch.setattr(ApiClient, "search", search)
    return state


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_shows_error_when_api_is_down(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    at = run_app()
    assert not at.exception
    assert "Cannot reach the API" in at.error[0].value


def test_app_empty_index_hint(fake_api) -> None:
    fake_api["health"].update(documents=0, chunks=0)
    fake_api["docs"] = []
    at = run_app()
    assert not at.exception
    assert any("No documents indexed yet" in i.value for i in at.info)


def test_app_answer_with_citations(fake_api) -> None:
    at = run_app()
    at.chat_input[0].set_value("How much leave do I get?").run()
    assert not at.exception
    assert any("You get twenty days [1]." in m.value for m in at.markdown)
    assert "cut off" in at.warning[0].value
    assert at.expander[0].label == "[1] handbook.pdf — page 2"
    assert fake_api["last_ask"] == ("How much leave do I get?", 5, [])


def test_app_passages_mode_and_source_filter(fake_api) -> None:
    at = run_app()
    at.sidebar.radio[0].set_value("Find passages only (no LLM)")
    at.sidebar.multiselect[0].select("handbook.pdf")
    at.run()
    at.chat_input[0].set_value("leave").run()
    assert not at.exception
    assert at.expander[0].label == "1. handbook.pdf — page 2"


def test_app_llm_unavailable_suggests_passage_mode(fake_api) -> None:
    fake_api["ask_error"] = ApiError(503, "The LLM account has no credits or quota left.")
    at = run_app()
    at.chat_input[0].set_value("question").run()
    assert not at.exception
    assert "no credits" in at.error[0].value and "Find passages only" in at.error[0].value


def test_app_sends_previous_turns_as_history(fake_api) -> None:
    at = run_app()
    at.chat_input[0].set_value("Which classifiers are used?").run()
    assert fake_api["last_history"] == []  # first question: no history
    at.chat_input[0].set_value("Which of them was better?").run()
    assert not at.exception
    assert fake_api["last_history"] == [
        {"question": "Which classifiers are used?", "answer": ANSWER["answer"]}
    ]


def test_app_shows_the_rewritten_question(fake_api) -> None:
    fake_api["ask"] = {**ANSWER, "searched_question": "Which classifier was more accurate?"}
    at = run_app()
    at.chat_input[0].set_value("Which was better?").run()
    assert any("Searched for: Which classifier was more accurate?" in c.value for c in at.caption)
