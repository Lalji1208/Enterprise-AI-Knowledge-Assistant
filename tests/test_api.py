"""Tests for Phase 10: FastAPI backend (app/main.py, api/routes.py, services/assistant.py).

Uses the fake embedding model and a fake LLM: no downloads, no API calls.
"""

from pathlib import Path

import pytest
from conftest import FakeModel, make_pdf
from fastapi.testclient import TestClient

from app.main import create_app
from config.settings import Settings
from rag.embedder import Embedder
from rag.llm import LLMError, LLMResponse
from services.assistant import AssistantService, UploadError, safe_filename


class FakeLLM:
    model = "fake-llm"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return LLMResponse(text="Employees get twenty days of annual leave [1].", model=self.model)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {"documents_dir": tmp_path / "docs", "processed_dir": tmp_path / "processed",
              "similarity_threshold": 0.0, "max_upload_size_mb": 1, "llm_provider": "openai",
              "openai_api_key": "", "_env_file": None}
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(tmp_path: Path, llm: FakeLLM):
    settings = make_settings(tmp_path)
    svc = AssistantService(settings, Embedder("fake-model", model=FakeModel()),
                           llm_factory=lambda: llm)
    with TestClient(create_app(svc)) as c:
        c.settings = settings
        yield c


def upload(client, name: str, content: bytes, replace: bool = False):
    return client.post("/api/documents", params={"replace": replace},
                       files={"file": (name, content, "application/octet-stream")})


# ---------- health & documents ----------

def test_health_on_empty_index(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["documents"] == 0 and body["chunks"] == 0
    assert body["llm_configured"] is False  # openai provider with empty key


def test_upload_list_and_delete(client) -> None:
    r = upload(client, "leave policy.txt", b"Employees get twenty days of annual leave.")
    assert r.status_code == 200, r.text
    assert r.json()["added"] == ["leave policy.txt"]
    assert (client.settings.documents_dir / "leave policy.txt").is_file()

    docs = client.get("/api/documents").json()
    assert [(d["source"], d["chunks"], d["pages"]) for d in docs] == [("leave policy.txt", 1, 1)]

    r = client.delete("/api/documents/leave policy.txt")
    assert r.status_code == 200 and r.json()["removed"] == ["leave policy.txt"]
    assert client.get("/api/documents").json() == []
    assert not (client.settings.documents_dir / "leave policy.txt").exists()


def test_upload_pdf(client) -> None:
    pdf = tmp_pdf_bytes(["Expense claims within thirty days", "Second page"])
    r = upload(client, "expenses.pdf", pdf)
    assert r.status_code == 200, r.text
    assert client.get("/api/documents").json()[0]["pages"] == 2


def tmp_pdf_bytes(pages: list[str]) -> bytes:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        return make_pdf(Path(d) / "x.pdf", pages).read_bytes()


def test_index_persists_across_restarts(client, tmp_path: Path) -> None:
    upload(client, "a.txt", b"alpha content")
    svc2 = AssistantService(client.settings, Embedder("fake-model", model=FakeModel()))
    assert [d["source"] for d in svc2.list_documents()] == ["a.txt"]


# ---------- upload validation ----------

@pytest.mark.parametrize(("name", "content", "status"), [
    ("notes.exe", b"MZ", 415),
    ("big.txt", b"x" * (1024 * 1024 + 1), 413),
    ("empty.txt", b"", 400),
    ("broken.pdf", b"not a pdf at all", 422),
    ("../escape.txt", b"x", 400),
    (".hidden.txt", b"x", 400),
    ("CON.txt", b"x", 400),
], ids=["wrong-type", "too-large", "empty", "corrupt-pdf", "path-traversal", "dotfile",
        "windows-reserved"])  # explicit ids: the 1 MB payload must not become the test name
def test_upload_rejections(client, name: str, content: bytes, status: int) -> None:
    r = upload(client, name, content)
    assert r.status_code == status, r.text
    assert list(client.settings.documents_dir.iterdir()) == []  # nothing written
    assert not list((client.settings.processed_dir / "tmp").glob("*"))  # no temp leftovers


def test_existing_file_needs_replace_and_bad_replacement_keeps_original(client) -> None:
    upload(client, "policy.pdf", tmp_pdf_bytes(["Original policy text"]))
    assert upload(client, "policy.pdf", tmp_pdf_bytes(["New"])).status_code == 409

    # A corrupt replacement is rejected and the original file stays intact.
    original = (client.settings.documents_dir / "policy.pdf").read_bytes()
    assert upload(client, "policy.pdf", b"corrupt", replace=True).status_code == 422
    assert (client.settings.documents_dir / "policy.pdf").read_bytes() == original

    r = upload(client, "policy.pdf", tmp_pdf_bytes(["Updated policy text"]), replace=True)
    assert r.status_code == 200
    assert r.json()["added"] == ["policy.pdf"] and r.json()["removed"] == ["policy.pdf"]


def test_delete_missing_and_traversal(client) -> None:
    assert client.delete("/api/documents/nope.txt").status_code == 404
    assert client.delete("/api/documents/..%5Csecret.txt").status_code == 400


def test_sync_picks_up_manually_copied_files(client) -> None:
    (client.settings.documents_dir / "manual.md").write_text("# Manual\nCopied in by hand.",
                                                            encoding="utf-8")
    r = client.post("/api/index/sync")
    assert r.status_code == 200 and r.json()["added"] == ["manual.md"]


# ---------- search & ask ----------

def test_search(client) -> None:
    upload(client, "leave.txt", b"annual leave days policy")
    upload(client, "food.txt", b"cafeteria lunch menu")
    r = client.post("/api/search", json={"query": "annual leave policy", "top_k": 1})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["chunk"]["source"] for x in results] == ["leave.txt"]
    assert {"score", "rerank_score"} <= results[0].keys()


def test_search_source_filter_and_validation(client) -> None:
    upload(client, "leave.txt", b"annual leave days policy")
    upload(client, "food.txt", b"cafeteria lunch menu")
    r = client.post("/api/search", json={"query": "annual leave", "sources": ["food.txt"]})
    assert [x["chunk"]["source"] for x in r.json()["results"]] == ["food.txt"]
    assert client.post("/api/search", json={"query": ""}).status_code == 422
    assert client.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_ask_returns_cited_answer(client, llm: FakeLLM) -> None:
    upload(client, "leave.txt", b"Employees get twenty days of annual leave.")
    r = client.post("/api/ask", json={"question": "How much annual leave?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    assert body["citations"][0]["source"] == "leave.txt" and body["citations"][0]["number"] == 1
    assert llm.calls == 1


def test_ask_with_empty_index_does_not_need_llm(tmp_path: Path) -> None:
    def broken_llm():
        raise LLMError("OPENAI_API_KEY is empty.")

    settings = make_settings(tmp_path)
    svc = AssistantService(settings, Embedder("fake-model", model=FakeModel()),
                           llm_factory=broken_llm)
    with TestClient(create_app(svc)) as c:
        # Upload/search still work although the LLM is unusable...
        upload(c, "a.txt", b"some content here")
        assert c.post("/api/search", json={"query": "content"}).status_code == 200
        # ...and asking reports the LLM problem as 503 with a helpful message.
        r = c.post("/api/ask", json={"question": "content?"})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]


def test_openapi_docs_available(client) -> None:
    assert client.get("/docs").status_code == 200
    assert "/api/ask" in client.get("/openapi.json").json()["paths"]


# ---------- unit: safe_filename ----------

@pytest.mark.parametrize("good", ["report.pdf", "Q3 results (final).pdf", "política_2026.txt"])
def test_safe_filename_accepts(good: str) -> None:
    assert safe_filename(good) == good


@pytest.mark.parametrize("bad", ["", "a/b.txt", "..\\x.txt", "..", ".env", "x.txt.", "nul.pdf",
                                 "a<b>.txt", "x" * 151 + ".txt"],
                         ids=["empty", "slash", "backslash", "dotdot", "dotfile", "trailing-dot",
                              "reserved", "bad-chars", "too-long"])
def test_safe_filename_rejects(bad: str) -> None:
    with pytest.raises(UploadError):
        safe_filename(bad)


def test_ask_passes_history_to_the_service(tmp_path: Path) -> None:
    seen = {}

    class Svc:
        settings = make_settings(tmp_path)

        def ask(self, question, top_k=None, sources=None, history=None):
            seen["history"] = history
            from models.answer import Answer
            return Answer(question=question, answer="ok", status="not_found")

    with TestClient(create_app(Svc())) as c:
        r = c.post("/api/ask", json={"question": "And it?",
                                     "history": [{"question": "Q1", "answer": "A1"}]})
    assert r.status_code == 200
    assert seen["history"] == [{"question": "Q1", "answer": "A1"}]
