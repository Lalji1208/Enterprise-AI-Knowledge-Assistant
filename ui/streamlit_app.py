"""Phase 11 - Streamlit chat UI for the Enterprise AI Knowledge Assistant.

Start the API first, then the UI (both from the project root):
    uvicorn app.main:app
    streamlit run ui/streamlit_app.py

API_URL (env var) points the UI at the backend; default http://127.0.0.1:8000.
"""

import os
import sys
from itertools import pairwise
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.api_client import ApiClient, ApiError

MODE_ANSWER = "Answer with citations (LLM)"
MODE_SEARCH = "Find passages only (no LLM)"
HISTORY_TURNS = 3  # earlier exchanges sent along so follow-up questions can be understood


@st.cache_resource
def get_client(base_url: str) -> ApiClient:
    return ApiClient(base_url)


def render_answer(payload: dict) -> None:
    overview = payload.get("mode") == "overview"
    if overview:
        st.caption("Summary based on the document's opening pages")
    if payload.get("searched_question"):
        st.caption(f"Searched for: {payload['searched_question']}")
    st.markdown(payload["answer"])
    if payload.get("auto_cited"):
        st.caption("Citations added automatically by matching each sentence to its passage")
    for warning in payload.get("warnings", []):
        st.warning(warning)
    citations = payload.get("citations", [])
    if citations:
        st.caption("Sources")
        for c in citations:
            with st.expander(f"[{c['number']}] {c['source']} — page {c['page_number']}"):
                st.write(c["text"])
                if not overview:  # overview passages are chosen by position, not scored
                    rerank = c.get("rerank_score")
                    st.caption(f"similarity {c['score']:.3f}"
                               + (f" · rerank {rerank:.2f}" if rerank is not None else ""))


def render_passages(payload: dict) -> None:
    results = payload.get("results", [])
    if not results:
        st.markdown("No relevant passages found in the documents.")
        return
    st.markdown(f"Top {len(results)} matching passage(s):")
    for i, r in enumerate(results, start=1):
        c = r["chunk"]
        with st.expander(f"{i}. {c['source']} — page {c['page_number']}", expanded=i == 1):
            st.write(c["text"])
            rerank = r.get("rerank_score")
            st.caption(f"similarity {r['score']:.3f}"
                       + (f" · rerank {rerank:.2f}" if rerank is not None else ""))


def chat_history(messages: list[dict], max_turns: int = HISTORY_TURNS) -> list[dict]:
    """The last question/answer pairs of the chat, oldest first."""
    turns = [
        {"question": prev["content"], "answer": nxt["payload"]["answer"]}
        for prev, nxt in pairwise(messages)
        if prev["role"] == "user" and nxt.get("kind") == "answer"
    ]
    return turns[-max_turns:]


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("kind") == "answer":
            render_answer(message["payload"])
        elif message.get("kind") == "passages":
            render_passages(message["payload"])
        elif message.get("kind") == "error":
            st.error(message["content"])
        else:
            st.markdown(message["content"])


def sidebar(client: ApiClient, health: dict) -> tuple[str, list[str], int]:
    with st.sidebar:
        st.subheader("Status")
        st.caption(f"{health['documents']} document(s) · {health['chunks']} chunks")
        st.caption(f"Embeddings: {health['embedding_model']}")
        st.caption(f"Reranking: {'on' if health['reranking_enabled'] else 'off'} · "
                   f"LLM: {health['llm_provider']} / {health['llm_model']}")
        if not health["llm_configured"]:
            st.warning("No LLM key configured: use 'Find passages only'.")

        mode = st.radio("Mode", [MODE_ANSWER, MODE_SEARCH])
        top_k = st.slider("Passages to use", min_value=1, max_value=10, value=5)

        st.subheader("Documents")
        try:
            documents = client.documents()
        except ApiError as exc:
            st.error(exc.message)
            documents = []
        sources = [d["source"] for d in documents]
        selected = st.multiselect("Search only in", sources,
                                  placeholder="All documents")
        for d in documents:
            col_name, col_btn = st.columns([5, 1])
            col_name.caption(f"{d['source']}  \n{d['pages']} page(s) · {d['chunks']} chunks")
            if col_btn.button("✕", key=f"del-{d['doc_id']}", help=f"Delete {d['source']}"):
                try:
                    client.delete(d["source"])
                    st.toast(f"Deleted {d['source']}")
                except ApiError as exc:
                    st.error(exc.message)
                st.rerun()

        st.subheader("Add documents")
        files = st.file_uploader("PDF, TXT or MD", accept_multiple_files=True,
                                 type=["pdf", "txt", "md"])
        replace = st.checkbox("Replace files with the same name")
        if st.button("Upload and index", disabled=not files):
            for f in files:
                with st.spinner(f"Indexing {f.name}..."):
                    try:
                        report = client.upload(f.name, f.getvalue(), replace=replace)
                        note = " (no extractable text)" if f.name in report["empty"] else ""
                        st.success(f"{f.name}: indexed{note}")
                    except ApiError as exc:
                        st.error(f"{f.name}: {exc.message}")
        if st.button("Re-scan documents folder"):
            with st.spinner("Syncing..."):
                try:
                    report = client.sync()
                    st.success(f"+{len(report['added'])} / -{len(report['removed'])} documents")
                except ApiError as exc:
                    st.error(exc.message)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    return mode, selected, top_k


def main() -> None:
    st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")
    st.title("📚 Enterprise AI Knowledge Assistant")
    st.caption("Answers come only from your documents, with page-level citations.")

    client = get_client(os.environ.get("API_URL", "http://127.0.0.1:8000"))
    try:
        health = client.health()
    except ApiError as exc:
        st.error(exc.message)
        st.info("Start the backend from the project root:  `uvicorn app.main:app`")
        st.stop()

    mode, sources, top_k = sidebar(client, health)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        render_message(message)

    if health["chunks"] == 0:
        st.info("No documents indexed yet. Add some in the sidebar to get started.")

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    history = chat_history(st.session_state.messages)
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.spinner("Searching documents..." if mode == MODE_SEARCH else "Thinking..."):
        try:
            if mode == MODE_SEARCH:
                reply = {"role": "assistant", "kind": "passages",
                         "payload": client.search(question, top_k=top_k, sources=sources)}
            else:
                reply = {"role": "assistant", "kind": "answer",
                         "payload": client.ask(question, top_k=top_k, sources=sources,
                                               history=history)}
        except ApiError as exc:
            hint = " Try 'Find passages only' mode." if exc.status == 503 else ""
            reply = {"role": "assistant", "kind": "error", "content": exc.message + hint}
    st.session_state.messages.append(reply)
    render_message(reply)


main()
