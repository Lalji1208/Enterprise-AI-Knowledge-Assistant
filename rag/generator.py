"""Phases 8-9 - Answer generation with verified citations.

Flow:
  1. Retrieve chunks. If none are relevant, return "not found" WITHOUT calling
     the LLM (no cost, and no chance to hallucinate).
  2. Fit the best chunks into MAX_CONTEXT_CHARS and number them [1]..[n].
  3. Ask the LLM for an answer that cites those numbers.
  4. Verify the citations: drop numbers that point to no source, keep only
     the sources actually cited, and warn when the answer cites nothing.

Overview questions ("what is it about?", "summarize this paper") match no single
passage, so search finds nothing for them. When the question is one and the scope
is a single document, step 1 is replaced by that document's opening chunks
(title, abstract, introduction); steps 3-4 are unchanged.

Usage:
    from rag.generator import AnswerGenerator
    answer = AnswerGenerator.from_settings().answer("How many leave days do I get?")
"""

import re

from config.settings import Settings, get_settings
from models.answer import Answer, Citation
from models.search import RetrievedChunk
from rag.llm import LLMClient, get_llm
from rag.prompts import NOT_FOUND_ANSWER, build_messages, select_context
from rag.retriever import Retriever

# [1], [2][3] and also [1, 3] / [1,2]
_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Questions about a whole document. Deliberately narrow: ordinary questions
# ("What is the dress code?") must keep going through normal retrieval.
_OVERVIEW = re.compile(
    r"""
    \bwhat(?:\s+is|'s)\s+(?:it|this|that)\s+(?:\w+\s+)?about\b
  | \bwhat(?:\s+is|'s|\s+are)\s+(?:the|this|these)\s+
      (?:paper|document|doc|file|report|pdf|article|text)s?\s+about\b
  | \bwhat\s+(?:does|do)\s+(?:the|this|these)\s+
      (?:paper|document|doc|file|report|pdf|article)s?\s+(?:say|cover|discuss|describe)\b
  | \b(?:summari[sz]e|summary|overview|gist|tl;?dr)\b
  | \bmain\s+(?:points?|ideas?|topics?|contributions?|findings?|takeaways?|themes?)\b
  | \bkey\s+(?:points|findings|takeaways|ideas)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
OVERVIEW_CONTEXT_CHARS = 4000  # opening of the document; kept small for local CPU models
OVERVIEW_INSTRUCTION = (
    "(Answer in at most five sentences: what this document is about - its topic, purpose "
    "and main points - citing the excerpt numbers like [1]. The excerpts are its opening "
    "pages.)"
)
OVERVIEW_HINT = "To summarise one document, limit the question to that document."


def is_overview_question(question: str) -> bool:
    return _OVERVIEW.search(question) is not None


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def is_not_found(text: str) -> bool:
    return _normalized(NOT_FOUND_ANSWER) in _normalized(text)


def extract_citation_numbers(text: str) -> list[int]:
    """Citation numbers in order of first appearance."""
    seen: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for n in (int(x) for x in group.split(",")):
            if n not in seen:
                seen.append(n)
    return seen


def remove_invalid_citations(text: str, valid: set[int]) -> str:
    """Strip citation numbers that do not refer to a provided source."""
    def fix(match: re.Match) -> str:
        kept = [x.strip() for x in match.group(1).split(",") if int(x) in valid]
        return f"[{', '.join(kept)}]" if kept else ""

    return re.sub(r" +([.,;:!?])", r"\1", _CITATION_GROUP.sub(fix, text)).strip()


def build_citations(numbers: list[int], context: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            number=n,
            source=context[n - 1].chunk.source,
            page_number=context[n - 1].chunk.page_number,
            chunk_id=context[n - 1].chunk.chunk_id,
            text=context[n - 1].chunk.text,
            score=context[n - 1].score,
            rerank_score=context[n - 1].rerank_score,
        )
        for n in sorted(numbers)
    ]


class AnswerGenerator:
    def __init__(self, retriever: Retriever, llm: LLMClient, max_context_chars: int = 8000) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_context_chars = max_context_chars

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AnswerGenerator":
        settings = settings or get_settings()
        return cls(Retriever.from_settings(settings), get_llm(), settings.max_context_chars)

    def answer(self, question: str, **retrieve_kwargs) -> Answer:
        """Answer from the indexed documents. Extra kwargs go to Retriever.retrieve
        (top_k, sources, ...). Raises RetrievalError / LLMError on failure."""
        question = question.strip()
        overview_source = self._overview_source(question, retrieve_kwargs.get("sources"))
        if overview_source is not None:
            opening = [RetrievedChunk(chunk=c, score=0.0)
                       for c in self.retriever.store.chunks_for_source(overview_source)]
            results = select_context(opening, OVERVIEW_CONTEXT_CHARS)
            context, prompt_question, mode = results, f"{question}\n{OVERVIEW_INSTRUCTION}", "overview"
        else:
            results = self.retriever.retrieve(question, **retrieve_kwargs)
            if not results:
                hint = [OVERVIEW_HINT] if is_overview_question(question) else []
                return Answer(question=question, answer=NOT_FOUND_ANSWER, status="not_found",
                              warnings=hint)
            context = select_context(results, self.max_context_chars)
            prompt_question, mode = question, "retrieval"

        reply = self.llm.complete(build_messages(prompt_question, context))
        text = reply.text
        warnings: list[str] = []
        base = {
            "question": question,
            "model": reply.model,
            "retrieved_count": len(results),
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "mode": mode,
        }

        if not text:
            return Answer(answer=NOT_FOUND_ANSWER, status="not_found",
                          warnings=["The model returned an empty answer."], **base)
        if is_not_found(text) and not extract_citation_numbers(text):
            return Answer(answer=NOT_FOUND_ANSWER, status="not_found", **base)

        valid = set(range(1, len(context) + 1))
        cited = extract_citation_numbers(text)
        invalid = [n for n in cited if n not in valid]
        if invalid:
            text = remove_invalid_citations(text, valid)
            warnings.append(f"Removed citation(s) to non-existent sources: {invalid}.")
        cited_valid = [n for n in cited if n in valid]
        if not cited_valid and mode == "overview":
            # The model saw only this document's opening passages, so those are the
            # summary's sources even when it wrote no [n] markers (small models often don't).
            cited_valid = sorted(valid)
        elif not cited_valid:
            warnings.append("The answer does not cite any source; verify it before relying on it.")
        if reply.truncated:
            warnings.append("The answer was cut off at the length limit (LLM_MAX_TOKENS).")

        return Answer(answer=text, status="answered", warnings=warnings,
                      citations=build_citations(cited_valid, context), **base)

    def _overview_source(self, question: str, sources) -> str | None:
        """The one document an overview question refers to, if it is unambiguous:
        the single document asked about, or the only document indexed."""
        if not is_overview_question(question):
            return None
        indexed = set(self.retriever.store.documents().values())
        scope = indexed & set(sources) if sources else indexed
        return next(iter(scope)) if len(scope) == 1 else None
