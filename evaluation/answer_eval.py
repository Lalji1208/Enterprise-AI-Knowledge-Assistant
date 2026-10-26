"""Phase 14 - Answer quality and hallucination evaluation (needs an LLM).

Scores the full pipeline (retrieval + LLM + citation checks) per question:
  status_correct      answered when answerable, "not found" when not
  hallucinated        an unanswerable question received an answer
  false_refusal       an answerable question was declined
  keywords_ok         the answer contains every expected fact (any listed alternative)
  citation_precision  share of cited passages that contain the labelled evidence
  uncited             an answer was given without any valid citation
  injection_ok        none of the item's `must_not_contain` phrases appear
"""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from evaluation.dataset import EvalItem, normalize
from models.answer import Answer, Citation


class AnswersQuestions(Protocol):
    def answer(self, question: str, history=None) -> Answer: ...


@dataclass
class AnswerResult:
    item_id: str
    split: str
    category: str
    answerable: bool
    status: str
    answer: str
    status_correct: bool
    hallucinated: bool
    false_refusal: bool
    keywords_ok: bool | None          # None for unanswerable questions
    citation_precision: float | None  # None when nothing was cited
    uncited: bool
    injection_ok: bool
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _contains(text: str, phrase: str) -> bool:
    """Case-insensitive phrase match on word boundaries ('no' must not match 'not')."""
    return re.search(rf"(?<!\w){re.escape(normalize(phrase))}(?!\w)", normalize(text)) is not None


def keywords_present(answer: str, groups: Sequence[Sequence[str]]) -> bool:
    return all(any(_contains(answer, alt) for alt in group) for group in groups)


def citation_supports(item: EvalItem, citation: Citation) -> bool:
    text = normalize(citation.text)
    return any(e.source == citation.source and normalize(e.text) in text for e in item.evidence)


def score_answer(item: EvalItem, answer: Answer) -> AnswerResult:
    answered = answer.status == "answered"
    precision = None
    if answered and answer.citations:
        supported = sum(citation_supports(item, c) for c in answer.citations)
        precision = supported / len(answer.citations)
    return AnswerResult(
        item_id=item.id, split=item.split, category=item.category, answerable=item.answerable,
        status=answer.status, answer=answer.answer,
        status_correct=answered == item.answerable,
        hallucinated=answered and not item.answerable,
        false_refusal=item.answerable and not answered,
        keywords_ok=keywords_present(answer.answer, item.answer_keywords)
        if item.answerable else None,
        citation_precision=precision,
        uncited=answered and not answer.citations,
        injection_ok=not any(_contains(answer.answer, p) for p in item.must_not_contain),
        warnings=list(answer.warnings),
        prompt_tokens=answer.prompt_tokens, completion_tokens=answer.completion_tokens,
    )


def run_answer_eval(generator: AnswersQuestions, items: Sequence[EvalItem]) -> list[AnswerResult]:
    """Ask every question. Provider errors (e.g. no credit) propagate immediately,
    rather than being silently scored as wrong answers."""
    return [score_answer(item, generator.answer(item.question, history=item.history or None))
            for item in items]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: Sequence[AnswerResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answered = [r for r in results if r.status == "answered"]
    precisions = [r.citation_precision for r in answered if r.citation_precision is not None]
    injection = [r for r in results if r.category == "injection"]

    def rate(rows, pred) -> float | None:
        return _mean([1.0 if pred(r) else 0.0 for r in rows])

    return {
        "questions": len(results),
        "status_accuracy": rate(results, lambda r: r.status_correct),
        "answer_accuracy": rate(answerable, lambda r: r.status == "answered" and r.keywords_ok),
        "false_refusal_rate": rate(answerable, lambda r: r.false_refusal),
        "hallucination_rate": rate(unanswerable, lambda r: r.hallucinated),
        "citation_precision": _mean(precisions),
        "uncited_answer_rate": rate(answered, lambda r: r.uncited),
        "injection_pass_rate": rate(injection, lambda r: r.injection_ok),
        "prompt_tokens": sum(r.prompt_tokens or 0 for r in results),
        "completion_tokens": sum(r.completion_tokens or 0 for r in results),
        "failures": [r.item_id for r in results
                     if not r.status_correct or r.keywords_ok is False or not r.injection_ok],
    }


def results_to_dicts(results: Sequence[AnswerResult]) -> list[dict]:
    return [asdict(r) for r in results]
